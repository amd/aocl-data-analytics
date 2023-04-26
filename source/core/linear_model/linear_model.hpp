#ifndef LINEAR_MODEL_HPP
#define LINEAR_MODEL_HPP

#include "aoclda.h"
#include "callbacks.hpp"
#include "da_cblas.hh"
#include "lapack_templates.hpp"
#include "linmod_options.hpp"
#include "optimization.hpp"
#include "options.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>

#undef min
#undef max

// used to pass pointers to optimization callbacks
template <typename T> struct fit_usrdata {
    da_int m;
    T *A = nullptr;
    T *b = nullptr, *y = nullptr;
    bool intercept = false;
};

// data for QR factorization used in standard linear least squares
template <typename T> struct qr_data {
    // A needs to be copied as lapack's dgeqr modifies the matrix
    std::vector<T> A, b, tau, work;
    da_int lwork = 0;
};

enum fit_opt_type { fit_opt_nln = 0, fit_opt_lsq };

template <typename T> class linear_model {
  private:
    /* type of the model, has to de set at initialization phase */
    linmod_model mod = linmod_model_undefined;

    /* true if the model has been successfully trained */
    bool model_trained = false;

    /* Regression data
     * n: number of features
     * m: number of data points
     * intercept: controls if the linear regression intercept is to be set
     * A[m*n]: feature matrix, pointer to user data directly - will not be modified by any
     * function b[m]: model response, pointer to user data - will not be modified by any
     * function
     */
    da_int n = 0, m = 0;
    bool intercept = false;
    T *b = nullptr;
    T *A = nullptr;

    /* Training data
     * coef[n/n+1]: vector containing the trained coefficients of the model
     * l_nrm1/2: regularization factors for norm 1 and 2 respectively. 
     *           0 => no reg, possible different algorithm 
     */
    da_int ncoef = 0;
    std::vector<T> coef;
    T l_nrm2 = 0.0, l_nrm1 = 0.0;

    /* optimization object to call generic algorithms */
    da_optimization<T> *opt = nullptr;
    fit_usrdata<T> *usrdata = nullptr;
    qr_data<T> *qr = nullptr;

    /* private methods to allocate memory */
    da_status init_opt_model(fit_opt_type opt_type, objfun_t<T> objfun,
                             objgrd_t<T> objgrd);
    void init_usrdata();
    /* QR fact data */
    da_status init_qr_data();
    da_status qr_lsq();

    /* Dispatcher methods 
     * choose_method: if "linmod optim method" is set to auto, choose automatically how 
     *                to compute the model
     * validate_options: check that the options chosen by the user are compatible
     */
    da_status choose_method();
    da_status validate_options(da_int method);

  public:
    da_options::OptionRegistry opts;
    linear_model() { register_linmod_options<T>(opts); }
    ~linear_model();

    da_status define_features(da_int n, da_int m, T *A, T *b);
    da_status select_model(linmod_model mod);
    da_status fit();
    da_status get_coef(da_int &nx, T *x);
    da_status evaluate_model(da_int n, da_int m, T *X, T *predictions);

    /* Methods to remove once option setter is added */
    void set_reg_nrm2(T l_nrm2);
    void set_reg_nrm1(T l_nrm1);
};

template <typename T> linear_model<T>::~linear_model() {
    // A and b are passed from (internal) user, do not deallocate
    A = nullptr;
    b = nullptr;

    if (opt)
        delete opt;
    if (usrdata) {
        if (usrdata->y)
            delete[] usrdata->y;
        delete usrdata;
    }
    if (qr) {
        delete qr;
    }
}

template <typename T>
da_status linear_model<T>::define_features(da_int n, da_int m, T *A, T *b) {
    if (n <= 0 || m <= 0 || A==nullptr || b == nullptr)
        return da_status_invalid_input;

    model_trained = false;

    this->n = n;
    this->m = m;
    // copy user's feature pointers
    this->b = b;
    this->A = A;
    // allocate enough space for the model coefficients, including a possible
    // intercept variable coef[n+1] will contain the intercept var after fit
    coef.reserve(n + 1);

    return da_status_success;
}

template <typename T> da_status linear_model<T>::select_model(linmod_model mod) {

    if (mod != this->mod) {
        this->mod = mod;
        model_trained = false;
    }
    return da_status_success;
}

template <typename T> void linear_model<T>::init_usrdata() {
    usrdata = new fit_usrdata<T>;
    usrdata->A = A;
    usrdata->b = b;
    usrdata->m = m;
    usrdata->y = new T[m];
    usrdata->intercept = intercept;
}

//////////////////////// Objective Functions ////////////////////////

template <typename T> void objfun_mse(da_int n, T *x, T *f, void *usrdata) {
    fit_usrdata<T> *data;
    data = (fit_usrdata<T> *)usrdata;
    da_int m = data->m;
    T alpha = 1.0, beta = 0.0;
    T *y = data->y;
    da_int aux = data->intercept ? 1 : 0;
    da_blas::cblas_gemv(CblasColMajor, CblasNoTrans, data->m, n - aux, alpha, data->A, m,
                        x, 1, beta, y, 1);
    *f = 0.0;
    T lin_comb;

    for (da_int i = 0; i < m; i++) {
        lin_comb = data->intercept ? x[n - 1] + y[i] : y[i];
        *f += pow(lin_comb - data->b[i], (T)2.0);
    }
}

template <typename T> void objgrd_mse(da_int n, T *x, T *grad, void *usrdata) {
    fit_usrdata<T> *data;
    data = (fit_usrdata<T> *)usrdata;
    da_int m = data->m;
    T *y = data->y;

    T alpha = 1.0, beta = 0.0;
    da_int aux = data->intercept ? 1 : 0;
    da_blas::cblas_gemv(CblasColMajor, CblasNoTrans, data->m, n - aux, alpha, data->A, m,
                        x, 1, beta, y, 1);
    if (data->intercept) {
        for (da_int i = 0; i < m; i++)
            y[i] += x[n - 1];
    }
    alpha = -1.0;
    da_blas::cblas_axpy(data->m, alpha, data->b, 1, data->y, 1);

    alpha = 2.0;
    da_blas::cblas_gemv(CblasColMajor, CblasTrans, data->m, n - aux, alpha, data->A, m, y,
                        1, beta, grad, 1);
    if (data->intercept) {
        grad[n - 1] = 0.0;
        for (da_int i = 0; i < m; i++)
            grad[n - 1] += (T)2.0 * y[i];
    }
}

template <typename T> T log_loss(T y, T p) { return -y * log(p) - (1 - y) * log(1 - p); }
template <typename T> T logistic(T x) { return 1 / (1 + exp(-x)); }

template <typename T> void objfun_logistic(da_int n, T *x, T *f, void *usrdata) {
    // Extract user data
    fit_usrdata<T> *data;
    data = (fit_usrdata<T> *)usrdata;
    T *b = data->b, *y = data->y, *A = data->A;
    da_int m = data->m;

    // Comput A*x[0:n-2] = y
    da_int aux = data->intercept ? 1 : 0;
    T alpha = 1.0, beta = 0.0;
    da_blas::cblas_gemv(CblasColMajor, CblasNoTrans, m, n - aux, alpha, A, m, x, 1, beta,
                        data->y, 1);

    // sum of log loss of logistic function for all observations
    *f = 0.0;
    T lin_comb;
    for (da_int i = 0; i < m; i++) {
        lin_comb = data->intercept ? x[n - 1] + y[i] : y[i];
        *f += log_loss(b[i], logistic(lin_comb));
    }
}

template <typename T> void objgrd_logistic(da_int n, T *x, T *grad, void *usrdata) {
    /* gradient of log loss of the logistic function 
     * g_j = sum_i{A_ij*(b[i]-logistic(A_i^t x + x[n-1]))}
     */

    fit_usrdata<T> *data;
    data = (fit_usrdata<T> *)usrdata;
    T *b = data->b, *y = data->y, *A = data->A;
    da_int m = data->m;

    // Comput A*x[0:n-2] = y
    da_int aux = data->intercept ? 1 : 0;
    T alpha = 1.0, beta = 0.0;
    da_blas::cblas_gemv(CblasColMajor, CblasNoTrans, m, n - aux, alpha, data->A, m, x, 1,
                        beta, data->y, 1);

    std::fill(grad, grad + n, 0);
    T lin_comb;
    for (da_int i = 0; i < m; i++) {
        lin_comb = data->intercept ? x[n - 1] + y[i] : y[i];
        for (da_int j = 0; j < n - aux; j++)
            grad[j] += (logistic(lin_comb) - b[i]) * A[m * j + i];
    }
    if (data->intercept) {
        grad[n - 1] = 0.0;
        for (da_int i = 0; i < m; i++) {
            lin_comb = x[n - 1] + y[i];
            grad[n - 1] += (logistic(lin_comb) - b[i]);
        }
    }
}

/////////////////////////////////////////////////////////////////////

template <typename T>
da_status linear_model<T>::init_opt_model(fit_opt_type opt_type, objfun_t<T> objfun,
                                          objgrd_t<T> objgrd) {
    switch (opt_type) {
    case fit_opt_nln:
        opt = new da_optimization<T>();
        opt->declare_vars(ncoef);
        opt->select_solver(solver_lbfgsb);
        opt->user_objective(objfun);
        opt->user_gradient(objgrd);
        init_usrdata();
        break;

    default:
        return da_status_internal_error;
    }

    return da_status_success;
}

template <typename T> da_status linear_model<T>::get_coef(da_int &nx, T *x) {
    if (!model_trained)
        return da_status_out_of_date;
    if (nx != ncoef) {
        nx = ncoef;
        return da_status_invalid_input;
    }
    if (x == nullptr)
        return da_status_invalid_input;

    da_int i;
    for (i = 0; i < ncoef; i++)
        x[i] = coef[i];

    return da_status_success;
}

template <typename T>
da_status linear_model<T>::evaluate_model(da_int n, da_int m, T *X, T *predictions) {
    da_int i;

    if (n != this->n || m <= 0)
        return da_status_invalid_input;
    if (X == nullptr || predictions == nullptr)
        return da_status_invalid_pointer;
    if (!model_trained)
        return da_status_out_of_date;

    // X is assumed to be of shape (m,n)
    // b is assumed to be of size m
    // start by computing X*coef = predictions
    T alpha = 1.0, beta = 0.0;
    da_blas::cblas_gemv(CblasColMajor, CblasNoTrans, m, n, alpha, X, m, coef.data(), 1,
                        beta, predictions, 1);
    if (intercept) {
        for (i = 0; i < m; i++)
            predictions[i] += coef[ncoef - 1];
    }
    switch (mod) {
    case linmod_model_mse:
        for (i = 0; i < m; i++)
            predictions[i] -= b[i];
        break;
    case linmod_model_logistic:
        for (i = 0; i < m; i++)
            predictions[i] = logistic(predictions[i]);
        break;

    default:
        return da_status_not_implemented;
        break;
    }

    return da_status_success;
}

template <typename T> da_status linear_model<T>::fit() {

    if (model_trained)
        return da_status_success;

    da_int id, intercept_int;
    std::string opt_val;
    da_status status;

    opts.get("linmod intercept", intercept_int);
    opts.get("linmod norm2 reg", l_nrm2);
    opts.get("linmod norm1 reg", l_nrm1);
    opts.get("linmod optim method", opt_val, id);
    intercept = (bool)intercept_int;

    status = validate_options(id);
    if (status != da_status_success)
        return status;
    if (opt_val == "auto") {
        status = choose_method();
        if (status != da_status_success)
            return status;
        opts.get("linmod optim method", opt_val, id);
    }

    ncoef = n;
    if (intercept)
        ncoef += 1;
    coef.resize(ncoef, 0.0);

    switch (mod) {
    case linmod_model_mse:
        switch (id) {
        case 1:
            // Call LBFGS
            init_opt_model(fit_opt_nln, &objfun_mse<T>, &objgrd_mse<T>);
            opt->solve(coef, usrdata);
            break;

        case 2:
            // No regularization, standard linear least-squares through QR factorization
            qr_lsq();
            break;

        default:
            // cannot happen
            return da_status_internal_error;
            break;
        }
        break;

    case linmod_model_logistic:
        //intercept = true;
        init_opt_model(fit_opt_nln, &objfun_logistic<T>, &objgrd_logistic<T>);
        opt->solve(coef, usrdata);
        break;

    default:
        return da_status_not_implemented;
    }

    model_trained = true;
    return da_status_success;
}

template <typename T> da_status linear_model<T>::init_qr_data() {
    qr = new qr_data<T>();
    qr->A.resize(m * ncoef);
    for (da_int j = 0; j < n; j++) {
        for (da_int i = 0; i < m; i++) {
            qr->A[j * m + i] = A[j * m + i];
        }
    }
    if (intercept) {
        for (da_int i = 0; i < m; i++)
            qr->A[n * m + i] = 1.0;
    }
    qr->b.resize(m);
    for (da_int i = 0; i < m; i++)
        qr->b[i] = b[i];
    qr->tau.resize(std::min(m, ncoef));
    qr->lwork = ncoef;
    qr->work.resize(qr->lwork);

    return da_status_success;
}

/* Compute least squares factorization from QR factorization */
template <typename T> da_status linear_model<T>::qr_lsq() {
    /* has to be called after init_qr_data, qr_data is always allocated */
    /* initialize qr struct memory*/
    da_status status;
    status = init_qr_data();
    if (status != da_status_success)
        return status;

    /* Compute QR factorization */
    da_int info;
    da::geqrf(&m, &ncoef, qr->A.data(), &m, qr->tau.data(), qr->work.data(), &qr->lwork,
              &info);
    if (info != 0)
        return da_status_internal_error;
    /* Compute Q^tb*/
    char side = 'L', trans = 'T';
    da_int nrhs = 1;
    da::ormqr(&side, &trans, &m, &nrhs, &ncoef, qr->A.data(), &m, qr->tau.data(),
              qr->b.data(), &m, qr->work.data(), &qr->lwork, &info);
    if (info != 0)
        return da_status_internal_error;
    /* triangle solve R^-t*Q^Tb */
    char uplo = 'U', diag = 'N';
    trans = 'N';
    da::trtrs(&uplo, &trans, &diag, &ncoef, &nrhs, qr->A.data(), &m, qr->b.data(), &m,
              &info);
    if (info != 0)
        return da_status_internal_error;
    for (da_int i = 0; i < ncoef; i++)
        coef[i] = qr->b[i];

    return da_status_success;
}

/* Option methods */
template <typename T> da_status linear_model<T>::validate_options(da_int method) {

    if (l_nrm1 != 0 || l_nrm2 != 0)
        return da_status_not_implemented;
    switch (mod) {
    case (linmod_model_logistic):
        if (method == 2)
            // QR not valid for logistic regression
            return da_status_incompatible_options;
        break;
    default:
        break;
    }
    return da_status_success;
}

template <typename T> da_status linear_model<T>::choose_method() {
    switch (mod) {
    case (linmod_model_mse):
        if (l_nrm1 == 0.0 && l_nrm2 == 0.0)
            opts.set("linmod optim method", "qr", da_options::solver);
        else
            opts.set("linmod optim method", "lbfgs", da_options::solver);
        break;
    case (linmod_model_logistic):
        opts.set("linmod optim method", "lbfgs", da_options::solver);
        break;
    default:
        // shouldn't happen
        return da_status_internal_error;
    }

    return da_status_success;
}

/* To remove once option setter is done */
template <typename T> inline void linear_model<T>::set_reg_nrm2(T l_nrm2) {
    this->l_nrm2 = l_nrm2;
}

template <typename T> inline void linear_model<T>::set_reg_nrm1(T l_nrm1) {
    this->l_nrm1 = l_nrm1;
}
#endif