#ifndef NLN_REG_DATA_HPP
#define NLN_REG_DATA_HPP

#include "aoclda.h"
#include "callbacks.hpp"
#include "da_cblas.hh"
#include "optimization.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>

// used to pass pointers to optimization callbacks
template <typename T> struct fit_usrdata {
    da_int m;
    T *A = nullptr;
    T *b = nullptr, *y = nullptr;
};

enum fit_opt_type { fit_opt_nln = 0, fit_opt_lsq };

template <typename T> class linear_model_data {
  private:
    /* type of the model, has to de set at initialization phase */
    linreg_model mod = linreg_model_undefined;

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
     */
    da_int ncoef = 0;
    std::vector<T> coef;

    /* optimization object to call generic algorithms */
    da_optimization<T> *opt = nullptr;
    fit_usrdata<T> *usrdata = nullptr;

    /* private methods to allocate memory */
    da_status
    init_opt_model(fit_opt_type opt_type,
                   std::function<void(da_int n, T *x, T *f, void *usrdata)> objfun,
                   std::function<void(da_int n, T *x, T *grad, void *usrdata)> objgrd);
    void init_usrdata();

  public:
    linear_model_data(){};
    ~linear_model_data();

    da_status define_features(da_int n, da_int m, T *A, T *b);
    da_status select_model(linreg_model mod);
    da_status fit();
    da_status get_coef(da_int &nx, T *x);
    da_status evaluate_model(da_int n, da_int m, T *X, T *predictions);
};

template <typename T> linear_model_data<T>::~linear_model_data() {
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
}

template <typename T>
da_status linear_model_data<T>::define_features(da_int n, da_int m, T *A, T *b) {
    if (n <= 0 || m <= 0 || b == nullptr)
        return da_status_invalid_input;

    model_trained = false;

    this->n = n;
    this->m = m;
    // copy user's feature pointers
    this->b = b;
    this->A = A;
    // allocate enough space for the model coefficients, including a possible
    // interceptvariable coef[n+1] will contain the intercept var after fit
    coef.reserve(n + 1);

    return da_status_success;
}

template <typename T> da_status linear_model_data<T>::select_model(linreg_model mod) {

    if (mod != this->mod) {
        this->mod = mod;
        model_trained = false;
    }
    return da_status_success;
}

template <typename T> void linear_model_data<T>::init_usrdata() {
    usrdata = new fit_usrdata<T>;
    usrdata->A = A;
    usrdata->b = b;
    usrdata->m = m;
    usrdata->y = new T[m];
}

////////////////////////////////////// Objective Function
///////////////////////////////////////

template <typename T> void objfun_mse(da_int n, T *x, T *f, void *usrdata) {
    fit_usrdata<T> *data;
    data = (fit_usrdata<T> *)usrdata;

    T alpha = 1.0, beta = 0.0;

    da_blas::cblas_gemv(CblasRowMajor, CblasNoTrans, data->m, n, alpha, data->A, n, x, 1,
                        beta, data->y, 1);
    *f = 0.0;
    for (da_int i = 0; i < data->m; i++)
        *f += pow(data->y[i] - data->b[i], 2.0);
}

template <typename T> void objgrd_mse(da_int n, T *x, T *grad, void *usrdata) {
    fit_usrdata<T> *data;
    data = (fit_usrdata<T> *)usrdata;

    T alpha = 1.0, beta = 0.0;
    da_blas::cblas_gemv(CblasRowMajor, CblasNoTrans, data->m, n, alpha, data->A, n, x, 1,
                        beta, data->y, 1);
    alpha = -1.0;
    da_blas::cblas_axpy(data->m, alpha, data->b, 1, data->y, 1);
    for (da_int i = 0; i < n; i++) {
        grad[i] = 2.0 * da_blas::cblas_dot(data->m, &data->A[i], n, data->y, 1);
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
    T alpha = 1.0, beta = 0.0;
    da_blas::cblas_gemv(CblasRowMajor, CblasNoTrans, m, n - 1, alpha, A, n - 1, x, 1,
                        beta, data->y, 1);

    // sum of log loss of logistic function for all observations
    *f = 0.0;
    for (da_int i = 0; i < m; i++) {
        *f += log_loss(b[i], logistic(x[n - 1] + y[i]));
    }
}

template <typename T> void objgrd_logistic(da_int n, T *x, T *grad, void *usrdata) {
    fit_usrdata<T> *data;
    data = (fit_usrdata<T> *)usrdata;
    T *b = data->b, *y = data->y, *A = data->A;
    da_int m = data->m;

    // Comput A*x[0:n-2] = y
    T alpha = 1.0, beta = 0.0;
    da_blas::cblas_gemv(CblasRowMajor, CblasNoTrans, m, n - 1, alpha, data->A, n - 1, x,
                        1, beta, data->y, 1);

    for (da_int i = 0; i < n - 1; i++) {
        grad[i] = 0.;
        for (da_int j = 0; j < m; j++) {
            grad[i] += (logistic(x[n - 1] + y[j]) - b[j]) * A[(n - 1) * i + j];
        }
    }
    grad[n - 1] = 0.0;
    for (da_int j = 0; j < m; j++) {
        grad[n - 1] += (logistic(x[n - 1] + y[j]) - b[j]);
    }
}

/////////////////////////////////////////////////////////////////

template <typename T>
da_status linear_model_data<T>::init_opt_model(
    fit_opt_type opt_type,
    std::function<void(da_int n, T *x, T *f, void *usrdata)> objfun,
    std::function<void(da_int n, T *x, T *grad, void *usrdata)> objgrd) {
    da_int nvar = n;
    switch (opt_type) {
    case fit_opt_nln:
        opt = new da_optimization<T>();
        if (intercept)
            nvar += 1;
        opt->declare_vars(nvar);
        opt->select_solver(solver_lbfgsb);
        if (objfun == nullptr || objgrd == nullptr)
            return da_status_invalid_input;
        opt->user_objective(objfun);
        opt->user_gradient(objgrd);
        init_usrdata();
        break;

    default:
        return da_status_not_implemented;
    }

    return da_status_success;
}

template <typename T> da_status linear_model_data<T>::get_coef(da_int &nx, T *x) {
    if (nx != ncoef) {
        nx = ncoef;
        return da_status_invalid_input;
    }
    if (!model_trained)
        return da_status_out_of_date;

    da_int i;
    for (i = 0; i < ncoef; i++)
        x[i] = coef[i];

    return da_status_success;
}

template <typename T>
da_status linear_model_data<T>::evaluate_model(da_int n, da_int m, T *X, T *predictions) {
    da_int i;

    if (n != this->n)
        return da_status_invalid_input;
    if (X == nullptr || predictions == nullptr)
        return da_status_invalid_pointer;
    if (!model_trained)
        return da_status_out_of_date;

    // X is assumed to be of shape (m,n)
    // b is assumed to be of size m
    // start by computing X*coef = predictions
    T alpha = 1.0, beta = 0.0;
    da_blas::cblas_gemv(CblasRowMajor, CblasNoTrans, m, n, alpha, X, n, coef.data(), 1,
                        beta, predictions, 1);
    if (intercept) {
        for (i = 0; i < m; i++)
            predictions[i] += coef[ncoef - 1];
    }
    switch (mod) {
    case linreg_model_mse:
        for (i = 0; i < m; i++)
            predictions[i] -= b[i];
        break;
    case linreg_model_logistic:
        for (i = 0; i < m; i++)
            predictions[i] = logistic(predictions[i]);
        break;

    default:
        return da_status_not_implemented;
        break;
    }

    return da_status_success;
}

template <typename T> da_status linear_model_data<T>::fit() {

    if (model_trained)
        return da_status_success;

    switch (mod) {
    case linreg_model_mse:
        intercept = false;
        ncoef = n;
        init_opt_model(fit_opt_nln, &objfun_mse<T>, &objgrd_mse<T>);
        coef.resize(ncoef, 0.0);
        opt->solve(coef, usrdata);
        break;

    case linreg_model_logistic:
        intercept = true;
        ncoef = n + 1;
        init_opt_model(fit_opt_nln, &objfun_logistic<T>, &objgrd_logistic<T>);
        coef.resize(ncoef, 0.0); // n+1 for intercept
        opt->solve(coef, usrdata);
        break;

    default:
        return da_status_not_implemented;
    }

    model_trained = true;
    return da_status_success;
}

#endif