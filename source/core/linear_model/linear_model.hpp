#ifndef LINEAR_MODEL_HPP
#define LINEAR_MODEL_HPP

#include "aoclda.h"
#include "basic_handle.hpp"
#include "callbacks.hpp"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "lapack_templates.hpp"
#include "linmod_options.hpp"
#include "optimization.hpp"
#include "options.hpp"
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

#undef min
#undef max

/* Linear Models
 *
 * Solve the problem   minimize   f(x) = \sum_{i=0}^{nres-1} \Xi ( \psi(yi, \phi(xi;t)) ) + eta(xi)
 *                   x \in R^nvar
 * where
 *  * \Xi() is the loss function
 *      * MSE (mean square error) or SEL (squared-error loss) or L2 loss
 *        \Xi(ri) = ri^2 [should not be used with logistic transform]
 *      * Logistic (uses log loss)
 *        \Xi(ri) = log_loss(bi, ri) [only to be used with logistic transform]
 *
 *  * \psi() estimates the transform of the residual,
 *         and \phi is the linear model e.g. \phi(x) = Ax
 *
 *  * \eta is the regularization term
 * 
 *  FIXME ADD BOX BOUNDS
 *
 */

/* Type of transforms for the residuals
 * residual(xi) = \psi(yi, \phi(xi;t))
 * with phi the linear model e.g. \phi(x) = Ax
 *
 * TRANSFORMS
 * identity / residual: \psi(u, v) = u - v
 * logistic: psi(u, v) = logistic(u)
 */
enum transform_t { trn_identity = 0, trn_residual = 0, trn_logistic = 1 };

/* Loss functions
 *      * MSE (mean square error) or SEL (squared-error loss) or L2 loss
 *        \Xi(ri) = ri^2 [should not be used with logistic transform]
 *      * Logistic (uses log loss)
 *        \Xi(ri) = log_loss(bi, ri) [only to be used with logistic transform]
 */
enum loss_t { loss_mse, loss_logistic };

/* User and solver data
 * struct to pass pointers to optimization callbacks
 */
template <typename T> struct fit_usrdata {
    // Number of parameters
    da_int m;
    // Feature matrix of size (m x n)
    T *A = nullptr;
    // Responce vector
    T *b = nullptr;
    // y=A*coef, but can also contain residuals. FIXME add a separate vector for res
    T *y = nullptr;
    // Intercept
    bool intercept = false;
    // Additional paremeters that enhance the model
    // Transform on the residuals, loss function and regularization
    T l1reg = 0.0;
    T l2reg = 0.0;
    enum transform_t trn = trn_identity;
    enum loss_t loss = loss_mse;
    // T chauchy_d = 0.0; Add Cauchy loss function (and also atan SmoothL1, quantile?, huber)
};

// data for QR factorization used in standard linear least squares
template <typename T> struct qr_data {
    // A needs to be copied as lapack's dgeqr modifies the matrix
    std::vector<T> A, b, tau, work;
    da_int lwork = 0;
};

enum fit_opt_type { fit_opt_nln = 0, fit_opt_lsq, fit_opt_coord };

template <typename T> class linear_model : public basic_handle<T> {
  private:
    /* type of the model, has to de set at initialization phase */
    linmod_model mod = linmod_model_undefined;

    /* pointer to error trace */
    da_errors::da_error_t *err = nullptr;

    /* true if the model has been successfully trained */
    bool model_trained = false;

    /* Regression data
     * n: number of features
     * m: number of data points
     * intercept: controls if the linear regression intercept is to be set
     * A[m*n]: feature matrix, pointer to user data directly - will not be modified by any function 
     * b[m]: model response, pointer to user data - will not be modified by any function
     */
    da_int n = 0, m = 0;
    bool intercept = false;
    T *b = nullptr;
    T *A = nullptr;

    /* Training data
     * coef[n/n+1]: vector containing the trained coefficients of the model
     * l1reg/2: regularization factors for norm 1 and 2 respectively.
     *           0 => no reg, possible different algorithm
     */
    da_int ncoef = 0;
    std::vector<T> coef;

    // Elastic net penalty parameters (Regularization L1: LASSO, L2: Ridge, combination => Elastic net)
    // Penalty parameters are: lambda ( (1-alpha)L2 + alpha*L1 )
    // lambda >= 0 and 0<=alpha<=1.
    T alpha, lambda;
    // Transform operator
    enum transform_t trn = trn_identity;
    // Loss function
    enum loss_t loss = loss_mse;

    /* optimization object to call generic algorithms */
    optim::da_optimization<T> *opt = nullptr;
    fit_usrdata<T> *usrdata = nullptr;
    qr_data<T> *qr = nullptr;

    /* private methods to allocate memory */
    da_status init_opt_method(std::string &method, da_int mid);
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
    linear_model(da_errors::da_error_t &err) {
        // assumes that err is valid
        this->err = &err;
        register_linmod_options<T>(opts);
    }
    ~linear_model();

    da_status define_features(da_int n, da_int m, T *A, T *b);
    da_status select_model(linmod_model mod);
    da_status fit(da_int ncoefs, const T *coefs);
    da_status get_coef(da_int &nx, T *x);
    da_status evaluate_model(da_int n, da_int m, T *X, T *predictions);

    /* get_result (required to be defined by basic_handle) */
    da_status get_result(da_result query, da_int *dim, T *result) {
        // Don't return anything if model not trained!
        if (!model_trained)
            return da_warn(this->err, da_status_unknown_query,
                           "Handle does not contain data relevant to this query. Was the "
                           "last call to the solver successful?");
        switch (query) {
        case da_result::da_rinfo:
            if (*dim < 100) {
                *dim = 100;
                return da_warn(this->err, da_status_invalid_array_dimension,
                               "Size of the array is too small, provide an array of at "
                               "least size: " +
                                   std::to_string(*dim) + ".");
            }
            //TODO FIXME ADD DOCUMENTATION of these "rinfo"
            result[0] = (T)this->n;
            result[1] = (T)this->m;
            result[2] = (T)this->ncoef;
            result[3] = (T)(this->intercept ? 1.0 : 0.0);
            result[4] = alpha;
            result[5] = lambda;
            result[6] = (T)this->trn;
            result[7] = (T)this->loss;
            // Reserved for future use
            for (auto i = 8; i < 100; i++)
                result[i] = static_cast<T>(0);
            return da_status_success;
            break;

        case da_result::da_linmod_coeff:
            return this->get_coef(*dim, result);
            break;

        default:
            return da_warn(this->err, da_status_unknown_query,
                           "The requested result could not be queried by this handle.");
        }
    };

    da_status get_result(da_result query, da_int *dim, da_int *result) {
        return da_warn(this->err, da_status_unknown_query,
                       "Handle does not contain data relevant to this query. Was the "
                       "last call to the solver successful?");
    };
};

template <typename T> linear_model<T>::~linear_model() {
    // A and b are passed from (internal) user, do not deallocate
    A = nullptr;
    b = nullptr;
    err = nullptr;

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
};

template <typename T>
da_status linear_model<T>::define_features(da_int n, da_int m, T *A, T *b) {
    if (n <= 0 || m <= 0 || A == nullptr || b == nullptr)
        return da_error(this->err, da_status_invalid_input,
                        "Either n, m, A, or b are not valid pointers.");

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
    usrdata->l1reg = lambda * alpha;
    usrdata->l2reg = lambda * ((T)1.0 - alpha) / (T)2.0;
    usrdata->loss = loss;
    usrdata->trn = trn;
}

/* Evaluate feature matrix and store result y = Ax
 * takes care of intercept
 */
template <typename T> void eval_feature_matrix(da_int n, T *x, void *usrdata) {
    fit_usrdata<T> *data;
    data = (fit_usrdata<T> *)usrdata;
    da_int m{data->m};
    T alpha{1}, beta{0};
    T *y{data->y};
    da_int aux = data->intercept ? 1 : 0;
    da_blas::cblas_gemv(CblasColMajor, CblasNoTrans, data->m, n - aux, alpha, data->A, m,
                        x, 1, beta, y, 1);
    if (data->intercept) {
        for (da_int i = 0; i < m; i++)
            y[i] += x[n - 1];
    }
}

template <typename T> T log_loss(T y, T p) { return -y * log(p) - (1 - y) * log(1 - p); }
template <typename T> T logistic(T x) { return 1 / (1 + exp(-x)); }

template <typename T>
void eval_residuals(void *usrdata, enum transform_t transform = trn_residual) {
    fit_usrdata<T> *data = (fit_usrdata<T> *)usrdata;
    da_int m = data->m;
    T *y = data->y;
    T alpha = -1.0;
    switch (transform) {
    case trn_identity: // same as trn_residual
        da_blas::cblas_axpy(m, alpha, data->b, 1, data->y, 1);
        break;
    case trn_logistic: // return y = logistic(A*x), b is NOT substracted here
        for (da_int i = 0; i < m; i++)
            y[i] = logistic(y[i]);
        break;
    }
}

template <typename T> T lossfun(void *usrdata, loss_t loss = loss_mse) {
    fit_usrdata<T> *data = (fit_usrdata<T> *)usrdata;
    da_int m = data->m;
    T *y = data->y;
    T *b = data->b;
    T f = 0.0;

    switch (loss) {
    case loss_mse:
        for (da_int i = 0; i < m; i++) {
            f += pow(data->y[i], (T)2.0);
        }
        break;
    case loss_logistic:
        // sum of log loss of logistic function for all observations
        for (da_int i = 0; i < m; i++) {
            f += log_loss(b[i], y[i]);
        }
        break;
    }
    return f;
}

template <typename T>
void lossgrd(da_int n, T *grad, void *usrdata, loss_t loss = loss_mse) {
    fit_usrdata<T> *data = (fit_usrdata<T> *)usrdata;
    da_int m = data->m;
    T *y = data->y;
    T *b = data->b;
    T *A = data->A;
    da_int aux = data->intercept ? 1 : 0;
    T alpha{0}, beta{0};

    switch (loss) {
    case loss_mse:
        alpha = 2;
        da_blas::cblas_gemv(CblasColMajor, CblasTrans, m, n - aux, alpha, data->A, m, y,
                            1, beta, grad, 1);
        if (data->intercept) {
            grad[n - 1] = 0;
            for (da_int i = 0; i < m; i++)
                grad[n - 1] += alpha * y[i];
        }
        break;
    case loss_logistic:
        /* gradient of log loss of the logistic function
         * g_j = sum_i{A_ij*(b[i]-logistic(A_i^t x + x[n-1]))}
         */
        std::fill(grad, grad + n, 0);
        for (da_int i = 0; i < m; i++) {
            for (da_int j = 0; j < n - aux; j++)
                grad[j] += (y[i] - b[i]) * A[m * j + i];
        }
        if (data->intercept) {
            grad[n - 1] = 0.0;
            for (da_int i = 0; i < m; i++) {
                grad[n - 1] += (y[i] - b[i]);
            }
        }
        break;
    }
}

/* Add regularization, l1 and l2 terms */
template <typename T> T regfun(void *usrdata, da_int n, T const *x) {
    fit_usrdata<T> *data = (fit_usrdata<T> *)usrdata;
    T l1{data->l1reg};
    T l2{data->l2reg};
    T f1{0}, f2{0};

    if (l1 > 0) {
        // Add LASSO term
        for (da_int i = 0; i < n; i++) {
            f1 += fabs(x[i]);
        }
        f1 *= l1;
    }

    if (l2 > 0) {
        // Add Ridge term
        for (da_int i = 0; i < n; i++) {
            f2 += x[i] * x[i];
        }
        f2 *= l2;
    }
    return f1 + f2;
}

/* Add regularization, l1 and l2 term derivatives */
template <typename T> void reggrd(void *usrdata, da_int n, T const *x, T *grad) {
    fit_usrdata<T> *data = (fit_usrdata<T> *)usrdata;
    T l1{data->l1reg};
    T l2{data->l2reg};

    if (l1 > 0) {
        // Add LASSO term
        for (da_int i = 0; i < n; i++) {
            // at xi = 0 there is no derivative => set to 0
            if (x[i] != 0)
                grad[i] += x[i] < 0 ? -l1 : l1;
        }
    }

    if (l2 > 0) {
        // Add Ridge term
        for (da_int i = 0; i < n; i++) {
            grad[i] += 2 * l2 * x[i];
        }
    }
}

//////////////////////// Objective Functions ////////////////////////
/* Helper callback to deal with model, transform, loss, regularization */
template <typename T> da_int objfun(da_int n, T *x, T *f, void *usrdata) {
    fit_usrdata<T> *data = (fit_usrdata<T> *)usrdata;
    // Compute A*x = *y (takes care of intercept)
    eval_feature_matrix(n, x, usrdata);
    // Evaluate residuals and apply transform, store in: *y
    eval_residuals<T>(usrdata, data->trn);
    // loss_function
    *f = lossfun<T>(usrdata, data->loss);
    // Add regularization (exclude intercept)
    da_int nmod = data->intercept ? n - 1 : n;
    *f += regfun(usrdata, nmod, x);
    return 0;
}

/* Helper callback to deal with model, transform, loss, regularization */
template <typename T> da_int objgrd(da_int n, T *x, T *grad, void *usrdata, da_int xnew) {
    fit_usrdata<T> *data = (fit_usrdata<T> *)usrdata;
    if (xnew) {
        // Compute A*x = *y (takes care of intercept)
        eval_feature_matrix(n, x, usrdata);
        // Evaluate residuals and apply transform, store in: *y
        eval_residuals<T>(usrdata, data->trn);
    }
    // loss_function
    lossgrd(n, grad, usrdata, data->loss);
    // Add regularization (exclude intercept)
    da_int nmod = data->intercept ? n - 1 : n;
    reggrd(usrdata, nmod, x, grad);
    return 0;
}

/* Helper callback to get step for coordinate descent method */
template <typename T>
da_int stepfun(da_int n, T *x, T *step, da_int k, T *f, void *usrdata, da_int action) {
    if (k >= n || k < 0) { // FIXME: Assume and remove this if
        return 1;
    }
    fit_usrdata<T> *data = (fit_usrdata<T> *)usrdata;
    // FIXME: we need a "sparse" eval_feature_matrix since not much changed
    // from the last call? This can be handled by the caller with active set.
    // POSSIBLE ACTIONS regarting feature matrix evaluation
    // action < 0 means that feature matrix was previously called and that only a low rank
    //            update is requested and -action contains the previous k that changed 
    //            kold = -action;
    // action = 0 means not to evaluate the feature matrix
    // action > 0 evaluate the matrix.
    // FIXME, for now action < 0 is an alias to action > 0.
    // FIXME, action for now MUST always be 1 since data->y is later reused to
    // store the residuals, destroying the y = Matrix-Vector product.
    // Add new entry to fit_usrdata to duplicate the vector?
    if (action) {
        // Compute A*x = *y (takes care of intercept)
        eval_feature_matrix(n, x, usrdata);
        // FIXME copy vector data->y
    }
    // FIXME: this all should be encapsulated into a "presidual<T>(usrdata, data->loss)"
    // For now it only deals with MSE Loss
    T betak{0};
    da_int nmod = data->intercept ? n - 1 : n;
    auto sign = [](T num) {
        const T absnum = std::abs(num);
        return (absnum == (T)0 ? (T)0 : num / absnum);
    };
    auto soft = [sign](T b, T Gamma) {
        return (sign(b) * std::max(std::abs(b) - Gamma, (T)0));
    };
    T ak;
    T presidual;
    da_int m = data->m;
    if (k < nmod) {
        // handle model coefficients beta1..betaN
        if (x[k] != (T)0) {
            for (da_int i = 0; i < m; i++) {
                ak = data->A[k * m + i];
                presidual = data->b[i] - (data->y[i] - ak * x[k]);
                betak += presidual * ak;
            }
        } else {
            for (da_int i = 0; i < m; i++) {
                ak = data->A[k * m + i];
                presidual = data->b[i] - data->y[i];
                betak += presidual * ak;
            }
        }
    } else {
        // handle intercept beta0 (last element of x)
        if (x[k] != (T)0) {
            for (da_int i = 0; i < m; i++) {
                presidual = data->b[i] - (data->y[i] - x[k]);
                betak += presidual;
            }
        } else {
            for (da_int i = 0; i < m; i++) {
                presidual = data->b[i] - data->y[i];
                betak += presidual;
            }
        }
    }

    // Evaluate residuals and apply transform, store in: *y
    eval_residuals<T>(usrdata, data->trn);
    // loss_function
    *f = lossfun<T>(usrdata, data->loss);
    // Add regularization (exclude intercept)
    *f += regfun(usrdata, nmod, x);

    *step = soft(betak, data->l1reg) / ((T)1 + data->l2reg);

    return 0;
}

/////////////////////////////////////////////////////////////////////

/*
 * Common setting for all optimization solvers for linear models
 */
template <typename T>
da_status linear_model<T>::init_opt_method(std::string &method, da_int mid) {
    da_status status;
    da_int maxit, prnlvl;
    std::string slv, prnopt, optstr;
    T tol, factr;

    switch (mid) {
    case optim::solvers::solver_lbfgsb:
        slv = "lbfgsb";
        [[fallthrough]];

    case optim::solvers::solver_coord:
        if (slv == "")
            slv = "coord";
        opt = new optim::da_optimization<T>(status, *(this->err));
        if (status != da_status_success) {
            opt = nullptr;
            // this->err is already populated
            return status;
        }
        if (opt->add_vars(ncoef) != da_status_success) {
            return da_error(opt->err, da_status_internal_error,
                            "Unexpectedly linear model provided for the optimization "
                            "problem has an invalid number of coefficients ncoef=" +
                                std::to_string(ncoef) + ", expecting ncoef > 0.");
        }
        // Set options here
        // Pass print level option from linmod to optimization
        if (this->opts.get("print level", prnlvl) != da_status_success) {
            return da_error(
                opt->err, da_status_internal_error,
                "Unexpectedly <print level> option not found in the linear model "
                "option registry.");
        }
        // Decrease print level for optimization stage
        if (opt->opts.set("print level", std::max((da_int)0, prnlvl - 1)) !=
            da_status_success) {
            return da_error(opt->err, da_status_internal_error,
                            "Unexpectedly linear model provided an invalid value to the "
                            "<print level> option.");
        }
        // Pass print options
        if (this->opts.get("print options", prnopt) != da_status_success) {
            return da_error(
                opt->err, da_status_internal_error,
                "Unexpectedly <print options> option not found in the linear model "
                "option registry.");
        }
        if (opt->opts.set("print options", prnopt) != da_status_success) {
            return da_error(opt->err, da_status_internal_error,
                            "Unexpectedly linear model provided an invalid value to the "
                            "<print options> option.");
        }
        // Setup optimization method
        if (opt->opts.set("optim method", method) != da_status_success) {
            return da_error(opt->err, da_status_internal_error,
                            "Unexpectedly linear model provided an invalid value to the "
                            "<optim method> option.");
        }
        // Pass convergence parameters
        if (this->opts.get("linmod optim iteration limit", maxit) != da_status_success) {
            return da_error(opt->err, da_status_internal_error,
                            "Unexpectedly <linmod optim iteration limit> option not "
                            "found in the linear model "
                            "option registry.");
        }
        optstr = slv + " iteration limit";
        if (opt->opts.set(optstr, maxit) != da_status_success) {
            return da_error(opt->err, da_status_internal_error,
                            "Unexpectedly linear model provided an invalid value to the "
                            "<" +
                                optstr + "> option.");
        }
        if (this->opts.get("linmod optim convergence tol", tol) != da_status_success) {
            return da_error(opt->err, da_status_internal_error,
                            "Unexpectedly <linmod optim convergence tol> option not "
                            "found in the linear model option registry.");
        }
        optstr = slv + " convergence tol";
        if (opt->opts.set(optstr, tol) != da_status_success) {
            return da_error(
                opt->err, da_status_internal_error,
                "Unexpectedly linear model provided an invalid value to the <" + optstr +
                    "> option.");
        }
        if (this->opts.get("linmod optim progress factor", factr) != da_status_success) {
            return da_error(opt->err, da_status_internal_error,
                            "Unexpectedly <linmod optim progress factor> option not "
                            "found in the linear model option registry.");
        }
        optstr = slv + " progress factor";
        if (opt->opts.set(optstr, factr) != da_status_success) {
            return da_error(
                opt->err, da_status_internal_error,
                "Unexpectedly linear model provided an invalid value to the <" + optstr +
                    "> option.");
        }

        init_usrdata();
        break;

    default:
        return da_error(opt->err, da_status_internal_error,
                        "Unexpected optimization problem class requested.");
    }

    return da_status_success;
}

template <typename T> da_status linear_model<T>::get_coef(da_int &nx, T *x) {
    if (!model_trained)
        return da_error(this->err, da_status_out_of_date,
                        "The data associated to the model is out of date.");
    if (nx != ncoef) {
        nx = ncoef;
        return da_warn(this->err, da_status_invalid_array_dimension,
                       "The number of coefficients is wrong, correct size is " +
                           std::to_string(ncoef) + ".");
    }
    if (x == nullptr)
        return da_error(this->err, da_status_invalid_input,
                        "Argument x needs to provide a valid pointer of at least size " +
                            std::to_string(ncoef) + ".");

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
        return da_error(this->err, da_status_not_implemented,
                        "No optimization solver for the requested linear model is "
                        "available, reformulate linear model problem.");
        break;
    }

    return da_status_success;
}

template <typename T> da_status linear_model<T>::fit(da_int ncoefs, const T *coefs) {

    if (model_trained)
        return da_status_success;

    da_int mid, prn, intercept_int;
    std::string val, method;
    da_status status;

    // For all opts.get() it is assumed they don't fail
    opts.get("print options", val, prn);
    if (prn != 0)
        opts.print_options();

    opts.get("linmod intercept", intercept_int);
    opts.get("linmod alpha", this->alpha);
    opts.get("linmod lambda", this->lambda);
    opts.get("linmod optim method", method, mid);
    this->intercept = (bool)intercept_int;

    status = validate_options(mid);
    if (status != da_status_success) {
        // this->err already populated
        return status;
    }
    if (method == "auto") {
        status = choose_method();
        if (status != da_status_success) {
            // this->err already populated
            return status;
        }
    }
    opts.get("linmod optim method", method, mid);

    ncoef = n;
    if (intercept)
        ncoef += 1;

    bool copycoefs = (coefs != nullptr) && (ncoefs >= n);

    if (copycoefs) {
        coef.resize(ncoef);
        // user provided starting coefficients, check, copy and use.
        // copy first n elements, then check the intercept
        for (da_int j = 0; j < n; j++)
            coef[j] = coefs[j];
        if (intercept) {
            coef[ncoef - 1] = ncoefs >= ncoef ? coefs[ncoef - 1] : (T)0;
        }
    } else {
        coef.resize(ncoef, (T)0);
    }

    switch (mod) {
    case linmod_model_mse:
        this->trn = trn_residual;
        this->loss = loss_mse;
        switch (mid) {
        case optim::solvers::solver_lbfgsb:
            // Call LBFGS
            status = init_opt_method(method, mid);
            if (status != da_status_success)
                return da_error(
                    this->err, da_status_internal_error,
                    "Unexpectedly could not initialize an optimization model.");
            // Add callbacks
            if (opt->add_objfun(objfun<T>) != da_status_success) {
                return da_error(opt->err, da_status_internal_error,
                                "Unexpectedly linear model provided an invalid objective "
                                "function pointer.");
            }
            if (opt->add_objgrd(objgrd<T>) != da_status_success) {
                return da_error(opt->err, da_status_internal_error,
                                "Unexpectedly linear model provided an invalid objective "
                                "gradient function pointer.");
            }
            status = opt->solve(coef, usrdata);
            if (status == da_status_success ||
                this->err->get_severity() != da_errors::severity_type::DA_ERROR)
                // either success or warning with usable solution, continue
                status = this->err->clear();
            else
                status = da_error(this->err, da_status_operation_failed,
                                  "Optimization step failed, rescale problem or request "
                                  "different solver.");
            break;

        case optim::solvers::solver_qr:
            // No regularization, standard linear least-squares through QR factorization
            qr_lsq();
            break;

        case optim::solvers::solver_coord:
            // Call Coordinate Descent Method (Elastic Nets)
            status = init_opt_method(method, mid);
            if (status != da_status_success)
                return da_error(
                    this->err, da_status_internal_error,
                    "Unexpectedly could not initialize an optimization model.");
            // Add callback
            if (opt->add_stepfun(stepfun<T>) != da_status_success) {
                return da_error(opt->err, da_status_internal_error,
                                "Unexpectedly linear model provided an invalid step "
                                "function pointer.");
            }
            status = opt->solve(coef, usrdata);
            if (status == da_status_success ||
                this->err->get_severity() != da_errors::severity_type::DA_ERROR)
                // either success or warning with usable solution, continue
                status = this->err->clear();
            else
                status = da_error(this->err, da_status_operation_failed,
                                  "Optimization step failed, rescale problem or request "
                                  "different solver.");
            break;

        default:
            // should not happen
            return da_error(this->err, da_status_internal_error,
                            "Unexpectedly an invalid optimization solver was requested.");
            break;
        }
        break;

    case linmod_model_logistic:
        this->trn = trn_logistic;
        this->loss = loss_logistic;
        //intercept = true;
        status = init_opt_method(method, mid);
        if (status != da_status_success) {
            // this->err already populated
            return status;
        }
        if (opt->add_objfun(objfun<T>) != da_status_success) {
            return da_error(opt->err, da_status_internal_error,
                            "Unexpectedly linear model provided an invalid objective "
                            "function pointer.");
        }
        if (opt->add_objgrd(objgrd<T>) != da_status_success) {
            return da_error(opt->err, da_status_internal_error,
                            "Unexpectedly linear model provided an invalid objective "
                            "gradient function pointer.");
        }
        status = opt->solve(coef, usrdata);
        if (status == da_status_success ||
            this->err->get_severity() != da_errors::severity_type::DA_ERROR) {
            // Solver managed to provide a usable solution
            // Reset status and continue
            status = this->err->clear();
        } else {
            // Hard error, no usable x, terminate.
            // this->err already populated
            return status;
        }
        break;

    default:
        return da_error(this->err, da_status_not_implemented,
                        "Unexpectedly an invalid linear model was requested.");
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
    da_int info = 1;
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
    switch (mod) {
    case (linmod_model_mse):
        if (method == 2 && lambda != 0)
            return da_error(
                this->err, da_status_incompatible_options,
                "The chosen solver, QR, is incompatible with regularization. Either "
                "remove regularization or choose different solver.");
        break;
    case (linmod_model_logistic):
        if (method == 2)
            // QR not valid for logistic regression
            return da_error(this->err, da_status_incompatible_options,
                            "The chosen solver, QR, is incompatible with the logistic "
                            "regression model. Either choose different linear model or "
                            "choose different solver.");
        break;
    default:
        break;
    }
    return da_status_success;
}

template <typename T> da_status linear_model<T>::choose_method() {
    switch (mod) {
    case (linmod_model_mse):
        if (lambda == (T)0)
            // QR direct method
            opts.set("linmod optim method", "qr", da_options::solver);
        else if (alpha == (T)0)
            // L-BFGS-B handles L2 regularization
            opts.set("linmod optim method", "lbfgs", da_options::solver);
        else
            // Coordinate Descent for L1 [and L2 combined: Elastic Net]
            opts.set("linmod optim method", "coord", da_options::solver);
        break;
    case (linmod_model_logistic):
        // Here we choose L-BFGS-B over Coordinate Descent
        if (alpha == (T)0)
            // L-BFGS-B handles L2 regularization
            opts.set("linmod optim method", "lbfgs", da_options::solver);
        else
            // Coordinate Descent for L1 [and L2 combined: Elastic Net]
            // opts.set("linmod optim method", "coord", da_options::solver);
            // TODO FIXME Enable this once coord+logistic is implemented
            // --> uncomment opts.set("linmod optim method", "coord", da_options::solver);
            return da_error(this->err, da_status_not_implemented, "Not yet implemented");
        break;
    default:
        // Shouldn't happen (would be nice to trap these with C++23 std::unreachable())
        return da_error(this->err, da_status_internal_error, "New linmod model?");
    }

    return da_status_success;
}
#endif
