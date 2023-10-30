#ifndef LINMOD_NLN_OPTIM_HPP
#define LINMOD_NLN_OPTIM_HPP

#include "aoclda.h"
#include "da_cblas.hh"

/* Data structure containing all the optimization problem information
 * Intended to be passed along all callbacks
 */
template <typename T> struct fit_usrdata {
    /* m: number of samples
     * nfeatures: number of features
     */
    da_int m, nfeatures;
    /* Feature matrix of size (m x n)*/
    T *A = nullptr;
    /* Responce vector */
    T *b = nullptr;
    /* y=A*coef, but can also contain residuals. */
    T *y = nullptr;

    /* additional auxiliary memory for logistic regression */
    std::vector<T> aux;

    /* Intercept */
    bool intercept = false;
    /* Additional paremeters that enhance the model
      Transform on the residuals, loss function and regularization */
    T l1reg = 0.0;
    T l2reg = 0.0;
    /* T chauchy_d = 0.0; Add Cauchy loss function (and also atan SmoothL1, quantile?, huber) */

    /* linear classification parameters */
    da_int nclass = 0;

    /* constructor */
    fit_usrdata(){};
    fit_usrdata(T *A, T *b, da_int m, da_int nfeatures, bool intercept, T lambda, T alpha,
                da_int nclass, linmod_model mod)
        : m(m), nfeatures(nfeatures), A(A), b(b), intercept(intercept), nclass(nclass) {
        l1reg = lambda * alpha;
        l2reg = lambda * ((T)1.0 - alpha) / (T)2.0;
        y = new T[m];
        if (mod == linmod_model_logistic)
            aux.resize(m);
    };
};

/* Evaluate feature matrix and store result in (fir_usrdata) usrdata 
 * result is stored in usrdata->y = Ax (+ o)
 * o is a vector of one added if the intercept variable is defined 
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

/* Callbacks for the various models 
 * Intended for a nonlinear unconstrained solver of AOCL-DA
 */

/* Logistic regression callbacks
 * Computes the inverse of the log-likelihood of the logistic refression model
 * and its gradient as defined in ESL
 */
template <typename T>
da_int objfun_logistic([[maybe_unused]] da_int n, T *x, T *f, void *usrdata) {

    // All data related to the regression problem is stored in the usrdata pointer
    // multinomial problem with K (nclass) classes (indexed in [0, K-1]), nfeat features and m samples.
    // x is of size (nfeat+itpt)*(K-1)
    // where itpt is 1 if the intercept is required and 0 otherwise
    // with nmod = (nfeat+itpt), the parameter corresponding to the class k (k in 0,..,K-2)

    fit_usrdata<T> *data = (fit_usrdata<T> *)usrdata;
    T *y = data->y;
    T *b = data->b;
    da_int nclass = data->nclass;
    da_int nfeat = data->nfeatures;
    da_int m = data->m;
    da_int nmod = data->intercept ? nfeat + 1 : nfeat;

    *f = 0;
    std::fill(data->aux.begin(), data->aux.end(), 1);
    // Store in data->aux: 1+sum_nclass(exp(Beta^T x)
    // for the m samples in the input matrix
    // Also add indicator(i, k) * X_i^T * Beta_k to the objective where k is the class of the sample i
    for (da_int k = 0; k < nclass - 1; k++) {
        eval_feature_matrix(nmod, &x[k * nmod], usrdata);
        for (da_int i = 0; i < data->m; i++) {
            if (std::round(b[i]) == k)
                *f -= y[i];
            data->aux[i] += exp(y[i]);
        }
    }
    for (da_int i = 0; i < m; i++) {
        *f += log(data->aux[i]);
    }

    // Add regularization (exclude intercept)
    *f += regfun(usrdata, data->nfeatures, x);

    return 0;
}

template <typename T>
da_int objgrd_logistic(da_int n, T *x, T *grad, void *usrdata,
                       [[maybe_unused]] da_int xnew) {

    fit_usrdata<T> *data = (fit_usrdata<T> *)usrdata;
    T *y = data->y;
    T *b = data->b;
    da_int m = data->m;
    T *A = data->A;
    da_int idc = data->intercept ? 1 : 0;
    da_int nclass = data->nclass;
    da_int nmod = data->intercept ? data->nfeatures + 1 : data->nfeatures;

    // Store in data->aux: 1+sum_nclass(exp(Beta^T x)
    // for the m samples in the input matrix
    std::fill(data->aux.begin(), data->aux.end(), 1.);
    for (da_int k = 0; k < nclass - 1; k++) {
        eval_feature_matrix(nmod, &x[k * nmod], usrdata);
        for (da_int i = 0; i < m; i++) {
            data->aux[i] += exp(y[i]);
        }
    }

    // compute for all samples i and all variables j with k being the class of sample i:
    // A_ij * (indicator(i, k) - prob(x_i=k|Beta))
    T c_exp;
    std::fill(grad, grad + n, 0);
    for (da_int i = 0; i < m; i++) {
        for (da_int k = 0; k < nclass - 1; k++) {
            c_exp = da_blas::cblas_dot(nmod - idc, &x[k * nmod], 1, &A[i], m);
            if (data->intercept)
                c_exp += x[(k + 1) * nmod - 1];
            c_exp = -exp(c_exp) / data->aux[i];
            if (std::round(b[i]) == k)
                c_exp += 1.;
            for (da_int j = 0; j < nmod - idc; j++) {
                grad[k * nmod + j] -= A[m * j + i] * c_exp;
            }
            if (data->intercept) {
                grad[(k + 1) * nmod - 1] -= c_exp;
            }
        }
    }

    // NOTE: This could be made simpler by using more working memory (nclass-1*m auxiliary vector) 

    // Add regularization (exclude intercept)
    reggrd(usrdata, data->nfeatures, x, grad);

    return 0;
}

/* Mean square error callbacks */
template <typename T> da_int objfun_mse(da_int n, T *x, T *f, void *usrdata) {

    fit_usrdata<T> *data = (fit_usrdata<T> *)usrdata;
    da_int m = data->m;
    T *y = data->y;
    T *b = data->b;
    *f = 0;

    // Compute y = A*x (+ itct)
    eval_feature_matrix(n, x, usrdata);

    // y = y - b
    T alpha = -1.0;
    da_blas::cblas_axpy(m, alpha, b, 1, y, 1);

    // sum (A * x (+itct) - b)^2
    for (da_int i = 0; i < m; i++) {
        *f += pow(data->y[i], (T)2.0);
    }

    // Add regularization (exclude intercept)
    da_int nmod = data->intercept ? n - 1 : n;
    *f += regfun(usrdata, nmod, x);

    return 0;
}

template <typename T>
da_int objgrd_mse(da_int n, T *x, T *grad, void *usrdata, [[maybe_unused]] da_int xnew) {

    fit_usrdata<T> *data = (fit_usrdata<T> *)usrdata;
    da_int m = data->m;
    T *y = data->y;

    // y = A*x (+ itct)
    eval_feature_matrix(n, x, usrdata);

    // y = y - b
    T alpha = -1.0;
    da_blas::cblas_axpy(m, alpha, data->b, 1, data->y, 1);

    alpha = 2.0;
    T beta = 0.0;
    da_int aux = data->intercept ? 1 : 0;
    da_blas::cblas_gemv(CblasColMajor, CblasTrans, m, n - aux, alpha, data->A, m, y, 1,
                        beta, grad, 1);
    if (data->intercept) {
        grad[n - 1] = 0;
        for (da_int i = 0; i < m; i++)
            grad[n - 1] += alpha * y[i];
    }

    // Add regularization (exclude intercept)
    da_int nmod = data->intercept ? n - 1 : n;
    reggrd(usrdata, nmod, x, grad);

    return 0;
}


#endif