/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef LINMOD_NLN_OPTIM_HPP
#define LINMOD_NLN_OPTIM_HPP

#undef max

#include "aoclda.h"
#include "da_cblas.hh"
#include "linmod_types.hpp"

/* Base class of the callback user data to be passed as void pointers
 * Contain all the basic data to compute */
template <class T> class usrdata_base {
  public:
    da_int nsamples = 0, nfeat = 0;
    // Feature matrix of size (nsamples x nfeat)
    const T *X = nullptr;
    // Response vector
    const T *y = nullptr;

    // Intercept
    bool intercept = false;

    // Additional parameters that enhance the model

    // Regularization
    T l1reg = T(0);
    T l2reg = T(0);
    /* Pointer to rescaled penalty factors for each coefficient
     * in the case of standardization these are scale[k]/scale[y],
     * and are used in the regularization terms.
     * See details in the standardization function
     * pointer to an array of at least nfeat
     */
    const T *xv{nullptr};
    da_linmod::scaling_t scaling{da_linmod::scaling_t::none};

    usrdata_base(const T *X, const T *y, da_int nsamples, da_int nfeat, bool intercept,
                 T lambda, T alpha, const T *xv = nullptr,
                 da_linmod::scaling_t scaling = da_linmod::scaling_t::none)
        : nsamples(nsamples), nfeat(nfeat), X(X), y(y), intercept(intercept), xv(xv),
          scaling(scaling) {
        l1reg = lambda * alpha;
        l2reg = lambda * (T(1) - alpha) / T(2);
    }
    virtual ~usrdata_base() {}
};

/* User data for the nonlinear optimization callbacks of the logistic regression */
template <class T> class cb_usrdata_logreg : public usrdata_base<T> {
  public:
    da_int nclass;
    /* Add 4 working memory arrays
     * maxexp[nsamples]: used to store the maximum values of each X_k beta_k (k as class index) for the logsumexp trick
     * sumexp[nsamples]: used to store the sum of the exponents of each X_k beta_k (k as class index) for the logsumexp trick
     * lincomb[nsamples*(nclass-1) OR nsamples*nclass]: used to store all the X_k beta_k values
     * gradients_p[nsamples*(nclass-1) OR nsamples*nclass]: used to store all the pointwise gradients
     */
    std::vector<T> maxexp, sumexp, lincomb, gradients_p;

    cb_usrdata_logreg(const T *X, const T *y, da_int nsamples, da_int nfeat,
                      bool intercept, T lambda, T alpha, da_int nclass, da_int nparam)
        : usrdata_base<T>(X, y, nsamples, nfeat, intercept, lambda, alpha),
          nclass(nclass) {
        maxexp.resize(nsamples);
        sumexp.resize(nsamples);
        lincomb.resize(nsamples * nparam);
        gradients_p.resize(nsamples * nparam);
    }
    ~cb_usrdata_logreg() {}
};

/* User data for the nonlinear optimization callbacks of the lineqr regression */
template <class T> class cb_usrdata_linreg : public usrdata_base<T> {
  public:
    /* Add a working memory arrays
     * matvec[nsamples]: typically used to compute the matrix vector product X * coef
     */
    std::vector<T> matvec;

    cb_usrdata_linreg(const T *X, const T *y, da_int nsamples, da_int nfeat,
                      bool intercept, T lambda, T alpha)
        : usrdata_base<T>(X, y, nsamples, nfeat, intercept, lambda, alpha) {
        matvec.resize(nsamples);
    }
    ~cb_usrdata_linreg() {}
};

/* User data for the coordinate descent step function of the linear regression */
template <class T> class stepfun_usrdata_linreg : public usrdata_base<T> {
  public:
    /* Add working memory array
     * residual[nsamples]: holds the residual (and all the intermediate calculations)
     */
    std::vector<T> residual;

    stepfun_usrdata_linreg(const T *X, const T *y, da_int nsamples, da_int nfeat,
                           bool intercept, T lambda, T alpha, const T *xv,
                           da_linmod::scaling_t scaling)
        : usrdata_base<T>(X, y, nsamples, nfeat, intercept, lambda, alpha, xv, scaling) {
        residual.resize(nsamples);
    }
    ~stepfun_usrdata_linreg() { usrdata_base<T>::xv = nullptr; }
};

/* This function evaluates the feature matrix X over the parameter vector x (taking into account the intercept),
 * it performs the GEMV operation
 *
 * v = [X, 1^T] * x OR v = [X, 1^T]^T x
 */
template <typename T>
void eval_feature_matrix(da_int n, const T *x, da_int nsamples, const T *X, T *v,
                         bool intercept, bool trans = false) {
    T alpha = 1.0, beta = 0.0;

    da_int aux = intercept ? 1 : 0;
    enum CBLAS_TRANSPOSE transpose = trans ? CblasTrans : CblasNoTrans;
    da_blas::cblas_gemv(CblasColMajor, transpose, nsamples, n - aux, alpha, X, nsamples,
                        x, 1, beta, v, 1);
    if (intercept && !trans) {
        for (da_int i = 0; i < nsamples; i++)
            v[i] += x[n - 1];
    } else if (intercept && trans) {
        v[n - 1] = T(0);
        for (da_int i = 0; i < nsamples; i++)
            v[n - 1] += x[i];
    }
}

/* Add regularization, l1 and l2 terms */
template <typename T> T regfun(da_int n, const T *x, const T l1reg, const T l2reg) {
    T f1{0}, f2{0};
    if (l1reg > 0) {
        // Add LASSO term
        for (da_int i = 0; i < n; i++) {
            f1 += fabs(x[i]);
        }
        f1 *= l1reg;
    }
    if (l2reg > 0) {
        // Add Ridge term
        for (da_int i = 0; i < n; i++) {
            f2 += x[i] * x[i];
        }
        f2 *= l2reg;
    }
    return f1 + f2;
}

/* Add regularization, l1 and l2 term derivatives */
template <typename T> void reggrd(da_int n, T const *x, T l1reg, T l2reg, T *grad) {
    if (l1reg > 0) {
        // Add LASSO term
        for (da_int i = 0; i < n; i++) {
            // At xi = 0 there is no derivative => set to 0
            if (x[i] != 0)
                grad[i] += x[i] < 0 ? -l1reg : l1reg;
        }
    }
    if (l2reg > 0) {
        // Add Ridge term
        const T l2term = T(2) * l2reg;
        for (da_int i = 0; i < n; i++) {
            grad[i] += l2term * x[i];
        }
    }
}

/* Callbacks for the various models to be passed to an (un)constrained
 * nonlinear solver of AOCL-DA
 */

/* Logistic regression callbacks
 * Computes the inverse of the log-likelihood of the logistic regression model
 * and its gradient as defined in Elements of Statistical Learning (Hastie & all)
 */
template <typename T>
da_int objfun_logistic_rsc([[maybe_unused]] da_int n, T *x, T *f, void *udata) {

    // All data related to the regression problem is stored in the udata pointer
    // multinomial problem with K (nclass) classes (indexed in [0, K-1]), nfeat features and nsamples samples.
    // x is of size (nfeat+itpt)*(K-1)
    // where itpt is 1 if the intercept is required and 0 otherwise
    // with nmod = (nfeat+itpt), the parameters corresponding to the class k (k in 0,..,K-2)

    cb_usrdata_logreg<T> *data = (cb_usrdata_logreg<T> *)udata;
    std::vector<T> &maxexp = data->maxexp;
    std::vector<T> &lincomb = data->lincomb;
    T *lincomb_ptr = data->lincomb.data();
    const T *y = data->y;
    da_int nclass = data->nclass;
    da_int nfeat = data->nfeat;
    da_int nsamples = data->nsamples;
    da_int nmod = data->intercept ? nfeat + 1 : nfeat;

    // lincomb is of size nsamples*(nclass-1)
    // Store in lincomb[:,k] the Beta_k^T * x for the nsamples samples in the input matrix
    // Store in maxexp the max of lincomb for each sample
    *f = 0;
    std::fill(maxexp.begin(), maxexp.end(), 0.);
    for (da_int k = 0; k < nclass - 1; k++) {
        da_int idx = k * nsamples;
        eval_feature_matrix(nmod, &x[k * nmod], nsamples, data->X,
                            &lincomb_ptr[k * nsamples], data->intercept);
        for (da_int i = 0; i < nsamples; i++) {
            if (maxexp[i] < lincomb[idx])
                maxexp[i] = lincomb[idx];
            // Indicator(i, k) * X * x[k*nmod:(k+1)*nmod-1] added to objective
            if (std::round(y[i]) == k)
                *f -= lincomb[idx];
            idx += 1;
        }
    }

    // Compute for each sample i ln(1+sum_{k=0}^{K-2} exp(lincomb[i][k]))
    // Use logsumexp trick to avoid overflow
    for (da_int i = 0; i < nsamples; i++) {
        T val = exp(-maxexp[i]);
        for (da_int k = 0; k < nclass - 1; k++) {
            val += exp(lincomb[k * nsamples + i] - maxexp[i]);
        }
        *f += maxexp[i] + log(val);
    }

    // Add regularization (exclude intercept)
    *f += regfun(data->nfeat, x, data->l1reg, data->l2reg);

    return 0;
}

template <typename T>
da_int objgrd_logistic_rsc(da_int n, T *x, T *grad, void *udata,
                           [[maybe_unused]] da_int xnew) {

    cb_usrdata_logreg<T> *data = (cb_usrdata_logreg<T> *)udata;
    std::vector<T> &maxexp = data->maxexp;
    const T *y = data->y;
    std::vector<T> &lincomb = data->lincomb;
    T *lincomb_ptr = data->lincomb.data();
    da_int nsamples = data->nsamples;
    const T *X = data->X;
    da_int idc = data->intercept ? 1 : 0;
    da_int nclass = data->nclass;
    da_int nmod = data->intercept ? data->nfeat + 1 : data->nfeat;

    if (xnew) {
        // Store in lincomb[:,k] the Beta_k^T * x for the nsamples samples in the input matrix
        // Store in maxexp the max of lincomb for each sample
        std::fill(maxexp.begin(), maxexp.end(), 0.);
        for (da_int k = 0; k < nclass - 1; k++) {
            da_int idx = k * nsamples;
            eval_feature_matrix(nmod, &x[k * nmod], nsamples, data->X,
                                &lincomb_ptr[k * nsamples], data->intercept);
            for (da_int i = 0; i < nsamples; i++) {
                if (maxexp[i] < lincomb[idx])
                    maxexp[i] = lincomb[idx];
                idx += 1;
            }
        }
    }

    // compute for all samples i and all variables j with k being the class of sample i:
    // A_ij * (indicator(i, k) - prob(x_i=k|Beta))
    std::fill(grad, grad + n, 0);
    for (da_int i = 0; i < nsamples; i++) {
        T lnsumexp = exp(-maxexp[i]);
        for (da_int k = 0; k < nclass - 1; k++) {
            lnsumexp += exp(lincomb[k * nsamples + i] - maxexp[i]);
        }
        lnsumexp = maxexp[i] + log(lnsumexp);

        for (da_int k = 0; k < nclass - 1; k++) {
            T val = -exp(lincomb[k * nsamples + i] - lnsumexp);
            if (std::round(y[i]) == k)
                val += 1.;
            for (da_int j = 0; j < nmod - idc; j++) {
                grad[k * nmod + j] -= X[nsamples * j + i] * val;
            }
            if (data->intercept) {
                grad[(k + 1) * nmod - 1] -= val;
            }
        }
    }

    // Add regularization (exclude intercept)
    reggrd(data->nfeat, x, data->l1reg, data->l2reg, grad);

    return 0;
}

template <typename T>
da_int objfun_logistic_two_class([[maybe_unused]] da_int n, T *x, T *f, void *udata) {

    // All data related to the regression problem is stored in the udata pointer
    // two class problem indexed as 0 and 1, nfeat features and nsamples samples.
    // x is of size (nfeat+itpt)
    // where itpt is 1 if the intercept is required and 0 otherwise

    cb_usrdata_logreg<T> *data = (cb_usrdata_logreg<T> *)udata;
    std::vector<T> &lincomb = data->lincomb;
    T *lincomb_ptr = data->lincomb.data();
    const T *y = data->y;
    da_int nfeat = data->nfeat;
    da_int nsamples = data->nsamples;
    da_int nmod = data->intercept ? nfeat + 1 : nfeat;

    // Loss value
    *f = 0;

    // lincomb is of size nsamples
    // Calculate licomb as X * Beta + Beta_0
    eval_feature_matrix(nmod, x, nsamples, data->X, lincomb_ptr, data->intercept);

    // Loss is sum of log(1+exp(lincomb[i])) - y_i*lincomb[i]
    // If-else codepath to avoid overflow
    // ln(1+exp(b^Tx)) = ln(exp(b^TX)[exp(-b^TX) + 1]) = b^TX + ln(1+exp(-b^TX))
    // look at private and shared variables
    for (da_int i = 0; i < nsamples; i++) {
        if (lincomb[i] < 0)
            *f += log(1 + exp(lincomb[i])) - std::round(y[i]) * lincomb[i];
        else
            *f += log(1 + exp(-lincomb[i])) + (1 - std::round(y[i])) * lincomb[i];
    }

    // Add regularization (exclude intercept)
    *f += regfun(nfeat, x, data->l1reg, data->l2reg);

    return 0;
}

template <typename T>
da_int objgrd_logistic_two_class(da_int n, T *x, T *grad, void *udata,
                                 [[maybe_unused]] da_int xnew) {

    cb_usrdata_logreg<T> *data = (cb_usrdata_logreg<T> *)udata;
    const T *y = data->y;
    std::vector<T> &lincomb = data->lincomb;
    std::vector<T> &gradients_p = data->gradients_p;
    T *lincomb_ptr = data->lincomb.data();
    da_int nsamples = data->nsamples;
    da_int nfeat = data->nfeat;
    da_int nmod = data->intercept ? data->nfeat + 1 : data->nfeat;
    T sum_of_gradients;

    if (xnew) {
        // Calculate licomb as X * Beta + Beta_0
        eval_feature_matrix(nmod, x, nsamples, data->X, lincomb_ptr, data->intercept);
    }

    // Compute for all samples i and all variables j with k being the class of sample i:
    // A_ij^T * (sigma(Beta*x)-y_i)
    std::fill(grad, grad + n, 0);
    sum_of_gradients = 0;

    // Trick to avoid overflow uses fact that:
    // sigma(x) = 1/(1+exp(-x)) = exp(x)/1+exp(x)
    for (da_int i = 0; i < nsamples; i++) {
        if (lincomb[i] < 0)
            gradients_p[i] = exp(lincomb[i]) / (1 + exp(lincomb[i])) - std::round(y[i]);
        else
            gradients_p[i] = 1 / (1 + exp(-lincomb[i])) - std::round(y[i]);
        sum_of_gradients += gradients_p[i];
    }

    da_blas::cblas_gemv(CblasColMajor, CblasTrans, nsamples, nfeat, 1.0, data->X,
                        nsamples, gradients_p.data(), 1, 1.0, grad, 1);

    if (data->intercept) {
        grad[n - 1] = sum_of_gradients;
    }
    // Add regularization (exclude intercept)
    reggrd(nfeat, x, data->l1reg, data->l2reg, grad);

    return 0;
}

template <typename T>
da_int objfun_logistic_ssc([[maybe_unused]] da_int n, T *x, T *f, void *udata) {

    // All data related to the regression problem is stored in the udata pointer
    // multinomial problem with K (nclass) classes (indexed in [0, K-1]), nfeat features and nsamples samples.
    // x is of size (nfeat+itpt)*(K)
    // where itpt is 1 if the intercept is required and 0 otherwise
    // with nmod = (nfeat+itpt), the parameters corresponding to the class k (k in 0,..,K-1)

    cb_usrdata_logreg<T> *data = (cb_usrdata_logreg<T> *)udata;
    std::vector<T> &maxexp = data->maxexp;
    std::vector<T> &sumexp = data->sumexp;
    std::vector<T> &lincomb = data->lincomb;
    T *lincomb_ptr = data->lincomb.data();
    const T *y = data->y;
    da_int nclass = data->nclass;
    da_int nfeat = data->nfeat;
    da_int nsamples = data->nsamples;

    // lincomb is of size nsamples*nclass
    // Store in lincomb[:,k] the Beta_k^T * x for the nsamples samples in the input matrix
    // Store in maxexp the max of lincomb for each sample
    *f = 0;
    std::fill(maxexp.begin(), maxexp.end(), 0.);
    std::fill(sumexp.begin(), sumexp.end(), 0.);
    if (data->intercept) {
        for (da_int k = 0; k < nclass; k++) {
            std::fill(lincomb.begin() + k * nsamples,
                      lincomb.begin() + (k + 1) * nsamples, x[n - (nclass - k)]);
        }
    } else {
        std::fill(lincomb.begin(), lincomb.end(), 0.);
    }

    // Calculate licomb as X * Beta + Beta_0
    da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, nsamples, nclass, nfeat,
                        1.0, data->X, nsamples, x, nclass, 1.0, lincomb_ptr, nsamples);

    // look at private and shared variables
    for (da_int i = 0; i < nsamples; i++) {
        for (da_int k = 0; k < nclass; k++) {
            // Find maxexp
            if (maxexp[i] < lincomb[k * nsamples + i]) {
                maxexp[i] = lincomb[k * nsamples + i];
            }
            // Subtract the residual of correct class
            if (std::round(y[i]) == k)
                *f -= lincomb[k * nsamples + i];
        }
        // Compute for each sample i ln(sum_{k=0}^{K-1} exp(lincomb[i][k]))
        // use logsumexp trick to avoid overflow
        sumexp[i] = 0;
        for (da_int k = 0; k < nclass; k++) {
            sumexp[i] += exp(lincomb[k * nsamples + i] - maxexp[i]);
        }
        *f += maxexp[i] + log(sumexp[i]);
    }

    // Add regularization (exclude intercept)
    *f += regfun(nfeat * nclass, x, data->l1reg, data->l2reg);

    return 0;
}

template <typename T>
da_int objgrd_logistic_ssc(da_int n, T *x, T *grad, void *udata,
                           [[maybe_unused]] da_int xnew) {

    cb_usrdata_logreg<T> *data = (cb_usrdata_logreg<T> *)udata;
    std::vector<T> &maxexp = data->maxexp;
    std::vector<T> &sumexp = data->sumexp;
    std::vector<T> &gradients_p = data->gradients_p;
    const T *y = data->y;
    std::vector<T> &lincomb = data->lincomb;
    T *lincomb_ptr = data->lincomb.data();
    da_int nsamples = data->nsamples;
    da_int nclass = data->nclass;
    da_int nfeat = data->nfeat;

    if (xnew) {
        std::fill(maxexp.begin(), maxexp.end(), 0.);
        if (data->intercept) {
            for (da_int k = 0; k < nclass; k++) {
                std::fill(lincomb.begin() + k * nsamples,
                          lincomb.begin() + (k + 1) * nsamples, x[n - (nclass - k)]);
            }
        } else {
            std::fill(lincomb.begin(), lincomb.end(), 0.);
        }
        // Calculate licomb as X * Beta + Beta_0
        da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, nsamples, nclass,
                            nfeat, 1.0, data->X, nsamples, x, nclass, 1.0, lincomb_ptr,
                            nsamples);
        for (da_int i = 0; i < nsamples; i++) {
            for (da_int k = 0; k < nclass; k++) {
                // Find maxexp
                if (maxexp[i] < lincomb[k * nsamples + i]) {
                    maxexp[i] = lincomb[k * nsamples + i];
                }
            }
            // Compute sumexp
            sumexp[i] = exp(-maxexp[i]);
            for (da_int k = 0; k < nclass; k++) {
                sumexp[i] += exp(lincomb[k * nsamples + i] - maxexp[i]);
            }
        }
    }

    // Compute for all samples i and all variables j with k being the class of sample i:
    // A_ij * (prob(x_i=k|Beta) - indicator(i, k))
    std::fill(grad, grad + n, 0);
    for (da_int i = 0; i < nsamples; i++) {
        for (da_int k = 0; k < nclass; k++) {
            gradients_p[k * nsamples + i] =
                exp(lincomb[k * nsamples + i] - maxexp[i]) / sumexp[i];
            if (std::round(y[i]) == k)
                gradients_p[k * nsamples + i] -= 1;
        }
    }
    da_blas::cblas_gemm(CblasColMajor, CblasTrans, CblasNoTrans, nclass, nfeat, nsamples,
                        1.0, gradients_p.data(), nsamples, data->X, nsamples, 0.0, grad,
                        nclass);
    if (data->intercept) {
        for (da_int i = 0; i < nclass; i++) {
            T sum = 0;
            for (da_int j = 0; j < nsamples; j++)
                sum += gradients_p[j + i * nsamples];
            grad[n - (nclass - i)] = sum;
        }
    }
    // Add regularization (exclude intercept)
    reggrd(nfeat * nclass, x, data->l1reg, data->l2reg, grad);

    return 0;
}

/* Mean square error callbacks
 * The MSE loss objective is
 * f = 1/2N \sum (MSE)^2 + lambda/2 (1-alpha) L2 + lambda alpha L1
 */
template <typename T> da_int objfun_mse(da_int n, T *x, T *loss, void *udata) {

    cb_usrdata_linreg<T> *data = (cb_usrdata_linreg<T> *)udata;
    da_int nsamples = data->nsamples;
    const T *y = data->y;
    T *matvec = data->matvec.data();
    *loss = 0;

    // Compute matvec = X*x (+ intercept)
    eval_feature_matrix(n, x, nsamples, data->X, matvec, data->intercept);

    // matvec = matvec - y
    T alpha = -1.0;
    da_blas::cblas_axpy(nsamples, alpha, y, 1, matvec, 1);

    // sum (X * x (+intr) - y)^2
    for (da_int i = 0; i < nsamples; i++) {
        *loss += pow(matvec[i], (T)2.0);
    }
    *loss /= T(2 * nsamples);

    // Add regularization (exclude intercept)
    da_int nmod = data->intercept ? n - 1 : n;
    *loss += regfun(nmod, x, data->l1reg, data->l2reg);

    return 0;
}

/* Evaluate model on a provided x and return loss and prediction
 * Input:
 *  * nsamples number of samples
 *  * ncoef number of coefficients (includes intercept coefficient)
 *  * coef[ncoef] vector of coefficients (includes beta0, intercept coefficient)
 *  * X matrix of size (nsamples times nfeat, nfeat = ncoef if intercept=false,
 *    otherwise nfeat = ncoef-1.
 *  * intercept true/false
 *  * l1reg regularization penalty associated with L1
 *  * l2reg regularization penalty associated with L2
 *  * y[nsamples] nullptr is no observations provided, otherwise a vector of
 *    nsamples observations
 *
 * Output
 *  * loss value of the loss for the predictions given the y observations or
 *         zero if y is nullptr
 *  * pred[nsamples] model predictions
 *
 * Assumes that all input is valid
 */
template <typename T>
da_int loss_mse(da_int nsamples, da_int nfeat, const T *X, bool intercept, T l1reg,
                T l2reg, const T *coef, const T *y, T *loss, T *pred) {

    const da_int ncoef = intercept ? nfeat + 1 : nfeat;

    // Compute predictions: X*coef (+ intercept)
    eval_feature_matrix(ncoef, coef, nsamples, X, pred, intercept);

    if (y) {
        *loss = 0;
        // Observation vector provided, return also the loss function value
        // sum (X * coef (+intr) - y)^2
        for (da_int i = 0; i < nsamples; i++) {
            T res = pred[i] - y[i];
            *loss += res * res;
        }
        *loss /= T(2 * nsamples);

        // Add regularization (exclude intercept coefficient)
        *loss += regfun(nfeat, coef, l1reg, l2reg);
    }

    return 0;
}

/* Mean square error callbacks (gradient)
 * The MSE loss objective gradient is
 * grad = 1/N \sum d(MSE) + lambda (1-alpha) d(L2) + lambda alpha d(L1)
 */
template <typename T>
da_int objgrd_mse(da_int n, T *x, T *grad, void *udata, [[maybe_unused]] da_int xnew) {

    cb_usrdata_linreg<T> *data = (cb_usrdata_linreg<T> *)udata;
    da_int nsamples = data->nsamples;
    T *matvec = data->matvec.data();

    // matvec = X*x (+ itct)
    eval_feature_matrix(n, x, nsamples, data->X, matvec, data->intercept);

    // matvec = matvec - y
    T alpha = -1.0;
    da_blas::cblas_axpy(nsamples, alpha, data->y, 1, matvec, 1);

    // alpha = 2.0;
    alpha = T(1) / T(nsamples);
    T beta = 0.0;
    da_int aux = data->intercept ? 1 : 0;
    da_blas::cblas_gemv(CblasColMajor, CblasTrans, nsamples, n - aux, alpha, data->X,
                        nsamples, matvec, 1, beta, grad, 1);
    if (data->intercept) {
        grad[n - 1] = 0;
        for (da_int i = 0; i < nsamples; i++)
            grad[n - 1] += alpha * matvec[i];
    }

    // Add regularization (exclude intercept)
    da_int nmod = data->intercept ? n - 1 : n;
    reggrd(nmod, x, data->l1reg, data->l2reg, grad);

    return 0;
}

/* coordinate descent method callback to get updated coefficient coef[k]
 *
 * Inputs:
 *  coef[nfeat], current iterate
 *  k, the coordinate to update, see details below
 *  udata, user data
 *  action, see below, and
 *  kdiff, coef[kold] - coef[k] only relevant if action < 0
 * Output:
 *  f, objective function value at current residual vector.
 *     if f = nullptr it provides the new x[k],
 *     otherwise it does not compute anything a part from f.
 *     It assumes that the first call to this function is with f = nullptr.
 *  knew, the new value for coef[k], only if f is nullptr.
 *
 * Actions regarding feature matrix evaluation
 * action < 0 means that feature matrix was previously called and that only a low rank
 *            update is requested and -(action+1) contains the previous k that changed
 *            kold = -(action+1);
 * action = 0 means step not changed. Residual up-to-date
 * action > 0 evaluate the matrix.
 *
 * Assumptions:
 *  * First call to this function is with f = nullptr.
 *  * udata->X is standardized (scaled):
 *    for each column j = 1:nfeat we have 1/nsamples sum xij^2 = 1, i=1:nsamples
 *  * if the model has intercept then udata->X is also centered:
 *    for each column j = 1:nfeat we have sum xij = 0, i=1:nsamples
 *  * udata->y is standardized (scaled):
 *    1/nsamples sum yi^2 = 1 i=1:nsamples
 *  * udata->residual is of size nsamples
 *  * udata->xv is of size nfeat+1 and has all the rescale factors
 *
 *  WARNING nfeat CAN include intercept, nmod provides the user coefficient count
 */
template <typename T>
da_int stepfun_linreg(da_int nfeat, T *coef, T *knew, da_int k, T *f, void *udata,
                      da_int action, T kdiff) {
    stepfun_usrdata_linreg<T> *data = (stepfun_usrdata_linreg<T> *)udata;

    const da_int nmod = data->intercept ? nfeat - 1 : nfeat;
    const da_int nsamples = data->nsamples;

    if (f) { // Quick exit, just provide f
        *f = (T)0;
        for (da_int i = 0; i < nsamples; ++i) {
            T res = data->residual[i];
            *f += res * res;
        }
        *f /= ((T)2 * (T)nsamples);
        // Add regularization (exclude intercept)
        *f += regfun(nmod, coef, data->l1reg, data->l2reg);
        return 0;
    }

    if (action > 0) {
        // Compute X*coef = *y (takes care of intercept)
        eval_feature_matrix(nfeat, coef, nsamples, data->X, data->residual.data(),
                            data->intercept);
        // Compute residuals
        for (da_int i = 0; i < nsamples; ++i) {
            data->residual[i] = data->y[i] - data->residual[i];
        }
    } else if (action < 0 && kdiff != (T)0) {
        /* Low rank update.
         * Only one single entry of coef[1..nmod;intercept]=coef[1..nfeat] has
         * changed and we have the entry and the amount.
         * data->matvec = data->aux + kdiff * X[:,kold];
         * Update the residual vector on the fly
         * residual[:] = residual[:] - kdiff * X[:,kold]
         */
        da_int kold = -(action + 1);
        if (kold < nmod) {
            for (da_int i = 0; i < nsamples; ++i) {
                data->residual[i] -= kdiff * data->X[kold * nsamples + i];
            }
        } else {
            // Change from intercept, X[:,nfeat]=1
            for (da_int i = 0; i < nsamples; ++i) {
                data->residual[i] -= kdiff;
            }
        }
    }

    auto sign = [](T num) {
        const T absnum = std::abs(num);
        return (absnum == (T)0 ? (T)0 : num / absnum);
    };
    auto soft = [sign](T z, T Gamma) {
        return (sign(z) * std::max(std::abs(z) - Gamma, (T)0));
    };

    T xk;
    T betak;
    T gk{T(0)};
    T xvk{T(1)};

    const bool standardized = data->scaling == da_linmod::scaling_t::standardize;
    const bool usexv = data->scaling == da_linmod::scaling_t::standardize ||
                       data->scaling == da_linmod::scaling_t::scale_only;
    const T l1{data->l1reg};        // lambdahat * alpha
    const T l2{T(2) * data->l2reg}; // Note that data->l2reg = lambdahat (1-alpha) / 2;

    if (k < nmod) {
        // handle model coefficients beta1..betaN=coef[0]..coef[nmod-1]
        for (da_int i = 0; i < nsamples; ++i) {
            xk = data->X[k * nsamples + i];
            gk += xk * data->residual[i];
        }
        if (standardized) {
            gk /= T(nsamples);
        }
        if (usexv) {
            xvk = data->xv[k];
        }
        // betak = gk + coef[k] * xv[k]; // see (8) paper GLM2010
        betak = gk + coef[k] * xvk; // <- scale by xv[k]
        betak = soft(betak, l1) / (xvk + l2);
    } else {
        // handle intercept beta0 = coef[nmod+1] = coef[nfeat]
        for (da_int i = 0; i < nsamples; ++i) {
            gk += data->residual[i];
        }
        gk /= (T)nsamples;
        betak = gk + coef[k];
    }
    *knew = betak;

    data = nullptr;
    return 0;
}

#endif
