/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#undef min

#include "aoclda.h"
#include "da_cblas.hh"

/* Base class of the callback user data to be passed as void pointers
 * Contain all the basic data to compute */
template <class T> class usrdata_base {
  public:
    da_int nsamples = 0, nfeat = 0;
    /* Feature matrix of size (nsamples x nfeat)*/
    const T *X = nullptr;
    /* Responce vector */
    const T *y = nullptr;

    /* Intercept */
    bool intercept = false;
    /* Additional paremeters that enhance the model
     * Transform on the residuals, loss function and regularization */
    T l1reg = 0.0;
    T l2reg = 0.0;

    usrdata_base(const T *X, const T *y, da_int nsamples, da_int nfeat, bool intercept,
                 T lambda, T alpha)
        : nsamples(nsamples), nfeat(nfeat), X(X), y(y), intercept(intercept) {
        l1reg = lambda * alpha;
        l2reg = lambda * ((T)1.0 - alpha) / (T)2.0;
    }
    virtual ~usrdata_base() {}
};

/* user data for the nonlinear optimization callbacks of the logistic regression */
template <class T> class cb_usrdata_logreg : public usrdata_base<T> {
  public:
    da_int nclass;
    /* Add 2 working memory arrays
     * maxexp[nsamples]: used to store the maximum values of each X_k beta_k (k as class index) for the logsumexp trick
     * lincomb[nsamples*(nclass-1)]: used to store all the X_k beta_k values
     */
    std::vector<T> maxexp, lincomb;

    cb_usrdata_logreg(const T *X, const T *y, da_int nsamples, da_int nfeat,
                      bool intercept, T lambda, T alpha, da_int nclass)
        : usrdata_base<T>(X, y, nsamples, nfeat, intercept, lambda, alpha),
          nclass(nclass) {
        maxexp.resize(nsamples);
        lincomb.resize(nsamples * (nclass - 1));
    }
    ~cb_usrdata_logreg() {}
};

/* user data for the nonlinear optimization callbacks of the lineqr regression */
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

/* user data for the coordinate descent step function of the linear regression */
template <class T> class stepfun_usrdata_linreg : public usrdata_base<T> {
  public:
    /* Add two working memory arrays
     * matvec[nsamples]: typically used to compute the matrix vector product X * coef
     * aux[nsamples]: auxilliary to hold intermediate results
     */
    std::vector<T> matvec, aux;

    stepfun_usrdata_linreg(const T *X, const T *y, da_int nsamples, da_int nfeat,
                           bool intercept, T lambda, T alpha)
        : usrdata_base<T>(X, y, nsamples, nfeat, intercept, lambda, alpha) {
        matvec.resize(nsamples);
        aux.resize(nsamples);
    }
    ~stepfun_usrdata_linreg() {}
};

/* This function evaluates the feature matrix X over the parameter vector x (taking into account the intercept),
 * it performs the GEMV operation
 *
 * v = [X, 1^T] * x
 */
template <typename T>
void eval_feature_matrix(da_int n, T *x, da_int nsamples, const T *X, T *v,
                         bool intercept) {
    T alpha = 1.0, beta = 0.0;

    da_int aux = intercept ? 1 : 0;
    da_blas::cblas_gemv(CblasColMajor, CblasNoTrans, nsamples, n - aux, alpha, X,
                        nsamples, x, 1, beta, v, 1);
    if (intercept) {
        for (da_int i = 0; i < nsamples; i++)
            v[i] += x[n - 1];
    }
}

/* Add regularization, l1 and l2 terms */
template <typename T> T regfun(usrdata_base<T> *data, da_int n, T const *x) {
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
template <typename T> void reggrd(usrdata_base<T> *data, da_int n, T const *x, T *grad) {
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

/* Callbacks for the various models to be passed to an (un)constrained
 * nonlinear solver of AOCL-DA
 */

/* Logistic regression callbacks
 * Computes the inverse of the log-likelihood of the logistic regression model
 * and its gradient as defined in Elements of Statistical Learning (Hastie & all)
 */
template <typename T>
da_int objfun_logistic([[maybe_unused]] da_int n, T *x, T *f, void *usrdata) {

    // All data related to the regression problem is stored in the usrdata pointer
    // multinomial problem with K (nclass) classes (indexed in [0, K-1]), nfeat features and nsamples samples.
    // x is of size (nfeat+itpt)*(K-1)
    // where itpt is 1 if the intercept is required and 0 otherwise
    // with nmod = (nfeat+itpt), the parameters corresponding to the class k (k in 0,..,K-2)

    cb_usrdata_logreg<T> *data = (cb_usrdata_logreg<T> *)usrdata;
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
    // use logsumexp trick to avoid overflow
    for (da_int i = 0; i < nsamples; i++) {
        T val = exp(-maxexp[i]);
        for (da_int k = 0; k < nclass - 1; k++) {
            val += exp(lincomb[k * nsamples + i] - maxexp[i]);
        }
        *f += maxexp[i] + log(val);
    }

    // Add regularization (exclude intercept)
    *f += regfun(data, data->nfeat, x);

    return 0;
}

template <typename T>
da_int objgrd_logistic(da_int n, T *x, T *grad, void *usrdata,
                       [[maybe_unused]] da_int xnew) {

    cb_usrdata_logreg<T> *data = (cb_usrdata_logreg<T> *)usrdata;
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
        // lnsumexp := log(1 + sum_k exp(Beta_k^T * x))
        T lnsumexp = exp(-maxexp[i]);
        for (da_int k = 0; k < nclass - 1; k++) {
            lnsumexp += exp(lincomb[k * nsamples + i] - maxexp[i]);
        }
        lnsumexp = maxexp[i] + log(lnsumexp);

        for (da_int k = 0; k < nclass - 1; k++) {
            // val := exp(Beta_k^T * x) / (1 + sum_j exp(Beta_j^T * x))
            T val = -exp(lincomb[k * nsamples + i] - lnsumexp);
            if (std::round(y[i]) == k)
                // indicator(i, k)
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
    reggrd(data, data->nfeat, x, grad);

    return 0;
}

/* Mean square error callbacks */
template <typename T> da_int objfun_mse(da_int n, T *x, T *f, void *usrdata) {

    cb_usrdata_linreg<T> *data = (cb_usrdata_linreg<T> *)usrdata;
    da_int nsamples = data->nsamples;
    const T *y = data->y;
    T *matvec = data->matvec.data();
    *f = 0;

    // Compute matvec = X*x (+ intercept)
    eval_feature_matrix(n, x, nsamples, data->X, matvec, data->intercept);

    // matvec = matvec - y
    T alpha = -1.0;
    da_blas::cblas_axpy(nsamples, alpha, y, 1, matvec, 1);

    // sum (X * x (+itct) - y)^2
    for (da_int i = 0; i < nsamples; i++) {
        *f += pow(matvec[i], (T)2.0);
    }

    // Add regularization (exclude intercept)
    da_int nmod = data->intercept ? n - 1 : n;
    *f += regfun(data, nmod, x);

    return 0;
}

template <typename T>
da_int objgrd_mse(da_int n, T *x, T *grad, void *usrdata, [[maybe_unused]] da_int xnew) {

    cb_usrdata_linreg<T> *data = (cb_usrdata_linreg<T> *)usrdata;
    da_int nsamples = data->nsamples;
    T *matvec = data->matvec.data();

    // matvec = X*x (+ itct)
    eval_feature_matrix(n, x, nsamples, data->X, matvec, data->intercept);

    // matvec = matvec - y
    T alpha = -1.0;
    da_blas::cblas_axpy(nsamples, alpha, data->y, 1, matvec, 1);

    alpha = 2.0;
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
    reggrd(data, nmod, x, grad);

    return 0;
}

/* Helper callback to get step for coordinate descent method */
template <typename T>
da_int stepfun_linreg(da_int nfeat, T *coef, T *step, da_int k, T *f, void *usrdata,
                      da_int action, T kdiff) {
    stepfun_usrdata_linreg<T> *data = (stepfun_usrdata_linreg<T> *)usrdata;
    /* Actions regarting feature matrix evaluation
     * action < 0 means that feature matrix was previously called and that only a low rank
     *            update is requested and -(action+1) contains the previous k that changed
     *            kold = -(action+1);
     * action = 0 means not to evaluate the feature matrix (restore matvec from aux)
     * action > 0 evaluate the matrix.
     *
     * Assumptions:
     *  * usrdata->X is standardized:
     *    for each column j = 1:nfeat we have
     *    zero-mean, and
     *    sum xij^2 = 1, i=1:nsamples
     *  * usrdata->aux is of size nsamples
     */

    da_int nsamples = data->nsamples;
    if (action > 0) {
        // Compute X*coef = *y (takes care of intercept)
        eval_feature_matrix(nfeat, coef, nsamples, data->X, data->matvec.data(),
                            data->intercept);
        // Copy vector data->y into data->aux
        for (da_int i = 0; i < nsamples; ++i)
            data->aux[i] = data->matvec[i];
    } else if (action < 0) {
        /* Low rank update.
         * Only one single entry of coef[1..nfeat] has changed and we have
         * the entry and the ammount.
         * data->matvec = data->aux + kdiff * X[:,kold];
         */
        da_int kold = -(action + 1);
        for (da_int i = 0; i < nsamples; ++i) {
            data->matvec[i] = data->aux[i] + kdiff * data->X[kold * nsamples + i];
            // Copy vector data->y into data->aux
            data->aux[i] = data->matvec[i];
        }
        /* Low-rank update validation
         * eval_feature_matrix(nfeat, coef, usrdata);
         * for (da_int i = 0; i < nsamples; ++i){
         *     T d = data->y[i] - data->aux[i];
         *     if (std::abs(d) > 1e-9) {
         *         return 99;
         *     }
         * }
         */
    } else {
        // Copy vector back from data->aux to data->y
        for (da_int i = 0; i < nsamples; ++i)
            data->matvec[i] = data->aux[i];
    }

    T betak{0};
    da_int nmod = data->intercept ? nfeat - 1 : nfeat;
    auto sign = [](T num) {
        const T absnum = std::abs(num);
        return (absnum == (T)0 ? (T)0 : num / absnum);
    };
    auto soft = [sign](T y, T Gamma) {
        return (sign(y) * std::max(std::abs(y) - Gamma, (T)0));
    };
    T xk;
    T presidual;
    if (k < nmod) {
        // handle model coefficients beta1..betaN
        if (coef[k] != (T)0) {
            for (da_int i = 0; i < nsamples; i++) {
                xk = data->X[k * nsamples + i];
                presidual = data->y[i] - (data->matvec[i] - xk * coef[k]);
                betak += presidual * xk;
            }
        } else {
            for (da_int i = 0; i < nsamples; i++) {
                xk = data->X[k * nsamples + i];
                presidual = data->y[i] - data->matvec[i];
                betak += presidual * xk;
            }
        }
    } else {
        // handle intercept beta0 (last element of coef)
        if (coef[k] != (T)0) {
            for (da_int i = 0; i < nsamples; i++) {
                presidual = data->y[i] - (data->matvec[i] - coef[k]);
                betak += presidual;
            }
        } else {
            for (da_int i = 0; i < nsamples; i++) {
                presidual = data->y[i] - data->matvec[i];
                betak += presidual;
            }
        }
    }

    // y = y - y
    T alpha = -1.0;
    da_blas::cblas_axpy(nsamples, alpha, data->y, 1, data->matvec.data(), 1);

    // sum (X * coef (+intercept) - y)^2
    for (da_int i = 0; i < nsamples; i++) {
        *f += pow(data->matvec[i], (T)2.0);
    }

    // Add regularization (exclude intercept)
    *f += regfun(data, nmod, coef);

    *step = soft(betak, data->l1reg) / ((T)1 + data->l2reg);

    data = nullptr;
    return 0;
}

#endif