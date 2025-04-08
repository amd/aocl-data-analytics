/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "aoclda_types.h"
#include "linmod_types.hpp"
#include "macros.h"
#include <vector>

namespace ARCH {

////////////////////////////////////////////////
///////     nln_optim declarations /////////////
////////////////////////////////////////////////

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
    da_linmod_types::scaling_t scaling{da_linmod_types::scaling_t::none};

    usrdata_base(const T *X, const T *y, da_int nsamples, da_int nfeat, bool intercept,
                 T lambda, T alpha, const T *xv = nullptr,
                 da_linmod_types::scaling_t scaling = da_linmod_types::scaling_t::none);
    virtual ~usrdata_base();
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
                      bool intercept, T lambda, T alpha, da_int nclass, da_int nparam);
    ~cb_usrdata_logreg();
};

/* User data for the nonlinear optimization callbacks of the linear regression */
template <class T> class cb_usrdata_linreg : public usrdata_base<T> {
  public:
    /* Add a working memory arrays
     * matvec[nsamples]: typically used to compute the matrix vector product X * coef
     */
    std::vector<T> matvec;

    cb_usrdata_linreg(const T *X, const T *y, da_int nsamples, da_int nfeat,
                      bool intercept, T lambda, T alpha);
    ~cb_usrdata_linreg();
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
                           da_linmod_types::scaling_t scaling);
    ~stepfun_usrdata_linreg();
};

/* This function evaluates the feature matrix X over the parameter vector x (taking into account the intercept),
 * it performs the GEMV operation
 *
 * v = alpha * [X, 1^T] * x + beta * v      OR
 * v = alpha * [X, 1^T]^T x + beta * v
 * if trans = false
 * x[n], X[m,n], v[m]
 * if trans = true
 * x[m], X[m,n], v[n]
 */
template <typename T>
void eval_feature_matrix(da_int n, const T *x, da_int m, const T *X, T *v, bool intercept,
                         bool trans = false, T alpha = 1.0, T beta = 0.0);

/* Add regularization, l1 and l2 terms */
template <typename T> T regfun(da_int n, const T *x, const T l1reg, const T l2reg);

/* Add regularization, l1 and l2 term derivatives */
template <typename T> void reggrd(da_int n, T const *x, T l1reg, T l2reg, T *grad);

/* Callbacks for the various models to be passed to an (un)constrained
 * nonlinear solver of AOCL-DA
 */

/* Logistic regression callbacks
 * Computes the inverse of the log-likelihood of the logistic regression model
 * and its gradient as defined in Elements of Statistical Learning (Hastie & all)
 */
template <typename T>
da_int objfun_logistic_rsc([[maybe_unused]] da_int n, T *x, T *f, void *udata);

template <typename T>
da_int objgrd_logistic_rsc(da_int n, T *x, T *grad, void *udata,
                           [[maybe_unused]] da_int xnew);

template <typename T>
da_int objfun_logistic_two_class([[maybe_unused]] da_int n, T *x, T *f, void *udata);

template <typename T>
da_int objgrd_logistic_two_class(da_int n, T *x, T *grad, void *udata,
                                 [[maybe_unused]] da_int xnew);

template <typename T>
da_int objfun_logistic_ssc([[maybe_unused]] da_int n, T *x, T *f, void *udata);

template <typename T>
da_int objgrd_logistic_ssc(da_int n, T *x, T *grad, void *udata,
                           [[maybe_unused]] da_int xnew);

/* Mean square error callbacks
 * The MSE loss objective is
 * f = 1/2N \sum (MSE)^2 + lambda/2 (1-alpha) L2 + lambda alpha L1
 */
template <typename T> da_int objfun_mse(da_int n, T *x, T *loss, void *udata);

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
                T l2reg, const T *coef, const T *y, T *loss, T *pred);

/* Mean square error callbacks (gradient)
 * The MSE loss objective gradient is
 * grad = 1/N \sum d(MSE) + lambda (1-alpha) d(L2) + lambda alpha d(L1)
 */
template <typename T>
da_int objgrd_mse(da_int n, T *x, T *grad, void *udata, [[maybe_unused]] da_int xnew);

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
da_int stepfun_linreg_glmnet(da_int nfeat, T *coef, T *knew, da_int k, T *f, void *udata,
                             da_int action, T kdiff);

template <typename T>
da_int stepfun_linreg_sklearn(da_int nfeat, T *coef, T *knew, da_int k, T *f, void *udata,
                              da_int action, [[maybe_unused]] T kdiff);

/* Dual gap for linear least squares (sklearn variant) */
template <typename T>
da_int stepchk_linreg_sklearn([[maybe_unused]] da_int nfeat,
                              [[maybe_unused]] const T *coef,
                              [[maybe_unused]] void *udata, T *gap);

} // namespace ARCH
