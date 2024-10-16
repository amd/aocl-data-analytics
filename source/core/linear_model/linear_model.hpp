/*
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclda.h"
#include "aoclsparse.h"
#include "basic_handle.hpp"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "lapack_templates.hpp"
#include "linmod_types.hpp"
#include "macros.h"
#include "nln_optim_callbacks.hpp"
#include "options.hpp"
#include "sparse_overloads.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

namespace ARCH {

namespace da_optim {
template <typename T> class da_optimization;
}

using namespace da_linmod_types;

using namespace ARCH;

/* Linear Models
 *
 * Solve the problem   minimize   f(x) = 1/2n_samples\sum_{i=0}^{nsamples-1} \Xi ( \psi(yi, \phi(xi;t)) ) + eta(xi)
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
 */
namespace da_linmod {

enum fit_opt_type { fit_opt_nln = 0, fit_opt_lsq, fit_opt_coord };
enum class coord_algo_t { undefined = 0, GLMNet = 1, sklearn = 2 };

template <typename T> struct cg_data {
    // Declare objects needed for conjugate gradient solver
    aoclsparse_itsol_handle handle;
    aoclsparse_itsol_rci_job ircomm;
    T *u, *v, rinfo[100], tol;
    T beta = T(0.0), alpha = T(1.0);
    da_int nsamples, ncoef, min_order, maxit;
    std::vector<T> coef, A, b;

    // Constructors
    cg_data(da_int nsamples, da_int ncoef, T tol, da_int maxit);
    ~cg_data();

    da_status compute_cg();

    da_status get_info([[maybe_unused]] da_int linfo, T *info);
};

// Data for cholesky solver in linear regression
template <typename T> struct cholesky_data {
    std::vector<T> A, b;
    da_int min_order;
    T alpha = 1.0, beta = 0.0;

    // Constructors
    cholesky_data(da_int nsamples, da_int ncoef);
};

// Data for QR factorization used in standard linear least squares
template <typename T> struct qr_data {
    // X needs to be copied as lapack's dgeqr modifies the matrix
    std::vector<T> tau, work;
    da_int lwork = 0, n_col, n_row;

    // Constructors
    qr_data(da_int nsamples, da_int nfeat);
};

// Data for svd used in linear regression
template <typename T> struct svd_data {
    std::vector<T> S, U, Vt, temp, work;
    std::vector<da_int> iwork;
    da_int lwork = 0, min_order;
    T alpha = 1.0, beta = 0.0;

    // Constructors
    svd_data(da_int nsamples, da_int nfeat);
};

template <typename T> class linear_model : public basic_handle<T> {
  private:
    // Type of the model, has to be set at initialization phase
    linmod_model mod = linmod_model_undefined;
    da_int method_id = linmod_method::undefined;
    logistic_constraint logistic_constraint_model = logistic_constraint::no;

    // True if the model has been successfully trained
    bool model_trained = false;
    bool is_well_determined;
    bool is_transposed = false;
    bool copycoefs = false;
    bool use_dual_coefs = false;

    /* Regression data
     * nfeat: number of features
     * nsamples: number of data points
     * nclass: number of different classes in the case of linear classification. unused otherwise
     * intercept: controls if the linear regression intercept is to be set
     * XUSR[nsamples*nfeat]: feature matrix, pointer to user data directly - will not be modified by any function
     * yusr[nsamples]: model response, pointer to user data - will not be modified by any function
     * X is a pointer to either XUSR or a modifiable copy of XUSR
     */
    da_int nfeat = 0, nsamples = 0;
    da_int nclass = 0;
    bool intercept = false;
    const T *yusr = nullptr;
    const T *XUSR = nullptr;
    T *y = nullptr; // May contain a modified copy of yusr
    T *X = nullptr; // May contain a modified copy of XUSR

    //Utility pointer to column major allocated copy of user's data
    T *X_temp = nullptr;

    T time; // Computation time

    /* Parameters used during the standardization of the problem
     * these are only defined if "scaling" is not "none" and populated
     * on the call to ::model_scaling(...)
     */
    scaling_t scaling = scaling_t::none;
    std::vector<T> std_shifts; // column-wise means [ X | y ], size nfeat + 1
    std::vector<T> std_scales; // column-wise scales stored as [ X | y ] size nfeat + 1
    // column-wise X (variance) "proportions" of size nfeat (or norm squared of X)
    std::vector<T> std_xv;
    /* Training data
     * coef: vector containing the trained coefficients of the model
       dual_coef: vector containing the trained dual coefficients of the model
     */
    da_int ncoef = 0;
    std::vector<T> coef;
    std::vector<T> dual_coef; // Currently only used to store user's initial start coefs

    /* Elastic net penalty parameters (Regularization L1: LASSO, L2: Ridge, combination => Elastic net)
     * Penalty parameters are: lambda ( (1-alpha)L2 + alpha*L1 )
     * lambda >= 0 and 0<=alpha<=1.
     */
    T alpha, lambda;

    // Optimization object to call generic algorithms
    ARCH::da_optim::da_optimization<T> *opt = nullptr;
    usrdata_base<T> *udata = nullptr;
    qr_data<T> *qr = nullptr;
    svd_data<T> *svd = nullptr;
    cg_data<T> *cg = nullptr;
    cholesky_data<T> *cholesky = nullptr;

    // Private methods to allocate memory
    da_status init_opt_method(linmod_method method);

    // QR fact data
    da_status init_qr_data();
    da_status qr_lsq();

    /* Dispatcher methods
     * choose_method: if "optim method" is set to auto, choose automatically how
     *                to compute the model
     * validate_options: check that the options chosen by the user are compatible
     */
    da_status choose_method();
    da_status validate_options(da_int method);

  public:
    linear_model(da_errors::da_error_t &err);
    ~linear_model();

    /* This function is called when data in the handle has changed, e.g. options
     * changed. We mark the model untrained and prepare the handle in a way that
     * it is suitable to solve again.
     */
    void refresh();

    da_status define_features(da_int nfeat, da_int nsamples, const T *X, const T *y);
    da_status select_model(linmod_model mod);
    da_status model_scaling(da_int method_id);
    void revert_scaling();
    void setup_xtx_xty(const T *X_input, const T *y_input, std::vector<T> &A,
                       std::vector<T> &b);
    void scale_warmstart();
    da_status fit(da_int usr_ncoefs, const T *coefs);
    da_status fit_logreg_lbfgs();
    da_status fit_linreg_lbfgs();
    da_status fit_linreg_coord();
    da_status fit_linreg_svd();
    da_status fit_linreg_cholesky();
    da_status fit_linreg_cg();
    da_status get_coef(da_int &nx, T *coef);
    da_status evaluate_model(da_int nfeat, da_int nsamples, const T *X, T *predictions,
                             T *observations, T *loss);

    da_status get_result(da_result query, da_int *dim, T *result);
    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] da_int *result);

    // Testing getters
    bool get_model_trained();
};

} // namespace da_linmod

} // namespace ARCH
