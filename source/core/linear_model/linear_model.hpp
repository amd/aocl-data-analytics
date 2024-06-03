/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef LINEAR_MODEL_HPP
#define LINEAR_MODEL_HPP

#include "aoclda.h"
#include "aoclsparse.h"
#include "basic_handle.hpp"
#include "callbacks.hpp"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "lapack_templates.hpp"
#include "linmod_cg.hpp"
#include "linmod_cholesky.hpp"
#include "linmod_nln_optim.hpp"
#include "linmod_options.hpp"
#include "linmod_qr.hpp"
#include "linmod_svd.hpp"
#include "linmod_types.hpp"
#include "optimization.hpp"
#include "options.hpp"
#include "statistical_utilities.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

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
 *  FIXME ADD BOX BOUNDS
 *
 */
namespace da_linmod {
enum fit_opt_type { fit_opt_nln = 0, fit_opt_lsq, fit_opt_coord };

template <typename T> class linear_model : public basic_handle<T> {
  private:
    /* type of the model, has to de set at initialization phase */
    linmod_model mod = linmod_model_undefined;
    da_int method_id = linmod_method::undefined;

    /* pointer to error trace */
    da_errors::da_error_t *err = nullptr;

    /* true if the model has been successfully trained */
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
    T *y = nullptr; // may contain a modified copy of yusr
    T *X = nullptr; // may contain a modified copy of XUSR

    T time; // computation time

    /* parameters used during the standardization of the problem
     * these are only defined if "scaling" is not "none" and populated
     * on the call to ::model_scaling(...)
     */
    scaling_t scaling = scaling_t::none;
    std::vector<T> std_shifts; // column-wise means [ X | y ], size nfeat + 1
    std::vector<T> std_scales; // column-wise scales stored as [ X | y ] size nfeat + 1
    std::vector<T> std_xv;     // column-wise X (variance) "proportions" of size nfeat

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

    /* optimization object to call generic algorithms */
    da_optim::da_optimization<T> *opt = nullptr;
    usrdata_base<T> *udata = nullptr;
    qr_data<T> *qr = nullptr;
    svd_data<T> *svd = nullptr;
    cg_data<T> *cg = nullptr;
    cholesky_data<T> *cholesky = nullptr;

    /* private methods to allocate memory */
    da_status init_opt_method(linmod_method method);

    /* QR fact data */
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
    da_options::OptionRegistry opts;
    linear_model(da_errors::da_error_t &err) {
        // assumes that err is valid
        this->err = &err;
        register_linmod_options<T>(opts); // FIXME verify return status!
    }
    ~linear_model();

    /* This function is called when data in the handle has changed, e.g. options
     * changed. We mark the model untrained and prepare the handle in a way that
     * it is suitable to solve again.
     *
     * TODO this needs to be optimized and
     * avoid rescaling when data was already prepared.
     * TODO we could avoid destroying the optimization options and user data at
     * every warm-start...
     */
    void refresh() {
        if (model_trained) {
            // Reset
            model_trained = false;
            if (X && X != XUSR)
                delete[] X;
            if (y && y != yusr)
                delete[] y;
            X = (T *)(XUSR);
            y = (T *)(yusr);
        }
        if (qr) {
            delete qr;
            qr = nullptr;
        }
        if (cholesky) {
            delete cholesky;
            cholesky = nullptr;
        }
        if (svd) {
            delete svd;
            svd = nullptr;
        }
        if (cg) {
            delete cg;
            cg = nullptr;
        }
        // Destroy optimization option registry
        if (opt) {
            delete opt;
            opt = nullptr;
        }

        // Destroy linear model data
        if (udata) {
            delete udata;
            udata = nullptr;
        }
    };

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
    da_status evaluate_model(da_int nfeat, da_int nsamples, T *X, T *predictions,
                             T *observations, T *loss);

    /* get_result (required to be defined by basic_handle) */
    da_status get_result(da_result query, da_int *dim, T *result);
    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] da_int *result);
};

template <typename T> linear_model<T>::~linear_model() {
    // XUSR and yusr are from user, do not deallocate
    // if X and y are not pointing XUSR and yusr then free up
    if (X && X != XUSR) {
        delete[] X;
        X = nullptr;
    }
    if (y && y != yusr) {
        delete[] y;
        y = nullptr;
    }
    XUSR = nullptr;
    yusr = nullptr;
    err = nullptr;

    if (qr)
        delete qr;

    if (svd)
        delete svd;

    if (cg)
        delete cg;

    if (cholesky)
        delete cholesky;

    if (opt)
        delete opt;

    if (udata)
        delete udata;
};

template <typename T>
da_status linear_model<T>::get_result(da_result query, da_int *dim, T *result) {
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
        for (da_int i = 0; i < 100; ++i)
            result[i] = T(-1);

        // Copy out the info array if available for optimisation solvers
        if (method_id == linmod_method::lbfgsb || method_id == linmod_method::coord)
            // Hopefully no opt solver will use more that the hard coded limit
            return opt->get_info(*dim, result);
        // For the rest of the solvers find loss value via loss_mse function and set compute time
        else {
            // Save information about loss function
            da_int status;
            T loss;
            std::vector<T> pred(nsamples);
            const T l1reg = alpha * lambda;
            const T l2reg = (T(1) - alpha) * lambda / T(2);
            // Call loss_mse
            status = loss_mse(nsamples, nfeat, X, intercept, l1reg, l2reg, coef.data(), y,
                              &loss, pred.data());
            if (status != 0) {
                return da_status_incorrect_output;
            }
            // Save information about the value of loss function
            result[0] = loss;
            // Save information about the computation time
            result[3] = time;
        }
        // For CG we have member function that fills n_iter and gradient of loss
        if (method_id == linmod_method::cg)
            return cg->get_info(*dim, result);

        return da_status_success;
        break;

    case da_result::da_linmod_coef:
        return this->get_coef(*dim, result);
        break;

    default:
        return da_warn(this->err, da_status_unknown_query,
                       "The requested result could not be queried by this handle.");
    }
};
template <typename T>
da_status linear_model<T>::get_result([[maybe_unused]] da_result query,
                                      [[maybe_unused]] da_int *dim,
                                      [[maybe_unused]] da_int *result) {
    return da_warn(this->err, da_status_unknown_query,
                   "Handle does not contain data relevant to this query. Was the "
                   "last call to the solver successful?");
};

/* Store the user data in X and y. No data is copied at this stage
 * possible fail:
 * - invalid input
 */
template <typename T>
da_status linear_model<T>::define_features(da_int nfeat, da_int nsamples, const T *X,
                                           const T *y) {
    if (nfeat <= 0)
        return da_error(this->err, da_status_invalid_input, "nfeat must be positive.");
    if (nsamples <= 0)
        return da_error(this->err, da_status_invalid_input, "nsamples must be positive.");
    if (X == nullptr || y == nullptr)
        return da_error(this->err, da_status_invalid_input,
                        "Either X or y are not valid pointers.");

    model_trained = false;

    this->nfeat = nfeat;
    this->nsamples = nsamples;
    this->is_well_determined = nsamples > nfeat;
    // assign user's feature pointers
    this->yusr = y;
    this->XUSR = X;
    // point copy X and y also to user data
    this->y = (T *)(y);
    this->X = (T *)(X);

    return da_status_success;
}

template <typename T> da_status linear_model<T>::select_model(linmod_model mod) {

    // reset model_trained only if the model is changed
    if (mod != this->mod) {
        this->mod = mod;
        model_trained = false;
    }
    return da_status_success;
}

/*
 * Common setting for all optimization solvers for linear models
 */
template <typename T> da_status linear_model<T>::init_opt_method(linmod_method method) {
    da_status status;
    da_int maxit, prnlvl, prnopt;
    std::string slv, prnopt_str, optstr;
    T tol, factr, maxtime;

    switch (method) {
    case (da_linmod::linmod_method::lbfgsb):
        slv = "lbfgsb";
        break;
    case (da_linmod::linmod_method::coord):
        slv = "coord";
        break;
    default:
        return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpected method.");
    }

    try {
        opt = new da_optim::da_optimization<T>(status, *(this->err));
    } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }

    if (status != da_status_success) {
        opt = nullptr;
        return status; // Error message already loaded
    }
    if (opt->add_vars(ncoef) != da_status_success) {
        return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly linear model provided for the optimization "
                        "problem has an invalid number of coefficients ncoef=" +
                            std::to_string(ncoef) + ", expecting ncoef > 0.");
    }
    // Set options here
    da_int dbg{0};
    if (this->opts.get("debug", dbg) != da_status_success) {
        return da_error( // LCOV_EXCL_LINE
            opt->err, da_status_internal_error,
            "Unexpectedly <debug> option not found in the linear model "
            "option registry.");
    }
    // Pass print level option from linmod to optimization
    if (this->opts.get("print level", prnlvl) != da_status_success) {
        return da_error( // LCOV_EXCL_LINE
            opt->err, da_status_internal_error,
            "Unexpectedly <print level> option not found in the linear model "
            "option registry.");
    }
    // Decrease print level for optimization stage
    if (opt->opts.set("print level", prnlvl) != da_status_success) {
        return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly linear model provided an invalid value to the "
                        "<print level> option.");
    }
    // Pass print options
    if (this->opts.get("print options", prnopt_str, prnopt) != da_status_success) {
        return da_error( // LCOV_EXCL_LINE
            opt->err, da_status_internal_error,
            "Unexpectedly <print options> option not found in the linear model "
            "option registry.");
    }
    if (dbg && prnopt) {
        // Request solver to also print options
        if (opt->opts.set("print options", prnopt_str) != da_status_success) {
            return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Unexpectedly linear model provided an invalid value to the "
                            "<print options> option.");
        }
    }
    // Setup optimization method
    if (opt->opts.set("optim method", slv) != da_status_success) {
        return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly linear model provided an invalid value to the "
                        "<optim method> option.");
    }
    // Pass convergence parameters
    if (this->opts.get("optim iteration limit", maxit) != da_status_success) {
        return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly <optim iteration limit> option not "
                        "found in the linear model "
                        "option registry.");
    }
    optstr = slv + " iteration limit";
    if (opt->opts.set(optstr, maxit) != da_status_success) {
        return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly linear model provided an invalid value to the "
                        "<" +
                            optstr + "> option.");
    }
    if (this->opts.get("optim convergence tol", tol) != da_status_success) {
        return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly <optim convergence tol> option not "
                        "found in the linear model option registry.");
    }
    optstr = slv + " convergence tol";
    if (opt->opts.set(optstr, tol) != da_status_success) {
        return da_error( // LCOV_EXCL_LINE
            opt->err, da_status_internal_error,
            "Unexpectedly linear model provided an invalid value to the <" + optstr +
                "> option.");
    }
    if (this->opts.get("optim progress factor", factr) != da_status_success) {
        return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly <optim progress factor> option not "
                        "found in the linear model option registry.");
    }
    optstr = slv + " progress factor";
    if (opt->opts.set(optstr, factr) != da_status_success) {
        return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly linear model provided an invalid value to the <" +
                            optstr + "> option.");
    }
    if (slv == "coord") {
        // specific options for coord
        optstr = slv + " skip tol";
        if (opt->opts.set(optstr, tol) != da_status_success) {
            return da_error(
                opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                "Unexpectedly linear model provided an invalid value to the <" + optstr +
                    "> option.");
        }
        // Pass ledger parameters
        da_int skipmin;
        da_int skipmax;
        if (this->opts.get("optim coord skip min", skipmin) != da_status_success) {
            return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Unexpectedly <optim coord skip min> option not "
                            "found in the linear model "
                            "option registry.");
        }
        optstr = "coord skip min";
        if (opt->opts.set(optstr, skipmin) != da_status_success) {
            return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Unexpectedly linear model provided an invalid value to the "
                            "<" +
                                optstr + "> option.");
        }
        if (this->opts.get("optim coord skip max", skipmax) != da_status_success) {
            return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Unexpectedly <optim coord skip max> option not "
                            "found in the linear model "
                            "option registry.");
        }
        optstr = "coord skip max";
        if (opt->opts.set(optstr, skipmax) != da_status_success) {
            return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Unexpectedly linear model provided an invalid value to the "
                            "<" +
                                optstr + "> option.");
        }
    }

    // Pass time limit
    if (this->opts.get("optim time limit", maxtime) != da_status_success) {
        return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly <optim time limit> option not "
                        "found in the linear model option registry.");
    }
    optstr = "time limit";
    if (opt->opts.set(optstr, maxtime) != da_status_success) {
        return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly linear model provided an invalid value to the "
                        "<" +
                            optstr + "> option.");
    }

    return da_status_success;
}

template <typename T> da_status linear_model<T>::get_coef(da_int &nx, T *coef) {
    if (!model_trained)
        return da_error(this->err, da_status_out_of_date,
                        "The data associated to the model is out of date.");
    if (nx != ncoef) {
        nx = ncoef;
        return da_warn(this->err, da_status_invalid_array_dimension,
                       "The number of coefficients is wrong, correct size is " +
                           std::to_string(ncoef) + ".");
    }
    if (coef == nullptr)
        return da_error(
            this->err, da_status_invalid_input,
            "Argument coef needs to provide a valid pointer of at least size " +
                std::to_string(ncoef) + ".");

    da_int i;
    for (i = 0; i < ncoef; i++)
        coef[i] = this->coef[i];

    return da_status_success;
}

template <typename T>
da_status linear_model<T>::evaluate_model(da_int nfeat, da_int nsamples, T *X,
                                          T *predictions, T *observations, T *loss) {
    da_int i, status;

    if (nfeat != this->nfeat)
        return da_error(this->err, da_status_invalid_input,
                        "nfeat = " + std::to_string(nfeat) +
                            ". it must match the number of features of the computed "
                            "model: nfeat = " +
                            std::to_string(this->nfeat) + ".");
    if (nsamples <= 0)
        return da_error(this->err, da_status_invalid_input, "nsamples must be positive.");
    if (X == nullptr || predictions == nullptr)
        return da_error(this->err, da_status_invalid_input,
                        "One of X or predictions was a null pointer.");
    if (!model_trained)
        return da_error(this->err, da_status_out_of_date,
                        "The model has not been trained yet.");

    // X is assumed to be of shape (nsamples, nfeat)
    // y is assumed to be of size nsamples

    const T l1reg = this->alpha * this->lambda;
    const T l2reg = (T(1) - this->alpha) * this->lambda / T(2);

    T alpha = 1.0, beta = 0.0;
    T aux;
    da_int nmod;
    std::vector<T> log_proba(0);
    switch (mod) {
    case linmod_model_mse:
        // Call loss_mse
        status = loss_mse(nsamples, nfeat, X, this->intercept, l1reg, l2reg,
                          this->coef.data(), observations, loss, predictions);
        if (status != 0) {
            return da_error(this->err, da_status_incorrect_output,
                            "Unexpected error at evaluating model.");
        }
        break;
    case linmod_model_logistic:
        nmod = intercept ? nfeat + 1 : nfeat;
        try {
            log_proba.resize(nsamples * nclass, 0);
        } catch (std::bad_alloc const &) {
            return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }
        std::fill(predictions, predictions + nsamples, T(0));
        //FIXME: move this to nln_optim header
        std::fill(log_proba.begin() + nsamples * (nclass - 1), log_proba.end(), T(1));
        for (da_int k = 0; k < nclass - 1; k++) {
            da_blas::cblas_gemv(CblasColMajor, CblasNoTrans, nsamples, nfeat, alpha, X,
                                nsamples, &coef[k * nmod], 1, beta,
                                &log_proba[k * nsamples], 1);
            if (intercept) {
                for (i = 0; i < nsamples; i++)
                    log_proba[k * nsamples + i] += coef[(k + 1) * nmod - 1];
            }
            for (i = 0; i < nsamples; i++)
                log_proba[k * nsamples + i] = exp(log_proba[k * nsamples + i]);
        }
        for (i = 0; i < nsamples; i++) {
            aux = 0;
            for (da_int k = 0; k < nclass; k++) {
                aux += log_proba[k * nsamples + i];
            }
            for (da_int k = 0; k < nclass; k++)
                log_proba[k * nsamples + i] /= aux;
        }
        for (i = 0; i < nsamples; i++) {
            aux = 0.0;
            for (da_int k = 0; k < nclass; k++) {
                if (log_proba[k * nsamples + i] > aux) {
                    aux = log_proba[k * nsamples + i];
                    predictions[i] = (T)k;
                }
            }
        }
        break;

    default:
        return da_error(this->err, da_status_not_implemented, // LCOV_EXCL_LINE
                        "The requested model is not supported.");
        break;
    }

    return da_status_success;
}

template <typename T> da_status linear_model<T>::fit(da_int usr_ncoefs, const T *coefs) {

    if (model_trained)
        return da_status_success;

    da_int prn, intercept_int, scalingint, dbg{0};
    std::string val, method, scalingstr;
    da_status status;
    auto clock = std::chrono::system_clock::now();

    // For all opts.get() it is assumed they don't fail
    opts.get("intercept", intercept_int);
    opts.get("alpha", this->alpha);
    opts.get("lambda", this->lambda);
    opts.get("optim method", method, method_id);
    this->intercept = (bool)intercept_int;

    if (method == "auto") {
        status = choose_method();
        if (status != da_status_success) {
            return status; // Error message already loaded
        }
    }
    opts.get("optim method", method, method_id);

    if (this->opts.get("scaling", scalingstr, scalingint) != da_status_success) {
        return da_error( // LCOV_EXCL_LINE
            this->err, da_status_internal_error,
            "Unexpectedly <scaling> option not found in the linear model "
            "option registry.");
    }
    scaling = scaling_t(scalingint);

    // Validation should be after reading user's chosen solvers and scaling
    status = validate_options(method_id);
    if (status != da_status_success) {
        return status; // Error message already loaded
    }

    switch (mod) {
    case linmod_model_mse:
        ncoef = nfeat;
        if (intercept)
            ncoef += 1;

        // FIXME Add box-bound

        // Scaling
        if (scaling == scaling_t::automatic) {
            switch (method_id) {
            // remove intercept by centering (and scaling) both X and y
            case linmod_method::coord:
                scaling = scaling_t::scale_only;
                scalingstr = "scale only";
                break;
            case linmod_method::svd:
            case linmod_method::qr:
                if (intercept) {
                    scaling = scaling_t::centering;
                    scalingstr = "centering";
                    break;
                } else {
                    scaling = scaling_t::none;
                    scalingstr = "none";
                    break;
                }
            case linmod_method::cholesky:
            case linmod_method::cg:
            case linmod_method::lbfgsb:
                if (!is_well_determined && intercept) {
                    scaling = scaling_t::centering;
                    scalingstr = "centering";
                    break;
                } else {
                    scaling = scaling_t::none;
                    scalingstr = "none";
                    break;
                }
            default:
                scaling = scaling_t::none;
                scalingstr = "none";
                break;
            }
            // Store back the option value
            this->opts.set("scaling", scalingstr, da_options::solver);
        }

        // Scales: X and y
        status = model_scaling(method_id);
        if (status != da_status_success) {
            return status; // message already loaded
        }

        // TODO: Before final push I'll try to reduce the following logic into one bigger "if branch" with nested ifs

        // Agreed standardising policy (matching glmnet and sklearn)
        // If scaling==standardise we are guaranteeing matching glmnet for lasso/ridge/elastic net
        // If scaling==scale_only for Lasso and Ridge only we are guaranteeing to match sklearn, and for Elastic Net we match glmnet (in future we will match sklearn)
        // If scaling==centering for Ridge we are guaranteeing sklearn solution except coord solver (which cannot be run on 'centering')

        // If L2 regression
        if (alpha == T(0.0) && lambda != T(0.0)) {
            if (scaling == scaling_t::scale_only) {
                lambda /= T(nsamples);
            }
            if (scaling == scaling_t::standardize) {
                lambda /= std_scales[nfeat];
                if (method_id == linmod_method::svd ||
                    method_id == linmod_method::cholesky ||
                    method_id == linmod_method::cg) {
                    lambda *= T(nsamples);
                }
            }
            // Rescale lambda when scaling != "standardize" and the solver is not coord, ie,
            // need the scaled lambda for the objective and gradient
            if ((method_id == linmod_method::lbfgsb) &&
                (scaling != scaling_t::standardize)) {
                lambda /= T(nsamples);
            }
        }
        // If Lasso or Elastic Net
        if (alpha != T(0.0) && lambda != T(0.0)) {
            if (scaling == scaling_t::standardize || scaling == scaling_t::scale_only) {
                // Rescale lambda
                lambda /= std_scales[nfeat];
            }
        }

        // copy if provide and solver can use it...
        copycoefs = coefs != nullptr &&
                    da_linmod::linmod_method_type::is_iterative(linmod_method(method_id));

        // We accept dual coefficients for underdetermined cg problem with initial coefficients
        if (copycoefs && method_id == linmod_method::cg && !is_well_determined) {
            copycoefs &= usr_ncoefs >= nsamples;
            use_dual_coefs = true;
            da_warn(this->err, da_status_invalid_input, // LCOV_EXCL_LINE
                    "In underdetermined system we are expecting dual coefficients as an "
                    "initial guess for a CG solver. If you want to use primal "
                    "coefficients as a starting point consider using LBFGS or Coordinate "
                    "Descent solver. ");
        } else {
            copycoefs &= usr_ncoefs >= nfeat;
        }

        try {
            if (copycoefs) {
                coef.resize(ncoef);
                dual_coef.resize(nsamples);
                // user provided starting coefficients, check, copy and use.
                // copy first nfeat elements, then check the intercept
                if (use_dual_coefs) {
                    for (da_int j = 0; j < nsamples; j++)
                        dual_coef[j] = coefs[j];
                } else {
                    for (da_int j = 0; j < nfeat; j++)
                        coef[j] = coefs[j];
                    if (intercept) {
                        coef[ncoef - 1] = usr_ncoefs >= ncoef ? coefs[ncoef - 1] : (T)0;
                    }
                    // Scale coefficient once we have the scaling factors
                    if (scaling != scaling_t::none)
                        scale_warmstart();
                }
            } else {
                coef.resize(ncoef, (T)0);
                dual_coef.resize(nsamples, (T)0);
            }
        } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
            return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation error");
        }

        // last so to capture all option changes by the solver
        opts.get("print options", val, prn);
        if (prn)
            opts.print_options();
        // Start clock
        clock = std::chrono::system_clock::now();
        switch (method_id) {
        case linmod_method::lbfgsb:
            // l2 regularization, standard linear least-squares using L-BFGS-B
            status = fit_linreg_lbfgs();
            break;

        case linmod_method::qr:

            // No regularization, standard linear least-squares through QR factorization
            status = qr_lsq();

            break;

        case linmod_method::coord:
            // Elastic Nets (l1 + l2 regularization) Coordinate Descent Method
            status = fit_linreg_coord();
            break;

        case linmod_method::svd:
            // Call SVD method to solve linear regression (L2 or no regularisation)
            status = fit_linreg_svd();
            break;

        case linmod_method::cholesky:
            // Call Cholesky method to solve linear regression (L2 or no regularisation)
            status = fit_linreg_cholesky();
            break;

        case linmod_method::cg:
            // Call Conjugate Gradient method to solve Ridge regression (L2)
            status = fit_linreg_cg();
            break;

        default:
            // should not happen
            return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Unexpectedly an invalid optimization solver was requested.");
            break;
        }
        // Record time
        time = std::chrono::duration<T>(std::chrono::system_clock::now() - clock).count();
        if (status != da_status_success)
            return status; // Error message already loaded

        // Revert scaling on coefficients
        if (scalingint) {
            revert_scaling();
            if (method_id == linmod_method::coord || method_id == linmod_method::lbfgsb) {
                // Update the objective value in info array
                T uloss{-1}; // unscalled loss
                const T l1regul = udata->l1reg;
                const T l2regul = udata->l2reg;
                T *tmp;
                if (method_id == linmod_method::coord) {
                    // use temporary storage of coord
                    stepfun_usrdata_linreg<T> *data = (stepfun_usrdata_linreg<T> *)udata;
                    tmp = data->residual.data();
                } else { // BFGS
                         // use temporary storage from BFGS
                    cb_usrdata_linreg<T> *data = (cb_usrdata_linreg<T> *)udata;
                    tmp = data->matvec.data();
                }
                loss_mse(nsamples, nfeat, XUSR, intercept, l1regul, l2regul, coef.data(),
                         yusr, &uloss, tmp);
                tmp = nullptr;
                status = opt->set_info(da_optim::info_t::info_objective, uloss);
                if (status != da_status_success)
                    return status;
            }
        }
        break;

    case linmod_model_logistic:

        // y rhs is assumed to only contain values from 0 to K-1 (K being the number of classes)
        nclass = (da_int)(std::round(*std::max_element(y, y + nsamples)) + 1);
        ncoef = (nclass - 1) * nfeat;
        if (intercept)
            ncoef += nclass - 1;
        copycoefs = (coefs != nullptr) && (usr_ncoefs >= ncoef);

        try {
            if (copycoefs) {
                coef.resize(ncoef);
                // user provided starting coefficients, check, copy and use.
                for (da_int j = 0; j < ncoef; j++)
                    coef[j] = coefs[j];
            } else {
                coef.resize(ncoef, (T)0);
            }
        } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
            return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation error");
        }

        // FIXME Add box-bound

        // FIXME Add Scaling for now (option is ignored)

        status = fit_logreg_lbfgs();
        if (status != da_status_success)
            return status; // Error message already loaded
        break;

        // FIXME Revert scaling on coefficients

    default:
        return da_error(this->err, da_status_not_implemented, // LCOV_EXCL_LINE
                        "Unexpectedly an invalid linear model was requested.");
    }

    model_trained = true;
    return da_status_success;
}

/* fit a linear regression model with the coordinate descent method */
template <class T> da_status linear_model<T>::fit_linreg_coord() {
    da_status status = da_status_success;
    try {
        udata = new stepfun_usrdata_linreg<T>(X, y, nsamples, nfeat, intercept, lambda,
                                              alpha, std_xv.data(), scaling);
    } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    status = init_opt_method(linmod_method::coord);
    if (status != da_status_success) {
        return status; // Error message already loaded
    }
    // Add callback
    if (opt->add_stepfun(stepfun_linreg<T>) != da_status_success) {
        return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly linear model provided an invalid step "
                        "function pointer.");
    }
    // ready to solve
    status = opt->solve(coef, udata);
    if (status == da_status_success || this->err->get_severity() != DA_ERROR) {
        // either success or warning with usable solution, continue
        status = this->err->clear();
    } else {
        status = da_error(this->err, da_status_operation_failed,
                          "Optimization step failed, check model or try "
                          "different solver.");
    }
    return status; // Error message already loaded
}

/* fit a linear regression model with the lbfgs method */
template <class T> da_status linear_model<T>::fit_linreg_lbfgs() {
    da_status status = da_status_success;
    try {
        udata = new cb_usrdata_linreg<T>(X, y, nsamples, nfeat, intercept, lambda, alpha);
    } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    status = init_opt_method(linmod_method::lbfgsb);
    if (status != da_status_success) {
        return status; // Error message already loaded
    }
    // Add callbacks
    if (opt->add_objfun(objfun_mse<T>) != da_status_success) {
        return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly linear model provided an invalid objective "
                        "function pointer.");
    }
    if (opt->add_objgrd(objgrd_mse<T>) != da_status_success) {
        return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly linear model provided an invalid objective "
                        "gradient function pointer.");
    }
    status = opt->solve(coef, udata);
    if (status == da_status_success || this->err->get_severity() != DA_ERROR)
        // either success or warning with usable solution, continue
        status = this->err->clear();
    else
        status = da_error(this->err, da_status_operation_failed,
                          "Optimization step failed, rescale problem or request "
                          "different solver.");

    return status; // Error message already loaded
}

/* fit a logistic regression model with the lbfgs method */
template <class T> da_status linear_model<T>::fit_logreg_lbfgs() {
    da_status status = da_status_success;
    status = init_opt_method(linmod_method::lbfgsb);
    if (status != da_status_success) {
        return status; // Error message already loaded
    }
    try {
        udata = new cb_usrdata_logreg<T>(X, y, nsamples, nfeat, intercept, lambda, alpha,
                                         nclass);
    } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }

    if (opt->add_objfun(objfun_logistic<T>) != da_status_success) {
        return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly linear model provided an invalid objective "
                        "function pointer.");
    }
    if (opt->add_objgrd(objgrd_logistic<T>) != da_status_success) {
        return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly linear model provided an invalid objective "
                        "gradient function pointer.");
    }
    status = opt->solve(coef, udata);
    if (status == da_status_success || this->err->get_severity() != DA_ERROR) {
        // Solver managed to provide a usable solution
        return this->err->clear(); // Clear warning and return
    } else {
        // Hard error, no usable coef, terminate.
        return status; // Error message already loaded
    }
}

/* Compute least squares factorization from QR factorization */
// TODO: QR variation for undetermined matrices
template <typename T> da_status linear_model<T>::qr_lsq() {
    try {
        qr = new qr_data<T>(nsamples, nfeat);
    } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }

    /* Compute QR factorization */
    da_int info = 1, nrhs = 1;
    da::geqrf(&qr->n_row, &qr->n_col, X, &qr->n_row, qr->tau.data(), qr->work.data(),
              &qr->lwork, &info);
    if (info != 0) {
        return da_error( // LCOV_EXCL_LINE
            err, da_status_internal_error,
            "encountered an unexpected error in the QR factorization (geqrf)");
    }
    if (is_well_determined) {
        /* Compute Q^tb*/
        char side = 'L', trans = 'T';
        da::ormqr(&side, &trans, &nsamples, &nrhs, &nfeat, X, &nsamples, qr->tau.data(),
                  y, &nsamples, qr->work.data(), &qr->lwork, &info);
        if (info != 0) {
            return da_error( // LCOV_EXCL_LINE
                err, da_status_internal_error,
                "encountered an unexpected error in the QR factorization (ormqr)");
        }
        /* triangle solve R^-1*Q^Tb */
        char uplo = 'U', diag = 'N';
        trans = 'N';
        da::trtrs(&uplo, &trans, &diag, &nfeat, &nrhs, X, &nsamples, y, &nsamples, &info);
        if (info != 0) {
            return da_error(
                err, da_status_internal_error, // LCOV_EXCL_LINE
                "encountered an unexpected error in the triangle solve (trtrs)");
        }
        for (da_int i = 0; i < nfeat; i++)
            coef[i] = y[i];
    } else {
        /* triangle solve R^-t*b */
        char uplo = 'U', diag = 'N', trans = 'T';
        da::trtrs(&uplo, &trans, &diag, &qr->n_col, &nrhs, X, &qr->n_row, y, &qr->n_col,
                  &info);
        if (info != 0) {
            return da_error(
                err, da_status_internal_error, // LCOV_EXCL_LINE
                "encountered an unexpected error in the triangle solve (trtrs)");
        }

        /* Compute Q*R^-t*b*/
        char side = 'L';
        trans = 'N';
        for (da_int i = 0; i < qr->n_col; i++) {
            coef[i] = y[i];
        }
        da::ormqr(&side, &trans, &nfeat, &nrhs, &nsamples, X, &nfeat, qr->tau.data(),
                  coef.data(), &nfeat, qr->work.data(), &qr->lwork, &info);
        if (info != 0) {
            return da_error( // LCOV_EXCL_LINE
                err, da_status_internal_error,
                "encountered an unexpected error in the QR factorization (ormqr)");
        }
    }
    return da_status_success;
}

template <typename T> da_status linear_model<T>::fit_linreg_cg() {
    da_status status = da_status_success;
    /* Get tolerance parameter */
    T tol;
    if (this->opts.get("optim convergence tol", tol) != da_status_success) {
        return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly <optim convergence tol> option not "
                        "found in the linear model option registry.");
    }
    /* Get maximum iterations */
    da_int maxit;
    if (this->opts.get("optim iteration limit", maxit) != da_status_success) {
        return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly <optim iteration limit> option not "
                        "found in the linear model "
                        "option registry.");
    }

    try {
        cg = new cg_data<T>(nsamples, ncoef, tol, maxit);
    } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    } catch (std::runtime_error &) {                   // LCOV_EXCL_LINE
        return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Internal error with CG solver");
    }

    setup_xtx_xty(X, y, cg->A, cg->b);

    // In case of providing initial coefficients we want to overwrite already initialised and filled with 0 cg->coef vector
    // otherwise we leave it filled with 0 as a starting point.
    if (copycoefs) {
        if (is_well_determined) {
            memcpy(cg->coef.data(), coef.data(), sizeof(T) * ncoef);
        } else {
            memcpy(cg->coef.data(), dual_coef.data(), sizeof(T) * nsamples);
        }
    }

    /* Solve Ax = b using CG solver*/
    status = cg->compute_cg();
    if (status != da_status_success) {
        switch (status) {
        case da_status_memory_error:
            return da_error(err, status, // LCOV_EXCL_LINE
                            "Encountered memory error in CG solver.");
        case da_status_numerical_difficulties:
            da_warn(err, status, // LCOV_EXCL_LINE
                    "Encountered numerically difficult problem, use SVD solver "
                    "for more stable solution.");
            break;
        case da_status_maxit:
            da_warn(err, status, // LCOV_EXCL_LINE
                    "Reached maximum number of iterations.");
            break;
        default:
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Encountered unexpected error in CG solver.");
        }
    }

    /* Save results into coefficient array */
    if (is_well_determined) {
        for (da_int i = 0; i < ncoef; i++)
            coef[i] = cg->coef[i];
    } else {
        /* Compute coefficient from dual coefficient */
        da_blas::cblas_gemv(CblasColMajor, CblasTrans, nsamples, nfeat, cg->alpha, X,
                            nsamples, cg->coef.data(), 1, cg->beta, coef.data(), 1);
    }

    return da_status_success;
}

/* Compute Ridge regression with SVD */
template <typename T> da_status linear_model<T>::fit_linreg_svd() {

    try {
        svd = new svd_data<T>(nsamples, nfeat);
    } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }

    /* Compute SVD s.t X = UDV^T */
    da_int info = 1;
    char jobz = 'S';

    da::gesdd(&jobz, &nsamples, &nfeat, X, &nsamples, svd->S.data(), svd->U.data(),
              &nsamples, svd->Vt.data(), &svd->min_order, svd->work.data(), &svd->lwork,
              svd->iwork.data(), &info);
    if (info != 0) {
        return da_error( // LCOV_EXCL_LINE
            err, da_status_internal_error,
            "encountered an unexpected error in the SVD (gesdd)");
    }

    /* Update diagonal entries of D = D/(D^2+lambda) */
    if (this->lambda != 0) {
        for (da_int i = 0; i < svd->min_order; i++)
            svd->S[i] /= svd->S[i] * svd->S[i] + this->lambda;
    } else {
        for (da_int i = 0; i < svd->min_order; i++) {
            // Small singular value causes large reciprocal
            if (svd->S[i] >
                1e2 * std::numeric_limits<T>::epsilon() * std::max(svd->S[0], (T)1)) {
                svd->S[i] = 1 / svd->S[i];
            } else {
                svd->S[i] = 0;
            }
        }
    }

    /* Compute vector of shape (min_order, 1) temp = U^t*y */
    da_blas::cblas_gemv(CblasColMajor, CblasTrans, nsamples, svd->min_order, svd->alpha,
                        svd->U.data(), nsamples, y, 1, svd->beta, svd->temp.data(), 1);

    /* Update vector of shape (min_order, 1) temp = D*temp */
    for (da_int i = 0; i < svd->min_order; i++)
        svd->temp[i] = svd->S[i] * svd->temp[i];

    /* Compute coeffient vector of shape (n, 1) coef = V*temp */
    da_blas::cblas_gemv(CblasColMajor, CblasTrans, svd->min_order, nfeat, svd->alpha,
                        svd->Vt.data(), svd->min_order, svd->temp.data(), 1, svd->beta,
                        coef.data(), 1);

    return da_status_success;
}

/* Compute Ridge regression with Cholesky */
template <typename T> da_status linear_model<T>::fit_linreg_cholesky() {
    try {
        cholesky = new cholesky_data<T>(nsamples, ncoef);
    } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }

    setup_xtx_xty(X, y, cholesky->A, cholesky->b);

    da_int info = 1;
    char uplo = 'U';
    /* Solve Ax=b with Cholesky method */
    da_int nrhs = 1;
    da::potrf(&uplo, &cholesky->min_order, cholesky->A.data(), &cholesky->min_order,
              &info);
    if (info != 0) {
        return da_error( // LCOV_EXCL_LINE
            err, da_status_numerical_difficulties,
            "Cannot perform Cholesky factorization (potrf). Matrix is not full rank. "
            "Consider choosing another solver.");
    }

    da::potrs(&uplo, &cholesky->min_order, &nrhs, cholesky->A.data(),
              &cholesky->min_order, cholesky->b.data(), &cholesky->min_order, &info);
    if (info != 0) {
        return da_error( // LCOV_EXCL_LINE
            err, da_status_internal_error,
            "Cannot solve linear equation with Cholesky method. (potrs)");
    }

    /* Save results into coefficient array */
    if (is_well_determined) {
        for (da_int i = 0; i < ncoef; i++)
            coef[i] = cholesky->b[i];
    } else {
        /* Compute coefficient from dual coefficient */
        da_blas::cblas_gemv(CblasColMajor, CblasTrans, nsamples, nfeat, cholesky->alpha,
                            X, nsamples, cholesky->b.data(), 1, cholesky->beta,
                            coef.data(), 1);
    }

    return da_status_success;
}

/* Option methods */
template <typename T> da_status linear_model<T>::validate_options(da_int method) {
    switch (mod) {
    case (linmod_model_mse):
        // User wants to solve Lasso/Elastic net with something else than coord
        if (method != linmod_method::coord && alpha > T(0) && lambda != T(0))
            return da_error(this->err, da_status_incompatible_options,
                            "This solver cannot be used for Lasso/Elastic Net "
                            "regression. Please use coordinate descent.");
        // User wants to use QR with regularisation
        else if (method == linmod_method::qr && lambda != T(0))
            return da_error(this->err, da_status_incompatible_options,
                            "The QR solver is incompatible with regularization.");
        // User wants to use coord with scaling other than scale only or standardise (or 'auto')
        else if (method == linmod_method::coord &&
                 (scaling == scaling_t::none || scaling == scaling_t::centering))
            return da_error(this->err, da_status_incompatible_options,
                            "Coordinate Descent solver can only be used with "
                            "scaling=='scale only' or 'standardize'.");
        // User wants to solve with intercept without scaling in underdetermined case, we cannot
        // do it since only correct strategy that don't penalise intercept is to center data
        else if (!is_well_determined && scaling == scaling_t::none && intercept &&
                 method != linmod_method::lbfgsb)
            // Excluded LBFGS from this if statement as it handles intercept internally
            return da_error(this->err, da_status_incompatible_options,
                            "Systems that are not over-determined cannot be solved with "
                            "intercept without centering.");
        // Extension of the test above to the well-determined situations
        else if ((method == linmod_method::qr || method == linmod_method::svd) &&
                 scaling == scaling_t::none && intercept)
            return da_error(
                this->err, da_status_incompatible_options,
                "This solver requires scaling = centering to compute intercept.");
        // User wants intercept from underdetermined QR
        else if (method == linmod_method::qr && !is_well_determined && intercept)
            return da_error(this->err, da_status_incompatible_options,
                            "The QR solver cannot compute intercept in "
                            "underdetermined situation.");
        // User wants QR in underdetermined and standardize scaling case (when centering underdetermined, matrix becomes low-rank)
        else if (method == linmod_method::qr && !is_well_determined &&
                 scaling == scaling_t::standardize)
            return da_error(this->err, da_status_incompatible_options,
                            "QR cannot solve underdetermined system with 'standardize' "
                            "scaling. For robustness try SVD solver");
        break;
    case (linmod_model_logistic):
        if (method != linmod_method::lbfgsb)
            // solver not valid for logistic regression
            return da_error(this->err, da_status_incompatible_options,
                            "This solver is incompatible with the logistic "
                            "regression model.");
        else if (method == linmod_method::lbfgsb && alpha != T(0) && lambda != T(0))
            return da_error(this->err, da_status_incompatible_options,
                            "The BFGS solver is incompatible with a 1-norm "
                            "regularization term.");
        break;
    default:
        return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly an invalid regression model was set as method.");
        break;
    }
    return da_status_success;
}

template <typename T> da_status linear_model<T>::choose_method() {
    switch (mod) {
    case (linmod_model_mse):
        // Cholesky for normal and L2 regression
        if (alpha == (T)0) {
            opts.set("optim method", "cholesky", da_options::solver);
        } else
            // Coordinate Descent for L1 [and L2 combined: Elastic Net]
            opts.set("optim method", "coord", da_options::solver);
        break;
    case (linmod_model_logistic):
        // Here we choose L-BFGS-B over Coordinate Descent
        if (alpha == (T)0)
            // L-BFGS-B handles L2 regularization
            opts.set("optim method", "lbfgs", da_options::solver);
        else
            // Coordinate Descent for L1 [and L2 combined: Elastic Net]
            // opts.set("optim method", "coord", da_options::solver);
            // TODO FIXME Enable this once coord+logistic is implemented
            // --> uncomment opts.set("optim method", "coord", da_options::solver);
            return da_error(this->err, da_status_not_implemented, // LCOV_EXCL_LINE
                            "Not yet implemented");
        break;
    default:
        // Shouldn't happen (would be nice to trap these with C++23 std::unreachable())
        return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "New linmod model?");
    }

    return da_status_success;
}

/* Transform the problem data and store extra information related to the rescaling.
     * For exact equations see documentation on standardization within the
     * Linear models section.
     *
     * Rescaling will interchangeably refer to scaling (only) and stardardizing.
     *
     * The rescaled model at exit of this function will modify
     * 1. X data matrix
     * 2. y responce vector
     * 3. box bounds l = user_l / yscale and is standardized the xscale[j] * user_l[j] / yscale, same for u
     *    For now there is not support for this feature
     *
     * N := nsamples.
     *
     * |------------------------------------------------------------------------------------------------------------------------------------|
     * |    Object        |                                             Transform type                                                      |
     * |                  |standardize+intrcpt|    standardize         | scale+intrcpt  | scale            |centering+intrcpt |  centering  |
     * |---------------------------------------------------------------------------------------------------|--------------------------------|
     * | X (copy of XUSR) |    1     X-mu(X)  |      1       X         | X-mu(X)        |    X             |  X - mu(X)       |     X       |
     * |                  | ------- --------- |   ------- --------     | -------        | -------          |                  |             |
     * |                  | sqrt(N)  sigma(X) |   sqrt(N) sigma(X)     | sqrt(N)        | sqrt(N)          |                  |             |
     * |------------------------------------------------------------------------------------------------------------------------------------|
     * | y (copy of yusr) |    1     Y-mu(Y)  |      1        Y        |   1     Y-mu(Y)|   1        Y     |  Y - mu(Y)       |     Y       |
     * |                  | ------- --------  |   ------- --------     |------- --------|------- --------- |                  |             |
     * |                  | sqrt(N) sigma(Y)  |   sqrt(N) norm(Y)      |sqrt(N) sigma(Y)|sqrt(N)  norm(Y)  |                  |             |
     * |------------------------------------------------------------------------------------------------------------------------------------|
     * | Storage scheme   |                    [ X[0], X[1], ..., X[N]; Y ]                                                                 |
     * |------------------------------------------------------------------------------------------------------------------------------------|
     * | std_shifts       | [ mu(X); mu(Y)]   |  [ 0,0,...,0; 0 ]      |[ mu(X); mu(Y)] |  [0,0,...,0;0]   | [mu(X); mu(Y)]   |      0      |
     * |-------------------------------------------------------------------------------------------------------------------------------------
     * | std_scales       |[sigma(X);sigma(Y)]|[sigma(X);nrm(Y)/sqrt(N)|[1;sigma(Y)]    |[1;nrm(Y)/sqrt(N)]|       1          |      1      |
     * |-------------------------------------------------------------------------------------------------------------------------------------
     * | std_xv[j]        |         1         |<X[j],X[j]>/N*var(X[j]) | var(X[j])      | <X[j],X[j]>/N    |       1          |      1      |
     * |-------------------------------------------------------------------------------------------------------------------------------------
     *
     * Note see reverse_scaling for reverting of the scaling on the model coefficients (solution)
     *
     */
template <typename T> da_status linear_model<T>::model_scaling(da_int method_id) {
    // For SVD and QR we still will want to copy X and y, even for scaling == none
    if (scaling == scaling_t::none && method_id != linmod_method::svd &&
        method_id != linmod_method::qr) {
        return da_status_success;
    }

    if (X != XUSR ||
        y != yusr) { /// multi call to solver no change? intercept change? fitted=T?
        // Expecting both to match!
        return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                        "X and y are not pointing to user data.");
    }
    try {
        std_scales.assign(nfeat + 1, T(0));
        std_shifts.assign(nfeat + 1, T(0));
        std_xv.assign(nfeat, T(0));
        X = new T[nsamples * nfeat];
        y = new T[nsamples];
    } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error.");
    }
    // Transpose matrix for undetermined QR case
    da_int nrow = nsamples, ncol = nfeat;
    da_axis axis = da_axis_col; // variables used later when computing standardisation
    if (method_id == linmod_method::qr && !is_well_determined) {
        for (da_int i = 0; i < nsamples; i++)
            for (da_int j = 0; j < nfeat; j++)
                X[i * nfeat + j] = XUSR[j * nsamples + i];
        is_transposed = true;
        nrow = nfeat;
        ncol = nsamples;
        axis = da_axis_row;
    }
    // Copy data. If we transposed the matrix, data is already copied into X
    if (!is_transposed) {
        if (memcpy(X, XUSR, nsamples * nfeat * sizeof(T)) != X) {
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Could not copy data from user.");
        }
    }
    if (memcpy(y, yusr, nsamples * sizeof(T)) != y) {
        return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Could not copy data from user.");
    }

    // Without this it proceeds to: else if (!standardize && intercept)
    if (scaling == scaling_t::none) {
        return da_status_success;
    }

    // FIXME std_xxxx can be assigned (resized) only once. Currently in some
    // places it done twice.

    if (scaling == scaling_t::centering) {
        std_scales.assign(nfeat + 1, T(1));
        std_shifts.assign(nfeat + 1, T(0));
        std_xv.assign(nfeat, T(1));
        if (!intercept) {
            // data copied XUSR -> X and yusr -> y, set-up scaling vectors and exit
            return da_status_success;
        }
        if (da_basic_statistics::standardize(axis, nrow, ncol, X, nrow, nrow, 0,
                                             std_shifts.data(),
                                             (T *)nullptr) != da_status_success) {
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Call to standardize on feature matrix unexpectedly failed.");
        }
        // intercept -> shift and scale Y
        if (da_basic_statistics::standardize(da_axis_col, nsamples, 1, y, nsamples,
                                             nsamples, 0, &std_shifts[nfeat],
                                             (T *)nullptr) != da_status_success) {
            return da_error(                   // LCOV_EXCL_LINE
                err, da_status_internal_error, // LCOV_EXCL_LINE
                "Call to standardize on responce vector unexpectedly failed.");
        }
        return da_status_success;
    }

    bool standardize = scaling == scaling_t::standardize;

    // 4 distinct cases to address the four cases can be compressed into a single
    // case obfuscating the understanding.
    // Standardizing with or without intercept
    // Scaling: with or without intercept
    T sqdof{T(0)};

    if (standardize && intercept) {
        // intercept -> shift and scale X
        std_xv.assign(nfeat, T(1));
        if (da_basic_statistics::standardize(axis, nrow, ncol, X, nrow, nrow, 0,
                                             std_shifts.data(),
                                             std_scales.data()) != da_status_success) {
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Call to standardize on feature matrix unexpectedly failed.");
        }
        // intercept -> shift and scale Y
        if (da_basic_statistics::standardize(axis, nsamples, 1, y, nsamples, nsamples, 0,
                                             &std_shifts[nfeat],
                                             &std_scales[nfeat]) != da_status_success) {
            return da_error(                   // LCOV_EXCL_LINE
                err, da_status_internal_error, // LCOV_EXCL_LINE
                "Call to standardize on response vector unexpectedly failed.");
        }
    } else if (standardize && !intercept) {
        // no intercept -> scale X
        for (da_int j = 0; j < nfeat; ++j) {
            sqdof = (T)0;
            T xcj = T(0);
            for (da_int i = 0; i < nsamples; ++i) {
                T xj = is_transposed ? X[i * nfeat + j] : X[j * nsamples + i];
                sqdof += xj * xj;
                xcj += xj;
            }
            // xcj = colmean(X[:,j])^2
            xcj /= nsamples;
            xcj *= xcj;
            sqdof = sqdof / nsamples;

            // These are used for updating the coefficients (betas).
            std_xv[j] = sqdof / (sqdof - xcj);

            sqdof =
                sqrt(sqdof -
                     xcj); // This is formula for standard deviation (after rearrangement)
            std_scales[j] = sqdof; // same as with intercept: stdev using 1/nsamples
            std_shifts[j] = (T)0;  // zero
        }
        if (da_basic_statistics::standardize(axis, nrow, ncol, X, nrow, nrow, 0,
                                             (T *)nullptr,
                                             std_scales.data()) != da_status_success) {
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Call to standardize on feature matrix unexpectedly failed.");
        }
        // no intercept -> scale Y
        T ynrm = da_blas::cblas_dot(nsamples, y, 1, y, 1);
        sqdof = sqrt(ynrm / T(nsamples));
        std_scales[nfeat] = sqdof;
        std_shifts[nfeat] = (T)0;
        if (da_basic_statistics::standardize(da_axis_col, nsamples, 1, y, nsamples,
                                             nsamples, 0, (T *)nullptr,
                                             &std_scales[nfeat]) != da_status_success) {
            return da_error(                   // LCOV_EXCL_LINE
                err, da_status_internal_error, // LCOV_EXCL_LINE
                "Call to standardize on responce vector unexpectedly failed.");
        }
    } else if (!standardize && intercept) {
        // intercept -> shift and scale X
        if (da_basic_statistics::variance(axis, nrow, ncol, X, nrow, nrow,
                                          std_shifts.data(),
                                          std_xv.data()) != da_status_success) {
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Call to variance on feature matrix unexpectedly failed.");
        }
        std_scales.assign(nfeat + 1, sqrt(T(nsamples)));
        if (da_basic_statistics::standardize(axis, nrow, ncol, X, nrow, 1, 0,
                                             std_shifts.data(),
                                             std_scales.data()) != da_status_success) {
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Call to standardize on feature matrix unexpectedly failed.");
        }
        std_scales.assign(nfeat + 1, T(1));

        // intercept -> shift and scale Y
        std_scales[nfeat] = T(0);
        std_shifts[nfeat] = T(0);
        if (da_basic_statistics::variance(da_axis_col, nsamples, 1, y, nsamples, nsamples,
                                          &std_shifts[nfeat],
                                          &std_scales[nfeat]) != da_status_success) {
            return da_error(                   // LCOV_EXCL_LINE
                err, da_status_internal_error, // LCOV_EXCL_LINE
                "Call to variance on response vector unexpectedly failed.");
        }
        std_scales[nfeat] = sqrt(std_scales[nfeat]);
        T ymean = std_shifts[nfeat];
        T ys_sqn = std_scales[nfeat] * sqrt(T(nsamples));
        for (da_int j = 0; j < nsamples; ++j) {
            y[j] = (y[j] - ymean) / ys_sqn;
        }
    } else if (!standardize && !intercept) {
        // no intercept -> scale X
        T sqrtn = sqrt(T(nsamples));
        for (da_int j = 0; j < nfeat; ++j) {
            T xjdot = T(0);
            for (da_int i = 0; i < nsamples; ++i) {
                T xj = is_transposed ? X[i * nfeat + j] : X[j * nsamples + i];
                xjdot += xj * xj;
                X[j * nsamples + i] /= sqrtn;
            }
            // These are used for updating the coefficients (betas).
            std_xv[j] = xjdot / T(nsamples);
            std_scales[j] = T(1);
            std_shifts[j] = T(0);
        }

        // no intercept -> scale Y
        T ynrm = sqrt(da_blas::cblas_dot(nsamples, y, 1, y, 1));
        std_scales[nfeat] = ynrm / sqrtn;
        std_shifts[nfeat] = (T)0;
        for (da_int j = 0; j < nsamples; ++j) {
            y[j] = y[j] / ynrm;
        }
    }

    // FIXME rescale box-bounds: l[j] <= x[j] <= u[j]
    // if (scale_only) l <- l / yscale
    // if (standardize) l <- xscale[j] * l[j] / yscale
    // same goes for u

    return da_status_success;
}

/* Revert scaling / standardization for coefficients so they are on the same
 * units of original problem.
 * The reversing is much more simpler and uses a single formula regarless of
 * the type of scaling used: (std_scales and std_shifts need to be setup correctly)
 *
 * beta[k] = ( beta[k] / scale[k] ) * scale[y]
 *         = ( scale[y] / scale[k] ) * beta[k]
 *
 * if (intercept)
 *    beta[intercept] = shift[y] = mean[y]
 *                    -= ( shift[k] * beta[k] / scale[k] ) * scale[y]
 *                    -= shift[k] * ( scale[y] / scale[k] ) * beta[k]
 */

template <typename T> void linear_model<T>::revert_scaling(void) {
    if (scaling != scaling_t::none) {
        T cum0{0};
        T yscale = std_scales[nfeat];
        for (da_int k = 0; k < nfeat; ++k) {
            coef[k] = yscale / std_scales[k] * coef[k];
            cum0 += std_shifts[k] * coef[k];
        }
        if (intercept) {
            coef[nfeat] = std_shifts[nfeat] + yscale * coef[nfeat] - cum0;
        }
    }
}

/* Function used at the beginning of cholesky and cg solver to get X'X (or XX') and X'y (or not)
    X_input and y_input is data provided by user, A and b are outputs that are later used to
    solve system of linear equations Ax=b where x is coefficient matrix. */
template <typename T>
void linear_model<T>::setup_xtx_xty(const T *X_input, const T *y_input, std::vector<T> &A,
                                    std::vector<T> &b) {
    if (is_well_determined) {
        /* Compute X'X */
        da_blas::cblas_syrk(CblasColMajor, CblasUpper, CblasTrans, nfeat, nsamples,
                            (T)1.0, X_input, nsamples, (T)0.0, A.data(), ncoef);
        /* In case of intercept, the last column of X'X needs to be filled.
            Each row of that column is equal to the sum of entries of respective
            column of original X matrix */
        if (intercept) {
            da_int end = ncoef * nfeat;
            const T *X_ptr = X_input;
            for (da_int i = 0; i < nfeat; i++, X_ptr += nsamples) {
#pragma omp simd
                for (da_int j = 0; j < nsamples; j++)
                    A[end + i] += X_ptr[j];
            }
            // The last entry is the number of rows in X
            A[ncoef * ncoef - 1] = nsamples;
        }

        /* Add lambda on diagonal */
        if (this->lambda > 0)
            for (da_int i = 0; i < nfeat; i++)
                A[i * ncoef + i] += this->lambda;

        /* Compute X'y */
        da_blas::cblas_gemv(CblasColMajor, CblasTrans, nsamples, nfeat, (T)1.0, X_input,
                            nsamples, y_input, 1, (T)0.0, b.data(), 1);
        if (intercept) {
#pragma omp simd
            for (da_int i = 0; i < nsamples; i++)
                b[nfeat] += y_input[i];
        }
        /* In case of underdetermined system, use Moore-Penrose pseudoinverse */
    } else {
        /* Compute XX' */
        da_blas::cblas_syrk(CblasColMajor, CblasUpper, CblasNoTrans, nsamples, nfeat,
                            (T)1.0, X_input, nsamples, (T)0.0, A.data(), nsamples);

        /* Add lambda on diagonal */
        for (da_int i = 0; i < nsamples; i++) {
            A[i * nsamples + i] += this->lambda;
            b[i] = y_input[i];
        }
    }
}

// Apply scaling for user provided warm start coefficients
template <typename T> void linear_model<T>::scale_warmstart(void) {
    T cum0{0};
    T yscale = std_scales[nfeat];
    for (da_int k = 0; k < nfeat; ++k) {
        cum0 += std_shifts[k] * coef[k];
        coef[k] = std_scales[k] * coef[k] / yscale;
    }
    if (intercept) {
        coef[nfeat] = (coef[nfeat] - std_shifts[nfeat] + cum0) / yscale;
    }
}

} // namespace da_linmod
#endif
