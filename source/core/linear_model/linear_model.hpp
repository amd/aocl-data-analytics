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
#include "basic_handle.hpp"
#include "callbacks.hpp"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "lapack_templates.hpp"
#include "linmod_nln_optim.hpp"
#include "linmod_options.hpp"
#include "linmod_qr.hpp"
#include "linmod_types.hpp"
#include "optimization.hpp"
#include "options.hpp"
#include "statistical_utilities.hpp"
#include <cstdlib>
#include <cstring>
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
namespace da_linmod {
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
     */
    da_int ncoef = 0;
    std::vector<T> coef;

    /* Elastic net penalty parameters (Regularization L1: LASSO, L2: Ridge, combination => Elastic net)
     * Penalty parameters are: lambda ( (1-alpha)L2 + alpha*L1 )
     * lambda >= 0 and 0<=alpha<=1.
     */
    T alpha, lambda;

    /* optimization object to call generic algorithms */
    da_optim::da_optimization<T> *opt = nullptr;
    usrdata_base<T> *udata = nullptr;
    qr_data<T> *qr = nullptr;

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
        register_linmod_options<T>(opts);
    }
    ~linear_model();

    /* This function is called when data in the handle has changed, e.g. options
     * changed. We mark the model untrained and prepare the handle in a way that
     * it is suitable to solve again.
     *
     * TODO this needs to be optimized and
     * avoid rescaling when data was already prepared.
     */
    void refresh() {
        if (model_trained) {
            // Reset
            model_trained = false;
            if (X != XUSR)
                free(X);
            if (y != yusr)
                free(y);
            X = (T *)(XUSR);
            y = (T *)(yusr);
        }
    };

    da_status define_features(da_int nfeat, da_int nsamples, const T *X, const T *y);
    da_status select_model(linmod_model mod);
    da_status model_scaling(da_int mid);
    void revert_scaling();
    da_status fit(da_int usr_ncoefs, const T *coefs);
    da_status fit_logreg_lbfgs();
    da_status fit_linreg_lbfgs();
    da_status fit_linreg_coord();
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

    if (opt)
        delete opt;
    if (qr)
        delete qr;
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
        result[0] = (T)this->nfeat;
        result[1] = (T)this->nsamples;
        result[2] = (T)this->ncoef;
        result[3] = (T)(this->intercept ? 1.0 : 0.0);
        result[4] = alpha;
        result[5] = lambda;
        // Reserved for future use
        for (auto i = 6; i < 100; i++)
            result[i] = static_cast<T>(0);
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
    da_int maxit, prnlvl;
    std::string slv, prnopt, optstr;
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
    if (this->opts.get("print options", prnopt) != da_status_success) {
        return da_error( // LCOV_EXCL_LINE
            opt->err, da_status_internal_error,
            "Unexpectedly <print options> option not found in the linear model "
            "option registry.");
    }
    if (opt->opts.set("print options", prnopt) != da_status_success) {
        return da_error(opt->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly linear model provided an invalid value to the "
                        "<print options> option.");
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

    da_int mid, prn, intercept_int;
    std::string val, method;
    da_status status;

    // For all opts.get() it is assumed they don't fail
    opts.get("intercept", intercept_int);
    opts.get("alpha", this->alpha);
    opts.get("lambda", this->lambda);
    opts.get("optim method", method, mid);
    this->intercept = (bool)intercept_int;

    status = validate_options(mid);
    if (status != da_status_success) {
        return status; // Error message already loaded
    }
    if (method == "auto") {
        status = choose_method();
        if (status != da_status_success) {
            return status; // Error message already loaded
        }
    }
    opts.get("optim method", method, mid);

    bool copycoefs = false;
    da_int scalingint;
    std::string scalingstr;

    switch (mod) {
    case linmod_model_mse:
        ncoef = nfeat;
        if (intercept)
            ncoef += 1;

        copycoefs = (coefs != nullptr) && (usr_ncoefs >= nfeat);
        // copy if solver can use it...
        copycoefs &= mid == linmod_method::coord || mid == linmod_method::lbfgsb;

        try {
            if (copycoefs) {
                coef.resize(ncoef);
                // user provided starting coefficients, check, copy and use.
                // copy first nfeat elements, then check the intercept
                for (da_int j = 0; j < nfeat; j++)
                    coef[j] = coefs[j];
                if (intercept) {
                    coef[ncoef - 1] = usr_ncoefs >= ncoef ? coefs[ncoef - 1] : (T)0;
                }

                // FIXME: if scaling -> scale back coefficients to problem scale

            } else {
                coef.resize(ncoef, (T)0);
            }
        } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
            return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation error");
        }

        // FIXME Add box-bound

        // Scaling
        if (this->opts.get("scaling", scalingstr, scalingint) != da_status_success) {
            return da_error( // LCOV_EXCL_LINE
                this->err, da_status_internal_error,
                "Unexpectedly <scaling> option not found in the linear model "
                "option registry.");
        }
        scaling = scaling_t(scalingint);
        if (scaling == scaling_t::automatic) {
            switch (mid) {
            case linmod_method::svd: // could also default to "scale only"...
            // remove intercept by centering (and scaling) both X and y
            case linmod_method::coord:
                scaling = scaling_t::standardize;
                scalingstr = "standardize";
                break;
            default:
                scaling = scaling_t::none;
                scalingstr = "none";
                break;
            }
            // Store back the option value
            this->opts.set("scaling", scalingstr, da_options::solver);
        }

        // Scales: X and y
        status = model_scaling(mid);
        if (status != da_status_success) {
            return status; // message already loaded
        }

        if (scaling == scaling_t::standardize || scaling == scaling_t::scale_only) {
            // Rescale lambda
            lambda /= std_scales[nfeat];
        }
        // Rescale lambda when scaling != "standardize" and the solver is not coord, ie,
        // need the scaled lambda for the objective and gradient
        if ((mid == linmod_method::lbfgsb) && (scaling != scaling_t::standardize)) {
            lambda /= T(nsamples);
        }

        // last so to capture all option changes by the solver
        opts.get("print options", val, prn);
        if (prn != 0)
            opts.print_options();

        switch (mid) {
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

        default:
            // should not happen
            return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Unexpectedly an invalid optimization solver was requested.");
            break;
        }
        if (status != da_status_success)
            return status; // Error message already loaded

        // Revert scaling on coefficients
        if (scalingint) {
            revert_scaling();
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
template <typename T> da_status linear_model<T>::qr_lsq() {

    try {
        qr = new qr_data<T>(nsamples, nfeat, X, y, intercept, ncoef);
    } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }

    /* Compute QR factorization */
    da_int info = 1;
    da::geqrf(&nsamples, &ncoef, qr->X.data(), &nsamples, qr->tau.data(), qr->work.data(),
              &qr->lwork, &info);
    if (info != 0) {
        return da_error( // LCOV_EXCL_LINE
            err, da_status_internal_error,
            "encountered an unexpected error in the QR factorization (geqrf)");
    }
    /* Compute Q^tb*/
    char side = 'L', trans = 'T';
    da_int nrhs = 1;
    da::ormqr(&side, &trans, &nsamples, &nrhs, &ncoef, qr->X.data(), &nsamples,
              qr->tau.data(), qr->y.data(), &nsamples, qr->work.data(), &qr->lwork,
              &info);
    if (info != 0) {
        return da_error( // LCOV_EXCL_LINE
            err, da_status_internal_error,
            "encountered an unexpected error in the QR factorization (ormqr)");
    }
    /* triangle solve R^-t*Q^Tb */
    char uplo = 'U', diag = 'N';
    trans = 'N';
    da::trtrs(&uplo, &trans, &diag, &ncoef, &nrhs, qr->X.data(), &nsamples, qr->y.data(),
              &nsamples, &info);
    if (info != 0) {
        return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                        "encountered an unexpected error in the triangle solve (trtrs)");
    }
    for (da_int i = 0; i < ncoef; i++)
        coef[i] = qr->y[i];

    return da_status_success;
}

/* Option methods */
template <typename T> da_status linear_model<T>::validate_options(da_int method) {
    switch (mod) {
    case (linmod_model_mse):
        if (method == linmod_method::qr && lambda != T(0))
            return da_error(this->err, da_status_incompatible_options,
                            "The QR solver is incompatible with regularization.");
        else if (method == linmod_method::svd && alpha != T(0) && lambda != T(0))
            return da_error(
                this->err, da_status_incompatible_options,
                "The SVD solver is incompatible with a 1-norm regularization term.");
        break;
    case (linmod_model_logistic):
        if (method == linmod_method::qr)
            // QR not valid for logistic regression
            return da_error(this->err, da_status_incompatible_options,
                            "The QR solver is incompatible with the logistic "
                            "regression model.");
        if (method == linmod_method::lbfgsb && lambda != 0 && alpha != 0)
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
        if (lambda == (T)0)
            // QR direct method
            opts.set("optim method", "qr", da_options::solver);
        else if (alpha == (T)0)
            // L-BFGS-B handles L2 regularization
            opts.set("optim method", "lbfgs", da_options::solver);
        else
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
template <typename T> da_status linear_model<T>::model_scaling(da_int mid) {

    if (scaling == scaling_t::none) {
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
    // Copy data
    if (memcpy(X, XUSR, nsamples * nfeat * sizeof(T)) != X) {
        return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Could not copy data from user.");
    }
    if (memcpy(y, yusr, nsamples * sizeof(T)) != y) {
        return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Could not copy data from user.");
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
        if (da_basic_statistics::standardize(da_axis_col, nsamples, nfeat, X, nsamples,
                                             nsamples, 0, std_shifts.data(),
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
        if (da_basic_statistics::standardize(da_axis_col, nsamples, nfeat, X, nsamples,
                                             nsamples, 0, std_shifts.data(),
                                             std_scales.data()) != da_status_success) {
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Call to standardize on feature matrix unexpectedly failed.");
        }
        // intercept -> shift and scale Y
        if (da_basic_statistics::standardize(da_axis_col, nsamples, 1, y, nsamples,
                                             nsamples, 0, &std_shifts[nfeat],
                                             &std_scales[nfeat]) != da_status_success) {
            return da_error(                   // LCOV_EXCL_LINE
                err, da_status_internal_error, // LCOV_EXCL_LINE
                "Call to standardize on responce vector unexpectedly failed.");
        }
    } else if (standardize && !intercept) {
        // no intercept -> scale X
        for (da_int j = 0; j < nfeat; ++j) {
            sqdof = (T)0;
            T xcj = T(0);
            for (da_int i = 0; i < nsamples; ++i) {
                T xj = X[j * nsamples + i];
                sqdof += xj * xj;
                xcj += xj;
            }
            // xcj = colmean(X[:,j])^2
            xcj /= nsamples;
            xcj *= xcj;
            sqdof = sqdof / nsamples;

            // These are used for updating the coefficients (betas).
            std_xv[j] = sqdof / (sqdof - xcj);

            sqdof = sqrt(sqdof - xcj);
            std_scales[j] = sqdof; // same as with intercept: stdev using 1/nsamples
            std_shifts[j] = (T)0;  // zero
        }
        if (da_basic_statistics::standardize(da_axis_col, nsamples, nfeat, X, nsamples,
                                             nsamples, 0, (T *)nullptr,
                                             std_scales.data()) != da_status_success) {
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Call to standardize on feature matrix unexpectedly failed.");
        }
        // no intercept -> scale Y
        sqdof = sqrt(da_blas::cblas_dot(nsamples, y, 1, y, 1) / T(nsamples));
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
        if (da_basic_statistics::variance(da_axis_col, nsamples, nfeat, X, nsamples,
                                          nsamples, std_shifts.data(),
                                          std_xv.data()) != da_status_success) {
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Call to variance on feature matrix unexpectedly failed.");
        }
        std_scales.assign(nfeat + 1, sqrt(T(nsamples)));
        if (da_basic_statistics::standardize(da_axis_col, nsamples, nfeat, X, nsamples, 1,
                                             0, std_shifts.data(),
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
                "Call to variance on responce vector unexpectedly failed.");
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
                auto xj = X[j * nsamples + i];
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

} // namespace da_linmod
#endif
