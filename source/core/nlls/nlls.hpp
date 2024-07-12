/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef NLLS_HPP
#define NLLS_HPP

#include "aoclda.h"
#include "basic_handle.hpp"
#include "callbacks.hpp"
#include "da_error.hpp"
#include "optimization.hpp"
#include "options.hpp"
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

/* Nonlinear Least Square Model and solver
 *
 * Solve the problem   minimize   F(x) = 1/2 \sum_{i=0}^{n_res-1} ri(x)^2_W + sigma/p ||x||_2^p
 *                   x \in R^n_coef
 * where
 *  * ri() are the model residuals
 *  * sigma > 0, p=2,3 are the regularization hyperparams
 *
 */
namespace da_nlls {

template <typename T> class nlls : public basic_handle<T> {
  private:
    /* pointer to error trace */
    da_errors::da_error_t *err{nullptr};

    /* true if the model has been successfully trained */
    bool model_trained{false};

    /* Regression data
     * n_res: number of residuals
     */
    da_int n_res{0};

    /* Training data
     * coef: vector containing the model coefficients
     */
    da_int n_coef{0};
    std::vector<T> coef;

    /* convenience pointers to model data defined in user-space */
    T *usrlower{nullptr};
    T *usrupper{nullptr};
    T *usrweights{nullptr};

    /* pointer to user data */
    void *udata{nullptr};

    /* pointers to the callbacks */
    resfun_t<T> resfun{nullptr};
    resgrd_t<T> resgrd{nullptr};
    reshes_t<T> reshes{nullptr};
    reshp_t<T> reshp{nullptr};

  public:
    /* optimization object */
    da_optim::da_optimization<T> *opt{nullptr};

    /* constructor*/
    nlls(da_errors::da_error_t &err, da_status &status) {
        // assumes that err is valid
        this->err = &err;
        // initialize the optimization framework and options registry
        status = init_opt_solver();
    }

    /* destructor */
    ~nlls(void) {
        this->err = nullptr;
        this->udata = nullptr;
        if (this->opt)
            delete this->opt;
    };

    /* This function is called when data in the handle has changed, e.g. options
     * changed. We mark the model untrained and prepare the handle in a way that
     * it is suitable to solve again.
     */
    void refresh() {
        if (model_trained) {
            // Reset
            model_trained = false;
        }
    };

    da_status define_residuals(da_int n_coef, da_int n_res);
    da_status define_callbacks(resfun_t<T> resfun, resgrd_t<T> resgrd, reshes_t<T> reshes,
                               reshp_t<T> reshp);
    da_status define_bounds(da_int n_coef, T *lower, T *upper);
    da_status define_weights(da_int n_res, T *weights);
    da_status init_opt_solver();
    da_status fit(da_int n_coef, T *coef, void *udata);
    /* get_result (required to be defined by basic_handle) */
    da_status get_result(da_result query, da_int *dim, T *result);
    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] da_int *result);
};

template <typename T>
da_status nlls<T>::get_result(da_result query, da_int *dim, T *result) {
    // Don't return anything if model not trained!
    if (!model_trained)
        return da_warn(this->err, da_status_unknown_query,
                       "Handle does not contain data relevant to this query. Was the "
                       "last call to the solver successful?");
    switch (query) {
    case da_result::da_rinfo:
        return this->opt->get_info(*dim, result);
        break;
    default:
        return da_warn(this->err, da_status_unknown_query,
                       "The requested result could not be queried by this handle.");
    }
};
template <typename T>
da_status nlls<T>::get_result([[maybe_unused]] da_result query,
                              [[maybe_unused]] da_int *dim,
                              [[maybe_unused]] da_int *result) {
    return da_warn(this->err, da_status_unknown_query,
                   "Handle does not contain data relevant to this query.");
};

/* Store model features */
template <typename T> da_status nlls<T>::define_residuals(da_int n_coef, da_int n_res) {
    if (n_coef <= 0)
        return da_error(this->err, da_status_invalid_input, "n_coef must be positive.");
    if (n_res <= 0)
        return da_error(this->err, da_status_invalid_input, "n_res must be positive.");

    this->n_coef = n_coef;
    this->n_res = n_res;
    model_trained = false;

    return da_status_success;
}

/* Store the user callbacks */
template <typename T>
da_status nlls<T>::define_callbacks(resfun_t<T> resfun, resgrd_t<T> resgrd,
                                    reshes_t<T> reshes, reshp_t<T> reshp) {
    if (!resfun)
        return da_error(this->err, da_status_invalid_input,
                        "resfun must point to the residual function.");

    this->resfun = resfun;
    this->resgrd = resgrd;
    this->reshes = reshes;
    this->reshp = reshp;
    this->model_trained = false;

    return da_status_success;
}
/* Define the box bounds */
template <typename T>
da_status nlls<T>::define_bounds(da_int n_coef, T *lower, T *upper) {
    if (n_coef == 0) {
        this->usrlower = nullptr;
        this->usrupper = nullptr;
    } else if (this->n_coef == n_coef) {
        this->usrlower = lower;
        this->usrupper = upper;
    } else {
        return da_error(this->err, da_status_invalid_input,
                        "Invalid size of n_coef, it must match zero or the number of "
                        "variables defined: " +
                            std::to_string(this->n_coef) + ".");
    }
    this->model_trained = false;
    return da_status_success;
}

/* Define the residual weights */
template <typename T> da_status nlls<T>::define_weights(da_int n_res, T *weights) {
    if (n_res == 0) {
        this->usrweights = nullptr;
    } else if (this->n_res == n_res) {
        if (weights) {
            this->usrweights = weights;
        } else {
            return da_error(this->err, da_status_invalid_pointer,
                            "Invalid pointer to weights array, n_res is positive yet "
                            "weights is invalid. To remove weights pass n_res=0.");
        }
    } else {
        return da_error(this->err, da_status_invalid_input,
                        "Invalid size of n_res, it must match zero or the "
                        "number of residuals defined: " +
                            std::to_string(this->n_res) + ".");
    }
    this->model_trained = false;
    return da_status_success;
}

/* Common settings for optimization solvers
 * Called only once per handle initialization
 */
template <typename T> da_status nlls<T>::init_opt_solver() {

    da_status status;

    // Initialize optimization framework
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

    // Select_solver based on problem and options
    if (opt->opts.set("optim method", "ralfit", da_options::setby_t::solver) !=
        da_status_success)
        return da_error( // LCOV_EXCL_LINE
            this->err, da_status_internal_error,
            "expected option not found: optim method in the optimization registry");

    return da_status_success;
}

template <typename T> da_status nlls<T>::fit(da_int n_coef, T *coef, void *udata) {

    // Copy the starting point
    if (n_coef != 0 && n_coef != this->n_coef)
        return da_error(err, da_status_invalid_array_dimension,
                        "n_coef must match zero or the number of defined features. Array "
                        "coef must be of size zero or " +
                            std::to_string(this->n_coef) + ".");
    if (n_coef > 0 && !coef) {
        // make sure it is a valid pointer
        if (!coef) {
            return da_error(err, da_status_invalid_pointer,
                            "Pointer coef must be valid.");
        }
    }
    try {
        this->coef.resize(this->n_coef);
    } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }

    if (n_coef == 0)
        // Initial guess is zeros...
        std::fill(this->coef.begin(), this->coef.end(), T(0));
    else {
        // Copy
        for (da_int i = 0; i < n_coef; ++i) {
            this->coef[i] = coef[i];
        }
    }

    this->udata = udata;

    da_status status;

    // optimization framework
    if (!this->opt) {
        return da_error( // LCOV_EXCL_LINE
            this->err, da_status_internal_error,
            "Unexpectedly nlls did not provided a valid optimization object?");
    }
    // Add features
    if (opt->add_vars(this->n_coef) != da_status_success) {
        return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly n_coef is invalid?");
    }
    if (opt->add_res(this->n_res) != da_status_success) {
        return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly n_res is invalid?");
    }
    // Add callbacks
    if (opt->add_resfun(resfun) != da_status_success) {
        return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly nlls provided an invalid residual "
                        "function pointer?");
    }
    // These don't fail
    opt->add_resgrd(resgrd);
    opt->add_reshes(reshes);
    opt->add_reshp(reshp);
    // Add optional features
    if (opt->add_bound_cons(this->n_coef, this->usrlower, this->usrupper) !=
        da_status_success) {
        return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly failed to set the bounds?");
    }
    if (opt->add_weights(n_res, this->usrweights) != da_status_success) {
        return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpectedly failed to set the weights?");
    }

    // Make sure user did not tamper with the optim method
    if (opt->opts.set("optim method", "ralfit", da_options::setby_t::solver) !=
        da_status_success)
        return da_error(                         // LCOV_EXCL_LINE
            this->err, da_status_internal_error, // LCOV_EXCL_LINE
            "expected option not found: <optim method> in the optimization registry?");

    status = opt->solve(this->coef, udata);

    if (this->err->get_severity() == DA_ERROR) {
        return status; // Error message already loaded
    }

    // status is either success or warning with usable solution, continue
    // copy out the solution found if user's n_coef is not zero.
    for (da_int i = 0; i < n_coef; ++i) {
        coef[i] = this->coef[i];
    }

    model_trained = true;
    return status;
}

} // namespace da_nlls
#endif