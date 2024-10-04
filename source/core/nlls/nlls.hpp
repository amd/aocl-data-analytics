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

using namespace da_optim;

template <typename T> class nlls : public da_optimization<T> {
  private:
  public:
    // Constructor
    nlls(da_status &status, da_errors::da_error_t &err)
        : da_optimization<T>(status, err) {
        status = this->opts.set("optim method", "ralfit", da_options::setby_t::solver);
        if (status != da_status_success)
            da_error( // LCOV_EXCL_LINE
                this->err, da_status_internal_error,
                "expected option not found: optim method in the optimization registry");
    }

    // da_status define_residuals(da_int n_coef, da_int n_res);
    da_status define_callbacks(resfun_t<T> resfun, resgrd_t<T> resgrd, reshes_t<T> reshes,
                               reshp_t<T> reshp);
    da_status fit(da_int n_coef, T *coef, void *udata);
    da_status get_result(da_result query, da_int *dim, T *result);
    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] da_int *result);
};

template <typename T>
da_status nlls<T>::get_result(da_result query, da_int *dim, T *result) {
    return da_optimization<T>::get_result(query, dim, result);
};
template <typename T>
da_status nlls<T>::get_result([[maybe_unused]] da_result query,
                              [[maybe_unused]] da_int *dim,
                              [[maybe_unused]] da_int *result) {
    return da_optimization<T>::get_result(query, dim, result);
};

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

template <typename T> da_status nlls<T>::fit(da_int n_coef, T *coef, void *udata) {

    da_status status;
    // Copy the starting point
    if (n_coef != 0 && n_coef != this->nvar)
        return da_error(this->err, da_status_invalid_array_dimension,
                        "n_coef must match zero or the number of defined features. Array "
                        "coef must be of size zero or " +
                            std::to_string(this->nvar) + ".");
    if (n_coef > 0 && !coef) {
        // Make sure it is a valid pointer
        return da_error(this->err, da_status_invalid_pointer,
                        "Pointer coef must be valid.");
    }
    try {
        this->coef.resize(this->nvar);
    } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }

    if (n_coef == 0)
        // Initial guess is zeros...
        std::fill(this->coef.begin(), this->coef.end(), T(0));
    else {
        status = this->check_1D_array(n_coef, coef, "n_coef", "coef", 0);
        if (status != da_status_success)
            return status;
        // Copy
        for (da_int i = 0; i < n_coef; ++i) {
            this->coef[i] = coef[i];
        }
    }

    this->udata = udata;

    status = this->solve(this->coef, udata);
    if (this->err->get_severity() == DA_ERROR) {
        return status; // Error message already loaded
    }

    // Status is either success or warning with usable solution, continue
    // copy out the solution found if user's n_coef is not zero.
    for (da_int i = 0; i < n_coef; ++i) {
        coef[i] = this->coef[i];
    }

    this->model_trained = true;
    return status;
}

} // namespace da_nlls
#endif
