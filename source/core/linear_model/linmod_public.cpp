/*
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "linmod_public.hpp"
#include "aoclda.h"
#include "da_handle.hpp"
#include "dynamic_dispatch.hpp"
#include "macros.h"

using namespace linmod_public;

template <typename T>
da_status da_linmod_select_model(da_handle handle, linmod_model mod) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err,
               return (linmod_select_model<da_linmod::linear_model<T>, T>(handle, mod)));
}

template <typename T>
da_status da_linmod_define_features(da_handle handle, da_int n_samples, da_int n_features,
                                    const T *X, da_int ldx, const T *y) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (linmod_define_features<da_linmod::linear_model<T>, T>(
                                handle, n_samples, n_features, X, ldx, y)));
}

template <typename T>
da_status da_linmod_fit_start(da_handle handle, da_int ncoefs, const T *coefs) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (linmod_fit_start<da_linmod::linear_model<T>, T>(
                                handle, ncoefs, coefs)));
}

template <typename T> da_status da_linmod_fit(da_handle handle) {
    // Call fit with no initial starting point
    return da_linmod_fit_start<T>(handle, 0, nullptr);
}

template <typename T>
da_status da_linmod_evaluate_model(da_handle handle, da_int n_samples, da_int n_features,
                                   const T *X, da_int ldx, T *predictions,
                                   const T *observations, T *loss) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (linmod_evaluate_model<da_linmod::linear_model<T>, T>(
                                handle, n_samples, n_features, X, ldx, predictions,
                                observations, loss)));
}

template da_status da_linmod_select_model<float>(da_handle, linmod_model);
template da_status da_linmod_select_model<double>(da_handle, linmod_model);
template da_status da_linmod_define_features<float>(da_handle, da_int, da_int,
                                                    const float *, da_int, const float *);
template da_status da_linmod_define_features<double>(da_handle, da_int, da_int,
                                                     const double *, da_int,
                                                     const double *);
template da_status da_linmod_fit_start<float>(da_handle, da_int, const float *);
template da_status da_linmod_fit_start<double>(da_handle, da_int, const double *);
template da_status da_linmod_fit<float>(da_handle);
template da_status da_linmod_fit<double>(da_handle);
template da_status da_linmod_evaluate_model<float>(da_handle, da_int, da_int,
                                                   const float *, da_int, float *,
                                                   const float *, float *);
template da_status da_linmod_evaluate_model<double>(da_handle, da_int, da_int,
                                                    const double *, da_int, double *,
                                                    const double *, double *);
