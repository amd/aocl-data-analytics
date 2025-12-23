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

#include "linmod_public.hpp"
#include "aoclda.h"
#include "da_handle.hpp"
#include "dynamic_dispatch.hpp"
#include "macros.h"

using namespace linmod_public;

da_status da_linmod_select_model_d(da_handle handle, linmod_model mod) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    DISPATCHER(handle->err,
               return (linmod_select_model<da_linmod::linear_model<double>, double>(
                   handle, mod)));
}

da_status da_linmod_select_model_s(da_handle handle, linmod_model mod) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    DISPATCHER(
        handle->err,
        return (linmod_select_model<da_linmod::linear_model<float>, float>(handle, mod)));
}

da_status da_linmod_define_features_d(da_handle handle, da_int n_samples,
                                      da_int n_features, const double *X, da_int ldx,
                                      const double *y) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    DISPATCHER(handle->err,
               return (linmod_define_features<da_linmod::linear_model<double>, double>(
                   handle, n_samples, n_features, X, ldx, y)));
}

da_status da_linmod_define_features_s(da_handle handle, da_int n_samples,
                                      da_int n_features, const float *X, da_int ldx,
                                      const float *y) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    DISPATCHER(handle->err,
               return (linmod_define_features<da_linmod::linear_model<float>, float>(
                   handle, n_samples, n_features, X, ldx, y)));
}

da_status da_linmod_fit_start_d(da_handle handle, da_int ncoefs, const double *coefs) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    DISPATCHER(handle->err,
               return (linmod_fit_start<da_linmod::linear_model<double>, double>(
                   handle, ncoefs, coefs)));
}

da_status da_linmod_fit_d(da_handle handle) {
    // Call fit with no initial starting point
    return da_linmod_fit_start_d(handle, 0, nullptr);
}

da_status da_linmod_fit_start_s(da_handle handle, da_int ncoefs, const float *coefs) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    DISPATCHER(handle->err,
               return (linmod_fit_start<da_linmod::linear_model<float>, float>(
                   handle, ncoefs, coefs)));
}

da_status da_linmod_fit_s(da_handle handle) {
    return da_linmod_fit_start_s(handle, 0, nullptr);
}

da_status da_linmod_evaluate_model_d(da_handle handle, da_int n_samples,
                                     da_int n_features, const double *X, da_int ldx,
                                     double *predictions, double *observations,
                                     double *loss) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    DISPATCHER(
        handle->err,
        return (linmod_evaluate_model<da_linmod::linear_model<double>, double>(
            handle, n_samples, n_features, X, ldx, predictions, observations, loss)));
}

da_status da_linmod_evaluate_model_s(da_handle handle, da_int n_samples,
                                     da_int n_features, const float *X, da_int ldx,
                                     float *predictions, float *observations,
                                     float *loss) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    DISPATCHER(
        handle->err,
        return (linmod_evaluate_model<da_linmod::linear_model<float>, float>(
            handle, n_samples, n_features, X, ldx, predictions, observations, loss)));
}
