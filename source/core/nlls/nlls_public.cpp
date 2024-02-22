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

#include "aoclda.h"
#include "da_handle.hpp"

da_status da_nlls_define_residuals_d(da_handle handle, da_int n_coef, da_int nres,
                                     da_resfun_t_d *resfun, da_resgrd_t_d *resgrd,
                                     da_reshes_t_d *reshes, da_reshp_t_d *reshp) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->nlls_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nlls or "
                        "handle is invalid.");

    da_status status = handle->nlls_d->define_residuals(n_coef, nres);
    if (status != da_status_success)
        return status; // Error message already loaded
    return handle->nlls_d->define_callbacks(resfun, resgrd, reshes, reshp);
}
da_status da_nlls_define_residuals_s(da_handle handle, da_int n_coef, da_int nres,
                                     da_resfun_t_s *resfun, da_resgrd_t_s *resgrd,
                                     da_reshes_t_s *reshes, da_reshp_t_s *reshp) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    if (handle->nlls_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nlls or "
                        "handle is invalid.");

    da_status status = handle->nlls_s->define_residuals(n_coef, nres);
    if (status != da_status_success)
        return status; // Error message already loaded
    return handle->nlls_s->define_callbacks(resfun, resgrd, reshes, reshp);
}

da_status da_nlls_define_bounds_d(da_handle handle, da_int n_coef, double *lower,
                                  double *upper) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->nlls_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nlls or "
                        "handle is invalid.");

    return handle->nlls_d->define_bounds(n_coef, lower, upper);
}
da_status da_nlls_define_bounds_s(da_handle handle, da_int n_coef, float *lower,
                                  float *upper) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    if (handle->nlls_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nlls or "
                        "handle is invalid.");

    return handle->nlls_s->define_bounds(n_coef, lower, upper);
}

da_status da_nlls_define_weights_d(da_handle handle, da_int n_coef, double *weights) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->nlls_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nlls or "
                        "handle is invalid.");

    return handle->nlls_d->define_weights(n_coef, weights);
}
da_status da_nlls_define_weights_s(da_handle handle, da_int n_coef, float *weights) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    if (handle->nlls_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nlls or "
                        "handle is invalid.");

    return handle->nlls_s->define_weights(n_coef, weights);
}

da_status da_nlls_fit_d(da_handle handle, da_int n_coef, double *coef, void *udata) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");
    if (handle->nlls_d == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nlls or "
                        "handle is invalid.");

    return handle->nlls_d->fit(n_coef, coef, udata);
}
da_status da_nlls_fit_s(da_handle handle, da_int n_coef, float *coef, void *udata) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");
    if (handle->nlls_s == nullptr)
        return da_error(handle->err, da_status_invalid_handle_type,
                        "handle was not initialized with handle_type=da_handle_nlls or "
                        "handle is invalid.");

    return handle->nlls_s->fit(n_coef, coef, udata);
}
