/*
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "nlls_public.hpp"
#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include "da_handle.hpp"

using namespace nlls_public;

template <typename T>
da_status da_nlls_define_residuals(da_handle handle, da_int n_coef, da_int n_res,
                                   da_resfun_t<T> *resfun, da_resgrd_t<T> *resgrd,
                                   da_reshes_t<T> *reshes, da_reshp_t<T> *reshp) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(
        handle->err,
        return (nlls_define_residuals<da_nlls::nlls<T>, da_resfun_t<T>, da_resgrd_t<T>,
                                      da_reshes_t<T>, da_reshp_t<T>, T>(
            handle, n_coef, n_res, resfun, resgrd, reshes, reshp)));
}

template <typename T>
da_status da_nlls_define_bounds(da_handle handle, da_int n_coef, T *lower, T *upper) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (nlls_define_bounds<da_nlls::nlls<T>, T>(
                                handle, n_coef, lower, upper)));
}

template <typename T>
da_status da_nlls_define_weights(da_handle handle, da_int n_coef, T *weights) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err, return (nlls_define_weights<da_nlls::nlls<T>, T>(
                                handle, n_coef, weights)));
}

template <typename T>
da_status da_nlls_fit(da_handle handle, da_int n_coef, T *coef, void *udata) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err,
               return (nlls_fit<da_nlls::nlls<T>, T>(handle, n_coef, coef, udata)));
}

template da_status da_nlls_define_residuals<float>(da_handle, da_int, da_int,
                                                   da_resfun_t_s *, da_resgrd_t_s *,
                                                   da_reshes_t_s *, da_reshp_t_s *);
template da_status da_nlls_define_residuals<double>(da_handle, da_int, da_int,
                                                    da_resfun_t_d *, da_resgrd_t_d *,
                                                    da_reshes_t_d *, da_reshp_t_d *);
template da_status da_nlls_define_bounds<float>(da_handle, da_int, float *, float *);
template da_status da_nlls_define_bounds<double>(da_handle, da_int, double *, double *);
template da_status da_nlls_define_weights<float>(da_handle, da_int, float *);
template da_status da_nlls_define_weights<double>(da_handle, da_int, double *);
template da_status da_nlls_fit<float>(da_handle, da_int, float *, void *);
template da_status da_nlls_fit<double>(da_handle, da_int, double *, void *);
