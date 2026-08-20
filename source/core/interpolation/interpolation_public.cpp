/*
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "interpolation_public.hpp"
#include "aoclda.h"
#include "da_handle.hpp"
#include "dynamic_dispatch.hpp"
#include "macros.h"

using namespace interpolation_public;

template <typename T>
da_status da_interpolation_select_model(da_handle handle, da_interpolation_model model) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    da_interpolation_model model_cpp = static_cast<da_interpolation_model>(model);

    DISPATCHER(
        handle->err,
        return (interpolation_select_model<da_interpolation::interpolation_p<T>, T>(
            handle, model_cpp)))

    return da_status_success;
}

template <typename T>
da_status da_interpolation_set_sites(da_handle handle, da_int n_sites, const T *x) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err,
               return (interpolation_set_sites<da_interpolation::interpolation_p<T>, T>(
                   handle, n_sites, x)))

    return da_status_success;
}

template <typename T>
da_status da_interpolation_set_sites_uniform(da_handle handle, da_int n_sites, T x_start,
                                             T x_end) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(
        handle->err,
        return (interpolation_set_sites_uniform<da_interpolation::interpolation_p<T>, T>(
            handle, n_sites, x_start, x_end)))

    return da_status_success;
}

template <typename T>
da_status da_interpolation_set_values(da_handle handle, da_int n, da_int dim,
                                      const T *y_data, da_int ldy, da_int order) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err,
               return (interpolation_set_values<da_interpolation::interpolation_p<T>, T>(
                   handle, n, dim, y_data, ldy, order)))

    return da_status_success;
}

template <typename T>
da_status da_interpolation_search_cells(da_handle handle, da_int n_eval, const T *x_eval,
                                        da_int *cells) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(
        handle->err,
        return (interpolation_search_cells<da_interpolation::interpolation_p<T>, T>(
            handle, n_eval, x_eval, cells)))

    return da_status_success;
}

template <typename T> da_status da_interpolation_interpolate(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err,
               return (interpolation_interpolate<da_interpolation::interpolation_p<T>, T>(
                   handle)))

    return da_status_success;
}

template <typename T>
da_status
da_interpolation_set_boundary_conditions(da_handle handle, da_int dim, da_int left_order,
                                         const T *left_values, da_int right_order,
                                         const T *right_values) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err,
               return (interpolation_set_boundary_conditions<
                       da_interpolation::interpolation_p<T>, T>(
                   handle, dim, left_order, left_values, right_order, right_values)))

    return da_status_success;
}

template <typename T>
da_status da_interpolation_evaluate(da_handle handle, da_int n_eval, const T *x_eval,
                                    T *y_eval, da_int n_orders, da_int *orders) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_status status = handle->check_precision<T>();
    if (status != da_status_success)
        return da_error_trace(handle->err, status, "Wrong precision type.");

    DISPATCHER(handle->err,
               return (interpolation_evaluate<da_interpolation::interpolation_p<T>, T>(
                   handle, n_eval, x_eval, y_eval, n_orders, orders)))

    return da_status_success;
}

template da_status da_interpolation_select_model<float>(da_handle,
                                                        da_interpolation_model);
template da_status da_interpolation_select_model<double>(da_handle,
                                                         da_interpolation_model);
template da_status da_interpolation_set_sites<float>(da_handle, da_int, const float *);
template da_status da_interpolation_set_sites<double>(da_handle, da_int, const double *);
template da_status da_interpolation_set_sites_uniform<float>(da_handle, da_int, float,
                                                             float);
template da_status da_interpolation_set_sites_uniform<double>(da_handle, da_int, double,
                                                              double);
template da_status da_interpolation_set_values<float>(da_handle, da_int, da_int,
                                                      const float *, da_int, da_int);
template da_status da_interpolation_set_values<double>(da_handle, da_int, da_int,
                                                       const double *, da_int, da_int);
template da_status da_interpolation_search_cells<float>(da_handle, da_int, const float *,
                                                        da_int *);
template da_status da_interpolation_search_cells<double>(da_handle, da_int,
                                                         const double *, da_int *);
template da_status da_interpolation_interpolate<float>(da_handle);
template da_status da_interpolation_interpolate<double>(da_handle);
template da_status da_interpolation_set_boundary_conditions<float>(da_handle, da_int,
                                                                   da_int, const float *,
                                                                   da_int, const float *);
template da_status
da_interpolation_set_boundary_conditions<double>(da_handle, da_int, da_int,
                                                 const double *, da_int, const double *);
template da_status da_interpolation_evaluate<float>(da_handle, da_int, const float *,
                                                    float *, da_int, da_int *);
template da_status da_interpolation_evaluate<double>(da_handle, da_int, const double *,
                                                     double *, da_int, da_int *);
