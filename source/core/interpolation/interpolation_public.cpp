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

da_status da_interpolation_select_model_d(da_handle handle,
                                          da_interpolation_model model) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");

    da_interpolation_model model_cpp = static_cast<da_interpolation_model>(model);

    DISPATCHER(
        handle->err,
        return (
            interpolation_select_model<da_interpolation::interpolation_p<double>, double>(
                handle, model_cpp)))

    return da_status_success;
}

da_status da_interpolation_select_model_s(da_handle handle,
                                          da_interpolation_model model) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");

    da_interpolation_model model_cpp = static_cast<da_interpolation_model>(model);

    DISPATCHER(
        handle->err,
        return (
            interpolation_select_model<da_interpolation::interpolation_p<float>, float>(
                handle, model_cpp)))

    return da_status_success;
}

da_status da_interpolation_set_sites_d(da_handle handle, da_int n_sites,
                                       const double *x) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");

    DISPATCHER(
        handle->err,
        return (
            interpolation_set_sites<da_interpolation::interpolation_p<double>, double>(
                handle, n_sites, x)))

    return da_status_success;
}

da_status da_interpolation_set_sites_s(da_handle handle, da_int n_sites, const float *x) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");

    DISPATCHER(
        handle->err,
        return (interpolation_set_sites<da_interpolation::interpolation_p<float>, float>(
            handle, n_sites, x)))

    return da_status_success;
}

da_status da_interpolation_set_sites_uniform_d(da_handle handle, da_int n_sites,
                                               double x_start, double x_end) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");

    DISPATCHER(
        handle->err,
        return (interpolation_set_sites_uniform<da_interpolation::interpolation_p<double>,
                                                double>(handle, n_sites, x_start, x_end)))

    return da_status_success;
}

da_status da_interpolation_set_sites_uniform_s(da_handle handle, da_int n_sites,
                                               float x_start, float x_end) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");

    DISPATCHER(
        handle->err,
        return (interpolation_set_sites_uniform<da_interpolation::interpolation_p<float>,
                                                float>(handle, n_sites, x_start, x_end)))

    return da_status_success;
}

da_status da_interpolation_set_values_d(da_handle handle, da_int n, da_int dim,
                                        const double *y_data, da_int ldy, da_int order) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");

    DISPATCHER(
        handle->err,
        return (
            interpolation_set_values<da_interpolation::interpolation_p<double>, double>(
                handle, n, dim, y_data, ldy, order)))

    return da_status_success;
}

da_status da_interpolation_set_values_s(da_handle handle, da_int n, da_int dim,
                                        const float *y_data, da_int ldy, da_int order) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");

    DISPATCHER(
        handle->err,
        return (interpolation_set_values<da_interpolation::interpolation_p<float>, float>(
            handle, n, dim, y_data, ldy, order)))

    return da_status_success;
}

da_status da_interpolation_search_cells_d(da_handle handle, da_int n_eval,
                                          const double *x_eval, da_int *cells) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");

    DISPATCHER(
        handle->err,
        return (
            interpolation_search_cells<da_interpolation::interpolation_p<double>, double>(
                handle, n_eval, x_eval, cells)))

    return da_status_success;
}

da_status da_interpolation_search_cells_s(da_handle handle, da_int n_eval,
                                          const float *x_eval, da_int *cells) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");

    DISPATCHER(
        handle->err,
        return (
            interpolation_search_cells<da_interpolation::interpolation_p<float>, float>(
                handle, n_eval, x_eval, cells)))

    return da_status_success;
}

da_status da_interpolation_interpolate_d(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");

    DISPATCHER(
        handle->err,
        return (
            interpolation_interpolate<da_interpolation::interpolation_p<double>, double>(
                handle)))

    return da_status_success;
}

da_status da_interpolation_interpolate_s(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");

    DISPATCHER(
        handle->err,
        return (
            interpolation_interpolate<da_interpolation::interpolation_p<float>, float>(
                handle)))

    return da_status_success;
}

da_status da_interpolation_set_boundary_conditions_d(da_handle handle, da_int dim,
                                                     da_int left_order,
                                                     const double *left_values,
                                                     da_int right_order,
                                                     const double *right_values) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");

    DISPATCHER(handle->err,
               return (interpolation_set_boundary_conditions<
                       da_interpolation::interpolation_p<double>, double>(
                   handle, dim, left_order, left_values, right_order, right_values)))

    return da_status_success;
}

da_status da_interpolation_set_boundary_conditions_s(da_handle handle, da_int dim,
                                                     da_int left_order,
                                                     const float *left_values,
                                                     da_int right_order,
                                                     const float *right_values) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");

    DISPATCHER(handle->err,
               return (interpolation_set_boundary_conditions<
                       da_interpolation::interpolation_p<float>, float>(
                   handle, dim, left_order, left_values, right_order, right_values)))

    return da_status_success;
}

da_status da_interpolation_evaluate_d(da_handle handle, da_int n_eval,
                                      const double *x_eval, double *y_eval,
                                      da_int n_orders, da_int *orders) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than double.");

    DISPATCHER(
        handle->err,
        return (interpolation_evaluate<da_interpolation::interpolation_p<double>, double>(
            handle, n_eval, x_eval, y_eval, n_orders, orders)))

    return da_status_success;
}

da_status da_interpolation_evaluate_s(da_handle handle, da_int n_eval,
                                      const float *x_eval, float *y_eval, da_int n_orders,
                                      da_int *orders) {
    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs
    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "The handle was initialized with a different precision type than single.");

    DISPATCHER(
        handle->err,
        return (interpolation_evaluate<da_interpolation::interpolation_p<float>, float>(
            handle, n_eval, x_eval, y_eval, n_orders, orders)))

    return da_status_success;
}
