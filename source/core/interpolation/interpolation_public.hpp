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

#include "aoclda.h"
#include "da_handle.hpp"
#include "dynamic_dispatch.hpp"
#include "interpolation.hpp"
#include "macros.h"

#ifndef INTERPOLATION_PUBLIC_HPP
#define INTERPOLATION_PUBLIC_HPP

namespace interpolation_public {
template <typename interp_class, typename T>
da_status interpolation_select_model(da_handle handle, da_interpolation_model model) {
    interp_class *interp = dynamic_cast<interp_class *>(handle->get_alg_handle<T>());
    if (interp == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_interpolation or "
            "handle is invalid.");

    return interp->select_model(model);
}

template <typename interp_class, typename T>
da_status interpolation_set_sites(da_handle handle, da_int n_sites, const T *x) {
    interp_class *interp = dynamic_cast<interp_class *>(handle->get_alg_handle<T>());
    if (interp == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_interpolation or "
            "handle is invalid.");

    return interp->set_sites(n_sites, x);
}

template <typename interp_class, typename T>
da_status interpolation_set_sites_uniform(da_handle handle, da_int n_sites, T x_start,
                                          T x_end) {
    interp_class *interp = dynamic_cast<interp_class *>(handle->get_alg_handle<T>());
    if (interp == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_interpolation or "
            "handle is invalid.");

    return interp->set_sites_uniform(n_sites, x_start, x_end);
}

template <typename interp_class, typename T>
da_status interpolation_set_values(da_handle handle, da_int n, da_int dim,
                                   const T *y_data, da_int ldy, da_int order) {
    interp_class *interp = dynamic_cast<interp_class *>(handle->get_alg_handle<T>());
    if (interp == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_interpolation or "
            "handle is invalid.");

    return interp->set_values(n, dim, y_data, ldy, order);
}

template <typename interp_class, typename T>
da_status interpolation_search_cells(da_handle handle, da_int n_eval, const T *x_eval,
                                     da_int *cells) {
    interp_class *interp = dynamic_cast<interp_class *>(handle->get_alg_handle<T>());
    if (interp == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_interpolation or "
            "handle is invalid.");

    return interp->search_cells(n_eval, x_eval, cells);
}

template <typename interp_class, typename T>
da_status interpolation_interpolate(da_handle handle) {
    interp_class *interp = dynamic_cast<interp_class *>(handle->get_alg_handle<T>());
    if (interp == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_interpolation or "
            "handle is invalid.");

    return interp->interpolate();
}

template <typename interp_class, typename T>
da_status interpolation_set_boundary_conditions(da_handle handle, da_int dim,
                                                da_int left_order, const T *left_values,
                                                da_int right_order,
                                                const T *right_values) {
    interp_class *interp = dynamic_cast<interp_class *>(handle->get_alg_handle<T>());
    if (interp == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_interpolation or "
            "handle is invalid.");

    return interp->set_boundary_conditions(dim, left_order, left_values, right_order,
                                           right_values);
}

template <typename interp_class, typename T>
da_status interpolation_evaluate(da_handle handle, da_int n_eval, const T *x_eval,
                                 T *y_eval, da_int n_orders, da_int *orders) {
    interp_class *interp = dynamic_cast<interp_class *>(handle->get_alg_handle<T>());
    if (interp == nullptr)
        return da_error(
            handle->err, da_status_invalid_handle_type,
            "handle was not initialized with handle_type=da_handle_interpolation or "
            "handle is invalid.");

    return interp->evaluate(n_eval, x_eval, y_eval, n_orders, orders);
}

} // namespace interpolation_public

#endif // INTERPOLATION_PUBLIC_HPP