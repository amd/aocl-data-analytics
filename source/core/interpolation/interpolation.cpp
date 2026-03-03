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

#include "interpolation.hpp"
#include "cubic_spline/cubic_spline.hpp"
#include "da_utils.hpp"
#include "macros.h"
#include "options.hpp"

namespace ARCH {

namespace da_interpolation {

template <typename T>
interpolation_p<T>::interpolation_p(da_errors::da_error_t *err) : basic_handle<T>(err) {
    register_interpolation_options<T>(this->opts, *this->err);
};
template <typename T>
interpolation_p<T>::interpolation_p(da_errors::da_error_t &err) : basic_handle<T>(err) {
    register_interpolation_options<T>(this->opts, *this->err);
};

template <typename T>
da_status interpolation_p<T>::select_model(da_interpolation_model model) {
    if (this->model != da_interpolation_model::interpolation_unset)
        return da_error(this->err, da_status_invalid_input,
                        "Cannot change the interpolation model once it has been set.");

    this->model = model;
    switch (model) {
    case da_interpolation_model::interpolation_cubic_spline:
        interp = std::make_unique<cubic_spline<T>>(*this->err, this->opts);
        break;

    default:
        return da_error(this->err, da_status_invalid_input,
                        "The model selected is invalid.");
        break;
    }

    return da_status_success;
}

// Set interpolation sites with custom x-coordinates
template <typename T> da_status interpolation_p<T>::set_sites(da_int n, const T *x) {

    da_status status = da_status_success;

    if (x == nullptr)
        return da_error(this->err, da_status_invalid_pointer,
                        "Interpolation site array cannot be null.");

    if (model != da_interpolation_model::interpolation_unset)
        status = interp->set_sites_1d(n, x);
    else
        return da_error(this->err, da_status_invalid_input,
                        "Interpolation model not selected. Cannot set sites.");
    return status;
}

template <typename T>
da_status interpolation_p<T>::set_sites_uniform(da_int n, T x_start, T x_end) {

    da_status status = da_status_success;

    if (model != da_interpolation_model::interpolation_unset)
        status = interp->set_sites_uniform_1d(n, x_start, x_end);
    else
        return da_error(this->err, da_status_invalid_input,
                        "Interpolation model not selected. Cannot set sites.");
    return status;
}
template <typename T>
da_status interpolation_p<T>::set_values(da_int n, da_int dim, const T *y_data,
                                         da_int ldy, da_int order) {

    da_status status = da_status_success;

    if (model != da_interpolation_model::interpolation_unset)
        status = interp->set_values_1d(n, dim, y_data, ldy, order);
    else
        return da_error(this->err, da_status_invalid_input,
                        "Interpolation model not selected. Cannot set values.");
    return status;
}

// Search for cells containing query points
template <typename T>
da_status interpolation_p<T>::search_cells(da_int n_eval, const T *x_eval,
                                           da_int *cells) {
    da_status status = da_status_success;

    if (model != da_interpolation_model::interpolation_unset)
        status = interp->search_cells(n_eval, x_eval, cells);
    else
        return da_error(this->err, da_status_invalid_input,
                        "Interpolation model not selected. Cannot search cells.");
    return status;
}
template <typename T>
da_status interpolation_p<T>::set_boundary_conditions(da_int dim, da_int left_order,
                                                      const T *left_values,
                                                      da_int right_order,
                                                      const T *right_values) {
    da_status status = da_status_success;

    if (model != da_interpolation_model::interpolation_unset)
        status = interp->set_boundary_conditions(dim, left_order, left_values,
                                                 right_order, right_values);
    else
        return da_error(
            this->err, da_status_invalid_input,
            "Interpolation model not selected. Cannot set boundary conditions.");
    return status;
}
template <typename T> da_status interpolation_p<T>::interpolate() {
    da_status status = da_status_success;

    if (model != da_interpolation_model::interpolation_unset)
        status = interp->interpolate();
    else
        return da_error(this->err, da_status_invalid_input,
                        "Interpolation model not selected. Cannot interpolate.");
    return status;
}

template <typename T>
da_status interpolation_p<T>::evaluate(da_int n_eval, const T *x_eval, T *y_eval,
                                       da_int n_orders, const da_int *order) {
    da_status status = da_status_success;
    if (model != da_interpolation_model::interpolation_unset)
        status = interp->evaluate(n_eval, x_eval, y_eval, n_orders, order);
    else
        return da_error(this->err, da_status_invalid_input,
                        "Interpolation model not selected. Cannot evaluate.");
    return status;
}

template <typename T> void interpolation_p<T>::refresh() {
    if (model != da_interpolation_model::interpolation_unset)
        interp->refresh();
}

template <typename T>
da_status interpolation_p<T>::get_result(da_result query, da_int *dim, T *result) {

    da_status status = da_status_success;

    if (model != da_interpolation_model::interpolation_unset)
        status = interp->get_result(query, dim, result);
    else
        return da_error(this->err, da_status_invalid_input,
                        "Interpolation model not selected. No results available.");

    return status;
}

template <typename T>
da_status interpolation_p<T>::get_result(da_result query, da_int *dim, da_int *result) {

    da_status status = da_status_success;

    if (model != da_interpolation_model::interpolation_unset)
        status = interp->get_result(query, dim, result);
    else
        return da_error(this->err, da_status_invalid_input,
                        "Interpolation model not selected. No results available.");

    return status;
}

template class interpolation_p<double>;
template class interpolation_p<float>;

} // namespace da_interpolation

} // namespace ARCH