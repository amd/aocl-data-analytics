/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclda_types.h"
#include "basic_handle.hpp"
#include "da_error.hpp"
#include "macros.h"
#include <cmath>
#include <vector>

#ifndef INTERPOLATION_GENERIC_HPP
#define INTERPOLATION_GENERIC_HPP
namespace ARCH {

namespace da_interpolation {
template <typename T> class interpolation_generic {
  protected:
    // Interpolation data
    // Dimensions
    da_int n_sites = 0; // Number of interpolation sites (data points)
    da_int dim_y = 0;   // number of dimensions of interpolation values y
    da_int ldy = 0;
    // Interpolation points (x_i, y_i)
    da_order storage_order = column_major;
    T *x_internal = nullptr;
    const T *x_sites = nullptr, *x_usr = nullptr;
    const T *y = nullptr;
    const T *y_deriv = nullptr;

    // Uniform distribution info
    bool uniform_sites = false;
    T x_start = 0;
    T x_end = 0;
    T x_range = 0;

    // non-uniform sites working data
    // Index to find the interval for a given x:
    // for i in [0, index_size-1]:
    // interval_index_lo[i] is the index of the largest site smaller than x_start + i*dx_index
    // and interval_index_up[i] is the index of the smallest site larger than x_start + i*dx_index
    da_int index_size = 0;
    std::vector<da_int> interval_index_lo, interval_index_up;

    // Working memory
    // dx(n_sites - 1), dy((n_sites - 1) * dim):
    //          Differences between consecutive x-coordinates (e.g.,  dx[i] = x[i+1] - x[i])
    std::vector<T> dx, dy;

    // State of the interpolation
    bool sites_set = false;
    bool values_set = false;
    bool derivatives_set = false;
    bool model_trained = false;

    // Handle data
    da_errors::da_error_t *err = nullptr;
    da_options::OptionRegistry *opts = nullptr;

  public:
    interpolation_generic(da_errors::da_error_t &err, da_options::OptionRegistry &opts)
        : err(&err), opts(&opts){};
    virtual ~interpolation_generic() {
        if (x_internal != nullptr)
            delete[] x_internal;
        x_sites = nullptr;
    };
    void refresh();
    da_status set_sites_1d(da_int n, const T *x);
    da_status set_sites_uniform_1d(da_int n, T x_start, T x_end);
    da_status set_values_1d(da_int n, da_int dim, const T *y_data, da_int ldy_data,
                            da_int order);

    da_status search_cells(da_int n_eval, const T *x_eval, da_int *cells);
    virtual da_status set_boundary_conditions(da_int dim, da_int left_order,
                                              const T *left_values, da_int right_order,
                                              const T *right_values);
    virtual da_status interpolate();
    virtual da_status evaluate(da_int n_eval, const T *x_eval, T *y_eval,
                               da_int n_orders = 1, const da_int *order = 0);

    virtual da_status get_result(da_result query, da_int *dim, T *result);
    virtual da_status get_result(da_result query, da_int *dim, da_int *result);
};

template <typename T> void interpolation_generic<T>::refresh() { model_trained = false; }

template <typename T>
da_status interpolation_generic<T>::set_sites_1d(da_int n, const T *x) {
    if (sites_set)
        return da_error(this->err, da_status_invalid_input,
                        "Cannot change the interpolation sites.");

    if (n < 2)
        return da_error(this->err, da_status_invalid_input,
                        "Number of interpolation sites must be at least 2.");

    if (n_sites > 0 && n != n_sites)
        return da_error(this->err, da_status_invalid_input,
                        "n_sites was set to " + std::to_string(n_sites) + " and n is " +
                            std::to_string(n) +
                            ". Cannot change the number of interpolation sites when "
                            "sites are already set.");

    n_sites = n;
    x_usr = x;
    x_sites = x_usr;
    index_size = n_sites + 1;

    try {
        dx.resize(n_sites - 1);
        interval_index_lo.resize(index_size);
        interval_index_up.resize(index_size);
    } catch (std::bad_alloc const &) {                     // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }

    // Mark sites as non-uniform
    uniform_sites = false;
    x_start = x_sites[0];
    x_end = x_sites[n_sites - 1];
    x_range = x_end - x_start;

    // Compute sites differences and check that the sites are sorted and do not contain duplicates
    // (all dx elements are strictly positive)
    for (da_int i = 0; i < n_sites - 1; i++) {
        dx[i] = x_sites[i + 1] - x_sites[i];
        if (dx[i] <= 0.)
            return da_error(
                this->err, da_status_invalid_input,
                "All sites must be provided in strictly increasing order. x[" +
                    std::to_string(i) + "] = " + std::to_string(x_sites[i]) + " and x[" +
                    std::to_string(i + 1) + "] = " + std::to_string(x_sites[i + 1]));
    }

    // Build index:
    // interval_index_lo[i] = idx where x_sites[idx] <= x_start + i*dx_index
    // interval_index_up[i] = idx where x_sites[idx] >= x_start + i*dx_index
    T dx_index = x_range / (T)(index_size - 2);
    T x_target = x_start;
    interval_index_lo[0] = 0;
    for (da_int i = 1; i < index_size - 1; i++) {
        x_target += dx_index;
        da_int left = interval_index_lo[i - 1];
        da_int right = n_sites - 1;
        while (right - left > 1) {
            da_int mid = (left + right) / 2;
            if (x_target < x_sites[mid])
                right = mid;
            else
                left = mid;
        }
        interval_index_lo[i] = left;
        interval_index_up[i - 1] = right;
    }
    // Last indices used to handle out of bounds requests
    // If x > x_sites[n_sites-1], return last interval (index n_sites - 2)
    interval_index_lo[index_size - 2] = n_sites - 2;
    interval_index_lo[index_size - 1] = n_sites - 2;
    interval_index_up[index_size - 2] = n_sites - 1;
    interval_index_up[index_size - 1] = n_sites - 1;

    model_trained = false;
    sites_set = true;
    return da_status_success;
}

template <typename T>
da_status interpolation_generic<T>::set_sites_uniform_1d(da_int n, T x_start, T x_end) {
    if (sites_set)
        return da_error(this->err, da_status_invalid_input,
                        "Cannot change the number of interpolation sites when sites are "
                        "already set.");

    if (n < 2)
        return da_error(this->err, da_status_invalid_input,
                        "Number of interpolation sites must be at least 2.");
    if (x_end <= x_start)
        return da_error(this->err, da_status_invalid_input,
                        "x_end must be greater than x_start.");

    if (n_sites > 0 && n != n_sites)
        return da_error(this->err, da_status_invalid_input,
                        "n_sites was set to " + std::to_string(n_sites) + " and n is " +
                            std::to_string(n) +
                            ". Cannot change the number of interpolation sites when "
                            "sites are already set.");

    n_sites = n;
    T dx_val = (x_end - x_start) / (T)(n - 1);
    x_usr = nullptr;
    try {
        x_internal = new T[n_sites];
        dx.resize(n_sites - 1, dx_val);
    } catch (std::bad_alloc const &) {                     // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }
    for (da_int i = 0; i < n; i++) {
        x_internal[i] = x_start + i * dx_val;
    }
    x_sites = x_internal;

    // Mark sites as uniform and store bounds
    uniform_sites = true;
    this->x_start = x_start;
    this->x_end = x_end;

    model_trained = false;
    sites_set = true;

    return da_status_success;
}

// Set interpolation values (y-values or derivatives)
template <typename T>
da_status interpolation_generic<T>::set_values_1d(da_int n, da_int dim, const T *y_data,
                                                  da_int ldy_data, da_int order) {
    // Validate order parameter
    if (order != 0 && order != 1)
        return da_error(this->err, da_status_invalid_input,
                        "Invalid order parameter. Use 0 for function values or 1 for "
                        "first derivatives.");

    if (this->n_sites != 0 && n != this->n_sites)
        return da_error(this->err, da_status_invalid_input,
                        "this->n_sites was set to " + std::to_string(this->n_sites) +
                            " and n is " + std::to_string(n) +
                            ". Cannot change the number of interpolation sites when "
                            "sites are already set.");

    if (y_data == nullptr)
        return da_error(this->err, da_status_invalid_pointer,
                        "y_data array cannot be null.");

    std::string opt_order, check_data_str;
    da_int check_data, sto;
    this->opts->get("storage order", opt_order, sto);
    this->opts->get("check data", check_data_str, check_data);
    this->storage_order = da_order(sto);

    da_status status = ARCH::da_utils::check_2D_array(
        check_data != 0, this->storage_order, this->err, n, dim, y_data, ldy_data,
        "n_sites", "dim", "y_data", "ldy_data", 2, 1);
    if (status != da_status_success)
        return status;

    this->n_sites = n;

    // Determine whether we're setting function values or derivatives
    if (order == 1) {
        this->dim_y = dim;
        this->ldy = ldy_data;
        y_deriv = y_data;
        derivatives_set = true;
    } else {
        try {
            this->dy.resize((this->n_sites - 1) * dim);
        } catch (std::bad_alloc const &) {                     // LCOV_EXCL_LINE
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation error");
        }

        // Copy usr values and pointer
        this->dim_y = dim;
        this->ldy = ldy_data;
        y = y_data;

        if (storage_order == column_major) {
            for (da_int j = 0; j < dim; j++) {
                da_int start_y = j * this->n_sites;
                da_int start_dy = j * (this->n_sites - 1);
#pragma omp simd
                for (da_int i = 0; i < this->n_sites - 1; i++)
                    this->dy[start_dy + i] =
                        this->y[start_y + i + 1] - this->y[start_y + i];
            }
        } else {
            // Always store dy in column major order
            for (da_int i = 0; i < this->n_sites - 1; i++) {
#pragma omp simd
                for (da_int j = 0; j < dim; j++)
                    dy[j * (n_sites - 1) + i] = y[(i + 1) * dim_y + j] - y[i * dim_y + j];
            }
        }
        values_set = true;
    }

    this->model_trained = false;

    return da_status_success;
}

// Search for cells containing query points
template <typename T>
da_status interpolation_generic<T>::search_cells(da_int n_eval, const T *x_eval,
                                                 da_int *cells) {
    if (!this->sites_set)
        return da_error(this->err, da_status_invalid_input,
                        "Interpolation sites must be set before searching cells.");

    if (n_eval < 1)
        return da_error(this->err, da_status_invalid_input,
                        "Number of evaluation points must be at least 1.");

    if (x_eval == nullptr || cells == nullptr)
        return da_error(this->err, da_status_invalid_pointer,
                        "x_eval or cells array cannot be null.");

    if (this->uniform_sites) {
        T dx_u = dx[0];
        for (da_int i = 0; i < n_eval; i++) {
            T x = x_eval[i];
            da_int idx = static_cast<da_int>((x - this->x_start) / dx_u);
            cells[i] = std::clamp(idx, (da_int)0, this->n_sites - 2);
        }
    } else {
        T i_dx_u = (T)(index_size - 2) / x_range;
        // Binary search inside the indexed interval.
        for (da_int i = 0; i < n_eval; i++) {
            T x = x_eval[i];
            da_int idx = std::clamp((da_int)((x - x_start) * i_dx_u), (da_int)0,
                                    (da_int)(index_size - 2));
            da_int left = interval_index_lo[idx];
            da_int right = interval_index_up[idx];
            while (right - left > 1) {
                da_int mid = (left + right) / 2;
                if (x < this->x_sites[mid]) {
                    right = mid;
                } else {
                    left = mid;
                }
            }
            cells[i] = left;
        }
    }

    return da_status_success;
}

template <typename T>
da_status interpolation_generic<T>::set_boundary_conditions(
    [[maybe_unused]] da_int dim, [[maybe_unused]] da_int left_order,
    [[maybe_unused]] const T *left_values, [[maybe_unused]] da_int right_order,
    [[maybe_unused]] const T *right_values) {
    return da_error(
        this->err, da_status_invalid_input,
        "Setting boundary conditions is not available for the generic interpolation "
        "model.");
}

template <typename T> da_status interpolation_generic<T>::interpolate() {
    return da_error(this->err, da_status_invalid_input,
                    "Interpolate method is not available for the generic interpolation "
                    "model.");
}

template <typename T>
da_status interpolation_generic<T>::evaluate([[maybe_unused]] da_int n_eval,
                                             [[maybe_unused]] const T *x_eval,
                                             [[maybe_unused]] T *y_eval,
                                             [[maybe_unused]] da_int n_orders,
                                             [[maybe_unused]] const da_int *order) {
    return da_error(this->err, da_status_invalid_input,
                    "Evaluate method is not available for the generic interpolation "
                    "model.");
}

template <typename T>
da_status interpolation_generic<T>::get_result([[maybe_unused]] da_result query,
                                               [[maybe_unused]] da_int *dim,
                                               [[maybe_unused]] T *result) {
    return da_error(this->err, da_status_invalid_input,
                    "Requested result is not available for this interpolation model.");
}
template <typename T>
da_status interpolation_generic<T>::get_result([[maybe_unused]] da_result query,
                                               [[maybe_unused]] da_int *dim,
                                               [[maybe_unused]] da_int *result) {
    return da_error(this->err, da_status_invalid_input,
                    "Requested result is not available for this interpolation model.");
}
} // namespace da_interpolation
} // namespace ARCH

#endif // INTERPOLATION_GENERIC_HPP