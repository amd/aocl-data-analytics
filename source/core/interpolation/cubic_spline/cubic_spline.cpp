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

#include "cubic_spline.hpp"
#include "cubic_spline_types.hpp"
#include "da_std.hpp"
#include "da_utils.hpp"
#include "lapack_templates.hpp"
#include "macros.h"
#include "options.hpp"
#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>

namespace ARCH {

namespace da_interpolation {

using namespace da_cubic_spline;

template <typename T>
cubic_spline<T>::cubic_spline(da_errors::da_error_t &err,
                              da_options::OptionRegistry &opts)
    : interpolation_generic<T>(err, opts) {}

template <typename T> void cubic_spline<T>::refresh() { this->model_trained = false; }

template <typename T>
da_status cubic_spline<T>::set_boundary_conditions(da_int dim, da_int left_order,
                                                   const T *left_values,
                                                   da_int right_order,
                                                   const T *right_values) {
    // Validate order values
    if (left_order != 1 && left_order != 2)
        return da_error(
            this->err, da_status_invalid_input,
            "left_order must be 1 (first derivative) or 2 (second derivative).");
    if (right_order != 1 && right_order != 2)
        return da_error(
            this->err, da_status_invalid_input,
            "right_order must be 1 (first derivative) or 2 (second derivative).");

    if ((this->values_set && dim != this->dim_y) || dim < 1)
        return da_error(this->err, da_status_invalid_input,
                        "dim must match the dimension of the values.");

    if (left_values == nullptr || right_values == nullptr)
        return da_error(this->err, da_status_invalid_pointer,
                        "left_values and right_values must not be null.");

    da_status status =
        this->opts->set("cubic spline type", "custom", da_options::setby_t::solver);
    if (status != da_status_success)
        return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Internal error in the option setter.");

    this->left_order = left_order;
    this->right_order = right_order;

    // Copy the arrays
    try {
        this->left_bc.assign(left_values, left_values + dim);
        this->right_bc.assign(right_values, right_values + dim);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, "Memory allocation error");
    }

    this->model_trained = false;
    this->custom_bc_set = true;

    return da_status_success;
}

/* Initialize memory and assign common values of the tri-diagonal matrix and rhs
 * diagonal     : [1:n-2] = 2 (h_{i-1} + h_i)
 * upper & lower: [0:n-2] = h_i
 * rhs (in z)   : [1:n-2] = 6(y_{i+1} - y_i)/h_i - 6(y_i - y_{i-1})/h_i
 */
template <typename T> da_status cubic_spline<T>::initialize_system() {

    try {
        diag.resize(this->n_sites);
        diag_up.resize(this->n_sites - 1);
        diag_lo.resize(this->n_sites - 1);
        z.resize(this->n_sites * this->dim_y);
        coeffs.resize(4 * (this->n_sites - 1) * this->dim_y);
    } catch (std::bad_alloc const &) {                     // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }

    if (this->uniform_sites) {
        T h = this->dx[0];
        T h4 = 4 * h;
#pragma omp simd
        for (da_int i = 1; i < this->n_sites - 1; i++)
            diag[i] = h4;

#pragma omp simd
        for (da_int i = 0; i < this->n_sites - 1; i++) {
            diag_up[i] = h;
            diag_lo[i] = h;
        }
    } else {
#pragma omp simd
        for (da_int i = 1; i < this->n_sites - 1; i++)
            diag[i] = 2 * (this->dx[i - 1] + this->dx[i]);

#pragma omp simd
        for (da_int i = 0; i < this->n_sites - 1; i++) {
            diag_up[i] = this->dx[i];
            diag_lo[i] = this->dx[i];
        }
    }

    for (da_int d = 0; d < this->dim_y; d++) {
        da_int start_z = d * this->n_sites;
        da_int start_dy = d * (this->n_sites - 1);
#pragma omp simd
        for (da_int i = 1; i < this->n_sites - 1; i++)
            z[start_z + i] = 6 * this->dy[start_dy + i] / this->dx[i] -
                             6 * this->dy[start_dy + i - 1] / this->dx[i - 1];
    }

    return da_status_success;
}

/* Retrieve the coefficients of the piecewise polynomials.
 * For X in[x_i, x_i + 1]
 * S_i(X) = D_i + C_i(X-x_i) + B_i(X-x_i)^2 + A_i(X-x_i)^3
 */
template <typename T> void cubic_spline<T>::coeffs_from_second_derivatives() {
    for (da_int d = 0; d < this->dim_y; d++) {
        da_int start_z = d * this->n_sites;
        da_int start_dy = d * (this->n_sites - 1);
        da_int start_dim = d * 4 * (this->n_sites - 1);
        for (da_int i = 0; i < this->n_sites - 1; i++) {
            da_int y_idx = this->storage_order == (da_int)row_major
                               ? (i * this->dim_y + d)
                               : (d * this->n_sites + i);
            coeffs[start_dim + 4 * i + 3] =
                (z[start_z + i + 1] - z[start_z + i]) / (6 * this->dx[i]); // A_i
            coeffs[start_dim + 4 * i + 2] = z[start_z + i] / 2;            // B_i
            coeffs[start_dim + 4 * i + 1] = this->dy[start_dy + i] / this->dx[i] -
                                            this->dx[i] * z[start_z + i + 1] / 6 -
                                            this->dx[i] * z[start_z + i] / 3; // C_i
            coeffs[start_dim + 4 * i + 0] = this->y[y_idx];                   // D_i
        }
    }
}

template <typename T> da_status cubic_spline<T>::compute_natural_spline() {

    // natural splines: set the second derivatives at the boudary to 0
    for (da_int i = 0; i < this->dim_y; i++) {
        z[i * this->n_sites] = 0;
        z[i * this->n_sites + this->n_sites - 1] = 0;
    }
    if (this->n_sites <= 2)
        return da_status_success;

    // Solve the system
    da_int info = 1;
    da_int n_eq = this->n_sites - 2;
    da_int n_rhs = this->dim_y;
    da_int ldz = this->n_sites;
    T *z_ptr = z.data();
    T *diag_ptr = diag.data();
    T *diag_lo_ptr = diag_lo.data();
    T *diag_up_ptr = diag_up.data();
    da::gtsv(&n_eq, &n_rhs, &diag_lo_ptr[1], &diag_ptr[1], &diag_up_ptr[1], &z_ptr[1],
             &ldz, &info);
    if (info != 0)
        return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "The tri-diagonal system could not be solved correctly. info = " +
                            std::to_string(info));

    return da_status_success;
}

template <typename T> da_status cubic_spline<T>::compute_clamped_spline() {

    // diagonal extreme values
    diag[0] = 2 * this->dx[0];
    diag[this->n_sites - 1] = 2 * this->dx[this->n_sites - 2];

    // Boundary conditions: 0 1st derivatives at both end
    for (da_int d = 0; d < this->dim_y; d++) {
        da_int start_z = d * this->n_sites;
        da_int start_dy = d * (this->n_sites - 1);
        z[start_z] = 6 * this->dy[start_dy] / this->dx[0];
        z[start_z + this->n_sites - 1] =
            -6 * this->dy[start_dy + this->n_sites - 2] / this->dx[this->n_sites - 2];
    }

    // Solve the system
    da_int info = 1;
    da_int n_eq = this->n_sites;
    da_int n_rhs = this->dim_y;
    da::gtsv(&n_eq, &n_rhs, diag_lo.data(), diag.data(), diag_up.data(), z.data(), &n_eq,
             &info);
    if (info != 0)
        return da_error(this->err, da_status_internal_error,
                        "The tri-diagonal system could not be solved correctly. info = " +
                            std::to_string(info));

    return da_status_success;
}

template <typename T> da_status cubic_spline<T>::compute_custom_bc_spline() {

    if (!this->custom_bc_set)
        return da_error(
            this->err, da_status_out_of_date,
            "Boundary conditions must be set for custom spline interpolation.");

    if (left_order == 1) {
        diag[0] = 2 * this->dx[0];
    } else {
        diag[0] = 1.;
        diag_up[0] = 0.;
    }

    if (right_order == 1) {
        diag[this->n_sites - 1] = 2 * this->dx[this->n_sites - 2];
    } else {
        diag[this->n_sites - 1] = 1.;
        diag_lo[this->n_sites - 2] = 0.;
    }

    for (da_int d = 0; d < this->dim_y; d++) {
        da_int start_z = d * this->n_sites;
        da_int start_dy = d * (this->n_sites - 1);
        if (left_order == 1) {
            z[start_z] = 6 * this->dy[start_dy] / this->dx[0] - 6 * left_bc[d];
        } else {
            z[start_z] = left_bc[d];
        }

        if (right_order == 1) {
            z[start_z + this->n_sites - 1] = -6 * this->dy[start_dy + this->n_sites - 2] /
                                                 this->dx[this->n_sites - 2] +
                                             6 * right_bc[d];
        } else {
            z[start_z + this->n_sites - 1] = right_bc[d];
        }
    }

    // Solve the system
    da_int info = 1;
    da_int n_eq = this->n_sites;
    da_int n_rhs = this->dim_y;
    da::gtsv(&n_eq, &n_rhs, diag_lo.data(), diag.data(), diag_up.data(), z.data(), &n_eq,
             &info);
    if (info != 0)
        return da_error(this->err, da_status_internal_error,
                        "The tri-diagonal system could not be solved correctly. info = " +
                            std::to_string(info));

    return da_status_success;
}

template <typename T> da_status cubic_spline<T>::compute_hermite_spline() {
    /* For Hermite cubic splines, we directly compute coefficients from
     * function values y and derivative values y_deriv at each site.
     * For interval [x_i, x_{i+1}] with h_i = x_{i+1} - x_i:
     *   S_i(X) = D_i + C_i*(X-x_i) + B_i*(X-x_i)^2 + A_i*(X-x_i)^3
     * where:
     *   D_i = y_i
     *   C_i = y'_i
     *   B_i = 3*(y_{i+1} - y_i)/h_i^2 - (2*y'_i + y'_{i+1})/h_i
     *   A_i = -2*(y_{i+1} - y_i)/h_i^3 + (y'_i + y'_{i+1})/h_i^2
     */

    if (!this->derivatives_set)
        return da_error(this->err, da_status_out_of_date,
                        "Derivatives must be set for Hermite spline interpolation.");

    try {
        coeffs.resize(4 * (this->n_sites - 1) * this->dim_y);
    } catch (std::bad_alloc const &) {                     // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }

    if (this->storage_order == column_major || this->dim_y <= 1) {
        for (da_int d = 0; d < this->dim_y; d++) {
            da_int start_dim = d * 4 * (this->n_sites - 1);
            da_int start_y = d * this->n_sites;
            da_int start_dy = d * (this->n_sites - 1);

            for (da_int i = 0; i < this->n_sites - 1; i++) {
                T h_i = this->dx[i];
                T y_i = this->y[start_y + i];
                T yp_i = this->y_deriv[start_y + i];
                T yp_ip1 = this->y_deriv[start_y + i + 1];
                T dy_i = this->dy[start_dy + i];

                T h_i2 = h_i * h_i;
                T h_i3 = h_i2 * h_i;

                coeffs[start_dim + 4 * i + 0] = y_i;  // D_i
                coeffs[start_dim + 4 * i + 1] = yp_i; // C_i
                coeffs[start_dim + 4 * i + 2] =
                    3 * dy_i / h_i2 - (2 * yp_i + yp_ip1) / h_i; // B_i
                coeffs[start_dim + 4 * i + 3] =
                    -2 * dy_i / h_i3 + (yp_i + yp_ip1) / h_i2; // A_i
            }
        }
    } else {
        for (da_int i = 0; i < this->n_sites - 1; i++) {
            da_int start_y = i * this->dim_y;
            da_int start_dim = i * 4;

            T h_i = this->dx[i];
            T h_i2 = h_i * h_i;
            T h_i3 = h_i2 * h_i;

            for (da_int d = 0; d < this->dim_y; d++) {
                da_int d_start = d * (this->n_sites - 1);
                T y_i = this->y[start_y + d];
                T yp_i = this->y_deriv[start_y + d];
                T yp_ip1 = this->y_deriv[start_y + this->dim_y + d];
                T dy_i = this->dy[d_start + i];

                coeffs[start_dim + 4 * d * (this->n_sites - 1) + 0] = y_i;  // D_i
                coeffs[start_dim + 4 * d * (this->n_sites - 1) + 1] = yp_i; // C_i
                coeffs[start_dim + 4 * d * (this->n_sites - 1) + 2] =
                    3 * dy_i / h_i2 - (2 * yp_i + yp_ip1) / h_i; // B_i
                coeffs[start_dim + 4 * d * (this->n_sites - 1) + 3] =
                    -2 * dy_i / h_i3 + (yp_i + yp_ip1) / h_i2; // A_i
            }
        }
    }

    return da_status_success;
}

template <typename T> da_status cubic_spline<T>::interpolate() {

    if (this->model_trained)
        return da_status_success;

    if (!this->sites_set || !this->values_set)
        return da_error(this->err, da_status_out_of_date,
                        "Both the interpolation sites and values need to be set before "
                        "interpolating.");

    // Get the spline type from options
    std::string opt_type;
    if (this->opts->get("cubic spline type", opt_type, spline_type) != da_status_success)
        return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Failed to read cubic spline type option.");

    da_status status;

    // Hermite splines compute coefficients directly, others need system initialization
    if (spline_type != da_cubic_spline::hermite) {
        status = initialize_system();
        if (status != da_status_success)
            // err is already filled
            return status; // LCOV_EXCL_LINE
    }

    switch (spline_type) {
    case da_cubic_spline::natural:
        status = compute_natural_spline();
        if (status != da_status_success)
            return status; // LCOV_EXCL_LINE
        coeffs_from_second_derivatives();
        break;
    case da_cubic_spline::clamped:
        status = compute_clamped_spline();
        if (status != da_status_success)
            return status; // LCOV_EXCL_LINE
        coeffs_from_second_derivatives();
        break;
    case da_cubic_spline::custom:
        status = compute_custom_bc_spline();
        if (status != da_status_success)
            return status; // LCOV_EXCL_LINE
        coeffs_from_second_derivatives();
        break;
    case da_cubic_spline::hermite:
        status = compute_hermite_spline();
        if (status != da_status_success)
            return status;
        break;
    default:
        return da_error(this->err, da_status_invalid_option, // LCOV_EXCL_LINE
                        "Unknown spline type.");
    }

    this->model_trained = true;

    return da_status_success;
}

// Convert a bitset to an integer_sequence of set bit positions
template <unsigned long BITSET, da_int BIT, da_int... ORD>
struct bitset_to_sequence_impl {
    // Recursive case : process the current bit
    // Check if the bit at index BIT is set in BITSET ((BITSET >> BIT) & 1)
    // If so, add BIT to the sequence in ORD... and continue with the next bit (BIT - 1)
    // stop with the base case when all bits have been processed (BIT - 1 == -1)
    using type = typename std::conditional_t<
        (BITSET >> BIT) & 1, bitset_to_sequence_impl<BITSET, BIT - 1, BIT, ORD...>,
        bitset_to_sequence_impl<BITSET, BIT - 1, ORD...>>::type;
};

// Base case, all bits processed: return the integer_sequence in ORD...
template <unsigned long BITSET, da_int... ORD>
struct bitset_to_sequence_impl<BITSET, -1, ORD...> {
    using type = std::integer_sequence<da_int, ORD...>;
};

// bitset_to_sequence: create a sequence of set bit positions from the first 4 bits of BITSET
// e.g., 1011 will return the type integer_sequence<da_int, 0, 1, 3>
template <unsigned long BITSET>
using bitset_to_sequence = typename bitset_to_sequence_impl<BITSET, 3>::type;

template <typename T, da_int order>
static inline void spline_eval(T A, T B, T C, T D, T h, T *val) {
    if constexpr (order == 0) {
        *val = D + h * (C + h * (B + h * A));
    } else if constexpr (order == 1) {
        *val = C + h * (2 * B + h * 3 * A);
    } else if constexpr (order == 2) {
        *val = 2 * B + 6 * A * h;
    } else if constexpr (order == 3) {
        *val = 6 * A;
    } else {
        *val = 0.; // LCOV_EXCL_LINE
    }
}

template <typename T, da_int... ORD, std::size_t... I>
inline void unroll_spline_eval_orders([[maybe_unused]] T A, [[maybe_unused]] T B,
                                      [[maybe_unused]] T C, [[maybe_unused]] T D,
                                      [[maybe_unused]] T h, [[maybe_unused]] T *y_eval,
                                      da_int &order_y_idx_start,
                                      std::integer_sequence<da_int, ORD...>,
                                      std::index_sequence<I...>) {
    ((spline_eval<T, ORD>(A, B, C, D, h, &y_eval[I * order_y_idx_start])), ...);
}

template <typename T, da_int BITSET>
void evaluate_all_orders(da_int n_eval, const T *x_eval, T *y_eval, const T *x_sites,
                         const std::vector<T> &coeffs, da_int &coeff_dim,
                         da_int &start_y_dim, da_int &order_y_idx_start,
                         const std::vector<da_int> &cells) {

    for (da_int i = 0; i < n_eval; i++) {
        da_int cell = cells[i];
        da_int coeff_idx = coeff_dim + 4 * cell;
        T h = x_eval[i] - x_sites[cell];

        T A = coeffs[coeff_idx + 3];
        T B = coeffs[coeff_idx + 2];
        T C = coeffs[coeff_idx + 1];
        T D = coeffs[coeff_idx + 0];

        unroll_spline_eval_orders(
            A, B, C, D, h, &y_eval[start_y_dim + i], order_y_idx_start,
            bitset_to_sequence<BITSET>{},
            std::make_index_sequence<bitset_to_sequence<BITSET>::size()>{});
    }
}

// Function pointer type for the lookup table
template <typename T>
using evaluate_fptr = void (*)(da_int, const T *, T *, const T *, const std::vector<T> &,
                               da_int &, da_int &, da_int &, const std::vector<da_int> &);

// Build the lookup table at compile time
// from a given 4 bit pattern, order_lookup_table[bit_pattern] gives
// a pointer to evaluate_all_orders<bit_pattern>
template <typename T, std::size_t... I>
constexpr std::array<evaluate_fptr<T>, sizeof...(I)>
make_lookup_table(std::index_sequence<I...>) {
    return {{&evaluate_all_orders<T, I>...}};
}

template <typename T>
constexpr auto order_lookup_table = make_lookup_table<T>(std::make_index_sequence<16>{});

template <typename T>
da_status cubic_spline<T>::evaluate(da_int n_eval, const T *x_eval, T *y_eval,
                                    da_int n_orders, const da_int *orders) {

    if (!this->model_trained)
        return da_error(this->err, da_status_out_of_date,
                        "Spline must be computed (interpolate) before evaluation.");

    if (n_eval < 1)
        return da_error(this->err, da_status_invalid_input,
                        "Number of evaluation points must be at least 1.");

    if (x_eval == nullptr || y_eval == nullptr || orders == nullptr)
        return da_error(this->err, da_status_invalid_pointer,
                        "x_eval, y_eval, or orders array cannot be null.");

    if (n_orders < 1)
        return da_error(this->err, da_status_invalid_input,
                        "Number of orders must be at least 1.");

    std::bitset<4> orders_bits = 0b0000;
    // Validate all orders
    for (da_int k = 0; k < n_orders; k++) {
        if (orders[k] < 0 || orders[k] > 3)
            return da_error(
                this->err, da_status_invalid_input,
                "Order must be 0 (function), 1 (1st deriv), 2 (2nd deriv), or 3 "
                "(3rd deriv).");
        orders_bits.set(orders[k]);
    }

    // Find the cells containing the evaluation points
    std::vector<da_int> cells;
    try {
        cells.resize(n_eval);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, "Memory allocation error");
    }

    da_status status = this->search_cells(n_eval, x_eval, cells.data());
    if (status != da_status_success)
        return status;

    // Evaluate the polynomial at each point for all orders
    da_int order_y_idx_start = n_eval * this->dim_y;
    for (da_int d = 0; d < this->dim_y; d++) {
        da_int coeff_dim = d * 4 * (this->n_sites - 1);
        da_int start_y_dim = d * n_eval;
        unsigned long bit_pattern = orders_bits.to_ulong();

        order_lookup_table<T>[bit_pattern](n_eval, x_eval, y_eval, this->x_sites,
                                           this->coeffs, coeff_dim, start_y_dim,
                                           order_y_idx_start, cells);
    }
    return da_status_success;
}

// Get results (floating point)
template <typename T>
da_status cubic_spline<T>::get_result(da_result query, da_int *dim, T *result) {

    if (!this->model_trained)
        return da_error(this->err, da_status_out_of_date,
                        "Handle does not contain data relevant to this query. Was the "
                        "last call to the solver successful?");

    if (dim == nullptr || result == nullptr)
        return da_error(this->err, da_status_invalid_pointer,
                        "Dimension and result arrays cannot be null.");

    da_int rinfo_size = 1;
    da_int c_size = (da_int)coeffs.size();

    switch (query) {
    case da_result::da_rinfo:
        if (*dim < rinfo_size) {
            *dim = rinfo_size;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(rinfo_size) + ".");
        }
        result[0] = (T)this->n_sites;
        break;

    case da_result::da_cubic_spline_coefficients:
        if (*dim < c_size) {
            *dim = c_size;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(c_size) + ".");
        }
        memcpy(result, coeffs.data(), coeffs.size() * sizeof(T));
        break;

    default:
        return da_error(this->err, da_status_unknown_query,
                        "Unknown query for interpolation results.");
    }

    return da_status_success;
}

// Get results (integer)
template <typename T>
da_status cubic_spline<T>::get_result([[maybe_unused]] da_result query,
                                      [[maybe_unused]] da_int *dim,
                                      [[maybe_unused]] da_int *result) {
    return da_error(this->err, da_status_unknown_query,
                    "No integer results available for interpolation.");
}

/* Instantiations */
template class cubic_spline<double>;
template class cubic_spline<float>;

} // namespace da_interpolation

} // namespace ARCH
