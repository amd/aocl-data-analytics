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

#ifndef INTERPOLATION_PY_HPP
#define INTERPOLATION_PY_HPP

#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include "internal_utilities_py.hpp"
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>

/*
 * Python binding for the AOCL-DA cubic spline interpolation functionality.
 */
class cubic_spline : public pyda_handle {
    /* Number of interpolation sites used in the most recent fit. */
    da_int n_sites = 0;
    /* Dimensionality of the values used in the most recent fit. */
    da_int dim_y = 0;

  public:
    cubic_spline(std::string spline_type = "natural", std::string prec = "double",
                 bool check_data = false) {
        da_status status;
        if (prec == "single") {
            precision = da_single;
            da_handle_init<float>(&handle, da_handle_interpolation);
            status =
                da_interpolation_select_model<float>(handle, interpolation_cubic_spline);
        } else {
            da_handle_init<double>(&handle, da_handle_interpolation);
            status =
                da_interpolation_select_model<double>(handle, interpolation_cubic_spline);
        }
        exception_check(status);

        status = da_options_set_string(handle, "cubic spline type", spline_type.c_str());
        exception_check(status);

        if (check_data) {
            status = da_options_set_string(handle, "check data", "yes");
            exception_check(status);
        }
    }
    ~cubic_spline() { da_handle_destroy(&handle); }

    /*
     * Fit the cubic spline.
     *
     * x          : 1D array of interpolation sites (sorted, strictly increasing).
     * y          : values at the sites, shape (n,) or (n, dim).
     * y_deriv    : optional first derivatives at the sites (Hermite splines),
     *              same shape as y.
     * left_order/right_order  : derivative orders for custom boundary conditions
     *                           (1 or 2); ignored when no boundary values supplied.
     * left_values/right_values: optional boundary derivative values, shape (dim,).
     */
    template <typename T>
    void fit(py::array_t<T> x, py::array_t<T> y, std::optional<py::array_t<T>> y_deriv,
             da_int left_order, std::optional<py::array_t<T>> left_values,
             da_int right_order, std::optional<py::array_t<T>> right_values) {
        da_status status;
        da_int n = (da_int)x.shape()[0];
        da_int dim, ldy;

        da_int n_rows, n_cols;
        get_numpy_array_properties(y, n_rows, n_cols, ldy);
        dim = n_cols;
        const char *so = (order == c_contiguous) ? "row-major" : "column-major";
        status = da_options_set_string(handle, "storage order", so);
        exception_check(status);

        status = da_interpolation_set_sites(handle, n, x.data());
        exception_check(status);

        status = da_interpolation_set_values(handle, n, dim, y.data(), ldy, (da_int)0);
        exception_check(status);

        if (y_deriv.has_value()) {
            py::array_t<T> &yd = y_deriv.value();
            da_int ldyd;
            da_int r, c;
            get_numpy_array_properties(yd, r, c, ldyd);
            status =
                da_interpolation_set_values(handle, n, dim, yd.data(), ldyd, (da_int)1);
            exception_check(status);
        }

        if (left_values.has_value() && right_values.has_value()) {
            py::array_t<T> &lv = left_values.value();
            py::array_t<T> &rv = right_values.value();
            status = da_interpolation_set_boundary_conditions(
                handle, dim, left_order, lv.data(), right_order, rv.data());
            exception_check(status);
        }

        status = da_interpolation_interpolate<T>(handle);
        exception_check(status);

        n_sites = n;
        dim_y = dim;
    }

    /*
     * Evaluate the spline (and/or its derivatives) at the query points x_eval.
     *
     * orders contains the derivative orders to evaluate (0=value, 1=1st, 2=2nd,
     * 3=3rd). The returned array has shape (n_orders, dim, n_eval) and is
     * C-contiguous, matching the fixed layout produced by the underlying library.
     */
    template <typename T>
    py::array_t<T> evaluate(py::array_t<T> x_eval, std::vector<da_int> orders) {
        da_int n_eval = (da_int)x_eval.shape()[0];
        da_int n_orders = (da_int)orders.size();

        auto y = py::array_t<T>(
            {(py::ssize_t)n_orders, (py::ssize_t)dim_y, (py::ssize_t)n_eval});

        da_status status = da_interpolation_evaluate(
            handle, n_eval, x_eval.data(), y.mutable_data(), n_orders, orders.data());
        exception_check(status);
        return y;
    }

    /*
     * For each query point return the index of the cell [x_sites[j], x_sites[j+1]]
     * that contains it. Out-of-bounds points are clamped to the first/last cell.
     */
    template <typename T> py::array_t<da_int> search_cells(py::array_t<T> x_eval) {
        da_int n_eval = (da_int)x_eval.shape()[0];
        auto cells = py::array_t<da_int>(n_eval);

        da_status status = da_interpolation_search_cells(handle, n_eval, x_eval.data(),
                                                         cells.mutable_data());
        exception_check(status);
        return cells;
    }

    /*
     * Return the piecewise polynomial coefficients of the fitted spline.
     *
     * The result has shape (dim, n_cells, 4) where n_cells = n_sites - 1, and the
     * trailing axis stores the coefficients in ascending power order
     * [constant, linear, quadratic, cubic] relative to the local coordinate
     * t = x - x_sites[cell].
     */
    py::array get_coefficients() {
        da_int n_cells = n_sites - 1;
        da_int dim = 4 * n_cells * dim_y;

        std::vector<size_t> shape = {(size_t)dim_y, (size_t)n_cells, (size_t)4};

        da_status status;
        if (precision == da_single) {
            auto coeffs = py::array_t<float>(shape);
            status = da_handle_get_result(handle, da_cubic_spline_coefficients, &dim,
                                          coeffs.mutable_data());
            exception_check(status);
            return py::reinterpret_borrow<py::array>(coeffs);
        } else {
            auto coeffs = py::array_t<double>(shape);
            status = da_handle_get_result(handle, da_cubic_spline_coefficients, &dim,
                                          coeffs.mutable_data());
            exception_check(status);
            return py::reinterpret_borrow<py::array>(coeffs);
        }
    }

    da_int get_n_sites() { return n_sites; }
    da_int get_dim() { return dim_y; }
};

#endif // INTERPOLATION_PY_HPP
