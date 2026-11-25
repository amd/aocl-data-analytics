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

#ifndef INTERPOLATION_R_HPP
#define INTERPOLATION_R_HPP

#include <Rcpp.h>
#include <vector>

#include "aoclda.h"
#include "internal_utilities_r.hpp"

using namespace Rcpp;

class CubicSpline : public rda_handle {
    // Number of interpolation sites used in the most recent fit.
    da_int n_sites = 0;
    // Dimensionality of the values used in the most recent fit.
    da_int dim_y = 0;

  public:
    CubicSpline(std::string spline_type = "natural", bool check_data = false) {
        da_status status = da_handle_init_d(&handle, da_handle_interpolation);
        exception_check(status);
        status = da_interpolation_select_model_d(handle, interpolation_cubic_spline);
        exception_check(status);
        status = da_options_set(handle, "cubic spline type", spline_type.c_str());
        exception_check(status);
        if (check_data) {
            status = da_options_set(handle, "check data", "yes");
            exception_check(status);
        }
    }

    /*
     * Fit the cubic spline.
     *
     * x          : vector of interpolation sites (sorted, strictly increasing).
     * y          : values at the sites, shape (n, dim).
     * y_deriv    : optional first derivatives at the sites (Hermite splines),
     *              same shape as y.
     * left_order/right_order  : derivative orders for custom boundary conditions
     *                           (1 or 2); ignored when no boundary values supplied.
     * left_values/right_values: optional boundary derivative values, shape (dim).
     */
    void fit(NumericVector x, NumericMatrix y, Nullable<NumericMatrix> y_deriv,
             da_int left_order, Nullable<NumericVector> left_values, da_int right_order,
             Nullable<NumericVector> right_values) {
        da_status status;
        da_int n = x.size();
        da_int dim = y.ncol();
        da_int ldy = y.nrow();

        status = da_options_set(handle, "storage order", "column-major");
        exception_check(status);

        status = da_interpolation_set_sites_d(handle, n, x.begin());
        exception_check(status);

        status = da_interpolation_set_values_d(handle, n, dim, y.begin(), ldy, (da_int)0);
        exception_check(status);

        if (y_deriv.isNotNull()) {
            NumericMatrix yd(y_deriv.get());
            status = da_interpolation_set_values_d(handle, n, dim, yd.begin(), yd.nrow(),
                                                   (da_int)1);
            exception_check(status);
        }

        if (left_values.isNotNull() && right_values.isNotNull()) {
            NumericVector lv(left_values.get());
            NumericVector rv(right_values.get());
            status = da_interpolation_set_boundary_conditions_d(
                handle, dim, left_order, lv.begin(), right_order, rv.begin());
            exception_check(status);
        }

        status = da_interpolation_interpolate_d(handle);
        exception_check(status);

        n_sites = n;
        dim_y = dim;
    }

    /*
     * Evaluate the spline (and/or its derivatives) at the query points x_eval.
     *
     * The library writes a fixed (n_orders, dim, n_eval) C-contiguous buffer;
     * read back as a column-major R array this is (n_eval, dim, n_orders). The R6
     * layer permutes this to the Python (n_orders, dim, n_eval) axis order.
     */
    NumericVector evaluate(NumericVector x_eval, IntegerVector orders) {
        da_int n_eval = x_eval.size();
        da_int n_orders = orders.size();

        NumericVector y(n_orders * dim_y * n_eval);
        std::vector<da_int> ord(orders.begin(), orders.end());
        da_status status = da_interpolation_evaluate_d(handle, n_eval, x_eval.begin(),
                                                       y.begin(), n_orders, ord.data());
        exception_check(status);

        y.attr("dim") = Dimension(n_eval, dim_y, n_orders);
        return y;
    }

    /*
     * For each query point return the index of the cell [x_sites[j], x_sites[j+1]]
     * that contains it. Out-of-bounds points are clamped to the first/last cell.
     * Indices are 0-based, matching the C and Python interfaces.
     */
    IntegerVector search_cells(NumericVector x_eval) {
        da_int n_eval = x_eval.size();
        IntegerVector cells(n_eval);
        da_status status = da_interpolation_search_cells_d(handle, n_eval, x_eval.begin(),
                                                           cells.begin());
        exception_check(status);
        return cells;
    }

    /*
     * Return the piecewise polynomial coefficients of the fitted spline.
     *
     * The library produces a fixed (dim, n_cells, 4) C-contiguous buffer; read back
     * as a column-major R array this is (4, n_cells, dim), with the coefficients in
     * ascending power order [c0, c1, c2, c3] along the first axis, relative to the
     * local coordinate t = x - x_sites[cell]. The R6 layer permutes this to the
     * Python (dim, n_cells, 4) axis order.
     */
    NumericVector get_coefficients() {
        da_int n_cells = n_sites - 1;
        da_int dim = 4 * n_cells * dim_y;

        NumericVector coeffs(dim);
        da_status status = da_handle_get_result_d(handle, da_cubic_spline_coefficients,
                                                  &dim, coeffs.begin());
        exception_check(status);

        coeffs.attr("dim") = Dimension(4, n_cells, dim_y);
        return coeffs;
    }

    da_int get_n_sites() { return n_sites; }
    da_int get_dim() { return dim_y; }
};

#endif // INTERPOLATION_R_HPP
