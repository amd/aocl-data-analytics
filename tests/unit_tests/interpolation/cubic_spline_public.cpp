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
#include "gtest/gtest.h"
#include <cmath>
#include <iomanip>
#include <list>
#include <numeric>
#include <vector>

#include "../utest_utils.hpp"

TEST(cubic_spline_pub, handle_life_cycle) {
    da_handle handle = nullptr;

    EXPECT_EQ(da_handle_init<double>(&handle, da_handle_interpolation),
              da_status_success);
    EXPECT_EQ(da_interpolation_select_model<double>(handle, interpolation_cubic_spline),
              da_status_success);

    /* Standard life cycle */
    da_int n_sites = 10;
    double x_start = 0;
    double x_end = 9.;
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, n_sites, x_start, x_end),
              da_status_success);
    std::vector<double> y_data(n_sites);
    for (da_int i = 0; i < n_sites; i++) {
        y_data[i] = std::sin(static_cast<double>(i));
    }
    da_int dim_y = 1, ldy = n_sites;
    EXPECT_EQ(da_interpolation_set_values(handle, n_sites, dim_y, y_data.data(), ldy, 0),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "cubic spline type", "natural"),
              da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<double>(handle), da_status_success);
    da_int dim = 4 * (n_sites - 1);
    std::vector<double> coeffs(dim);
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients, &dim,
                                   coeffs.data()),
              da_status_success);

    /* Try to change sites */
    x_start = 1.;
    x_end = 10.;
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, n_sites, x_start, x_end),
              da_status_invalid_input);
    std::vector<double> x_sites(n_sites);
    for (da_int i = 0; i < n_sites; i++) {
        x_sites[i] = static_cast<double>(i * i); // non-uniform spacing
    }
    EXPECT_EQ(da_interpolation_set_sites(handle, n_sites, x_sites.data()),
              da_status_invalid_input);

    // Set values again
    EXPECT_EQ(da_interpolation_set_values(handle, n_sites, dim_y, y_data.data(), ldy, 0),
              da_status_success);
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients,
                                   &dim_y, coeffs.data()),
              da_status_out_of_date);
    EXPECT_EQ(da_interpolation_interpolate<double>(handle), da_status_success);
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients, &dim,
                                   coeffs.data()),
              da_status_success);

    // Try to change values with incorrect n_sites
    n_sites = 5;
    EXPECT_EQ(da_interpolation_set_values(handle, n_sites, dim_y, y_data.data(), ldy, 0),
              da_status_invalid_input);
    da_handle_destroy(&handle);

    // Start a new handle, setting the values first
    EXPECT_EQ(da_handle_init<double>(&handle, da_handle_interpolation),
              da_status_success);
    EXPECT_EQ(da_interpolation_select_model<double>(handle, interpolation_cubic_spline),
              da_status_success);
    n_sites = 10;
    for (da_int i = 0; i < n_sites; i++) {
        y_data[i] = std::sin(static_cast<double>(i));
    }
    EXPECT_EQ(da_interpolation_set_values(handle, n_sites, dim_y, y_data.data(), ldy, 0),
              da_status_success);
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, n_sites, x_start, x_end),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "cubic spline type", "natural"),
              da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<double>(handle), da_status_success);
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients, &dim,
                                   coeffs.data()),
              da_status_success);

    da_handle_destroy(&handle);
}

template <typename T> class cubic_spline_public : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(cubic_spline_public, FloatTypes);

TYPED_TEST(cubic_spline_public, search_cells_uniform) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_interpolation),
              da_status_success);
    EXPECT_EQ(
        da_interpolation_select_model<TypeParam>(handle, interpolation_cubic_spline),
        da_status_success);

    // Uniform sites
    da_int n_sites = 11;
    TypeParam x_start = 0.0;
    TypeParam x_end = 10.0;
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, n_sites, x_start, x_end),
              da_status_success);

    // Test search_cells with various evaluation points
    da_int n_eval = 8;
    std::vector<TypeParam> x_eval = {0.5, 1.5, 2.3, 5.0, 7.8, 9.9, -1.0, 12.0};
    std::vector<da_int> cells(n_eval);
    std::vector<da_int> expected_cells = {0, 1, 2, 5, 7, 9, 0, 9};

    EXPECT_EQ(da_interpolation_search_cells(handle, n_eval, x_eval.data(), cells.data()),
              da_status_success);
    EXPECT_ARR_EQ(n_eval, cells.data(), expected_cells.data(), 1, 1, 0, 0);

    da_handle_destroy(&handle);
}

TYPED_TEST(cubic_spline_public, search_cells_custom) {
    da_handle handle = nullptr;

    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_interpolation),
              da_status_success);
    EXPECT_EQ(
        da_interpolation_select_model<TypeParam>(handle, interpolation_cubic_spline),
        da_status_success);
    da_int n_sites = 6;
    std::vector<TypeParam> x_sites = {0.0, 1.0, 2.0, 5.0, 8.0, 10.0};
    EXPECT_EQ(da_interpolation_set_sites(handle, n_sites, x_sites.data()),
              da_status_success);

    // Test search_cells with various evaluation points
    da_int n_eval = 11;
    std::vector<TypeParam> x_eval = {2.0, 0.5, 1.5,  2.5,  4.0, 5.5,
                                     7.0, 9.0, -1.0, 11.0, 6.5};
    std::vector<da_int> cells(n_eval);
    std::vector<da_int> expected_cells = {2, 0, 1, 2, 2, 3, 3, 4, 0, 4, 3};

    EXPECT_EQ(da_interpolation_search_cells(handle, n_eval, x_eval.data(), cells.data()),
              da_status_success);

    EXPECT_ARR_EQ(n_eval, cells.data(), expected_cells.data(), 1, 1, 0, 0);

    da_handle_destroy(&handle);

    // Test again with very unbalanced sites
    handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_interpolation),
              da_status_success);
    EXPECT_EQ(
        da_interpolation_select_model<TypeParam>(handle, interpolation_cubic_spline),
        da_status_success);
    x_sites = {0.0, 0.2, 0.4, 2.0, 3.0, 10.0};
    EXPECT_EQ(da_interpolation_set_sites(handle, n_sites, x_sites.data()),
              da_status_success);
    x_eval = {-1.0, 0.1, 0.15, 0.3, 1.99, 2.5, 3.5, 4.0, 9.0, 10.0, 11.0};
    expected_cells = {0, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4};
    EXPECT_EQ(da_interpolation_search_cells(handle, n_eval, x_eval.data(), cells.data()),
              da_status_success);
    EXPECT_ARR_EQ(n_eval, cells.data(), expected_cells.data(), 1, 1, 0, 0);

    da_handle_destroy(&handle);
}

TYPED_TEST(cubic_spline_public, set_sites_errors) {
    da_handle handle = nullptr;

    // Test with uninitialized handle
    da_int n_sites = 5;
    std::vector<TypeParam> x_sites(n_sites, 1.0);
    EXPECT_EQ(da_interpolation_set_sites(handle, n_sites, x_sites.data()),
              da_status_handle_not_initialized);

    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_interpolation),
              da_status_success);
    EXPECT_EQ(
        da_interpolation_select_model<TypeParam>(handle, interpolation_cubic_spline),
        da_status_success);

    // Test with null pointer
    TypeParam *ptr = nullptr;
    EXPECT_EQ(da_interpolation_set_sites(handle, n_sites, ptr),
              da_status_invalid_pointer);

    // Test with invalid n_sites (< 2)
    EXPECT_EQ(da_interpolation_set_sites(handle, 1, x_sites.data()),
              da_status_invalid_input);
    EXPECT_EQ(da_interpolation_set_sites(handle, 0, x_sites.data()),
              da_status_invalid_input);
    EXPECT_EQ(da_interpolation_set_sites(handle, -1, x_sites.data()),
              da_status_invalid_input);

    // Test with non increasing x
    EXPECT_EQ(da_interpolation_set_sites(handle, n_sites, x_sites.data()),
              da_status_invalid_input);

    da_handle_destroy(&handle);
}

TYPED_TEST(cubic_spline_public, set_sites_uniform_errors) {
    da_handle handle = nullptr;

    // Test with uninitialized handle
    da_int n_sites = 5;
    TypeParam x_start = 0.0;
    TypeParam x_end = 10.0;
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, n_sites, x_start, x_end),
              da_status_handle_not_initialized);

    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_interpolation),
              da_status_success);
    EXPECT_EQ(
        da_interpolation_select_model<TypeParam>(handle, interpolation_cubic_spline),
        da_status_success);

    // Test with invalid n_sites (< 2)
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, 1, x_start, x_end),
              da_status_invalid_input);
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, 0, x_start, x_end),
              da_status_invalid_input);
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, -1, x_start, x_end),
              da_status_invalid_input);

    // Test with x_end <= x_start
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, n_sites, x_end, x_start),
              da_status_invalid_input);
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, n_sites, x_start, x_start),
              da_status_invalid_input);

    // Valid call
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, n_sites, x_start, x_end),
              da_status_success);

    da_handle_destroy(&handle);
}

TYPED_TEST(cubic_spline_public, search_cells_errors) {
    da_handle handle = nullptr;

    // Test with uninitialized handle
    da_int n_eval = 5;
    std::vector<TypeParam> x_eval(n_eval, 1.0);
    std::vector<da_int> cells(n_eval);
    EXPECT_EQ(da_interpolation_search_cells(handle, n_eval, x_eval.data(), cells.data()),
              da_status_handle_not_initialized);

    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_interpolation),
              da_status_success);
    EXPECT_EQ(
        da_interpolation_select_model<TypeParam>(handle, interpolation_cubic_spline),
        da_status_success);
    // Sites not set
    EXPECT_EQ(da_interpolation_search_cells(handle, n_eval, x_eval.data(), cells.data()),
              da_status_invalid_input);

    // Set sites
    EXPECT_EQ(
        da_interpolation_set_sites_uniform(handle, 10, (TypeParam)0.0, (TypeParam)10.0),
        da_status_success);

    // Test with null pointers
    TypeParam *ptr = nullptr;
    EXPECT_EQ(da_interpolation_search_cells(handle, n_eval, ptr, cells.data()),
              da_status_invalid_pointer);
    EXPECT_EQ(da_interpolation_search_cells(handle, n_eval, x_eval.data(), nullptr),
              da_status_invalid_pointer);

    // Test with invalid n_eval
    EXPECT_EQ(da_interpolation_search_cells(handle, 0, x_eval.data(), cells.data()),
              da_status_invalid_input);

    da_handle_destroy(&handle);
}

TYPED_TEST(cubic_spline_public, set_boundary_conditions_errors) {
    using T = TypeParam;
    da_handle h = nullptr;

    EXPECT_EQ(da_handle_init<T>(&h, da_handle_interpolation), da_status_success);
    EXPECT_EQ(da_interpolation_select_model<T>(h, interpolation_cubic_spline),
              da_status_success);

    da_int n_sites = 5;
    T x_start = 0.0;
    T x_end = 4.0;
    EXPECT_EQ(da_interpolation_set_sites_uniform(h, n_sites, x_start, x_end),
              da_status_success);

    std::vector<T> y_data(n_sites);
    for (da_int i = 0; i < n_sites; i++) {
        y_data[i] = static_cast<T>(i);
    }
    da_int dim_y = 1, ldy = n_sites;
    EXPECT_EQ(da_interpolation_set_values(h, n_sites, dim_y, y_data.data(), ldy, 0),
              da_status_success);

    // Test invalid order
    T left_bc[] = {(T)1.0};
    T right_bc[] = {(T)1.0};
    EXPECT_EQ(da_interpolation_set_boundary_conditions(h, 1, 0, left_bc, 1, right_bc),
              da_status_invalid_input);
    EXPECT_EQ(da_interpolation_set_boundary_conditions(h, 1, 3, left_bc, 1, right_bc),
              da_status_invalid_input);
    EXPECT_EQ(da_interpolation_set_boundary_conditions(h, 1, 1, left_bc, 0, right_bc),
              da_status_invalid_input);
    EXPECT_EQ(da_interpolation_set_boundary_conditions(h, 1, 1, left_bc, 3, right_bc),
              da_status_invalid_input);

    // Test null pointers
    EXPECT_EQ(da_interpolation_set_boundary_conditions(h, 1, 1, nullptr, 1, right_bc),
              da_status_invalid_pointer);
    EXPECT_EQ(da_interpolation_set_boundary_conditions(h, 1, 1, left_bc, 1, nullptr),
              da_status_invalid_pointer);

    // Test invalid dim
    EXPECT_EQ(da_interpolation_set_boundary_conditions(h, 2, 1, left_bc, 1, right_bc),
              da_status_invalid_input);

    da_handle_destroy(&h);
}

TYPED_TEST(cubic_spline_public, interpolate_error) {

    using T = TypeParam;
    da_handle h = nullptr;

    EXPECT_EQ(da_handle_init<T>(&h, da_handle_interpolation), da_status_success);
    EXPECT_EQ(da_interpolation_select_model<T>(h, interpolation_cubic_spline),
              da_status_success);

    // sites not set
    EXPECT_EQ(da_interpolation_interpolate<T>(h), da_status_out_of_date);
    da_int n_sites = 5;
    T x_start = 0.0;
    T x_end = 4.0;
    EXPECT_EQ(da_interpolation_set_sites_uniform(h, n_sites, x_start, x_end),
              da_status_success);
    // values not set
    EXPECT_EQ(da_interpolation_interpolate<T>(h), da_status_out_of_date);
    std::vector<T> y_data(n_sites);
    for (da_int i = 0; i < n_sites; i++) {
        y_data[i] = static_cast<T>(i);
    }
    da_int dim_y = 1, ldy = n_sites;
    EXPECT_EQ(da_interpolation_set_values(h, n_sites, dim_y, y_data.data(), ldy, 0),
              da_status_success);

    // set custom without defining boundary conditions
    EXPECT_EQ(da_options_set_string(h, "cubic spline type", "custom"), da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<T>(h), da_status_out_of_date);

    // set valid boundary conditions
    T left_bc[] = {(T)0.0};
    T right_bc[] = {(T)0.0};
    EXPECT_EQ(da_interpolation_set_boundary_conditions(h, 1, 1, left_bc, 1, right_bc),
              da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<T>(h), da_status_success);

    // set splines to hermite without setting derivatives
    EXPECT_EQ(da_options_set_string(h, "cubic spline type", "hermite"),
              da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<T>(h), da_status_out_of_date);

    da_handle_destroy(&h);
}

TYPED_TEST(cubic_spline_public, set_values_error) {
    using T = TypeParam;
    da_handle h = nullptr;

    EXPECT_EQ(da_handle_init<T>(&h, da_handle_interpolation), da_status_success);
    EXPECT_EQ(da_interpolation_select_model<T>(h, interpolation_cubic_spline),
              da_status_success);

    da_int n_sites = 5;
    std::vector<T> y_data(n_sites, (T)1.0);

    // set wrong order
    T *NU = nullptr;
    da_int dim = 1;
    da_int ldy = n_sites;
    EXPECT_EQ(da_interpolation_set_values(h, n_sites, dim, NU, ldy, 0),
              da_status_invalid_pointer);
    EXPECT_EQ(da_interpolation_set_values(h, n_sites, dim, y_data.data(), ldy, -1),
              da_status_invalid_input);
    EXPECT_EQ(da_interpolation_set_values(h, n_sites, dim, y_data.data(), ldy, 2),
              da_status_invalid_input);
    EXPECT_EQ(da_interpolation_set_values(h, 1, dim, y_data.data(), ldy, 0),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_interpolation_set_values(h, n_sites, 0, y_data.data(), ldy, 0),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_interpolation_set_values(h, n_sites, dim, y_data.data(), n_sites - 1, 0),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_interpolation_set_values(h, n_sites, dim, y_data.data(), n_sites, 0),
              da_status_success);

    // set values before sites
    EXPECT_EQ(da_interpolation_set_values(h, n_sites, dim, y_data.data(), ldy, 0),
              da_status_success);
    // set values again with different n_sites
    EXPECT_EQ(da_interpolation_set_values(h, n_sites + 1, dim, y_data.data(), ldy, 0),
              da_status_invalid_input);

    da_handle_destroy(&h);
}

TYPED_TEST(cubic_spline_public, evaluate_error) {
    using T = TypeParam;
    da_handle h = nullptr;

    EXPECT_EQ(da_handle_init<T>(&h, da_handle_interpolation), da_status_success);
    EXPECT_EQ(da_interpolation_select_model<T>(h, interpolation_cubic_spline),
              da_status_success);

    // Model not trained
    da_int n_eval = 5;
    std::vector<T> x_eval(n_eval);
    da_int n_orders = 4;
    da_int orders[] = {0, 1, 2, 3};
    std::vector<T> y_eval(n_orders * n_eval);
    std::iota(x_eval.begin(), x_eval.end(), (T)0.0);
    EXPECT_EQ(da_interpolation_evaluate(h, n_eval, x_eval.data(), y_eval.data(), n_orders,
                                        orders),
              da_status_out_of_date);
    // Set sites and values but call evaluate before interpolate
    da_int n_sites = 5;
    T x_start = 0.0;
    T x_end = 4.0;
    EXPECT_EQ(da_interpolation_set_sites_uniform(h, n_sites, x_start, x_end),
              da_status_success);
    std::vector<T> y_data(n_sites);
    for (da_int i = 0; i < n_sites; i++) {
        y_data[i] = static_cast<T>(i);
    }
    da_int dim_y = 1, ldy = n_sites;
    EXPECT_EQ(da_interpolation_set_values(h, n_sites, dim_y, y_data.data(), ldy, 0),
              da_status_success);
    EXPECT_EQ(da_interpolation_evaluate(h, n_eval, x_eval.data(), y_eval.data(), n_orders,
                                        orders),
              da_status_out_of_date);

    // Interpolate and check that evaluate passes
    EXPECT_EQ(da_interpolation_interpolate<T>(h), da_status_success);
    EXPECT_EQ(da_interpolation_evaluate(h, n_eval, x_eval.data(), y_eval.data(), n_orders,
                                        orders),
              da_status_success);

    // Test with invalid inputs
    EXPECT_EQ(
        da_interpolation_evaluate(h, 0, x_eval.data(), y_eval.data(), n_orders, orders),
        da_status_invalid_input);
    EXPECT_EQ(
        da_interpolation_evaluate(h, n_eval, nullptr, y_eval.data(), n_orders, orders),
        da_status_invalid_pointer);
    EXPECT_EQ(
        da_interpolation_evaluate(h, n_eval, x_eval.data(), nullptr, n_orders, orders),
        da_status_invalid_pointer);
    EXPECT_EQ(da_interpolation_evaluate(h, n_eval, x_eval.data(), y_eval.data(), n_orders,
                                        nullptr),
              da_status_invalid_pointer);
    EXPECT_EQ(
        da_interpolation_evaluate(h, n_eval, x_eval.data(), y_eval.data(), 0, orders),
        da_status_invalid_input);

    da_handle_destroy(&h);
}

template <typename T>
int read_coeffs(std::string filename, T **coeffs, da_int &n_row, da_int &n_col) {

    da_datastore csv_store = nullptr;
    int exit_status = 1;
    da_status status = da_datastore_init(&csv_store);
    EXPECT_EQ(status, da_status_success);
    FILE *file = fopen(filename.c_str(), "r");
    if (file) {
        fclose(file);

        if (status == da_status_success) {
            // read the expected coefficients
            status =
                da_read_csv(csv_store, filename.c_str(), coeffs, &n_row, &n_col, nullptr);
            EXPECT_EQ(status, da_status_success);
            if (status == da_status_success)
                exit_status = 0;
        }
    }

    da_datastore_destroy(&csv_store);
    if (status != da_status_success)
        exit_status = 1;

    return exit_status;
}

TYPED_TEST(cubic_spline_public, interpolate_few_sites) {
    using T = TypeParam;

    da_numeric::tolerance<T> tol_struct;
    T tol = tol_struct.tol(1.0, 1.0);

    da_handle handle = nullptr;

    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_interpolation), da_status_success);
    EXPECT_EQ(da_interpolation_select_model<T>(handle, interpolation_cubic_spline),
              da_status_success);

    da_int n_sites = 3;
    T x_start = 0;
    T x_end = 2.;
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, n_sites, x_start, x_end),
              da_status_success);

    std::vector<T> y_data(n_sites);
    for (da_int i = 0; i < n_sites; i++) {
        y_data[i] = std::sin(static_cast<T>(i));
    }
    da_int dim_y = 1, ldy = n_sites;
    EXPECT_EQ(da_interpolation_set_values(handle, n_sites, dim_y, y_data.data(), ldy, 0),
              da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<T>(handle), da_status_success);
    da_int dim = 4 * (n_sites - 1);
    std::vector<T> coeffs(dim);
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients, &dim,
                                   coeffs.data()),
              da_status_success);
    std::vector<T> expected_coeffs = {
        (T)0.000000000000000000e+00,  (T)1.034882120505424163e+00,
        (T)4.440892098500626162e-16,  (T)-1.934111356975281026e-01,
        (T)8.414709848078965049e-01,  (T)4.546487134128407992e-01,
        (T)-5.802334070925834197e-01, (T)1.934111356975278251e-01};

    EXPECT_ARR_NEAR(dim, coeffs.data(), expected_coeffs.data(), tol);
    da_handle_destroy(&handle);

    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_interpolation), da_status_success);
    EXPECT_EQ(da_interpolation_select_model<T>(handle, interpolation_cubic_spline),
              da_status_success);

    // Test with 2 sites
    n_sites = 2;
    x_start = 0;
    x_end = 1.;
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, n_sites, x_start, x_end),
              da_status_success);

    y_data.resize(n_sites);
    for (da_int i = 0; i < n_sites; i++) {
        y_data[i] = std::sin(static_cast<T>(i));
    }
    ldy = n_sites;
    EXPECT_EQ(da_interpolation_set_values(handle, n_sites, dim_y, y_data.data(), ldy, 0),
              da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<T>(handle), da_status_success);
    dim = 4 * (n_sites - 1);
    coeffs.resize(dim);
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients, &dim,
                                   coeffs.data()),
              da_status_success);
    expected_coeffs = {(T)0.000000000000000000e+00, (T)8.414709848078965049e-01,
                       (T)0.000000000000000000e+00, (T)0.000000000000000000e+00};
    EXPECT_ARR_NEAR(dim, coeffs.data(), expected_coeffs.data(), tol);

    // Test with clamped zero spline type (zero derivatives at boundaries)
    EXPECT_EQ(da_options_set_string(handle, "cubic spline type", "clamped zero"),
              da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<T>(handle), da_status_success);
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients, &dim,
                                   coeffs.data()),
              da_status_success);
    expected_coeffs = {(T)0.000000000000000000e+00, (T)0.000000000000000000e+00,
                       (T)2.524412954423689293e+00, (T)-1.682941969615793010e+00};
    EXPECT_ARR_NEAR(dim, coeffs.data(), expected_coeffs.data(), tol);

    // Test with custom boundary conditions (first derivatives = 1.0 at both ends)
    T left_bc[] = {(T)1.0};
    T right_bc[] = {(T)1.0};
    EXPECT_EQ(
        da_interpolation_set_boundary_conditions(handle, 1, 1, left_bc, 1, right_bc),
        da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<T>(handle), da_status_success);
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients, &dim,
                                   coeffs.data()),
              da_status_success);
    expected_coeffs = {(T)0.000000000000000000e+00, (T)1.000000000000000000e+00,
                       (T)-4.755870455763104854e-01, (T)3.170580303842069902e-01};
    EXPECT_ARR_NEAR(dim, coeffs.data(), expected_coeffs.data(), tol);
    da_handle_destroy(&handle);
}

TYPED_TEST(cubic_spline_public, interpolate_natural) {
    using T = TypeParam;
    da_numeric::tolerance<T> tol_struct;
    T tol = tol_struct.tol(10.0, 1.0);

    da_handle handle = nullptr;
    da_int n_row, n_col;

    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_interpolation), da_status_success);
    EXPECT_EQ(da_interpolation_select_model<T>(handle, interpolation_cubic_spline),
              da_status_success);

    /****
     * Sin
     * uniform
     * natural
     */
    da_int n_sites = 10;
    T x_start = 0;
    T x_end = 9.;
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, n_sites, x_start, x_end),
              da_status_success);
    std::vector<T> y_data(n_sites);
    for (da_int i = 0; i < n_sites; i++) {
        y_data[i] = std::sin(static_cast<T>(i));
    }
    da_int dim_y = 1, ldy = n_sites;
    EXPECT_EQ(da_interpolation_set_values(handle, n_sites, dim_y, y_data.data(), ldy, 0),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "cubic spline type", "natural"),
              da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<T>(handle), da_status_success);
    da_int dim = 4 * (n_sites - 1);
    std::vector<T> coeffs(dim);
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients, &dim,
                                   coeffs.data()),
              da_status_success);
    // Read expected coefficients from CSV file (generate with scipy cubic splines)
    std::string filename =
        std::string(DATA_DIR) + "/interpolation/coeffs_sin_uni_natural.csv";
    T *expected_coeffs = nullptr;
    if (read_coeffs(filename, &expected_coeffs, n_row, n_col) != 0) {
        da_handle_destroy(&handle);
        FAIL() << "Failed reading the file " << filename;
    }
    // Compare computed coefficients with expected values
    EXPECT_ARR_NEAR(dim, coeffs.data(), expected_coeffs, tol);

    /*****************************************************************
     * Same but natural is defined through custom boundary conditions */
    EXPECT_EQ(da_options_set_string(handle, "cubic spline type", "custom"),
              da_status_success);
    T left_bc[] = {(T)0.0};
    T right_bc[] = {(T)0.0};
    EXPECT_EQ(
        da_interpolation_set_boundary_conditions(handle, 1, 2, left_bc, 2, right_bc),
        da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<T>(handle), da_status_success);
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients, &dim,
                                   coeffs.data()),
              da_status_success);
    EXPECT_ARR_NEAR(dim, coeffs.data(), expected_coeffs, tol);

    /*****************************************************************
     * Same but interpolationpoints are defined manually */
    da_handle_destroy(&handle);
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_interpolation), da_status_success);
    EXPECT_EQ(da_interpolation_select_model<T>(handle, interpolation_cubic_spline),
              da_status_success);
    n_sites = 10;
    std::vector<T> x = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    ldy = n_sites;
    EXPECT_EQ(da_interpolation_set_sites(handle, n_sites, x.data()), da_status_success);
    EXPECT_EQ(da_interpolation_set_values(handle, n_sites, dim_y, y_data.data(), ldy, 0),
              da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<T>(handle), da_status_success);
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients, &dim,
                                   coeffs.data()),
              da_status_success);
    EXPECT_ARR_NEAR(dim, coeffs.data(), expected_coeffs, tol);

    da_handle_destroy(&handle);
    if (expected_coeffs != nullptr) {
        free(expected_coeffs);
        expected_coeffs = nullptr;
    }
    /****
     * J0_sqrt
     * uniform
     * natural
     */
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_interpolation),
              da_status_success);
    EXPECT_EQ(da_interpolation_select_model<T>(handle, interpolation_cubic_spline),
              da_status_success);
    n_sites = 11;
    x_start = 0;
    x_end = 50.;
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, n_sites, x_start, x_end),
              da_status_success);
    y_data.resize(n_sites);
    for (da_int i = 0; i < n_sites; i++) {
        y_data[i] = std::cyl_bessel_j(0, std::sqrt(static_cast<TypeParam>(5 * i)));
    }
    ldy = n_sites;
    EXPECT_EQ(da_interpolation_set_values(handle, n_sites, dim_y, y_data.data(), ldy, 0),
              da_status_success);

    EXPECT_EQ(da_options_set_string(handle, "cubic spline type", "natural"),
              da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<T>(handle), da_status_success);
    dim = 4 * (n_sites - 1);
    coeffs.resize(dim);
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients, &dim,
                                   coeffs.data()),
              da_status_success);
    // Read expected coefficients from CSV file (generate with scipy cubic splines)
    filename = std::string(DATA_DIR) + "/interpolation/coeffs_J0_uni_natural.csv";
    expected_coeffs = nullptr;
    if (read_coeffs(filename, &expected_coeffs, n_row, n_col) != 0) {
        da_handle_destroy(&handle);
        FAIL() << "Failed reading the file " << filename;
    }
    // Compare computed coefficients with expected values
    EXPECT_ARR_NEAR(dim, coeffs.data(), expected_coeffs, tol);
    da_handle_destroy(&handle);
    if (expected_coeffs != nullptr)
        free(expected_coeffs);
}

TYPED_TEST(cubic_spline_public, interpolate_clamped) {
    using T = TypeParam;

    da_numeric::tolerance<T> tol_struct;
    T tol = tol_struct.tol(10.0, 1.0);

    da_handle handle = nullptr;
    da_int n_row, n_col;

    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_interpolation), da_status_success);
    EXPECT_EQ(da_interpolation_select_model<T>(handle, interpolation_cubic_spline),
              da_status_success);
    da_int n_sites = 11;
    T x_start = 0.;
    T x_end = 50.;
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, n_sites, x_start, x_end),
              da_status_success);
    // y = J_0(sqrt(x))
    std::vector<T> y_data(n_sites);
    for (da_int i = 0; i < n_sites; i++) {
        y_data[i] = std::cyl_bessel_j(0, std::sqrt(static_cast<T>(5 * i)));
    }
    da_int dim_y = 1, ldy = n_sites;
    EXPECT_EQ(da_interpolation_set_values(handle, n_sites, dim_y, y_data.data(), ldy, 0),
              da_status_success);
    // known first derivative values
    T left_bc[] = {(T)-0.25};
    T right_bc[] = {(T)-0.00117217};
    EXPECT_EQ(
        da_interpolation_set_boundary_conditions(handle, 1, 1, left_bc, 1, right_bc),
        da_status_success);
    char opt_val[32];
    da_int iopt = 32;
    EXPECT_EQ(da_options_get(handle, "cubic spline type", opt_val, &iopt),
              da_status_success);
    // Check that the option was correctly set by the solver
    EXPECT_STREQ(opt_val, "custom");
    EXPECT_EQ(da_interpolation_interpolate<T>(handle), da_status_success);
    da_int dim = 4 * (n_sites - 1);
    std::vector<T> coeffs(dim);
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients, &dim,
                                   coeffs.data()),
              da_status_success);
    T *expected_coeffs = nullptr;
    std::string filename =
        std::string(DATA_DIR) + "/interpolation/coeffs_J0_uni_custom_clamped.csv";
    if (read_coeffs(filename, &expected_coeffs, n_row, n_col) != 0)
        FAIL() << "Failed reading the file " + filename;
    EXPECT_ARR_NEAR(dim, coeffs.data(), expected_coeffs, tol);

    // Use of clamped zero option (0. first derivatives on the boundary)
    if (expected_coeffs != nullptr) {
        free(expected_coeffs);
        expected_coeffs = nullptr;
    }
    EXPECT_EQ(da_options_set_string(handle, "cubic spline type", "clamped zero"),
              da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<T>(handle), da_status_success);
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients, &dim,
                                   coeffs.data()),
              da_status_success);
    filename = std::string(DATA_DIR) + "/interpolation/coeffs_J0_uni_clamped.csv";
    if (read_coeffs(filename, &expected_coeffs, n_row, n_col) != 0)
        FAIL() << "Failed reading the file " + filename;
    EXPECT_ARR_NEAR(dim, coeffs.data(), expected_coeffs, tol);

    da_handle_destroy(&handle);
    if (expected_coeffs != nullptr)
        free(expected_coeffs);
}

TYPED_TEST(cubic_spline_public, interpolate_custom) {
    using T = TypeParam;
    da_numeric::tolerance<T> tol_struct;
    T tol = tol_struct.tol(10.0, 1.0);

    da_handle handle = nullptr;

    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_interpolation), da_status_success);
    EXPECT_EQ(da_interpolation_select_model<T>(handle, interpolation_cubic_spline),
              da_status_success);

    // Custom clamped
    da_int n_sites, n_col, n_row;
    std::string filename = std::string(DATA_DIR) + "/interpolation/coord_exp.csv";
    T *coord = nullptr;
    if (read_coeffs(filename, &coord, n_sites, n_col) != 0)
        FAIL() << "Failed reading the file " + filename;
    EXPECT_EQ(da_interpolation_set_sites(handle, n_sites, coord), da_status_success);
    EXPECT_EQ(
        da_interpolation_set_values(handle, n_sites, 1, &coord[n_sites], n_sites, 0),
        da_status_success);
    T left_bc[] = {(T)-1.0};
    T right_bc[] = {(T)2.0};
    EXPECT_EQ(
        da_interpolation_set_boundary_conditions(handle, 1, 1, left_bc, 1, right_bc),
        da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<T>(handle), da_status_success);
    da_int dim = 4 * (n_sites - 1);
    std::vector<T> coeffs(dim);
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients, &dim,
                                   coeffs.data()),
              da_status_success);
    T *expected_coeffs = nullptr;
    filename = std::string(DATA_DIR) + "/interpolation/coeffs_exp_nonuni_ff.csv";
    if (read_coeffs(filename, &expected_coeffs, n_col, n_row) != 0)
        FAIL() << "Failed reading the file " + filename;
    EXPECT_ARR_NEAR(dim, coeffs.data(), expected_coeffs, tol);

    // set first/second derivatives boundary conditions
    if (expected_coeffs != nullptr) {
        free(expected_coeffs);
        expected_coeffs = nullptr;
    }
    T left_bc2[] = {(T)-1.0};
    T right_bc2[] = {(T)1.0};
    EXPECT_EQ(
        da_interpolation_set_boundary_conditions(handle, 1, 1, left_bc2, 2, right_bc2),
        da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<T>(handle), da_status_success);
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients, &dim,
                                   coeffs.data()),
              da_status_success);
    filename = std::string(DATA_DIR) + "/interpolation/coeffs_exp_nonuni_fs.csv";
    if (read_coeffs(filename, &expected_coeffs, n_col, n_row) != 0)
        FAIL() << "Failed reading the file " + filename;
    EXPECT_ARR_NEAR(dim, coeffs.data(), expected_coeffs, tol);

    // set second/first derivatives boundary conditions
    if (expected_coeffs != nullptr) {
        free(expected_coeffs);
        expected_coeffs = nullptr;
    }
    EXPECT_EQ(
        da_interpolation_set_boundary_conditions(handle, 1, 2, left_bc2, 1, right_bc2),
        da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<T>(handle), da_status_success);
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients, &dim,
                                   coeffs.data()),
              da_status_success);
    filename = std::string(DATA_DIR) + "/interpolation/coeffs_exp_nonuni_sf.csv";
    if (read_coeffs(filename, &expected_coeffs, n_col, n_row) != 0)
        FAIL() << "Failed reading the file " + filename;
    EXPECT_ARR_NEAR(dim, coeffs.data(), expected_coeffs, tol);

    // set second/second derivatives boundary conditions
    if (expected_coeffs != nullptr) {
        free(expected_coeffs);
        expected_coeffs = nullptr;
    }
    EXPECT_EQ(
        da_interpolation_set_boundary_conditions(handle, 1, 2, left_bc2, 2, right_bc2),
        da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<T>(handle), da_status_success);
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients, &dim,
                                   coeffs.data()),
              da_status_success);
    filename = std::string(DATA_DIR) + "/interpolation/coeffs_exp_nonuni_ss.csv";
    if (read_coeffs(filename, &expected_coeffs, n_col, n_row) != 0)
        FAIL() << "Failed reading the file " + filename;
    EXPECT_ARR_NEAR(dim, coeffs.data(), expected_coeffs, tol);

    da_handle_destroy(&handle);
    if (coord != nullptr) {
        free(coord);
        coord = nullptr;
    }
    if (expected_coeffs != nullptr) {
        free(expected_coeffs);
        expected_coeffs = nullptr;
    }
}

TYPED_TEST(cubic_spline_public, evaluate) {
    using T = TypeParam;

    da_numeric::tolerance<T> tol_struct;
    T tol = tol_struct.tol(10.0, 1.0);

    da_handle handle = nullptr;

    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_interpolation), da_status_success);
    EXPECT_EQ(da_interpolation_select_model<T>(handle, interpolation_cubic_spline),
              da_status_success);
    // Set up a simple cubic spline: y = 1*x^3 + 2*x^2 + 3*x + 4 over [0, 3] with 4 points
    da_int n_sites = 4;
    T x_start = (T)0.0;
    T x_end = (T)3.0;
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, n_sites, x_start, x_end),
              da_status_success);
    // y = 1*x^3 + 2*x^2 + 3*x + 4 at x = 0, 1, 2, 3
    std::vector<T> y_data = {(T)4.0, (T)10.0, (T)26.0, (T)58.0};
    da_int dim_y = 1, ldy = n_sites;
    EXPECT_EQ(da_interpolation_set_values(handle, n_sites, dim_y, y_data.data(), ldy, 0),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "cubic spline type", "custom"),
              da_status_success);
    T left_bc[] = {(T)3.0};
    T right_bc[] = {(T)42.0};
    EXPECT_EQ(
        da_interpolation_set_boundary_conditions(handle, 1, 1, left_bc, 1, right_bc),
        da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<T>(handle), da_status_success);

    // Evaluate at specific points
    da_int n_eval = 7;
    std::vector<T> x_eval = {(T)-1.0, (T)0.5, (T)1.5, (T)2.0, (T)2.5, (T)3.0, (T)4.0};
    std::vector<T> y_eval(n_eval);
    da_int order = 0;
    EXPECT_EQ(da_interpolation_evaluate(handle, n_eval, x_eval.data(), y_eval.data(), 1,
                                        &order),
              da_status_success);
    // Expected values: y = 1*x^3 + 2*x^2 + 3*x + 4
    std::vector<T> expected_y(n_eval);
    for (da_int i = 0; i < n_eval; i++) {
        expected_y[i] = (T)(x_eval[i] * x_eval[i] * x_eval[i] +
                            2 * x_eval[i] * x_eval[i] + 3 * x_eval[i] + 4);
    }
    EXPECT_ARR_NEAR(n_eval, y_eval.data(), expected_y.data(), tol);

    // First derivatives
    order = 1;
    EXPECT_EQ(da_interpolation_evaluate(handle, n_eval, x_eval.data(), y_eval.data(), 1,
                                        &order),
              da_status_success);
    // Expected values: y' = 3*x^2 + 4*x + 3
    for (da_int i = 0; i < n_eval; i++) {
        expected_y[i] = (T)(3 * x_eval[i] * x_eval[i] + 4 * x_eval[i] + 3);
    }
    EXPECT_ARR_NEAR(n_eval, y_eval.data(), expected_y.data(), tol);

    // Second derivatives
    order = 2;
    EXPECT_EQ(da_interpolation_evaluate(handle, n_eval, x_eval.data(), y_eval.data(), 1,
                                        &order),
              da_status_success);
    // Expected values: y = 6*x + 4
    for (da_int i = 0; i < n_eval; i++) {
        expected_y[i] = (T)(6 * x_eval[i] + 4);
    }
    EXPECT_ARR_NEAR(n_eval, y_eval.data(), expected_y.data(), tol);

    // Third derivatives
    order = 3;
    EXPECT_EQ(da_interpolation_evaluate(handle, n_eval, x_eval.data(), y_eval.data(), 1,
                                        &order),
              da_status_success);
    // Expected values: y = 6
    for (da_int i = 0; i < n_eval; i++) {
        expected_y[i] = (T)6.0;
    }
    EXPECT_ARR_NEAR(n_eval, y_eval.data(), expected_y.data(), tol);

    // Test multiple orders at once: evaluate orders 0, 1, 2, 3 simultaneously
    da_int n_orders = 4;
    std::vector<da_int> orders = {0, 1, 2, 3};
    std::vector<T> y_eval_multi(n_eval * n_orders);
    EXPECT_EQ(da_interpolation_evaluate(handle, n_eval, x_eval.data(),
                                        y_eval_multi.data(), n_orders, orders.data()),
              da_status_success);

    for (da_int k = 0; k < n_orders; k++) {
        std::vector<T> expected(n_eval);
        for (da_int i = 0; i < n_eval; i++) {
            T x = x_eval[i];
            switch (orders[k]) {
            case 0: // y = x^3 + 2*x^2 + 3*x + 4
                expected[i] = x * x * x + 2 * x * x + 3 * x + 4;
                break;
            case 1: // y' = 3*x^2 + 4*x + 3
                expected[i] = 3 * x * x + 4 * x + 3;
                break;
            case 2: // y'' = 6*x + 4
                expected[i] = 6 * x + 4;
                break;
            case 3: // y''' = 6
                expected[i] = (T)6.0;
                break;
            }
        }
        T *cmp = &y_eval_multi.data()[k * n_eval];
        EXPECT_ARR_NEAR(n_eval, cmp, expected.data(), tol);
    }

    da_handle_destroy(&handle);
}

TYPED_TEST(cubic_spline_public, hermite) {
    using T = TypeParam;
    da_handle handle = nullptr;

    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_interpolation), da_status_success);
    EXPECT_EQ(da_interpolation_select_model<T>(handle, interpolation_cubic_spline),
              da_status_success);

    da_int n_sites = 10;
    T x_start = 0.;
    T x_end = 9.;
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, n_sites, x_start, x_end),
              da_status_success);
    // y = J_0(sqrt(x))
    std::vector<T> y_data(n_sites);
    for (da_int i = 0; i < n_sites; i++) {
        T x = static_cast<T>(i);
        y_data[i] = x * x * x + 2 * x * x + 3 * x + 4;
    }
    std::vector<T> y_deriv(n_sites);
    for (da_int i = 0; i < n_sites; i++) {
        T x = static_cast<T>(i);
        y_deriv[i] = 3 * x * x + 4 * x + 3;
    }
    da_int dim_y = 1, ldy = n_sites;
    EXPECT_EQ(da_interpolation_set_values(handle, n_sites, dim_y, y_data.data(), ldy, 0),
              da_status_success);
    EXPECT_EQ(da_interpolation_set_values(handle, n_sites, dim_y, y_deriv.data(), ldy, 1),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "cubic spline type", "hermite"),
              da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<T>(handle), da_status_success);
    da_int dim = 4 * (n_sites - 1);
    std::vector<T> coeffs(dim);
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients, &dim,
                                   coeffs.data()),
              da_status_success);
    // Expected coefficients for cubic Hermite spline interpolation of the given data
    std::vector<T> expected_coeffs = {
        (T)4.0,   (T)3.0,   (T)2.0,  (T)1.0, (T)10.0,  (T)10.0,  (T)5.0,  (T)1.0,
        (T)26.0,  (T)23.0,  (T)8.0,  (T)1.0, (T)58.0,  (T)42.0,  (T)11.0, (T)1.0,
        (T)112.0, (T)67.0,  (T)14.0, (T)1.0, (T)194.0, (T)98.0,  (T)17.0, (T)1.0,
        (T)310.0, (T)135.0, (T)20.0, (T)1.0, (T)466.0, (T)178.0, (T)23.0, (T)1.0,
        (T)668.0, (T)227.0, (T)26.0, (T)1.0};
    T tol = (T)1.0e-06;
    EXPECT_ARR_NEAR(dim, coeffs.data(), expected_coeffs.data(), tol);

    std::vector<T> x_eval = {(T)0.5, (T)1.5, (T)2.5, (T)3.5, (T)4.5,
                             (T)5.5, (T)6.5, (T)7.5, (T)8.5};
    da_int n_eval = static_cast<da_int>(x_eval.size());
    std::vector<T> y_eval(n_eval * 2);
    da_int order[] = {0, 1};
    EXPECT_EQ(
        da_interpolation_evaluate(handle, n_eval, x_eval.data(), y_eval.data(), 2, order),
        da_status_success);
    // Expected values: y = x^3 + 2*x^2 + 3*x + 4
    std::vector<T> expected_y(n_eval);
    for (da_int i = 0; i < n_eval; i++) {
        T x = x_eval[i];
        expected_y[i] = x * x * x + 2 * x * x + 3 * x + 4;
    }
    EXPECT_ARR_NEAR(n_eval, y_eval.data(), expected_y.data(), tol);
    // Expected first derivatives: y' = 3*x^2 + 4*x + 3
    for (da_int i = 0; i < n_eval; i++) {
        T x = x_eval[i];
        expected_y[i] = 3 * x * x + 4 * x + 3;
    }
    T *yp_exp = &y_eval[n_eval];
    EXPECT_ARR_NEAR(n_eval, yp_exp, expected_y.data(), tol);

    da_handle_destroy(&handle);
}

TYPED_TEST(cubic_spline_public, no_model) {
    using T = TypeParam;
    da_handle handle = nullptr;

    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_interpolation), da_status_success);

    // Try all the functions without selecting a model
    da_int n_sites = 10;
    T x_start = 0.;
    T x_end = 9.;
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, n_sites, x_start, x_end),
              da_status_invalid_input);
    std::vector<T> y_data(n_sites);
    for (da_int i = 0; i < n_sites; i++) {
        y_data[i] = std::sin(static_cast<T>(i));
    }
    EXPECT_EQ(da_interpolation_set_sites(handle, n_sites, y_data.data()),
              da_status_invalid_input);
    da_int dim_y = 1, ldy = n_sites;
    EXPECT_EQ(da_interpolation_set_values(handle, n_sites, dim_y, y_data.data(), ldy, 0),
              da_status_invalid_input);
    EXPECT_EQ(da_interpolation_interpolate<T>(handle), da_status_invalid_input);
    da_int order = 0;
    std::vector<T> x = {(T)0.5, (T)1.5, (T)2.5};
    std::vector<T> y(3);
    EXPECT_EQ(da_interpolation_evaluate(handle, (da_int)3, x.data(), y.data(), 1, &order),
              da_status_invalid_input);
    std::vector<da_int> cells(3);
    EXPECT_EQ(da_interpolation_search_cells(handle, (da_int)3, x.data(), cells.data()),
              da_status_invalid_input);
    T left_bc[] = {(T)1.0};
    T right_bc[] = {(T)1.0};
    EXPECT_EQ(
        da_interpolation_set_boundary_conditions(handle, 1, 1, left_bc, 1, right_bc),
        da_status_invalid_input);

    // Try to unset the model
    EXPECT_EQ(da_interpolation_select_model<T>(handle, interpolation_unset),
              da_status_invalid_input);
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, n_sites, x_start, x_end),
              da_status_invalid_input);

    da_handle_destroy(&handle);
}

TYPED_TEST(cubic_spline_public, multiple_dimensions) {
    using T = TypeParam;

    std::vector<std::string> spline_types = {"natural", "clamped zero", "custom",
                                             "hermite"};
    for (const auto &spline_type : spline_types) {
        std::cout << "Testing spline type: " << spline_type << std::endl;
        da_handle handle = nullptr;
        EXPECT_EQ(da_handle_init<T>(&handle, da_handle_interpolation), da_status_success);
        EXPECT_EQ(da_interpolation_select_model<T>(handle, interpolation_cubic_spline),
                  da_status_success);
        da_int n_sites = 5;
        std::vector<T> x_data = {0., 1.5, 2.2, 3., 4.};
        EXPECT_EQ(da_interpolation_set_sites(handle, n_sites, x_data.data()),
                  da_status_success);

        da_int dim_y = 3;
        std::vector<T> y_data(n_sites * dim_y);

        for (da_int i = 0; i < n_sites; i++) {
            T x = x_data[i];
            y_data[i + 0 * n_sites] = 4 * x * x * x + 3 * x * x + 2 * x + 1;
            y_data[i + 1 * n_sites] = std::sin(x);
            y_data[i + 2 * n_sites] = std::exp(x);
        }
        da_int order = 0;
        EXPECT_EQ(da_interpolation_set_values(handle, n_sites, dim_y, y_data.data(),
                                              n_sites, order),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "cubic spline type", spline_type.c_str()),
                  da_status_success);
        T left_bc[] = {(T)1.0, (T)2.0, (T)3.0};
        T right_bc[] = {(T)0.0, (T)2.0, (T)4.0};
        if (spline_type == "custom") {
            // Set natural boundary conditions for custom
            EXPECT_EQ(da_interpolation_set_boundary_conditions(handle, dim_y, 1, left_bc,
                                                               1, right_bc),
                      da_status_success);
        }
        std::vector<T> y_deriv(n_sites * dim_y);
        if (spline_type == "hermite") {
            // Set first derivatives for hermite
            for (da_int i = 0; i < n_sites; i++) {
                T x = x_data[i];
                y_deriv[0 * n_sites + i] = 12 * x * x + 6 * x + 2;
                y_deriv[1 * n_sites + i] = std::cos(x);
                y_deriv[2 * n_sites + i] = std::exp(x);
            }
            EXPECT_EQ(da_interpolation_set_values(handle, n_sites, dim_y, y_deriv.data(),
                                                  n_sites, 1),
                      da_status_success);
        }

        EXPECT_EQ(da_interpolation_interpolate<T>(handle), da_status_success);
        da_int dim = 4 * (n_sites - 1) * dim_y;
        std::vector<T> coeffs(dim);
        EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients,
                                       &dim, coeffs.data()),
                  da_status_success);

        std::vector<T> x_eval = {(T)0.,  (T)0.5, (T)0.7, (T)1.8,
                                 (T)2.1, (T)3.3, (T)4.0, (T)5.1};
        da_int n_eval = static_cast<da_int>(x_eval.size());
        std::vector<T> y_eval(4 * n_eval * dim_y);
        da_int order_eval[] = {0, 1, 2, 3};
        EXPECT_EQ(da_interpolation_evaluate(handle, n_eval, x_eval.data(), y_eval.data(),
                                            4, order_eval),
                  da_status_success);

        for (da_int d = 0; d < dim_y; d++) {
            std::cout << "Checking dimension " << d << std::endl;
            da_handle h = nullptr;
            EXPECT_EQ(da_handle_init<T>(&h, da_handle_interpolation), da_status_success);
            EXPECT_EQ(da_interpolation_select_model<T>(h, interpolation_cubic_spline),
                      da_status_success);
            EXPECT_EQ(da_interpolation_set_sites(h, n_sites, x_data.data()),
                      da_status_success);
            EXPECT_EQ(da_options_set_string(h, "cubic spline type", spline_type.c_str()),
                      da_status_success);
            if (spline_type == "custom") {
                // Set natural boundary conditions for custom
                EXPECT_EQ(da_interpolation_set_boundary_conditions(h, 1, 1, &left_bc[d],
                                                                   1, &right_bc[d]),
                          da_status_success);
            }
            EXPECT_EQ(da_interpolation_set_values(
                          h, n_sites, 1, &y_data.data()[n_sites * d], n_sites, order),
                      da_status_success);
            std::vector<T> y_deriv_1d(n_sites);
            if (spline_type == "hermite") {
                // Set first derivatives for hermite
                for (da_int i = 0; i < n_sites; i++) {
                    T x = x_data[i];
                    if (d == 0) {
                        y_deriv_1d[i] = 12 * x * x + 6 * x + 2;
                    } else if (d == 1) {
                        y_deriv_1d[i] = std::cos(x);
                    } else if (d == 2) {
                        y_deriv_1d[i] = std::exp(x);
                    }
                }
                EXPECT_EQ(da_interpolation_set_values(h, n_sites, 1, y_deriv_1d.data(),
                                                      n_sites, 1),
                          da_status_success);
            }
            EXPECT_EQ(da_interpolation_interpolate<T>(h), da_status_success);
            dim = 4 * (n_sites - 1);
            std::vector<T> coeffs_1d(dim);
            EXPECT_EQ(da_handle_get_result(h, da_result::da_cubic_spline_coefficients,
                                           &dim, coeffs_1d.data()),
                      da_status_success);
            T *coeff_ptr = &coeffs.data()[d * dim];
            EXPECT_ARR_NEAR(dim, coeff_ptr, coeffs_1d.data(), (T)1.0e-06);

            std::vector<T> y_eval_1d(4 * n_eval);
            EXPECT_EQ(da_interpolation_evaluate(h, n_eval, x_eval.data(),
                                                y_eval_1d.data(), 4, order_eval),
                      da_status_success);
            for (da_int ord = 0; ord < 4; ord++) {
                T *eval_ptr = &y_eval.data()[(ord * dim_y + d) * n_eval];
                T *eval_ptr_1d = &y_eval_1d.data()[ord * n_eval];
                EXPECT_ARR_NEAR(n_eval, eval_ptr, eval_ptr_1d, (T)1.0e-06);
            }
            da_handle_destroy(&h);
        }
        da_handle_destroy(&handle);
    }
}

TYPED_TEST(cubic_spline_public, row_major_multiple_dimensions) {
    using T = TypeParam;

    std::vector<std::string> spline_types = {"natural", "clamped zero", "custom",
                                             "hermite"};
    for (const auto &spline_type : spline_types) {
        std::cout << "Testing spline type: " << spline_type << std::endl;
        da_handle handle = nullptr;
        EXPECT_EQ(da_handle_init<T>(&handle, da_handle_interpolation), da_status_success);
        EXPECT_EQ(da_interpolation_select_model<T>(handle, interpolation_cubic_spline),
                  da_status_success);
        da_int n_sites = 5;
        std::vector<T> x_data = {0., 1.5, 2.2, 3., 4.};
        EXPECT_EQ(da_interpolation_set_sites(handle, n_sites, x_data.data()),
                  da_status_success);

        da_int dim_y = 3;
        std::vector<T> y_data(n_sites * dim_y);

        for (da_int i = 0; i < n_sites; i++) {
            T x = x_data[i];
            y_data[i * dim_y + 0] = 4 * x * x * x + 3 * x * x + 2 * x + 1;
            y_data[i * dim_y + 1] = std::sin(x);
            y_data[i * dim_y + 2] = std::exp(x);
        }
        EXPECT_EQ(da_options_set_string(handle, "storage order", "row-major"),
                  da_status_success);
        da_int order = 0;
        EXPECT_EQ(da_interpolation_set_values(handle, n_sites, dim_y, y_data.data(),
                                              dim_y, order),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "cubic spline type", spline_type.c_str()),
                  da_status_success);
        if (spline_type == "custom") {
            // Set natural boundary conditions for custom
            std::vector<T> left_bc(dim_y, (T)1.0);
            std::vector<T> right_bc(dim_y, (T)1.0);
            EXPECT_EQ(da_interpolation_set_boundary_conditions(
                          handle, dim_y, 1, left_bc.data(), 1, right_bc.data()),
                      da_status_success);
        }
        std::vector<T> y_deriv(n_sites * dim_y);
        if (spline_type == "hermite") {
            for (da_int i = 0; i < n_sites; i++) {
                T x = x_data[i];
                y_deriv[i * dim_y + 0] = 12 * x * x + 6 * x + 2;
                y_deriv[i * dim_y + 1] = std::cos(x);
                y_deriv[i * dim_y + 2] = std::exp(x);
            }
            EXPECT_EQ(da_interpolation_set_values(handle, n_sites, dim_y, y_deriv.data(),
                                                  dim_y, 1),
                      da_status_success);
        }

        EXPECT_EQ(da_interpolation_interpolate<T>(handle), da_status_success);
        da_int dim = 4 * (n_sites - 1) * dim_y;
        std::vector<T> coeffs(dim);
        EXPECT_EQ(da_handle_get_result(handle, da_result::da_cubic_spline_coefficients,
                                       &dim, coeffs.data()),
                  da_status_success);

        std::vector<T> x_eval = {(T)0.,  (T)0.5, (T)0.7, (T)1.8,
                                 (T)2.1, (T)3.3, (T)4.0, (T)5.1};
        da_int n_eval = static_cast<da_int>(x_eval.size());
        std::vector<T> y_eval(4 * n_eval * dim_y);
        da_int order_eval[] = {0, 1, 2, 3};
        EXPECT_EQ(da_interpolation_evaluate(handle, n_eval, x_eval.data(), y_eval.data(),
                                            4, order_eval),
                  da_status_success);

        for (da_int d = 0; d < dim_y; d++) {
            std::cout << "Checking dimension " << d << std::endl;
            da_handle h = nullptr;
            EXPECT_EQ(da_handle_init<T>(&h, da_handle_interpolation), da_status_success);
            EXPECT_EQ(da_interpolation_select_model<T>(h, interpolation_cubic_spline),
                      da_status_success);
            EXPECT_EQ(da_interpolation_set_sites(h, n_sites, x_data.data()),
                      da_status_success);
            EXPECT_EQ(da_options_set_string(h, "cubic spline type", spline_type.c_str()),
                      da_status_success);
            if (spline_type == "custom") {
                // Set natural boundary conditions for custom
                T left_bc[] = {(T)1.0};
                T right_bc[] = {(T)1.0};
                EXPECT_EQ(da_interpolation_set_boundary_conditions(h, 1, 1, left_bc, 1,
                                                                   right_bc),
                          da_status_success);
            }
            std::vector<T> y_data_1d(n_sites);
            for (da_int i = 0; i < n_sites; i++) {
                y_data_1d[i] = y_data[i * dim_y + d];
            }
            EXPECT_EQ(da_interpolation_set_values(h, n_sites, 1, y_data_1d.data(),
                                                  n_sites, order),
                      da_status_success);
            std::vector<T> y_deriv_1d(n_sites);
            if (spline_type == "hermite") {
                for (da_int i = 0; i < n_sites; i++) {
                    T x = x_data[i];
                    if (d == 0) {
                        y_deriv_1d[i] = 12 * x * x + 6 * x + 2;
                    } else if (d == 1) {
                        y_deriv_1d[i] = std::cos(x);
                    } else if (d == 2) {
                        y_deriv_1d[i] = std::exp(x);
                    }
                }
                EXPECT_EQ(da_interpolation_set_values(h, n_sites, 1, y_deriv_1d.data(),
                                                      n_sites, 1),
                          da_status_success);
            }
            EXPECT_EQ(da_interpolation_interpolate<T>(h), da_status_success);
            dim = 4 * (n_sites - 1);
            std::vector<T> coeffs_1d(dim);
            EXPECT_EQ(da_handle_get_result(h, da_result::da_cubic_spline_coefficients,
                                           &dim, coeffs_1d.data()),
                      da_status_success);
            T *coeff_ptr = &coeffs.data()[d * dim];
            EXPECT_ARR_NEAR(dim, coeff_ptr, coeffs_1d.data(), (T)1.0e-06);
            std::vector<T> y_eval_1d(4 * n_eval);
            EXPECT_EQ(da_interpolation_evaluate(h, n_eval, x_eval.data(),
                                                y_eval_1d.data(), 4, order_eval),
                      da_status_success);
            for (da_int ord = 0; ord < 4; ord++) {
                T *eval_ptr = &y_eval.data()[(ord * dim_y + d) * n_eval];
                T *eval_ptr_1d = &y_eval_1d.data()[ord * n_eval];
                EXPECT_ARR_NEAR(n_eval, eval_ptr, eval_ptr_1d, (T)1.0e-06);
            }

            da_handle_destroy(&h);
        }

        da_handle_destroy(&handle);
    }
}

TYPED_TEST(cubic_spline_public, multiple_orders) {

    using T = TypeParam;

    da_numeric::tolerance<T> tol_struct;
    T tol = tol_struct.tol(10.0, 1.0);

    // interpolation data:
    // x uniform in [0,9] with 10 points
    // y[:,0] = x^3 + 2*x^2 + 3*x +4
    // y[:,1] = 3*x^2 + 4*x +3
    T x_start = (T)0.0;
    T x_end = (T)2.0;
    da_int n_sites = 10;
    std::vector<T> y_data(n_sites * 2);
    T dx = (x_end - x_start) / (n_sites - 1);
    for (da_int i = 0; i < n_sites; i++) {
        T x = x_start + i * dx;
        y_data[i + 0 * n_sites] = x * x * x + 2 * x * x + 3 * x + 4;
        y_data[i + 1 * n_sites] = 3 * x * x + 4 * x + 3;
    }

    // Initialize handle
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_interpolation), da_status_success);
    EXPECT_EQ(da_interpolation_select_model<T>(handle, interpolation_cubic_spline),
              da_status_success);
    EXPECT_EQ(da_interpolation_set_sites_uniform(handle, n_sites, x_start, x_end),
              da_status_success);
    da_int dim_y = 2, ldy = n_sites;
    EXPECT_EQ(da_interpolation_set_values(handle, n_sites, dim_y, y_data.data(), ldy, 0),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "cubic spline type", "custom"),
              da_status_success);
    T left_bc[] = {(T)3.0, (T)4.0};
    T right_bc[] = {(T)23.0, (T)16.0};
    EXPECT_EQ(
        da_interpolation_set_boundary_conditions(handle, dim_y, 1, left_bc, 1, right_bc),
        da_status_success);
    EXPECT_EQ(da_interpolation_interpolate<T>(handle), da_status_success);

    // Set up evaluation points
    da_int n_eval = 20;
    std::vector<T> x_eval(n_eval);
    T step = (x_end - x_start) / ((n_eval - 1));
    for (da_int i = 0; i < n_eval; i++) {
        x_eval[i] = (T)0.02 + i * step;
    }
    std::vector<T> y_eval(4 * n_eval * dim_y);
    std::vector<T> expected_y(4 * n_eval * dim_y);
    for (da_int ord = 0; ord < 4; ord++) {
        for (da_int i = 0; i < n_eval; i++) {
            T x = x_eval[i];
            // First dimension
            switch (ord) {
            case 0:
                expected_y[(ord * dim_y + 0) * n_eval + i] =
                    x * x * x + 2 * x * x + 3 * x + 4;
                break;
            case 1:
                expected_y[(ord * dim_y + 0) * n_eval + i] = 3 * x * x + 4 * x + 3;
                break;
            case 2:
                expected_y[(ord * dim_y + 0) * n_eval + i] = 6 * x + 4;
                break;
            case 3:
                expected_y[(ord * dim_y + 0) * n_eval + i] = (T)6.0;
                break;
            }
            // Second dimension
            switch (ord) {
            case 0:
                expected_y[(ord * dim_y + 1) * n_eval + i] = 3 * x * x + 4 * x + 3;
                break;
            case 1:
                expected_y[(ord * dim_y + 1) * n_eval + i] = 6 * x + 4;
                break;
            case 2:
                expected_y[(ord * dim_y + 1) * n_eval + i] = (T)6.0;
                break;
            case 3:
                expected_y[(ord * dim_y + 1) * n_eval + i] = (T)0.0;
                break;
            }
        }
    }

    typedef struct order_test {
        da_int n_orders;
        da_int orders[4];
    } order_test_t;

    order_test_t tests[] = {
        {1, {0}},       {1, {1}},       {1, {2}},       {1, {3}},          {2, {0, 1}},
        {2, {0, 2}},    {2, {0, 3}},    {2, {1, 2}},    {2, {3, 1}},       {2, {2, 3}},
        {3, {3, 2, 0}}, {3, {0, 2, 1}}, {3, {3, 2, 1}}, {4, {0, 1, 2, 3}},
    };

    for (auto &test : tests) {
        std::cout << "Evaluating for orders: ";
        for (da_int i = 0; i < test.n_orders; i++) {
            std::cout << test.orders[i] << " ";
        }
        std::cout << std::endl;
        EXPECT_EQ(da_interpolation_evaluate(handle, n_eval, x_eval.data(), y_eval.data(),
                                            test.n_orders, test.orders),
                  da_status_success);
        std::vector<da_int> sorted_idx = {0, 1, 2, 3};
        std::sort(
            sorted_idx.begin(), sorted_idx.begin() + test.n_orders,
            [&test](da_int a, da_int b) { return test.orders[a] < test.orders[b]; });
        for (da_int ord_idx = 0; ord_idx < test.n_orders; ord_idx++) {
            da_int ord = test.orders[ord_idx];
            da_int ord_y_idx = sorted_idx[ord_idx];
            for (da_int d = 0; d < dim_y; d++) {
                std::cout << "Checking order " << ord << " dimension " << d << std::endl;
                T *eval_ptr = &y_eval.data()[(ord_y_idx * dim_y + d) * n_eval];
                T *exp_ptr = &expected_y.data()[(ord * dim_y + d) * n_eval];
                EXPECT_ARR_NEAR(n_eval, eval_ptr, exp_ptr, tol);
            }
        }
    }

    da_handle_destroy(&handle);
}
