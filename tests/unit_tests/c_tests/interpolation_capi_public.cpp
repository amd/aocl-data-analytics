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
#include "gtest/gtest.h"
#include <cmath>

/*
 * Test the interpolation C API (double precision).
 * Based on tests/examples/cubic_spline.cpp
 */
TEST(InterpolationCAPI, CubicSplineDouble) {
    da_handle handle = nullptr;

    EXPECT_EQ(da_handle_init_d(&handle, da_handle_interpolation), da_status_success);
    EXPECT_EQ(da_interpolation_select_model_d(handle, interpolation_cubic_spline),
              da_status_success);

    // Uniform sites: sin(x) on [0, 9]
    da_int n_sites = 10;
    double x_start = 0.0, x_end = 9.0;
    EXPECT_EQ(da_interpolation_set_sites_uniform_d(handle, n_sites, x_start, x_end),
              da_status_success);

    // Values: sin at uniform points
    double y[10];
    double step = (x_end - x_start) / (n_sites - 1);
    for (da_int i = 0; i < n_sites; i++)
        y[i] = std::sin(x_start + i * step);

    EXPECT_EQ(da_interpolation_set_values_d(handle, n_sites, 1, y, n_sites, 0),
              da_status_success);

    // Interpolate
    EXPECT_EQ(da_interpolation_interpolate_d(handle), da_status_success);

    // Evaluate
    da_int n_eval = 5;
    double x_eval[5] = {0.5, 1.5, 3.0, 5.5, 8.0};
    double y_eval[5];
    da_int orders = 0;
    EXPECT_EQ(da_interpolation_evaluate_d(handle, n_eval, x_eval, y_eval, 1, &orders),
              da_status_success);

    // Check interpolation is reasonable (within 0.1 of sin)
    for (da_int i = 0; i < n_eval; i++)
        EXPECT_NEAR(y_eval[i], std::sin(x_eval[i]), 0.1);

    da_handle_destroy(&handle);
}

/*
 * Test interpolation with explicit sites (double).
 */
TEST(InterpolationCAPI, ExplicitSitesDouble) {
    da_handle handle = nullptr;

    EXPECT_EQ(da_handle_init_d(&handle, da_handle_interpolation), da_status_success);
    EXPECT_EQ(da_interpolation_select_model_d(handle, interpolation_cubic_spline),
              da_status_success);

    // Explicit sites
    da_int n_sites = 5;
    double x_sites[5] = {0.0, 1.0, 2.0, 3.0, 4.0};
    EXPECT_EQ(da_interpolation_set_sites_d(handle, n_sites, x_sites), da_status_success);

    // Values: x^2
    double y[5] = {0.0, 1.0, 4.0, 9.0, 16.0};
    EXPECT_EQ(da_interpolation_set_values_d(handle, n_sites, 1, y, n_sites, 0),
              da_status_success);

    // Set boundary conditions
    double left_val[1] = {0.0};
    double right_val[1] = {8.0};
    EXPECT_EQ(
        da_interpolation_set_boundary_conditions_d(handle, 1, 1, left_val, 1, right_val),
        da_status_success);

    EXPECT_EQ(da_interpolation_interpolate_d(handle), da_status_success);

    // Search cells
    da_int n_eval = 3;
    double x_eval[3] = {0.5, 1.5, 3.5};
    da_int cells[3];
    EXPECT_EQ(da_interpolation_search_cells_d(handle, n_eval, x_eval, cells),
              da_status_success);

    // Evaluate
    double y_eval[3];
    da_int orders = 0;
    EXPECT_EQ(da_interpolation_evaluate_d(handle, n_eval, x_eval, y_eval, 1, &orders),
              da_status_success);

    // x^2 at 0.5, 1.5, 3.5 = 0.25, 2.25, 12.25
    EXPECT_NEAR(y_eval[0], 0.25, 1.0e-10);
    EXPECT_NEAR(y_eval[1], 2.25, 1.0e-10);
    EXPECT_NEAR(y_eval[2], 12.25, 1.0e-10);

    da_handle_destroy(&handle);
}

/*
 * Test the interpolation C API (single precision).
 */
TEST(InterpolationCAPI, CubicSplineFloat) {
    da_handle handle = nullptr;

    EXPECT_EQ(da_handle_init_s(&handle, da_handle_interpolation), da_status_success);
    EXPECT_EQ(da_interpolation_select_model_s(handle, interpolation_cubic_spline),
              da_status_success);

    da_int n_sites = 10;
    float x_start = 0.0f, x_end = 9.0f;
    EXPECT_EQ(da_interpolation_set_sites_uniform_s(handle, n_sites, x_start, x_end),
              da_status_success);

    float y[10];
    float step = (x_end - x_start) / (n_sites - 1);
    for (da_int i = 0; i < n_sites; i++)
        y[i] = sinf(x_start + i * step);

    EXPECT_EQ(da_interpolation_set_values_s(handle, n_sites, 1, y, n_sites, 0),
              da_status_success);
    EXPECT_EQ(da_interpolation_interpolate_s(handle), da_status_success);

    da_int n_eval = 5;
    float x_eval[5] = {0.5f, 1.5f, 3.0f, 5.5f, 8.0f};
    float y_eval[5];
    da_int orders = 0;
    EXPECT_EQ(da_interpolation_evaluate_s(handle, n_eval, x_eval, y_eval, 1, &orders),
              da_status_success);

    for (da_int i = 0; i < n_eval; i++)
        EXPECT_NEAR(y_eval[i], sinf(x_eval[i]), 0.1f);

    da_handle_destroy(&handle);
}

/*
 * Test interpolation with explicit sites (single precision).
 */
TEST(InterpolationCAPI, ExplicitSitesFloat) {
    da_handle handle = nullptr;

    EXPECT_EQ(da_handle_init_s(&handle, da_handle_interpolation), da_status_success);
    EXPECT_EQ(da_interpolation_select_model_s(handle, interpolation_cubic_spline),
              da_status_success);

    da_int n_sites = 5;
    float x_sites[5] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_EQ(da_interpolation_set_sites_s(handle, n_sites, x_sites), da_status_success);

    float y[5] = {0.0f, 1.0f, 4.0f, 9.0f, 16.0f};
    EXPECT_EQ(da_interpolation_set_values_s(handle, n_sites, 1, y, n_sites, 0),
              da_status_success);

    float left_val[1] = {0.0f};
    float right_val[1] = {8.0f};
    EXPECT_EQ(
        da_interpolation_set_boundary_conditions_s(handle, 1, 1, left_val, 1, right_val),
        da_status_success);

    EXPECT_EQ(da_interpolation_interpolate_s(handle), da_status_success);

    da_int n_eval = 3;
    float x_eval[3] = {0.5f, 1.5f, 3.5f};
    da_int cells[3];
    EXPECT_EQ(da_interpolation_search_cells_s(handle, n_eval, x_eval, cells),
              da_status_success);

    float y_eval[3];
    da_int orders = 0;
    EXPECT_EQ(da_interpolation_evaluate_s(handle, n_eval, x_eval, y_eval, 1, &orders),
              da_status_success);

    // x^2 at 0.5, 1.5, 3.5 = 0.25, 2.25, 12.25
    EXPECT_NEAR(y_eval[0], 0.25f, 1.0e-4f);
    EXPECT_NEAR(y_eval[1], 2.25f, 1.0e-4f);
    EXPECT_NEAR(y_eval[2], 12.25f, 1.0e-4f);

    da_handle_destroy(&handle);
}
