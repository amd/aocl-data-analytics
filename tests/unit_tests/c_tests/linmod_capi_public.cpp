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
 * Test the linear model C API (double precision).
 * Based on tests/examples/linear_model.cpp
 */
TEST(LinmodCAPI, MseDouble) {
    da_handle handle = nullptr;

    // Problem data: 5 samples, 2 features
    da_int m = 5, n = 2, ldA = 6;
    double Ad[12] = {1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 3.0, 5.0, 1.0, 1.0, 0.0};
    double bd[5] = {1.0, 1.0, 1.0, 1.0, 1.0};

    EXPECT_EQ(da_handle_init_d(&handle, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_select_model_d(handle, linmod_model_mse), da_status_success);
    EXPECT_EQ(da_linmod_define_features_d(handle, m, n, Ad, ldA, bd), da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle), da_status_success);

    // Get coefficients
    da_int nx = 2;
    double x[2];
    EXPECT_EQ(da_handle_get_result_d(handle, da_linmod_coef, &nx, x), da_status_success);

    // Expected coefficients (approximate)
    EXPECT_NEAR(x[0], 0.199256, 1.0e-4);
    EXPECT_NEAR(x[1], 0.130354, 1.0e-4);

    // Evaluate model on test data
    double X_test[4] = {1.0, 2.0, 2.0, 3.0};
    double predictions[2];
    double observations[2] = {0.5, 0.8};
    double loss = 0.0;
    EXPECT_EQ(da_linmod_evaluate_model_d(handle, 2, n, X_test, 2, predictions,
                                         observations, &loss),
              da_status_success);

    da_handle_destroy(&handle);
}

/*
 * Test the linear model C API (single precision).
 */
TEST(LinmodCAPI, MseFloat) {
    da_handle handle = nullptr;

    da_int m = 5, n = 2, ldA = 6;
    float As[12] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                    0.0f, 1.0f, 3.0f, 5.0f, 1.0f, 1.0f};
    float bs[5] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    EXPECT_EQ(da_handle_init_s(&handle, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_select_model_s(handle, linmod_model_mse), da_status_success);
    EXPECT_EQ(da_linmod_define_features_s(handle, m, n, &As[1], ldA, bs),
              da_status_success);
    EXPECT_EQ(da_linmod_fit_s(handle), da_status_success);

    da_int nx = 2;
    float xs[2];
    EXPECT_EQ(da_handle_get_result_s(handle, da_linmod_coef, &nx, xs), da_status_success);

    EXPECT_NEAR(xs[0], 0.20f, 0.01f);
    EXPECT_NEAR(xs[1], 0.13f, 0.01f);

    // Evaluate model
    float X_test[4] = {1.0f, 2.0f, 2.0f, 3.0f};
    float predictions[2];
    float observations[2] = {0.5f, 0.8f};
    float loss = 0.0f;
    EXPECT_EQ(da_linmod_evaluate_model_s(handle, 2, n, X_test, 2, predictions,
                                         observations, &loss),
              da_status_success);

    da_handle_destroy(&handle);
}

/*
 * Test linmod_fit_start (warm start).
 */
TEST(LinmodCAPI, FitStartDouble) {
    da_handle handle = nullptr;

    da_int m = 5, n = 2, ldA = 5;
    double A[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 3.0, 5.0, 1.0, 1.0};
    double b[5] = {1.0, 1.0, 1.0, 1.0, 1.0};

    EXPECT_EQ(da_handle_init_d(&handle, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_select_model_d(handle, linmod_model_mse), da_status_success);
    EXPECT_EQ(da_linmod_define_features_d(handle, m, n, A, ldA, b), da_status_success);

    // Provide a starting point
    double coefs[2] = {0.1, 0.1};
    EXPECT_EQ(da_linmod_fit_start_d(handle, 2, coefs), da_status_success);
    EXPECT_EQ(da_linmod_fit_d(handle), da_status_success);

    da_int nx = 2;
    double x[2];
    EXPECT_EQ(da_handle_get_result_d(handle, da_linmod_coef, &nx, x), da_status_success);

    da_handle_destroy(&handle);
}

/*
 * Test linmod_fit_start (warm start, single precision).
 */
TEST(LinmodCAPI, FitStartFloat) {
    da_handle handle = nullptr;

    da_int m = 5, n = 2, ldA = 5;
    float A[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 1.0f, 3.0f, 5.0f, 1.0f, 1.0f};
    float b[5] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    EXPECT_EQ(da_handle_init_s(&handle, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_select_model_s(handle, linmod_model_mse), da_status_success);
    EXPECT_EQ(da_linmod_define_features_s(handle, m, n, A, ldA, b), da_status_success);

    // Provide a starting point
    float coefs[2] = {0.1f, 0.1f};
    EXPECT_EQ(da_linmod_fit_start_s(handle, 2, coefs), da_status_success);
    EXPECT_EQ(da_linmod_fit_s(handle), da_status_success);

    da_int nx = 2;
    float x[2];
    EXPECT_EQ(da_handle_get_result_s(handle, da_linmod_coef, &nx, x), da_status_success);

    da_handle_destroy(&handle);
}
