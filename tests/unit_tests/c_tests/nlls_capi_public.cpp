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

#ifdef NO_FORTRAN
TEST(NllsCAPI, NotImplemented) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_nlls), da_status_not_implemented);
    da_handle_destroy(&handle);
}

#else

/* Callback: r_i(x) = x_1 * exp(x_2 * t_i) - y_i */
static da_int eval_r_d(da_int n_coef, da_int n_res, void *params, const double *x,
                       double *r) {
    (void)n_coef;
    double *t = ((double **)params)[0];
    double *y = ((double **)params)[1];
    for (da_int i = 0; i < n_res; i++)
        r[i] = x[0] * exp(x[1] * t[i]) - y[i];
    return 0;
}

/* Jacobian */
static da_int eval_J_d(da_int n_coef, da_int n_res, void *params, const double *x,
                       double *J) {
    (void)n_coef;
    double *t = ((double **)params)[0];
    for (da_int i = 0; i < n_res; i++) {
        J[0 * n_res + i] = exp(x[1] * t[i]);
        J[1 * n_res + i] = t[i] * x[0] * exp(x[1] * t[i]);
    }
    return 0;
}

/* Hessian-residual product */
static da_int eval_HF_d(da_int n_coef, da_int n_res, void *params, const double *x,
                        const double *r, double *HF) {
    double *t = ((double **)params)[0];
    for (da_int i = 0; i < n_coef * n_coef; i++)
        HF[i] = 0.0;
    for (da_int i = 0; i < n_res; i++) {
        HF[1] += r[i] * t[i] * exp(x[1] * t[i]);
        HF[1 * n_coef + 1] += r[i] * t[i] * t[i] * x[0] * exp(x[1] * t[i]);
    }
    HF[1 * n_coef + 0] = HF[1];
    return 0;
}

/* Single precision callbacks */
static da_int eval_r_s(da_int n_coef, da_int n_res, void *params, const float *x,
                       float *r) {
    (void)n_coef;
    float *t = ((float **)params)[0];
    float *y = ((float **)params)[1];
    for (da_int i = 0; i < n_res; i++)
        r[i] = x[0] * expf(x[1] * t[i]) - y[i];
    return 0;
}

static da_int eval_J_s(da_int n_coef, da_int n_res, void *params, const float *x,
                       float *J) {
    (void)n_coef;
    float *t = ((float **)params)[0];
    for (da_int i = 0; i < n_res; i++) {
        J[0 * n_res + i] = expf(x[1] * t[i]);
        J[1 * n_res + i] = t[i] * x[0] * expf(x[1] * t[i]);
    }
    return 0;
}

/*
 * Test the NLLS C API (double precision).
 * Based on tests/examples/nlls.cpp and c_compatibility_nog_public.c
 */
TEST(NllsCAPI, BasicDouble) {
    da_handle handle = nullptr;

    double t[5] = {1.0, 2.0, 4.0, 5.0, 8.0};
    double y[5] = {3.0, 4.0, 6.0, 11.0, 20.0};
    double *params[2] = {t, y};
    double x[2] = {0.5, 0.5};
    double lower_bounds[2] = {0.0, 0.0};
    double upper_bounds[2] = {10.0, 10.0};

    EXPECT_EQ(da_handle_init_d(&handle, da_handle_nlls), da_status_success);
    EXPECT_EQ(
        da_nlls_define_residuals_d(handle, 2, 5, eval_r_d, eval_J_d, eval_HF_d, nullptr),
        da_status_success);
    EXPECT_EQ(da_nlls_define_bounds_d(handle, 2, lower_bounds, upper_bounds),
              da_status_success);

    // Set options
    EXPECT_EQ(da_options_set_int(handle, "print level", 0), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "Storage Order", "Fortran"),
              da_status_success);

    // Define weights
    double weights[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    EXPECT_EQ(da_nlls_define_weights_d(handle, 5, weights), da_status_success);

    // Fit
    EXPECT_EQ(da_nlls_fit_d(handle, 2, x, params), da_status_success);

    // Get results
    da_int dim = 100;
    double info[100];
    EXPECT_EQ(da_handle_get_result_d(handle, da_rinfo, &dim, info), da_status_success);

    da_handle_destroy(&handle);
}

/*
 * Test the NLLS C API (single precision).
 */
TEST(NllsCAPI, BasicFloat) {
    da_handle handle = nullptr;

    float t[5] = {1.0f, 2.0f, 4.0f, 5.0f, 8.0f};
    float y[5] = {3.0f, 4.0f, 6.0f, 11.0f, 20.0f};
    float *params[2] = {t, y};
    float x[2] = {0.5f, 0.5f};
    float lower_bounds[2] = {0.0f, 0.0f};
    float upper_bounds[2] = {10.0f, 10.0f};

    EXPECT_EQ(da_handle_init_s(&handle, da_handle_nlls), da_status_success);
    EXPECT_EQ(
        da_nlls_define_residuals_s(handle, 2, 5, eval_r_s, eval_J_s, nullptr, nullptr),
        da_status_success);
    EXPECT_EQ(da_nlls_define_bounds_s(handle, 2, lower_bounds, upper_bounds),
              da_status_success);

    EXPECT_EQ(da_options_set_int(handle, "print level", 0), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "Storage Order", "Fortran"),
              da_status_success);

    float weights[5] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    EXPECT_EQ(da_nlls_define_weights_s(handle, 5, weights), da_status_success);

    EXPECT_EQ(da_nlls_fit_s(handle, 2, x, params), da_status_success);

    da_int dim = 100;
    float info[100];
    EXPECT_EQ(da_handle_get_result_s(handle, da_rinfo, &dim, info), da_status_success);

    da_handle_destroy(&handle);
}

#endif
