/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

/* UT are not for the solver but rather to exercise the interface */

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "nlls_functions.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <vector>

namespace {

#ifdef NO_FORTRAN
TEST(nlls, not_implemented) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_nlls), da_status_not_implemented);
    da_handle_destroy(&handle);
}

#else

/* RALFit examples as test of interface */
TEST(nlls, double_nlls_example_box_fortran) {
    using namespace double_nlls_example_box_fortran;
    const double t[5]{1.0, 2.0, 4.0, 5.0, 8.0};
    const double y[5]{3.0, 4.0, 6.0, 11.0, 20.0};
    const struct udata_t udata = {t, y};

    const da_int n_coef = 2;
    const da_int n_res = 5;
    double coef[n_coef]{1.0, 0.15};
    const double coef_exp[n_coef]{2.541046, 0.2595048};

    double blx[2]{0.0, 0.0};
    double bux[2]{3.0, 10.0};
    const double tol{1.0e-2};

    // Initialize handle for nonlinear regression
    da_handle handle = nullptr;

    EXPECT_EQ(da_handle_init_d(&handle, da_handle_nlls), da_status_success);
    EXPECT_EQ(da_nlls_define_residuals_d(handle, n_coef, n_res, eval_r, nullptr, nullptr,
                                         nullptr),
              da_status_success);
    EXPECT_EQ(da_nlls_define_bounds_d(handle, n_coef, blx, bux), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "print options", "yes"), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "storage order", "fortran"),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "print level", (da_int)3), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "ralfit iteration limit", (da_int)200),
              da_status_success);
    EXPECT_EQ(da_options_set_real_d(handle, "finite differences step", 1e-6),
              da_status_success);

    EXPECT_EQ(da_nlls_fit_d(handle, n_coef, coef, (void *)&udata), da_status_success);

    EXPECT_NEAR(coef[0], coef_exp[0], tol);
    EXPECT_NEAR(coef[1], coef_exp[1], tol);

    // Get info out of handle
    std::vector<double> info(100);
    da_int size = info.size();
    EXPECT_EQ(da_handle_get_result_d(handle, da_result::da_rinfo, &size, info.data()),
              da_status_success);

    EXPECT_LT(info[da_optim_info_t::info_objective], 2.3);
    EXPECT_LT(info[da_optim_info_t::info_grad_norm], 1.0e-4);
    EXPECT_GT(info[da_optim_info_t::info_nevalf], 1);
    EXPECT_GT(info[da_optim_info_t::info_nevalfd], 3);

    da_handle_destroy(&handle);
}

TEST(nlls, nlls_example_box_c) {
    template_nlls_example_box_c::driver<double>();
    template_nlls_example_box_c::driver<float>();
}

TEST(nlls, lm_example_c) {
    template_lm_example_c::driver<double>();
    // disabled template_lm_example_c::driver<float>();
}

TEST(nlls, ifaceChecks) {
    using namespace template_nlls_example_box_c;
    using T = double;
    using S = float;
    da_handle handle_d{nullptr};
    da_handle handle_s{nullptr};
    EXPECT_EQ(da_handle_init<T>(&handle_d, da_handle_type::da_handle_nlls),
              da_status_success);
    EXPECT_EQ(da_handle_init<S>(&handle_s, da_handle_type::da_handle_nlls),
              da_status_success);
    // exercise define_residuals
    da_int n = 1;
    da_int m = 1;
    EXPECT_EQ(
        da_nlls_define_residuals(nullptr, n, m, eval_r<T>, eval_J<T>, nullptr, nullptr),
        da_status_handle_not_initialized);
    EXPECT_EQ(
        da_nlls_define_residuals(nullptr, n, m, eval_r<S>, eval_J<S>, nullptr, nullptr),
        da_status_handle_not_initialized);

    // get results without training
    da_int dim{2};
    T result_d[2];
    S result_s[2];
    EXPECT_EQ(da_handle_get_result(handle_d, da_result::da_rinfo, &dim, result_d),
              da_status_unknown_query);
    EXPECT_EQ(da_handle_get_result(handle_s, da_result::da_rinfo, &dim, result_s),
              da_status_unknown_query);

    da_int iresult[2];
    EXPECT_EQ(
        da_handle_get_result(handle_d, da_result::da_pca_total_variance, &dim, iresult),
        da_status_unknown_query);
    EXPECT_EQ(
        da_handle_get_result(handle_d, da_result::da_pca_total_variance, &dim, iresult),
        da_status_unknown_query);
    // eval_r
    EXPECT_EQ(
        da_nlls_define_residuals(handle_d, n, m, nullptr, eval_J<T>, nullptr, nullptr),
        da_status_invalid_input);
    EXPECT_EQ(
        da_nlls_define_residuals(handle_s, n, m, nullptr, eval_J<S>, nullptr, nullptr),
        da_status_invalid_input);
    n = -1;
    EXPECT_EQ(
        da_nlls_define_residuals(handle_d, n, m, eval_r<T>, eval_J<T>, nullptr, nullptr),
        da_status_invalid_input);
    EXPECT_EQ(
        da_nlls_define_residuals(handle_s, n, m, eval_r<S>, eval_J<S>, nullptr, nullptr),
        da_status_invalid_input);
    n = 1;
    m = -1;
    EXPECT_EQ(
        da_nlls_define_residuals(handle_d, n, m, eval_r<T>, eval_J<T>, nullptr, nullptr),
        da_status_invalid_input);
    EXPECT_EQ(
        da_nlls_define_residuals(handle_s, n, m, eval_r<S>, eval_J<S>, nullptr, nullptr),
        da_status_invalid_input);
    m = 5;
    EXPECT_EQ(
        da_nlls_define_residuals(handle_d, n, m, eval_r<T>, nullptr, nullptr, nullptr),
        da_status_success);
    EXPECT_EQ(
        da_nlls_define_residuals(handle_s, n, m, eval_r<S>, nullptr, nullptr, nullptr),
        da_status_success);
    n = 2;
    EXPECT_EQ(
        da_nlls_define_residuals(handle_d, n, m, eval_r<T>, eval_J<T>, nullptr, nullptr),
        da_status_success);
    EXPECT_EQ(
        da_nlls_define_residuals(handle_s, n, m, eval_r<S>, eval_J<S>, nullptr, nullptr),
        da_status_success);

    // exercise define bounds
    std::vector<T> lower_bounds_d = {0.0, -1.0}, upper_bounds_d = {1.0, 2.0};
    std::vector<S> lower_bounds_s = {0.0f, -1.0f}, upper_bounds_s = {1.0f, 2.0f};
    EXPECT_EQ(
        da_nlls_define_bounds(nullptr, n, lower_bounds_d.data(), upper_bounds_d.data()),
        da_status_handle_not_initialized);
    n = 0; // remove bounds
    EXPECT_EQ(
        da_nlls_define_bounds(handle_d, n, lower_bounds_d.data(), upper_bounds_d.data()),
        da_status_success);
    n = 1; // wrong size n_coef
    EXPECT_EQ(
        da_nlls_define_bounds(handle_s, n, lower_bounds_s.data(), upper_bounds_s.data()),
        da_status_invalid_input);
    n = 2;
    EXPECT_EQ(da_nlls_define_bounds(handle_d, n, nullptr, upper_bounds_d.data()),
              da_status_success);
    EXPECT_EQ(da_nlls_define_bounds(handle_s, n, lower_bounds_s.data(), nullptr),
              da_status_success);

    T weights_d[5];
    S weights_s[5];
    // Exercise weights
    EXPECT_EQ(da_nlls_define_weights(nullptr, m, weights_d),
              da_status_handle_not_initialized);
    m = 2; // wrong nsamples
    EXPECT_EQ(da_nlls_define_weights(handle_d, m, weights_d), da_status_invalid_input);
    m = 5;
    // correct nres but wrong pointer
    EXPECT_EQ(da_nlls_define_weights(handle_d, m, (T *)nullptr),
              da_status_invalid_pointer);
    // add weights
    EXPECT_EQ(da_nlls_define_weights(handle_d, m, weights_d), da_status_success);
    // remove weights
    m = 0;
    EXPECT_EQ(da_nlls_define_weights(handle_d, m, (T *)nullptr), da_status_success);
    // add weights
    EXPECT_EQ(da_nlls_define_weights(handle_s, m, weights_s), da_status_success);
    // remove weights
    m = 0;
    EXPECT_EQ(da_nlls_define_weights(handle_s, m, (S *)nullptr), da_status_success);
    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}

TEST(nlls, solverCheckX0Rubbish) {
    using namespace template_nlls_example_box_c;
    using namespace template_nlls_cb_errors;
    using T = double;
    da_handle handle{nullptr};
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_type::da_handle_nlls),
              da_status_success);
    // exercise define_residuals
    da_int n = 1;
    da_int m = 1;
    T x[1]{0};
    EXPECT_EQ(da_nlls_define_residuals(handle, n, m, eval_r_fail<T>, eval_J<T>, nullptr,
                                       nullptr),
              da_status_success);
    EXPECT_EQ(da_nlls_fit(handle, n, x, nullptr), da_status_operation_failed);

    T t[]{1.0, 2.0, 4.0, 5.0, 8.0};
    T y[]{3.0, 4.0, 6.0, 11.0, 20.0};

    struct params_type<T> params {
        t, y, 0
    };
    n = 2;
    m = 5;
    T x2[2]{0, 0};
    EXPECT_EQ(da_nlls_define_residuals(
                  handle, n, m, template_nlls_example_box_c::eval_r<T>,
                  template_nlls_example_box_c::eval_J_wrong<T>, eval_HF<T>, nullptr),
              da_status_success);
    EXPECT_EQ(da_options_set(handle, "check derivatives", "yes"), da_status_success);
    EXPECT_EQ(da_nlls_fit(handle, n, x2, &params), da_status_operation_failed);
    da_handle_destroy(&handle);
}

TEST(nlls, solverCheckMaxIt) {
    using namespace template_nlls_example_box_c;
    using T = double;
    // Data to be fitted
    const da_int m = 5;
    const da_int n = 2;
    T t[]{1.0, 2.0, 4.0, 5.0, 8.0};
    T y[]{3.0, 4.0, 6.0, 11.0, 20.0};
    struct params_type<T> params {
        t, y
    };

    // Call fitting routine
    T x[n]{1.0, 1.0}; // Initial guess

    da_handle handle{nullptr};
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_type::da_handle_nlls),
              da_status_success);
    EXPECT_EQ(da_nlls_define_residuals(handle, n, m, eval_r<T>, eval_J_wrong<T>,
                                       eval_HF<T>, nullptr),
              da_status_success);
    EXPECT_EQ(da_options_set(handle, "ralfit iteration limit", da_int(1)),
              da_status_success);
    EXPECT_EQ(da_options_set(handle, "Storage Order", "Fortran"), da_status_success);
    EXPECT_EQ(da_nlls_fit(handle, n, x, &params), da_status_maxit);
    da_handle_destroy(&handle);
}

TEST(nlls, solverCheckUsrStop) {
    using namespace template_nlls_example_box_c;
    using T = double;
    // Data to be fitted
    const da_int m = 5;
    const da_int n = 2;
    T t[]{1.0, 2.0, 4.0, 5.0, 8.0};
    T y[]{3.0, 4.0, 6.0, 11.0, 20.0};
    struct params_type<T> params {
        t, y, 2, 1
    };

    // Call fitting routine
    T x[n]{1.0, 1.0}; // Initial guess

    T lower_bounds[n]{0.0, 1.0};
    T upper_bounds[n]{1.0, 10.0};
    T weights[m]{0.1, 0.1, 0.1, 0.1, 0.1};

    da_handle handle{nullptr};
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_type::da_handle_nlls),
              da_status_success);
    EXPECT_EQ(da_nlls_define_residuals(handle, n, m, eval_r<T>, eval_J_wrong<T>,
                                       eval_HF<T>, nullptr),
              da_status_success);
    EXPECT_EQ(da_nlls_define_bounds(handle, n, lower_bounds, upper_bounds),
              da_status_success);
    EXPECT_EQ(da_nlls_define_weights(handle, m, weights), da_status_success);
    EXPECT_EQ(da_options_set(handle, "ralfit iteration limit", da_int(10)),
              da_status_success);
    EXPECT_EQ(da_options_set(handle, "Storage Order", "Fortran"), da_status_success);
    EXPECT_EQ(da_nlls_fit(handle, n, x, &params), da_status_optimization_usrstop);

    // Check during FD check derivatives
    params.fcnt = 1;
    params.jcnt = 1;
    EXPECT_EQ(da_options_set(handle, "check derivatives", "yes"), da_status_success);
    EXPECT_EQ(da_nlls_fit(handle, n, x, &params), da_status_optimization_usrstop);
    da_handle_destroy(&handle);
}

TEST(nlls, solverCheckNumDifficulties) {
    using namespace template_nlls_example_box_c;
    using T = double;
    // Data to be fitted
    const da_int m = 5;
    const da_int n = 2;
    T t[]{1.0, 2.0, 4.0, 5.0, 8.0};
    T y[]{3.0, 4.0, 6.0, 11.0, 20.0};
    struct params_type<T> params {
        t, y
    };

    // Call fitting routine
    T x[n]{0.5, 0.0}; // Initial guess

    T lower_bounds[n]{0.0, 1.0};
    T upper_bounds[n]{1.0, 10.0};

    da_handle handle{nullptr};
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_type::da_handle_nlls),
              da_status_success);
    EXPECT_EQ(da_nlls_define_residuals(handle, n, m, eval_r<T>, eval_J_bad<T>, eval_HF<T>,
                                       nullptr),
              da_status_success);
    EXPECT_EQ(da_nlls_define_bounds(handle, n, lower_bounds, upper_bounds),
              da_status_success);
    EXPECT_EQ(da_options_set(handle, "Storage Order", "Fortran"), da_status_success);
    EXPECT_EQ(da_nlls_fit(handle, n, x, &params), da_status_numerical_difficulties);
    da_handle_destroy(&handle);
}

TEST(nlls, wrongType) {
    using namespace template_nlls_example_box_c;
    da_handle handle{nullptr};
    EXPECT_EQ(da_handle_init<float>(&handle, da_handle_type::da_handle_nlls),
              da_status_success);
    da_int n{2}, m{5};
    EXPECT_EQ(da_nlls_define_residuals(handle, n, m, eval_r<double>, eval_J<double>,
                                       nullptr, nullptr),
              da_status_wrong_type);
    double lower_bounds[2]{0};
    EXPECT_EQ(da_nlls_define_bounds(handle, n, lower_bounds, nullptr),
              da_status_wrong_type);
    double x[2]{0};
    EXPECT_EQ(da_nlls_fit(handle, n, x, nullptr), da_status_wrong_type);
    da_handle_destroy(&handle);
}

#endif

} // namespace
