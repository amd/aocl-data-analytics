/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

/* UT are not for the solver but rather to exersize the interface */

#include "nlls_tests.hpp"
#include "aoclda.h"
#include "da_handle.hpp"
#include "utest_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <vector>

namespace {

// FIXME ADD Float version of the tests

/* RALFit examples as test of interface */
TEST(nlls, template_double_nlls_example_box_2d) {
    using namespace template_nlls_example_box_fortran;
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
    EXPECT_EQ(da_options_set_string(handle, "storage scheme", "fortran"),
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

    EXPECT_LT(info[0], 2.3);
    EXPECT_LT(info[1], 1.0e-4);
    EXPECT_GT(info[4], 1);
    EXPECT_GT(info[12], 3);

    da_handle_destroy(&handle);
}

TEST(nlls, template_double_nlls_example_box_c) {
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

    T lower_bounds[n]{0.0, 1.0};
    T upper_bounds[n]{1.0, 10.0};
    T weights[m]{0.1, 0.1, 0.1, 0.1, 0.1};

    da_handle handle{nullptr};
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_type::da_handle_nlls),
              da_status_success);
    EXPECT_EQ(
        da_nlls_define_residuals(handle, n, m, eval_r<T>, eval_J<T>, eval_HF<T>, nullptr),
        da_status_success);
    EXPECT_EQ(da_nlls_define_bounds(handle, n, lower_bounds, upper_bounds),
              da_status_success);
    EXPECT_EQ(da_nlls_define_weights(handle, m, weights), da_status_success);
    EXPECT_EQ(da_options_set(handle, "print level", (da_int)2), da_status_success);
    EXPECT_EQ(da_options_set(handle, "print options", "yes"), da_status_success);
    EXPECT_EQ(da_options_set(handle, "Storage Scheme", "Fortran"), da_status_success);
    EXPECT_EQ(da_nlls_fit(handle, n, x, &params), da_status_success);
    // Check output
    std::vector<T> info(100);
    da_int dim{100};
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_rinfo, &dim, info.data()),
              da_status_success);
    std::vector<T> info_exp(0);
    if (std::is_same_v<T, double>)
        //                   0     1   2  3    4  5  6  7   8   9 10    11
        info_exp.assign({0.779, 6e-6, 24, 0, 212, 0, 0, 0, 33, 12, 0, 5e-6});
    else
        info_exp.assign(da_optim_info_t::info_number, T(0));

    // relaxed lower bounds
    EXPECT_LE(std::abs(info[0] - info_exp[0]), 0.1);
    EXPECT_LE(std::abs(info[1] - info_exp[1]), 0.001);
    EXPECT_LE(std::abs(info[11] - info_exp[11]), 0.001);

    EXPECT_EQ(info[3], 0);
    EXPECT_EQ(info[5], 0);
    EXPECT_EQ(info[6], 0);
    EXPECT_EQ(info[7], 0);
    EXPECT_EQ(info[10], 0);

    EXPECT_GT(info[2], 5);
    EXPECT_GT(info[4], 15);
    EXPECT_GT(info[8], 10);
    EXPECT_GT(info[9], 4);

    // double call Warm start
    EXPECT_EQ(da_options_set(handle, "print options", "no"), da_status_success);
    EXPECT_EQ(da_options_set(handle, "print level", (da_int)5), da_status_success);
    EXPECT_EQ(da_nlls_fit(handle, n, x, &params), da_status_success);
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_rinfo, &dim, info.data()),
              da_status_success);
    EXPECT_EQ(info[da_optim_info_t::info_iter], T(0));
    EXPECT_EQ(info[da_optim_info_t::info_nevalf], T(1));
    EXPECT_EQ(info[da_optim_info_t::info_nevalg], T(1));
    EXPECT_EQ(info[da_optim_info_t::info_nevalh], T(0));
    EXPECT_EQ(info[da_optim_info_t::info_nevalhp], T(0));
    EXPECT_EQ(da_options_set(handle, "print level", (da_int)0), da_status_success);

    // initial x0 not provided
    EXPECT_EQ(da_nlls_fit(handle, 0, x, &params), da_status_success);
    EXPECT_EQ(da_nlls_fit_d(handle, 0, nullptr, &params), da_status_success);
    EXPECT_EQ(da_nlls_fit_d(handle, n, nullptr, &params), da_status_invalid_pointer);
    EXPECT_EQ(da_nlls_fit(handle, n - 1, x, &params), da_status_invalid_array_dimension);

    // call with wrong bounds
    EXPECT_EQ(da_nlls_define_bounds(handle, n, upper_bounds, lower_bounds),
              da_status_success);
    EXPECT_EQ(da_nlls_fit(handle, n, x, &params), da_status_option_invalid_bounds);

    // call with no search space
    EXPECT_EQ(da_nlls_define_bounds(handle, n, lower_bounds, lower_bounds),
              da_status_success);
    EXPECT_EQ(da_nlls_fit(handle, n, x, &params), da_status_success);

    // call with missing bound
    EXPECT_EQ(da_nlls_define_bounds(handle, n, nullptr, upper_bounds), da_status_success);
    EXPECT_EQ(da_nlls_fit(handle, n, x, &params), da_status_success);

    EXPECT_EQ(da_nlls_define_bounds(handle, n, lower_bounds, nullptr), da_status_success);
    EXPECT_EQ(da_nlls_fit(handle, n, x, &params), da_status_success);

    da_handle_destroy(&handle);
}

TEST(nlls, template_double_lm_example_c) {
    using namespace template_lm_example_c;
    using T = double;
    // Data to be fitted
    const da_int m = 40;
    const da_int n = 3;
    const double rnorm[m]{
        0.042609947,  -0.022738876, 0.036553029,  0.025512666,  0.086793270,
        0.047511025,  -0.119396222, -0.042148599, -0.060072244, 0.034911810,
        -0.101209931, -0.103685375, 0.245487401,  -0.038353027, -0.119823715,
        -0.262366501, -0.191863895, -0.015469065, -0.200587427, 0.029074121,
        -0.231842121, 0.056358818,  -0.035592133, -0.105945032, -0.132918722,
        -0.040054318, 0.060915270,  0.041010165,  0.087690256,  0.041471613,
        -0.015124534, 0.090526818,  -0.086582542, -0.026412243, 0.005523387,
        0.006404224,  -0.030465898, 0.097183478,  0.136050209,  -0.038862787};
    double sigma[m];
    double y[m];
    /* Model
     * for (i = 0; i < n; i++)
     *   double t = i;
     *   sigma[i] = 0.1;
     *   y[i] = 1 + 5 * exp (-sigma[i] * t) + rnorm(0.1);
     *   A = amplitude = 5.0
     *   sigma = lambda = 0.1
     *   b = intercept = 1.0
     */
    const double Amplitude{5};
    const double lambda{0.1};
    const double intercept{1};
    for (da_int i = 0; i < m; ++i) {
        double t = double(i);
        sigma[i] = lambda;
        y[i] = intercept + Amplitude * std::exp(-sigma[i] * t) + rnorm[i];
    }

    struct usertype params {
        sigma, y
    };

    double x[n]{1.0, 0.0, 0.0};

    da_handle handle{nullptr};
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_type::da_handle_nlls),
              da_status_success);
    EXPECT_EQ(da_nlls_define_residuals(handle, n, m, eval_r, eval_J, nullptr, nullptr),
              da_status_success);
    EXPECT_EQ(da_options_set(handle, "ralfit model", "gauss-newton"), da_status_success);
    EXPECT_EQ(da_options_set(handle, "ralfit nlls method", "more-sorensen"),
              da_status_success);
    EXPECT_EQ(da_options_set(handle, "Storage Scheme", "C"), da_status_success);
    EXPECT_EQ(da_options_set(handle, "print level", da_int(2)), da_status_success);
    EXPECT_EQ(da_options_set(handle, "check derivatives", "yes"), da_status_success);
    EXPECT_EQ(da_options_set(handle, "derivative test tol", 9.0e-5), da_status_success);
    EXPECT_EQ(da_nlls_fit(handle, n, x, &params), da_status_success);
    // Check output
    std::vector<T> info(100);
    da_int dim{100};
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_rinfo, &dim, info.data()),
              da_status_success);

    EXPECT_GE(info[da_optim_info_t::info_iter], 5.0);
    EXPECT_LE(info[da_optim_info_t::info_objective], 25.0);
    EXPECT_LE(info[da_optim_info_t::info_grad_norm], 1.0e-3);

    // wrong query...
    T result[2];
    EXPECT_EQ(
        da_handle_get_result(handle, da_result::da_pca_total_variance, &dim, result),
        da_status_unknown_query);

    // Check solution point
    std::cout << "Amplitude A  = " << x[0] << std::endl;
    std::cout << "sigma/lambda = " << x[1] << std::endl;
    std::cout << "intercept b  = " << x[2] << std::endl;

    EXPECT_LE(std::abs(x[0] - Amplitude), 0.1);
    EXPECT_LE(std::abs(x[1] - lambda), 0.01);
    EXPECT_LE(std::abs(x[2] - intercept), 0.1);

    // solve again without initial guess
    EXPECT_EQ(da_options_set(handle, "check derivatives", "no"), da_status_success);
    EXPECT_EQ(da_nlls_fit_d(handle, 0, nullptr, &params), da_status_success);

    // solve again using fd
    EXPECT_EQ(da_nlls_define_residuals(handle, n, m, eval_r, nullptr, nullptr, nullptr),
              da_status_success);
    x[0] = 1.0;
    x[1] = 0.0;
    x[2] = 0.0;
    EXPECT_EQ(da_options_set(handle, "finite differences step", 1.0e-7),
              da_status_success);
    EXPECT_EQ(da_nlls_fit(handle, n, x, &params), da_status_success);
    // Check output
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_rinfo, &dim, info.data()),
              da_status_success);

    EXPECT_GE(info[da_optim_info_t::info_iter], 5.0);
    EXPECT_LE(info[da_optim_info_t::info_objective], 25.0);
    EXPECT_LE(info[da_optim_info_t::info_grad_norm], 1.0e-3);

    // Check solution point
    std::cout << "FD: Amplitude A  = " << x[0] << std::endl;
    std::cout << "FD: sigma/lambda = " << x[1] << std::endl;
    std::cout << "FD: intercept b  = " << x[2] << std::endl;

    EXPECT_LE(std::abs(x[0] - Amplitude), 0.1);
    EXPECT_LE(std::abs(x[1] - lambda), 0.01);
    EXPECT_LE(std::abs(x[2] - intercept), 0.1);

    // solve again using fd (with Fortran storage scheme)
    EXPECT_EQ(da_nlls_define_residuals(handle, n, m, eval_r, nullptr, nullptr, nullptr),
              da_status_success);
    x[0] = 1.0;
    x[1] = 0.0;
    x[2] = 0.0;
    EXPECT_EQ(da_options_set(handle, "storage scheme", "Fortran"), da_status_success);
    EXPECT_EQ(da_nlls_fit(handle, n, x, &params), da_status_success);
    // Check output
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_rinfo, &dim, info.data()),
              da_status_success);

    EXPECT_GE(info[da_optim_info_t::info_iter], 5.0);
    EXPECT_LE(info[da_optim_info_t::info_objective], 25.0);
    EXPECT_LE(info[da_optim_info_t::info_grad_norm], 1.0e-3);

    // Check solution point
    std::cout << "F/FD: Amplitude A  = " << x[0] << std::endl;
    std::cout << "F/FD: sigma/lambda = " << x[1] << std::endl;
    std::cout << "F/FD: intercept b  = " << x[2] << std::endl;

    EXPECT_LE(std::abs(x[0] - Amplitude), 0.1);
    EXPECT_LE(std::abs(x[1] - lambda), 0.01);
    EXPECT_LE(std::abs(x[2] - intercept), 0.1);

    // Check for errors in eval_j
    EXPECT_EQ(da_options_set(handle, "check derivatives", "yes"), da_status_success);
    EXPECT_EQ(
        da_nlls_define_residuals(handle, n, m, eval_r, eval_J_bad, nullptr, nullptr),
        da_status_success);
    EXPECT_EQ(da_nlls_fit(handle, n, x, &params), da_status_bad_derivatives);

    da_handle_destroy(&handle);
}

TEST(nlls, ifaceChecks) {
    using namespace template_nlls_example_box_c;
    using T = double;
    da_handle handle{nullptr};
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_type::da_handle_nlls),
              da_status_success);
    // exersize define_residuals
    da_int n = 1;
    da_int m = 1;
    EXPECT_EQ(
        da_nlls_define_residuals(nullptr, n, m, eval_r<T>, eval_J<T>, nullptr, nullptr),
        da_status_handle_not_initialized);

    // get results without training
    da_int dim{2};
    T result[2];
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_rinfo, &dim, result),
              da_status_unknown_query);

    da_int iresult[2];
    EXPECT_EQ(
        da_handle_get_result(handle, da_result::da_pca_total_variance, &dim, iresult),
        da_status_unknown_query);
    // eval_r
    EXPECT_EQ(
        da_nlls_define_residuals(handle, n, m, nullptr, eval_J<T>, nullptr, nullptr),
        da_status_invalid_input);
    n = -1;
    EXPECT_EQ(
        da_nlls_define_residuals(handle, n, m, eval_r<T>, eval_J<T>, nullptr, nullptr),
        da_status_invalid_input);
    n = 1;
    m = -1;
    EXPECT_EQ(
        da_nlls_define_residuals(handle, n, m, eval_r<T>, eval_J<T>, nullptr, nullptr),
        da_status_invalid_input);
    m = 5;
    EXPECT_EQ(
        da_nlls_define_residuals(handle, n, m, eval_r<T>, nullptr, nullptr, nullptr),
        da_status_success);
    n = 2;
    EXPECT_EQ(
        da_nlls_define_residuals(handle, n, m, eval_r<T>, eval_J<T>, nullptr, nullptr),
        da_status_success);

    // exersize define bounds
    std::vector<T> lower_bounds = {0.0, -1.0}, upper_bounds = {1.0, 2.0};
    EXPECT_EQ(da_nlls_define_bounds(nullptr, n, lower_bounds.data(), upper_bounds.data()),
              da_status_handle_not_initialized);
    n = 0; // remove bounds
    EXPECT_EQ(da_nlls_define_bounds(handle, n, lower_bounds.data(), upper_bounds.data()),
              da_status_success);
    n = 1; // wrong size n_coef
    EXPECT_EQ(da_nlls_define_bounds(handle, n, lower_bounds.data(), upper_bounds.data()),
              da_status_invalid_input);
    n = 2;
    EXPECT_EQ(da_nlls_define_bounds(handle, n, nullptr, upper_bounds.data()),
              da_status_success);
    EXPECT_EQ(da_nlls_define_bounds(handle, n, lower_bounds.data(), nullptr),
              da_status_success);

    T weights[5];
    // Exersize weights
    EXPECT_EQ(da_nlls_define_weights(nullptr, m, weights),
              da_status_handle_not_initialized);
    m = 2; // wrong nsamples
    EXPECT_EQ(da_nlls_define_weights(handle, m, weights), da_status_invalid_input);
    m = 5;
    // correct nres but wrong pointer
    EXPECT_EQ(da_nlls_define_weights(handle, m, (double *)nullptr),
              da_status_invalid_pointer);
    // add weights
    EXPECT_EQ(da_nlls_define_weights(handle, m, weights), da_status_success);
    // remove weights
    m = 0;
    EXPECT_EQ(da_nlls_define_weights(handle, m, (double *)nullptr), da_status_success);
    da_handle_destroy(&handle);
}

TEST(nlls, solverCheckX0Rubbish) {
    using namespace template_nlls_example_box_c;
    using namespace template_nlls_cb_errors;
    using T = double;
    da_handle handle{nullptr};
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_type::da_handle_nlls),
              da_status_success);
    // exersize define_residuals
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
    EXPECT_EQ(da_options_set(handle, "Storage Scheme", "Fortran"), da_status_success);
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
    EXPECT_EQ(da_options_set(handle, "Storage Scheme", "Fortran"), da_status_success);
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
    EXPECT_EQ(da_options_set(handle, "Storage Scheme", "Fortran"), da_status_success);
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

TEST(nlls, tamperNllsHandle) {
    using namespace template_nlls_example_box_c;
    da_handle handle_s{nullptr};
    da_handle handle_d{nullptr};
    EXPECT_EQ(da_handle_init<float>(&handle_s, da_handle_type::da_handle_nlls),
              da_status_success);
    EXPECT_EQ(da_handle_init<double>(&handle_d, da_handle_type::da_handle_nlls),
              da_status_success);

    // save...
    da_nlls::nlls<float> *nlls_s = handle_s->nlls_s;
    da_nlls::nlls<double> *nlls_d = handle_d->nlls_d;
    // tamper...
    handle_s->nlls_s = nullptr;
    handle_d->nlls_d = nullptr;

    da_int n{2}, m{5};
    EXPECT_EQ(da_nlls_define_residuals(handle_s, n, m, eval_r<float>, eval_J<float>,
                                       nullptr, nullptr),
              da_status_invalid_handle_type);
    EXPECT_EQ(da_nlls_define_residuals(handle_d, n, m, eval_r<double>, eval_J<double>,
                                       nullptr, nullptr),
              da_status_invalid_handle_type);
    float lower_s[2]{0};
    double lower_d[2]{0};
    EXPECT_EQ(da_nlls_define_bounds(handle_s, n, lower_s, nullptr),
              da_status_invalid_handle_type);
    EXPECT_EQ(da_nlls_define_bounds(handle_d, n, lower_d, nullptr),
              da_status_invalid_handle_type);
    float w_s[5]{0};
    double w_d[5]{0};
    EXPECT_EQ(da_nlls_define_weights(handle_s, m, w_s), da_status_invalid_handle_type);
    EXPECT_EQ(da_nlls_define_weights(handle_d, m, w_d), da_status_invalid_handle_type);
    float x_s[2]{0};
    double x_d[2]{0};
    EXPECT_EQ(da_nlls_fit(handle_s, n, x_s, nullptr), da_status_invalid_handle_type);
    EXPECT_EQ(da_nlls_fit(handle_d, n, x_d, nullptr), da_status_invalid_handle_type);
    // restore...
    handle_s->nlls_s = nlls_s;
    handle_d->nlls_d = nlls_d;
    da_handle_destroy(&handle_s);
    da_handle_destroy(&handle_d);
}

TEST(nlls, tempNotImplemented) {
    using T = float;
    using namespace template_nlls_example_box_c;
    da_handle handle{nullptr};
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_type::da_handle_nlls),
              da_status_success);
    da_int n{2}, m{5};
    EXPECT_EQ(
        da_nlls_define_residuals(handle, n, m, eval_r<T>, eval_J<T>, nullptr, nullptr),
        da_status_success);
    float lower_bounds[2]{0};
    EXPECT_EQ(da_nlls_define_bounds(handle, n, lower_bounds, nullptr), da_status_success);
    float weights[5]{0};
    EXPECT_EQ(da_nlls_define_weights(handle, m, weights), da_status_success);
    float x[2]{0};
    EXPECT_EQ(da_nlls_fit(handle, n, x, nullptr), da_status_not_implemented);
    da_handle_destroy(&handle);
}

} // namespace
