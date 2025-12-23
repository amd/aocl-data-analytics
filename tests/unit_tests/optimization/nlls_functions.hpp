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

#pragma once
#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <math.h>
#include <vector>

namespace template_nlls_cb_errors {
template <typename T>
da_int eval_r_fail([[maybe_unused]] da_int n, [[maybe_unused]] da_int m,
                   [[maybe_unused]] void *params, [[maybe_unused]] T const *x,
                   [[maybe_unused]] T *r) {
    return 1; // fail...
}
} // namespace template_nlls_cb_errors

namespace template_nlls_example_box_c {

template <typename T> struct params_type {
    T *t; // The m data points t_i
    T *y; // The m data points y_i
    da_int fcnt{100000}, jcnt{100000};
};

// Calculate r_i(x; t_i, y_i) = x_1 e^(x_2 * t_i) - y_i
template <typename T>
da_int eval_r([[maybe_unused]] da_int n, [[maybe_unused]] da_int m,
              [[maybe_unused]] void *params, [[maybe_unused]] T const *x,
              [[maybe_unused]] T *r) {
    T x1 = x[0];
    T x2 = x[1];
    T const *t = ((struct params_type<T> *)params)->t;
    T const *y = ((struct params_type<T> *)params)->y;
    static da_int count_down{0};
    da_int fcnt = ((struct params_type<T> *)params)->fcnt;

    if (fcnt >= 0) {
        count_down = fcnt;
        ((struct params_type<T> *)params)->fcnt = -1;
    }
    if (count_down-- <= 0) {
        return 1;
    }
    for (da_int i = 0; i < m; i++)
        r[i] = x1 * exp(x2 * t[i]) - y[i];

    return 0; // Success
}

// Calculate:
// J_i1 = e^(x_2 * t_i)
// J_i2 = t_i x_1 e^(x_2 * t_i)
template <typename T>
da_int eval_J([[maybe_unused]] da_int n, da_int m, void *params, T const *x, T *J) {
    T x1 = x[0];
    T x2 = x[1];
    T const *t = ((struct params_type<T> *)params)->t;

    for (da_int i = 0; i < m; i++) {
        J[0 * m + i] = exp(x2 * t[i]);             // J_i1
        J[1 * m + i] = t[i] * x1 * exp(x2 * t[i]); // J_i2
    }

    return 0; // Success
}

// User Stop...
template <typename T>
da_int eval_J_wrong([[maybe_unused]] da_int n, da_int m, void *params, T const *x, T *J) {
    T x1 = x[0];
    T x2 = x[1];
    T const *t = ((struct params_type<T> *)params)->t;
    static da_int count_down{0};
    da_int jcnt = ((struct params_type<T> *)params)->jcnt;

    if (jcnt >= 0) {
        count_down = jcnt;
        ((struct params_type<T> *)params)->jcnt = -1;
    }
    if (count_down-- <= 0) {
        return 1;
    }
    for (da_int i = 0; i < m; i++) {
        J[0 * m + i] = exp(x2 * t[i]);
        J[1 * m + i] = t[i] * x1 * exp(x2 * t[i]);
    }
    return 0; // Success
}

// Num difficulties...
template <typename T>
da_int eval_J_bad([[maybe_unused]] da_int n, da_int m, void *params, T const *x, T *J) {
    T x1 = x[0];
    T x2 = x[1];
    T const *t = ((struct params_type<T> *)params)->t;

    for (da_int i = 0; i < m; i++) {
        J[0 * m + i] = exp(x2 * t[i]) + x2 * x2;
        J[1 * m + i] = t[i] * x1 * exp(x2 * t[i]) + x1 * x2;
    }
    return 0; // Success
}

// Calculate:
// HF = sum_i r_i H_i
// Where H_i = [ 1                t_i e^(x_2 t_i)    ]
//             [ t_i e^(x_2 t_i)  t_i^2 e^(x_2 t_i)  ]
template <typename T>
da_int eval_HF(da_int n, da_int m, void *params, T const *x, T const *r, T *HF) {
    T x1 = x[0];
    T x2 = x[1];
    T const *t = ((struct params_type<T> *)params)->t;

    for (da_int i = 0; i < n * n; i++)
        HF[i] = T(0);
    for (da_int i = 0; i < m; i++) {
        HF[0] += T(0);                                             // H_11
        HF[1] += r[i] * t[i] * exp(x2 * t[i]);                     // H_21
        HF[1 * n + 1] += r[i] * t[i] * t[i] * x1 * exp(x2 * t[i]); // H_22
    }
    HF[1 * n + 0] = HF[1]; // H_12 by symmetry of Hessian

    return 0; // Success
}
template <typename T> void driver(void) {
    T t[5]{1.0, 2.0, 4.0, 5.0, 8.0};
    T y[5]{3.0, 4.0, 6.0, 11.0, 20.0};
    const struct params_type<T> udata = {
        t, y
    };

    const da_int n_coef = 2;
    const da_int n_res = 5;
    T coef[n_coef]{1.0, 0.15};
    const T coef_exp[n_coef]{2.541046, 0.2595048};

    T blx[2]{0.0, 0.0};
    T bux[2]{3.0, 10.0};
    T tol{1.0e-2};

    std::cout << "Driver(T=" << typeid(T).name() << ")\n";

    // Initialize handle for nonlinear regression
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_nlls), da_status_success);
    EXPECT_EQ(da_nlls_define_residuals(handle, n_coef, n_res, eval_r<T>, nullptr, nullptr,
                                       nullptr),
              da_status_success);
    EXPECT_EQ(da_nlls_define_bounds(handle, n_coef, blx, bux), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "print options", "yes"), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "storage order", "fortran"),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "print level", (da_int)3), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "ralfit iteration limit", (da_int)300),
              da_status_success);
    if constexpr (std::is_same_v<T, float>) {
        EXPECT_EQ(da_options_set_real_s(handle, "finite differences step", 1e-3),
                  da_status_success);
        EXPECT_EQ(da_options_set_real_s(handle, "ralfit convergence abs tol grd", 1e-8f),
                  da_status_success);
        EXPECT_EQ(da_nlls_fit_s(handle, n_coef, coef, (void *)&udata), da_status_success);
    } else {
        EXPECT_EQ(da_options_set_real_d(handle, "finite differences step", 1e-6f),
                  da_status_success);

        EXPECT_EQ(da_nlls_fit_d(handle, n_coef, coef, (void *)&udata), da_status_success);
    }

    EXPECT_NEAR(coef[0], coef_exp[0], tol);
    EXPECT_NEAR(coef[1], coef_exp[1], tol);

    // Get info out of handle
    std::vector<T> info(100);
    da_int size = info.size();
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_rinfo, &size, info.data()),
              da_status_success);
    if constexpr (std::is_same_v<T, float>) {
        tol = 5.0e-3f;
    } else {
        tol = 1.0e-4;
    }

    EXPECT_LT(info[da_optim_info_t::info_objective], T(2.3));
    EXPECT_LT(info[da_optim_info_t::info_grad_norm], tol);
    EXPECT_GT(info[da_optim_info_t::info_nevalf], T(1));
    EXPECT_GT(info[da_optim_info_t::info_nevalfd], T(3));

    da_handle_destroy(&handle);
}

} // namespace template_nlls_example_box_c

namespace template_lm_example_c {

template <typename T> struct usertype {
    T *sigma;
    T *y;
};

template <typename T>
da_int eval_r([[maybe_unused]] da_int n, da_int m, void *params, T const *x, T *r) {
    T *y = ((struct usertype<T> *)params)->y;
    T *sigma = ((struct usertype<T> *)params)->sigma;
    T A{x[0]};
    T lambda{x[1]};
    T b{x[2]};

    for (da_int i = 0; i < m; i++) {
        /* Model Yi = A * exp(-lambda * i) + b */
        T t = i;
        T Yi = A * exp(-lambda * t) + b;
        r[i] = (Yi - y[i]) / sigma[i];
    }
    return 0;
}

template <typename T> da_int eval_J(da_int n, da_int m, void *params, T const *x, T *J) {
    T *sigma = ((struct usertype<T> *)params)->sigma;
    T A{x[0]};
    T lambda{x[1]};

    for (da_int i = 0; i < m; i++) {
        /* Jacobian matrix J(i,j) = dfi / dxj, */
        /* where fi = (Yi - yi)/sigma[i],      */
        /*       Yi = A * exp(-lambda * i) + b  */
        /* and the xj are the parameters (A,lambda,b) */
        T t = i;
        T s = sigma[i];
        T e = exp(-lambda * t);
        J[n * i + 0] = e / s;
        J[n * i + 1] = -t * A * e / s;
        J[n * i + 2] = 1 / s;
    }
    return 0;
}

template <typename T>
da_int eval_J_bad(da_int n, da_int m, void *params, T const *x, T *J) {
    T *sigma = ((struct usertype<T> *)params)->sigma;
    T A{x[0]};
    T lambda{x[1]};

    for (da_int i = 0; i < m; i++) {
        /* Jacobian matrix J(i,j) = dfi / dxj, */
        /* where fi = (Yi - yi)/sigma[i],      */
        /*       Yi = A * exp(-lambda * i) + b  */
        /* and the xj are the parameters (A,lambda,b) */
        T t = i;
        T s = sigma[i];
        T e = exp(-lambda * t);
        J[n * i + 0] = -e / s;
        J[n * i + 1] = -t * A * e / s;
        J[n * i + 2] = 1 / s;
    }
    return 0;
}

template <typename T> void driver(void) {
    // Data to be fitted
    const da_int m = 40;
    const da_int n = 3;
    const T rnorm[m]{
        0.042609947,  -0.022738876, 0.036553029,  0.025512666,  0.086793270,
        0.047511025,  -0.119396222, -0.042148599, -0.060072244, 0.034911810,
        -0.101209931, -0.103685375, 0.245487401,  -0.038353027, -0.119823715,
        -0.262366501, -0.191863895, -0.015469065, -0.200587427, 0.029074121,
        -0.231842121, 0.056358818,  -0.035592133, -0.105945032, -0.132918722,
        -0.040054318, 0.060915270,  0.041010165,  0.087690256,  0.041471613,
        -0.015124534, 0.090526818,  -0.086582542, -0.026412243, 0.005523387,
        0.006404224,  -0.030465898, 0.097183478,  0.136050209,  -0.038862787};
    T sigma[m];
    T y[m];
    /* Model
     * for (i = 0; i < n; i++)
     *   T t = i;
     *   sigma[i] = 0.1;
     *   y[i] = 1 + 5 * exp (-sigma[i] * t) + rnorm(0.1);
     *   A = amplitude = 5.0
     *   sigma = lambda = 0.1
     *   b = intercept = 1.0
     */
    const T Amplitude{5};
    const T lambda{0.1};
    const T intercept{1};
    for (da_int i = 0; i < m; ++i) {
        T t = T(i);
        sigma[i] = lambda;
        y[i] = intercept + Amplitude * std::exp(-sigma[i] * t) + rnorm[i];
    }

    struct usertype<T> params {
        sigma, y
    };

    T x[n]{3.0, 0.1, 1.0};
    T gtol{1.0e-3};

    std::cout << "Driver(T=" << typeid(T).name() << ")\n";

    da_handle handle{nullptr};
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_type::da_handle_nlls),
              da_status_success);
    EXPECT_EQ(
        da_nlls_define_residuals(handle, n, m, eval_r<T>, eval_J<T>, nullptr, nullptr),
        da_status_success);
    EXPECT_EQ(da_options_set(handle, "ralfit model", "gauss-newton"), da_status_success);
    EXPECT_EQ(da_options_set(handle, "ralfit nlls method", "more-sorensen"),
              da_status_success);
    EXPECT_EQ(da_options_set(handle, "Storage Order", "C"), da_status_success);
    EXPECT_EQ(da_options_set(handle, "print level", da_int(2)), da_status_success);
    EXPECT_EQ(da_options_set(handle, "check derivatives", "yes"), da_status_success);
    if constexpr (std::is_same_v<T, float>) {
        gtol = 0.02f;
        EXPECT_EQ(da_options_set(handle, "derivative test tol", T(5.0e-2)),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle, "finite differences step", T(1.0e-4)),
                  da_status_success);
    } else {
        EXPECT_EQ(da_options_set(handle, "derivative test tol", T(9.0e-5)),
                  da_status_success);
    }
    EXPECT_EQ(da_nlls_fit(handle, n, x, &params), da_status_success);
    // Check output
    std::vector<T> info(100);
    da_int dim{100};
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_rinfo, &dim, info.data()),
              da_status_success);

    EXPECT_GE(info[da_optim_info_t::info_iter], T(5.0));
    EXPECT_LE(info[da_optim_info_t::info_objective], T(25.0));
    EXPECT_LE(info[da_optim_info_t::info_grad_norm], gtol);

    // wrong query...
    T result[2];
    EXPECT_EQ(
        da_handle_get_result(handle, da_result::da_pca_total_variance, &dim, result),
        da_status_unknown_query);

    // Check solution point
    std::cout << "Amplitude A  = " << x[0] << std::endl;
    std::cout << "sigma/lambda = " << x[1] << std::endl;
    std::cout << "intercept b  = " << x[2] << std::endl;

    EXPECT_LE(std::abs(x[0] - Amplitude), T(0.1));
    EXPECT_LE(std::abs(x[1] - lambda), T(0.01));
    EXPECT_LE(std::abs(x[2] - intercept), T(0.1));

    // solve again without initial guess (only for double)
    EXPECT_EQ(da_options_set(handle, "check derivatives", "no"), da_status_success);
    if constexpr (std::is_same_v<T, double>) {
        EXPECT_EQ(da_nlls_fit(handle, 0, (T *)nullptr, &params), da_status_success);
    }

    // solve again using fd
    EXPECT_EQ(
        da_nlls_define_residuals(handle, n, m, eval_r<T>, nullptr, nullptr, nullptr),
        da_status_success);
    if constexpr (std::is_same_v<T, float>) {
        x[0] = 4.0f;
        x[1] = 0.1f;
        x[2] = 1.0f;
        EXPECT_EQ(da_options_set(handle, "finite differences step", 1.0e-3f),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle, "ralfit convergence rel tol grd", 5.0e-6f),
                  da_status_success);
        gtol = 0.1f;
    } else {
        x[0] = 1.0;
        x[1] = 0.0;
        x[2] = 0.0;
        EXPECT_EQ(da_options_set(handle, "finite differences step", T(1.0e-7)),
                  da_status_success);
    }
    EXPECT_EQ(da_nlls_fit(handle, n, x, &params), da_status_success);
    // Check output
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_rinfo, &dim, info.data()),
              da_status_success);

    EXPECT_GE(info[da_optim_info_t::info_iter], T(5.0));
    EXPECT_LE(info[da_optim_info_t::info_objective], T(25.0));
    EXPECT_LE(info[da_optim_info_t::info_grad_norm], 2.f * gtol);

    // Check solution point
    std::cout << "FD: Amplitude A  = " << x[0] << std::endl;
    std::cout << "FD: sigma/lambda = " << x[1] << std::endl;
    std::cout << "FD: intercept b  = " << x[2] << std::endl;

    EXPECT_LE(std::abs(x[0] - Amplitude), T(0.1));
    EXPECT_LE(std::abs(x[1] - lambda), T(0.01));
    EXPECT_LE(std::abs(x[2] - intercept), T(0.1));

    // solve again using fd (with Fortran storage scheme)
    std::cout << "\nsolve again using fd (with Fortran storage scheme)\n";
    EXPECT_EQ(
        da_nlls_define_residuals(handle, n, m, eval_r<T>, nullptr, nullptr, nullptr),
        da_status_success);
    EXPECT_EQ(da_options_set(handle, "storage order", "Fortran"), da_status_success);
    if constexpr (std::is_same_v<T, float>) {
        x[0] = 4.0f;
        x[1] = 0.1f;
        x[2] = 1.0f;
        EXPECT_EQ(da_options_set(handle, "finite differences step", 1.0e-3f),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle, "ralfit convergence rel tol grd", 1.0e-8f),
                  da_status_success);
        gtol = 0.1f;
    } else {
        x[0] = 1.0;
        x[1] = 0.0;
        x[2] = 0.0;
        EXPECT_EQ(da_options_set(handle, "finite differences step", T(1.0e-7)),
                  da_status_success);
    }
    EXPECT_EQ(da_nlls_fit(handle, n, x, &params), da_status_success);
    // Check output
    EXPECT_EQ(da_handle_get_result(handle, da_result::da_rinfo, &dim, info.data()),
              da_status_success);

    EXPECT_GE(info[da_optim_info_t::info_iter], T(5.0));
    EXPECT_LE(info[da_optim_info_t::info_objective], T(25.0));
    EXPECT_LE(info[da_optim_info_t::info_grad_norm], gtol);

    // Check solution point
    std::cout << "F/FD: Amplitude A  = " << x[0] << std::endl;
    std::cout << "F/FD: sigma/lambda = " << x[1] << std::endl;
    std::cout << "F/FD: intercept b  = " << x[2] << std::endl;

    EXPECT_LE(std::abs(x[0] - Amplitude), T(0.1));
    EXPECT_LE(std::abs(x[1] - lambda), T(0.01));
    EXPECT_LE(std::abs(x[2] - intercept), T(0.1));

    // Check for errors in eval_j
    EXPECT_EQ(da_options_set(handle, "check derivatives", "yes"), da_status_success);
    EXPECT_EQ(da_nlls_define_residuals(handle, n, m, eval_r<T>, eval_J_bad<T>, nullptr,
                                       nullptr),
              da_status_success);
    EXPECT_EQ(da_nlls_fit(handle, n, x, &params), da_status_bad_derivatives);

    da_handle_destroy(&handle);
}

} // namespace template_lm_example_c

namespace double_nlls_example_box_fortran {
struct udata_t {
    const double *t;
    const double *y;
};

da_int eval_r(da_int n_coef, da_int n_res, void *udata, double const *x, double *r) {
    double x1 = x[0];
    double x2 = x[n_coef - 1];
    double const *t = ((struct udata_t *)udata)->t;
    double const *y = ((struct udata_t *)udata)->y;

    for (da_int i = 0; i < n_res; i++)
        r[i] = x1 * exp(x2 * t[i]) - y[i];

    return 0;
}
} // namespace double_nlls_example_box_fortran