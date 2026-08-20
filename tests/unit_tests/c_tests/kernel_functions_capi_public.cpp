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

/*
 * Test the kernel functions C API (double precision).
 * Based on tests/examples/kernel_functions.cpp
 */
TEST(KernelFunctionsCAPI, LinearKernelDouble) {
    // X: 3 samples, 2 features (row-major)
    double X[6] = {1.0, 2.0, -1.0, 0.0, 2.0, -2.0};
    // Y: 2 samples, 2 features (row-major)
    double Y[4] = {0.5, -0.5, 1.0, 3.0};

    da_int m = 3, n = 2, k = 2;
    da_int ldx = k, ldy = k, ldd = n;
    double D[6]; // m x n = 3 x 2

    EXPECT_EQ(da_linear_kernel_d(row_major, m, n, k, X, ldx, Y, ldy, D, ldd),
              da_status_success);
}

TEST(KernelFunctionsCAPI, LinearKernelFloat) {
    float X[6] = {1.0f, 2.0f, -1.0f, 0.0f, 2.0f, -2.0f};
    float Y[4] = {0.5f, -0.5f, 1.0f, 3.0f};

    da_int m = 3, n = 2, k = 2;
    da_int ldx = k, ldy = k, ldd = n;
    float D[6];

    EXPECT_EQ(da_linear_kernel_s(row_major, m, n, k, X, ldx, Y, ldy, D, ldd),
              da_status_success);
}

TEST(KernelFunctionsCAPI, RbfKernelDouble) {
    double X[6] = {1.0, 2.0, -1.0, 0.0, 2.0, -2.0};
    double Y[4] = {0.5, -0.5, 1.0, 3.0};

    da_int m = 3, n = 2, k = 2;
    da_int ldx = k, ldy = k, ldd = n;
    double D[6];
    double gamma = 0.5;

    EXPECT_EQ(da_rbf_kernel_d(row_major, m, n, k, X, ldx, Y, ldy, D, ldd, gamma),
              da_status_success);

    // All values should be in (0, 1]
    for (da_int i = 0; i < m * n; i++) {
        EXPECT_GT(D[i], 0.0);
        EXPECT_LE(D[i], 1.0);
    }
}

TEST(KernelFunctionsCAPI, RbfKernelFloat) {
    float X[6] = {1.0f, 2.0f, -1.0f, 0.0f, 2.0f, -2.0f};
    float Y[4] = {0.5f, -0.5f, 1.0f, 3.0f};

    da_int m = 3, n = 2, k = 2;
    da_int ldx = k, ldy = k, ldd = n;
    float D[6];
    float gamma = 0.5f;

    EXPECT_EQ(da_rbf_kernel_s(row_major, m, n, k, X, ldx, Y, ldy, D, ldd, gamma),
              da_status_success);

    for (da_int i = 0; i < m * n; i++) {
        EXPECT_GT(D[i], 0.0f);
        EXPECT_LE(D[i], 1.0f);
    }
}

TEST(KernelFunctionsCAPI, PolynomialKernelDouble) {
    double X[6] = {1.0, 2.0, -1.0, 0.0, 2.0, -2.0};
    double Y[4] = {0.5, -0.5, 1.0, 3.0};

    da_int m = 3, n = 2, k = 2;
    da_int ldx = k, ldy = k, ldd = n;
    double D[6];
    double gamma = 1.0, coef0 = 0.0;
    da_int degree = 3;

    EXPECT_EQ(da_polynomial_kernel_d(row_major, m, n, k, X, ldx, Y, ldy, D, ldd, gamma,
                                     degree, coef0),
              da_status_success);
}

TEST(KernelFunctionsCAPI, PolynomialKernelFloat) {
    float X[6] = {1.0f, 2.0f, -1.0f, 0.0f, 2.0f, -2.0f};
    float Y[4] = {0.5f, -0.5f, 1.0f, 3.0f};

    da_int m = 3, n = 2, k = 2;
    da_int ldx = k, ldy = k, ldd = n;
    float D[6];
    float gamma = 1.0f, coef0 = 0.0f;
    da_int degree = 3;

    EXPECT_EQ(da_polynomial_kernel_s(row_major, m, n, k, X, ldx, Y, ldy, D, ldd, gamma,
                                     degree, coef0),
              da_status_success);
}

TEST(KernelFunctionsCAPI, SigmoidKernelDouble) {
    double X[6] = {1.0, 2.0, -1.0, 0.0, 2.0, -2.0};
    double Y[4] = {0.5, -0.5, 1.0, 3.0};

    da_int m = 3, n = 2, k = 2;
    da_int ldx = k, ldy = k, ldd = n;
    double D[6];
    double gamma = 0.1, coef0 = 0.0;

    EXPECT_EQ(
        da_sigmoid_kernel_d(row_major, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, coef0),
        da_status_success);
}

TEST(KernelFunctionsCAPI, SigmoidKernelFloat) {
    float X[6] = {1.0f, 2.0f, -1.0f, 0.0f, 2.0f, -2.0f};
    float Y[4] = {0.5f, -0.5f, 1.0f, 3.0f};

    da_int m = 3, n = 2, k = 2;
    da_int ldx = k, ldy = k, ldd = n;
    float D[6];
    float gamma = 0.1f, coef0 = 0.0f;

    EXPECT_EQ(
        da_sigmoid_kernel_s(row_major, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, coef0),
        da_status_success);
}
