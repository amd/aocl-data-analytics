# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

library(testthat)
library(aoclda)

n_X <- 8
n_Y <- 6
p <- 4

X <- matrix(c(
    1.43, 0.82, -0.51, 1.17,
    0.37, -1.24, 0.68, -0.43,
    -0.89, 1.56, 0.23, -0.71,
    1.72, -0.34, -1.18, 0.52,
    -0.55, 0.93, 1.41, -1.06,
    0.84, -1.47, 0.29, 0.78,
    1.28, 0.61, -0.74, 1.34,
    -1.13, 0.44, 1.57, -0.92
), nrow = n_X, ncol = p, byrow = TRUE)
Y <- matrix(c(
    0.73, -0.58, 1.22, -0.81,
    -1.34, 0.47, 0.63, -0.94,
    0.56, 1.18, -0.37, 0.85,
    -0.72, -1.41, 0.54, 1.29,
    1.07, 0.34, -1.53, 0.66,
    -0.48, 0.91, 1.37, -0.57
), nrow = n_Y, ncol = p, byrow = TRUE)

# ============================================================================
# rbf_kernel
# ============================================================================
test_that("rbf_kernel X vs Y returns correct shape", {
    K <- rbf_kernel(X, Y, gamma = 1.0)
    expect_equal(dim(K), c(n_X, n_Y))
    expect_true(all(K >= 0))
    expect_true(all(K <= 1))
})

test_that("rbf_kernel self (Y=NULL) is square, symmetric and diag=1", {
    K <- rbf_kernel(X, gamma = 0.5)
    expect_equal(dim(K), c(n_X, n_X))
    expect_equal(K, t(K), tolerance = 1e-10)
    expect_equal(diag(K), rep(1, n_X), tolerance = 1e-10)
})

test_that("rbf_kernel gamma=0 gives all ones", {
    K <- rbf_kernel(X, Y, gamma = 0.0)
    expect_true(all(abs(K - 1) < 1e-10))
})

test_that("rbf_kernel gives exact expected values", {
    x <- matrix(c(10, 10, 11, 11), nrow = 2, ncol = 2, byrow = TRUE)
    y <- matrix(c(10.1, 10.1, 11.1, 11.1), nrow = 2, ncol = 2, byrow = TRUE)

    K_xy <- rbf_kernel(x, y, gamma = 1.0)
    expect_equal(K_xy, matrix(c(
        0.9801986733067731, 0.08892161745938744,
        0.1978986990836138, 0.9801986733067453
    ), nrow = 2, ncol = 2, byrow = TRUE), tolerance = 1e-10)

    K_xx <- rbf_kernel(x, gamma = 1.0)
    expect_equal(K_xx, matrix(c(
        1.0, 0.1353352832366127,
        0.1353352832366127, 1.0
    ), nrow = 2, ncol = 2, byrow = TRUE), tolerance = 1e-10)
})

# ============================================================================
# linear_kernel
# ============================================================================
test_that("linear_kernel X vs Y returns correct shape", {
    K <- linear_kernel(X, Y)
    expect_equal(dim(K), c(n_X, n_Y))
})

test_that("linear_kernel self (Y=NULL) matches X %*% t(X)", {
    K <- linear_kernel(X)
    K_ref <- X %*% t(X)
    expect_equal(dim(K), c(n_X, n_X))
    expect_equal(K, K_ref, tolerance = 1e-5)
})

test_that("linear_kernel accepts data.frame input", {
    expect_no_error(linear_kernel(as.data.frame(X), as.data.frame(Y)))
})

test_that("linear_kernel gives exact expected values", {
    x <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, ncol = 2, byrow = TRUE)
    y <- matrix(c(7, 8, 9, 10), nrow = 2, ncol = 2, byrow = TRUE)

    K_xy <- linear_kernel(x, y)
    expect_equal(K_xy, matrix(c(
        23.0, 29.0,
        53.0, 67.0,
        83.0, 105.0
    ), nrow = 3, ncol = 2, byrow = TRUE), tolerance = 1e-10)

    K_xx <- linear_kernel(x)
    expect_equal(K_xx, matrix(c(
        5.0, 11.0, 17.0,
        11.0, 25.0, 39.0,
        17.0, 39.0, 61.0
    ), nrow = 3, ncol = 3, byrow = TRUE), tolerance = 1e-10)
})

# ============================================================================
# polynomial_kernel
# ============================================================================
test_that("polynomial_kernel X vs Y returns correct shape", {
    K <- polynomial_kernel(X, Y, degree = 3, gamma = 1.0, coef0 = 1.0)
    expect_equal(dim(K), c(n_X, n_Y))
})

test_that("polynomial_kernel self (Y=NULL) is symmetric", {
    K <- polynomial_kernel(X, degree = 2, gamma = 0.5, coef0 = 0.0)
    expect_equal(dim(K), c(n_X, n_X))
    expect_equal(K, t(K), tolerance = 1e-10)
})

test_that("polynomial_kernel different degrees run without error", {
    for (d in c(1L, 2L, 3L, 4L)) {
        expect_no_error(polynomial_kernel(X, Y, degree = d))
    }
})

test_that("polynomial_kernel gives exact expected values", {
    x <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, ncol = 2, byrow = TRUE)
    y <- matrix(c(7, 8, 9, 10), nrow = 2, ncol = 2, byrow = TRUE)

    K_xy <- polynomial_kernel(x, y, gamma = 0.25, degree = 2, coef0 = 0.1)
    expect_equal(K_xy, matrix(c(
        34.2225, 54.0225,
        178.2225, 283.9225,
        434.7225, 694.3225
    ), nrow = 3, ncol = 2, byrow = TRUE), tolerance = 1e-10)

    K_xx <- polynomial_kernel(x, gamma = 0.25, degree = 2, coef0 = 0.1)
    expect_equal(K_xx, matrix(c(
        1.8225, 8.1225, 18.9225,
        8.1225, 40.3225, 97.0225,
        18.9225, 97.0225, 235.6225
    ), nrow = 3, ncol = 3, byrow = TRUE), tolerance = 1e-10)
})

# ============================================================================
# sigmoid_kernel
# ============================================================================
test_that("sigmoid_kernel X vs Y returns correct shape", {
    K <- sigmoid_kernel(X, Y, gamma = 0.1, coef0 = 0.0)
    expect_equal(dim(K), c(n_X, n_Y))
})

test_that("sigmoid_kernel self (Y=NULL) is square", {
    K <- sigmoid_kernel(X, gamma = 0.1, coef0 = 0.0)
    expect_equal(dim(K), c(n_X, n_X))
})

test_that("sigmoid_kernel gives exact expected values", {
    x <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, ncol = 2, byrow = TRUE)
    y <- matrix(c(7, 8, 9, 10), nrow = 2, ncol = 2, byrow = TRUE)

    K_xy <- sigmoid_kernel(x, y, gamma = 0.01, coef0 = 0.01)
    expect_equal(K_xy, matrix(c(
        0.23549574953849797, 0.2913126124515909,
        0.49298796667532435, 0.5915193954318164,
        0.6858090622290945, 0.7856638590269437
    ), nrow = 3, ncol = 2, byrow = TRUE), tolerance = 1e-10)

    K_xx <- sigmoid_kernel(x, gamma = 0.01, coef0 = 0.01)
    expect_equal(K_xx, matrix(c(
        0.0599281035291435, 0.11942729853438588, 0.1780808681173302,
        0.11942729853438588, 0.25429553262639115, 0.3799489622552249,
        0.1780808681173302, 0.3799489622552249, 0.5511280285381469
    ), nrow = 3, ncol = 3, byrow = TRUE), tolerance = 1e-10)
})

test_that("kernel functions error on mismatched feature dimensions", {
    X <- matrix(1:6, nrow = 2, ncol = 3)
    Y_bad <- matrix(1:4, nrow = 2, ncol = 2)
    expect_error(rbf_kernel(X, Y_bad), "same number of features")
    expect_error(linear_kernel(X, Y_bad), "same number of features")
    expect_error(polynomial_kernel(X, Y_bad), "same number of features")
    expect_error(sigmoid_kernel(X, Y_bad), "same number of features")
})
