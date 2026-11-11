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
p <- 3

X <- matrix(c(
    1.43, 0.82, -0.51,
    0.37, -1.24, 0.68,
    -0.89, 1.56, 0.23,
    1.72, -0.34, -1.18,
    -0.55, 0.93, 1.41,
    0.84, -1.47, 0.29,
    1.28, 0.61, -0.74,
    -1.13, 0.44, 1.57
), nrow = n_X, ncol = p, byrow = TRUE)
Y <- matrix(c(
    0.73, -0.58, 1.22,
    -1.34, 0.47, 0.63,
    0.56, 1.18, -0.37,
    -0.72, -1.41, 0.54,
    1.07, 0.34, -1.53,
    -0.48, 0.91, 1.37
), nrow = n_Y, ncol = p, byrow = TRUE)

# ============================================================================
# pairwise_distances — X vs Y
# ============================================================================
test_that("pairwise_distances returns correct shape for X and Y", {
    D <- pairwise_distances(X, Y, metric = "euclidean")
    expect_equal(dim(D), c(n_X, n_Y))
    expect_true(all(D >= 0))
})

test_that("pairwise_distances self-distance (Y=NULL) is square and symmetric", {
    D <- pairwise_distances(X, metric = "euclidean")
    expect_equal(dim(D), c(n_X, n_X))
    expect_equal(D, t(D), tolerance = 1e-10)
    expect_equal(diag(D), rep(0, n_X), tolerance = 1e-10)
})

test_that("pairwise_distances manhattan metric works", {
    D <- pairwise_distances(X, Y, metric = "manhattan")
    expect_equal(dim(D), c(n_X, n_Y))
    expect_true(all(D >= 0))
})

test_that("pairwise_distances cosine metric works", {
    D <- pairwise_distances(X, Y, metric = "cosine")
    expect_equal(dim(D), c(n_X, n_Y))
    expect_true(all(D >= 0))
})

test_that("pairwise_distances sqeuclidean metric works", {
    D_sq <- pairwise_distances(X, Y, metric = "sqeuclidean")
    D_euc <- pairwise_distances(X, Y, metric = "euclidean")
    expect_equal(dim(D_sq), c(n_X, n_Y))
    expect_equal(D_sq, D_euc^2, tolerance = 1e-6)
})

test_that("pairwise_distances minkowski p=1 matches manhattan", {
    D_min <- pairwise_distances(X, Y, metric = "minkowski", p = 1.0)
    D_man <- pairwise_distances(X, Y, metric = "manhattan")
    expect_equal(D_min, D_man, tolerance = 1e-6)
})

test_that("pairwise_distances minkowski p=2 matches euclidean", {
    D_min <- pairwise_distances(X, Y, metric = "minkowski", p = 2.0)
    D_euc <- pairwise_distances(X, Y, metric = "euclidean")
    expect_equal(D_min, D_euc, tolerance = 1e-6)
})

test_that("pairwise_distances accepts data.frame input", {
    df_X <- as.data.frame(X)
    df_Y <- as.data.frame(Y)
    expect_no_error(pairwise_distances(df_X, df_Y, metric = "euclidean"))
})

# ============================================================================
# pairwise_distances — exact expected values
# ============================================================================
test_that("pairwise_distances euclidean gives exact values for X-Y and X-X", {
    x <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, ncol = 2, byrow = TRUE)
    y <- matrix(c(7, 8, 9, 10), nrow = 2, ncol = 2, byrow = TRUE)
    s2 <- sqrt(2)

    D_xy <- pairwise_distances(x, y, metric = "euclidean")
    expect_equal(D_xy, matrix(c(
        6 * s2, 8 * s2,
        4 * s2, 6 * s2,
        2 * s2, 4 * s2
    ), nrow = 3, ncol = 2, byrow = TRUE), tolerance = 1e-10)

    D_xx <- pairwise_distances(x, metric = "euclidean")
    expect_equal(D_xx, matrix(c(
        0, 2 * s2, 4 * s2,
        2 * s2, 0, 2 * s2,
        4 * s2, 2 * s2, 0
    ), nrow = 3, ncol = 3, byrow = TRUE), tolerance = 1e-10)
})

test_that("pairwise_distances sqeuclidean gives exact values for X-Y and X-X", {
    x <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, ncol = 2, byrow = TRUE)
    y <- matrix(c(7, 8, 9, 10), nrow = 2, ncol = 2, byrow = TRUE)

    D_xy <- pairwise_distances(x, y, metric = "sqeuclidean")
    expect_equal(D_xy, matrix(c(
        72, 128,
        32, 72,
        8, 32
    ), nrow = 3, ncol = 2, byrow = TRUE), tolerance = 1e-10)

    D_xx <- pairwise_distances(x, metric = "sqeuclidean")
    expect_equal(D_xx, matrix(c(
        0, 8, 32,
        8, 0, 8,
        32, 8, 0
    ), nrow = 3, ncol = 3, byrow = TRUE), tolerance = 1e-10)
})

test_that("pairwise_distances manhattan gives exact values for X-Y and X-X", {
    x <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, ncol = 2, byrow = TRUE)
    y <- matrix(c(7, 8, 9, 10), nrow = 2, ncol = 2, byrow = TRUE)

    D_xy <- pairwise_distances(x, y, metric = "manhattan")
    expect_equal(D_xy, matrix(c(
        12, 16,
        8, 12,
        4, 8
    ), nrow = 3, ncol = 2, byrow = TRUE), tolerance = 1e-10)

    D_xx <- pairwise_distances(x, metric = "manhattan")
    expect_equal(D_xx, matrix(c(
        0, 4, 8,
        4, 0, 4,
        8, 4, 0
    ), nrow = 3, ncol = 3, byrow = TRUE), tolerance = 1e-10)
})

test_that("pairwise_distances cosine gives exact values for X-Y and X-X", {
    x <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, ncol = 2, byrow = TRUE)
    y <- matrix(c(7, 8, 9, 10), nrow = 2, ncol = 2, byrow = TRUE)

    D_xy <- pairwise_distances(x, y, metric = "cosine")
    expect_equal(D_xy, matrix(c(
        0.0323827276031562, 0.0360073817939262,
        0.0028358795133868, 0.0039858440287971,
        0.0002901915325176, 0.0007248347423349
    ), nrow = 3, ncol = 2, byrow = TRUE), tolerance = 1e-10)

    D_xx <- pairwise_distances(x, metric = "cosine")
    expect_equal(D_xx, matrix(c(
        0, 0.0161300899000926, 0.0265828316664239,
        0.0161300899000926, 0, 0.0013123365234113,
        0.0265828316664239, 0.0013123365234113, 0
    ), nrow = 3, ncol = 3, byrow = TRUE), tolerance = 1e-10)
})

test_that("pairwise_distances minkowski p=4 gives exact values for X-Y and X-X", {
    x <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, ncol = 2, byrow = TRUE)
    y <- matrix(c(7, 8, 9, 10), nrow = 2, ncol = 2, byrow = TRUE)

    D_xy <- pairwise_distances(x, y, metric = "minkowski", p = 4.0)
    expect_equal(D_xy, matrix(c(
        7.135242690016327, 9.513656920021768,
        4.756828460010884, 7.135242690016327,
        2.378414230005442, 4.756828460010884
    ), nrow = 3, ncol = 2, byrow = TRUE), tolerance = 1e-10)

    D_xx <- pairwise_distances(x, metric = "minkowski", p = 4.0)
    expect_equal(D_xx, matrix(c(
        0, 2.378414230005442, 4.756828460010884,
        2.378414230005442, 0, 2.378414230005442,
        4.756828460010884, 2.378414230005442, 0
    ), nrow = 3, ncol = 3, byrow = TRUE), tolerance = 1e-10)
})

# --- Metric alias tests ---

X_pd <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, ncol = 2, byrow = TRUE)

test_that("pairwise_distances l1 equals manhattan", {
    D <- pairwise_distances(X_pd, metric = "l1")
    expect_equal(D, pairwise_distances(X_pd, metric = "manhattan"), tolerance = 1e-10)
    expect_equal(D[1, 2], 4.0, tolerance = 1e-10)
})

test_that("pairwise_distances l2 equals euclidean", {
    D <- pairwise_distances(X_pd, metric = "l2")
    expect_equal(D, pairwise_distances(X_pd, metric = "euclidean"), tolerance = 1e-10)
    expect_equal(D[1, 2], 2.8284271247461903, tolerance = 1e-10)
})

test_that("pairwise_distances cityblock equals manhattan", {
    D <- pairwise_distances(X_pd, metric = "cityblock")
    expect_equal(D, pairwise_distances(X_pd, metric = "l1"), tolerance = 1e-10)
})

test_that("pairwise_distances euclidean_gemm equals euclidean", {
    D <- pairwise_distances(X_pd, metric = "euclidean_gemm")
    expect_equal(D, pairwise_distances(X_pd, metric = "euclidean"), tolerance = 1e-10)
})

test_that("pairwise_distances sqeuclidean_gemm equals sqeuclidean", {
    D <- pairwise_distances(X_pd, metric = "sqeuclidean_gemm")
    expect_equal(D, pairwise_distances(X_pd, metric = "sqeuclidean"), tolerance = 1e-10)
})

test_that("pairwise_distances inner_product matches Python reference", {
    D <- pairwise_distances(X_pd, metric = "inner_product")
    expect_equal(D[1, 1], 5.0, tolerance = 1e-10)
    expect_equal(D[1, 2], 11.0, tolerance = 1e-10)
})

test_that("pairwise_distances errors on invalid metric name", {
    expect_error(
        pairwise_distances(X_pd, metric = "not_a_metric"),
        "Given metric does not exist"
    )
})

test_that("pairwise_distances errors on feature dimension mismatch", {
    Y_bad <- matrix(1:6, nrow = 2, ncol = 3)
    expect_error(
        pairwise_distances(X_pd, Y = Y_bad),
        "same number of features"
    )
})
