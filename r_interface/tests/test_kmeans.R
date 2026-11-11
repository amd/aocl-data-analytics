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

tol <- .Machine$double.eps * 1000

x_train <- matrix(c(
    2.0, 1.0,
    -1.0, -2.0,
    3.0, 2.0,
    2.0, 3.0,
    -3.0, -2.0,
    -2.0, -1.0,
    -2.0, -3.0,
    1.0, 2.0
), nrow = 8, ncol = 2, byrow = TRUE)

initial_centres <- matrix(c(
    1.0, 1.0,
    -3.0, -3.0
), nrow = 2, ncol = 2, byrow = TRUE)

x_test <- matrix(c(
    0.0, 1.0,
    0.0, -1.0
), nrow = 2, ncol = 2, byrow = TRUE)

expected_centres <- matrix(c(
    2.0, 2.0,
    -2.0, -2.0
), nrow = 2, ncol = 2, byrow = TRUE)

expected_train_labels <- c(0L, 1L, 0L, 0L, 1L, 1L, 1L, 0L)

expected_test_distances <- matrix(c(
    2.23606797749979, 3.605551275463989,
    3.605551275463989, 2.23606797749979
), nrow = 2, ncol = 2, byrow = TRUE)

expected_test_labels <- c(0L, 1L)

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------
test_that("kmeans constructor defaults work", {
    m <- kmeans$new()
    expect_s3_class(m, "kmeans")
})

test_that("kmeans constructor custom params accepted", {
    m <- kmeans$new(
        n_clusters = 2, initialization_method = "random",
        C = initial_centres, n_init = 5, max_iter = 100,
        seed = 23, algorithm = "lloyd", tol = 1e-4, check_data = TRUE
    )
    expect_s3_class(m, "kmeans")
})

test_that("kmeans public API returns expected fitted values", {
    m <- kmeans$new(
        n_clusters = 2L,
        C = initial_centres,
        tol = 1.0e-4,
        seed = 23L
    )
    m$fit(x_train)

    expect_equal(m$cluster_centres, expected_centres, tolerance = tol)
    expect_equal(m$labels, expected_train_labels)
    expect_true(is.numeric(m$inertia) && m$inertia >= 0)
    expect_lte(m$n_iter, 1L)
    expect_equal(m$n_samples, 8L)
    expect_equal(m$n_features, 2L)
    expect_equal(m$n_clusters, 2L)
    expect_equal(m$transform(x_test), expected_test_distances, tolerance = tol)
    expect_equal(m$predict(x_test), expected_test_labels)
    expect_true(m$lp_n_iter >= 0)
})

# --- Algorithm and init variant tests ---

test_that("kmeans init=random", {
    m <- kmeans$new(n_clusters = 2L, initialization_method = "random",
                    seed = 42L, algorithm = "lloyd")
    m$fit(x_train)
    expect_equal(m$labels, c(0L, 1L, 0L, 0L, 1L, 1L, 1L, 0L))
    expect_equal(m$cluster_centres,
                 matrix(c(2, 2, -2, -2), nrow = 2, byrow = TRUE),
                 tolerance = 1e-6)
})

test_that("kmeans init=random partitions", {
    m <- kmeans$new(n_clusters = 2L, initialization_method = "random partitions",
                    seed = 42L, algorithm = "lloyd")
    m$fit(x_train)
    expect_equal(m$labels, c(0L, 1L, 0L, 0L, 1L, 1L, 1L, 0L))
})

test_that("kmeans algorithm=macqueen", {
    m <- kmeans$new(n_clusters = 2L, seed = 42L, algorithm = "macqueen")
    m$fit(x_train)
    centres_sorted <- m$cluster_centres[order(m$cluster_centres[, 1]), ]
    expect_equal(centres_sorted[1, ], c(-2.0, -2.0), tolerance = 1e-6)
    expect_equal(centres_sorted[2, ], c(2.0, 2.0), tolerance = 1e-6)
})

test_that("kmeans algorithm=hartigan-wong runs without error", {
    m <- kmeans$new(n_clusters = 2L, seed = 42L, algorithm = "hartigan-wong")
    m$fit(x_train)
    centres_sorted <- m$cluster_centres[order(m$cluster_centres[, 1]), ]
    expect_equal(centres_sorted[1, ], c(-2.0, -2.0), tolerance = 1e-6)
    expect_equal(centres_sorted[2, ], c(2.0, 2.0), tolerance = 1e-6)
})

test_that("kmeans cosine distance produces unit-normalized centres", {
    m <- kmeans$new(n_clusters = 2L, seed = 42L, distance = "cosine",
                    algorithm = "lloyd")
    m$fit(x_train)
    norms <- apply(m$cluster_centres, 1, function(r) sqrt(sum(r^2)))
    expect_equal(norms, c(1.0, 1.0), tolerance = 1e-6)
})

test_that("kmeans algorithm=elkan matches lloyd results", {
    m <- kmeans$new(n_clusters = 2L, C = initial_centres, algorithm = "elkan")
    m$fit(x_train)
    expect_equal(m$cluster_centres, expected_centres, tolerance = tol)
    expect_equal(m$labels, expected_train_labels)
})