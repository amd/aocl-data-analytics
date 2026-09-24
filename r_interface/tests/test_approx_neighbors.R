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

# ============================================================================
# approximate_neighbors
# ============================================================================

tol <- sqrt(.Machine$double.eps) * 100

x_train <- matrix(c(
    10.1, 10.2, 10.4,
    11.1, 12.2, 11.4,
    9.1, 9.2, 9.4,
    -20.2, -20.3, -20.3,
    -21.2, -21.3, -21.3,
    -19.2, -19.3, -18.3,
    5.0, -10.1, 5.2,
    5.0, -11.1, 11.2,
    5.0, -9.1, 9.2
), nrow = 9, ncol = 3, byrow = TRUE)

x_test <- matrix(c(
    10.1, 11.2, 9.4,
    -20.2, -21.3, -19.3,
    6.0, -10.1, 9.2
), nrow = 3, ncol = 3, byrow = TRUE)

expected_indices <- matrix(c(
    0L, 2L, 1L,
    3L, 4L, 5L,
    8L, 7L, 6L
), nrow = 3, ncol = 3, byrow = TRUE)

expected_distances <- matrix(c(
    2.0, 5.0, 6.0,
    2.0, 5.0, 6.0,
    2.0, 6.0, 17.0
), nrow = 3, ncol = 3, byrow = TRUE)

test_that("approximate_neighbors constructor defaults work", {
    ann <- approximate_neighbors$new()
    expect_true(R6::is.R6(ann))
    expect_s3_class(ann, "approximate_neighbors")
})

test_that("approximate_neighbors constructorcustom params and n_probe active binding work", {
    ann <- approximate_neighbors$new(
        n_neighbors = 3L,
        algorithm = "ivfflat",
        metric = "sqeuclidean",
        n_list = 3L,
        n_probe = 1L,
        kmeans_iter = 10L,
        train_fraction = 1.0,
        seed = 0L
    )
    expect_true(R6::is.R6(ann))
    expect_equal(ann$n_probe, 1L)

    ann$n_probe <- 2L
    expect_equal(ann$n_probe, 2L)
})

test_that("approximate_neighbors train_and_add and kneighbors variants work", {
    ann <- approximate_neighbors$new(
        n_neighbors = 3L,
        algorithm = "ivfflat",
        metric = "sqeuclidean",
        n_list = 3L,
        n_probe = 2L,
        kmeans_iter = 10L,
        train_fraction = 1.0,
        seed = 0L
    )
    ann$train_and_add(x_train)

    result <- ann$kneighbors(x_test)

    expect_named(result, c("distances", "indices"))
    expect_equal(result$indices, expected_indices)
    expect_equal(result$distances, expected_distances, tolerance = tol)

    result_indices_only <- ann$kneighbors(x_test, return_distance = FALSE)

    expect_true(is.matrix(result_indices_only))
    expect_equal(result_indices_only, expected_indices)

    result_explicit_neighbors <- ann$kneighbors(x_test, n_neighbors = 3L)

    expect_named(result_explicit_neighbors, c("distances", "indices"))
    expect_equal(result_explicit_neighbors$indices, expected_indices)
    expect_equal(result_explicit_neighbors$distances, expected_distances, tolerance = tol)
})

test_that("approximate_neighbors train() then add() workflow works", {
    ann <- approximate_neighbors$new(
        n_neighbors = 3L,
        algorithm = "ivfflat",
        metric = "sqeuclidean",
        n_list = 3L,
        n_probe = 2L,
        kmeans_iter = 10L,
        train_fraction = 1.0,
        seed = 0L
    )
    ann$train(x_train)
    ann$add(x_train)

    result <- ann$kneighbors(x_test)

    expect_named(result, c("distances", "indices"))
    expect_equal(result$indices, expected_indices)
    expect_equal(result$distances, expected_distances, tolerance = tol)
})

test_that("approximate_neighbors active fields all accessible", {
    ann <- approximate_neighbors$new(
        n_neighbors = 3L, algorithm = "ivfflat", metric = "sqeuclidean",
        n_list = 3L, n_probe = 2L, kmeans_iter = 10L,
        train_fraction = 1.0, seed = 0L
    )
    ann$train_and_add(x_train)

    expect_true(is.matrix(ann$cluster_centroids))
    expect_length(ann$list_sizes, ann$n_list)
    expect_equal(ann$n_index, nrow(x_train))
    expect_equal(ann$n_list, 3L)
    expect_true(ann$kmeans_iter >= 1L)
    expect_equal(ann$n_features, ncol(x_train))
})

test_that("approximate_neighbors ivfsq quantization variants run", {
    n_samples <- 32L
    n_features <- 5L
    n_queries <- 7L

    combos <- list(
        list(q = "int8", s = "per_dimension"),
        list(q = "int8", s = "uniform"),
        list(q = "int8", s = "direct_unsigned"),
        list(q = "int8", s = "direct_signed"),
        list(q = "bfloat16", s = "direct_signed")
    )

    for (combo in combos) {
        for (residual in c(TRUE, FALSE)) {
            base_values <- if (combo$s == "direct_unsigned")
                c(25, 70, 115, 160) else c(-75, -30, 15, 60)

            xt <- matrix(0.0, nrow = n_samples, ncol = n_features)
            for (s in 0:(n_samples - 1)) {
                cluster <- s %% 4
                for (f in 0:(n_features - 1)) {
                    xt[s + 1, f + 1] <- base_values[cluster + 1] + 3 * f +
                        (s %/% 4) * 0.37 + (f + 1) * (s + 1) * 0.003
                }
            }

            xq <- matrix(0.0, nrow = n_queries, ncol = n_features)
            for (q in 0:(n_queries - 1)) {
                src <- (q * 5 + 1) %% n_samples
                for (f in 0:(n_features - 1)) {
                    xq[q + 1, f + 1] <- xt[src + 1, f + 1] +
                        (q + 1) * (f + 2) * 0.007
                }
            }

            ann <- approximate_neighbors$new(
                n_neighbors = 4L,
                algorithm = "ivfsq",
                n_list = 4L,
                n_probe = 4L,
                kmeans_iter = 20L,
                seed = 42L,
                quantization = combo$q,
                quantization_scheme = combo$s,
                quantization_residual = residual
            )
            ann$train_and_add(xt)

            result <- ann$kneighbors(xq)
            expect_equal(dim(result$distances), c(n_queries, 4L))
            expect_equal(dim(result$indices), c(n_queries, 4L))
            expect_true(all(is.finite(result$distances)))
            expect_true(all(result$indices >= 0L & result$indices < n_samples))
        }
    }
})

test_that("approximate_neighbors quantization error exits", {
    # Invalid quantization format
    expect_error(approximate_neighbors$new(quantization = "none"))

    # Invalid quantization scheme
    expect_error(approximate_neighbors$new(quantization_scheme = "nonexistent"))

    # Quantization is ignored when ivfflat is selected.
    ann <- approximate_neighbors$new(algorithm = "ivfflat", quantization = "int8")
    expect_no_error(ann$train(x_train))

    # ivfsq with defaults trains (centroids only, no quantization at train time).
    ann <- approximate_neighbors$new(algorithm = "ivfsq")
    expect_no_error(ann$train(x_train))
})

