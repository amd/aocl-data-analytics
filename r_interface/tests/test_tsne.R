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

x_iris <- as.matrix(iris[, 1:4])

x_rect <- matrix(c(
    0.0, 0.0,
    0.0, 1.0,
    2.0, 0.0,
    2.0, 1.0
), nrow = 4, ncol = 2, byrow = TRUE)

test_that("tsne constructor defaults work", {
    m <- tsne$new()
    expect_s3_class(m, "tsne")
})

test_that("tsne constructor all params accepted", {
    m <- tsne$new(
        n_components = 2L, perplexity = 5.0, learning_rate = -1.0,
        max_iter = 500L, n_iter_without_progress = 100L, min_grad_norm = 1e-7,
        early_exaggeration = 12.0, theta = 0.5, init = "pca", seed = 42L,
        mixed_precision = FALSE, low_precision_max_iter = 200L,
        low_precision_min_grad_norm = 1e-4, check_data = FALSE
    )
    expect_s3_class(m, "tsne")
})

test_that("tsne exact method produces expected KL divergence", {
    m <- tsne$new(n_components = 2L, perplexity = 30.0, max_iter = 300L,
                  theta = 0.0, seed = 42L)
    m$fit_transform(x_iris)

    expected_kl <- 0.15038418722650815
    expect_true(m$kl_divergence <= 1.6 * expected_kl)
    expect_equal(m$n_samples, 150L)
    expect_equal(m$n_features, 4L)
    expect_equal(m$n_components, 2L)
    expect_equal(m$n_iter, 300L)
    expect_true(is.finite(m$kl_divergence))
})

test_that("tsne barnes-hut method produces expected KL divergence", {
    m <- tsne$new(n_components = 2L, perplexity = 30.0, max_iter = 300L,
                  theta = 0.5, seed = 42L)
    m$fit_transform(x_iris)

    expected_kl <- 0.16197189688682556
    expect_true(m$kl_divergence <= 1.6 * expected_kl)
})

test_that("tsne rectangle symmetry preserved", {
    m <- tsne$new(n_components = 2L, perplexity = 1.0, max_iter = 50L,
                  theta = 0.0, seed = 0L, learning_rate = 200.0, init = "pca")
    emb <- m$fit_transform(x_rect)

    c0 <- emb[1, 1]
    c1 <- emb[1, 2]
    expect_true(abs(c0) > 1.0)
    expect_true(abs(c1) > 1.0)

    expected_pattern <- matrix(c(
        c0,  c1,
        c0, -c1,
       -c0,  c1,
       -c0, -c1
    ), nrow = 4, ncol = 2, byrow = TRUE)
    expect_equal(emb, expected_pattern, tolerance = 1e-3)
})

test_that("tsne supplied init with identical points stays at init", {
    x <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8), nrow = 4, ncol = 2, byrow = TRUE)
    y_init <- matrix(rep(c(2.0, -3.0), 4), nrow = 4, ncol = 2, byrow = TRUE)

    m <- tsne$new(n_components = 2L, perplexity = 1.0, max_iter = 50L,
                  theta = 0.0, seed = 0L, learning_rate = 200.0, init = y_init)
    embedding <- m$fit_transform(x)

    expect_equal(nrow(embedding), 4L)
    expect_equal(ncol(embedding), 2L)
    expect_equal(embedding, y_init, tolerance = 1e-3)
    expect_equal(m$n_samples, 4L)
    expect_equal(m$n_components, 2L)
    expect_equal(m$embedding, embedding)
    expect_true(m$n_iter >= 0)
    expect_true(m$lp_n_iter >= 0)
    expect_true(is.numeric(m$kl_divergence))
    expect_equal(m$n_features, 2L)
})

test_that("tsne seed reproducibility", {
    m1 <- tsne$new(n_components = 2L, perplexity = 30.0, max_iter = 150L,
                   theta = 0.5, seed = 7L, init = "random")
    m2 <- tsne$new(n_components = 2L, perplexity = 30.0, max_iter = 150L,
                   theta = 0.5, seed = 7L, init = "random")
    emb1 <- m1$fit_transform(x_iris)
    emb2 <- m2$fit_transform(x_iris)

    expect_equal(emb1, emb2)
})

test_that("tsne mixed precision", {
    m <- tsne$new(n_components = 2L, perplexity = 5.0, max_iter = 300L,
                  theta = 0.0, seed = 42L, mixed_precision = TRUE,
                  low_precision_max_iter = 100L, low_precision_min_grad_norm = 1e-3)
    embedding <- m$fit_transform(x_iris[1:20, ])

    expect_equal(nrow(embedding), 20L)
    expect_equal(ncol(embedding), 2L)
    expect_true(all(is.finite(embedding)))
    expect_true(is.finite(m$kl_divergence))
    expect_true(m$lp_n_iter >= 1L)
})

test_that("tsne rejects init with wrong columns", {
    expect_error(
        tsne$new(n_components = 2L, init = matrix(1:12, 4, 3)),
        "columns but n_components=2"
    )
})

test_that("tsne rejects init with non-finite values", {
    expect_error(
        tsne$new(n_components = 2L, init = matrix(c(1, NA, 3, 4, 5, 6), 3, 2)),
        "finite values"
    )
})

test_that("tsne rejects invalid learning_rate string", {
    m <- tsne$new(n_components = 2L, learning_rate = "bad")
    expect_error(m$fit(x_iris), "learning_rate must be")
})

test_that("tsne learning_rate auto works", {
    m <- tsne$new(n_components = 2L, perplexity = 30.0, learning_rate = "auto",
                  seed = 42L, max_iter = 50L)
    embedding <- m$fit_transform(x_iris)
    expect_equal(nrow(embedding), 150L)
})

test_that("tsne init embedding row mismatch rejected", {
    init <- matrix(1:20, nrow = 10, ncol = 2)
    m <- tsne$new(n_components = 2L, perplexity = 30.0, init = init)
    expect_error(m$fit(x_iris), "rows but X has")
})
