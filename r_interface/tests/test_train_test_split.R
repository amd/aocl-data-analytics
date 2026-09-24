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

n <- 24
p <- 4
X <- matrix(c(
    1.43, 0.82, -0.51, 1.17,
    0.37, -1.24, 0.68, -0.43,
    -0.89, 1.56, 0.23, -0.71,
    1.72, -0.34, -1.18, 0.52,
    -0.55, 0.93, 1.41, -1.06,
    0.84, -1.47, 0.29, 0.78,
    1.28, 0.61, -0.74, 1.34,
    -1.13, 0.44, 1.57, -0.92,
    0.62, -0.38, 1.74, -0.67,
    -0.47, 1.23, -0.84, 0.51,
    1.87, 0.14, -0.63, 1.42,
    -0.73, 1.68, 0.37, -0.91,
    0.48, -1.32, 0.93, 0.74,
    -1.17, 0.56, -0.48, 1.23,
    0.91, 1.37, -1.14, 0.34,
    -0.34, 0.78, 1.52, -0.67,
    1.53, -0.47, 0.81, -1.28,
    -0.82, 1.14, -0.37, 0.95,
    0.27, 0.67, -1.43, 0.58,
    -1.48, 0.31, 0.76, -0.54,
    0.73, 1.42, -0.58, 0.83,
    -0.61, -0.94, 1.37, 0.42,
    1.14, -0.73, 0.52, -1.31,
    -0.43, 0.88, -1.27, 0.64
), nrow = n, ncol = p, byrow = TRUE)
y <- c(
    1.23, -0.87, 2.14, -1.45, 0.56, -0.92,
    1.78, -0.63, 0.94, -1.17, 0.73, -0.48,
    1.56, -0.84, 0.31, 1.19, -0.91, 2.08,
    -1.52, 0.61, -0.88, 1.83, -0.59, 0.97
)
cls <- c(rep("a", 8), rep("b", 8), rep("c", 8))

test_that("train_test_split default 75/25 split preserves total samples", {
    res <- train_test_split(X, y, seed = 1L)
    X_tr <- res[[1]]
    X_te <- res[[2]]
    y_tr <- res[[3]]
    y_te <- res[[4]]

    expect_equal(nrow(X_tr) + nrow(X_te), n)
    expect_equal(length(y_tr) + length(y_te), n)
    expect_equal(ncol(X_tr), p)
    expect_equal(ncol(X_te), p)
    expect_equal(nrow(X_tr), n - as.integer(ceiling(n * 0.25)))
    expect_equal(nrow(X_te), as.integer(ceiling(n * 0.25)))
})

test_that("train_test_split test_size as fraction works", {
    res <- train_test_split(X, test_size = 0.2, seed = 1L)
    X_te <- res[[2]]
    expect_equal(nrow(X_te), as.integer(ceiling(n * 0.2)))
})

test_that("train_test_split test_size as integer count works", {
    res <- train_test_split(X, test_size = 10L, seed = 1L)
    expect_equal(nrow(res[[2]]), 10L)
    expect_equal(nrow(res[[1]]), n - 10L)
})

test_that("train_test_split same seed produces same result", {
    res1 <- train_test_split(X, seed = 7L)
    res2 <- train_test_split(X, seed = 7L)
    expect_equal(res1[[1]], res2[[1]])
    expect_equal(res1[[2]], res2[[2]])
})

test_that("train_test_split shuffle=FALSE returns first rows in train", {
    res <- train_test_split(X, test_size = 10L, shuffle = FALSE)
    expect_equal(res[[1]], X[1:14, ])
    expect_equal(res[[2]], X[15:24, ])
})

test_that("train_test_split with stratify preserves class proportions", {
    X_strat <- matrix(rnorm(300), nrow = 150, ncol = 2)
    y_strat <- as.integer(rep(0:2, each = 50))
    cls_strat <- as.character(y_strat)

    res <- train_test_split(X_strat, y_strat, stratify = cls_strat,
                            test_size = 30L, seed = 42L)
    y_tr <- res[[3]]
    y_te <- res[[4]]

    expect_equal(length(y_tr), 120L)
    expect_equal(length(y_te), 30L)
    expect_equal(sum(y_te == 0L), 10L)
    expect_equal(sum(y_te == 1L), 10L)
    expect_equal(sum(y_te == 2L), 10L)
})

test_that("train_test_split errors when stratify used with shuffle=FALSE", {
    expect_error(
        train_test_split(X, shuffle = FALSE, stratify = cls),
        regexp = "shuffle"
    )
})

test_that("train_test_split errors on mismatched array lengths", {
    y_bad <- rnorm(n + 5)
    expect_error(train_test_split(X, y_bad, seed = 1L))
})

test_that("train_test_split accepts data.frame input", {
    df <- as.data.frame(X)
    res <- train_test_split(df, seed = 1L)
    expect_equal(nrow(res[[1]]) + nrow(res[[2]]), n)
})

test_that("train_test_split errors with no arrays", {
    expect_error(train_test_split(), "At least one array must be provided")
})

test_that("train_test_split errors when stratify length mismatches data", {
    expect_error(
        train_test_split(X, stratify = c(0L, 1L)),
        "same number of samples"
    )
})

test_that("train_test_split errors on invalid train_size float", {
    expect_error(
        train_test_split(X, train_size = -0.5),
        "train_size cannot be a float outside the range"
    )
})

test_that("train_test_split errors on invalid test_size float", {
    expect_error(
        train_test_split(X, test_size = 1.5),
        "test_size cannot be a float outside the range"
    )
})

test_that("train_test_split errors when train_size + test_size > 1", {
    expect_error(
        train_test_split(X, train_size = 0.7, test_size = 0.5),
        "train_size \\+ test_size"
    )
})
