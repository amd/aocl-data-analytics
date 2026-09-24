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

x_train <- matrix(c(
    0.0, 0.0,
    1.0, 1.0
), nrow = 2, ncol = 2, byrow = TRUE)

y_train <- c(0L, 1L)
x_test <- matrix(c(2.0, 2.0), nrow = 1, ncol = 2)
y_test <- c(1L)
expected_pred <- c(1L)
expected_proba <- matrix(c(0.0, 1.0), nrow = 1, ncol = 2, byrow = TRUE)

test_that("decision_forest constructor defaults work", {
    m <- decision_forest$new()
    expect_s3_class(m, "decision_forest")
})

test_that("decision_forest constructor custom params accepted", {
    m <- decision_forest$new(
        n_trees = 10L, criterion = "cross-entropy",
        seed = 7L, max_depth = 8L, bootstrap = FALSE,
        features_selection = "all", check_data = TRUE
    )
    expect_s3_class(m, "decision_forest")
})

test_that("decision_forest read-write properties work", {
    m <- decision_forest$new()
    expect_equal(m$max_features, 0L)
    m$max_features <- 2L
    expect_equal(m$max_features, 2L)

    expect_equal(m$features_selection, "sqrt")
    m$features_selection <- "all"
    expect_equal(m$features_selection, "all")

    expect_equal(m$samples_factor, 1.0)
    m$samples_factor <- 0.7
    expect_equal(m$samples_factor, 0.7)
})

test_that("decision_forest public API returns expected predictions", {
    m <- decision_forest$new(
        min_samples_split = 1L, n_trees = 10L,
        bootstrap = FALSE, features_selection = "all",
        seed = 42L
    )
    m$fit(x_train, y_train)

    expect_equal(m$predict(x_test), expected_pred)
    expect_equal(m$predict_proba(x_test), expected_proba)

    log_proba <- m$predict_log_proba(x_test)
    expect_equal(exp(log_proba), m$predict_proba(x_test), tolerance = 1e-10)

    expect_equal(m$score(x_test, y_test), 1.0)
})

test_that("decision_forest works with all features_selection options", {
    for (fs in c("all", "sqrt", "log2")) {
        m <- decision_forest$new(
            n_trees = 5L, features_selection = fs,
            bootstrap = FALSE, min_samples_split = 1L, seed = 1L
        )
        expect_no_error(m$fit(x_train, y_train))
    }
})

test_that("decision_forest works with all criteria", {
    for (crit in c("gini", "cross-entropy")) {
        m <- decision_forest$new(
            n_trees = 5L, criterion = crit,
            bootstrap = FALSE, min_samples_split = 1L, seed = 1L
        )
        expect_no_error(m$fit(x_train, y_train))
    }
})

test_that("decision_forest features_selection=log2 matches Python reference", {
    X <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                  1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
                  2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5),
                nrow = 12, ncol = 3, byrow = TRUE)
    y <- as.integer(c(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2))
    m <- decision_forest$new(n_trees = 10L, features_selection = "log2", seed = 42L)
    m$fit(X, y)
    expect_equal(m$score(X, y), 1.0)
})

test_that("decision_forest bootstrap=TRUE runs and predicts", {
    X <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                  1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
                  2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5),
                nrow = 12, ncol = 3, byrow = TRUE)
    y <- as.integer(c(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2))
    m <- decision_forest$new(n_trees = 10L, bootstrap = TRUE, seed = 42L)
    m$fit(X, y)
    expect_equal(m$score(X, y), 1.0)
})
