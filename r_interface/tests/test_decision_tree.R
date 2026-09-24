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

test_that("decision_tree constructor defaults work", {
    m <- decision_tree$new()
    expect_s3_class(m, "decision_tree")
})

test_that("decision_tree constructor custom params accepted", {
    m <- decision_tree$new(
        criterion = "cross-entropy", seed = 1,
        max_depth = 10, min_samples_split = 1,
        max_features = 2, check_data = TRUE
    )
    expect_s3_class(m, "decision_tree")
})

test_that("decision_tree max_features binding works", {
    m <- decision_tree$new()
    expect_equal(m$max_features, 0L)
    m$max_features <- 2L
    expect_equal(m$max_features, 2L)
})

test_that("decision_tree public API returns expected predictions and metadata", {
    m <- decision_tree$new(min_samples_split = 1L)
    m$fit(x_train, y_train)

    expect_equal(m$predict(x_test), expected_pred)
    expect_equal(m$predict_proba(x_test), expected_proba)

    log_proba <- m$predict_log_proba(x_test)
    expect_equal(dim(log_proba), c(1L, 2L))
    expect_equal(exp(log_proba), expected_proba, tolerance = 1e-12)

    expect_equal(m$score(x_test, y_test), 1.0)
    expect_equal(m$n_samples, 2L)
    expect_equal(m$n_obs, 2L)
    expect_equal(m$n_features, 2L)
    expect_true(m$depth >= 1L)
    expect_equal(m$n_nodes, 3L)
    expect_equal(m$n_leaves, 2L)
    expect_true(m$n_threads >= 0)
})

test_that("decision_tree works with all criteria", {
    for (crit in c("gini", "cross-entropy", "misclassification")) {
        m <- decision_tree$new(criterion = crit, min_samples_split = 1L, seed = 1)
        expect_no_error(m$fit(x_train, y_train))
    }
})

test_that("decision_tree accepts data.frame X", {
    df <- as.data.frame(x_train)
    m <- decision_tree$new(min_samples_split = 1L)
    expect_no_error(m$fit(df, y_train))
})
