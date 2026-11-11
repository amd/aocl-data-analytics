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

tol <- sqrt(.Machine$double.eps)

x_train <- matrix(c(
    -1, -1, 2,
    -2, -1, 3,
    -3, -2, -1,
    1, 3, 1,
    2, 5, 1,
    3, -1, 2
), nrow = 6, ncol = 3, byrow = TRUE)

y_train_cls <- c(1L, 2L, 0L, 1L, 2L, 2L)
y_train_reg <- c(1.5, 2.3, 0.6, 1.0, 2.8, 5.2)

x_test <- matrix(c(
    -2, 2, 3,
    -1, -2, -1,
    2, 1, -3
), nrow = 3, ncol = 3, byrow = TRUE)

expected_indices <- matrix(c(
    1L, 0L, 3L,
    2L, 0L, 1L,
    3L, 5L, 4L
), nrow = 3, ncol = 3, byrow = TRUE)

expected_distances <- matrix(c(
    3.0, 3.31662479, 3.74165739,
    2.0, 3.16227766, 4.24264069,
    4.58257569, 5.47722558, 5.65685425
), nrow = 3, ncol = 3, byrow = TRUE)

expected_proba_knn <- matrix(c(
    0.2, 0.4, 0.4,
    0.2, 0.4, 0.4,
    0.2, 0.4, 0.4
), nrow = 3, ncol = 3, byrow = TRUE)

expected_labels_knn <- c(1L, 1L, 1L)
expected_targets_knn <- c(1.64, 2.12, 2.22)

expected_proba_radius <- matrix(c(
    0.0, 0.66666667, 0.33333333,
    0.33333333, 0.33333333, 0.33333333,
    0.0, 1.0, 0.0
), nrow = 3, ncol = 3, byrow = TRUE)

expected_labels_radius <- c(1L, 0L, 1L)
expected_targets_radius <- c(1.6, 1.46666667, 1.0)

test_that("nearest_neighbors constructor defaults work", {
    m <- nearest_neighbors$new()
    expect_s3_class(m, "nearest_neighbors")
})

test_that("nearest_neighbors constructor validates n_neighbors and radius", {
    expect_error(
        nearest_neighbors$new(n_neighbors = 1.5),
        "Number of neighbors must be an integer or NULL"
    )
    expect_error(
        nearest_neighbors$new(n_neighbors = c(1, 2)),
        "Number of neighbors must be an integer or NULL"
    )
    expect_error(
        nearest_neighbors$new(n_neighbors = NA),
        "Number of neighbors must be an integer or NULL"
    )
    expect_error(
        nearest_neighbors$new(radius = "bad"),
        "Radius must be a number or NULL"
    )
    expect_error(
        nearest_neighbors$new(radius = -1),
        "Radius must be positive"
    )
})

test_that("nearest_neighbors supports NULL constructor values", {
    m <- nearest_neighbors$new(n_neighbors = NULL, radius = NULL)
    m$fit(x_train)
    expect_error(
        m$kneighbors(x_test),
        "n_neighbors cannot be NULL in kneighbors\\(\\) if it was NULL in the constructor"
    )
    expect_error(
        m$radius_neighbors(x_test),
        "radius cannot be NULL in radius_neighbors\\(\\) if it was NULL in the constructor"
    )
    expect_no_error(m$kneighbors(x_test, n_neighbors = 3L))
    expect_no_error(m$radius_neighbors(x_test, radius = 3.0))
})

test_that("nearest_neighbors classifier public API returns expected results", {
    m <- nearest_neighbors$new(
        algorithm = "auto",
        metric = "euclidean",
        radius = 5.0
    )
    m$fit(x_train, y_train_cls)

    expect_equal(m$classifier_predict(x_test, "knn"), expected_labels_knn)
    expect_equal(m$classifier_predict_proba(x_test, "knn"),
        expected_proba_knn,
        tolerance = tol
    )
    expect_equal(
        m$classifier_predict(x_test, "radius_neighbors"),
        expected_labels_radius
    )
    expect_equal(m$classifier_predict_proba(x_test, "radius_neighbors"),
        expected_proba_radius,
        tolerance = tol
    )

    result <- m$kneighbors(x_test, n_neighbors = 3L, return_distance = TRUE)
    expect_named(result, c("distances", "indices"))
    expect_equal(result$indices, expected_indices)
    expect_equal(result$distances, expected_distances, tolerance = tol)
    expect_equal(
        m$kneighbors(x_test, n_neighbors = 3L, return_distance = FALSE),
        expected_indices
    )
})

test_that("nearest_neighbors query methods validate null inputs", {
    m <- nearest_neighbors$new(
        n_neighbors = 3L,
        algorithm = "auto",
        metric = "euclidean",
        radius = 5.0
    )
    m$fit(x_train, y_train_cls)

    expect_error(m$kneighbors(NULL), "kneighbors\\(\\) is not implemented for X=NULL")
    expect_error(
        m$classifier_predict(NULL),
        "Predicting labels is not implemented for X=NULL"
    )
    expect_error(
        m$classifier_predict_proba(NULL),
        "Predicting probabilities is not implemented for X=NULL"
    )
})

test_that("nearest_neighbors works with different weights and metrics", {
    for (w in c("uniform", "distance")) {
        m <- nearest_neighbors$new(n_neighbors = 3L, weights = w, metric = "euclidean")
        expect_no_error(m$fit(x_train, y_train_cls))
        expect_no_error(m$classifier_predict(x_test))
    }
    for (met in c("euclidean", "manhattan", "cosine")) {
        m <- nearest_neighbors$new(n_neighbors = 3L, algorithm = "brute", metric = met)
        expect_no_error(m$fit(x_train, y_train_cls))
        expect_no_error(m$classifier_predict(x_test))
    }
})

test_that("nearest_neighbors regressor public API returns expected results", {
    m <- nearest_neighbors$new(
        algorithm = "auto",
        metric = "euclidean",
        radius = 5.0
    )
    m$fit(x_train, y_train_reg)

    expect_equal(m$regressor_predict(x_test, "knn"),
        expected_targets_knn,
        tolerance = tol
    )
    expect_equal(m$regressor_predict(x_test, "radius_neighbors"),
        expected_targets_radius,
        tolerance = tol
    )

    result <- m$kneighbors(x_test, n_neighbors = 3L, return_distance = TRUE)
    expect_named(result, c("distances", "indices"))
    expect_equal(result$indices, expected_indices)
    expect_equal(result$distances, expected_distances, tolerance = tol)
})

test_that("nearest_neighbors regressor prediction validates null input", {
    m <- nearest_neighbors$new(
        n_neighbors = 3L,
        algorithm = "auto",
        metric = "euclidean",
        radius = 5.0
    )
    m$fit(x_train, y_train_reg)
    expect_error(
        m$regressor_predict(NULL),
        "Predicting target values is not implemented for X=NULL"
    )
})

test_that("nearest_neighbors validates manual outlier handling", {
    cls_model <- nearest_neighbors$new(
        n_neighbors = 3, metric = "euclidean", radius = 1.0,
        outlier_handling = 1.5
    )
    cls_model$fit(x_train, y_train_cls)
    expect_error(
        cls_model$classifier_predict(x_test, search_mode = "radius_neighbors"),
        "outlier_handling must be NULL, 'most_frequent', or an integer for classification"
    )

    reg_model <- nearest_neighbors$new(
        n_neighbors = 3, metric = "euclidean", radius = 1.0,
        outlier_handling = "bad"
    )
    reg_model$fit(x_train, y_train_reg)
    expect_error(
        reg_model$regressor_predict(x_test, search_mode = "radius_neighbors"),
        "outlier_handling must be NULL, 'most_frequent', or numeric for regression"
    )
})

test_that("nearest_neighbors unsupervised API returns expected kneighbors", {
    m <- nearest_neighbors$new(
        n_neighbors = 3L,
        algorithm = "auto",
        metric = "euclidean",
        radius = 5.0
    )
    m$fit(x_train)

    result <- m$kneighbors(x_test, n_neighbors = 3L, return_distance = TRUE)
    expect_named(result, c("distances", "indices"))
    expect_equal(result$indices, expected_indices)
    expect_equal(result$distances, expected_distances, tolerance = tol)
    expect_equal(
        m$kneighbors(x_test, n_neighbors = 3L, return_distance = FALSE),
        expected_indices
    )
})

# --- Algorithm variant tests ---

test_that("nearest_neighbors algorithm=kd_tree", {
    X <- matrix(c(0, 0, 1, 0, 0, 1, 1, 1, 0.5, 0.5),
                nrow = 5, ncol = 2, byrow = TRUE)
    y <- as.integer(c(0, 0, 1, 1, 0))
    m <- nearest_neighbors$new(n_neighbors = 3L, algorithm = "kd_tree")
    m$fit(X, y)
    query <- matrix(c(0.6, 0.6), nrow = 1)
    result <- m$kneighbors(query)
    expect_equal(result$indices[1, 1], 4L)
    expect_equal(result$distances[1, 1], 0.14142135623730948, tolerance = 1e-10)
    pred <- m$classifier_predict(query)
    expect_equal(pred, 1L)
})

# --- radius_neighbors direct API ---

test_that("radius_neighbors returns per-query variable-length results", {
    nn <- nearest_neighbors$new(n_neighbors = 3L, metric = "euclidean", radius = 3.0)
    nn$fit(x_train)

    result <- nn$radius_neighbors(x_test, radius = 3.0, sort_results = TRUE)
    expect_type(result, "list")
    expect_length(result, 3L)

    for (i in seq_along(result)) {
        entry <- result[[i]]
        expect_named(entry, c("distances", "indices"))
        if (length(entry$distances) > 1L) {
            expect_true(all(diff(entry$distances) >= 0))
        }
    }
    expect_equal(result[[1]]$distances, c(3.0), tolerance = tol)
    expect_equal(result[[1]]$indices, c(1L))
    expect_equal(result[[2]]$distances, c(2.0), tolerance = tol)
    expect_equal(result[[2]]$indices, c(2L))
    expect_length(result[[3]]$indices, 0L)
})

test_that("radius_neighbors return_distance=FALSE returns indices only", {
    nn <- nearest_neighbors$new(n_neighbors = 3L, metric = "euclidean", radius = 3.0)
    nn$fit(x_train)

    result <- nn$radius_neighbors(x_test, radius = 3.0, return_distance = FALSE)
    expect_type(result, "list")
    for (i in seq_along(result)) {
        expect_type(result[[i]], "integer")
    }
})

test_that("radius_neighbors errors on sort_results=TRUE with return_distance=FALSE", {
    nn <- nearest_neighbors$new(n_neighbors = 3L, metric = "euclidean", radius = 3.0)
    nn$fit(x_train)

    expect_error(
        nn$radius_neighbors(x_test, radius = 3.0,
                            return_distance = FALSE, sort_results = TRUE),
        regexp = "sort_results"
    )
})

# --- Error path tests ---

test_that("fit() rejects X=NULL", {
    m <- nearest_neighbors$new()
    expect_error(m$fit(NULL), "fit\\(\\) is not implemented for X=NULL")
})

test_that("kneighbors validates n_neighbors type and sign", {
    m <- nearest_neighbors$new(n_neighbors = 3L)
    m$fit(x_train)
    expect_error(
        m$kneighbors(x_test, n_neighbors = 1.5),
        "Number of neighbors must be an integer or NULL"
    )
    expect_error(
        m$kneighbors(x_test, n_neighbors = -2L),
        "Number of neighbors must be positive or NULL"
    )
})

test_that("radius_neighbors rejects X=NULL and invalid radius", {
    m <- nearest_neighbors$new(n_neighbors = 3L, radius = 5.0)
    m$fit(x_train)
    expect_error(
        m$radius_neighbors(NULL),
        "radius_neighbors\\(\\) is not implemented for X=NULL"
    )
    expect_error(
        m$radius_neighbors(x_test, radius = "bad"),
        "Radius must be a number or NULL"
    )
    expect_error(
        m$radius_neighbors(x_test, radius = -1.0),
        "Radius must be non-negative or NULL"
    )
})

test_that("predict methods error when y not provided during fit", {
    m <- nearest_neighbors$new(n_neighbors = 3L, radius = 5.0)
    m$fit(x_train)
    expect_error(
        m$classifier_predict_proba(x_test),
        "Labels y were not provided during fit"
    )
    expect_error(
        m$classifier_predict(x_test),
        "Labels y were not provided during fit"
    )
    expect_error(
        m$regressor_predict(x_test),
        "Targets y were not provided during fit"
    )
})

test_that("predict with radius_neighbors mode errors when radius=NULL in constructor", {
    m <- nearest_neighbors$new(n_neighbors = 3L, radius = NULL)
    m$fit(x_train, y_train_cls)
    expect_error(
        m$classifier_predict_proba(x_test, search_mode = "radius_neighbors"),
        "radius was NULL in the constructor"
    )
    expect_error(
        m$classifier_predict(x_test, search_mode = "radius_neighbors"),
        "radius was NULL in the constructor"
    )

    m2 <- nearest_neighbors$new(n_neighbors = 3L, radius = NULL)
    m2$fit(x_train, y_train_reg)
    expect_error(
        m2$regressor_predict(x_test, search_mode = "radius_neighbors"),
        "radius was NULL in the constructor"
    )
})
