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
# Model persistence tests
#
# Tests that save_model / load_model (functional API) and $save / load_model
# (method + functional) round-trip correctly for every model class.
#
# One test (SVC) exercises the fully-functional approach with a named file and
# manual cleanup.  All others use $save() + tempfile + on.exit(unlink()).
# ============================================================================

tol <- sqrt(.Machine$double.eps) * 100
svm_test_tol <- 1e-6

check_model_metadata <- function(path) {
    out <- capture.output(print_model_metadata(path))
    txt <- paste(out, collapse = "\n")
    expect_true(grepl("Header keyword:", txt))
    expect_true(grepl("Integer size:", txt))
    expect_true(grepl("Serialization version:", txt))
    expect_true(grepl("Handle type:", txt))
    expect_true(grepl("Precision:", txt))
    expect_true(grepl("Saved AOCL-DA build version:", txt))
}

check_model_versions <- function(model) {
    out <- capture.output(model$print_model_versions())
    txt <- paste(out, collapse = "\n")
    expect_true(grepl("Serialization version:", txt))
    expect_true(grepl("Saved AOCL-DA build version:", txt))
}

# ============================================================================
# SVC — functional approach (save_model / load_model with named file)
# ============================================================================

test_that("SVC save_model and load_model reproduce expected predictions", {
    x_train <- matrix(c(
        1.92, -0.52,
        1.76, 0.84,
        -1.02, 0.94,
        0.79, 1.34,
        2.86, -0.43
    ), nrow = 5, ncol = 2, byrow = TRUE)
    x_test <- matrix(c(
        -0.15, 2.33,
        -1.14, 1.11
    ), nrow = 2, ncol = 2, byrow = TRUE)
    y_train <- c(0, 0, 2, 1, 0)

    m <- SVC$new(
        C = 0.5, kernel = "rbf", gamma = 1, degree = 2,
        coef0 = 1.0, tol = 1e-6, cache_size = 500.0,
        max_ws_size = 11L, mixed_precision = TRUE,
        low_precision_max_iter = 7L, low_precision_tol = 0.25,
        check_data = TRUE
    )
    m$fit(x_train, y_train)

    path <- file.path(tempdir(), "svc_persistence_test.rds")
    save_model(m, path)
    expect_true(file.exists(path))
    check_model_metadata(path)

    m2 <- load_model(path)
    check_model_versions(m2)
    expect_equal(m2$predict(x_test), m$predict(x_test))
    expect_equal(m2$bias, m$bias)
    expect_equal(m2$n_support_per_class, m$n_support_per_class)
    expect_equal(m2$support_vectors_idx, m$support_vectors_idx)
    expect_equal(m2$dual_coef, m$dual_coef)
    expect_equal(m2$support_vectors, m$support_vectors)
    expect_equal(m2$n_classes, m$n_classes)
    expect_equal(m2$n_support, m$n_support)
    expect_equal(m2$n_samples, m$n_samples)
    expect_equal(m2$n_features, m$n_features)

    unlink(path)
    expect_false(file.exists(path))
})

# ============================================================================
# SVR
# ============================================================================

test_that("SVR save and load_model reproduce expected predictions", {
    x_train <- matrix(c(
        1.29, -0.73,
        2.3, -0.47,
        -1.73, 1.66,
        0.11, -0.38,
        -1.03, 0.48
    ), nrow = 5, ncol = 2, byrow = TRUE)
    x_test <- matrix(c(
        -1.61, 0.99,
        0.11, 0.35
    ), nrow = 2, ncol = 2, byrow = TRUE)
    y_train <- c(-5.66, 45.94, 53.25, -21.96, -2.46)

    m <- SVR$new(
        C = 2.0, kernel = "linear", epsilon = 0.5, coef0 = 0.5,
        tol = 1e-6, cache_size = 100.0, mixed_precision = TRUE,
        low_precision_max_iter = 5L, low_precision_tol = 0.05,
        check_data = TRUE
    )
    m$fit(x_train, y_train)

    tmp <- tempfile(fileext = ".rds")
    on.exit(unlink(tmp))

    m$save(tmp)
    check_model_metadata(tmp)

    m2 <- load_model(tmp)
    check_model_versions(m2)
    expect_equal(m2$predict(x_test), m$predict(x_test), tolerance = svm_test_tol * 10)
    expect_equal(m2$bias, m$bias)
    expect_equal(m2$support_vectors_idx, m$support_vectors_idx)
    expect_equal(m2$dual_coef, m$dual_coef)
    expect_equal(m2$support_vectors, m$support_vectors)
    expect_equal(m2$n_support, m$n_support)
    expect_equal(m2$n_samples, m$n_samples)
    expect_equal(m2$n_features, m$n_features)
})

# ============================================================================
# NuSVC
# ============================================================================

test_that("NuSVC save and load_model reproduce expected predictions", {
    x_train <- matrix(c(
        1.2, 0.92,
        1.39, 1.06,
        1.24, 0.91,
        1.21, 1.05,
        0.65, -0.75
    ), nrow = 5, ncol = 2, byrow = TRUE)
    x_test <- matrix(c(
        -0.02, 0.57,
        0.89, -0.77
    ), nrow = 2, ncol = 2, byrow = TRUE)
    y_train <- c(0, 0, 0, 0, 1)

    m <- NuSVC$new(
        nu = 0.3, kernel = "sigmoid", degree = 2, coef0 = 1,
        tol = 1e-6, cache_size = 100.0, max_ws_size = 5L,
        mixed_precision = TRUE, low_precision_max_iter = 3L,
        low_precision_tol = 0.05
    )
    m$fit(x_train, y_train)

    tmp <- tempfile(fileext = ".rds")
    on.exit(unlink(tmp))

    m$save(tmp)
    check_model_metadata(tmp)

    m2 <- load_model(tmp)
    check_model_versions(m2)
    expect_equal(m2$predict(x_test), m$predict(x_test))
    expect_equal(m2$bias, m$bias)
    expect_equal(m2$n_support_per_class, m$n_support_per_class)
    expect_equal(m2$support_vectors_idx, m$support_vectors_idx)
    expect_equal(m2$dual_coef, m$dual_coef)
    expect_equal(m2$support_vectors, m$support_vectors)
    expect_equal(m2$n_classes, m$n_classes)
    expect_equal(m2$n_support, m$n_support)
    expect_equal(m2$n_samples, m$n_samples)
    expect_equal(m2$n_features, m$n_features)
})

# ============================================================================
# NuSVR
# ============================================================================

test_that("NuSVR save and load_model reproduce expected predictions", {
    x_train <- matrix(c(
        2.07, 0.43,
        -0.43, 0.3,
        -0.04, 0.39,
        0.46, 0.09,
        -1.38, -0.54
    ), nrow = 5, ncol = 2, byrow = TRUE)
    x_test <- matrix(c(
        -0.87, 1.73,
        -1.1, -0.82
    ), nrow = 2, ncol = 2, byrow = TRUE)
    y_train <- c(125.09, -1.89, 22.09, 27.03, -98.74)

    m <- NuSVR$new(
        C = 2.0, nu = 0.5, kernel = "poly", degree = 2,
        coef0 = 0.5, tol = 1e-6, cache_size = 100.0,
        mixed_precision = TRUE, low_precision_max_iter = 5L,
        low_precision_tol = 0.05
    )
    m$fit(x_train, y_train)

    tmp <- tempfile(fileext = ".rds")
    on.exit(unlink(tmp))

    m$save(tmp)
    check_model_metadata(tmp)

    m2 <- load_model(tmp)
    check_model_versions(m2)
    expect_equal(m2$predict(x_test), m$predict(x_test), tolerance = svm_test_tol * 10)
    expect_equal(m2$bias, m$bias)
    expect_equal(m2$support_vectors_idx, m$support_vectors_idx)
    expect_equal(m2$dual_coef, m$dual_coef)
    expect_equal(m2$support_vectors, m$support_vectors)
    expect_equal(m2$n_support, m$n_support)
    expect_equal(m2$n_samples, m$n_samples)
    expect_equal(m2$n_features, m$n_features)
})

# ============================================================================
# kmeans
# ============================================================================

test_that("kmeans save and load_model reproduce expected results", {
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

    m <- kmeans$new(
        n_clusters = 2L,
        C = initial_centres,
        n_init = 1L,
        max_iter = 100,
        tol = 1.0e-4,
        seed = 23L,
        algorithm = "elkan",
        mixed_precision = TRUE,
        low_precision_max_iter = 200L,
        low_precision_tol = 1e-2,
    )
    m$fit(x_train)

    tmp <- tempfile(fileext = ".rds")
    on.exit(unlink(tmp))

    m$save(tmp)
    expect_true(file.exists(tmp))
    check_model_metadata(tmp)

    m2 <- load_model(tmp)
    check_model_versions(m2)

    expect_equal(m2$cluster_centres, m$cluster_centres, tolerance = tol)
    expect_equal(m2$labels, m$labels)
    expect_equal(m2$inertia, m$inertia, tolerance = tol)
    expect_equal(m2$n_iter, m$n_iter)
    expect_equal(m2$n_samples, m$n_samples)
    expect_equal(m2$n_features, m$n_features)
    expect_equal(m2$n_clusters, m$n_clusters)
    expect_equal(m2$transform(x_test), m$transform(x_test), tolerance = tol)
    expect_equal(m2$predict(x_test), m$predict(x_test))
})

# ============================================================================
# linmod
# ============================================================================

test_that("linmod save and load_model reproduce exact predictions", {
    x_train <- matrix(c(
        1, 1,
        2, 3,
        3, 5,
        4, 8,
        5, 7,
        6, 9
    ), nrow = 6, ncol = 2, byrow = TRUE)
    y_train <- c(3.0, 6.5, 10.0, 12.0, 13.0, 19.0)

    m <- linmod$new("mse",
        intercept = TRUE, scaling = "standardize",
        reg_lambda = 0.1, reg_alpha = 0.5, tol = 1e-6,
        check_data = TRUE
    )
    m$fit(x_train, y_train)

    tmp <- tempfile(fileext = ".rds")
    on.exit(unlink(tmp))

    m$save(tmp)
    check_model_metadata(tmp)

    m2 <- load_model(tmp)
    check_model_versions(m2)
    expect_equal(m2$predict(x_train), m$predict(x_train), tolerance = 1e-4)
    expect_equal(m2$coef, m$coef)
})

# ============================================================================
# decision_tree
# ============================================================================

test_that("decision_tree save and load_model reproduce expected predictions", {
    x_train <- matrix(c(
        0.0, 0.0,
        1.0, 1.0
    ), nrow = 2, ncol = 2, byrow = TRUE)
    y_train <- c(0L, 1L)
    x_test <- matrix(c(2.0, 2.0), nrow = 1, ncol = 2)
    y_test <- c(1L)

    m <- decision_tree$new(
        criterion = "cross-entropy", min_samples_split = 1L,
        max_depth = 10, max_features = 2L, seed = 1,
        feat_thresh = 1e-3, check_data = TRUE
    )
    m$fit(x_train, y_train)

    tmp <- tempfile(fileext = ".rds")
    on.exit(unlink(tmp))

    m$save(tmp)
    check_model_metadata(tmp)

    m2 <- load_model(tmp)
    check_model_versions(m2)
    expect_equal(m2$predict(x_test), m$predict(x_test))
    expect_equal(m2$score(x_test, y_test), m$score(x_test, y_test))
    expect_equal(m2$n_samples, m$n_samples)
    expect_equal(m2$n_obs, m$n_obs)
    expect_equal(m2$n_features, m$n_features)
    expect_equal(m2$depth, m$depth)
    expect_equal(m2$n_nodes, m$n_nodes)
    expect_equal(m2$n_leaves, m$n_leaves)
})

# ============================================================================
# decision_forest
# ============================================================================

test_that("decision_forest save and load_model reproduce expected predictions", {
    x_train <- matrix(c(
        0.0, 0.0,
        1.0, 1.0
    ), nrow = 2, ncol = 2, byrow = TRUE)
    y_train <- c(0L, 1L)
    x_test <- matrix(c(2.0, 2.0), nrow = 1, ncol = 2)
    y_test <- c(1L)

    m <- decision_forest$new(
        n_trees = 10L, criterion = "cross-entropy",
        min_samples_split = 1L, max_depth = 15,
        bootstrap = FALSE, features_selection = "all",
        samples_factor = 0.8, seed = 42L,
        check_data = TRUE
    )
    m$fit(x_train, y_train)

    tmp <- tempfile(fileext = ".rds")
    on.exit(unlink(tmp))

    m$save(tmp)
    check_model_metadata(tmp)

    m2 <- load_model(tmp)
    check_model_versions(m2)
    expect_equal(m2$predict(x_test), m$predict(x_test))
    expect_equal(m2$score(x_test, y_test), m$score(x_test, y_test))
    expect_equal(m2$max_features, m$max_features)
    expect_equal(m2$features_selection, m$features_selection)
    expect_equal(m2$samples_factor, m$samples_factor)
})

# ============================================================================
# PCA
# ============================================================================

test_that("PCA save and load_model reproduce same transform", {
    x_train <- matrix(c(
        1, 2, 3,
        2, 5, 4,
        3, 6, 1
    ), nrow = 3, ncol = 3, byrow = TRUE)
    x_test <- matrix(c(
        1, 1, 4,
        3, 2, 3,
        0, 2, 3,
        1, 0, 1
    ), nrow = 4, ncol = 3, byrow = TRUE)

    m <- PCA$new(
        n_components = 2, bias = "biased", solver = "gesdd",
        store_U = TRUE, check_data = TRUE
    )
    m$fit(x_train)
    Xt_before <- m$transform(x_test)

    tmp <- tempfile(fileext = ".rds")
    on.exit(unlink(tmp))

    m$save(tmp)
    check_model_metadata(tmp)

    m2 <- load_model(tmp)
    check_model_versions(m2)
    expect_equal(m2$transform(x_test), Xt_before, tolerance = tol)
    expect_equal(m2$principal_components, m$principal_components, tolerance = tol)
    expect_equal(m2$variance, m$variance, tolerance = tol)
    expect_equal(m2$total_variance, m$total_variance, tolerance = tol)
    expect_equal(m2$column_means, m$column_means, tolerance = tol)
    expect_equal(m2$sigma, m$sigma, tolerance = tol)
    expect_equal(m2$n_samples, m$n_samples)
    expect_equal(m2$n_features, m$n_features)
    expect_equal(m2$n_components, m$n_components)
})

# ============================================================================
# approximate_neighbors
# ============================================================================

test_that("approximate_neighbors save and load_model reproduce results", {
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
    result_before <- ann$kneighbors(x_test)

    tmp <- tempfile(fileext = ".rds")
    on.exit(unlink(tmp))

    ann$save(tmp)
    expect_true(file.exists(tmp))
    check_model_metadata(tmp)

    ann2 <- load_model(tmp)
    check_model_versions(ann2)
    expect_equal(ann2$n_probe, 2L)

    result_after <- ann2$kneighbors(x_test)
    expect_equal(result_after$indices, result_before$indices)
    expect_equal(
        result_after$distances,
        result_before$distances,
        tolerance = tol
    )
})

# ============================================================================
# nearest_neighbors — NULL preservation
# ============================================================================

test_that("nearest_neighbors preserves NULL constructor values across save and load", {
    x_train <- matrix(c(
        -1, -1, 2,
        -2, -1, 3,
        -3, -2, -1,
        1, 3, 1,
        2, 5, 1,
        3, -1, 2
    ), nrow = 6, ncol = 3, byrow = TRUE)

    m <- nearest_neighbors$new(n_neighbors = NULL, radius = NULL)
    m$fit(x_train)

    tmp <- tempfile(fileext = ".rds")
    on.exit(unlink(tmp))

    m$save(tmp)
    check_model_metadata(tmp)

    m2 <- load_model(tmp)
    check_model_versions(m2)

    expect_error(
        m2$kneighbors(x_train),
        "n_neighbors cannot be NULL in kneighbors\\(\\) if it was NULL in the constructor"
    )
    expect_error(
        m2$radius_neighbors(x_train),
        "radius cannot be NULL in radius_neighbors\\(\\) if it was NULL in the constructor"
    )
})

# ============================================================================
# nearest_neighbors — classifier
# ============================================================================

test_that("nearest_neighbors classifier save and load_model reproduce expected predictions", {
    x_train <- matrix(c(
        -1, -1, 2,
        -2, -1, 3,
        -3, -2, -1,
        1, 3, 1,
        2, 5, 1,
        3, -1, 2
    ), nrow = 6, ncol = 3, byrow = TRUE)
    x_test <- matrix(c(
        -2, 2, 3,
        -1, -2, -1,
        2, 1, -3
    ), nrow = 3, ncol = 3, byrow = TRUE)
    y_train_cls <- c(1L, 2L, 0L, 1L, 2L, 2L)

    m <- nearest_neighbors$new(
        n_neighbors = 3L,
        weights = "distance",
        algorithm = "brute",
        metric = "minkowski",
        p = 3.0,
        leaf_size = 10,
        radius = 5.0,
        check_data = TRUE
    )
    m$fit(x_train, y_train_cls)

    tmp <- tempfile(fileext = ".rds")
    on.exit(unlink(tmp))

    m$save(tmp)
    check_model_metadata(tmp)

    m2 <- load_model(tmp)
    check_model_versions(m2)
    expect_equal(
        m2$classifier_predict(x_test, "knn"),
        m$classifier_predict(x_test, "knn")
    )
    expect_equal(m2$classifier_predict_proba(x_test, "knn"),
        m$classifier_predict_proba(x_test, "knn"),
        tolerance = tol
    )
})

# ============================================================================
# nearest_neighbors — regressor
# ============================================================================

test_that("nearest_neighbors regressor save and load_model reproduce expected predictions", {
    x_train <- matrix(c(
        -1, -1, 2,
        -2, -1, 3,
        -3, -2, -1,
        1, 3, 1,
        2, 5, 1,
        3, -1, 2
    ), nrow = 6, ncol = 3, byrow = TRUE)
    x_test <- matrix(c(
        -2, 2, 3,
        -1, -2, -1,
        2, 1, -3
    ), nrow = 3, ncol = 3, byrow = TRUE)
    y_train_reg <- c(1.5, 2.3, 0.6, 1.0, 2.8, 5.2)

    m <- nearest_neighbors$new(
        n_neighbors = 3L,
        weights = "distance",
        algorithm = "brute",
        metric = "minkowski",
        p = 3.0,
        leaf_size = 10,
        radius = 5.0,
        check_data = TRUE
    )
    m$fit(x_train, y_train_reg)

    tmp <- tempfile(fileext = ".rds")
    on.exit(unlink(tmp))

    m$save(tmp)
    check_model_metadata(tmp)

    m2 <- load_model(tmp)
    check_model_versions(m2)
    expect_equal(m2$regressor_predict(x_test, "knn"),
        m$regressor_predict(x_test, "knn"),
        tolerance = tol
    )
    expect_equal(m2$regressor_predict(x_test, "radius_neighbors"),
        m$regressor_predict(x_test, "radius_neighbors"),
        tolerance = tol
    )
})

# ============================================================================
# nearest_neighbors — unsupervised
# ============================================================================

test_that("nearest_neighbors unsupervised save and load_model reproduce results", {
    x_train <- matrix(c(
        -1, -1, 2,
        -2, -1, 3,
        -3, -2, -1,
        1, 3, 1,
        2, 5, 1,
        3, -1, 2
    ), nrow = 6, ncol = 3, byrow = TRUE)
    x_test <- matrix(c(
        -2, 2, 3,
        -1, -2, -1,
        2, 1, -3
    ), nrow = 3, ncol = 3, byrow = TRUE)
    m <- nearest_neighbors$new(
        n_neighbors = 3L,
        weights = "distance",
        algorithm = "brute",
        metric = "minkowski",
        p = 3.0,
        leaf_size = 10,
        radius = 5.0,
        check_data = TRUE
    )
    m$fit(x_train)
    result_before <- m$kneighbors(x_test, n_neighbors = 3L, return_distance = TRUE)

    tmp <- tempfile(fileext = ".rds")
    on.exit(unlink(tmp))

    m$save(tmp)
    check_model_metadata(tmp)

    m2 <- load_model(tmp)
    check_model_versions(m2)

    result <- m2$kneighbors(x_test, n_neighbors = 3L, return_distance = TRUE)
    expect_named(result, c("distances", "indices"))
    expect_equal(result$indices, result_before$indices)
    expect_equal(result$distances, result_before$distances, tolerance = tol)
})

test_that("KernelPCA save and load_model reproduce same transform", {
    A <- matrix(c(
        1, 0, 0,
        0, 2, 0,
        0, 0, 3,
        1, 1, 1,
        2, 2, 2
    ), nrow = 5, ncol = 3, byrow = TRUE)

    m <- KernelPCA$new(n_components = 2L, kernel = "rbf", gamma = 0.5)
    m$fit(A)
    scores_before <- m$scores
    eigenvalues_before <- m$eigenvalues
    transform_before <- m$transform(A[1:2, , drop = FALSE])

    tmp <- tempfile(fileext = ".rds")
    on.exit(unlink(tmp))

    m$save(tmp)
    check_model_metadata(tmp)

    m2 <- load_model(tmp)
    check_model_versions(m2)

    expect_equal(m2$eigenvalues, eigenvalues_before, tolerance = tol)
    expect_equal(m2$scores, scores_before, tolerance = tol)
    expect_equal(m2$transform(A[1:2, , drop = FALSE]), transform_before,
                 tolerance = tol)
    expect_equal(m2$n_components, 2L)
    expect_equal(m2$n_samples, 5L)
    expect_equal(m2$n_features, 3L)
})

# ============================================================================
# NMF
# ============================================================================

test_that("NMF save and load_model reproduce factors and transforms", {
    A_train <- matrix(c(
        1, 2, 3, 4,
        2, 4, 6, 8,
        3, 6, 9, 12,
        4, 1, 5, 2,
        5, 3, 8, 4
    ), nrow = 5, ncol = 4, byrow = TRUE)

    A_new <- matrix(c(
        2, 3, 5, 4,
        1, 1, 2, 1
    ), nrow = 2, ncol = 4, byrow = TRUE)

    for (init in c("nndsvda", "nndsvd", "random", "nndsvdar")) {
        m <- NMF$new(n_components = 2L, init = init, max_iter = 200L,
                     tol = 1.0e-6, random_state = 42L)
        W_before <- m$fit_transform(A_train)
        H_before <- m$h
        recon_before <- m$reconstruction_err
        n_iter_before <- m$n_iter
        transform_before <- m$transform(A_new)
        inv_before <- m$inverse_transform(transform_before)

        tmp <- tempfile(fileext = ".rds")
        m$save(tmp)
        check_model_metadata(tmp)

        m2 <- load_model(tmp)
        check_model_versions(m2)

        expect_equal(m2$w, W_before, tolerance = tol)
        expect_equal(m2$h, H_before, tolerance = tol)
        expect_equal(m2$reconstruction_err, recon_before, tolerance = tol)
        expect_equal(m2$n_iter, n_iter_before)
        expect_equal(m2$n_components, 2L)

        transform_after <- m2$transform(A_new)
        inv_after <- m2$inverse_transform(transform_after)
        expect_equal(transform_after, transform_before, tolerance = tol)
        expect_equal(inv_after, inv_before, tolerance = tol)

        unlink(tmp)
    }
})

test_that("NMF fit after load succeeds", {
    A_train <- matrix(c(
        1, 2, 3, 4,
        2, 4, 6, 8,
        3, 6, 9, 12,
        4, 1, 5, 2
    ), nrow = 4, ncol = 4, byrow = TRUE)

    m <- NMF$new(n_components = 2L, random_state = 0L)
    m$fit(A_train)

    tmp <- tempfile(fileext = ".rds")
    on.exit(unlink(tmp))
    m$save(tmp)

    m2 <- load_model(tmp)

    # Refit on new data
    A_new <- matrix(c(
        1, 0, 2, 1,
        0, 2, 1, 3,
        2, 1, 0, 2,
        1, 3, 2, 0
    ), nrow = 4, ncol = 4, byrow = TRUE)
    W_new <- m2$fit_transform(A_new)
    expect_equal(dim(W_new), c(4L, 2L))
    result <- m2$transform(A_new)
    expect_equal(dim(result), c(4L, 2L))
})

# ============================================================================
# quantized approximate_neighbors (ivfsq)
# ============================================================================

test_that("quantized approximate_neighbors save and load_model reproduce results", {
    x_train <- matrix(c(
        -1.5, -1.2, 2.6,
        -2.5, -1.3, 3.8,
        -3.6, -2.8, -1.7,
        1.6, 3.4, 1.2,
        2.8, 5.3, 1.3,
        3.0, -1.2, 2.4,
        0.5, 0.3, 1.1,
        -0.5, 0.8, -0.9
    ), nrow = 8, ncol = 3, byrow = TRUE)
    x_test <- matrix(c(
        -2.5, 2.2, 3.6,
        -1.4, -2.7, -1.0,
        2.6, 1.2, -3.7
    ), nrow = 3, ncol = 3, byrow = TRUE)

    for (quantization in c("int8", "int4", "bfloat16")) {
        scheme <- if (quantization == "bfloat16") "direct_signed" else "per_dimension"
        ann <- approximate_neighbors$new(
            n_neighbors = 3L,
            algorithm = "ivfsq",
            n_list = 2L,
            n_probe = 2L,
            seed = 42L,
            quantization = quantization,
            quantization_scheme = scheme,
            quantization_residual = TRUE
        )
        ann$train_and_add(x_train)
        result_before <- ann$kneighbors(x_test)

        tmp <- tempfile(fileext = ".rds")
        ann$save(tmp)
        check_model_metadata(tmp)

        ann2 <- load_model(tmp)
        result_after <- ann2$kneighbors(x_test)
        expect_equal(result_after$indices, result_before$indices)
        expect_equal(result_after$distances, result_before$distances, tolerance = tol)

        unlink(tmp)
    }
})
