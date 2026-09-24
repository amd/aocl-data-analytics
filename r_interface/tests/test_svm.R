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

svm_test_tol <- 1e-6

svc_x_train <- matrix(c(
    1.92, -0.52,
    1.76, 0.84,
    -1.02, 0.94,
    0.79, 1.34,
    2.86, -0.43
), nrow = 5, ncol = 2, byrow = TRUE)

svc_x_test <- matrix(c(
    -0.15, 2.33,
    -1.14, 1.11
), nrow = 2, ncol = 2, byrow = TRUE)

svc_y_train <- c(0, 0, 2, 1, 0)
svc_y_test <- c(1, 2)
svc_expected_bias <- c(0.6614310488724978, 0.5365371320535124, 0.0)
svc_expected_n_support_per_class <- c(3L, 1L, 1L)
svc_expected_support_idx <- c(0L, 1L, 4L, 3L, 2L)
svc_expected_dual_coef <- matrix(c(
    0.1567003658615869, 0.6042853359149776, 0.23901429822343534, -1.0, -1.0,
    0.2661922344693693, 0.4034513140618167, 0.3303564514688140, 1.0, -1.0
), nrow = 2, ncol = 5, byrow = TRUE)

svc_expected_support_vectors <- matrix(c(
    1.92, -0.52,
    1.76, 0.84,
    2.86, -0.43,
    0.79, 1.34,
    -1.02, 0.94
), nrow = 5, ncol = 2, byrow = TRUE)

svc_expected_predict <- c(0, 2)
svc_expected_decision_ovr <- matrix(c(
    2.1647936191078374, 0.9012607489781155, -0.1192298001005942,
    1.0595892420507806, -0.20380478276768527, 2.1918357549767650
), nrow = 2, ncol = 3, byrow = TRUE)

svc_expected_decision_ovo <- matrix(c(
    0.5080428914586723, 0.4697303287293784, 0.08714885163887187,
    0.6386841115554920, -0.42100181649081425, -0.9347511031340464
), nrow = 2, ncol = 3, byrow = TRUE)

svc_prob_x_train <- matrix(c(
    -1.2, -1.08, -2.92,
    1.84, -0.08, 0.03,
    0.91, 0.92, 0.75,
    3.31, 2.22, -1.48,
    -0.01, -0.85, -2.22,
    2.14, 0.35, -1.52,
    -0.54, -1.05, -2.06,
    1.46, 1.38, 1.83,
    0.44, 0.67, 1.86,
    -0.99, 1.13, 1.35,
    0.09, -1.69, 0.36,
    -0.67, 0.01, 0.46,
    -1.9, -1.64, 0.6,
    0.47, 0.92, 0.44,
    -1.53, -0.3, -0.93,
    -1.81, 0.92, 0.65,
    0.32, 0.99, 1.41
), nrow = 17, ncol = 3, byrow = TRUE)

svc_prob_x_test <- matrix(c(
    -0.14, -0.01, 1.93,
    1.98, 1.55, 0.69,
    -0.66, -0.61, 0.41,
    1.89, -1.54, -0.08,
    -1.48, -2.81, 0.77,
    0.62, 0.8, 2.02,
    0.81, 0.8, 0.44,
    -0.8, 0.64, 0.56
), nrow = 8, ncol = 3, byrow = TRUE)

svc_prob_y_train <- c(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1)
svc_prob_expected_probaA <- -3.31093987
svc_prob_expected_probaB <- 0.36261956
svc_prob_expected_proba <- matrix(c(
    0.187326, 0.812674,
    0.968047, 0.031953,
    0.01328, 0.98672,
    0.766831, 0.233169,
    0.093423, 0.906577,
    0.747862, 0.252138,
    0.65693, 0.34307,
    0.012665, 0.987335
), nrow = 8, ncol = 2, byrow = TRUE)
svc_prob_expected_log_proba <- matrix(c(
    -1.674904, -0.207425,
    -0.032474, -3.443504,
    -4.321526, -0.013369,
    -0.265489, -1.455991,
    -2.37062, -0.098079,
    -0.290537, -1.377779,
    -0.420177, -1.069822,
    -4.368891, -0.012746
), nrow = 8, ncol = 2, byrow = TRUE)

svr_x_train <- matrix(c(
    1.29, -0.73,
    2.3, -0.47,
    -1.73, 1.66,
    0.11, -0.38,
    -1.03, 0.48
), nrow = 5, ncol = 2, byrow = TRUE)

svr_x_test <- matrix(c(
    -1.61, 0.99,
    0.11, 0.35
), nrow = 2, ncol = 2, byrow = TRUE)

svr_y_train <- c(-5.66, 45.94, 53.25, -21.96, -2.46)
svr_y_test <- c(12.5, 27.19)
svr_expected_bias <- -3.664383850678176
svr_expected_support_idx <- c(0L, 1L, 2L, 3L, 4L)
svr_expected_dual_coef <- matrix(c(-0.7796538205625038, 1.0, 1.0, -1.0, -0.22034617943749626),
    nrow = 1, ncol = 5
)
svr_expected_support_vectors <- matrix(c(
    1.29, -0.73,
    2.3, -0.47,
    -1.73, 1.66,
    0.11, -0.38,
    -1.03, 0.48
), nrow = 5, ncol = 2, byrow = TRUE)

svr_expected_predict <- c(-1.1380735884612876, -2.9877681126775064)
svr_expected_score <- -9.16417571463822

nusvc_x_train <- matrix(c(
    1.2, 0.92,
    1.39, 1.06,
    1.24, 0.91,
    1.21, 1.05,
    0.65, -0.75
), nrow = 5, ncol = 2, byrow = TRUE)

nusvc_x_test <- matrix(c(
    -0.02, 0.57,
    0.89, -0.77
), nrow = 2, ncol = 2, byrow = TRUE)

nusvc_y_train <- c(0, 0, 0, 0, 1)
nusvc_y_test <- c(1, 1)
nusvc_expected_bias <- 0.04340625628910816
nusvc_expected_n_support_per_class <- c(1L, 1L)
nusvc_expected_support_idx <- c(2L, 4L)
nusvc_expected_dual_coef <- matrix(c(-6.089160022858101, 6.089160022858101), nrow = 1, ncol = 2)
nusvc_expected_support_vectors <- matrix(c(
    1.24, 0.91,
    0.65, -0.75
), nrow = 2, ncol = 2, byrow = TRUE)

nusvc_expected_predict <- c(0, 1)
nusvc_expected_decision <- c(-3.642897716170645, 0.4697240910420319)

nusvr_x_train <- matrix(c(
    2.07, 0.43,
    -0.43, 0.3,
    -0.04, 0.39,
    0.46, 0.09,
    -1.38, -0.54
), nrow = 5, ncol = 2, byrow = TRUE)

nusvr_x_test <- matrix(c(
    -0.87, 1.73,
    -1.1, -0.82
), nrow = 2, ncol = 2, byrow = TRUE)

nusvr_y_train <- c(125.09, -1.89, 22.09, 27.03, -98.74)
nusvr_y_test <- c(65.67, -103.03)
nusvr_expected_bias <- 12.437607185762609
nusvr_expected_support_idx <- c(0L, 1L, 3L, 4L)
nusvr_expected_dual_coef <- matrix(c(1.0, -0.25, 0.25, -1.0), nrow = 1, ncol = 4)
nusvr_expected_support_vectors <- matrix(c(
    2.07, 0.43,
    -0.43, 0.3,
    0.46, 0.09,
    -1.38, -0.54
), nrow = 4, ncol = 2, byrow = TRUE)

nusvr_expected_predict <- c(11.96152679112588, 3.5562811749938152)
nusvr_expected_score <- -0.001081559858395753

test_that("SVC public API returns expected fitted values", {
    m <- SVC$new(kernel = "rbf", gamma = 1, tol = 1e-6)
    m$fit(svc_x_train, svc_y_train)

    expect_equal(m$bias, svc_expected_bias, tolerance = svm_test_tol)
    expect_equal(m$n_support_per_class, svc_expected_n_support_per_class)
    expect_equal(m$support_vectors_idx, svc_expected_support_idx)
    expect_equal(m$dual_coef, svc_expected_dual_coef, tolerance = svm_test_tol * 200)
    expect_equal(m$support_vectors, svc_expected_support_vectors, tolerance = svm_test_tol)
    expect_equal(m$predict(svc_x_test), svc_expected_predict)
    expect_equal(m$score(svc_x_test, svc_y_test), 0.5)
    expect_equal(m$decision_function(svc_x_test), svc_expected_decision_ovr,
        tolerance = svm_test_tol * 10
    )
    expect_equal(m$decision_function(svc_x_test, "ovo"), svc_expected_decision_ovo,
        tolerance = svm_test_tol * 20
    )
    expect_equal(m$n_classes, 3L)
    expect_equal(m$n_support, 5L)
    expect_equal(m$n_samples, 5L)
    expect_equal(m$n_features, 2L)
    expect_true(all(m$lp_n_iter >= 0))
    expect_true(all(m$n_iter >= 0))
})

test_that("SVC probability API works when probability=TRUE", {
    m <- SVC$new(kernel = "rbf", tol = 1e-5, probability = TRUE, random_state = 1)
    m$fit(svc_prob_x_train, svc_prob_y_train)

    proba <- m$predict_proba(svc_prob_x_test)
    expect_equal(dim(proba), c(8L, 2L))
    expect_equal(m$probA, svc_prob_expected_probaA, tolerance = 1e-4)
    expect_equal(m$probB, svc_prob_expected_probaB, tolerance = 1e-4)
    expect_equal(proba, svc_prob_expected_proba, tolerance = 1e-4)

    log_proba <- m$predict_log_proba(svc_prob_x_test)
    expect_equal(dim(log_proba), c(8L, 2L))
    expect_equal(log_proba, svc_prob_expected_log_proba, tolerance = 1e-4)
    expect_equal(exp(log_proba), proba, tolerance = 1e-4)

    expect_true(is.numeric(m$probA))
    expect_true(is.numeric(m$probB))
})

test_that("SVR public API returns expected fitted values", {
    m <- SVR$new(kernel = "linear", epsilon = 0.1, tol = 1e-6)
    m$fit(svr_x_train, svr_y_train)

    expect_equal(m$bias, svr_expected_bias, tolerance = svm_test_tol)
    expect_equal(m$support_vectors_idx, svr_expected_support_idx)
    expect_equal(m$dual_coef, svr_expected_dual_coef, tolerance = svm_test_tol * 20)
    expect_equal(m$support_vectors, svr_expected_support_vectors, tolerance = svm_test_tol)
    expect_equal(m$predict(svr_x_test), svr_expected_predict, tolerance = svm_test_tol * 10)
    expect_equal(m$score(svr_x_test, svr_y_test), svr_expected_score, tolerance = svm_test_tol * 10)
    expect_equal(m$n_support, 5L)
    expect_equal(m$n_samples, 5L)
    expect_equal(m$n_features, 2L)
})

test_that("NuSVC public API returns expected fitted values", {
    m <- NuSVC$new(nu = 0.3, kernel = "sigmoid", coef0 = 1, tol = 1e-6)
    m$fit(nusvc_x_train, nusvc_y_train)

    expect_equal(as.numeric(m$bias), nusvc_expected_bias, tolerance = svm_test_tol)
    expect_equal(m$n_support_per_class, nusvc_expected_n_support_per_class)
    expect_equal(m$support_vectors_idx, nusvc_expected_support_idx)
    expect_equal(m$dual_coef, nusvc_expected_dual_coef, tolerance = svm_test_tol * 200)
    expect_equal(m$support_vectors, nusvc_expected_support_vectors, tolerance = svm_test_tol)
    expect_equal(m$predict(nusvc_x_test), nusvc_expected_predict)
    expect_equal(m$score(nusvc_x_test, nusvc_y_test), 0.5)
    expect_equal(as.numeric(m$decision_function(nusvc_x_test)), nusvc_expected_decision,
        tolerance = svm_test_tol * 200
    )
    expect_equal(as.numeric(m$decision_function(nusvc_x_test, "ovo")), nusvc_expected_decision,
        tolerance = svm_test_tol * 200
    )
    expect_equal(m$n_classes, 2L)
    expect_equal(m$n_support, 2L)
    expect_equal(m$n_samples, 5L)
    expect_equal(m$n_features, 2L)
})

test_that("NuSVC probability API works when probability=TRUE", {
    m <- NuSVC$new(
        nu = 0.3, kernel = "sigmoid", coef0 = 1, tol = 1e-6,
        probability = TRUE
    )
    m$fit(nusvc_x_train, nusvc_y_train)

    proba <- m$predict_proba(nusvc_x_test)
    expect_equal(dim(proba), c(2L, 2L))
    expect_equal(rowSums(proba), c(1, 1), tolerance = 1e-10)

    log_proba <- m$predict_log_proba(nusvc_x_test)
    expect_equal(dim(log_proba), c(2L, 2L))
    expect_equal(exp(log_proba), proba, tolerance = 1e-10)

    expect_true(is.numeric(m$probA))
    expect_true(is.numeric(m$probB))
})

test_that("NuSVR public API returns expected fitted values", {
    m <- NuSVR$new(nu = 0.5, kernel = "poly", tol = 1e-6)
    m$fit(nusvr_x_train, nusvr_y_train)

    expect_equal(m$bias, nusvr_expected_bias, tolerance = svm_test_tol)
    expect_equal(m$support_vectors_idx, nusvr_expected_support_idx)
    expect_equal(m$dual_coef, nusvr_expected_dual_coef, tolerance = svm_test_tol)
    expect_equal(m$support_vectors, nusvr_expected_support_vectors, tolerance = svm_test_tol)
    expect_equal(m$predict(nusvr_x_test), nusvr_expected_predict, tolerance = svm_test_tol * 10)
    expect_equal(m$score(nusvr_x_test, nusvr_y_test), nusvr_expected_score, tolerance = svm_test_tol * 10)
    expect_equal(m$n_support, 4L)
    expect_equal(m$n_samples, 5L)
    expect_equal(m$n_features, 2L)
    expect_true(m$lp_n_iter >= 0)
})

test_that("SVC works with all kernel types", {
    for (k in c("linear", "poly", "rbf", "sigmoid")) {
        m <- SVC$new(C = 1.0, kernel = k, gamma = 1.0, coef0 = 0.0, tol = 1e-6)
        m$fit(svc_x_train, svc_y_train)
        preds <- m$predict(svc_x_test)
        expect_length(preds, 2)
    }
})

test_that("SVR works with all kernel types", {
    for (k in c("linear", "poly", "rbf", "sigmoid")) {
        m <- SVR$new(C = 1.0, epsilon = 0.1, kernel = k, gamma = 1.0,
                     coef0 = 1.0, tol = 1e-6)
        m$fit(svr_x_train, svr_y_train)
        preds <- m$predict(svr_x_test)
        expect_length(preds, 2)
    }
})
