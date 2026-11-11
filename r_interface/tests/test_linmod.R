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

tol <- 1e-4

x_train <- matrix(c(
    1, 1,
    2, 3,
    3, 5,
    4, 8,
    5, 7,
    6, 9
), nrow = 6, ncol = 2, byrow = TRUE)

y_train <- c(3.0, 6.5, 10.0, 12.0, 13.0, 19.0)
expected_coef <- c(2.45, 0.35)
expected_preds <- c(2.8, 5.95, 9.1, 12.6, 14.7, 17.85)
expected_coef_intercept <- c(2.35, 0.35, 0.43333333333333535)
expected_preds_intercept <- c(
    3.133333333333335, 6.183333333333335, 9.233333333333334,
    12.633333333333335, 14.633333333333335, 17.683333333333334
)

x_reg <- matrix(c(
    0.497, -0.138,
    0.648, 1.523,
    -0.469, 0.543,
    1.579, 0.767,
    -0.234, -0.234,
    -0.463, -0.466
), nrow = 6, ncol = 2, byrow = TRUE)

y_reg <- c(6.132, 99.065, 13.28, 88.632, -20.423, -39.951)

regularized_cases <- list(
    list(
        name = "ElasticNet", reg_lambda = 0.1, reg_alpha = 0.5,
        tol = 1e-6, intercept = TRUE, solver = "auto",
        coef = c(30.42698430323763, 47.92432984823038, 0.6201200680560284)
    ),
    list(
        name = "Lasso", reg_lambda = 1.0, reg_alpha = 1.0,
        tol = 1e-6, intercept = TRUE, solver = "auto",
        coef = c(29.751069, 51.37881, -0.3529817356770124)
    ),
    list(
        name = "Ridge", reg_lambda = 1.0, reg_alpha = 0.0,
        tol = 1e-6, intercept = TRUE, solver = "sparse_cg",
        coef = c(28.49233212, 40.06618967, 3.7353163594626566)
    )
)

x_under <- matrix(c(
    0.497, -0.138, -0.421,
    0.648, 1.523, 2.123
), nrow = 2, ncol = 3, byrow = TRUE)

x_under_y <- c(6.132, 99.065)
expected_under_primal <- c(10.31036367, 16.26075041, 21.27136156)
expected_under_dual <- c(6.10348802, 11.22983044)

y_train_cls <- as.double(c(0, 1, 1, 1, 1, 1))

test_that("linmod constructor defaults work (mse)", {
    m <- linmod$new("mse")
    expect_s3_class(m, "linmod")
})

test_that("linmod constructor all params accepted", {
    m <- linmod$new("mse",
        intercept = TRUE, solver = "svd",
        scaling = "standardize", reg_lambda = 0.1, reg_alpha = 0.0,
        warm_start = FALSE, tol = 1e-5, check_data = TRUE
    )
    expect_s3_class(m, "linmod")
})

test_that("linmod mse public API returns exact coefficients without intercept", {
    m <- linmod$new("mse", intercept = FALSE, solver = "svd")
    m$fit(x_train, y_train)

    expect_equal(m$coef, expected_coef, tolerance = tol)
    expect_equal(m$predict(x_train), expected_preds, tolerance = tol)
    expect_type(m$loss, "double")
    expect_length(m$loss, 1L)
    expect_type(m$nrm_gradient_loss, "double")
    expect_length(m$nrm_gradient_loss, 1L)
    expect_true(is.numeric(m$n_iter))
    expect_length(m$n_iter, 1L)
    expect_true(is.numeric(m$time))
    expect_length(m$time, 1L)
})

test_that("linmod mse public API returns exact coefficients with intercept", {
    m <- linmod$new("mse", intercept = TRUE, solver = "svd")
    m$fit(x_train, y_train)

    expect_equal(m$coef, expected_coef_intercept, tolerance = tol)
    expect_equal(m$predict(x_train), expected_preds_intercept, tolerance = tol)
})

test_that("linmod regularized solvers return expected coefficients", {
    for (case in regularized_cases) {
        solver_arg <- if (identical(case$solver, "auto")) NULL else case$solver
        model <- do.call(linmod$new, c(
            list(
                mod = "mse",
                reg_lambda = case$reg_lambda,
                reg_alpha = case$reg_alpha,
                tol = case$tol,
                intercept = case$intercept
            ),
            if (is.null(solver_arg)) list() else list(solver = solver_arg)
        ))
        model$fit(x_reg, y_reg)
        expect_equal(model$coef, case$coef, tolerance = tol)
    }
})

test_that("linmod coord solver skip options converge to the lasso solution", {
    lasso_coef <- c(29.751069, 51.37881, -0.3529817356770124)
    for (skip_tol in c(-4.0, 0.0, 1.0e-4)) {
        model <- linmod$new("mse",
            reg_lambda = 1.0, reg_alpha = 1.0, tol = 1e-6, dual_tol = 1e-4,
            solver = "coord", intercept = TRUE, optim_coord_skip_min = 3L,
            optim_coord_skip_max = 13L, optim_coord_skip_tol = skip_tol
        )
        model$fit(x_reg, y_reg)
        expect_equal(model$coef, lasso_coef, tolerance = tol)
    }
})

test_that("linmod warm_start and x0 use the expected primal and dual coefficients", {
    model <- linmod$new("mse",
        solver = "sparse_cg", reg_lambda = 2.0,
        intercept = FALSE, warm_start = TRUE
    )
    model$fit(x_under, x_under_y)
    expect_equal(model$coef, expected_under_primal, tolerance = tol)
    expect_equal(model$dual_coef, expected_under_dual, tolerance = tol)
    expect_true(model$n_iter > 1)

    model$fit(x_under, x_under_y)
    expect_equal(model$coef, expected_under_primal, tolerance = tol)
    expect_equal(model$dual_coef, expected_under_dual, tolerance = tol)
    expect_equal(model$n_iter, 0)

    model2 <- linmod$new("mse",
        solver = "sparse_cg", reg_lambda = 2.0,
        intercept = FALSE, warm_start = FALSE
    )
    model2$fit(x_under, x_under_y, x0 = expected_under_dual)
    expect_equal(model2$coef, expected_under_primal, tolerance = tol)
    expect_equal(model2$dual_coef, expected_under_dual, tolerance = tol)
    expect_equal(model2$n_iter, 0)
})

test_that("linmod logistic fit and predict run without error", {
    m <- linmod$new("logistic", solver = "lbfgs", intercept = TRUE)
    m$fit(x_train, y_train_cls)
    expect_true(m$trained)
    preds <- m$predict(x_train)
    expect_length(preds, nrow(x_train))
})

test_that("linmod accepts integer matrix and data.frame input", {
    Xi <- matrix(as.integer(round(x_train * 10)), nrow = nrow(x_train), ncol = ncol(x_train))
    m <- linmod$new("mse")
    expect_no_error(m$fit(Xi, y_train))

    df <- as.data.frame(x_train)
    m2 <- linmod$new("mse")
    expect_no_error(m2$fit(df, y_train))
})

# --- Tests for scaling variants ---

X_lm <- matrix(c(1, 1.2, 2, 1.4, 3, 1.6, 4, 1.8, 5, 2.0, 6, 2.2),
               nrow = 6, ncol = 2, byrow = TRUE)
y_lm <- c(3.0, 6.5, 10.0, 12.0, 13.0, 19.0)

test_that("linmod scaling=none", {
    m <- linmod$new("mse", intercept = TRUE, solver = "auto", scaling = "none")
    m$fit(X_lm, y_lm)
    expect_equal(m$coef, c(6.421856899477051, -17.609284497385243, 18.042617830718566),
                 tolerance = 1e-4)
})

test_that("linmod scaling=centering", {
    m <- linmod$new("mse", intercept = TRUE, solver = "svd", scaling = "centering")
    m$fit(X_lm, y_lm)
    expect_equal(m$coef, c(2.78846154, 0.55769231, -0.12435897), tolerance = 1e-4)
})

test_that("linmod scaling=scale only", {
    m <- linmod$new("mse", intercept = TRUE, solver = "svd", scaling = "scale only")
    m$fit(X_lm, y_lm)
    expect_equal(m$coef, c(2.78846154, 0.55769231, -0.12435897), tolerance = 1e-4)
})

test_that("linmod solver=cholesky", {
    m <- linmod$new("mse", intercept = TRUE, solver = "cholesky")
    m$fit(X_lm, y_lm)
    expect_equal(m$coef, c(6.421856899477051, -17.609284497385243, 18.042617830718566),
                 tolerance = 1e-4)
})

test_that("linmod solver=qr runs without error", {
    m <- linmod$new("mse", intercept = TRUE, solver = "qr")
    m$fit(X_lm, y_lm)
    expect_length(m$coef, 3)
    expect_true(m$trained)
})

test_that("linmod solver=coord runs without error", {
    m <- linmod$new("mse", intercept = TRUE, solver = "coord",
                    reg_lambda = 0.1, reg_alpha = 1.0)
    m$fit(X_lm, y_lm)
    expect_length(m$coef, 3)
    expect_true(m$trained)
})

test_that("linmod active fields all accessible after fit", {
    m <- linmod$new("mse", intercept = TRUE, solver = "svd")
    m$fit(X_lm, y_lm)
    expect_true(m$trained)
    expect_length(m$coef, 3)
    expect_type(m$loss, "double")
    expect_type(m$nrm_gradient_loss, "double")
    expect_true(is.numeric(m$n_iter))
    expect_true(m$lp_n_iter >= 0)
    expect_type(m$time, "double")
})

test_that("linmod score returns a numeric value", {
    m <- linmod$new("mse", intercept = FALSE, solver = "svd")
    m$fit(x_train, y_train)
    s <- m$score(x_train, y_train)
    expect_type(s, "double")
})

# --- Error path tests ---

test_that("linmod score errors on mismatched y length", {
    m <- linmod$new("mse", intercept = FALSE, solver = "svd")
    m$fit(x_train, y_train)
    expect_error(
        m$score(x_train, c(1.0, 2.0)),
        "Observations array has invalid shape"
    )
})

test_that("linmod warm_start errors when n_classes changes (logistic)", {
    m <- linmod$new("logistic", solver = "lbfgs", warm_start = TRUE)
    y2 <- as.double(c(0, 1, 1, 1, 1, 1))
    m$fit(x_train, y2)
    y3 <- as.double(c(0, 1, 2, 2, 1, 0))
    expect_error(
        m$fit(x_train, y3),
        "number of classes has changed"
    )
})

test_that("linmod warm_start errors when n_features changes", {
    m <- linmod$new("mse", solver = "auto", warm_start = TRUE)
    m$fit(x_train, y_train)
    X_wider <- cbind(x_train, x_train[, 1])
    expect_error(
        m$fit(X_wider, y_train),
        "number of features has changed"
    )
})

test_that("linmod warm_start errors when well-determined status changes (CG)", {
    X_under <- matrix(c(0.497, -0.138, -0.421,
                        0.648, 1.523, 2.123), nrow = 2, ncol = 3, byrow = TRUE)
    y_under <- c(6.132, 99.065)
    X_over <- matrix(c(0.497, -0.138, -0.421,
                       0.648, 1.523, 2.123,
                       1.0, 0.5, -0.3,
                       -0.2, 0.8, 1.5), nrow = 4, ncol = 3, byrow = TRUE)
    y_over <- c(6.132, 99.065, 5.0, 10.0)

    m <- linmod$new("mse", solver = "sparse_cg", reg_lambda = 2.0,
                    intercept = FALSE, warm_start = TRUE)
    m$fit(X_over, y_over)
    expect_error(
        m$fit(X_under, y_under),
        "Previously fit data was well determined"
    )
})
