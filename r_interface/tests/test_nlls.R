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

expected_coef <- c(2.5410455, 0.25950481)
expected_n_eval <- list(f = 18L, j = 18L, h = 17L, hp = 0L, fd_f = 0L)

exp_data <- list(
    t = c(1.0, 2.0, 4.0, 5.0, 8.0),
    y = c(3.0, 4.0, 6.0, 11.0, 20.0)
)

exp_fun <- function(x, data) {
    x[1] * exp(x[2] * data$t) - data$y
}

exp_jac <- function(x, data) {
    cbind(
        exp(x[2] * data$t),
        data$t * x[1] * exp(x[2] * data$t)
    )
}

exp_hes <- function(x, wr, data) {
    Hr <- matrix(0, nrow = 2, ncol = 2)
    v <- data$t * exp(x[2] * data$t)
    Hr[2, 1] <- sum(wr * v)
    Hr[1, 2] <- Hr[2, 1]
    Hr[2, 2] <- sum(wr * (data$t * x[1]) * v)
    Hr
}

test_that("nlls delays backend initialization until fit", {
    model <- nlls$new(n_coef = 2L, n_res = 5L)

    expect_error(model$n_iter, "fit\\(\\) to be called first")
    expect_error(model$n_eval, "fit\\(\\) to be called first")
    expect_error(model$metrics, "fit\\(\\) to be called first")
})

test_that("nlls fits example with expected coefficients", {
    model <- nlls$new(
        n_coef = 2L,
        n_res = 5L,
        weights = rep(0.12, 5),
        lower_bounds = c(0.0, 0.0),
        upper_bounds = c(5.0, 3.0)
    )

    coef <- model$fit(
        x = c(2.5, 0.25),
        fun = exp_fun,
        jac = exp_jac,
        hes = exp_hes,
        data = exp_data,
        abs_gtol = 1e-7,
        gtol = 1e-9,
        maxit = 20,
    )

    expect_equal(model$coef, expected_coef, tolerance = 1e-5)
    expect_equal(model$n_iter, 17L)
    expect_type(model$n_eval, "list")
    expect_true(all(c("f", "j", "h", "hp", "fd_f") %in% names(model$n_eval)))
    expect_equal(model$n_eval$f, expected_n_eval$f)
    expect_equal(model$n_eval$j, expected_n_eval$j)
    expect_equal(model$n_eval$h, expected_n_eval$h)
    expect_equal(model$n_eval$hp, expected_n_eval$hp)
    expect_equal(model$n_eval$fd_f, expected_n_eval$fd_f)
    expect_type(model$metrics, "list")
    expect_true(all(c("obj", "norm_g", "scl_norm_g") %in% names(model$metrics)))
    expect_equal(model$metrics$obj < 0.04, TRUE)
    expect_equal(model$metrics$norm_g < 1e-6, TRUE)
    expect_equal(model$metrics$scl_norm_g < 1e-6, TRUE)
})

test_that("nlls works without Jacobian (numerical differentiation)", {
    x_data <- seq(0, 2, length.out = 15)
    y_data <- 1.5 * x_data^2

    fun <- function(coef, data) coef[1] * data$x^2 - data$y

    model <- nlls$new(n_coef = 1L, n_res = length(x_data))
    model$fit(x = 1.0, fun = fun, data = list(x = x_data, y = y_data))

    expect_length(model$coef, 1)
    expect_equal(model$coef[1], 1.5, tolerance = 1e-3)
})

test_that("nlls with hessian callback", {
    model <- nlls$new(n_coef = 2L, n_res = 5L,
                      model = "hybrid", method = "galahad", glob_strategy = "tr")
    model$fit(x = c(2.5, 0.25), fun = exp_fun, jac = exp_jac, hes = exp_hes,
              data = exp_data)

    expect_equal(model$coef, c(2.5410456815441385, 0.25950480130239280),
                 tolerance = 1e-7)
    expect_equal(model$n_iter, 4L)
    expect_true(model$metrics$obj < 2.25)
    expect_true(model$metrics$norm_g < 1e-10)
})

test_that("nlls model=gauss-newton", {
    model <- nlls$new(n_coef = 2L, n_res = 5L, model = "gauss-newton")
    model$fit(x = c(2.5, 0.25), fun = exp_fun, jac = exp_jac, data = exp_data)

    expect_equal(model$coef, c(2.541045637671064, 0.25950480381444563),
                 tolerance = 1e-7)
    expect_equal(model$n_iter, 5L)
    expect_true(model$metrics$obj < 2.25)
})

test_that("nlls model=quasi-newton", {
    model <- nlls$new(n_coef = 2L, n_res = 5L, model = "quasi-newton")
    model$fit(x = c(2.5, 0.25), fun = exp_fun, jac = exp_jac, data = exp_data)

    expect_equal(model$coef, c(2.541045686361007, 0.259504800907488),
                 tolerance = 1e-7)
    expect_equal(model$n_iter, 4L)
})

test_that("nlls model=tensor-newton requires hessian", {
    model <- nlls$new(n_coef = 2L, n_res = 5L, model = "tensor-newton")
    model$fit(x = c(2.5, 0.25), fun = exp_fun, jac = exp_jac, hes = exp_hes,
              data = exp_data)

    expect_equal(model$coef, c(2.5410456820529292, 0.2595048012899708),
                 tolerance = 1e-7)
    expect_equal(model$n_iter, 2L)
})

test_that("nlls glob_strategy=reg", {
    model <- nlls$new(n_coef = 2L, n_res = 5L, glob_strategy = "reg")
    model$fit(x = c(2.5, 0.25), fun = exp_fun, jac = exp_jac, data = exp_data)

    expect_equal(model$coef, c(2.541045681772276, 0.25950480129434367),
                 tolerance = 1e-7)
    expect_equal(model$n_iter, 4L)
})

test_that("nlls glob_strategy=reg reg_power=cubic", {
    model <- nlls$new(n_coef = 2L, n_res = 5L,
                      glob_strategy = "reg", reg_power = "cubic")
    model$fit(x = c(2.5, 0.25), fun = exp_fun, jac = exp_jac, data = exp_data)

    expect_equal(model$coef, c(2.541045681772276, 0.25950480129434367),
                 tolerance = 1e-7)
    expect_equal(model$n_iter, 4L)
})

test_that("nlls fit returns self for method chaining", {
    model <- nlls$new(n_coef = 2L, n_res = 5L)
    result <- model$fit(x = c(2.5, 0.25), fun = exp_fun, jac = exp_jac,
                        data = exp_data)
    expect_identical(result, model)
})

test_that("nlls check_derivatives option runs without error", {
    model <- nlls$new(n_coef = 2L, n_res = 5L, check_derivatives = "yes")
    expect_no_error(
        model$fit(x = c(2.5, 0.25), fun = exp_fun, jac = exp_jac,
                  data = exp_data)
    )
})

test_that("nlls maxtime parameter is accepted", {
    model <- nlls$new(n_coef = 2L, n_res = 5L)
    expect_no_error(
        model$fit(x = c(2.5, 0.25), fun = exp_fun, jac = exp_jac,
                  data = exp_data, maxtime = 60.0)
    )
})

# --- Error path tests ---

test_that("nlls coef errors before fit", {
    model <- nlls$new(n_coef = 2L, n_res = 5L)
    expect_error(model$coef, "fit\\(\\) to be called first")
})

test_that("nlls constructor rejects matrix weights", {
    w_mat <- matrix(1, nrow = 5, ncol = 1)
    expect_error(
        nlls$new(n_coef = 2L, n_res = 5L, weights = w_mat),
        "1D vector, not a matrix"
    )
})

test_that("nlls constructor rejects matrix lower_bounds", {
    lb_mat <- matrix(0, nrow = 2, ncol = 1)
    expect_error(
        nlls$new(n_coef = 2L, n_res = 5L, lower_bounds = lb_mat),
        "1D vector, not a matrix"
    )
})

test_that("nlls constructor rejects empty lower_bounds", {
    expect_error(
        nlls$new(n_coef = 2L, n_res = 5L, lower_bounds = numeric(0)),
        "must not be an empty vector"
    )
})

test_that("nlls constructor rejects matrix upper_bounds", {
    ub_mat <- matrix(1, nrow = 2, ncol = 1)
    expect_error(
        nlls$new(n_coef = 2L, n_res = 5L, upper_bounds = ub_mat),
        "1D vector, not a matrix"
    )
})

test_that("nlls constructor rejects empty upper_bounds", {
    expect_error(
        nlls$new(n_coef = 2L, n_res = 5L, upper_bounds = numeric(0)),
        "must not be an empty vector"
    )
})

test_that("nlls constructor rejects mismatched bounds sizes", {
    expect_error(
        nlls$new(n_coef = 2L, n_res = 5L,
                 lower_bounds = c(0, 0), upper_bounds = c(1, 1, 1)),
        "must be the same size"
    )
})
