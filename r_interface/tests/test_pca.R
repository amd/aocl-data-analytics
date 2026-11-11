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

tol <- .Machine$double.eps * 1000

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

expected_components <- matrix(c(
    0.4082482904638631, 0.8164965809277260, -0.4082482904638630,
    0.0, 0.4472135954999580, 0.8944271909999159
), nrow = 2, ncol = 3, byrow = TRUE)

expected_variance <- c(5.999999999999999, 1.6666666666666665)
expected_mean <- c(2.0, 4.333333333333333, 2.6666666666666665)
expected_sigma <- c(3.4641016151377544, 1.8257418583505536)

expected_u <- matrix(c(
    -7.0710678118654724e-01, -4.0824829046386296e-01,
    3.7336028004349413e-17, 8.1649658092772626e-01,
    7.0710678118654746e-01, -4.0824829046386302e-01
), nrow = 3, ncol = 2, byrow = TRUE)

expected_scores <- matrix(c(
    -2.4494897427831770e+00, -7.4535599249992979e-01,
    1.2933579491269523e-16, 1.4907119849998600e+00,
    2.4494897427831779e+00, -7.4535599249992990e-01
), nrow = 3, ncol = 2, byrow = TRUE)

expected_reconstruction <- matrix(c(
    0.5, 1.2000000000000002, 3.8999999999999995,
    1.3333333333333335, 2.6666666666666665, 2.6666666666666665,
    0.8333333333333333, 1.6666666666666665, 3.1666666666666665,
    0.6666666666666665, 0.1333333333333329, 0.9333333333333333
), nrow = 4, ncol = 3, byrow = TRUE)

test_that("PCA constructor defaults work", {
    m <- PCA$new()
    expect_s3_class(m, "PCA")
})

test_that("PCA constructor all params accepted", {
    m <- PCA$new(
        n_components = 2, bias = "biased", method = "covariance",
        solver = "gesdd", store_U = TRUE, whiten = FALSE, check_data = TRUE
    )
    expect_s3_class(m, "PCA")
})

test_that("PCA public API returns expected fitted values", {
    m <- PCA$new(n_components = 2, store_U = TRUE)
    m$fit(x_train)

    expect_equal(abs(m$principal_components), abs(expected_components), tolerance = tol)
    expect_equal(abs(m$vt), abs(expected_components), tolerance = tol)
    expect_equal(m$column_means, expected_mean, tolerance = tol)
    expect_equal(m$sigma, expected_sigma, tolerance = tol)
    expect_equal(m$variance, expected_variance, tolerance = tol)
    expect_equal(abs(m$u), abs(expected_u), tolerance = tol)
    expect_equal(abs(m$scores), abs(expected_scores), tolerance = tol)
    expect_equal(as.numeric(m$total_variance)[1], sum(expected_variance), tolerance = tol)
    expect_equal(m$n_samples, 3L)
    expect_equal(m$n_features, 3L)
    expect_equal(m$n_components, 2L)
})

test_that("PCA inverse_transform reproduces the expected reconstruction", {
    m <- PCA$new(n_components = 2, store_U = TRUE)
    m$fit(x_train)

    Xt <- m$transform(x_test)
    expect_equal(dim(Xt), c(4L, 2L))
    expect_equal(m$inverse_transform(Xt), expected_reconstruction, tolerance = tol)
})

test_that("PCA correlation method exposes exact column_sdevs on simple data", {
    X <- matrix(c(
        1, 1, 1,
        2, 2, 2,
        3, 3, 3
    ), nrow = 3, ncol = 3, byrow = TRUE)

    m <- PCA$new(n_components = 3, method = "correlation")
    m$fit(X)
    expect_equal(m$column_sdevs, c(1, 1, 1), tolerance = tol)
})

test_that("PCA works with correlation and svd methods", {
    for (meth in c("correlation", "svd")) {
        m <- PCA$new(n_components = 2, method = meth)
        expect_no_error(m$fit(x_train))
        expect_equal(nrow(m$transform(x_train)), nrow(x_train))
    }
})

test_that("PCA accepts data.frame input", {
    df <- as.data.frame(x_train)
    m <- PCA$new(n_components = 2)
    expect_no_error(m$fit(df))
})

test_that("PCA whiten=TRUE produces unit-covariance transformed data", {
    X <- matrix(c(1,0,2, 0,3,1, 4,1,0, 2,2,3, 1,4,2), nrow = 5, ncol = 3, byrow = TRUE)
    m <- PCA$new(n_components = 2, whiten = TRUE, store_U = TRUE)
    m$fit(X)
    Xt <- m$transform(X)
    cov_Xt <- cov(Xt)
    expect_equal(diag(cov_Xt), c(1, 1), tolerance = 1e-6)
})

test_that("PCA bias=biased runs and returns smaller variance", {
    m_unbiased <- PCA$new(n_components = 2, bias = "unbiased")
    m_unbiased$fit(x_train)
    m_biased <- PCA$new(n_components = 2, bias = "biased")
    m_biased$fit(x_train)
    expect_true(all(m_biased$variance <= m_unbiased$variance))
})

test_that("PCA all solver options run without error", {
    for (slv in c("syevd", "gesvd", "randomized")) {
        X <- matrix(rnorm(200), nrow = 50, ncol = 4)
        m <- PCA$new(n_components = 2, solver = slv, seed = 42L)
        expect_no_error(m$fit(X))
        expect_length(m$variance, 2)
    }
})
