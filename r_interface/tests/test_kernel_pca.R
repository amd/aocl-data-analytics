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

A <- matrix(c(
    1.0,  2.0, -1.0,
    3.0, -1.0,  0.5,
    0.5,  1.5,  2.0,
   -2.0,  0.0,  1.0,
    1.5, -0.5, -1.5,
   -1.0,  1.0,  0.0
), nrow = 6, ncol = 3, byrow = TRUE)

X <- matrix(c(
    0.5, -1.0,  1.5,
   -0.5,  2.0, -1.0,
    1.0,  0.0,  0.5
), nrow = 3, ncol = 3, byrow = TRUE)

expected_eigenvalues_linear <- c(
    1.83055715040013816e+01, 7.30102995247190556e+00, 5.72673187686005924e+00)

expected_eigenvectors_linear <- matrix(c(
    8.62948669540869462e-02, 6.06456873889861248e-01, -2.05409257584859228e-01,
   -5.52530436625913945e-01, 4.03508842196679263e-01, -3.38320888829854005e-01,
    5.26237765855806838e-01, -4.89792827467531167e-01, -5.16718413116041830e-01,
   -1.33917212983749284e-01, 3.76485404837801518e-01,  2.37705282873712398e-01,
    5.45314575607418472e-01, -1.48924748101957716e-01,  5.35040989459531158e-01,
   -5.07508208089862967e-01, -3.64874355212676194e-01, -5.90482536624547596e-02
), nrow = 6, ncol = 3)

expected_scores_linear <- matrix(c(
    3.69212681586973235e-01,  2.59472639079286527e+00, -8.78843730717443372e-01,
   -2.36400207064011569e+00,  1.72641301771466504e+00, -1.44750628873694298e+00,
    1.42191644626689495e+00, -1.32344069891494143e+00, -1.39619475714320584e+00,
   -3.61849908795781716e-01,  1.01727930538722422e+00,  6.42289613199805709e-01,
    1.30497000988612277e+00, -3.56385724306790885e-01,  1.28038471102070939e+00,
   -1.21449713789624814e+00, -8.73165897681538516e-01, -1.41305961022259535e-01
), nrow = 6, ncol = 3)

expected_transform_linear <- matrix(c(
    2.82450111998761681e-02, -9.81122101044141104e-01,  4.94629198892678312e-01,
   -1.68857555355862798e+00,  1.75868547965963362e+00, -5.72179687098797740e-01,
   -1.08429359356415156e+00,  7.45342818834525689e-01, -1.97644813707366113e-01
), nrow = 3, ncol = 3)

expected_inv_transform_linear <- matrix(c(
    4.48863636363635132e-01,  2.36297928262213919e+00, -6.77952999381563198e-02,
   -2.33219310451453232e+00,  9.95439084724799250e-01, -1.40729359925788500e+00,
    1.27840909090908750e+00, -1.36359771181199863e+00,  8.64950525664811343e-01,
   -3.60138373531232348e-01, -8.92779839208410642e-01,  4.73156307977736668e-01,
   -1.04166666666666607e+00,  2.38945578231292755e-01,  1.62414965986394622e+00,
    7.91241496598640293e-01, -1.49829931972789132e+00, -1.14370748299320424e-01
), nrow = 6, ncol = 3)

test_that("KernelPCA constructor defaults work", {
    m <- KernelPCA$new()
    expect_s3_class(m, "KernelPCA")
})

test_that("KernelPCA constructor all params accepted", {
    m <- KernelPCA$new(
        n_components = 2L, kernel = "rbf", eigensolver = "syevd",
        gamma = 0.5, degree = 3L, coef0 = 1.0,
        fit_inverse_transform = FALSE, alpha = 1.0,
        remove_zero_eig = FALSE, copy_X = TRUE,
        n_oversamples = 10L, power_iterations = -1L,
        power_normalization = "QR", seed = -1L, check_data = FALSE
    )
    expect_s3_class(m, "KernelPCA")
})

test_that("KernelPCA linear kernel returns expected values", {
    m <- KernelPCA$new(n_components = 3L, kernel = "linear",
                       fit_inverse_transform = TRUE, eigensolver = "syevd")
    m$fit(A)

    expect_equal(m$n_samples, 6L)
    expect_equal(m$n_features, 3L)
    expect_equal(m$n_components, 3L)

    expect_equal(m$eigenvalues, expected_eigenvalues_linear, tolerance = tol)
    expect_equal(abs(m$eigenvectors), abs(expected_eigenvectors_linear), tolerance = tol)
    expect_equal(abs(m$scores), abs(expected_scores_linear), tolerance = tol)
    expect_equal(m$gamma_, 1.0 / 3.0, tolerance = tol)
    expect_true(is.matrix(m$dual_coef))
    expect_equal(m$X_fit_, A, tolerance = tol)
})

test_that("KernelPCA linear kernel transform returns expected values", {
    m <- KernelPCA$new(n_components = 3L, kernel = "linear",
                       fit_inverse_transform = TRUE, eigensolver = "syevd")
    m$fit(A)

    transformed <- m$transform(X)
    expect_equal(nrow(transformed), 3L)
    expect_equal(ncol(transformed), 3L)
    expect_equal(abs(transformed), abs(expected_transform_linear), tolerance = tol)
})

test_that("KernelPCA linear kernel inverse_transform returns expected values", {
    m <- KernelPCA$new(n_components = 3L, kernel = "linear",
                       fit_inverse_transform = TRUE, eigensolver = "syevd")
    m$fit(A)

    transformed <- m$transform(A)
    reconstructed <- m$inverse_transform(transformed)
    expect_equal(nrow(reconstructed), 6L)
    expect_equal(ncol(reconstructed), 3L)
    expect_equal(reconstructed, expected_inv_transform_linear, tolerance = tol)
})

test_that("KernelPCA rbf kernel produces positive eigenvalues", {
    m <- KernelPCA$new(n_components = 5L, kernel = "rbf", gamma = 0.5,
                       eigensolver = "syevd")
    m$fit(A)

    expect_true(all(m$eigenvalues > 0))
    expect_equal(length(m$eigenvalues), 5L)
    expect_equal(m$gamma_, 0.5, tolerance = tol)
})

test_that("KernelPCA polynomial kernel", {
    m <- KernelPCA$new(n_components = 3L, kernel = "poly", degree = 2L,
                       gamma = 0.5, coef0 = 1.0, fit_inverse_transform = TRUE)
    m$fit(A)
    transformed <- m$transform(X)

    expect_equal(nrow(transformed), 3L)
    expect_equal(ncol(transformed), 3L)
})

test_that("KernelPCA with randomized eigensolver", {
    m <- KernelPCA$new(n_components = 2L, kernel = "rbf", gamma = 0.5,
                       eigensolver = "randomized", seed = 42L)
    m$fit(A)
    expect_equal(m$n_components, 2L)
})

test_that("KernelPCA kernel=sigmoid matches known values", {
    X <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                nrow = 4, ncol = 3, byrow = TRUE)
    m <- KernelPCA$new(n_components = 2L, kernel = "sigmoid",
                       gamma = 0.01, coef0 = 0.0)
    m$fit(X)
    expect_equal(m$n_components, 2L)
    expect_equal(m$gamma_, 0.01)
    expect_equal(m$eigenvalues,
                 c(0.1260169729481583, 0.00046525148152055273),
                 tolerance = 1e-6)
})
