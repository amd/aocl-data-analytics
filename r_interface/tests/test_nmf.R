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
# Nonnegative matrix factorization (NMF)
# ============================================================================

frob <- function(M) sqrt(sum(M^2))

test_that("NMF constructor defaults work", {
    m <- NMF$new()
    expect_s3_class(m, "NMF")
})

test_that("NMF public API returns expected factors for each init method", {
    # Construct the factors W and H and use them to create matrix A.
    W_true <- matrix(c(2, 0, 2, 0, 0, 3, 0, 3), nrow = 4, ncol = 2, byrow = TRUE)
    H_true <- matrix(c(1, 2, 0, 0, 0, 1), nrow = 2, ncol = 3, byrow = TRUE)
    A <- W_true %*% H_true

    solve_tol <- .Machine$double.eps * 1000
    tol <- 5e-7
    num_tol <- sqrt(.Machine$double.eps) * 10

    # X_new = [1, 1] @ H_true = [1, 2, 1] lies in the column space of H_true, so
    # inverse_transform(transform(X_new)) recovers X_new exactly.
    X_new <- matrix(c(1, 2, 1), nrow = 1, ncol = 3)
    expected_rec_new <- matrix(c(1, 2, 1), nrow = 1, ncol = 3)

    for (init in c("nndsvd", "nndsvda", "nndsvdar", "random", "custom")) {
        W0 <- NULL
        H0 <- NULL
        if (init == "custom") {
            set.seed(42)
            W0 <- matrix(runif(8), nrow = 4, ncol = 2)
            H0 <- matrix(runif(6), nrow = 2, ncol = 3)
        }

        m <- NMF$new(n_components = 2L, solver = "mu", beta_loss = "frobenius",
                     max_iter = 500L, tol = solve_tol, init = init, random_state = 0L)
        m$fit(A, W = W0, H = H0)

        H <- m$h
        W <- m$w
        WH <- W %*% H
        expect_lt(frob(WH - A), tol)
        expect_lt(abs(m$reconstruction_err - frob(A - WH)), num_tol)
        expect_equal(dim(H), c(2L, 3L))
        expect_equal(dim(W), c(4L, 2L))
        expect_true(all(H >= 0.0))
        expect_true(all(W >= 0.0))
        expect_equal(m$n_components, 2L)
        expect_equal(m$n_samples, 4L)
        expect_equal(m$n_features, 3L)

        # transform(A) re-solves for W with H fixed; must converge close to stored W.
        Xt <- m$transform(A)
        expect_equal(dim(Xt), c(4L, 2L))
        expect_true(all(Xt >= 0.0))
        expect_lt(frob(Xt - W) / frob(W), tol)

        A_rec <- m$inverse_transform(Xt)
        expect_equal(dim(A_rec), dim(A))
        expect_true(all(A_rec >= 0.0))

        Xt_new <- m$transform(X_new)
        expect_equal(dim(Xt_new), c(1L, 2L))
        expect_true(all(Xt_new >= 0.0))
        X_rec <- m$inverse_transform(Xt_new)
        expect_equal(dim(X_rec), dim(X_new))
        expect_true(all(X_rec >= 0.0))
        expect_lt(frob(X_rec - expected_rec_new) / frob(expected_rec_new), tol)

        # fit_transform() must return exactly the stored W.
        m2 <- NMF$new(n_components = 2L, solver = "mu", beta_loss = "frobenius",
                      max_iter = 500L, tol = solve_tol, init = init, random_state = 0L)
        W_ft <- m2$fit_transform(A, W = W0, H = H0)
        expect_equal(dim(W_ft), c(4L, 2L))
        expect_true(all(W_ft >= 0.0))
        expect_equal(W_ft, m2$w)
        WH_2 <- W_ft %*% m2$h
        expect_lt(frob(WH_2 - A), tol)
    }
})

test_that("NMF n_components 'auto' and NULL resolve to min(n_samples, n_features)", {
    for (n_components in list("auto", NULL)) {
        # Tall matrix: min(n_samples = 4, n_features = 3) = 3
        A_tall <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                         nrow = 4, ncol = 3, byrow = TRUE)
        m <- NMF$new(n_components = n_components)
        m$fit(A_tall)
        expect_equal(m$n_components, 3L)
        expect_equal(m$n_samples, 4L)
        expect_equal(m$n_features, 3L)
        expect_equal(dim(m$w), c(4L, 3L))
        expect_true(all(m$w >= 0.0))
        expect_equal(dim(m$h), c(3L, 3L))
        Xt <- m$transform(A_tall)
        expect_equal(dim(Xt), c(4L, 3L))
        expect_true(all(Xt >= 0.0))

        # Wide matrix: min(n_samples = 2, n_features = 5) = 2
        A_wide <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                         nrow = 2, ncol = 5, byrow = TRUE)
        m <- NMF$new(n_components = n_components)
        m$fit(A_wide)
        expect_equal(m$n_components, 2L)
        expect_equal(m$n_samples, 2L)
        expect_equal(m$n_features, 5L)
        expect_equal(dim(m$w), c(2L, 2L))
        expect_equal(dim(m$h), c(2L, 5L))
        X_new <- matrix(c(2, 4, 6, 8, 10), nrow = 1, ncol = 5)
        Xt <- m$transform(X_new)
        expect_equal(dim(Xt), c(1L, 2L))
        expect_true(all(Xt >= 0.0))
    }
})

test_that("NMF custom W and H are accepted", {
    set.seed(42)
    A <- matrix(c(1.3, 2.53, 3.86, 2.4, 5.5, 4.5,
                  3.33, 6.21, 1.76, 1.0, 2.0, 3.0),
                nrow = 4, ncol = 3, byrow = TRUE)
    W0 <- matrix(runif(8), nrow = 4, ncol = 2)
    H0 <- matrix(runif(6), nrow = 2, ncol = 3)

    m <- NMF$new(n_components = 2L, init = "custom", random_state = 0L, max_iter = 500L)
    m$fit(A, W = W0, H = H0)
    expect_equal(dim(m$w), c(4L, 2L))
    expect_equal(dim(m$h), c(2L, 3L))
    expect_true(all(m$w >= 0.0))
    expect_true(all(m$h >= 0.0))

    W_ft <- m$fit_transform(A, W = W0, H = H0)
    expect_equal(W_ft, m$w)
})

test_that("NMF error exits", {
    # Invalid n_components
    expect_error(NMF$new(n_components = 0))
    expect_error(NMF$new(n_components = -1))
    expect_error(NMF$new(n_components = 1.5))
    expect_error(NMF$new(n_components = "bad"))

    # Invalid random_state
    expect_error(NMF$new(random_state = 1.5))
    expect_error(NMF$new(random_state = "seed"))

    # Invalid check_every
    expect_error(NMF$new(check_every = 0))
    expect_error(NMF$new(check_every = -1))
    expect_error(NMF$new(check_every = 1.5))

    # Supplying only W or only H must raise an error.
    A <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2, byrow = TRUE)
    m <- NMF$new(n_components = 1L)
    expect_error(m$fit_transform(A, W = matrix(1, nrow = 2, ncol = 1)))
    expect_error(m$fit_transform(A, H = matrix(1, nrow = 1, ncol = 2)))
    expect_error(m$fit(A, W = matrix(1, nrow = 2, ncol = 1)))
    expect_error(m$fit(A, H = matrix(1, nrow = 1, ncol = 2)))
})

test_that("NMF accepts data.frame input", {
    A <- as.data.frame(matrix(c(1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12),
                              nrow = 3, ncol = 4, byrow = TRUE))
    m <- NMF$new(n_components = 2L, random_state = 0L)
    expect_no_error(m$fit(A))
    expect_equal(m$n_components, 2L)
})
