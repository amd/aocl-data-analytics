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

tol <- 1e-10

# ============================================================================
# fit / evaluate
# ============================================================================
test_that("spline reproduces the data values at the interpolation sites", {
    for (spline_type in c("natural", "clamped zero")) {
        x <- seq(0, 2 * pi, length.out = 11)
        y <- sin(x)

        spline <- cubic_spline$new(spline_type = spline_type)
        spline$fit(x, y)

        values <- spline$evaluate(x)
        expect_equal(length(values), length(x))
        expect_null(dim(values))
        expect_equal(values, y, tolerance = tol)
    }
})

test_that("single-order evaluation returns a vector for 1D data", {
    x <- seq(0, 1, length.out = 6)
    y <- x^2

    spline <- cubic_spline$new()
    spline$fit(x, y)

    x_eval <- seq(0, 1, length.out = 20)
    values <- spline$evaluate(x_eval, orders = 0)
    expect_equal(length(values), 20)
    expect_null(dim(values))

    first_deriv <- spline$evaluate(x_eval, orders = 1)
    expect_equal(length(first_deriv), 20)
})

test_that("multiple orders retain an axis and respect the requested order", {
    x <- seq(0, 1, length.out = 6)
    y <- x^3

    spline <- cubic_spline$new()
    spline$fit(x, y)

    x_eval <- seq(0, 1, length.out = 7)
    result <- spline$evaluate(x_eval, orders = c(0, 1, 2))
    expect_equal(dim(result), c(3, 7))

    # Order of the returned rows must match the requested orders.
    reordered <- spline$evaluate(x_eval, orders = c(2, 0))
    expect_equal(dim(reordered), c(2, 7))
    expect_equal(reordered[2, ], result[1, ], tolerance = tol)
    expect_equal(reordered[1, ], result[3, ], tolerance = tol)
})

test_that("multiple dimensions can be interpolated with a single set of sites", {
    x <- seq(0, 1, length.out = 8)
    y <- cbind(sin(x), cos(x), x^2)

    spline <- cubic_spline$new()
    spline$fit(x, y)
    expect_equal(spline$n_dim, 3)

    values <- spline$evaluate(x)
    expect_equal(dim(values), c(3, length(x)))
    expect_equal(values, t(y), tolerance = tol)
})

# ============================================================================
# coefficients
# ============================================================================
test_that("coefficients have shape (n_cells, 4) for 1D data", {
    x <- seq(0, 1, length.out = 5)
    y <- sin(x)

    spline <- cubic_spline$new()
    spline$fit(x, y)

    coeffs <- spline$coefficients
    n_cells <- length(x) - 1
    expect_equal(dim(coeffs), c(n_cells, 4))

    # The constant coefficient of each cell equals the value at its left site.
    expect_equal(coeffs[, 1], y[1:n_cells], tolerance = tol)
})

test_that("coefficients raise before the spline is fitted", {
    spline <- cubic_spline$new()
    expect_error(spline$coefficients)
})

# ============================================================================
# hermite / boundary conditions
# ============================================================================
test_that("Hermite splines accept user-supplied first derivatives", {
    x <- seq(0, 2 * pi, length.out = 9)
    y <- sin(x)
    dydx <- cos(x)

    spline <- cubic_spline$new(spline_type = "hermite")
    spline$fit(x, y, y_derivative = dydx)

    values <- spline$evaluate(x)
    derivs <- spline$evaluate(x, orders = 1)
    expect_equal(values, y, tolerance = tol)
    expect_equal(derivs, dydx, tolerance = tol)
})

test_that("custom boundary conditions impose the requested end derivatives", {
    x <- seq(0, 1, length.out = 6)
    y <- x^2

    # Clamp the first derivative to 0 at the left and 2 at the right.
    bcs <- list(list(1, 0.0), list(1, 2.0))

    spline <- cubic_spline$new(spline_type = "custom")
    spline$fit(x, y, boundary_conditions = bcs)

    deriv <- spline$evaluate(c(0.0, 1.0), orders = 1)
    expect_equal(deriv, c(0.0, 2.0), tolerance = tol)
})

test_that("supplying only one of the left/right boundary values raises an error", {
    x <- seq(0, 1, length.out = 6)
    y <- x^2

    # Only the left boundary value is supplied; the right is missing.
    bcs <- list(list(1, 0.0), list(1, NULL))

    spline <- cubic_spline$new(spline_type = "custom")
    expect_error(spline$fit(x, y, boundary_conditions = bcs))
})

# ============================================================================
# search_cells
# ============================================================================
test_that("search_cells returns the containing cell index for each query point", {
    x <- c(0, 1, 2, 3)
    y <- c(0, 1, 4, 9)

    spline <- cubic_spline$new()
    spline$fit(x, y)

    cells <- spline$search_cells(c(0.5, 1.5, 2.5))
    expect_equal(cells, c(0L, 1L, 2L))

    # Out-of-bounds points are clamped to the first and last cells.
    clamped <- spline$search_cells(c(-1.0, 10.0))
    expect_equal(clamped, c(0L, 2L))
})
