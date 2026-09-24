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

# shared test data
m34 <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), nrow = 3, ncol = 4)
pos <- matrix(c(1, 2, 4, 8, 16), nrow = 5, ncol = 1) # all positive
m_sq <- matrix(c(
    1.23, -0.45, 0.78, 0.34,
    -0.87, 1.56, -1.23, -0.67,
    2.14, 0.23, 0.91, 1.47,
    -1.45, -0.78, 0.14, -1.23,
    0.56, 1.89, -0.43, 0.81,
    -0.92, -0.34, 1.67, -0.45,
    1.78, 0.91, -0.54, 0.92,
    -0.63, -1.12, -0.89, -0.34,
    0.94, 0.45, 1.34, 0.56,
    -1.17, -0.67, -0.34, -0.78,
    0.73, 1.24, 0.47, 1.14,
    -0.48, -0.89, -1.67, -0.56
), nrow = 12, ncol = 4, byrow = TRUE)

# ---------------------------------------------------------------------------
# da_mean
# ---------------------------------------------------------------------------
test_that("da_mean returns correct types and shapes", {
    r <- da_mean(m34, axis = "col")
    expect_type(r, "double")
    expect_length(r, 4)

    r_row <- da_mean(m34, axis = "row")
    expect_length(r_row, 3)

    r_all <- da_mean(m34, axis = "all")
    expect_length(r_all, 1)
})

test_that("da_mean equals known values for col, row and all axes", {
    x <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3, byrow = TRUE)
    expect_equal(da_mean(x, "col"), c(2.5, 3.5, 4.5))
    expect_equal(da_mean(x, "row"), c(2.0, 5.0))
    expect_equal(da_mean(x, "all"), 3.5)
})

test_that("da_mean accepts integer matrix input", {
    x <- matrix(1L:12L, nrow = 3, ncol = 4)
    expect_equal(da_mean(x, axis = "col"), c(2, 5, 8, 11))
    expect_equal(da_mean(x, axis = "row"), c(5.5, 6.5, 7.5))
    expect_equal(drop(da_mean(x, axis = "all")), 6.5)
})

test_that("da_mean works on 1D vector and single value", {
    v <- c(1, 2, 3, 4, 5)
    expect_equal(drop(suppressWarnings(da_mean(v, "col"))), 3.0)
    expect_equal(drop(suppressWarnings(da_mean(v, "row"))), 3.0)
    expect_equal(drop(da_mean(v, "all")), 3.0)

    expect_equal(drop(suppressWarnings(da_mean(42.0, "col"))), 42.0)
    expect_equal(drop(suppressWarnings(da_mean(matrix(42.0, 1, 1), "col"))), 42.0)
})

# ---------------------------------------------------------------------------
# da_geometric_mean
# ---------------------------------------------------------------------------
test_that("da_geometric_mean returns correct shapes", {
    r <- da_geometric_mean(abs(m34), axis = "col")
    expect_length(r, 4)
    expect_true(all(r > 0))

    r_row <- da_geometric_mean(abs(m34), axis = "row")
    expect_length(r_row, 3)
    expect_true(all(r_row > 0))

    r_all <- da_geometric_mean(abs(m34), axis = "all")
    expect_length(r_all, 1)
})

test_that("da_geometric_mean equals known values for col, row and all axes", {
    x <- matrix(c(1, 2, 4, 8, 16, 32), nrow = 2, ncol = 3, byrow = TRUE)
    expect_equal(
        da_geometric_mean(x, "col"),
        sqrt(c(8, 32, 128)),
        tolerance = 1e-5
    )
    expect_equal(
        da_geometric_mean(x, "row"),
        c(2.0, 16.0),
        tolerance = 1e-5
    )
    expect_equal(
        drop(da_geometric_mean(x, "all")),
        2^2.5,
        tolerance = 1e-5
    )
})

test_that("da_geometric_mean works on 1D vector and single value", {
    expect_equal(drop(suppressWarnings(da_geometric_mean(c(1, 2, 4), "col"))), 2.0, tolerance = 1e-10)

    expect_equal(drop(suppressWarnings(da_geometric_mean(5.0, "col"))), 5.0)
    expect_equal(drop(suppressWarnings(da_geometric_mean(matrix(5.0, 1, 1), "col"))), 5.0)
})

# ---------------------------------------------------------------------------
# da_harmonic_mean
# ---------------------------------------------------------------------------
test_that("da_harmonic_mean returns correct shapes", {
    x <- matrix(c(1, 2, 4, 8, 16, 32), nrow = 2, ncol = 3, byrow = TRUE)
    r <- da_harmonic_mean(x, axis = "col")
    expect_length(r, 3)
    expect_true(all(r > 0))

    r_row <- da_harmonic_mean(x, axis = "row")
    expect_length(r_row, 2)
    expect_true(all(r_row > 0))

    r_all <- da_harmonic_mean(x, axis = "all")
    expect_length(r_all, 1)
})

test_that("da_harmonic_mean equals known values for col, row and all axes", {
    x <- matrix(c(1, 2, 4, 8, 16, 32), nrow = 2, ncol = 3, byrow = TRUE)
    expect_equal(
        da_harmonic_mean(x, "col"),
        c(16 / 9, 32 / 9, 64 / 9),
        tolerance = 1e-10
    )
    expect_equal(
        da_harmonic_mean(x, "row"),
        c(12 / 7, 96 / 7),
        tolerance = 1e-10
    )
    expect_equal(
        drop(da_harmonic_mean(x, "all")),
        6 / sum(1 / c(1, 2, 4, 8, 16, 32)),
        tolerance = 1e-10
    )
})

test_that("da_harmonic_mean works on 1D vector and single value", {
    expect_equal(drop(suppressWarnings(da_harmonic_mean(c(1, 2, 4), "col"))),
                 3 / (1 + 0.5 + 0.25), tolerance = 1e-10)

    expect_equal(drop(suppressWarnings(da_harmonic_mean(7.0, "col"))), 7.0)
    expect_equal(drop(suppressWarnings(da_harmonic_mean(matrix(7.0, 1, 1), "col"))), 7.0)
})

# ---------------------------------------------------------------------------
# da_variance
# ---------------------------------------------------------------------------
test_that("da_variance returns correct shapes for all axes", {
    expect_length(da_variance(m34, dof = 0, axis = "col"), 4)
    expect_length(da_variance(m34, dof = 0, axis = "row"), 3)
    expect_length(da_variance(m34, dof = 0, axis = "all"), 1)
})

test_that("da_variance dof=0 equals known values for col, row and all axes", {
    x <- matrix(c(2, 4, 4, 4, 5, 5, 7, 9), nrow = 2, ncol = 4, byrow = TRUE)
    expect_equal(da_variance(x, dof = 0, axis = "col"), c(4.5, 0.5, 4.5, 12.5), tolerance = 1e-10)
    expect_equal(da_variance(x, dof = 0, axis = "row"), c(1.0, 11 / 3), tolerance = 1e-10)
    expect_equal(drop(da_variance(x, dof = 0, axis = "all")), 32 / 7, tolerance = 1e-10)
})

test_that("da_variance dof=2 equals known values for col, row and all axes", {
    x <- matrix(c(2, 4, 4, 4, 5, 5, 7, 9), nrow = 2, ncol = 4, byrow = TRUE)
    expect_equal(da_variance(x, dof = 2, axis = "col"), c(2.25, 0.25, 2.25, 6.25), tolerance = 1e-10)
    expect_equal(da_variance(x, dof = 2, axis = "row"), c(1.5, 5.5), tolerance = 1e-10)
    expect_equal(drop(da_variance(x, dof = 2, axis = "all")), 16, tolerance = 1e-10)
})

test_that("da_variance of constant data is zero for col, row and all axes", {
    x <- matrix(rep(5, 6), nrow = 2, ncol = 3)
    expect_equal(da_variance(x, dof = 0, axis = "col"), c(0, 0, 0), tolerance = 1e-14)
    expect_equal(da_variance(x, dof = 0, axis = "row"), c(0, 0), tolerance = 1e-14)
    expect_equal(drop(da_variance(x, dof = 0, axis = "all")), 0, tolerance = 1e-14)
})

test_that("da_variance works on 1D vector and single value", {
    expect_equal(drop(suppressWarnings(da_variance(c(2, 4, 6), dof = 0, axis = "col"))),
                 4.0, tolerance = 1e-10)

    expect_equal(drop(suppressWarnings(da_variance(3.0, dof = -1, axis = "col"))), 0.0)
    expect_equal(drop(suppressWarnings(da_variance(matrix(3.0, 1, 1), dof = -1, axis = "col"))), 0.0)
})

# ---------------------------------------------------------------------------
# da_skewness
# ---------------------------------------------------------------------------
test_that("da_skewness returns correct shapes", {
    expect_length(da_skewness(m34, axis = "col"), 4)
    expect_length(da_skewness(m34, axis = "row"), 3)
    expect_length(da_skewness(m34, axis = "all"), 1)
})

test_that("da_skewness of symmetric data is near zero for col, row and all axes", {
    x <- matrix(c(-3, -2, -1, 1, 2, 3), nrow = 2, ncol = 3, byrow = TRUE)
    expect_equal(da_skewness(x, axis = "col"), c(0, 0, 0), tolerance = 1e-10)
    expect_equal(da_skewness(x, axis = "row"), c(0, 0), tolerance = 1e-10)
    expect_equal(drop(da_skewness(x, axis = "all")), 0, tolerance = 1e-10)
})

test_that("da_skewness equals known values for col, row and all axes", {
    x <- matrix(c(0, 0, 0, 0, 0, 1), nrow = 2, ncol = 3, byrow = TRUE)
    expect_equal(da_skewness(x, axis = "col"), c(0, 0, 0), tolerance = 1e-10)
    expect_equal(da_skewness(x, axis = "row"), c(0, 1 / sqrt(2)), tolerance = 1e-10)
    expect_equal(drop(da_skewness(x, axis = "all")), 4 / sqrt(5), tolerance = 1e-10)
})

test_that("da_skewness works on 1D vector", {
    expect_equal(drop(suppressWarnings(da_skewness(c(-1, 0, 1), axis = "col"))), 0.0, tolerance = 1e-10)
})

# ---------------------------------------------------------------------------
# da_kurtosis
# ---------------------------------------------------------------------------
test_that("da_kurtosis returns correct shapes", {
    expect_length(da_kurtosis(m34, axis = "col"), 4)
    expect_length(da_kurtosis(m34, axis = "row"), 3)
    expect_length(da_kurtosis(m34, axis = "all"), 1)
})

test_that("da_kurtosis returns numeric", {
    r <- da_kurtosis(m_sq, axis = "col")
    expect_type(r, "double")

    r_row <- da_kurtosis(m_sq, axis = "row")
    expect_type(r_row, "double")

    r_all <- da_kurtosis(m_sq, axis = "all")
    expect_type(r_all, "double")
})

test_that("da_kurtosis equals known values for col, row and all axes", {
    x <- matrix(c(-1, -1, -1, 1, 1, 1), nrow = 2, ncol = 3, byrow = TRUE)
    expect_equal(da_kurtosis(x, axis = "col"), c(-2, -2, -2), tolerance = 1e-10)
    expect_equal(da_kurtosis(x, axis = "row"), c(-3, -3), tolerance = 1e-10)
    expect_equal(drop(da_kurtosis(x, axis = "all")), -2.0, tolerance = 1e-10)
})

test_that("da_kurtosis works on 1D vector", {
    r <- suppressWarnings(da_kurtosis(c(-1, 0, 1), axis = "col"))
    expect_type(r, "double")
})

# ---------------------------------------------------------------------------
# da_moment
# ---------------------------------------------------------------------------
test_that("da_moment order=1 is zero for col, row and all axes", {
    x <- matrix(c(
        0.37, -0.82, 1.13,
        -1.13, 0.82, -0.37
    ), nrow = 2, ncol = 3, byrow = TRUE)
    expect_equal(da_moment(x, k = 1, axis = "col"), c(0, 0, 0), tolerance = 1e-10)
    expect_equal(da_moment(x, k = 1, axis = "row"), c(0, 0), tolerance = 1e-10)
    expect_equal(drop(da_moment(x, k = 1, axis = "all")), 0, tolerance = 1e-10)
})

test_that("da_moment returns correct shapes", {
    expect_length(da_moment(m34, k = 2, axis = "col"), 4)
    expect_length(da_moment(m34, k = 2, axis = "row"), 3)
    expect_length(da_moment(m34, k = 2, axis = "all"), 1)
})

test_that("da_moment order 2 equals known values for col, row and all axes", {
    x <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3, byrow = TRUE)
    expect_equal(da_moment(x, k = 2, axis = "col"), c(2.25, 2.25, 2.25), tolerance = 1e-10)
    expect_equal(da_moment(x, k = 2, axis = "row"), c(2 / 3, 2 / 3), tolerance = 1e-10)
    expect_equal(drop(da_moment(x, k = 2, axis = "all")), 35 / 12, tolerance = 1e-10)
})

test_that("da_moment works on 1D vector and single value", {
    expect_equal(drop(suppressWarnings(da_moment(c(1, 2, 3), k = 2, axis = "col"))),
                 2 / 3, tolerance = 1e-10)

    expect_equal(drop(suppressWarnings(da_moment(9.0, k = 2, axis = "col"))), 0.0)
    expect_equal(drop(suppressWarnings(da_moment(matrix(9.0, 1, 1), k = 2, axis = "col"))), 0.0)
})

test_that("da_moment with precomputed mean matches auto-computed mean", {
    means_col <- da_mean(m34, axis = "col")
    means_row <- da_mean(m34, axis = "row")
    mean_all <- da_mean(m34, axis = "all")

    expect_equal(
        da_moment(m34, k = 2, mean = means_col, axis = "col"),
        da_moment(m34, k = 2, axis = "col"),
        tolerance = 1e-10
    )
    expect_equal(
        da_moment(m34, k = 2, mean = means_row, axis = "row"),
        da_moment(m34, k = 2, axis = "row"),
        tolerance = 1e-10
    )
    expect_equal(
        da_moment(m34, k = 2, mean = mean_all, axis = "all"),
        da_moment(m34, k = 2, axis = "all"),
        tolerance = 1e-10
    )
})

test_that("da_moment with shifted mean gives moment + 1 for order 2", {
    # For k=2: E[(X-(mu+1))^2] = E[(X-mu)^2] + 1, so shifting the mean by +1
    # should increase the second moment by exactly 1.
    means_col <- da_mean(m34, axis = "col")
    means_row <- da_mean(m34, axis = "row")
    mean_all <- da_mean(m34, axis = "all")

    expect_equal(
        da_moment(m34, k = 2, mean = means_col + 1, axis = "col"),
        da_moment(m34, k = 2, axis = "col") + 1,
        tolerance = 1e-10
    )
    expect_equal(
        da_moment(m34, k = 2, mean = means_row + 1, axis = "row"),
        da_moment(m34, k = 2, axis = "row") + 1,
        tolerance = 1e-10
    )
    expect_equal(
        da_moment(m34, k = 2, mean = mean_all + 1, axis = "all"),
        da_moment(m34, k = 2, axis = "all") + 1,
        tolerance = 1e-10
    )
})

# ---------------------------------------------------------------------------
# da_quantile
# ---------------------------------------------------------------------------
test_that("da_quantile returns correct shapes", {
    expect_length(da_quantile(m34, q = 0.5, axis = "col"), 4)
    expect_length(da_quantile(m34, q = 0.5, axis = "row"), 3)
    expect_length(da_quantile(m34, q = 0.5, axis = "all"), 1)
})

test_that("da_quantile q=0 and q=1 give known minima and maxima for all axes", {
    x <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3, byrow = TRUE)
    expect_equal(da_quantile(x, q = 0, axis = "col"), c(1, 2, 3))
    expect_equal(da_quantile(x, q = 1, axis = "col"), c(4, 5, 6))
    expect_equal(da_quantile(x, q = 0, axis = "row"), c(1, 4))
    expect_equal(da_quantile(x, q = 1, axis = "row"), c(3, 6))
    expect_equal(drop(da_quantile(x, q = 0, axis = "all")), 1)
    expect_equal(drop(da_quantile(x, q = 1, axis = "all")), 6)
})

test_that("da_quantile q=0.5 gives known medians for all axes", {
    x <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3, byrow = TRUE)
    expect_equal(da_quantile(x, q = 0.5, axis = "col"), c(2.5, 3.5, 4.5), tolerance = 1e-10)
    expect_equal(da_quantile(x, q = 0.5, axis = "row"), c(2.0, 5.0), tolerance = 1e-10)
    expect_equal(drop(da_quantile(x, q = 0.5, axis = "all")), 3.5, tolerance = 1e-10)
})

test_that("da_quantile methods agree on median for odd-count axis", {
    # m34 has 3 rows, so column medians (odd count) are unambiguous
    expected_col_median <- c(2, 5, 8, 11)
    methods <- c(
        "inverted_cdf", "averaged_inverted_cdf",
        "hazen", "weibull", "linear",
        "median_unbiased", "normal_unbiased"
    )
    for (m in methods) {
        expect_equal(
            da_quantile(m34, q = 0.5, method = m, axis = "col"),
            expected_col_median,
            tolerance = 1e-10,
            info = paste("method:", m)
        )
    }

    expect_equal(
        da_quantile(m34, q = 0.5, method = "interpolated_inverted_cdf", axis = "col"),
        c(1.5, 4.5, 7.5, 10.5),
        tolerance = 1e-10
    )

    expect_equal(
        da_quantile(m34, q = 0.5, method = "closest_observation", axis = "col"),
        c(1, 4, 7, 10),
        tolerance = 1e-10
    )
})

test_that("da_quantile method parameter affects result for even-count axis", {
    # m34 rows have 4 elements (even count), so interpolation methods should differ
    r_linear <- da_quantile(m34, q = 0.5, method = "linear", axis = "row")
    r_inverted <- da_quantile(m34, q = 0.5, method = "inverted_cdf", axis = "row")
    expect_false(isTRUE(all.equal(r_linear, r_inverted)))

    r_all_linear <- da_quantile(m34, q = 0.5, method = "linear", axis = "all")
    r_all_inverted <- da_quantile(m34, q = 0.5, method = "inverted_cdf", axis = "all")
    expect_false(isTRUE(all.equal(r_all_linear, r_all_inverted)))
})

test_that("da_quantile works on 1D vector and single value", {
    expect_equal(drop(suppressWarnings(da_quantile(c(1, 2, 3, 4, 5), q = 0.5, axis = "col"))), 3.0)

    expect_equal(drop(suppressWarnings(da_quantile(4.0, q = 0.5, axis = "col"))), 4.0)
    expect_equal(drop(suppressWarnings(da_quantile(matrix(4.0, 1, 1), q = 0.5, axis = "col"))), 4.0)
})

# ---------------------------------------------------------------------------
# da_five_point_summary
# ---------------------------------------------------------------------------
test_that("da_five_point_summary returns a list of length 5", {
    r <- da_five_point_summary(m34, axis = "col")
    expect_type(r, "list")
    expect_length(r, 5)

    r_row <- da_five_point_summary(m34, axis = "row")
    expect_type(r_row, "list")
    expect_length(r_row, 5)

    r_all <- da_five_point_summary(m34, axis = "all")
    expect_type(r_all, "list")
    expect_length(r_all, 5)
})

test_that("da_five_point_summary gives exact values for all axes", {
    # Use 4x5 so both col (n=4) and row (n=5) axes have distinct Q1/Q3
    # Implementation uses Weibull method (type 6): h = (n+1)*q - 1
    x <- matrix(1:20, nrow = 4, ncol = 5, byrow = TRUE)

    r_col <- da_five_point_summary(x, axis = "col")
    expect_equal(r_col[[1]], c(1, 2, 3, 4, 5)) # min
    expect_equal(r_col[[2]], c(2.25, 3.25, 4.25, 5.25, 6.25)) # Q1
    expect_equal(r_col[[3]], c(8.5, 9.5, 10.5, 11.5, 12.5)) # median
    expect_equal(r_col[[4]], c(14.75, 15.75, 16.75, 17.75, 18.75)) # Q3
    expect_equal(r_col[[5]], c(16, 17, 18, 19, 20)) # max

    r_row <- da_five_point_summary(x, axis = "row")
    expect_equal(r_row[[1]], c(1, 6, 11, 16)) # min
    expect_equal(r_row[[2]], c(1.5, 6.5, 11.5, 16.5)) # Q1
    expect_equal(r_row[[3]], c(3, 8, 13, 18)) # median
    expect_equal(r_row[[4]], c(4.5, 9.5, 14.5, 19.5)) # Q3
    expect_equal(r_row[[5]], c(5, 10, 15, 20)) # max

    r_all <- da_five_point_summary(x, axis = "all")
    expect_equal(drop(r_all[[1]]), 1.0) # min
    expect_equal(drop(r_all[[2]]), 5.25) # Q1
    expect_equal(drop(r_all[[3]]), 10.5) # median
    expect_equal(drop(r_all[[4]]), 15.75) # Q3
    expect_equal(drop(r_all[[5]]), 20.0) # max
})

test_that("da_five_point_summary works on 1D vector", {
    v <- c(1, 2, 3, 4, 5)
    r <- suppressWarnings(da_five_point_summary(v, axis = "col"))
    expect_equal(drop(r[[1]]), 1.0)
    expect_equal(drop(r[[2]]), 1.5)
    expect_equal(drop(r[[3]]), 3.0)
    expect_equal(drop(r[[4]]), 4.5)
    expect_equal(drop(r[[5]]), 5.0)
})

# ---------------------------------------------------------------------------
# da_standardize
# ---------------------------------------------------------------------------
test_that("da_standardize returns matrix of same shape", {
    r <- da_standardize(m_sq, axis = "col")
    expect_equal(dim(r), dim(m_sq))
    expect_type(r, "double")

    r_row <- da_standardize(m_sq, axis = "row")
    expect_equal(dim(r_row), dim(m_sq))
    expect_type(r_row, "double")

    r_all <- da_standardize(m_sq, axis = "all")
    expect_equal(dim(r_all), dim(m_sq))
    expect_type(r_all, "double")
})

test_that("da_standardize centers data for col, row and all axes", {
    r_col <- da_standardize(m_sq, axis = "col")
    expect_equal(colMeans(r_col), rep(0, ncol(m_sq)), tolerance = 1e-10)

    r_row <- da_standardize(m_sq, axis = "row")
    expect_equal(rowMeans(r_row), rep(0, nrow(m_sq)), tolerance = 1e-10)

    r_all <- da_standardize(m_sq, axis = "all")
    expect_equal(mean(r_all), 0, tolerance = 1e-10)
})

test_that("da_standardize with external shift and scale matches default", {
    shift_col <- da_mean(m_sq, axis = "col")
    scale_col <- sqrt(da_variance(m_sq, dof = 0, axis = "col"))
    r_col <- da_standardize(m_sq, shift = shift_col, scale = scale_col, axis = "col")
    expect_equal(r_col, da_standardize(m_sq, axis = "col"), tolerance = 1e-10)

    shift_row <- da_mean(m_sq, axis = "row")
    scale_row <- sqrt(da_variance(m_sq, dof = 0, axis = "row"))
    r_row <- da_standardize(m_sq, shift = shift_row, scale = scale_row, axis = "row")
    expect_equal(r_row, da_standardize(m_sq, axis = "row"), tolerance = 1e-10)

    shift_all <- da_mean(m_sq, axis = "all")
    scale_all <- sqrt(da_variance(m_sq, dof = 0, axis = "all"))
    r_all <- da_standardize(m_sq, shift = shift_all, scale = scale_all, axis = "all")
    expect_equal(r_all, da_standardize(m_sq, axis = "all"), tolerance = 1e-10)
})

test_that("da_standardize reverse recovers original data for all axes", {
    x <- matrix(c(1, 3, 5, 7), nrow = 2, ncol = 2, byrow = TRUE)

    fwd_col <- da_standardize(x, axis = "col")
    shift_col <- da_mean(x, axis = "col")
    scale_col <- sqrt(da_variance(x, dof = 0, axis = "col"))
    rev_col <- da_standardize(fwd_col,
        shift = shift_col, scale = scale_col,
        reverse = TRUE, axis = "col"
    )
    expect_equal(rev_col, x, tolerance = 1e-10)

    fwd_row <- da_standardize(x, axis = "row")
    shift_row <- da_mean(x, axis = "row")
    scale_row <- sqrt(da_variance(x, dof = 0, axis = "row"))
    rev_row <- da_standardize(fwd_row,
        shift = shift_row, scale = scale_row,
        reverse = TRUE, axis = "row"
    )
    expect_equal(rev_row, x, tolerance = 1e-10)

    fwd_all <- da_standardize(x, axis = "all")
    shift_all <- da_mean(x, axis = "all")
    scale_all <- sqrt(da_variance(x, dof = 0, axis = "all"))
    rev_all <- da_standardize(fwd_all,
        shift = shift_all, scale = scale_all,
        reverse = TRUE, axis = "all"
    )
    expect_equal(rev_all, x, tolerance = 1e-10)
})

test_that("da_standardize inplace modifies input and returns same result", {
    x_col <- m_sq + 0
    r_col <- da_standardize(x_col, inplace = TRUE, axis = "col")
    expect_equal(r_col, da_standardize(m_sq, axis = "col"), tolerance = 1e-10)
    expect_equal(x_col, r_col, tolerance = 1e-10)

    x_row <- m_sq + 0
    r_row <- da_standardize(x_row, inplace = TRUE, axis = "row")
    expect_equal(r_row, da_standardize(m_sq, axis = "row"), tolerance = 1e-10)
    expect_equal(x_row, r_row, tolerance = 1e-10)

    x_all <- m_sq + 0
    r_all <- da_standardize(x_all, inplace = TRUE, axis = "all")
    expect_equal(r_all, da_standardize(m_sq, axis = "all"), tolerance = 1e-10)
    expect_equal(x_all, r_all, tolerance = 1e-10)
})

test_that("da_standardize dof=-1 gives exact values for col, row and all axes", {
    # x = [[1,3],[5,7]]; dof=-1 => divisor = n_obs
    # col (n=2): means=[3,5], var=8/2=4, std=2 => [[-1,-1],[1,1]]
    # row (n=2): means=[2,6], var=2/2=1, std=1 => [[-1,1],[-1,1]]
    x <- matrix(c(1, 3, 5, 7), nrow = 2, ncol = 2, byrow = TRUE)

    r_col <- da_standardize(x, dof = -1, axis = "col")
    expect_equal(r_col, matrix(c(-1, -1, 1, 1), nrow = 2, ncol = 2, byrow = TRUE),
        tolerance = 1e-10
    )

    r_row <- da_standardize(x, dof = -1, axis = "row")
    expect_equal(r_row, matrix(c(-1, 1, -1, 1), nrow = 2, ncol = 2, byrow = TRUE),
        tolerance = 1e-10
    )

    # dof=-1 must differ from default dof=0 (divisor n-1=1 vs n=2)
    expect_false(isTRUE(all.equal(r_col, da_standardize(x, axis = "col"))))
    expect_false(isTRUE(all.equal(r_row, da_standardize(x, axis = "row"))))
})

test_that("da_standardize gives known values for col, row and all axes", {
    x <- matrix(c(1, 3, 5, 7), nrow = 2, ncol = 2, byrow = TRUE)
    expect_equal(
        da_standardize(x, axis = "col"),
        matrix(c(
            -1 / sqrt(2), -1 / sqrt(2),
            1 / sqrt(2), 1 / sqrt(2)
        ), nrow = 2, ncol = 2, byrow = TRUE),
        tolerance = 1e-10
    )
    expect_equal(
        da_standardize(x, axis = "row"),
        matrix(c(
            -1 / sqrt(2), 1 / sqrt(2),
            -1 / sqrt(2), 1 / sqrt(2)
        ), nrow = 2, ncol = 2, byrow = TRUE),
        tolerance = 1e-10
    )
    expect_equal(
        da_standardize(x, axis = "all"),
        matrix(c(
            -1.16189500386223, -0.387298334620742,
            0.387298334620742, 1.16189500386223
        ), nrow = 2, ncol = 2, byrow = TRUE),
        tolerance = 1e-10
    )
})

test_that("da_standardize works on 1D vector", {
    v <- c(1, 3, 5)
    r <- suppressWarnings(da_standardize(v, axis = "col"))
    expect_equal(drop(mean(r)), 0.0, tolerance = 1e-10)
})

# ---------------------------------------------------------------------------
# da_covariance_matrix
# ---------------------------------------------------------------------------
test_that("da_covariance_matrix returns square symmetric matrix", {
    r <- da_covariance_matrix(m_sq)
    expect_equal(dim(r), c(4, 4))
    expect_equal(r, t(r), tolerance = 1e-12)
})

test_that("da_covariance_matrix diagonal matches da_variance", {
    v <- da_variance(m_sq, dof = 0, axis = "col")
    cov <- da_covariance_matrix(m_sq, dof = 0)
    expect_equal(diag(cov), v, tolerance = 1e-10)
})

test_that("da_covariance_matrix gives exact values for linearly related columns", {
    # col2 = 2*col1; with dof=0 the implementation uses n_rows - 1 in the divisor.
    x <- matrix(c(1, 2, 3, 2, 4, 6), nrow = 3, ncol = 2)
    cov <- da_covariance_matrix(x, dof = 0)
    expect_equal(cov[1, 1], 1.0, tolerance = 1e-10)
    expect_equal(cov[1, 2], 2.0, tolerance = 1e-10)
    expect_equal(cov[2, 2], 4.0, tolerance = 1e-10)
})

test_that("da_covariance_matrix assume_centered gives exact values", {
    # col2 = 2*col1; assume_centered skips mean subtraction so computes X'X/(n-1)
    # X'X = [[1+4+9, 2+8+18],[2+8+18, 4+16+36]] = [[14,28],[28,56]]
    # dof=0 => divisor = n-1 = 2, so result = [[7,14],[14,28]]
    x <- matrix(c(1, 2, 3, 2, 4, 6), nrow = 3, ncol = 2)
    cov_ac <- da_covariance_matrix(x, dof = 0, assume_centered = TRUE)
    expect_equal(cov_ac[1, 1], 7.0, tolerance = 1e-10)
    expect_equal(cov_ac[1, 2], 14.0, tolerance = 1e-10)
    expect_equal(cov_ac[2, 2], 28.0, tolerance = 1e-10)

    # Must differ from the centered version
    cov_normal <- da_covariance_matrix(x, dof = 0)
    expect_false(isTRUE(all.equal(cov_ac, cov_normal)))
})

test_that("da_covariance_matrix dof parameter changes result", {
    x <- matrix(c(1, 2, 3, 2, 4, 6), nrow = 3, ncol = 2)
    # dof=0 => divisor n-1=2; dof=-1 => divisor n=3
    cov_dof0 <- da_covariance_matrix(x, dof = 0)
    cov_neg <- da_covariance_matrix(x, dof = -1)
    # dof=-1 uses n=3 as divisor: cov = [[2/3, 4/3],[4/3, 8/3]]
    expect_equal(cov_neg[1, 1], 2 / 3, tolerance = 1e-10)
    expect_equal(cov_neg[1, 2], 4 / 3, tolerance = 1e-10)
    expect_equal(cov_neg[2, 2], 8 / 3, tolerance = 1e-10)
    expect_false(isTRUE(all.equal(cov_dof0, cov_neg)))
})

# ---------------------------------------------------------------------------
# da_correlation_matrix
# ---------------------------------------------------------------------------
test_that("da_correlation_matrix returns square matrix with diagonal 1", {
    r <- da_correlation_matrix(m_sq)
    expect_equal(dim(r), c(4, 4))
    expect_equal(diag(r), rep(1, 4), tolerance = 1e-10)
})

test_that("da_correlation_matrix is symmetric and values in [-1, 1]", {
    r <- da_correlation_matrix(m_sq)
    expect_equal(r, t(r), tolerance = 1e-12)
    expect_true(all(r >= -1 - 1e-10 & r <= 1 + 1e-10))
})

test_that("da_correlation_matrix off-diagonal is 1 for perfectly correlated columns", {
    # col2 = 2*col1, so correlation = 1
    x <- matrix(c(1, 2, 3, 2, 4, 6), nrow = 3, ncol = 2)
    r <- da_correlation_matrix(x)
    expect_equal(r[1, 2], 1.0, tolerance = 1e-10)
})

test_that("da_correlation_matrix accepts data.frame input", {
    df <- as.data.frame(m_sq)
    expect_equal(da_correlation_matrix(df), da_correlation_matrix(m_sq), tolerance = 1e-12)
})

test_that("da_covariance_matrix on single column returns 1x1 matrix", {
    x <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
    r <- da_covariance_matrix(x)
    expect_equal(dim(r), c(1, 1))
    expect_equal(r[1, 1], 1.0, tolerance = 1e-10)
})

test_that("da_correlation_matrix on single column returns 1x1 identity", {
    x <- matrix(c(1, 2, 3), nrow = 3, ncol = 1)
    r <- da_correlation_matrix(x)
    expect_equal(dim(r), c(1, 1))
    expect_equal(r[1, 1], 1.0, tolerance = 1e-10)
})

# ---------------------------------------------------------------------------
# Error path tests
# ---------------------------------------------------------------------------
test_that("invalid axis string raises error", {
    expect_error(da_mean(m34, axis = "bad"), "Invalid axis")
})

test_that("da_moment with wrong-sized mean raises error", {
    x <- matrix(1:12, nrow = 3, ncol = 4)
    expect_error(da_moment(x, k = 2, mean = c(1, 2), axis = "col"),
                 "mean array does not match")
})

test_that("da_quantile with invalid method raises error", {
    expect_error(da_quantile(m34, q = 0.5, method = "foo"), "Invalid quantile method")
})

test_that("da_standardize with wrong-sized shift raises error", {
    x <- matrix(1:12, nrow = 3, ncol = 4)
    expect_error(da_standardize(x, shift = c(1, 2), axis = "col"),
                 "size of shift or scale")
})

test_that("da_standardize reverse without shift or scale raises error", {
    x <- matrix(1:12, nrow = 3, ncol = 4)
    expect_error(da_standardize(x, reverse = TRUE, axis = "col"),
                 "Reverse standardization")
})
