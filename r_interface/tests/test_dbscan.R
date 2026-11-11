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

x <- matrix(c(
    2.0, 1.0,
    -1.0, -2.0,
    3.0, 2.0,
    2.0, 3.0,
    -3.0, -2.0,
    -2.0, -1.0,
    -2.0, -3.0,
    1.0, 2.0
), nrow = 8, ncol = 2, byrow = TRUE)

expected_labels <- c(0L, 1L, 0L, 0L, 1L, 1L, 1L, 0L)
expected_core_indices <- 0:7

test_that("DBSCAN constructor defaults work", {
    m <- DBSCAN$new()
    expect_s3_class(m, "DBSCAN")
})

test_that("DBSCAN constructor custom params accepted", {
    m <- DBSCAN$new(
        min_samples = 2, metric = "minkowski", algorithm = "kd_tree",
        leaf_size = 20, eps = 2.0, power = 2.0, check_data = TRUE
    )
    expect_s3_class(m, "DBSCAN")
})

test_that("DBSCAN public API returns expected clustering state", {
    m <- DBSCAN$new(eps = 2.0, min_samples = 2)
    m$fit(x)

    expect_equal(m$n_clusters, 2L)
    expect_equal(m$n_core_samples, 8L)
    expect_equal(m$n_samples, 8L)
    expect_equal(m$n_features, 2L)
    expect_equal(m$labels, expected_labels)
    expect_equal(m$core_sample_indices, expected_core_indices)
})

test_that("DBSCAN kd_tree and ball_tree configurations match expected clustering", {
    for (algo in c("kd_tree", "ball_tree")) {
        m <- DBSCAN$new(
            eps = 2.0, min_samples = 2,
            algorithm = algo, metric = "minkowski"
        )
        m$fit(x)
        expect_equal(m$n_clusters, 2L)
        expect_equal(m$n_core_samples, 8L)
        expect_equal(m$n_samples, 8L)
        expect_equal(m$n_features, 2L)
        expect_equal(m$labels, expected_labels)
        expect_equal(m$core_sample_indices, expected_core_indices)
    }
})

test_that("DBSCAN accepts data.frame input", {
    df <- as.data.frame(x)
    m <- DBSCAN$new(eps = 2.0, min_samples = 2)
    expect_no_error(m$fit(df))
})
