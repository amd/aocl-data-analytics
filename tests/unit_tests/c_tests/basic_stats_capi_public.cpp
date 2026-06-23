/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "aoclda.h"
#include "gtest/gtest.h"
#include <cmath>

/*
 * Test the basic statistics C API (double precision).
 * Based on tests/examples/basic_statistics.cpp
 */
TEST(BasicStatsCAPI, MeanDouble) {
    double X[20] = {1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 8.0,
                    4.0, 6.0, 9.0, 5.0, 4.0, 3.0, 1.0, 1.0, 2.0, 2.0};
    da_int n_rows = 4, n_cols = 5, ldx = 4;

    double mean[5];
    EXPECT_EQ(da_mean_d(column_major, da_axis_col, n_rows, n_cols, X, ldx, mean),
              da_status_success);
}

TEST(BasicStatsCAPI, MeanFloat) {
    float X[20] = {1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 8.0f,
                   4.0f, 6.0f, 9.0f, 5.0f, 4.0f, 3.0f, 1.0f, 1.0f, 2.0f, 2.0f};
    da_int n_rows = 4, n_cols = 5, ldx = 4;

    float mean[5];
    EXPECT_EQ(da_mean_s(column_major, da_axis_col, n_rows, n_cols, X, ldx, mean),
              da_status_success);
}

TEST(BasicStatsCAPI, GeometricMeanDouble) {
    double X[20] = {1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 8.0,
                    4.0, 6.0, 9.0, 5.0, 4.0, 3.0, 1.0, 1.0, 2.0, 2.0};
    da_int n_rows = 4, n_cols = 5, ldx = 4;

    double gmean[5];
    EXPECT_EQ(
        da_geometric_mean_d(column_major, da_axis_col, n_rows, n_cols, X, ldx, gmean),
        da_status_success);
}

TEST(BasicStatsCAPI, GeometricMeanFloat) {
    float X[20] = {1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 8.0f,
                   4.0f, 6.0f, 9.0f, 5.0f, 4.0f, 3.0f, 1.0f, 1.0f, 2.0f, 2.0f};
    da_int n_rows = 4, n_cols = 5, ldx = 4;

    float gmean[5];
    EXPECT_EQ(
        da_geometric_mean_s(column_major, da_axis_col, n_rows, n_cols, X, ldx, gmean),
        da_status_success);
}

TEST(BasicStatsCAPI, HarmonicMeanDouble) {
    double X[20] = {1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 8.0,
                    4.0, 6.0, 9.0, 5.0, 4.0, 3.0, 1.0, 1.0, 2.0, 2.0};
    da_int n_rows = 4, n_cols = 5, ldx = 4;

    double hmean[5];
    EXPECT_EQ(
        da_harmonic_mean_d(column_major, da_axis_col, n_rows, n_cols, X, ldx, hmean),
        da_status_success);
}

TEST(BasicStatsCAPI, HarmonicMeanFloat) {
    float X[20] = {1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 8.0f,
                   4.0f, 6.0f, 9.0f, 5.0f, 4.0f, 3.0f, 1.0f, 1.0f, 2.0f, 2.0f};
    da_int n_rows = 4, n_cols = 5, ldx = 4;

    float hmean[5];
    EXPECT_EQ(
        da_harmonic_mean_s(column_major, da_axis_col, n_rows, n_cols, X, ldx, hmean),
        da_status_success);
}

TEST(BasicStatsCAPI, VarianceDouble) {
    double X[20] = {1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 8.0,
                    4.0, 6.0, 9.0, 5.0, 4.0, 3.0, 1.0, 1.0, 2.0, 2.0};
    da_int n_rows = 4, n_cols = 5, ldx = 4, dof = 0;

    double mean[4], variance[4];
    EXPECT_EQ(da_variance_d(column_major, da_axis_row, n_rows, n_cols, X, ldx, dof, mean,
                            variance),
              da_status_success);
}

TEST(BasicStatsCAPI, VarianceFloat) {
    float X[20] = {1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 8.0f,
                   4.0f, 6.0f, 9.0f, 5.0f, 4.0f, 3.0f, 1.0f, 1.0f, 2.0f, 2.0f};
    da_int n_rows = 4, n_cols = 5, ldx = 4, dof = 0;

    float mean[4], variance[4];
    EXPECT_EQ(da_variance_s(column_major, da_axis_row, n_rows, n_cols, X, ldx, dof, mean,
                            variance),
              da_status_success);
}

TEST(BasicStatsCAPI, SkewnessDouble) {
    double X[20] = {1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 8.0,
                    4.0, 6.0, 9.0, 5.0, 4.0, 3.0, 1.0, 1.0, 2.0, 2.0};
    da_int n_rows = 4, n_cols = 5, ldx = 4;

    double mean[4], variance[4], skewness[4];
    EXPECT_EQ(da_skewness_d(column_major, da_axis_row, n_rows, n_cols, X, ldx, mean,
                            variance, skewness),
              da_status_success);
}

TEST(BasicStatsCAPI, SkewnessFloat) {
    float X[20] = {1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 8.0f,
                   4.0f, 6.0f, 9.0f, 5.0f, 4.0f, 3.0f, 1.0f, 1.0f, 2.0f, 2.0f};
    da_int n_rows = 4, n_cols = 5, ldx = 4;

    float mean[4], variance[4], skewness[4];
    EXPECT_EQ(da_skewness_s(column_major, da_axis_row, n_rows, n_cols, X, ldx, mean,
                            variance, skewness),
              da_status_success);
}

TEST(BasicStatsCAPI, KurtosisDouble) {
    double X[20] = {1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 8.0,
                    4.0, 6.0, 9.0, 5.0, 4.0, 3.0, 1.0, 1.0, 2.0, 2.0};
    da_int n_rows = 4, n_cols = 5, ldx = 4;

    double mean[4], variance[4], kurtosis[4];
    EXPECT_EQ(da_kurtosis_d(column_major, da_axis_row, n_rows, n_cols, X, ldx, mean,
                            variance, kurtosis),
              da_status_success);
}

TEST(BasicStatsCAPI, KurtosisFloat) {
    float X[20] = {1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 8.0f,
                   4.0f, 6.0f, 9.0f, 5.0f, 4.0f, 3.0f, 1.0f, 1.0f, 2.0f, 2.0f};
    da_int n_rows = 4, n_cols = 5, ldx = 4;

    float mean[4], variance[4], kurtosis[4];
    EXPECT_EQ(da_kurtosis_s(column_major, da_axis_row, n_rows, n_cols, X, ldx, mean,
                            variance, kurtosis),
              da_status_success);
}

TEST(BasicStatsCAPI, MomentDouble) {
    double X[20] = {1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 8.0,
                    4.0, 6.0, 9.0, 5.0, 4.0, 3.0, 1.0, 1.0, 2.0, 2.0};
    da_int n_rows = 4, n_cols = 5, ldx = 4;

    double mean[4], moment[4];
    EXPECT_EQ(da_moment_d(column_major, da_axis_row, n_rows, n_cols, X, ldx, 3, 0, mean,
                          moment),
              da_status_success);
}

TEST(BasicStatsCAPI, MomentFloat) {
    float X[20] = {1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 8.0f,
                   4.0f, 6.0f, 9.0f, 5.0f, 4.0f, 3.0f, 1.0f, 1.0f, 2.0f, 2.0f};
    da_int n_rows = 4, n_cols = 5, ldx = 4;

    float mean[4], moment[4];
    EXPECT_EQ(da_moment_s(column_major, da_axis_row, n_rows, n_cols, X, ldx, 3, 0, mean,
                          moment),
              da_status_success);
}

TEST(BasicStatsCAPI, QuantileDouble) {
    double X[20] = {1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 8.0,
                    4.0, 6.0, 9.0, 5.0, 4.0, 3.0, 1.0, 1.0, 2.0, 2.0};
    da_int n_rows = 4, n_cols = 5, ldx = 4;

    double quantile[5];
    EXPECT_EQ(da_quantile_d(column_major, da_axis_col, n_rows, n_cols, X, ldx, 0.5,
                            quantile, da_quantile_type_7),
              da_status_success);
}

TEST(BasicStatsCAPI, QuantileFloat) {
    float X[20] = {1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 8.0f,
                   4.0f, 6.0f, 9.0f, 5.0f, 4.0f, 3.0f, 1.0f, 1.0f, 2.0f, 2.0f};
    da_int n_rows = 4, n_cols = 5, ldx = 4;

    float quantile[5];
    EXPECT_EQ(da_quantile_s(column_major, da_axis_col, n_rows, n_cols, X, ldx, 0.5f,
                            quantile, da_quantile_type_7),
              da_status_success);
}

TEST(BasicStatsCAPI, FivePointSummaryDouble) {
    double X[20] = {1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 8.0,
                    4.0, 6.0, 9.0, 5.0, 4.0, 3.0, 1.0, 1.0, 2.0, 2.0};
    da_int n_rows = 4, n_cols = 5, ldx = 4;

    double minimum[1], lower_hinge[1], median[1], upper_hinge[1], maximum[1];
    EXPECT_EQ(da_five_point_summary_d(column_major, da_axis_all, n_rows, n_cols, X, ldx,
                                      minimum, lower_hinge, median, upper_hinge, maximum),
              da_status_success);
}

TEST(BasicStatsCAPI, FivePointSummaryFloat) {
    float X[20] = {1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 8.0f,
                   4.0f, 6.0f, 9.0f, 5.0f, 4.0f, 3.0f, 1.0f, 1.0f, 2.0f, 2.0f};
    da_int n_rows = 4, n_cols = 5, ldx = 4;

    float minimum[1], lower_hinge[1], median[1], upper_hinge[1], maximum[1];
    EXPECT_EQ(da_five_point_summary_s(column_major, da_axis_all, n_rows, n_cols, X, ldx,
                                      minimum, lower_hinge, median, upper_hinge, maximum),
              da_status_success);
}

TEST(BasicStatsCAPI, StandardizeDouble) {
    double X[20] = {1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 8.0,
                    4.0, 6.0, 9.0, 5.0, 4.0, 3.0, 1.0, 1.0, 2.0, 2.0};
    da_int n_rows = 4, n_cols = 5, ldx = 4, dof = 0, mode = 0;

    double shift[5], scale[5];
    EXPECT_EQ(da_standardize_d(column_major, da_axis_col, n_rows, n_cols, X, ldx, dof,
                               mode, shift, scale),
              da_status_success);
}

TEST(BasicStatsCAPI, StandardizeFloat) {
    float X[20] = {1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 8.0f,
                   4.0f, 6.0f, 9.0f, 5.0f, 4.0f, 3.0f, 1.0f, 1.0f, 2.0f, 2.0f};
    da_int n_rows = 4, n_cols = 5, ldx = 4, dof = 0, mode = 0;

    float shift[5], scale[5];
    EXPECT_EQ(da_standardize_s(column_major, da_axis_col, n_rows, n_cols, X, ldx, dof,
                               mode, shift, scale),
              da_status_success);
}

TEST(BasicStatsCAPI, CovarianceMatrixDouble) {
    double X[20] = {1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 8.0,
                    4.0, 6.0, 9.0, 5.0, 4.0, 3.0, 1.0, 1.0, 2.0, 2.0};
    da_int n_rows = 4, n_cols = 5, ldx = 4, ldcov = 5, dof = 0, assume_centered = 0;

    double cov[25];
    EXPECT_EQ(da_covariance_matrix_d(column_major, n_rows, n_cols, X, ldx, dof, cov,
                                     ldcov, assume_centered),
              da_status_success);
}

TEST(BasicStatsCAPI, CovarianceMatrixFloat) {
    float X[20] = {1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 8.0f,
                   4.0f, 6.0f, 9.0f, 5.0f, 4.0f, 3.0f, 1.0f, 1.0f, 2.0f, 2.0f};
    da_int n_rows = 4, n_cols = 5, ldx = 4, ldcov = 5, dof = 0, assume_centered = 0;

    float cov[25];
    EXPECT_EQ(da_covariance_matrix_s(column_major, n_rows, n_cols, X, ldx, dof, cov,
                                     ldcov, assume_centered),
              da_status_success);
}

TEST(BasicStatsCAPI, CorrelationMatrixDouble) {
    double X[20] = {1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 8.0,
                    4.0, 6.0, 9.0, 5.0, 4.0, 3.0, 1.0, 1.0, 2.0, 2.0};
    da_int n_rows = 4, n_cols = 5, ldx = 4, ldcorr = 5;

    double corr[25];
    EXPECT_EQ(da_correlation_matrix_d(column_major, n_rows, n_cols, X, ldx, corr, ldcorr),
              da_status_success);
}

TEST(BasicStatsCAPI, CorrelationMatrixFloat) {
    float X[20] = {1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 8.0f,
                   4.0f, 6.0f, 9.0f, 5.0f, 4.0f, 3.0f, 1.0f, 1.0f, 2.0f, 2.0f};
    da_int n_rows = 4, n_cols = 5, ldx = 4, ldcorr = 5;

    float corr[25];
    EXPECT_EQ(da_correlation_matrix_s(column_major, n_rows, n_cols, X, ldx, corr, ldcorr),
              da_status_success);
}
