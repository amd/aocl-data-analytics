/*
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "correlation_and_covariance.hpp"
#include "moment_statistics.hpp"
#include "order_statistics.hpp"
#include "statistical_utilities.hpp"

da_status da_mean_d(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                    da_int ldx, double *amean) {
    return da_basic_statistics::mean(axis, n_rows, n_cols, X, ldx, amean);
}

da_status da_mean_s(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                    da_int ldx, float *amean) {
    return da_basic_statistics::mean(axis, n_rows, n_cols, X, ldx, amean);
}

da_status da_geometric_mean_d(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                              da_int ldx, double *gmean) {
    return da_basic_statistics::geometric_mean(axis, n_rows, n_cols, X, ldx, gmean);
}

da_status da_geometric_mean_s(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                              da_int ldx, float *gmean) {
    return da_basic_statistics::geometric_mean(axis, n_rows, n_cols, X, ldx, gmean);
}

da_status da_harmonic_mean_d(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                             da_int ldx, double *hmean) {
    return da_basic_statistics::harmonic_mean(axis, n_rows, n_cols, X, ldx, hmean);
}

da_status da_harmonic_mean_s(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                             da_int ldx, float *hmean) {
    return da_basic_statistics::harmonic_mean(axis, n_rows, n_cols, X, ldx, hmean);
}

da_status da_variance_d(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                        da_int ldx, double *mean, double *var) {
    return da_basic_statistics::variance(axis, n_rows, n_cols, X, ldx, mean, var);
}

da_status da_variance_s(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                        da_int ldx, float *mean, float *var) {
    return da_basic_statistics::variance(axis, n_rows, n_cols, X, ldx, mean, var);
}

da_status da_skewness_d(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                        da_int ldx, double *mean, double *var, double *skew) {
    return da_basic_statistics::skewness(axis, n_rows, n_cols, X, ldx, mean, var, skew);
}

da_status da_skewness_s(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                        da_int ldx, float *mean, float *var, float *skew) {
    return da_basic_statistics::skewness(axis, n_rows, n_cols, X, ldx, mean, var, skew);
}

da_status da_kurtosis_d(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                        da_int ldx, double *mean, double *var, double *kurt) {
    return da_basic_statistics::kurtosis(axis, n_rows, n_cols, X, ldx, mean, var, kurt);
}

da_status da_kurtosis_s(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                        da_int ldx, float *mean, float *var, float *kurt) {
    return da_basic_statistics::kurtosis(axis, n_rows, n_cols, X, ldx, mean, var, kurt);
}

da_status da_moment_d(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                      da_int ldx, da_int k, da_int use_precomputed_mean, double *mean,
                      double *mom) {
    return da_basic_statistics::moment(axis, n_rows, n_cols, X, ldx, k,
                                       use_precomputed_mean, mean, mom);
}

da_status da_moment_s(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                      da_int ldx, da_int k, da_int use_precomputed_mean, float *mean,
                      float *mom) {
    return da_basic_statistics::moment(axis, n_rows, n_cols, X, ldx, k,
                                       use_precomputed_mean, mean, mom);
}

da_status da_quantile_d(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                        da_int ldx, double q, double *quant,
                        da_quantile_type quantile_type) {
    return da_basic_statistics::quantile(axis, n_rows, n_cols, X, ldx, q, quant,
                                         quantile_type);
}

da_status da_quantile_s(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                        da_int ldx, float q, float *quant,
                        da_quantile_type quantile_type) {
    return da_basic_statistics::quantile(axis, n_rows, n_cols, X, ldx, q, quant,
                                         quantile_type);
}

da_status da_five_point_summary_d(da_axis axis, da_int n_rows, da_int n_cols,
                                  const double *X, da_int ldx, double *minimum,
                                  double *lower_hinge, double *median,
                                  double *upper_hinge, double *maximum) {
    return da_basic_statistics::five_point_summary(
        axis, n_rows, n_cols, X, ldx, minimum, lower_hinge, median, upper_hinge, maximum);
}

da_status da_five_point_summary_s(da_axis axis, da_int n_rows, da_int n_cols,
                                  const float *X, da_int ldx, float *minimum,
                                  float *lower_hinge, float *median, float *upper_hinge,
                                  float *maximum) {
    return da_basic_statistics::five_point_summary(
        axis, n_rows, n_cols, X, ldx, minimum, lower_hinge, median, upper_hinge, maximum);
}

/* Shift by a constant amount and scale */
da_status da_standardize_d(da_axis axis, da_int n_rows, da_int n_cols, double *X,
                           da_int ldx, double *shift, double *scale) {
    return da_basic_statistics::standardize(axis, n_rows, n_cols, X, ldx, shift, scale);
}

da_status da_standardize_s(da_axis axis, da_int n_rows, da_int n_cols, float *X,
                           da_int ldx, float *shift, float *scale) {
    return da_basic_statistics::standardize(axis, n_rows, n_cols, X, ldx, shift, scale);
}

da_status da_covariance_matrix_d(da_int n_rows, da_int n_cols, const double *X,
                                 da_int ldx, double *cov, da_int ldcov) {
    return da_basic_statistics::covariance_matrix(n_rows, n_cols, X, ldx, cov, ldcov);
}

da_status da_covariance_matrix_s(da_int n_rows, da_int n_cols, const float *X, da_int ldx,
                                 float *cov, da_int ldcov) {
    return da_basic_statistics::covariance_matrix(n_rows, n_cols, X, ldx, cov, ldcov);
}

da_status da_correlation_matrix_d(da_int n_rows, da_int n_cols, const double *X,
                                  da_int ldx, double *corr, da_int ldcorr) {
    return da_basic_statistics::correlation_matrix(n_rows, n_cols, X, ldx, corr, ldcorr);
}

da_status da_correlation_matrix_s(da_int n_rows, da_int n_cols, const float *X,
                                  da_int ldx, float *corr, da_int ldcorr) {
    return da_basic_statistics::correlation_matrix(n_rows, n_cols, X, ldx, corr, ldcorr);
}
