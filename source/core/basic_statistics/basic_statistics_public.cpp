/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "context.hpp"
#include "da_error.hpp"
#include "dynamic_dispatch.hpp"
#include "macros.h"

da_errors::error_bypass_t *nosave_stats(nullptr);

da_status da_mean_d(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                    const double *X, da_int ldx, double *amean) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::mean(order, axis, n_rows,
                                                               n_cols, X, ldx, amean)));
}

da_status da_mean_s(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                    const float *X, da_int ldx, float *amean) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::mean(order, axis, n_rows,
                                                               n_cols, X, ldx, amean)));
}

da_status da_geometric_mean_d(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                              const double *X, da_int ldx, double *gmean) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::geometric_mean(
                                 order, axis, n_rows, n_cols, X, ldx, gmean)));
}

da_status da_geometric_mean_s(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                              const float *X, da_int ldx, float *gmean) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::geometric_mean(
                                 order, axis, n_rows, n_cols, X, ldx, gmean)));
}

da_status da_harmonic_mean_d(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                             const double *X, da_int ldx, double *hmean) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::harmonic_mean(
                                 order, axis, n_rows, n_cols, X, ldx, hmean)));
}

da_status da_harmonic_mean_s(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                             const float *X, da_int ldx, float *hmean) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::harmonic_mean(
                                 order, axis, n_rows, n_cols, X, ldx, hmean)));
}

da_status da_variance_d(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                        const double *X, da_int ldx, da_int dof, double *mean,
                        double *var) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::variance(
                                 order, axis, n_rows, n_cols, X, ldx, dof, mean, var)));
}

da_status da_variance_s(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                        const float *X, da_int ldx, da_int dof, float *mean, float *var) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::variance(
                                 order, axis, n_rows, n_cols, X, ldx, dof, mean, var)));
}

da_status da_skewness_d(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                        const double *X, da_int ldx, double *mean, double *var,
                        double *skew) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::skewness(
                                 order, axis, n_rows, n_cols, X, ldx, mean, var, skew)));
}

da_status da_skewness_s(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                        const float *X, da_int ldx, float *mean, float *var,
                        float *skew) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::skewness(
                                 order, axis, n_rows, n_cols, X, ldx, mean, var, skew)));
}

da_status da_kurtosis_d(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                        const double *X, da_int ldx, double *mean, double *var,
                        double *kurt) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::kurtosis(
                                 order, axis, n_rows, n_cols, X, ldx, mean, var, kurt)));
}

da_status da_kurtosis_s(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                        const float *X, da_int ldx, float *mean, float *var,
                        float *kurt) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::kurtosis(
                                 order, axis, n_rows, n_cols, X, ldx, mean, var, kurt)));
}

da_status da_moment_d(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                      const double *X, da_int ldx, da_int k, da_int use_precomputed_mean,
                      double *mean, double *mom) {
    DISPATCHER(nosave_stats,
               return (da_basic_statistics::moment(order, axis, n_rows, n_cols, X, ldx, k,
                                                   use_precomputed_mean, mean, mom)));
}

da_status da_moment_s(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                      const float *X, da_int ldx, da_int k, da_int use_precomputed_mean,
                      float *mean, float *mom) {
    DISPATCHER(nosave_stats,
               return (da_basic_statistics::moment(order, axis, n_rows, n_cols, X, ldx, k,
                                                   use_precomputed_mean, mean, mom)));
}

da_status da_quantile_d(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                        const double *X, da_int ldx, double q, double *quant,
                        da_quantile_type quantile_type) {
    DISPATCHER(nosave_stats,
               return (da_basic_statistics::quantile(order, axis, n_rows, n_cols, X, ldx,
                                                     q, quant, quantile_type)));
}

da_status da_quantile_s(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                        const float *X, da_int ldx, float q, float *quant,
                        da_quantile_type quantile_type) {
    DISPATCHER(nosave_stats,
               return (da_basic_statistics::quantile(order, axis, n_rows, n_cols, X, ldx,
                                                     q, quant, quantile_type)));
}

da_status da_five_point_summary_d(da_order order, da_axis axis, da_int n_rows,
                                  da_int n_cols, const double *X, da_int ldx,
                                  double *minimum, double *lower_hinge, double *median,
                                  double *upper_hinge, double *maximum) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::five_point_summary(
                                 order, axis, n_rows, n_cols, X, ldx, minimum,
                                 lower_hinge, median, upper_hinge, maximum)));
}

da_status da_five_point_summary_s(da_order order, da_axis axis, da_int n_rows,
                                  da_int n_cols, const float *X, da_int ldx,
                                  float *minimum, float *lower_hinge, float *median,
                                  float *upper_hinge, float *maximum) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::five_point_summary(
                                 order, axis, n_rows, n_cols, X, ldx, minimum,
                                 lower_hinge, median, upper_hinge, maximum)));
}

da_status da_standardize_d(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                           double *X, da_int ldx, da_int dof, da_int mode, double *shift,
                           double *scale) {
    DISPATCHER(nosave_stats,
               return (da_basic_statistics::standardize(order, axis, n_rows, n_cols, X,
                                                        ldx, dof, mode, shift, scale)));
}

da_status da_standardize_s(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                           float *X, da_int ldx, da_int dof, da_int mode, float *shift,
                           float *scale) {
    DISPATCHER(nosave_stats,
               return (da_basic_statistics::standardize(order, axis, n_rows, n_cols, X,
                                                        ldx, dof, mode, shift, scale)));
}

da_status da_covariance_matrix_d(da_order order, da_int n_rows, da_int n_cols,
                                 const double *X, da_int ldx, da_int dof, double *cov,
                                 da_int ldcov) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::covariance_matrix(
                                 order, n_rows, n_cols, X, ldx, dof, cov, ldcov)));
}

da_status da_covariance_matrix_s(da_order order, da_int n_rows, da_int n_cols,
                                 const float *X, da_int ldx, da_int dof, float *cov,
                                 da_int ldcov) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::covariance_matrix(
                                 order, n_rows, n_cols, X, ldx, dof, cov, ldcov)));
}

da_status da_correlation_matrix_d(da_order order, da_int n_rows, da_int n_cols,
                                  const double *X, da_int ldx, double *corr,
                                  da_int ldcorr) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::correlation_matrix(
                                 order, n_rows, n_cols, X, ldx, corr, ldcorr)));
}

da_status da_correlation_matrix_s(da_order order, da_int n_rows, da_int n_cols,
                                  const float *X, da_int ldx, float *corr,
                                  da_int ldcorr) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::correlation_matrix(
                                 order, n_rows, n_cols, X, ldx, corr, ldcorr)));
}
