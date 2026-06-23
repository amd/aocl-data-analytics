/*
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

template <typename T>
da_status da_mean(da_order order, da_axis axis, da_int n_rows, da_int n_cols, const T *X,
                  da_int ldx, T *amean) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::mean(order, axis, n_rows,
                                                               n_cols, X, ldx, amean)));
}

template <typename T>
da_status da_geometric_mean(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                            const T *X, da_int ldx, T *gmean) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::geometric_mean(
                                 order, axis, n_rows, n_cols, X, ldx, gmean)));
}

template <typename T>
da_status da_harmonic_mean(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                           const T *X, da_int ldx, T *hmean) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::harmonic_mean(
                                 order, axis, n_rows, n_cols, X, ldx, hmean)));
}

template <typename T>
da_status da_variance(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                      const T *X, da_int ldx, da_int dof, T *mean, T *var) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::variance(
                                 order, axis, n_rows, n_cols, X, ldx, dof, mean, var)));
}

template <typename T>
da_status da_skewness(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                      const T *X, da_int ldx, T *mean, T *var, T *skew) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::skewness(
                                 order, axis, n_rows, n_cols, X, ldx, mean, var, skew)));
}

template <typename T>
da_status da_kurtosis(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                      const T *X, da_int ldx, T *mean, T *var, T *kurt) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::kurtosis(
                                 order, axis, n_rows, n_cols, X, ldx, mean, var, kurt)));
}

template <typename T>
da_status da_moment(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                    const T *X, da_int ldx, da_int k, da_int use_precomputed_mean,
                    T *mean, T *mom) {
    DISPATCHER(nosave_stats,
               return (da_basic_statistics::moment(order, axis, n_rows, n_cols, X, ldx, k,
                                                   use_precomputed_mean, mean, mom)));
}

template <typename T>
da_status da_quantile(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                      const T *X, da_int ldx, T q, T *quant,
                      da_quantile_type quantile_type) {
    DISPATCHER(nosave_stats,
               return (da_basic_statistics::quantile(order, axis, n_rows, n_cols, X, ldx,
                                                     q, quant, quantile_type)));
}

template <typename T>
da_status da_five_point_summary(da_order order, da_axis axis, da_int n_rows,
                                da_int n_cols, const T *X, da_int ldx, T *minimum,
                                T *lower_hinge, T *median, T *upper_hinge, T *maximum) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::five_point_summary(
                                 order, axis, n_rows, n_cols, X, ldx, minimum,
                                 lower_hinge, median, upper_hinge, maximum)));
}

template <typename T>
da_status da_standardize(da_order order, da_axis axis, da_int n_rows, da_int n_cols, T *X,
                         da_int ldx, da_int dof, da_int mode, T *shift, T *scale) {
    DISPATCHER(nosave_stats,
               return (da_basic_statistics::standardize(order, axis, n_rows, n_cols, X,
                                                        ldx, dof, mode, shift, scale)));
}

template <typename T>
da_status da_covariance_matrix(da_order order, da_int n_rows, da_int n_cols, const T *X,
                               da_int ldx, da_int dof, T *cov, da_int ldcov,
                               da_int assume_centered) {
    DISPATCHER(nosave_stats,
               return (da_basic_statistics::covariance_matrix(
                   order, n_rows, n_cols, X, ldx, dof, cov, ldcov, assume_centered)));
}

template <typename T>
da_status da_correlation_matrix(da_order order, da_int n_rows, da_int n_cols, const T *X,
                                da_int ldx, T *corr, da_int ldcorr) {
    DISPATCHER(nosave_stats, return (da_basic_statistics::correlation_matrix(
                                 order, n_rows, n_cols, X, ldx, corr, ldcorr)));
}

template da_status da_mean<float>(da_order, da_axis, da_int, da_int, const float *,
                                  da_int, float *);
template da_status da_mean<double>(da_order, da_axis, da_int, da_int, const double *,
                                   da_int, double *);
template da_status da_geometric_mean<float>(da_order, da_axis, da_int, da_int,
                                            const float *, da_int, float *);
template da_status da_geometric_mean<double>(da_order, da_axis, da_int, da_int,
                                             const double *, da_int, double *);
template da_status da_harmonic_mean<float>(da_order, da_axis, da_int, da_int,
                                           const float *, da_int, float *);
template da_status da_harmonic_mean<double>(da_order, da_axis, da_int, da_int,
                                            const double *, da_int, double *);
template da_status da_variance<float>(da_order, da_axis, da_int, da_int, const float *,
                                      da_int, da_int, float *, float *);
template da_status da_variance<double>(da_order, da_axis, da_int, da_int, const double *,
                                       da_int, da_int, double *, double *);
template da_status da_skewness<float>(da_order, da_axis, da_int, da_int, const float *,
                                      da_int, float *, float *, float *);
template da_status da_skewness<double>(da_order, da_axis, da_int, da_int, const double *,
                                       da_int, double *, double *, double *);
template da_status da_kurtosis<float>(da_order, da_axis, da_int, da_int, const float *,
                                      da_int, float *, float *, float *);
template da_status da_kurtosis<double>(da_order, da_axis, da_int, da_int, const double *,
                                       da_int, double *, double *, double *);
template da_status da_moment<float>(da_order, da_axis, da_int, da_int, const float *,
                                    da_int, da_int, da_int, float *, float *);
template da_status da_moment<double>(da_order, da_axis, da_int, da_int, const double *,
                                     da_int, da_int, da_int, double *, double *);
template da_status da_quantile<float>(da_order, da_axis, da_int, da_int, const float *,
                                      da_int, float, float *, da_quantile_type);
template da_status da_quantile<double>(da_order, da_axis, da_int, da_int, const double *,
                                       da_int, double, double *, da_quantile_type);
template da_status da_five_point_summary<float>(da_order, da_axis, da_int, da_int,
                                                const float *, da_int, float *, float *,
                                                float *, float *, float *);
template da_status da_five_point_summary<double>(da_order, da_axis, da_int, da_int,
                                                 const double *, da_int, double *,
                                                 double *, double *, double *, double *);
template da_status da_standardize<float>(da_order, da_axis, da_int, da_int, float *,
                                         da_int, da_int, da_int, float *, float *);
template da_status da_standardize<double>(da_order, da_axis, da_int, da_int, double *,
                                          da_int, da_int, da_int, double *, double *);
template da_status da_covariance_matrix<float>(da_order, da_int, da_int, const float *,
                                               da_int, da_int, float *, da_int, da_int);
template da_status da_covariance_matrix<double>(da_order, da_int, da_int, const double *,
                                                da_int, da_int, double *, da_int, da_int);
template da_status da_correlation_matrix<float>(da_order, da_int, da_int, const float *,
                                                da_int, float *, da_int);
template da_status da_correlation_matrix<double>(da_order, da_int, da_int, const double *,
                                                 da_int, double *, da_int);
