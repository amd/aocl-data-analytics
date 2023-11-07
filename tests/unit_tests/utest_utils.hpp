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

#ifndef UTEST_UTILS_HPP
#define UTEST_UTILS_HPP
#include "aoclda.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <vector>

#define EXPECT_ARR_NEAR(n, x, y, abs_error)                                              \
    for (da_int j = 0; j < (n); j++)                                                     \
    EXPECT_NEAR((x[j]), (y[j]), abs_error)                                               \
        << "Vectors " #x " and " #y " different at index j=" << j << "."

#define EXPECT_ARR_EQ(n, x, y, incx, incy, startx, starty)                               \
    for (da_int j = 0; j < (n); j++)                                                     \
    EXPECT_EQ((x[startx + j * incx]), (y[starty + j * incy]))                            \
        << "Vectors " #x " and " #y " different at index j=" << j << "."

#define EXPECT_ARR_ABS_NEAR(n, x, y, abs_error)                                          \
    for (da_int j = 0; j < (n); j++)                                                     \
    EXPECT_NEAR((std::abs(x[j])), (std::abs(y[j])), abs_error)                           \
        << "Vectors " #x " and " #y " different at index j=" << j << "."

/* Convert std::vector from one type to another, to avoid warnings in templated tests*/
template <class T_in, class T_out>
std::vector<T_out> convert_vector(const std::vector<T_in> &input) {
    std::vector<T_out> output(input.size());
    std::transform(input.begin(), input.end(), output.begin(),
                   [](T_out x) { return static_cast<T_in>(x); });
    return output;
}

/* handle overloaded functions */
template <class T>
da_status da_handle_init(da_handle *handle, da_handle_type handle_type);
template <>
da_status da_handle_init<double>(da_handle *handle, da_handle_type handle_type) {
    return da_handle_init_d(handle, handle_type);
}
template <>
da_status da_handle_init<float>(da_handle *handle, da_handle_type handle_type) {
    return da_handle_init_s(handle, handle_type);
}

inline da_status da_handle_get_result(da_handle handle, da_result query, da_int *dim,
                                      double *result) {
    return da_handle_get_result_d(handle, query, dim, result);
}
inline da_status da_handle_get_result(da_handle handle, da_result query, da_int *dim,
                                      float *result) {
    return da_handle_get_result_s(handle, query, dim, result);
}
inline da_status da_handle_get_result(da_handle handle, da_result query, da_int *dim,
                                      da_int *result) {
    return da_handle_get_result_int(handle, query, dim, result);
}

/* Options overloaded functons */
template <class T>
inline da_status da_options_set_real(da_handle handle, const char *option, T value);
template <>
inline da_status da_options_set_real<float>(da_handle handle, const char *option,
                                            float value) {
    return da_options_set_real_s(handle, option, value);
}
template <>
inline da_status da_options_set_real<double>(da_handle handle, const char *option,
                                             double value) {
    return da_options_set_real_d(handle, option, value);
}

/* FIXME The tests should be able to include directly read_csv.hpp
 * This is a workaround for now
 */

inline da_status da_read_csv(da_datastore store, const char *filename, double **a,
                             da_int *nrows, da_int *ncols, char ***headings) {
    return da_read_csv_d(store, filename, a, nrows, ncols, headings);
}

inline da_status da_read_csv(da_datastore store, const char *filename, float **a,
                             da_int *nrows, da_int *ncols, char ***headings) {
    return da_read_csv_s(store, filename, a, nrows, ncols, headings);
}

inline da_status da_read_csv(da_datastore store, const char *filename, da_int **a,
                             da_int *nrows, da_int *ncols, char ***headings) {
    return da_read_csv_int(store, filename, a, nrows, ncols, headings);
}

inline da_status da_read_csv(da_datastore store, const char *filename, uint8_t **a,
                             da_int *nrows, da_int *ncols, char ***headings) {
    return da_read_csv_uint8(store, filename, a, nrows, ncols, headings);
}

inline da_status da_read_csv(da_datastore store, const char *filename, char ***a,
                             da_int *nrows, da_int *ncols, char ***headings) {
    return da_read_csv_char(store, filename, a, nrows, ncols, headings);
}

/* basic statistics overloaded functions */
inline da_status da_mean(da_axis axis, da_int n, da_int p, double *x, da_int ldx,
                         double *mean) {
    return da_mean_d(axis, n, p, x, ldx, mean);
}

inline da_status da_mean(da_axis axis, da_int n, da_int p, float *x, da_int ldx,
                         float *mean) {
    return da_mean_s(axis, n, p, x, ldx, mean);
}

inline da_status da_harmonic_mean(da_axis axis, da_int n, da_int p, double *x, da_int ldx,
                                  double *harmonic_mean) {
    return da_harmonic_mean_d(axis, n, p, x, ldx, harmonic_mean);
}

inline da_status da_harmonic_mean(da_axis axis, da_int n, da_int p, float *x, da_int ldx,
                                  float *harmonic_mean) {
    return da_harmonic_mean_s(axis, n, p, x, ldx, harmonic_mean);
}

inline da_status da_geometric_mean(da_axis axis, da_int n, da_int p, double *x,
                                   da_int ldx, double *geometric_mean) {
    return da_geometric_mean_d(axis, n, p, x, ldx, geometric_mean);
}

inline da_status da_geometric_mean(da_axis axis, da_int n, da_int p, float *x, da_int ldx,
                                   float *geometric_mean) {
    return da_geometric_mean_s(axis, n, p, x, ldx, geometric_mean);
}

inline da_status da_variance(da_axis axis, da_int n, da_int p, double *x, da_int ldx,
                             double *mean, double *variance) {
    return da_variance_d(axis, n, p, x, ldx, mean, variance);
}

inline da_status da_variance(da_axis axis, da_int n, da_int p, float *x, da_int ldx,
                             float *mean, float *variance) {
    return da_variance_s(axis, n, p, x, ldx, mean, variance);
}

inline da_status da_skewness(da_axis axis, da_int n, da_int p, double *x, da_int ldx,
                             double *mean, double *variance, double *skewness) {
    return da_skewness_d(axis, n, p, x, ldx, mean, variance, skewness);
}

inline da_status da_skewness(da_axis axis, da_int n, da_int p, float *x, da_int ldx,
                             float *mean, float *variance, float *skewness) {
    return da_skewness_s(axis, n, p, x, ldx, mean, variance, skewness);
}

inline da_status da_kurtosis(da_axis axis, da_int n, da_int p, double *x, da_int ldx,
                             double *mean, double *variance, double *kurtosis) {
    return da_kurtosis_d(axis, n, p, x, ldx, mean, variance, kurtosis);
}

inline da_status da_kurtosis(da_axis axis, da_int n, da_int p, float *x, da_int ldx,
                             float *mean, float *variance, float *kurtosis) {
    return da_kurtosis_s(axis, n, p, x, ldx, mean, variance, kurtosis);
}

inline da_status da_moment(da_axis axis, da_int n, da_int p, double *x, da_int ldx,
                           da_int k, da_int use_precomputed_mean, double *mean,
                           double *moment) {
    return da_moment_d(axis, n, p, x, ldx, k, use_precomputed_mean, mean, moment);
}

inline da_status da_moment(da_axis axis, da_int n, da_int p, float *x, da_int ldx,
                           da_int k, da_int use_precomputed_mean, float *mean,
                           float *moment) {
    return da_moment_s(axis, n, p, x, ldx, k, use_precomputed_mean, mean, moment);
}

inline da_status da_quantile(da_axis axis, da_int n, da_int p, double *x, da_int ldx,
                             double q, double *quantile, da_quantile_type quantile_type) {
    return da_quantile_d(axis, n, p, x, ldx, q, quantile, quantile_type);
}

inline da_status da_quantile(da_axis axis, da_int n, da_int p, float *x, da_int ldx,
                             float q, float *quantile, da_quantile_type quantile_type) {
    return da_quantile_s(axis, n, p, x, ldx, q, quantile, quantile_type);
}

inline da_status da_five_point_summary(da_axis axis, da_int n, da_int p, double *x,
                                       da_int ldx, double *minimum, double *lower_hinge,
                                       double *median, double *upper_hinge,
                                       double *maximum) {
    return da_five_point_summary_d(axis, n, p, x, ldx, minimum, lower_hinge, median,
                                   upper_hinge, maximum);
}

inline da_status da_five_point_summary(da_axis axis, da_int n, da_int p, float *x,
                                       da_int ldx, float *minimum, float *lower_hinge,
                                       float *median, float *upper_hinge,
                                       float *maximum) {
    return da_five_point_summary_s(axis, n, p, x, ldx, minimum, lower_hinge, median,
                                   upper_hinge, maximum);
}

inline da_status da_standardize(da_axis axis, da_int n, da_int p, double *x, da_int ldx,
                                double *shift, double *scale) {
    return da_standardize_d(axis, n, p, x, ldx, shift, scale);
}

inline da_status da_standardize(da_axis axis, da_int n, da_int p, float *x, da_int ldx,
                                float *shift, float *scale) {
    return da_standardize_s(axis, n, p, x, ldx, shift, scale);
}

inline da_status da_covariance_matrix(da_int n, da_int p, float *x, da_int ldx,
                                      float *cov, da_int ldcov) {
    return da_covariance_matrix_s(n, p, x, ldx, cov, ldcov);
}

inline da_status da_covariance_matrix(da_int n, da_int p, double *x, da_int ldx,
                                      double *cov, da_int ldcov) {
    return da_covariance_matrix_d(n, p, x, ldx, cov, ldcov);
}

inline da_status da_correlation_matrix(da_int n, da_int p, float *x, da_int ldx,
                                       float *corr, da_int ldcorr) {
    return da_correlation_matrix_s(n, p, x, ldx, corr, ldcorr);
}

inline da_status da_correlation_matrix(da_int n, da_int p, double *x, da_int ldx,
                                       double *corr, da_int ldcorr) {
    return da_correlation_matrix_d(n, p, x, ldx, corr, ldcorr);
}

/* linmod overloaded functions */
template <class T> da_status da_linmod_select_model(da_handle handle, linmod_model mod);
template <> da_status da_linmod_select_model<double>(da_handle handle, linmod_model mod) {
    return da_linmod_select_model_d(handle, mod);
}
template <> da_status da_linmod_select_model<float>(da_handle handle, linmod_model mod) {
    return da_linmod_select_model_s(handle, mod);
}

inline da_status da_linreg_define_features(da_handle handle, da_int n, da_int m,
                                           double *A, double *b) {
    return da_linmod_define_features_d(handle, n, m, A, b);
}
inline da_status da_linreg_define_features(da_handle handle, da_int n, da_int m, float *A,
                                           float *b) {
    return da_linmod_define_features_s(handle, n, m, A, b);
}

template <class T> da_status da_linreg_fit(da_handle handle);
template <> da_status da_linreg_fit<double>(da_handle handle) {
    return da_linmod_fit_d(handle);
}
template <> da_status da_linreg_fit<float>(da_handle handle) {
    return da_linmod_fit_s(handle);
}

inline da_status da_linmod_get_coef(da_handle handle, da_int *nc, double *x) {
    return da_handle_get_result_d(handle, da_result::da_linmod_coeff, nc, x);
}
inline da_status da_linmod_get_coef(da_handle handle, da_int *nc, float *x) {
    return da_handle_get_result_s(handle, da_result::da_linmod_coeff, nc, x);
}

inline da_status da_linmod_evaluate_model(da_handle handle, da_int n, da_int m, double *X,
                                          double *predictions) {
    return da_linmod_evaluate_model_d(handle, n, m, X, predictions);
}
inline da_status da_linmod_evaluate_model(da_handle handle, da_int n, da_int m, float *X,
                                          float *predictions) {
    return da_linmod_evaluate_model_s(handle, n, m, X, predictions);
}

/* Datastore overloaded functions */
inline da_status da_data_extract_column(da_datastore store, da_int idx, da_int m,
                                        da_int *col) {
    return da_data_extract_column_int(store, idx, m, col);
}
inline da_status da_data_extract_column(da_datastore store, da_int idx, da_int m,
                                        float *col) {
    return da_data_extract_column_real_s(store, idx, m, col);
}
inline da_status da_data_extract_column(da_datastore store, da_int idx, da_int m,
                                        double *col) {
    return da_data_extract_column_real_d(store, idx, m, col);
}
inline da_status da_data_extract_column(da_datastore store, da_int idx, da_int m,
                                        uint8_t *col) {
    return da_data_extract_column_uint8(store, idx, m, col);
}
inline da_status da_data_extract_column(da_datastore store, da_int idx, da_int m,
                                        char **col) {
    return da_data_extract_column_str(store, idx, m, col);
}

inline da_status da_data_extract_selection(da_datastore store, const char *key, da_int ld,
                                           da_int *data) {
    return da_data_extract_selection_int(store, key, ld, data);
}
inline da_status da_data_extract_selection(da_datastore store, const char *key, da_int ld,
                                           float *data) {
    return da_data_extract_selection_real_s(store, key, ld, data);
}
inline da_status da_data_extract_selection(da_datastore store, const char *key, da_int ld,
                                           double *data) {
    return da_data_extract_selection_real_d(store, key, ld, data);
}
inline da_status da_data_extract_selection(da_datastore store, const char *key, da_int ld,
                                           uint8_t *data) {
    return da_data_extract_selection_uint8(store, key, ld, data);
}

/* PCA overloaded functions */
inline da_status da_pca_set_data(da_handle handle, da_int n, da_int p, double *A,
                                 da_int lda) {
    return da_pca_set_data_d(handle, n, p, A, lda);
}

inline da_status da_pca_set_data(da_handle handle, da_int n, da_int p, float *A,
                                 da_int lda) {
    return da_pca_set_data_s(handle, n, p, A, lda);
}

template <class T> inline da_status da_pca_compute(da_handle handle);

template <> inline da_status da_pca_compute<double>(da_handle handle) {
    return da_pca_compute_d(handle);
}

template <> inline da_status da_pca_compute<float>(da_handle handle) {
    return da_pca_compute_s(handle);
}

inline da_status da_pca_transform(da_handle handle, da_int m, da_int p, double *X,
                                  da_int ldx, double *X_transform, da_int ldx_transform) {
    return da_pca_transform_d(handle, m, p, X, ldx, X_transform, ldx_transform);
}

inline da_status da_pca_transform(da_handle handle, da_int m, da_int p, float *X,
                                  da_int ldx, float *X_transform, da_int ldx_transform) {
    return da_pca_transform_s(handle, m, p, X, ldx, X_transform, ldx_transform);
}

inline da_status da_pca_inverse_transform(da_handle handle, da_int m, da_int r, double *X,
                                          da_int ldx, double *Xinv_transform,
                                          da_int ldxinv_transform) {
    return da_pca_inverse_transform_d(handle, m, r, X, ldx, Xinv_transform,
                                      ldxinv_transform);
}

inline da_status da_pca_inverse_transform(da_handle handle, da_int m, da_int r, float *X,
                                          da_int ldx, float *Xinv_transform,
                                          da_int ldxinv_transform) {
    return da_pca_inverse_transform_s(handle, m, r, X, ldx, Xinv_transform,
                                      ldxinv_transform);
}

#endif