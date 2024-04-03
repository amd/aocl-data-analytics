/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef AOCLDA_CPP_OVERLOADS
#define AOCLDA_CPP_OVERLOADS

#include "aoclda.h"
#include <iostream>

/* da_handle overloaded functions */
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

/* Options overloaded functions */
inline da_status da_options_set(da_handle handle, const char *option, float value) {
    return da_options_set_real_s(handle, option, value);
}
inline da_status da_options_set(da_handle handle, const char *option, double value) {
    return da_options_set_real_d(handle, option, value);
}
inline da_status da_options_set(da_handle handle, const char *option, da_int value) {
    return da_options_set_int(handle, option, value);
}
inline da_status da_options_set(da_handle handle, const char *option, const char *value) {
    return da_options_set_string(handle, option, value);
}
inline da_status da_options_get(da_handle handle, const char *option, double *value) {
    return da_options_get_real_d(handle, option, value);
}
inline da_status da_options_get(da_handle handle, const char *option, float *value) {
    return da_options_get_real_s(handle, option, value);
}
inline da_status da_options_get(da_handle handle, const char *option, da_int *value) {
    return da_options_get_int(handle, option, value);
}
inline da_status da_options_get(da_handle handle, const char *option, char *value,
                                da_int *lvalue) {
    return da_options_get_string(handle, option, value, lvalue);
}

inline da_status da_read_csv(da_datastore store, const char *filename, double **A,
                             da_int *n_rows, da_int *n_cols, char ***headings) {
    return da_read_csv_d(store, filename, A, n_rows, n_cols, headings);
}

inline da_status da_read_csv(da_datastore store, const char *filename, float **A,
                             da_int *n_rows, da_int *n_cols, char ***headings) {
    return da_read_csv_s(store, filename, A, n_rows, n_cols, headings);
}

inline da_status da_read_csv(da_datastore store, const char *filename, da_int **A,
                             da_int *n_rows, da_int *n_cols, char ***headings) {
    return da_read_csv_int(store, filename, A, n_rows, n_cols, headings);
}

inline da_status da_read_csv(da_datastore store, const char *filename, uint8_t **A,
                             da_int *n_rows, da_int *n_cols, char ***headings) {
    return da_read_csv_uint8(store, filename, A, n_rows, n_cols, headings);
}

inline da_status da_read_csv(da_datastore store, const char *filename, char ***A,
                             da_int *n_rows, da_int *n_cols, char ***headings) {
    return da_read_csv_string(store, filename, A, n_rows, n_cols, headings);
}

/* Basic statistics overloaded functions */
inline da_status da_mean(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                         da_int ldx, double *mean) {
    return da_mean_d(axis, n_rows, n_cols, X, ldx, mean);
}

inline da_status da_mean(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                         da_int ldx, float *mean) {
    return da_mean_s(axis, n_rows, n_cols, X, ldx, mean);
}

inline da_status da_harmonic_mean(da_axis axis, da_int n_rows, da_int n_cols,
                                  const double *X, da_int ldx, double *harmonic_mean) {
    return da_harmonic_mean_d(axis, n_rows, n_cols, X, ldx, harmonic_mean);
}

inline da_status da_harmonic_mean(da_axis axis, da_int n_rows, da_int n_cols,
                                  const float *X, da_int ldx, float *harmonic_mean) {
    return da_harmonic_mean_s(axis, n_rows, n_cols, X, ldx, harmonic_mean);
}

inline da_status da_geometric_mean(da_axis axis, da_int n_rows, da_int n_cols,
                                   const double *X, da_int ldx, double *geometric_mean) {
    return da_geometric_mean_d(axis, n_rows, n_cols, X, ldx, geometric_mean);
}

inline da_status da_geometric_mean(da_axis axis, da_int n_rows, da_int n_cols,
                                   const float *X, da_int ldx, float *geometric_mean) {
    return da_geometric_mean_s(axis, n_rows, n_cols, X, ldx, geometric_mean);
}

inline da_status da_variance(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                             da_int ldx, da_int dof, double *mean, double *variance) {
    return da_variance_d(axis, n_rows, n_cols, X, ldx, dof, mean, variance);
}

inline da_status da_variance(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                             da_int ldx, da_int dof, float *mean, float *variance) {
    return da_variance_s(axis, n_rows, n_cols, X, ldx, dof, mean, variance);
}

inline da_status da_skewness(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                             da_int ldx, double *mean, double *variance,
                             double *skewness) {
    return da_skewness_d(axis, n_rows, n_cols, X, ldx, mean, variance, skewness);
}

inline da_status da_skewness(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                             da_int ldx, float *mean, float *variance, float *skewness) {
    return da_skewness_s(axis, n_rows, n_cols, X, ldx, mean, variance, skewness);
}

inline da_status da_kurtosis(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                             da_int ldx, double *mean, double *variance,
                             double *kurtosis) {
    return da_kurtosis_d(axis, n_rows, n_cols, X, ldx, mean, variance, kurtosis);
}

inline da_status da_kurtosis(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                             da_int ldx, float *mean, float *variance, float *kurtosis) {
    return da_kurtosis_s(axis, n_rows, n_cols, X, ldx, mean, variance, kurtosis);
}

inline da_status da_moment(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                           da_int ldx, da_int k, da_int use_precomputed_mean,
                           double *mean, double *moment) {
    return da_moment_d(axis, n_rows, n_cols, X, ldx, k, use_precomputed_mean, mean,
                       moment);
}

inline da_status da_moment(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                           da_int ldx, da_int k, da_int use_precomputed_mean, float *mean,
                           float *moment) {
    return da_moment_s(axis, n_rows, n_cols, X, ldx, k, use_precomputed_mean, mean,
                       moment);
}

inline da_status da_quantile(da_axis axis, da_int n_rows, da_int n_cols, const double *X,
                             da_int ldx, double q, double *quantile,
                             da_quantile_type quantile_type) {
    return da_quantile_d(axis, n_rows, n_cols, X, ldx, q, quantile, quantile_type);
}

inline da_status da_quantile(da_axis axis, da_int n_rows, da_int n_cols, const float *X,
                             da_int ldx, float q, float *quantile,
                             da_quantile_type quantile_type) {
    return da_quantile_s(axis, n_rows, n_cols, X, ldx, q, quantile, quantile_type);
}

inline da_status da_five_point_summary(da_axis axis, da_int n_rows, da_int n_cols,
                                       const double *X, da_int ldx, double *minimum,
                                       double *lower_hinge, double *median,
                                       double *upper_hinge, double *maximum) {
    return da_five_point_summary_d(axis, n_rows, n_cols, X, ldx, minimum, lower_hinge,
                                   median, upper_hinge, maximum);
}

inline da_status da_five_point_summary(da_axis axis, da_int n_rows, da_int n_cols,
                                       const float *X, da_int ldx, float *minimum,
                                       float *lower_hinge, float *median,
                                       float *upper_hinge, float *maximum) {
    return da_five_point_summary_s(axis, n_rows, n_cols, X, ldx, minimum, lower_hinge,
                                   median, upper_hinge, maximum);
}

inline da_status da_standardize(da_axis axis, da_int n_rows, da_int n_cols, double *X,
                                da_int ldx, da_int dof, da_int mode, double *shift,
                                double *scale) {
    return da_standardize_d(axis, n_rows, n_cols, X, ldx, dof, mode, shift, scale);
}

inline da_status da_standardize(da_axis axis, da_int n_rows, da_int n_cols, float *X,
                                da_int ldx, da_int dof, da_int mode, float *shift,
                                float *scale) {
    return da_standardize_s(axis, n_rows, n_cols, X, ldx, dof, mode, shift, scale);
}

inline da_status da_covariance_matrix(da_int n_rows, da_int n_cols, const float *X,
                                      da_int ldx, da_int dof, float *cov, da_int ldcov) {
    return da_covariance_matrix_s(n_rows, n_cols, X, ldx, dof, cov, ldcov);
}

inline da_status da_covariance_matrix(da_int n_rows, da_int n_cols, const double *X,
                                      da_int ldx, da_int dof, double *cov, da_int ldcov) {
    return da_covariance_matrix_d(n_rows, n_cols, X, ldx, dof, cov, ldcov);
}

inline da_status da_correlation_matrix(da_int n_rows, da_int n_cols, const float *X,
                                       da_int ldx, float *corr, da_int ldcorr) {
    return da_correlation_matrix_s(n_rows, n_cols, X, ldx, corr, ldcorr);
}

inline da_status da_correlation_matrix(da_int n_rows, da_int n_cols, const double *X,
                                       da_int ldx, double *corr, da_int ldcorr) {
    return da_correlation_matrix_d(n_rows, n_cols, X, ldx, corr, ldcorr);
}

/* Linear model overloaded functions */
template <class T> da_status da_linmod_select_model(da_handle handle, linmod_model mod);
template <> da_status da_linmod_select_model<double>(da_handle handle, linmod_model mod) {
    return da_linmod_select_model_d(handle, mod);
}
template <> da_status da_linmod_select_model<float>(da_handle handle, linmod_model mod) {
    return da_linmod_select_model_s(handle, mod);
}

inline da_status da_linmod_define_features(da_handle handle, da_int n_samples,
                                           da_int n_features, float *X, float *y) {
    return da_linmod_define_features_s(handle, n_samples, n_features, X, y);
}
inline da_status da_linmod_define_features(da_handle handle, da_int n_samples,
                                           da_int n_features, double *X, double *y) {
    return da_linmod_define_features_d(handle, n_samples, n_features, X, y);
}

template <class T> da_status da_linmod_fit(da_handle handle);
template <> da_status da_linmod_fit<double>(da_handle handle) {
    return da_linmod_fit_d(handle);
}
template <> da_status da_linmod_fit<float>(da_handle handle) {
    return da_linmod_fit_s(handle);
}

template <class T>
da_status da_linmod_fit_start(da_handle handle, da_int ncoef, T *coefs);
template <>
da_status da_linmod_fit_start<double>(da_handle handle, da_int ncoef, double *coefs) {
    return da_linmod_fit_start_d(handle, ncoef, coefs);
}
template <>
da_status da_linmod_fit_start<float>(da_handle handle, da_int ncoef, float *coefs) {
    return da_linmod_fit_start_s(handle, ncoef, coefs);
}

inline da_status da_linmod_evaluate_model(da_handle handle, da_int nsamples, da_int nfeat,
                                          double *X, double *predictions,
                                          double *observations = nullptr,
                                          double *loss = nullptr) {
    return da_linmod_evaluate_model_d(handle, nsamples, nfeat, X, predictions,
                                      observations, loss);
}

inline da_status da_linmod_evaluate_model(da_handle handle, da_int nsamples, da_int nfeat,
                                          float *X, float *predictions,
                                          float *observations = nullptr,
                                          float *loss = nullptr) {
    return da_linmod_evaluate_model_s(handle, nsamples, nfeat, X, predictions,
                                      observations, loss);
}

/* Datastore overloaded functions */
inline da_status da_data_extract_column(da_datastore store, da_int idx, da_int dim,
                                        da_int *col) {
    return da_data_extract_column_int(store, idx, dim, col);
}
inline da_status da_data_extract_column(da_datastore store, da_int idx, da_int dim,
                                        float *col) {
    return da_data_extract_column_real_s(store, idx, dim, col);
}
inline da_status da_data_extract_column(da_datastore store, da_int idx, da_int dim,
                                        double *col) {
    return da_data_extract_column_real_d(store, idx, dim, col);
}
inline da_status da_data_extract_column(da_datastore store, da_int idx, da_int dim,
                                        uint8_t *col) {
    return da_data_extract_column_uint8(store, idx, dim, col);
}
inline da_status da_data_extract_column(da_datastore store, da_int idx, da_int dim,
                                        char **col) {
    return da_data_extract_column_str(store, idx, dim, col);
}

inline da_status da_data_extract_selection(da_datastore store, const char *key,
                                           da_int *data, da_int lddata) {
    return da_data_extract_selection_int(store, key, data, lddata);
}
inline da_status da_data_extract_selection(da_datastore store, const char *key,
                                           float *data, da_int lddata) {
    return da_data_extract_selection_real_s(store, key, data, lddata);
}
inline da_status da_data_extract_selection(da_datastore store, const char *key,
                                           double *data, da_int lddata) {
    return da_data_extract_selection_real_d(store, key, data, lddata);
}
inline da_status da_data_extract_selection(da_datastore store, const char *key,
                                           uint8_t *data, da_int lddata) {
    return da_data_extract_selection_uint8(store, key, data, lddata);
}

/* PCA overloaded functions */
inline da_status da_pca_set_data(da_handle handle, da_int n_samples, da_int n_features,
                                 const double *A, da_int lda) {
    return da_pca_set_data_d(handle, n_samples, n_features, A, lda);
}

inline da_status da_pca_set_data(da_handle handle, da_int n_samples, da_int n_features,
                                 const float *A, da_int lda) {
    return da_pca_set_data_s(handle, n_samples, n_features, A, lda);
}

template <class T> inline da_status da_pca_compute(da_handle handle);

template <> inline da_status da_pca_compute<double>(da_handle handle) {
    return da_pca_compute_d(handle);
}

template <> inline da_status da_pca_compute<float>(da_handle handle) {
    return da_pca_compute_s(handle);
}

inline da_status da_pca_transform(da_handle handle, da_int m_samples, da_int m_features,
                                  const double *X, da_int ldx, double *X_transform,
                                  da_int ldx_transform) {
    return da_pca_transform_d(handle, m_samples, m_features, X, ldx, X_transform,
                              ldx_transform);
}

inline da_status da_pca_transform(da_handle handle, da_int m_samples, da_int m_features,
                                  const float *X, da_int ldx, float *X_transform,
                                  da_int ldx_transform) {
    return da_pca_transform_s(handle, m_samples, m_features, X, ldx, X_transform,
                              ldx_transform);
}

inline da_status da_pca_inverse_transform(da_handle handle, da_int k_samples,
                                          da_int k_features, const double *Y, da_int ldy,
                                          double *Y_inv_transform,
                                          da_int ldy_inv_transform) {
    return da_pca_inverse_transform_d(handle, k_samples, k_features, Y, ldy,
                                      Y_inv_transform, ldy_inv_transform);
}

inline da_status da_pca_inverse_transform(da_handle handle, da_int k_samples,
                                          da_int k_features, const float *Y, da_int ldy,
                                          float *Y_inv_transform,
                                          da_int ldy_inv_transform) {
    return da_pca_inverse_transform_s(handle, k_samples, k_features, Y, ldy,
                                      Y_inv_transform, ldy_inv_transform);
}

/* k-means overloaded functions */
inline da_status da_kmeans_set_data(da_handle handle, da_int n_samples, da_int n_features,
                                    const double *A, da_int lda) {
    return da_kmeans_set_data_d(handle, n_samples, n_features, A, lda);
}

inline da_status da_kmeans_set_data(da_handle handle, da_int n_samples, da_int n_features,
                                    const float *A, da_int lda) {
    return da_kmeans_set_data_s(handle, n_samples, n_features, A, lda);
}

inline da_status da_kmeans_set_init_centres(da_handle handle, const float *C,
                                            da_int ldc) {
    return da_kmeans_set_init_centres_s(handle, C, ldc);
}

inline da_status da_kmeans_set_init_centres(da_handle handle, const double *C,
                                            da_int ldc) {
    return da_kmeans_set_init_centres_d(handle, C, ldc);
}

template <class T> inline da_status da_kmeans_compute(da_handle handle);

template <> inline da_status da_kmeans_compute<double>(da_handle handle) {
    return da_kmeans_compute_d(handle);
}

template <> inline da_status da_kmeans_compute<float>(da_handle handle) {
    return da_kmeans_compute_s(handle);
}

inline da_status da_kmeans_transform(da_handle handle, da_int m_samples,
                                     da_int m_features, const double *X, da_int ldx,
                                     double *X_transform, da_int ldx_transform) {
    return da_kmeans_transform_d(handle, m_samples, m_features, X, ldx, X_transform,
                                 ldx_transform);
}

inline da_status da_kmeans_transform(da_handle handle, da_int m_samples,
                                     da_int m_features, const float *X, da_int ldx,
                                     float *X_transform, da_int ldx_transform) {
    return da_kmeans_transform_s(handle, m_samples, m_features, X, ldx, X_transform,
                                 ldx_transform);
}

inline da_status da_kmeans_predict(da_handle handle, da_int k_samples, da_int k_features,
                                   const double *Y, da_int ldy, da_int *Y_labels) {
    return da_kmeans_predict_d(handle, k_samples, k_features, Y, ldy, Y_labels);
}

inline da_status da_kmeans_predict(da_handle handle, da_int k_samples, da_int k_features,
                                   const float *Y, da_int ldy, da_int *Y_labels) {
    return da_kmeans_predict_s(handle, k_samples, k_features, Y, ldy, Y_labels);
}

/* Decision Forest overloaded functions */

inline da_status da_df_set_training_data(da_handle handle, da_int n_obs,
                                         da_int n_features, double *X, da_int ldx,
                                         uint8_t *y) {
    return da_df_set_training_data_d(handle, n_obs, n_features, X, ldx, y);
}

inline da_status da_df_set_training_data(da_handle handle, da_int n_obs,
                                         da_int n_features, float *X, da_int ldx,
                                         uint8_t *y) {
    return da_df_set_training_data_s(handle, n_obs, n_features, X, ldx, y);
}

template <typename T> inline da_status da_df_fit(da_handle handle);

template <> inline da_status da_df_fit<float>(da_handle handle) {
    return da_df_fit_s(handle);
}

template <> inline da_status da_df_fit<double>(da_handle handle) {
    return da_df_fit_d(handle);
}

inline da_status da_df_score(da_handle handle, da_int n_obs, da_int n_features,
                             double *X_test, da_int ldx_test, uint8_t *y_test,
                             double *score) {
    return da_df_score_d(handle, n_obs, n_features, X_test, ldx_test, y_test, score);
}

inline da_status da_df_score(da_handle handle, da_int n_obs, da_int n_features,
                             float *X_test, da_int ldx_test, uint8_t *y_test,
                             float *score) {
    return da_df_score_s(handle, n_obs, n_features, X_test, ldx_test, y_test, score);
}

inline da_status da_df_predict(da_handle handle, da_int n_obs, da_int n_features,
                               double *X_test, da_int ldx_test, uint8_t *y_pred) {
    return da_df_predict_d(handle, n_obs, n_features, X_test, ldx_test, y_pred);
}

inline da_status da_df_predict(da_handle handle, da_int n_obs, da_int n_features,
                               float *X_test, da_int ldx_test, uint8_t *y_pred) {
    return da_df_predict_s(handle, n_obs, n_features, X_test, ldx_test, y_pred);
}

#endif // AOCLDA_CPP_OVERLOADS
