/*
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
inline da_status da_options_get(da_handle handle, const char *option, char *value,
                                da_int *lvalue, da_int *key) {
    return da_options_get_string_key(handle, option, value, lvalue, key);
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
inline da_status da_mean(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                         const double *X, da_int ldx, double *mean) {
    return da_mean_d(order, axis, n_rows, n_cols, X, ldx, mean);
}

inline da_status da_mean(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                         const float *X, da_int ldx, float *mean) {
    return da_mean_s(order, axis, n_rows, n_cols, X, ldx, mean);
}

inline da_status da_harmonic_mean(da_order order, da_axis axis, da_int n_rows,
                                  da_int n_cols, const double *X, da_int ldx,
                                  double *harmonic_mean) {
    return da_harmonic_mean_d(order, axis, n_rows, n_cols, X, ldx, harmonic_mean);
}

inline da_status da_harmonic_mean(da_order order, da_axis axis, da_int n_rows,
                                  da_int n_cols, const float *X, da_int ldx,
                                  float *harmonic_mean) {
    return da_harmonic_mean_s(order, axis, n_rows, n_cols, X, ldx, harmonic_mean);
}

inline da_status da_geometric_mean(da_order order, da_axis axis, da_int n_rows,
                                   da_int n_cols, const double *X, da_int ldx,
                                   double *geometric_mean) {
    return da_geometric_mean_d(order, axis, n_rows, n_cols, X, ldx, geometric_mean);
}

inline da_status da_geometric_mean(da_order order, da_axis axis, da_int n_rows,
                                   da_int n_cols, const float *X, da_int ldx,
                                   float *geometric_mean) {
    return da_geometric_mean_s(order, axis, n_rows, n_cols, X, ldx, geometric_mean);
}

inline da_status da_variance(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                             const double *X, da_int ldx, da_int dof, double *mean,
                             double *variance) {
    return da_variance_d(order, axis, n_rows, n_cols, X, ldx, dof, mean, variance);
}

inline da_status da_variance(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                             const float *X, da_int ldx, da_int dof, float *mean,
                             float *variance) {
    return da_variance_s(order, axis, n_rows, n_cols, X, ldx, dof, mean, variance);
}

inline da_status da_skewness(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                             const double *X, da_int ldx, double *mean, double *variance,
                             double *skewness) {
    return da_skewness_d(order, axis, n_rows, n_cols, X, ldx, mean, variance, skewness);
}

inline da_status da_skewness(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                             const float *X, da_int ldx, float *mean, float *variance,
                             float *skewness) {
    return da_skewness_s(order, axis, n_rows, n_cols, X, ldx, mean, variance, skewness);
}

inline da_status da_kurtosis(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                             const double *X, da_int ldx, double *mean, double *variance,
                             double *kurtosis) {
    return da_kurtosis_d(order, axis, n_rows, n_cols, X, ldx, mean, variance, kurtosis);
}

inline da_status da_kurtosis(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                             const float *X, da_int ldx, float *mean, float *variance,
                             float *kurtosis) {
    return da_kurtosis_s(order, axis, n_rows, n_cols, X, ldx, mean, variance, kurtosis);
}

inline da_status da_moment(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                           const double *X, da_int ldx, da_int k,
                           da_int use_precomputed_mean, double *mean, double *moment) {
    return da_moment_d(order, axis, n_rows, n_cols, X, ldx, k, use_precomputed_mean, mean,
                       moment);
}

inline da_status da_moment(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                           const float *X, da_int ldx, da_int k,
                           da_int use_precomputed_mean, float *mean, float *moment) {
    return da_moment_s(order, axis, n_rows, n_cols, X, ldx, k, use_precomputed_mean, mean,
                       moment);
}

inline da_status da_quantile(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                             const double *X, da_int ldx, double q, double *quantile,
                             da_quantile_type quantile_type) {
    return da_quantile_d(order, axis, n_rows, n_cols, X, ldx, q, quantile, quantile_type);
}

inline da_status da_quantile(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                             const float *X, da_int ldx, float q, float *quantile,
                             da_quantile_type quantile_type) {
    return da_quantile_s(order, axis, n_rows, n_cols, X, ldx, q, quantile, quantile_type);
}

inline da_status da_five_point_summary(da_order order, da_axis axis, da_int n_rows,
                                       da_int n_cols, const double *X, da_int ldx,
                                       double *minimum, double *lower_hinge,
                                       double *median, double *upper_hinge,
                                       double *maximum) {
    return da_five_point_summary_d(order, axis, n_rows, n_cols, X, ldx, minimum,
                                   lower_hinge, median, upper_hinge, maximum);
}

inline da_status da_five_point_summary(da_order order, da_axis axis, da_int n_rows,
                                       da_int n_cols, const float *X, da_int ldx,
                                       float *minimum, float *lower_hinge, float *median,
                                       float *upper_hinge, float *maximum) {
    return da_five_point_summary_s(order, axis, n_rows, n_cols, X, ldx, minimum,
                                   lower_hinge, median, upper_hinge, maximum);
}

inline da_status da_standardize(da_order order, da_axis axis, da_int n_rows,
                                da_int n_cols, double *X, da_int ldx, da_int dof,
                                da_int mode, double *shift, double *scale) {
    return da_standardize_d(order, axis, n_rows, n_cols, X, ldx, dof, mode, shift, scale);
}

inline da_status da_standardize(da_order order, da_axis axis, da_int n_rows,
                                da_int n_cols, float *X, da_int ldx, da_int dof,
                                da_int mode, float *shift, float *scale) {
    return da_standardize_s(order, axis, n_rows, n_cols, X, ldx, dof, mode, shift, scale);
}

inline da_status da_covariance_matrix(da_order order, da_int n_rows, da_int n_cols,
                                      const float *X, da_int ldx, da_int dof, float *cov,
                                      da_int ldcov, da_int assume_centered) {
    return da_covariance_matrix_s(order, n_rows, n_cols, X, ldx, dof, cov, ldcov,
                                  assume_centered);
}

inline da_status da_covariance_matrix(da_order order, da_int n_rows, da_int n_cols,
                                      const double *X, da_int ldx, da_int dof,
                                      double *cov, da_int ldcov, da_int assume_centered) {
    return da_covariance_matrix_d(order, n_rows, n_cols, X, ldx, dof, cov, ldcov,
                                  assume_centered);
}

inline da_status da_correlation_matrix(da_order order, da_int n_rows, da_int n_cols,
                                       const float *X, da_int ldx, float *corr,
                                       da_int ldcorr) {
    return da_correlation_matrix_s(order, n_rows, n_cols, X, ldx, corr, ldcorr);
}

inline da_status da_correlation_matrix(da_order order, da_int n_rows, da_int n_cols,
                                       const double *X, da_int ldx, double *corr,
                                       da_int ldcorr) {
    return da_correlation_matrix_d(order, n_rows, n_cols, X, ldx, corr, ldcorr);
}

/* Linear model overloaded functions */
template <class T> da_status da_linmod_select_model(da_handle handle, linmod_model mod);

inline da_status da_linmod_define_features(da_handle handle, da_int n_samples,
                                           da_int n_features, const float *X, da_int ldx,
                                           const float *y) {
    return da_linmod_define_features_s(handle, n_samples, n_features, X, ldx, y);
}
inline da_status da_linmod_define_features(da_handle handle, da_int n_samples,
                                           da_int n_features, const double *X, da_int ldx,
                                           const double *y) {
    return da_linmod_define_features_d(handle, n_samples, n_features, X, ldx, y);
}

template <class T> da_status da_linmod_fit(da_handle handle);

template <class T>
da_status da_linmod_fit_start(da_handle handle, da_int ncoef, const T *coefs);

inline da_status da_linmod_evaluate_model(da_handle handle, da_int n_samples,
                                          da_int n_features, const double *X, da_int ldx,
                                          double *predictions,
                                          double *observations = nullptr,
                                          double *loss = nullptr) {
    return da_linmod_evaluate_model_d(handle, n_samples, n_features, X, ldx, predictions,
                                      observations, loss);
}

inline da_status da_linmod_evaluate_model(da_handle handle, da_int n_samples,
                                          da_int n_features, const float *X, da_int ldx,
                                          float *predictions,
                                          float *observations = nullptr,
                                          float *loss = nullptr) {
    return da_linmod_evaluate_model_s(handle, n_samples, n_features, X, ldx, predictions,
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
                                           da_order order, da_int *data, da_int lddata) {
    return da_data_extract_selection_int(store, key, order, data, lddata);
}
inline da_status da_data_extract_selection(da_datastore store, const char *key,
                                           da_order order, float *data, da_int lddata) {
    return da_data_extract_selection_real_s(store, key, order, data, lddata);
}
inline da_status da_data_extract_selection(da_datastore store, const char *key,
                                           da_order order, double *data, da_int lddata) {
    return da_data_extract_selection_real_d(store, key, order, data, lddata);
}
inline da_status da_data_extract_selection(da_datastore store, const char *key,
                                           da_order order, uint8_t *data, da_int lddata) {
    return da_data_extract_selection_uint8(store, key, order, data, lddata);
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

template <class T> da_status da_pca_compute(da_handle handle);

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

template <class T> da_status da_kmeans_compute(da_handle handle);

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

/* DBSCAN overloaded functions */
inline da_status da_dbscan_set_data(da_handle handle, da_int n_samples, da_int n_features,
                                    const double *A, da_int lda) {
    return da_dbscan_set_data_d(handle, n_samples, n_features, A, lda);
}

inline da_status da_dbscan_set_data(da_handle handle, da_int n_samples, da_int n_features,
                                    const float *A, da_int lda) {
    return da_dbscan_set_data_s(handle, n_samples, n_features, A, lda);
}

template <class T> da_status da_dbscan_compute(da_handle handle);

/* Decision Forest overloaded functions */
/* Decision tree */
inline da_status da_tree_set_training_data(da_handle handle, da_int n_samples,
                                           da_int n_features, da_int n_class,
                                           const double *X, da_int ldx, const da_int *y,
                                           const da_int *categorical_feat = nullptr) {
    return da_tree_set_training_data_d(handle, n_samples, n_features, n_class, X, ldx, y,
                                       categorical_feat);
}
inline da_status da_tree_set_training_data(da_handle handle, da_int n_samples,
                                           da_int n_features, da_int n_class,
                                           const float *X, da_int ldx, const da_int *y,
                                           const da_int *categorical_feat = nullptr) {
    return da_tree_set_training_data_s(handle, n_samples, n_features, n_class, X, ldx, y,
                                       categorical_feat);
}

template <typename T> da_status da_tree_fit(da_handle handle);

inline da_status da_tree_predict(da_handle handle, da_int n_samples, da_int n_features,
                                 const double *X_test, da_int ldx_test, da_int *y_pred) {
    return da_tree_predict_d(handle, n_samples, n_features, X_test, ldx_test, y_pred);
}
inline da_status da_tree_predict(da_handle handle, da_int n_samples, da_int n_features,
                                 const float *X_test, da_int ldx_test, da_int *y_pred) {
    return da_tree_predict_s(handle, n_samples, n_features, X_test, ldx_test, y_pred);
}

inline da_status da_tree_predict_proba(da_handle handle, da_int n_obs, da_int n_features,
                                       const double *X_test, da_int ldx_test,
                                       double *y_pred, da_int n_class, da_int ldy) {
    return da_tree_predict_proba_d(handle, n_obs, n_features, X_test, ldx_test, y_pred,
                                   n_class, ldy);
}

inline da_status da_tree_predict_proba(da_handle handle, da_int n_obs, da_int n_features,
                                       const float *X_test, da_int ldx_test,
                                       float *y_pred, da_int n_class, da_int ldy) {
    return da_tree_predict_proba_s(handle, n_obs, n_features, X_test, ldx_test, y_pred,
                                   n_class, ldy);
}

inline da_status da_tree_predict_log_proba(da_handle handle, da_int n_obs,
                                           da_int n_features, const double *X_test,
                                           da_int ldx_test, double *y_pred,
                                           da_int n_class, da_int ldy) {
    return da_tree_predict_log_proba_d(handle, n_obs, n_features, X_test, ldx_test,
                                       y_pred, n_class, ldy);
}

inline da_status da_tree_predict_log_proba(da_handle handle, da_int n_obs,
                                           da_int n_features, const float *X_test,
                                           da_int ldx_test, float *y_pred, da_int n_class,
                                           da_int ldy) {
    return da_tree_predict_log_proba_s(handle, n_obs, n_features, X_test, ldx_test,
                                       y_pred, n_class, ldy);
}

inline da_status da_tree_score(da_handle handle, da_int n_samples, da_int n_features,
                               const double *X_test, da_int ldx_test,
                               const da_int *y_test, double *mean_accuracy) {
    return da_tree_score_d(handle, n_samples, n_features, X_test, ldx_test, y_test,
                           mean_accuracy);
}
inline da_status da_tree_score(da_handle handle, da_int n_samples, da_int n_features,
                               const float *X_test, da_int ldx_test, const da_int *y_test,
                               float *mean_accuracy) {
    return da_tree_score_s(handle, n_samples, n_features, X_test, ldx_test, y_test,
                           mean_accuracy);
}

/* Random forest */
inline da_status da_forest_set_training_data(da_handle handle, da_int n_samples,
                                             da_int n_features, da_int n_class,
                                             const double *X, da_int ldx, const da_int *y,
                                             const da_int *categorical_feat = nullptr) {
    return da_forest_set_training_data_d(handle, n_samples, n_features, n_class, X, ldx,
                                         y, categorical_feat);
}
inline da_status da_forest_set_training_data(da_handle handle, da_int n_samples,
                                             da_int n_features, da_int n_class,
                                             const float *X, da_int ldx, const da_int *y,
                                             const da_int *categorical_feat = nullptr) {
    return da_forest_set_training_data_s(handle, n_samples, n_features, n_class, X, ldx,
                                         y, categorical_feat);
}

template <typename T> da_status da_forest_fit(da_handle handle);

inline da_status da_forest_predict(da_handle handle, da_int n_samples, da_int n_features,
                                   const double *X_test, da_int ldx_test,
                                   da_int *y_pred) {
    return da_forest_predict_d(handle, n_samples, n_features, X_test, ldx_test, y_pred);
}
inline da_status da_forest_predict(da_handle handle, da_int n_samples, da_int n_features,
                                   const float *X_test, da_int ldx_test, da_int *y_pred) {
    return da_forest_predict_s(handle, n_samples, n_features, X_test, ldx_test, y_pred);
}

inline da_status da_forest_predict_proba(da_handle handle, da_int n_obs,
                                         da_int n_features, const double *X_test,
                                         da_int ldx_test, double *y_pred, da_int n_class,
                                         da_int ldy) {
    return da_forest_predict_proba_d(handle, n_obs, n_features, X_test, ldx_test, y_pred,
                                     n_class, ldy);
}

inline da_status da_forest_predict_proba(da_handle handle, da_int n_obs,
                                         da_int n_features, const float *X_test,
                                         da_int ldx_test, float *y_pred, da_int n_class,
                                         da_int ldy) {
    return da_forest_predict_proba_s(handle, n_obs, n_features, X_test, ldx_test, y_pred,
                                     n_class, ldy);
}

inline da_status da_forest_predict_log_proba(da_handle handle, da_int n_obs,
                                             da_int n_features, const double *X_test,
                                             da_int ldx_test, double *y_pred,
                                             da_int n_class, da_int ldy) {
    return da_forest_predict_log_proba_d(handle, n_obs, n_features, X_test, ldx_test,
                                         y_pred, n_class, ldy);
}

inline da_status da_forest_predict_log_proba(da_handle handle, da_int n_obs,
                                             da_int n_features, const float *X_test,
                                             da_int ldx_test, float *y_pred,
                                             da_int n_class, da_int ldy) {
    return da_forest_predict_log_proba_s(handle, n_obs, n_features, X_test, ldx_test,
                                         y_pred, n_class, ldy);
}

inline da_status da_forest_score(da_handle handle, da_int n_samples, da_int n_features,
                                 const double *X_test, da_int ldx_test,
                                 const da_int *y_test, double *mean_accuracy) {
    return da_forest_score_d(handle, n_samples, n_features, X_test, ldx_test, y_test,
                             mean_accuracy);
}
inline da_status da_forest_score(da_handle handle, da_int n_samples, da_int n_features,
                                 const float *X_test, da_int ldx_test,
                                 const da_int *y_test, float *mean_accuracy) {
    return da_forest_score_s(handle, n_samples, n_features, X_test, ldx_test, y_test,
                             mean_accuracy);
}

inline da_status da_nlls_define_residuals(da_handle handle, da_int n_coef, da_int n_res,
                                          da_resfun_t_d *resfun, da_resgrd_t_d *resgrd,
                                          da_reshes_t_d *reshes, da_reshp_t_d *reshp) {
    return da_nlls_define_residuals_d(handle, n_coef, n_res, resfun, resgrd, *reshes,
                                      *reshp);
}

inline da_status da_nlls_define_residuals(da_handle handle, da_int n_coef, da_int n_res,
                                          da_resfun_t_s *resfun, da_resgrd_t_s *resgrd,
                                          da_reshes_t_s *reshes, da_reshp_t_s *reshp) {
    return da_nlls_define_residuals_s(handle, n_coef, n_res, resfun, resgrd, reshes,
                                      reshp);
}

inline da_status da_nlls_define_bounds(da_handle handle, da_int n_coef, double *lower,
                                       double *upper) {
    return da_nlls_define_bounds_d(handle, n_coef, lower, upper);
}

inline da_status da_nlls_define_bounds(da_handle handle, da_int n_coef, float *lower,
                                       float *upper) {
    return da_nlls_define_bounds_s(handle, n_coef, lower, upper);
}

inline da_status da_nlls_define_weights(da_handle handle, da_int n_coef,
                                        double *weights) {
    return da_nlls_define_weights_d(handle, n_coef, weights);
}

inline da_status da_nlls_define_weights(da_handle handle, da_int n_coef, float *weights) {
    return da_nlls_define_weights_s(handle, n_coef, weights);
}

inline da_status da_nlls_fit(da_handle handle, da_int n_coef, double *coef, void *udata) {
    return da_nlls_fit_d(handle, n_coef, coef, udata);
}

inline da_status da_nlls_fit(da_handle handle, da_int n_coefs, float *coefs,
                             void *udata) {
    return da_nlls_fit_s(handle, n_coefs, coefs, udata);
}

/* Pairwise distances overloaded functions */
inline da_status da_pairwise_distances(da_order order, da_int m, da_int n, da_int k,
                                       const double *X, da_int ldx, const double *Y,
                                       da_int ldy, double *D, da_int ldd, double p,
                                       da_metric metric = da_euclidean) {
    return da_pairwise_distances_d(order, m, n, k, X, ldx, Y, ldy, D, ldd, p, metric);
}

inline da_status da_pairwise_distances(da_order order, da_int m, da_int n, da_int k,
                                       const float *X, da_int ldx, const float *Y,
                                       da_int ldy, float *D, da_int ldd, float p,
                                       da_metric metric = da_euclidean) {
    return da_pairwise_distances_s(order, m, n, k, X, ldx, Y, ldy, D, ldd, p, metric);
}

/* k-NN for classification and regression functions */
inline da_status da_nn_set_data(da_handle handle, da_int n_samples, da_int n_features,
                                const float *X_train, da_int ldx_train) {
    return da_nn_set_data_s(handle, n_samples, n_features, X_train, ldx_train);
}

inline da_status da_nn_set_data(da_handle handle, da_int n_samples, da_int n_features,
                                const double *X_train, da_int ldx_train) {
    return da_nn_set_data_d(handle, n_samples, n_features, X_train, ldx_train);
}

template <class T>
da_status da_nn_set_labels(da_handle handle, da_int n_samples, const da_int *y_train);

inline da_status da_nn_set_targets(da_handle handle, da_int n_samples,
                                   const float *y_train) {
    return da_nn_set_targets_s(handle, n_samples, y_train);
}

inline da_status da_nn_set_targets(da_handle handle, da_int n_samples,
                                   const double *y_train) {
    return da_nn_set_targets_d(handle, n_samples, y_train);
}

inline da_status da_nn_kneighbors(da_handle handle, da_int n_queries, da_int n_features,
                                  const double *X_test, da_int ldx_test, da_int *n_ind,
                                  double *n_dist, da_int k, da_int return_distance) {
    return da_nn_kneighbors_d(handle, n_queries, n_features, X_test, ldx_test, n_ind,
                              n_dist, k, return_distance);
}

inline da_status da_nn_kneighbors(da_handle handle, da_int n_queries, da_int n_features,
                                  const float *X_test, da_int ldx_test, da_int *n_ind,
                                  float *n_dist, da_int k, da_int return_distance) {
    return da_nn_kneighbors_s(handle, n_queries, n_features, X_test, ldx_test, n_ind,
                              n_dist, k, return_distance);
}

template <class T>
da_status da_nn_classes(da_handle handle, da_int *n_classes, da_int *classes);

inline da_status da_nn_classifier_predict_proba(da_handle handle, da_int n_queries,
                                                da_int n_features, const double *X_test,
                                                da_int ldx_test, double *proba,
                                                da_nn_search_mode search_mode) {
    return da_nn_classifier_predict_proba_d(handle, n_queries, n_features, X_test,
                                            ldx_test, proba, search_mode);
}

inline da_status da_nn_classifier_predict_proba(da_handle handle, da_int n_queries,
                                                da_int n_features, const float *X_test,
                                                da_int ldx_test, float *proba,
                                                da_nn_search_mode search_mode) {
    return da_nn_classifier_predict_proba_s(handle, n_queries, n_features, X_test,
                                            ldx_test, proba, search_mode);
}

inline da_status da_nn_classifier_predict(da_handle handle, da_int n_queries,
                                          da_int n_features, const double *X_test,
                                          da_int ldx_test, da_int *y_test,
                                          da_nn_search_mode search_mode) {
    return da_nn_classifier_predict_d(handle, n_queries, n_features, X_test, ldx_test,
                                      y_test, search_mode);
}

inline da_status da_nn_classifier_predict(da_handle handle, da_int n_queries,
                                          da_int n_features, const float *X_test,
                                          da_int ldx_test, da_int *y_test,
                                          da_nn_search_mode search_mode) {
    return da_nn_classifier_predict_s(handle, n_queries, n_features, X_test, ldx_test,
                                      y_test, search_mode);
}
inline da_status da_nn_regressor_predict(da_handle handle, da_int n_queries,
                                         da_int n_features, const double *X_test,
                                         da_int ldx_test, double *y_test,
                                         da_nn_search_mode search_mode) {
    return da_nn_regressor_predict_d(handle, n_queries, n_features, X_test, ldx_test,
                                     y_test, search_mode);
}

inline da_status da_nn_regressor_predict(da_handle handle, da_int n_queries,
                                         da_int n_features, const float *X_test,
                                         da_int ldx_test, float *y_test,
                                         da_nn_search_mode search_mode) {
    return da_nn_regressor_predict_s(handle, n_queries, n_features, X_test, ldx_test,
                                     y_test, search_mode);
}

inline da_status da_nn_radius_neighbors(da_handle handle, da_int n_queries,
                                        da_int n_features, const double *X_test,
                                        da_int ldx_test, double radius,
                                        da_int return_distance, da_int sort_results) {
    return da_nn_radius_neighbors_d(handle, n_queries, n_features, X_test, ldx_test,
                                    radius, return_distance, sort_results);
}

inline da_status da_nn_radius_neighbors(da_handle handle, da_int n_queries,
                                        da_int n_features, const float *X_test,
                                        da_int ldx_test, float radius,
                                        da_int return_distance, da_int sort_results) {
    return da_nn_radius_neighbors_s(handle, n_queries, n_features, X_test, ldx_test,
                                    radius, return_distance, sort_results);
}

/* Utility overloaded functions */
inline da_status da_check_data(da_order order, da_int n_rows, da_int n_cols,
                               const double *X, da_int ldx) {
    return da_check_data_d(order, n_rows, n_cols, X, ldx);
}
inline da_status da_check_data(da_order order, da_int n_rows, da_int n_cols,
                               const float *X, da_int ldx) {
    return da_check_data_s(order, n_rows, n_cols, X, ldx);
}
inline da_status da_switch_order_copy(da_order order_X, da_int n_rows, da_int n_cols,
                                      const float *X, da_int ldx, float *Y, da_int ldy) {
    return da_switch_order_copy_s(order_X, n_rows, n_cols, X, ldx, Y, ldy);
}
inline da_status da_switch_order_copy(da_order order_X, da_int n_rows, da_int n_cols,
                                      const double *X, da_int ldx, double *Y,
                                      da_int ldy) {
    return da_switch_order_copy_d(order_X, n_rows, n_cols, X, ldx, Y, ldy);
}
inline da_status da_switch_order_in_place(da_order order_X_in, da_int n_rows,
                                          da_int n_cols, float *X, da_int ldx_in,
                                          da_int ldx_out) {
    return da_switch_order_in_place_s(order_X_in, n_rows, n_cols, X, ldx_in, ldx_out);
}
inline da_status da_switch_order_in_place(da_order order_X_in, da_int n_rows,
                                          da_int n_cols, double *X, da_int ldx_in,
                                          da_int ldx_out) {
    return da_switch_order_in_place_d(order_X_in, n_rows, n_cols, X, ldx_in, ldx_out);
}

inline da_status da_get_shuffled_indices(da_int m, da_int seed, da_int train_size,
                                         da_int test_size, da_int fp_precision,
                                         const da_int *classes, da_int *shuffle_array) {
    return da_get_shuffled_indices_int(m, seed, train_size, test_size, fp_precision,
                                       classes, shuffle_array);
};
inline da_status da_get_shuffled_indices(da_int m, da_int seed, da_int train_size,
                                         da_int test_size, da_int fp_precision,
                                         const float *classes, da_int *shuffle_array) {
    return da_get_shuffled_indices_s(m, seed, train_size, test_size, fp_precision,
                                     classes, shuffle_array);
};
inline da_status da_get_shuffled_indices(da_int m, da_int seed, da_int train_size,
                                         da_int test_size, da_int fp_precision,
                                         const double *classes, da_int *shuffle_array) {
    return da_get_shuffled_indices_d(m, seed, train_size, test_size, fp_precision,
                                     classes, shuffle_array);
};

inline da_status da_train_test_split(da_order order, da_int m, da_int n, const da_int *X,
                                     da_int ldx, da_int train_size, da_int test_size,
                                     const da_int *shuffle_array, da_int *X_train,
                                     da_int ldx_train, da_int *X_test, da_int ldx_test) {
    return da_train_test_split_int(order, m, n, X, ldx, train_size, test_size,
                                   shuffle_array, X_train, ldx_train, X_test, ldx_test);
}
inline da_status da_train_test_split(da_order order, da_int m, da_int n, const double *X,
                                     da_int ldx, da_int train_size, da_int test_size,
                                     const da_int *shuffle_array, double *X_train,
                                     da_int ldx_train, double *X_test, da_int ldx_test) {
    return da_train_test_split_d(order, m, n, X, ldx, train_size, test_size,
                                 shuffle_array, X_train, ldx_train, X_test, ldx_test);
}
inline da_status da_train_test_split(da_order order, da_int m, da_int n, const float *X,
                                     da_int ldx, da_int train_size, da_int test_size,
                                     const da_int *shuffle_array, float *X_train,
                                     da_int ldx_train, float *X_test, da_int ldx_test) {
    return da_train_test_split_s(order, m, n, X, ldx, train_size, test_size,
                                 shuffle_array, X_train, ldx_train, X_test, ldx_test);
}

/* Kernel functions */
inline da_status da_rbf_kernel(da_order order, da_int m, da_int n, da_int k,
                               const double *X, da_int ldx, const double *Y, da_int ldy,
                               double *D, da_int ldd, double gamma) {
    return da_rbf_kernel_d(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma);
}

inline da_status da_rbf_kernel(da_order order, da_int m, da_int n, da_int k,
                               const float *X, da_int ldx, const float *Y, da_int ldy,
                               float *D, da_int ldd, float gamma) {
    return da_rbf_kernel_s(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma);
}
inline da_status da_linear_kernel(da_order order, da_int m, da_int n, da_int k,
                                  const double *X, da_int ldx, const double *Y,
                                  da_int ldy, double *D, da_int ldd) {
    return da_linear_kernel_d(order, m, n, k, X, ldx, Y, ldy, D, ldd);
}

inline da_status da_linear_kernel(da_order order, da_int m, da_int n, da_int k,
                                  const float *X, da_int ldx, const float *Y, da_int ldy,
                                  float *D, da_int ldd) {
    return da_linear_kernel_s(order, m, n, k, X, ldx, Y, ldy, D, ldd);
}
inline da_status da_polynomial_kernel(da_order order, da_int m, da_int n, da_int k,
                                      const double *X, da_int ldx, const double *Y,
                                      da_int ldy, double *D, da_int ldd, double gamma,
                                      da_int degree, double coef0) {
    return da_polynomial_kernel_d(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, degree,
                                  coef0);
}
inline da_status da_polynomial_kernel(da_order order, da_int m, da_int n, da_int k,
                                      const float *X, da_int ldx, const float *Y,
                                      da_int ldy, float *D, da_int ldd, float gamma,
                                      da_int degree, float coef0) {
    return da_polynomial_kernel_s(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, degree,
                                  coef0);
}
inline da_status da_sigmoid_kernel(da_order order, da_int m, da_int n, da_int k,
                                   const double *X, da_int ldx, const double *Y,
                                   da_int ldy, double *D, da_int ldd, double gamma,
                                   double coef0) {
    return da_sigmoid_kernel_d(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, coef0);
}
inline da_status da_sigmoid_kernel(da_order order, da_int m, da_int n, da_int k,
                                   const float *X, da_int ldx, const float *Y, da_int ldy,
                                   float *D, da_int ldd, float gamma, float coef0) {
    return da_sigmoid_kernel_s(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, coef0);
}

/* SVM functions */
template <class T> da_status da_svm_select_model(da_handle handle, da_svm_model mod);

inline da_status da_svm_set_data(da_handle handle, da_int n_samples, da_int n_features,
                                 const double *X, da_int ldx, const double *y) {
    return da_svm_set_data_d(handle, n_samples, n_features, X, ldx, y);
}

inline da_status da_svm_set_data(da_handle handle, da_int n_samples, da_int n_features,
                                 const float *X, da_int ldx, const float *y) {
    return da_svm_set_data_s(handle, n_samples, n_features, X, ldx, y);
}
template <class T> da_status da_svm_compute(da_handle handle);

inline da_status da_svm_predict(da_handle handle, da_int n_samples, da_int n_features,
                                const double *X_test, da_int ldx_test,
                                double *predictions) {
    return da_svm_predict_d(handle, n_samples, n_features, X_test, ldx_test, predictions);
}

inline da_status da_svm_predict(da_handle handle, da_int n_samples, da_int n_features,
                                const float *X_test, da_int ldx_test,
                                float *predictions) {
    return da_svm_predict_s(handle, n_samples, n_features, X_test, ldx_test, predictions);
}

inline da_status da_svm_decision_function(da_handle handle, da_int n_samples,
                                          da_int n_features, const double *X_test,
                                          da_int ldx_test,
                                          da_svm_decision_function_shape shape,
                                          double *decision_values, da_int ldd) {
    return da_svm_decision_function_d(handle, n_samples, n_features, X_test, ldx_test,
                                      shape, decision_values, ldd);
}

inline da_status da_svm_decision_function(da_handle handle, da_int n_samples,
                                          da_int n_features, const float *X_test,
                                          da_int ldx_test,
                                          da_svm_decision_function_shape shape,
                                          float *decision_values, da_int ldd) {
    return da_svm_decision_function_s(handle, n_samples, n_features, X_test, ldx_test,
                                      shape, decision_values, ldd);
}

inline da_status da_svm_score(da_handle handle, da_int n_samples, da_int n_features,
                              const double *X_test, da_int ldx_test, const double *y_test,
                              double *score) {
    return da_svm_score_d(handle, n_samples, n_features, X_test, ldx_test, y_test, score);
}

inline da_status da_svm_score(da_handle handle, da_int n_samples, da_int n_features,
                              const float *X_test, da_int ldx_test, const float *y_test,
                              float *score) {
    return da_svm_score_s(handle, n_samples, n_features, X_test, ldx_test, y_test, score);
}

inline da_status da_svm_predict_proba(da_handle handle, da_int n_samples,
                                      da_int n_features, const double *X_test,
                                      da_int ldx_test, double *y_proba, da_int ldy) {
    return da_svm_predict_proba_d(handle, n_samples, n_features, X_test, ldx_test,
                                  y_proba, ldy);
}

inline da_status da_svm_predict_proba(da_handle handle, da_int n_samples,
                                      da_int n_features, const float *X_test,
                                      da_int ldx_test, float *y_proba, da_int ldy) {
    return da_svm_predict_proba_s(handle, n_samples, n_features, X_test, ldx_test,
                                  y_proba, ldy);
}

inline da_status da_svm_predict_log_proba(da_handle handle, da_int n_samples,
                                          da_int n_features, const double *X_test,
                                          da_int ldx_test, double *y_log_proba,
                                          da_int ldy) {
    return da_svm_predict_log_proba_d(handle, n_samples, n_features, X_test, ldx_test,
                                      y_log_proba, ldy);
}

inline da_status da_svm_predict_log_proba(da_handle handle, da_int n_samples,
                                          da_int n_features, const float *X_test,
                                          da_int ldx_test, float *y_log_proba,
                                          da_int ldy) {
    return da_svm_predict_log_proba_s(handle, n_samples, n_features, X_test, ldx_test,
                                      y_log_proba, ldy);
}

/* Interpolation overloaded functions */
template <typename T>
da_status da_interpolation_select_model(da_handle handle, da_interpolation_model model);

inline da_status da_interpolation_set_sites(da_handle handle, da_int n_sites,
                                            const double *x) {
    return da_interpolation_set_sites_d(handle, n_sites, x);
}

inline da_status da_interpolation_set_sites(da_handle handle, da_int n_sites,
                                            const float *x) {
    return da_interpolation_set_sites_s(handle, n_sites, x);
}

inline da_status da_interpolation_set_sites_uniform(da_handle handle, da_int n_sites,
                                                    double x_start, double x_end) {
    return da_interpolation_set_sites_uniform_d(handle, n_sites, x_start, x_end);
}

inline da_status da_interpolation_set_sites_uniform(da_handle handle, da_int n_sites,
                                                    float x_start, float x_end) {
    return da_interpolation_set_sites_uniform_s(handle, n_sites, x_start, x_end);
}

inline da_status da_interpolation_set_values(da_handle handle, da_int n, da_int dim,
                                             const double *y_data, da_int ldy_data,
                                             da_int order) {
    return da_interpolation_set_values_d(handle, n, dim, y_data, ldy_data, order);
}

inline da_status da_interpolation_set_values(da_handle handle, da_int n, da_int dim,
                                             const float *y_data, da_int ldy_data,
                                             da_int order) {
    return da_interpolation_set_values_s(handle, n, dim, y_data, ldy_data, order);
}

inline da_status da_interpolation_search_cells(da_handle handle, da_int n_eval,
                                               const double *x_eval, da_int *cells) {
    return da_interpolation_search_cells_d(handle, n_eval, x_eval, cells);
}

inline da_status da_interpolation_search_cells(da_handle handle, da_int n_eval,
                                               const float *x_eval, da_int *cells) {
    return da_interpolation_search_cells_s(handle, n_eval, x_eval, cells);
}

template <typename T> da_status da_interpolation_interpolate(da_handle handle);

inline da_status da_interpolation_set_boundary_conditions(da_handle handle, da_int dim,
                                                          da_int left_order,
                                                          const double *left_values,
                                                          da_int right_order,
                                                          const double *right_values) {
    return da_interpolation_set_boundary_conditions_d(
        handle, dim, left_order, left_values, right_order, right_values);
}

inline da_status da_interpolation_set_boundary_conditions(da_handle handle, da_int dim,
                                                          da_int left_order,
                                                          const float *left_values,
                                                          da_int right_order,
                                                          const float *right_values) {
    return da_interpolation_set_boundary_conditions_s(
        handle, dim, left_order, left_values, right_order, right_values);
}

inline da_status da_interpolation_evaluate(da_handle handle, da_int n_eval,
                                           const double *x_eval, double *y_eval,
                                           da_int n_orders, da_int *orders) {
    return da_interpolation_evaluate_d(handle, n_eval, x_eval, y_eval, n_orders, orders);
}

inline da_status da_interpolation_evaluate(da_handle handle, da_int n_eval,
                                           const float *x_eval, float *y_eval,
                                           da_int n_orders, da_int *orders) {
    return da_interpolation_evaluate_s(handle, n_eval, x_eval, y_eval, n_orders, orders);
}
/* Approximate Neighbors functions*/
inline da_status da_approx_nn_set_training_data(da_handle handle, da_int n_samples,
                                                da_int n_features, const double *X_train,
                                                da_int ldx_train) {
    return da_approx_nn_set_training_data_d(handle, n_samples, n_features, X_train,
                                            ldx_train);
}

inline da_status da_approx_nn_set_training_data(da_handle handle, da_int n_samples,
                                                da_int n_features, const float *X_train,
                                                da_int ldx_train) {
    return da_approx_nn_set_training_data_s(handle, n_samples, n_features, X_train,
                                            ldx_train);
}

template <class T> da_status da_approx_nn_train(da_handle handle);

inline da_status da_approx_nn_add(da_handle handle, da_int n_samples_add,
                                  da_int n_features, const double *X_add,
                                  da_int ldx_add) {
    return da_approx_nn_add_d(handle, n_samples_add, n_features, X_add, ldx_add);
}

inline da_status da_approx_nn_add(da_handle handle, da_int n_samples_add,
                                  da_int n_features, const float *X_add, da_int ldx_add) {
    return da_approx_nn_add_s(handle, n_samples_add, n_features, X_add, ldx_add);
}

template <class T> da_status da_approx_nn_train_and_add(da_handle handle);

inline da_status da_approx_nn_kneighbors(da_handle handle, da_int n_queries,
                                         da_int n_features, const double *X_test,
                                         da_int ldx_test, da_int *n_ind, double *n_dist,
                                         da_int k = 0, bool return_distance = 0) {
    return da_approx_nn_kneighbors_d(handle, n_queries, n_features, X_test, ldx_test,
                                     n_ind, n_dist, k, return_distance);
}

inline da_status da_approx_nn_kneighbors(da_handle handle, da_int n_queries,
                                         da_int n_features, const float *X_test,
                                         da_int ldx_test, da_int *n_ind, float *n_dist,
                                         da_int k = 0, bool return_distance = 0) {
    return da_approx_nn_kneighbors_s(handle, n_queries, n_features, X_test, ldx_test,
                                     n_ind, n_dist, k, return_distance);
}

#endif // AOCLDA_CPP_OVERLOADS
