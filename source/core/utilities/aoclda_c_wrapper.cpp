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

/*
 * C API wrappers: thin extern "C" functions that forward to C++ template instantiations.
 * Each _d/_s/_int/_uint8 C function calls the corresponding template<T> with
 * T = double/float/da_int/uint8_t.
 */

#include "aoclda.hpp"

extern "C" {

/* ======================== Handle (aoclda_handle.h) ======================== */

da_status da_handle_init_d(da_handle *handle, da_handle_type handle_type) {
    return da_handle_init<double>(handle, handle_type);
}
da_status da_handle_init_s(da_handle *handle, da_handle_type handle_type) {
    return da_handle_init<float>(handle, handle_type);
}

/* ======================== Result (aoclda_result.h) ======================== */

da_status da_handle_get_result_d(const da_handle handle, da_result query, da_int *dim,
                                 double *result) {
    return da_handle_get_result<double>(handle, query, dim, result);
}
da_status da_handle_get_result_s(const da_handle handle, da_result query, da_int *dim,
                                 float *result) {
    return da_handle_get_result<float>(handle, query, dim, result);
}
da_status da_handle_get_result_int(const da_handle handle, da_result query, da_int *dim,
                                   da_int *result) {
    return da_handle_get_result<da_int>(handle, query, dim, result);
}

/* ======================== Options (aoclda_options.h) ======================== */

da_status da_options_set_real_d(da_handle handle, const char *option, double value) {
    return da_options_set<double>(handle, option, value);
}
da_status da_options_set_real_s(da_handle handle, const char *option, float value) {
    return da_options_set<float>(handle, option, value);
}
da_status da_options_set_int(da_handle handle, const char *option, da_int value) {
    return da_options_set<da_int>(handle, option, value);
}
da_status da_options_set_string(da_handle handle, const char *option, const char *value) {
    return da_options_set<const char *>(handle, option, value);
}

da_status da_options_get_real_d(da_handle handle, const char *option, double *value) {
    return da_options_get<double>(handle, option, value);
}
da_status da_options_get_real_s(da_handle handle, const char *option, float *value) {
    return da_options_get<float>(handle, option, value);
}
da_status da_options_get_int(da_handle handle, const char *option, da_int *value) {
    return da_options_get<da_int>(handle, option, value);
}
da_status da_options_get_string(da_handle handle, const char *option, char *value,
                                da_int *lvalue) {
    return da_options_get(handle, option, value, lvalue);
}
da_status da_options_get_string_key(da_handle handle, const char *option, char *value,
                                    da_int *lvalue, da_int *key) {
    return da_options_get(handle, option, value, lvalue, key);
}

/* Datastore options */
da_status da_datastore_options_set_real_d(da_datastore store, const char *option,
                                          double value) {
    return da_datastore_options_set<double>(store, option, value);
}
da_status da_datastore_options_set_real_s(da_datastore store, const char *option,
                                          float value) {
    return da_datastore_options_set<float>(store, option, value);
}
da_status da_datastore_options_set_int(da_datastore store, const char *option,
                                       da_int value) {
    return da_datastore_options_set<da_int>(store, option, value);
}
da_status da_datastore_options_set_string(da_datastore store, const char *option,
                                          const char *value) {
    return da_datastore_options_set<const char *>(store, option, value);
}

da_status da_datastore_options_get_real_d(da_datastore store, const char *option,
                                          double *value) {
    return da_datastore_options_get<double>(store, option, value);
}
da_status da_datastore_options_get_real_s(da_datastore store, const char *option,
                                          float *value) {
    return da_datastore_options_get<float>(store, option, value);
}
da_status da_datastore_options_get_int(da_datastore store, const char *option,
                                       da_int *value) {
    return da_datastore_options_get<da_int>(store, option, value);
}
da_status da_datastore_options_get_string(da_datastore store, const char *option,
                                          char *value, da_int lvalue) {
    return da_datastore_options_get(store, option, value, &lvalue);
}

/* ======================== CSV (aoclda_csv.h) ======================== */

da_status da_read_csv_d(da_datastore store, const char *filename, double **A,
                        da_int *n_rows, da_int *n_cols, char ***headings) {
    return da_read_csv<double>(store, filename, A, n_rows, n_cols, headings);
}
da_status da_read_csv_s(da_datastore store, const char *filename, float **A,
                        da_int *n_rows, da_int *n_cols, char ***headings) {
    return da_read_csv<float>(store, filename, A, n_rows, n_cols, headings);
}
da_status da_read_csv_int(da_datastore store, const char *filename, da_int **A,
                          da_int *n_rows, da_int *n_cols, char ***headings) {
    return da_read_csv<da_int>(store, filename, A, n_rows, n_cols, headings);
}
da_status da_read_csv_uint8(da_datastore store, const char *filename, uint8_t **A,
                            da_int *n_rows, da_int *n_cols, char ***headings) {
    return da_read_csv<uint8_t>(store, filename, A, n_rows, n_cols, headings);
}
da_status da_read_csv_string(da_datastore store, const char *filename, char ***A,
                             da_int *n_rows, da_int *n_cols, char ***headings) {
    return da_read_csv(store, filename, A, n_rows, n_cols, headings);
}

/* ======================== Datastore (aoclda_datastore.h) ======================== */

/* da_data_load_col */
da_status da_data_load_col_real_d(da_datastore store, da_int n_rows, da_int n_cols,
                                  double *block, da_order order, da_int copy_data) {
    return da_data_load_col<double>(store, n_rows, n_cols, block, order, copy_data);
}
da_status da_data_load_col_real_s(da_datastore store, da_int n_rows, da_int n_cols,
                                  float *block, da_order order, da_int copy_data) {
    return da_data_load_col<float>(store, n_rows, n_cols, block, order, copy_data);
}
da_status da_data_load_col_int(da_datastore store, da_int n_rows, da_int n_cols,
                               da_int *block, da_order order, da_int copy_data) {
    return da_data_load_col<da_int>(store, n_rows, n_cols, block, order, copy_data);
}
da_status da_data_load_col_uint8(da_datastore store, da_int n_rows, da_int n_cols,
                                 uint8_t *block, da_order order, da_int copy_data) {
    return da_data_load_col<uint8_t>(store, n_rows, n_cols, block, order, copy_data);
}
da_status da_data_load_col_str(da_datastore store, da_int n_rows, da_int n_cols,
                               const char **block, da_order order) {
    return da_data_load_col(store, n_rows, n_cols, block, order);
}

/* da_data_load_row */
da_status da_data_load_row_real_d(da_datastore store, da_int n_rows, da_int n_cols,
                                  double *block, da_order order, da_int copy_data) {
    return da_data_load_row<double>(store, n_rows, n_cols, block, order, copy_data);
}
da_status da_data_load_row_real_s(da_datastore store, da_int n_rows, da_int n_cols,
                                  float *block, da_order order, da_int copy_data) {
    return da_data_load_row<float>(store, n_rows, n_cols, block, order, copy_data);
}
da_status da_data_load_row_int(da_datastore store, da_int n_rows, da_int n_cols,
                               da_int *block, da_order order, da_int copy_data) {
    return da_data_load_row<da_int>(store, n_rows, n_cols, block, order, copy_data);
}
da_status da_data_load_row_uint8(da_datastore store, da_int n_rows, da_int n_cols,
                                 uint8_t *block, da_order order, da_int copy_data) {
    return da_data_load_row<uint8_t>(store, n_rows, n_cols, block, order, copy_data);
}
da_status da_data_load_row_str(da_datastore store, da_int n_rows, da_int n_cols,
                               const char **block, da_order order) {
    return da_data_load_row(store, n_rows, n_cols, block, order);
}

/* da_data_extract_column */
da_status da_data_extract_column_real_d(da_datastore store, da_int idx, da_int dim,
                                        double *col) {
    return da_data_extract_column<double>(store, idx, dim, col);
}
da_status da_data_extract_column_real_s(da_datastore store, da_int idx, da_int dim,
                                        float *col) {
    return da_data_extract_column<float>(store, idx, dim, col);
}
da_status da_data_extract_column_int(da_datastore store, da_int idx, da_int dim,
                                     da_int *col) {
    return da_data_extract_column<da_int>(store, idx, dim, col);
}
da_status da_data_extract_column_uint8(da_datastore store, da_int idx, da_int dim,
                                       uint8_t *col) {
    return da_data_extract_column<uint8_t>(store, idx, dim, col);
}
da_status da_data_extract_column_str(da_datastore store, da_int idx, da_int dim,
                                     char **col) {
    return da_data_extract_column(store, idx, dim, col);
}

/* da_data_extract_selection */
da_status da_data_extract_selection_real_d(da_datastore store, const char *key,
                                           da_order order, double *data, da_int lddata) {
    return da_data_extract_selection<double>(store, key, order, data, lddata);
}
da_status da_data_extract_selection_real_s(da_datastore store, const char *key,
                                           da_order order, float *data, da_int lddata) {
    return da_data_extract_selection<float>(store, key, order, data, lddata);
}
da_status da_data_extract_selection_int(da_datastore store, const char *key,
                                        da_order order, da_int *data, da_int lddata) {
    return da_data_extract_selection<da_int>(store, key, order, data, lddata);
}
da_status da_data_extract_selection_uint8(da_datastore store, const char *key,
                                          da_order order, uint8_t *data, da_int lddata) {
    return da_data_extract_selection<uint8_t>(store, key, order, data, lddata);
}

/* da_data_get_element */
da_status da_data_get_element_real_d(da_datastore store, da_int i, da_int j,
                                     double *elem) {
    return da_data_get_element<double>(store, i, j, elem);
}
da_status da_data_get_element_real_s(da_datastore store, da_int i, da_int j,
                                     float *elem) {
    return da_data_get_element<float>(store, i, j, elem);
}
da_status da_data_get_element_int(da_datastore store, da_int i, da_int j, da_int *elem) {
    return da_data_get_element<da_int>(store, i, j, elem);
}
da_status da_data_get_element_uint8(da_datastore store, da_int i, da_int j,
                                    uint8_t *elem) {
    return da_data_get_element<uint8_t>(store, i, j, elem);
}

/* da_data_set_element */
da_status da_data_set_element_real_d(da_datastore store, da_int i, da_int j,
                                     double elem) {
    return da_data_set_element<double>(store, i, j, elem);
}
da_status da_data_set_element_real_s(da_datastore store, da_int i, da_int j, float elem) {
    return da_data_set_element<float>(store, i, j, elem);
}
da_status da_data_set_element_int(da_datastore store, da_int i, da_int j, da_int elem) {
    return da_data_set_element<da_int>(store, i, j, elem);
}
da_status da_data_set_element_uint8(da_datastore store, da_int i, da_int j,
                                    uint8_t elem) {
    return da_data_set_element<uint8_t>(store, i, j, elem);
}

/* ======================== Basic Statistics (aoclda_basic_statistics.h) ======================== */

da_status da_mean_d(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                    const double *X, da_int ldx, double *mean) {
    return da_mean<double>(order, axis, n_rows, n_cols, X, ldx, mean);
}
da_status da_mean_s(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                    const float *X, da_int ldx, float *mean) {
    return da_mean<float>(order, axis, n_rows, n_cols, X, ldx, mean);
}

da_status da_harmonic_mean_d(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                             const double *X, da_int ldx, double *harmonic_mean) {
    return da_harmonic_mean<double>(order, axis, n_rows, n_cols, X, ldx, harmonic_mean);
}
da_status da_harmonic_mean_s(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                             const float *X, da_int ldx, float *harmonic_mean) {
    return da_harmonic_mean<float>(order, axis, n_rows, n_cols, X, ldx, harmonic_mean);
}

da_status da_geometric_mean_d(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                              const double *X, da_int ldx, double *geometric_mean) {
    return da_geometric_mean<double>(order, axis, n_rows, n_cols, X, ldx, geometric_mean);
}
da_status da_geometric_mean_s(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                              const float *X, da_int ldx, float *geometric_mean) {
    return da_geometric_mean<float>(order, axis, n_rows, n_cols, X, ldx, geometric_mean);
}

da_status da_variance_d(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                        const double *X, da_int ldx, da_int dof, double *mean,
                        double *variance) {
    return da_variance<double>(order, axis, n_rows, n_cols, X, ldx, dof, mean, variance);
}
da_status da_variance_s(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                        const float *X, da_int ldx, da_int dof, float *mean,
                        float *variance) {
    return da_variance<float>(order, axis, n_rows, n_cols, X, ldx, dof, mean, variance);
}

da_status da_skewness_d(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                        const double *X, da_int ldx, double *mean, double *variance,
                        double *skewness) {
    return da_skewness<double>(order, axis, n_rows, n_cols, X, ldx, mean, variance,
                               skewness);
}
da_status da_skewness_s(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                        const float *X, da_int ldx, float *mean, float *variance,
                        float *skewness) {
    return da_skewness<float>(order, axis, n_rows, n_cols, X, ldx, mean, variance,
                              skewness);
}

da_status da_kurtosis_d(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                        const double *X, da_int ldx, double *mean, double *variance,
                        double *kurtosis) {
    return da_kurtosis<double>(order, axis, n_rows, n_cols, X, ldx, mean, variance,
                               kurtosis);
}
da_status da_kurtosis_s(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                        const float *X, da_int ldx, float *mean, float *variance,
                        float *kurtosis) {
    return da_kurtosis<float>(order, axis, n_rows, n_cols, X, ldx, mean, variance,
                              kurtosis);
}

da_status da_moment_d(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                      const double *X, da_int ldx, da_int k, da_int use_precomputed_mean,
                      double *mean, double *moment) {
    return da_moment<double>(order, axis, n_rows, n_cols, X, ldx, k, use_precomputed_mean,
                             mean, moment);
}
da_status da_moment_s(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                      const float *X, da_int ldx, da_int k, da_int use_precomputed_mean,
                      float *mean, float *moment) {
    return da_moment<float>(order, axis, n_rows, n_cols, X, ldx, k, use_precomputed_mean,
                            mean, moment);
}

da_status da_quantile_d(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                        const double *X, da_int ldx, double q, double *quantile,
                        da_quantile_type quantile_type) {
    return da_quantile<double>(order, axis, n_rows, n_cols, X, ldx, q, quantile,
                               quantile_type);
}
da_status da_quantile_s(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                        const float *X, da_int ldx, float q, float *quantile,
                        da_quantile_type quantile_type) {
    return da_quantile<float>(order, axis, n_rows, n_cols, X, ldx, q, quantile,
                              quantile_type);
}

da_status da_five_point_summary_d(da_order order, da_axis axis, da_int n_rows,
                                  da_int n_cols, const double *X, da_int ldx,
                                  double *minimum, double *lower_hinge, double *median,
                                  double *upper_hinge, double *maximum) {
    return da_five_point_summary<double>(order, axis, n_rows, n_cols, X, ldx, minimum,
                                         lower_hinge, median, upper_hinge, maximum);
}
da_status da_five_point_summary_s(da_order order, da_axis axis, da_int n_rows,
                                  da_int n_cols, const float *X, da_int ldx,
                                  float *minimum, float *lower_hinge, float *median,
                                  float *upper_hinge, float *maximum) {
    return da_five_point_summary<float>(order, axis, n_rows, n_cols, X, ldx, minimum,
                                        lower_hinge, median, upper_hinge, maximum);
}

da_status da_standardize_d(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                           double *X, da_int ldx, da_int dof, da_int mode, double *shift,
                           double *scale) {
    return da_standardize<double>(order, axis, n_rows, n_cols, X, ldx, dof, mode, shift,
                                  scale);
}
da_status da_standardize_s(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                           float *X, da_int ldx, da_int dof, da_int mode, float *shift,
                           float *scale) {
    return da_standardize<float>(order, axis, n_rows, n_cols, X, ldx, dof, mode, shift,
                                 scale);
}

da_status da_covariance_matrix_d(da_order order, da_int n_rows, da_int n_cols,
                                 const double *X, da_int ldx, da_int dof, double *cov,
                                 da_int ldcov, da_int assume_centered) {
    return da_covariance_matrix<double>(order, n_rows, n_cols, X, ldx, dof, cov, ldcov,
                                        assume_centered);
}
da_status da_covariance_matrix_s(da_order order, da_int n_rows, da_int n_cols,
                                 const float *X, da_int ldx, da_int dof, float *cov,
                                 da_int ldcov, da_int assume_centered) {
    return da_covariance_matrix<float>(order, n_rows, n_cols, X, ldx, dof, cov, ldcov,
                                       assume_centered);
}

da_status da_correlation_matrix_d(da_order order, da_int n_rows, da_int n_cols,
                                  const double *X, da_int ldx, double *corr,
                                  da_int ldcorr) {
    return da_correlation_matrix<double>(order, n_rows, n_cols, X, ldx, corr, ldcorr);
}
da_status da_correlation_matrix_s(da_order order, da_int n_rows, da_int n_cols,
                                  const float *X, da_int ldx, float *corr,
                                  da_int ldcorr) {
    return da_correlation_matrix<float>(order, n_rows, n_cols, X, ldx, corr, ldcorr);
}

/* ======================== Linear Model (aoclda_linmod.h) ======================== */

da_status da_linmod_select_model_d(da_handle handle, linmod_model mod) {
    return da_linmod_select_model<double>(handle, mod);
}
da_status da_linmod_select_model_s(da_handle handle, linmod_model mod) {
    return da_linmod_select_model<float>(handle, mod);
}

da_status da_linmod_define_features_d(da_handle handle, da_int n_samples,
                                      da_int n_features, const double *X, da_int ldx,
                                      const double *y) {
    return da_linmod_define_features<double>(handle, n_samples, n_features, X, ldx, y);
}
da_status da_linmod_define_features_s(da_handle handle, da_int n_samples,
                                      da_int n_features, const float *X, da_int ldx,
                                      const float *y) {
    return da_linmod_define_features<float>(handle, n_samples, n_features, X, ldx, y);
}

da_status da_linmod_fit_d(da_handle handle) { return da_linmod_fit<double>(handle); }
da_status da_linmod_fit_s(da_handle handle) { return da_linmod_fit<float>(handle); }

da_status da_linmod_fit_start_d(da_handle handle, da_int n_coefs, const double *coefs) {
    return da_linmod_fit_start<double>(handle, n_coefs, coefs);
}
da_status da_linmod_fit_start_s(da_handle handle, da_int n_coefs, const float *coefs) {
    return da_linmod_fit_start<float>(handle, n_coefs, coefs);
}

da_status da_linmod_evaluate_model_d(da_handle handle, da_int n_samples,
                                     da_int n_features, const double *X, da_int ldx,
                                     double *predictions, const double *observations,
                                     double *loss) {
    return da_linmod_evaluate_model<double>(handle, n_samples, n_features, X, ldx,
                                            predictions, observations, loss);
}
da_status da_linmod_evaluate_model_s(da_handle handle, da_int n_samples,
                                     da_int n_features, const float *X, da_int ldx,
                                     float *predictions, const float *observations,
                                     float *loss) {
    return da_linmod_evaluate_model<float>(handle, n_samples, n_features, X, ldx,
                                           predictions, observations, loss);
}

/* ======================== PCA (aoclda_pca.h) ======================== */

da_status da_pca_set_data_d(da_handle handle, da_int n_samples, da_int n_features,
                            const double *A, da_int lda) {
    return da_pca_set_data<double>(handle, n_samples, n_features, A, lda);
}
da_status da_pca_set_data_s(da_handle handle, da_int n_samples, da_int n_features,
                            const float *A, da_int lda) {
    return da_pca_set_data<float>(handle, n_samples, n_features, A, lda);
}

da_status da_pca_compute_d(da_handle handle) { return da_pca_compute<double>(handle); }
da_status da_pca_compute_s(da_handle handle) { return da_pca_compute<float>(handle); }

da_status da_pca_transform_d(da_handle handle, da_int m_samples, da_int m_features,
                             const double *X, da_int ldx, double *X_transform,
                             da_int ldx_transform) {
    return da_pca_transform<double>(handle, m_samples, m_features, X, ldx, X_transform,
                                    ldx_transform);
}
da_status da_pca_transform_s(da_handle handle, da_int m_samples, da_int m_features,
                             const float *X, da_int ldx, float *X_transform,
                             da_int ldx_transform) {
    return da_pca_transform<float>(handle, m_samples, m_features, X, ldx, X_transform,
                                   ldx_transform);
}

da_status da_pca_inverse_transform_d(da_handle handle, da_int k_samples,
                                     da_int k_features, const double *Y, da_int ldy,
                                     double *Y_inv_transform, da_int ldy_inv_transform) {
    return da_pca_inverse_transform<double>(handle, k_samples, k_features, Y, ldy,
                                            Y_inv_transform, ldy_inv_transform);
}
da_status da_pca_inverse_transform_s(da_handle handle, da_int k_samples,
                                     da_int k_features, const float *Y, da_int ldy,
                                     float *Y_inv_transform, da_int ldy_inv_transform) {
    return da_pca_inverse_transform<float>(handle, k_samples, k_features, Y, ldy,
                                           Y_inv_transform, ldy_inv_transform);
}

/* ======================== Kernel PCA (aoclda_pca.h) ======================== */

da_status da_kernel_pca_set_data_d(da_handle handle, da_int n_samples, da_int n_features,
                                   const double *A, da_int lda) {
    return da_kernel_pca_set_data<double>(handle, n_samples, n_features, A, lda);
}
da_status da_kernel_pca_set_data_s(da_handle handle, da_int n_samples, da_int n_features,
                                   const float *A, da_int lda) {
    return da_kernel_pca_set_data<float>(handle, n_samples, n_features, A, lda);
}

da_status da_kernel_pca_compute_d(da_handle handle) {
    return da_kernel_pca_compute<double>(handle);
}
da_status da_kernel_pca_compute_s(da_handle handle) {
    return da_kernel_pca_compute<float>(handle);
}

da_status da_kernel_pca_transform_d(da_handle handle, da_int m_samples, da_int m_features,
                                    const double *X, da_int ldx, double *X_transform,
                                    da_int ldx_transform) {
    return da_kernel_pca_transform<double>(handle, m_samples, m_features, X, ldx,
                                           X_transform, ldx_transform);
}
da_status da_kernel_pca_transform_s(da_handle handle, da_int m_samples, da_int m_features,
                                    const float *X, da_int ldx, float *X_transform,
                                    da_int ldx_transform) {
    return da_kernel_pca_transform<float>(handle, m_samples, m_features, X, ldx,
                                          X_transform, ldx_transform);
}

da_status da_kernel_pca_inverse_transform_d(da_handle handle, da_int k_samples,
                                            da_int k_components, const double *Y,
                                            da_int ldy, double *Y_inv_transform,
                                            da_int ldy_inv_transform) {
    return da_kernel_pca_inverse_transform<double>(
        handle, k_samples, k_components, Y, ldy, Y_inv_transform, ldy_inv_transform);
}
da_status da_kernel_pca_inverse_transform_s(da_handle handle, da_int k_samples,
                                            da_int k_components, const float *Y,
                                            da_int ldy, float *Y_inv_transform,
                                            da_int ldy_inv_transform) {
    return da_kernel_pca_inverse_transform<float>(handle, k_samples, k_components, Y, ldy,
                                                  Y_inv_transform, ldy_inv_transform);
}

/* ======================== t-SNE (aoclda_tsne.h) ======================== */

da_status da_tsne_set_data_d(da_handle handle, da_int n_samples, da_int n_features,
                             const double *X, da_int ldx) {
    return da_tsne_set_data<double>(handle, n_samples, n_features, X, ldx);
}
da_status da_tsne_set_data_s(da_handle handle, da_int n_samples, da_int n_features,
                             const float *X, da_int ldx) {
    return da_tsne_set_data<float>(handle, n_samples, n_features, X, ldx);
}

da_status da_tsne_set_init_embedding_d(da_handle handle, const double *Y, da_int ldy) {
    return da_tsne_set_init_embedding<double>(handle, Y, ldy);
}
da_status da_tsne_set_init_embedding_s(da_handle handle, const float *Y, da_int ldy) {
    return da_tsne_set_init_embedding<float>(handle, Y, ldy);
}

da_status da_tsne_compute_d(da_handle handle) { return da_tsne_compute<double>(handle); }
da_status da_tsne_compute_s(da_handle handle) { return da_tsne_compute<float>(handle); }

/* ======================== k-means (aoclda_kmeans.h) ======================== */

da_status da_kmeans_set_data_d(da_handle handle, da_int n_samples, da_int n_features,
                               const double *A, da_int lda) {
    return da_kmeans_set_data<double>(handle, n_samples, n_features, A, lda);
}
da_status da_kmeans_set_data_s(da_handle handle, da_int n_samples, da_int n_features,
                               const float *A, da_int lda) {
    return da_kmeans_set_data<float>(handle, n_samples, n_features, A, lda);
}

da_status da_kmeans_set_init_centres_d(da_handle handle, const double *C, da_int ldc) {
    return da_kmeans_set_init_centres<double>(handle, C, ldc);
}
da_status da_kmeans_set_init_centres_s(da_handle handle, const float *C, da_int ldc) {
    return da_kmeans_set_init_centres<float>(handle, C, ldc);
}

da_status da_kmeans_compute_d(da_handle handle) {
    return da_kmeans_compute<double>(handle);
}
da_status da_kmeans_compute_s(da_handle handle) {
    return da_kmeans_compute<float>(handle);
}

da_status da_kmeans_transform_d(da_handle handle, da_int m_samples, da_int m_features,
                                const double *X, da_int ldx, double *X_transform,
                                da_int ldx_transform) {
    return da_kmeans_transform<double>(handle, m_samples, m_features, X, ldx, X_transform,
                                       ldx_transform);
}
da_status da_kmeans_transform_s(da_handle handle, da_int m_samples, da_int m_features,
                                const float *X, da_int ldx, float *X_transform,
                                da_int ldx_transform) {
    return da_kmeans_transform<float>(handle, m_samples, m_features, X, ldx, X_transform,
                                      ldx_transform);
}

da_status da_kmeans_predict_d(da_handle handle, da_int k_samples, da_int k_features,
                              const double *Y, da_int ldy, da_int *Y_labels) {
    return da_kmeans_predict<double>(handle, k_samples, k_features, Y, ldy, Y_labels);
}
da_status da_kmeans_predict_s(da_handle handle, da_int k_samples, da_int k_features,
                              const float *Y, da_int ldy, da_int *Y_labels) {
    return da_kmeans_predict<float>(handle, k_samples, k_features, Y, ldy, Y_labels);
}

/* ======================== DBSCAN (aoclda_dbscan.h) ======================== */

da_status da_dbscan_set_data_d(da_handle handle, da_int n_samples, da_int n_features,
                               const double *A, da_int lda) {
    return da_dbscan_set_data<double>(handle, n_samples, n_features, A, lda);
}
da_status da_dbscan_set_data_s(da_handle handle, da_int n_samples, da_int n_features,
                               const float *A, da_int lda) {
    return da_dbscan_set_data<float>(handle, n_samples, n_features, A, lda);
}

da_status da_dbscan_compute_d(da_handle handle) {
    return da_dbscan_compute<double>(handle);
}
da_status da_dbscan_compute_s(da_handle handle) {
    return da_dbscan_compute<float>(handle);
}

/* ======================== Decision Tree (aoclda_decision_forest.h) ======================== */

da_status da_tree_set_training_data_d(da_handle handle, da_int n_samples,
                                      da_int n_features, da_int n_class, const double *X,
                                      da_int ldx, const da_int *y,
                                      const da_int *categorical_features) {
    return da_tree_set_training_data<double>(handle, n_samples, n_features, n_class, X,
                                             ldx, y, categorical_features);
}
da_status da_tree_set_training_data_s(da_handle handle, da_int n_samples,
                                      da_int n_features, da_int n_class, const float *X,
                                      da_int ldx, const da_int *y,
                                      const da_int *categorical_features) {
    return da_tree_set_training_data<float>(handle, n_samples, n_features, n_class, X,
                                            ldx, y, categorical_features);
}

da_status da_tree_fit_d(da_handle handle) { return da_tree_fit<double>(handle); }
da_status da_tree_fit_s(da_handle handle) { return da_tree_fit<float>(handle); }

da_status da_tree_predict_d(da_handle handle, da_int n_samples, da_int n_features,
                            const double *X_test, da_int ldx_test, da_int *y_pred) {
    return da_tree_predict<double>(handle, n_samples, n_features, X_test, ldx_test,
                                   y_pred);
}
da_status da_tree_predict_s(da_handle handle, da_int n_samples, da_int n_features,
                            const float *X_test, da_int ldx_test, da_int *y_pred) {
    return da_tree_predict<float>(handle, n_samples, n_features, X_test, ldx_test,
                                  y_pred);
}

da_status da_tree_predict_proba_d(da_handle handle, da_int n_samples, da_int n_features,
                                  const double *X_test, da_int ldx_test, double *y_proba,
                                  da_int n_class, da_int ldy) {
    return da_tree_predict_proba<double>(handle, n_samples, n_features, X_test, ldx_test,
                                         y_proba, n_class, ldy);
}
da_status da_tree_predict_proba_s(da_handle handle, da_int n_samples, da_int n_features,
                                  const float *X_test, da_int ldx_test, float *y_proba,
                                  da_int n_class, da_int ldy) {
    return da_tree_predict_proba<float>(handle, n_samples, n_features, X_test, ldx_test,
                                        y_proba, n_class, ldy);
}

da_status da_tree_predict_log_proba_d(da_handle handle, da_int n_samples,
                                      da_int n_features, const double *X_test,
                                      da_int ldx_test, double *y_log_proba,
                                      da_int n_class, da_int ldy) {
    return da_tree_predict_log_proba<double>(handle, n_samples, n_features, X_test,
                                             ldx_test, y_log_proba, n_class, ldy);
}
da_status da_tree_predict_log_proba_s(da_handle handle, da_int n_samples,
                                      da_int n_features, const float *X_test,
                                      da_int ldx_test, float *y_log_proba, da_int n_class,
                                      da_int ldy) {
    return da_tree_predict_log_proba<float>(handle, n_samples, n_features, X_test,
                                            ldx_test, y_log_proba, n_class, ldy);
}

da_status da_tree_score_d(da_handle handle, da_int n_samples, da_int n_features,
                          const double *X_test, da_int ldx_test, const da_int *y_test,
                          double *mean_accuracy) {
    return da_tree_score<double>(handle, n_samples, n_features, X_test, ldx_test, y_test,
                                 mean_accuracy);
}
da_status da_tree_score_s(da_handle handle, da_int n_samples, da_int n_features,
                          const float *X_test, da_int ldx_test, const da_int *y_test,
                          float *mean_accuracy) {
    return da_tree_score<float>(handle, n_samples, n_features, X_test, ldx_test, y_test,
                                mean_accuracy);
}

/* ======================== Decision Forest (aoclda_decision_forest.h) ======================== */

da_status da_forest_set_training_data_d(da_handle handle, da_int n_samples,
                                        da_int n_features, da_int n_class,
                                        const double *X, da_int ldx, const da_int *y,
                                        const da_int *categorical_features) {
    return da_forest_set_training_data<double>(handle, n_samples, n_features, n_class, X,
                                               ldx, y, categorical_features);
}
da_status da_forest_set_training_data_s(da_handle handle, da_int n_samples,
                                        da_int n_features, da_int n_class, const float *X,
                                        da_int ldx, const da_int *y,
                                        const da_int *categorical_features) {
    return da_forest_set_training_data<float>(handle, n_samples, n_features, n_class, X,
                                              ldx, y, categorical_features);
}

da_status da_forest_fit_d(da_handle handle) { return da_forest_fit<double>(handle); }
da_status da_forest_fit_s(da_handle handle) { return da_forest_fit<float>(handle); }

da_status da_forest_predict_d(da_handle handle, da_int n_samples, da_int n_features,
                              const double *X_test, da_int ldx_test, da_int *y_pred) {
    return da_forest_predict<double>(handle, n_samples, n_features, X_test, ldx_test,
                                     y_pred);
}
da_status da_forest_predict_s(da_handle handle, da_int n_samples, da_int n_features,
                              const float *X_test, da_int ldx_test, da_int *y_pred) {
    return da_forest_predict<float>(handle, n_samples, n_features, X_test, ldx_test,
                                    y_pred);
}

da_status da_forest_predict_proba_d(da_handle handle, da_int n_samples, da_int n_features,
                                    const double *X_test, da_int ldx_test,
                                    double *y_proba, da_int n_class, da_int ldy) {
    return da_forest_predict_proba<double>(handle, n_samples, n_features, X_test,
                                           ldx_test, y_proba, n_class, ldy);
}
da_status da_forest_predict_proba_s(da_handle handle, da_int n_samples, da_int n_features,
                                    const float *X_test, da_int ldx_test, float *y_proba,
                                    da_int n_class, da_int ldy) {
    return da_forest_predict_proba<float>(handle, n_samples, n_features, X_test, ldx_test,
                                          y_proba, n_class, ldy);
}

da_status da_forest_predict_log_proba_d(da_handle handle, da_int n_samples,
                                        da_int n_features, const double *X_test,
                                        da_int ldx_test, double *y_log_proba,
                                        da_int n_class, da_int ldy) {
    return da_forest_predict_log_proba<double>(handle, n_samples, n_features, X_test,
                                               ldx_test, y_log_proba, n_class, ldy);
}
da_status da_forest_predict_log_proba_s(da_handle handle, da_int n_samples,
                                        da_int n_features, const float *X_test,
                                        da_int ldx_test, float *y_log_proba,
                                        da_int n_class, da_int ldy) {
    return da_forest_predict_log_proba<float>(handle, n_samples, n_features, X_test,
                                              ldx_test, y_log_proba, n_class, ldy);
}

da_status da_forest_score_d(da_handle handle, da_int n_samples, da_int n_features,
                            const double *X_test, da_int ldx_test, const da_int *y_test,
                            double *mean_accuracy) {
    return da_forest_score<double>(handle, n_samples, n_features, X_test, ldx_test,
                                   y_test, mean_accuracy);
}
da_status da_forest_score_s(da_handle handle, da_int n_samples, da_int n_features,
                            const float *X_test, da_int ldx_test, const da_int *y_test,
                            float *mean_accuracy) {
    return da_forest_score<float>(handle, n_samples, n_features, X_test, ldx_test, y_test,
                                  mean_accuracy);
}

/* ======================== NLLS (aoclda_nlls.h) ======================== */

da_status da_nlls_define_residuals_d(da_handle handle, da_int n_coef, da_int n_res,
                                     da_resfun_t_d *resfun, da_resgrd_t_d *resgrd,
                                     da_reshes_t_d *reshes, da_reshp_t_d *reshp) {
    return da_nlls_define_residuals<double>(handle, n_coef, n_res, resfun, resgrd, reshes,
                                            reshp);
}
da_status da_nlls_define_residuals_s(da_handle handle, da_int n_coef, da_int n_res,
                                     da_resfun_t_s *resfun, da_resgrd_t_s *resgrd,
                                     da_reshes_t_s *reshes, da_reshp_t_s *reshp) {
    return da_nlls_define_residuals<float>(handle, n_coef, n_res, resfun, resgrd, reshes,
                                           reshp);
}

da_status da_nlls_define_bounds_d(da_handle handle, da_int n_coef, double *lower,
                                  double *upper) {
    return da_nlls_define_bounds<double>(handle, n_coef, lower, upper);
}
da_status da_nlls_define_bounds_s(da_handle handle, da_int n_coef, float *lower,
                                  float *upper) {
    return da_nlls_define_bounds<float>(handle, n_coef, lower, upper);
}

da_status da_nlls_define_weights_d(da_handle handle, da_int n_res, double *weights) {
    return da_nlls_define_weights<double>(handle, n_res, weights);
}
da_status da_nlls_define_weights_s(da_handle handle, da_int n_res, float *weights) {
    return da_nlls_define_weights<float>(handle, n_res, weights);
}

da_status da_nlls_fit_d(da_handle handle, da_int n_coef, double *coef, void *udata) {
    return da_nlls_fit<double>(handle, n_coef, coef, udata);
}
da_status da_nlls_fit_s(da_handle handle, da_int n_coef, float *coef, void *udata) {
    return da_nlls_fit<float>(handle, n_coef, coef, udata);
}

/* ======================== Pairwise Distances (aoclda_metrics.h) ======================== */

da_status da_pairwise_distances_d(da_order order, da_int m, da_int n, da_int k,
                                  const double *X, da_int ldx, const double *Y,
                                  da_int ldy, double *D, da_int ldd, double p,
                                  da_metric metric) {
    return da_pairwise_distances<double>(order, m, n, k, X, ldx, Y, ldy, D, ldd, p,
                                         metric);
}
da_status da_pairwise_distances_s(da_order order, da_int m, da_int n, da_int k,
                                  const float *X, da_int ldx, const float *Y, da_int ldy,
                                  float *D, da_int ldd, float p, da_metric metric) {
    return da_pairwise_distances<float>(order, m, n, k, X, ldx, Y, ldy, D, ldd, p,
                                        metric);
}

/* ======================== k-NN (aoclda_nearest_neighbors.h) ======================== */

da_status da_nn_set_data_d(da_handle handle, da_int n_samples, da_int n_features,
                           const double *X_train, da_int ldx_train) {
    return da_nn_set_data<double>(handle, n_samples, n_features, X_train, ldx_train);
}
da_status da_nn_set_data_s(da_handle handle, da_int n_samples, da_int n_features,
                           const float *X_train, da_int ldx_train) {
    return da_nn_set_data<float>(handle, n_samples, n_features, X_train, ldx_train);
}

da_status da_nn_set_labels_d(da_handle handle, da_int n_samples, const da_int *y_train) {
    return da_nn_set_labels<double>(handle, n_samples, y_train);
}
da_status da_nn_set_labels_s(da_handle handle, da_int n_samples, const da_int *y_train) {
    return da_nn_set_labels<float>(handle, n_samples, y_train);
}

da_status da_nn_set_targets_d(da_handle handle, da_int n_samples, const double *y_train) {
    return da_nn_set_targets<double>(handle, n_samples, y_train);
}
da_status da_nn_set_targets_s(da_handle handle, da_int n_samples, const float *y_train) {
    return da_nn_set_targets<float>(handle, n_samples, y_train);
}

da_status da_nn_kneighbors_d(da_handle handle, da_int n_queries, da_int n_features,
                             const double *X_test, da_int ldx_test, da_int *n_ind,
                             double *n_dist, da_int k, da_int return_distance) {
    return da_nn_kneighbors<double>(handle, n_queries, n_features, X_test, ldx_test,
                                    n_ind, n_dist, k, return_distance);
}
da_status da_nn_kneighbors_s(da_handle handle, da_int n_queries, da_int n_features,
                             const float *X_test, da_int ldx_test, da_int *n_ind,
                             float *n_dist, da_int k, da_int return_distance) {
    return da_nn_kneighbors<float>(handle, n_queries, n_features, X_test, ldx_test, n_ind,
                                   n_dist, k, return_distance);
}

da_status da_nn_radius_neighbors_d(da_handle handle, da_int n_queries, da_int n_features,
                                   const double *X_test, da_int ldx_test, double radius,
                                   da_int return_distance, da_int sort_results) {
    return da_nn_radius_neighbors<double>(handle, n_queries, n_features, X_test, ldx_test,
                                          radius, return_distance, sort_results);
}
da_status da_nn_radius_neighbors_s(da_handle handle, da_int n_queries, da_int n_features,
                                   const float *X_test, da_int ldx_test, float radius,
                                   da_int return_distance, da_int sort_results) {
    return da_nn_radius_neighbors<float>(handle, n_queries, n_features, X_test, ldx_test,
                                         radius, return_distance, sort_results);
}

da_status da_nn_classes_d(da_handle handle, da_int *n_classes, da_int *classes) {
    return da_nn_classes<double>(handle, n_classes, classes);
}
da_status da_nn_classes_s(da_handle handle, da_int *n_classes, da_int *classes) {
    return da_nn_classes<float>(handle, n_classes, classes);
}

da_status da_nn_classifier_predict_proba_d(da_handle handle, da_int n_queries,
                                           da_int n_features, const double *X_test,
                                           da_int ldx_test, double *proba,
                                           da_nn_search_mode search_mode) {
    return da_nn_classifier_predict_proba<double>(handle, n_queries, n_features, X_test,
                                                  ldx_test, proba, search_mode);
}
da_status da_nn_classifier_predict_proba_s(da_handle handle, da_int n_queries,
                                           da_int n_features, const float *X_test,
                                           da_int ldx_test, float *proba,
                                           da_nn_search_mode search_mode) {
    return da_nn_classifier_predict_proba<float>(handle, n_queries, n_features, X_test,
                                                 ldx_test, proba, search_mode);
}

da_status da_nn_classifier_predict_d(da_handle handle, da_int n_queries,
                                     da_int n_features, const double *X_test,
                                     da_int ldx_test, da_int *y_test,
                                     da_nn_search_mode search_mode) {
    return da_nn_classifier_predict<double>(handle, n_queries, n_features, X_test,
                                            ldx_test, y_test, search_mode);
}
da_status da_nn_classifier_predict_s(da_handle handle, da_int n_queries,
                                     da_int n_features, const float *X_test,
                                     da_int ldx_test, da_int *y_test,
                                     da_nn_search_mode search_mode) {
    return da_nn_classifier_predict<float>(handle, n_queries, n_features, X_test,
                                           ldx_test, y_test, search_mode);
}

da_status da_nn_regressor_predict_d(da_handle handle, da_int n_queries, da_int n_features,
                                    const double *X_test, da_int ldx_test, double *y_test,
                                    da_nn_search_mode search_mode) {
    return da_nn_regressor_predict<double>(handle, n_queries, n_features, X_test,
                                           ldx_test, y_test, search_mode);
}
da_status da_nn_regressor_predict_s(da_handle handle, da_int n_queries, da_int n_features,
                                    const float *X_test, da_int ldx_test, float *y_test,
                                    da_nn_search_mode search_mode) {
    return da_nn_regressor_predict<float>(handle, n_queries, n_features, X_test, ldx_test,
                                          y_test, search_mode);
}

/* ======================== Utilities (aoclda_utils.h) ======================== */

da_status da_check_data_d(da_order order, da_int n_rows, da_int n_cols, const double *X,
                          da_int ldx) {
    return da_check_data<double>(order, n_rows, n_cols, X, ldx);
}
da_status da_check_data_s(da_order order, da_int n_rows, da_int n_cols, const float *X,
                          da_int ldx) {
    return da_check_data<float>(order, n_rows, n_cols, X, ldx);
}

da_status da_switch_order_copy_d(da_order order_X, da_int n_rows, da_int n_cols,
                                 const double *X, da_int ldx, double *Y, da_int ldy) {
    return da_switch_order_copy<double>(order_X, n_rows, n_cols, X, ldx, Y, ldy);
}
da_status da_switch_order_copy_s(da_order order_X, da_int n_rows, da_int n_cols,
                                 const float *X, da_int ldx, float *Y, da_int ldy) {
    return da_switch_order_copy<float>(order_X, n_rows, n_cols, X, ldx, Y, ldy);
}

da_status da_switch_order_in_place_d(da_order order_X_in, da_int n_rows, da_int n_cols,
                                     double *X, da_int ldx_in, da_int ldx_out) {
    return da_switch_order_in_place<double>(order_X_in, n_rows, n_cols, X, ldx_in,
                                            ldx_out);
}
da_status da_switch_order_in_place_s(da_order order_X_in, da_int n_rows, da_int n_cols,
                                     float *X, da_int ldx_in, da_int ldx_out) {
    return da_switch_order_in_place<float>(order_X_in, n_rows, n_cols, X, ldx_in,
                                           ldx_out);
}

da_status da_get_shuffled_indices_d(da_int m, da_int seed, da_int train_size,
                                    da_int test_size, da_int fp_precision,
                                    const double *classes, da_int *shuffle_array) {
    return da_get_shuffled_indices<double>(m, seed, train_size, test_size, fp_precision,
                                           classes, shuffle_array);
}
da_status da_get_shuffled_indices_s(da_int m, da_int seed, da_int train_size,
                                    da_int test_size, da_int fp_precision,
                                    const float *classes, da_int *shuffle_array) {
    return da_get_shuffled_indices<float>(m, seed, train_size, test_size, fp_precision,
                                          classes, shuffle_array);
}
da_status da_get_shuffled_indices_int(da_int m, da_int seed, da_int train_size,
                                      da_int test_size, da_int fp_precision,
                                      const da_int *classes, da_int *shuffle_array) {
    return da_get_shuffled_indices<da_int>(m, seed, train_size, test_size, fp_precision,
                                           classes, shuffle_array);
}

da_status da_train_test_split_d(da_order order, da_int m, da_int n, const double *X,
                                da_int ldx, da_int train_size, da_int test_size,
                                const da_int *shuffle_array, double *X_train,
                                da_int ldx_train, double *X_test, da_int ldx_test) {
    return da_train_test_split<double>(order, m, n, X, ldx, train_size, test_size,
                                       shuffle_array, X_train, ldx_train, X_test,
                                       ldx_test);
}
da_status da_train_test_split_s(da_order order, da_int m, da_int n, const float *X,
                                da_int ldx, da_int train_size, da_int test_size,
                                const da_int *shuffle_array, float *X_train,
                                da_int ldx_train, float *X_test, da_int ldx_test) {
    return da_train_test_split<float>(order, m, n, X, ldx, train_size, test_size,
                                      shuffle_array, X_train, ldx_train, X_test,
                                      ldx_test);
}
da_status da_train_test_split_int(da_order order, da_int m, da_int n, const da_int *X,
                                  da_int ldx, da_int train_size, da_int test_size,
                                  const da_int *shuffle_array, da_int *X_train,
                                  da_int ldx_train, da_int *X_test, da_int ldx_test) {
    return da_train_test_split<da_int>(order, m, n, X, ldx, train_size, test_size,
                                       shuffle_array, X_train, ldx_train, X_test,
                                       ldx_test);
}

/* ======================== Kernel Functions (aoclda_kernel_functions.h) ======================== */

da_status da_rbf_kernel_d(da_order order, da_int m, da_int n, da_int k, const double *X,
                          da_int ldx, const double *Y, da_int ldy, double *D, da_int ldd,
                          double gamma) {
    return da_rbf_kernel<double>(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma);
}
da_status da_rbf_kernel_s(da_order order, da_int m, da_int n, da_int k, const float *X,
                          da_int ldx, const float *Y, da_int ldy, float *D, da_int ldd,
                          float gamma) {
    return da_rbf_kernel<float>(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma);
}

da_status da_linear_kernel_d(da_order order, da_int m, da_int n, da_int k,
                             const double *X, da_int ldx, const double *Y, da_int ldy,
                             double *D, da_int ldd) {
    return da_linear_kernel<double>(order, m, n, k, X, ldx, Y, ldy, D, ldd);
}
da_status da_linear_kernel_s(da_order order, da_int m, da_int n, da_int k, const float *X,
                             da_int ldx, const float *Y, da_int ldy, float *D,
                             da_int ldd) {
    return da_linear_kernel<float>(order, m, n, k, X, ldx, Y, ldy, D, ldd);
}

da_status da_polynomial_kernel_d(da_order order, da_int m, da_int n, da_int k,
                                 const double *X, da_int ldx, const double *Y, da_int ldy,
                                 double *D, da_int ldd, double gamma, da_int degree,
                                 double coef0) {
    return da_polynomial_kernel<double>(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma,
                                        degree, coef0);
}
da_status da_polynomial_kernel_s(da_order order, da_int m, da_int n, da_int k,
                                 const float *X, da_int ldx, const float *Y, da_int ldy,
                                 float *D, da_int ldd, float gamma, da_int degree,
                                 float coef0) {
    return da_polynomial_kernel<float>(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma,
                                       degree, coef0);
}

da_status da_sigmoid_kernel_d(da_order order, da_int m, da_int n, da_int k,
                              const double *X, da_int ldx, const double *Y, da_int ldy,
                              double *D, da_int ldd, double gamma, double coef0) {
    return da_sigmoid_kernel<double>(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma,
                                     coef0);
}
da_status da_sigmoid_kernel_s(da_order order, da_int m, da_int n, da_int k,
                              const float *X, da_int ldx, const float *Y, da_int ldy,
                              float *D, da_int ldd, float gamma, float coef0) {
    return da_sigmoid_kernel<float>(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, coef0);
}

/* ======================== SVM (aoclda_svm.h) ======================== */

da_status da_svm_select_model_d(da_handle handle, da_svm_model mod) {
    return da_svm_select_model<double>(handle, mod);
}
da_status da_svm_select_model_s(da_handle handle, da_svm_model mod) {
    return da_svm_select_model<float>(handle, mod);
}

da_status da_svm_set_data_d(da_handle handle, da_int n_samples, da_int n_features,
                            const double *X, da_int ldx, const double *y) {
    return da_svm_set_data<double>(handle, n_samples, n_features, X, ldx, y);
}
da_status da_svm_set_data_s(da_handle handle, da_int n_samples, da_int n_features,
                            const float *X, da_int ldx, const float *y) {
    return da_svm_set_data<float>(handle, n_samples, n_features, X, ldx, y);
}

da_status da_svm_compute_d(da_handle handle) { return da_svm_compute<double>(handle); }
da_status da_svm_compute_s(da_handle handle) { return da_svm_compute<float>(handle); }

da_status da_svm_predict_d(da_handle handle, da_int n_samples, da_int n_features,
                           const double *X_test, da_int ldx_test, double *predictions) {
    return da_svm_predict<double>(handle, n_samples, n_features, X_test, ldx_test,
                                  predictions);
}
da_status da_svm_predict_s(da_handle handle, da_int n_samples, da_int n_features,
                           const float *X_test, da_int ldx_test, float *predictions) {
    return da_svm_predict<float>(handle, n_samples, n_features, X_test, ldx_test,
                                 predictions);
}

da_status da_svm_decision_function_d(da_handle handle, da_int n_samples,
                                     da_int n_features, const double *X_test,
                                     da_int ldx_test,
                                     da_svm_decision_function_shape shape,
                                     double *decision_values, da_int ldd) {
    return da_svm_decision_function<double>(handle, n_samples, n_features, X_test,
                                            ldx_test, shape, decision_values, ldd);
}
da_status da_svm_decision_function_s(da_handle handle, da_int n_samples,
                                     da_int n_features, const float *X_test,
                                     da_int ldx_test,
                                     da_svm_decision_function_shape shape,
                                     float *decision_values, da_int ldd) {
    return da_svm_decision_function<float>(handle, n_samples, n_features, X_test,
                                           ldx_test, shape, decision_values, ldd);
}

da_status da_svm_score_d(da_handle handle, da_int n_samples, da_int n_features,
                         const double *X_test, da_int ldx_test, const double *y_test,
                         double *score) {
    return da_svm_score<double>(handle, n_samples, n_features, X_test, ldx_test, y_test,
                                score);
}
da_status da_svm_score_s(da_handle handle, da_int n_samples, da_int n_features,
                         const float *X_test, da_int ldx_test, const float *y_test,
                         float *score) {
    return da_svm_score<float>(handle, n_samples, n_features, X_test, ldx_test, y_test,
                               score);
}

da_status da_svm_predict_proba_d(da_handle handle, da_int n_samples, da_int n_features,
                                 const double *X_test, da_int ldx_test, double *y_proba,
                                 da_int ldy) {
    return da_svm_predict_proba<double>(handle, n_samples, n_features, X_test, ldx_test,
                                        y_proba, ldy);
}
da_status da_svm_predict_proba_s(da_handle handle, da_int n_samples, da_int n_features,
                                 const float *X_test, da_int ldx_test, float *y_proba,
                                 da_int ldy) {
    return da_svm_predict_proba<float>(handle, n_samples, n_features, X_test, ldx_test,
                                       y_proba, ldy);
}

da_status da_svm_predict_log_proba_d(da_handle handle, da_int n_samples,
                                     da_int n_features, const double *X_test,
                                     da_int ldx_test, double *y_log_proba, da_int ldy) {
    return da_svm_predict_log_proba<double>(handle, n_samples, n_features, X_test,
                                            ldx_test, y_log_proba, ldy);
}
da_status da_svm_predict_log_proba_s(da_handle handle, da_int n_samples,
                                     da_int n_features, const float *X_test,
                                     da_int ldx_test, float *y_log_proba, da_int ldy) {
    return da_svm_predict_log_proba<float>(handle, n_samples, n_features, X_test,
                                           ldx_test, y_log_proba, ldy);
}

/* ======================== Interpolation (aoclda_interpolation.h) ======================== */

da_status da_interpolation_select_model_d(da_handle handle,
                                          da_interpolation_model model) {
    return da_interpolation_select_model<double>(handle, model);
}
da_status da_interpolation_select_model_s(da_handle handle,
                                          da_interpolation_model model) {
    return da_interpolation_select_model<float>(handle, model);
}

da_status da_interpolation_set_sites_d(da_handle handle, da_int n_sites,
                                       const double *x) {
    return da_interpolation_set_sites<double>(handle, n_sites, x);
}
da_status da_interpolation_set_sites_s(da_handle handle, da_int n_sites, const float *x) {
    return da_interpolation_set_sites<float>(handle, n_sites, x);
}

da_status da_interpolation_set_sites_uniform_d(da_handle handle, da_int n_sites,
                                               double x_start, double x_end) {
    return da_interpolation_set_sites_uniform<double>(handle, n_sites, x_start, x_end);
}
da_status da_interpolation_set_sites_uniform_s(da_handle handle, da_int n_sites,
                                               float x_start, float x_end) {
    return da_interpolation_set_sites_uniform<float>(handle, n_sites, x_start, x_end);
}

da_status da_interpolation_set_values_d(da_handle handle, da_int n, da_int dim,
                                        const double *y_data, da_int ldy_data,
                                        da_int order) {
    return da_interpolation_set_values<double>(handle, n, dim, y_data, ldy_data, order);
}
da_status da_interpolation_set_values_s(da_handle handle, da_int n, da_int dim,
                                        const float *y_data, da_int ldy_data,
                                        da_int order) {
    return da_interpolation_set_values<float>(handle, n, dim, y_data, ldy_data, order);
}

da_status da_interpolation_search_cells_d(da_handle handle, da_int n_eval,
                                          const double *x_eval, da_int *cells) {
    return da_interpolation_search_cells<double>(handle, n_eval, x_eval, cells);
}
da_status da_interpolation_search_cells_s(da_handle handle, da_int n_eval,
                                          const float *x_eval, da_int *cells) {
    return da_interpolation_search_cells<float>(handle, n_eval, x_eval, cells);
}

da_status da_interpolation_interpolate_d(da_handle handle) {
    return da_interpolation_interpolate<double>(handle);
}
da_status da_interpolation_interpolate_s(da_handle handle) {
    return da_interpolation_interpolate<float>(handle);
}

da_status da_interpolation_set_boundary_conditions_d(da_handle handle, da_int dim,
                                                     da_int left_order,
                                                     const double *left_values,
                                                     da_int right_order,
                                                     const double *right_values) {
    return da_interpolation_set_boundary_conditions<double>(
        handle, dim, left_order, left_values, right_order, right_values);
}
da_status da_interpolation_set_boundary_conditions_s(da_handle handle, da_int dim,
                                                     da_int left_order,
                                                     const float *left_values,
                                                     da_int right_order,
                                                     const float *right_values) {
    return da_interpolation_set_boundary_conditions<float>(
        handle, dim, left_order, left_values, right_order, right_values);
}

da_status da_interpolation_evaluate_d(da_handle handle, da_int n_eval,
                                      const double *x_eval, double *y_eval,
                                      da_int n_orders, da_int *orders) {
    return da_interpolation_evaluate<double>(handle, n_eval, x_eval, y_eval, n_orders,
                                             orders);
}
da_status da_interpolation_evaluate_s(da_handle handle, da_int n_eval,
                                      const float *x_eval, float *y_eval, da_int n_orders,
                                      da_int *orders) {
    return da_interpolation_evaluate<float>(handle, n_eval, x_eval, y_eval, n_orders,
                                            orders);
}

/* ======================== Approximate Neighbors (aoclda_approximate_neighbors.h) ======================== */

da_status da_approx_nn_set_training_data_d(da_handle handle, da_int n_samples,
                                           da_int n_features, const double *X_train,
                                           da_int ldx_train) {
    return da_approx_nn_set_training_data<double>(handle, n_samples, n_features, X_train,
                                                  ldx_train);
}
da_status da_approx_nn_set_training_data_s(da_handle handle, da_int n_samples,
                                           da_int n_features, const float *X_train,
                                           da_int ldx_train) {
    return da_approx_nn_set_training_data<float>(handle, n_samples, n_features, X_train,
                                                 ldx_train);
}

da_status da_approx_nn_train_d(da_handle handle) {
    return da_approx_nn_train<double>(handle);
}
da_status da_approx_nn_train_s(da_handle handle) {
    return da_approx_nn_train<float>(handle);
}

da_status da_approx_nn_add_d(da_handle handle, da_int n_samples_add, da_int n_features,
                             const double *X_add, da_int ldx_add) {
    return da_approx_nn_add<double>(handle, n_samples_add, n_features, X_add, ldx_add);
}
da_status da_approx_nn_add_s(da_handle handle, da_int n_samples_add, da_int n_features,
                             const float *X_add, da_int ldx_add) {
    return da_approx_nn_add<float>(handle, n_samples_add, n_features, X_add, ldx_add);
}

da_status da_approx_nn_train_and_add_d(da_handle handle) {
    return da_approx_nn_train_and_add<double>(handle);
}
da_status da_approx_nn_train_and_add_s(da_handle handle) {
    return da_approx_nn_train_and_add<float>(handle);
}

da_status da_approx_nn_kneighbors_d(da_handle handle, da_int n_queries, da_int n_features,
                                    const double *X_test, da_int ldx_test, da_int *n_ind,
                                    double *n_dist, da_int k, da_int return_distance) {
    return da_approx_nn_kneighbors<double>(handle, n_queries, n_features, X_test,
                                           ldx_test, n_ind, n_dist, k,
                                           static_cast<bool>(return_distance));
}
da_status da_approx_nn_kneighbors_s(da_handle handle, da_int n_queries, da_int n_features,
                                    const float *X_test, da_int ldx_test, da_int *n_ind,
                                    float *n_dist, da_int k, da_int return_distance) {
    return da_approx_nn_kneighbors<float>(handle, n_queries, n_features, X_test, ldx_test,
                                          n_ind, n_dist, k,
                                          static_cast<bool>(return_distance));
}

} // extern "C"
