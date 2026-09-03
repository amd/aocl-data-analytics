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

#ifndef AOCLDA_CPP
#define AOCLDA_CPP

#include "aoclda.h"

#include <vector>

/* da_handle declarations */
template <typename T>
da_status da_handle_init(da_handle *handle, da_handle_type handle_type);
template <typename T>
da_status da_handle_get_result(da_handle handle, da_result query, da_int *dim, T *result);

/* Save and Load overloads */
da_status da_handle_save_model(da_handle handle, std::vector<char> &buffer);
da_status da_handle_load_model(da_handle *handle, const char *buffer_data,
                               const size_t data_size);

/* Options declarations */
template <typename T>
da_status da_options_set(da_handle handle, const char *option, T value);
template <typename T>
da_status da_options_get(da_handle handle, const char *option, T *value);
da_status da_options_get(da_handle handle, const char *option, char *value,
                         da_int *lvalue, da_int *key = nullptr);

/* Datastore options declarations */
template <typename T>
da_status da_datastore_options_set(da_datastore store, const char *option, T value);
template <typename T>
da_status da_datastore_options_get(da_datastore store, const char *option, T *value);
da_status da_datastore_options_get(da_datastore store, const char *option, char *value,
                                   da_int *lvalue);

/* CSV declarations */
template <typename T>
da_status da_read_csv(da_datastore store, const char *filename, T **A, da_int *n_rows,
                      da_int *n_cols, char ***headings);

/* Basic statistics declarations */
template <typename T>
da_status da_mean(da_order order, da_axis axis, da_int n_rows, da_int n_cols, const T *X,
                  da_int ldx, T *mean);
template <typename T>
da_status da_harmonic_mean(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                           const T *X, da_int ldx, T *harmonic_mean);
template <typename T>
da_status da_geometric_mean(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                            const T *X, da_int ldx, T *geometric_mean);
template <typename T>
da_status da_variance(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                      const T *X, da_int ldx, da_int dof, T *mean, T *variance);
template <typename T>
da_status da_skewness(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                      const T *X, da_int ldx, T *mean, T *variance, T *skewness);
template <typename T>
da_status da_kurtosis(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                      const T *X, da_int ldx, T *mean, T *variance, T *kurtosis);
template <typename T>
da_status da_moment(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                    const T *X, da_int ldx, da_int k, da_int use_precomputed_mean,
                    T *mean, T *moment);
template <typename T>
da_status da_quantile(da_order order, da_axis axis, da_int n_rows, da_int n_cols,
                      const T *X, da_int ldx, const T *q, da_int n_q, T *quantiles,
                      da_quantile_type quantile_type);
template <typename T>
da_status da_five_point_summary(da_order order, da_axis axis, da_int n_rows,
                                da_int n_cols, const T *X, da_int ldx, T *minimum,
                                T *lower_hinge, T *median, T *upper_hinge, T *maximum);
template <typename T>
da_status da_standardize(da_order order, da_axis axis, da_int n_rows, da_int n_cols, T *X,
                         da_int ldx, da_int dof, da_int mode, T *shift, T *scale);
template <typename T>
da_status da_covariance_matrix(da_order order, da_int n_rows, da_int n_cols, const T *X,
                               da_int ldx, da_int dof, T *cov, da_int ldcov,
                               da_int assume_centered);
template <typename T>
da_status da_correlation_matrix(da_order order, da_int n_rows, da_int n_cols, const T *X,
                                da_int ldx, T *corr, da_int ldcorr);

/* Linear model declarations */
template <typename T>
da_status da_linmod_select_model(da_handle handle, linmod_model mod);
template <typename T>
da_status da_linmod_define_features(da_handle handle, da_int n_samples, da_int n_features,
                                    const T *X, da_int ldx, const T *y);
template <typename T> da_status da_linmod_fit(da_handle handle);
template <typename T>
da_status da_linmod_fit_start(da_handle handle, da_int ncoef, const T *coefs);
template <typename T>
da_status da_linmod_evaluate_model(da_handle handle, da_int n_samples, da_int n_features,
                                   const T *X, da_int ldx, T *predictions,
                                   const T *observations = nullptr, T *loss = nullptr);

/* Datastore declarations */
template <typename T>
da_status da_data_load_col(da_datastore store, da_int n_rows, da_int n_cols, T *block,
                           da_order order, da_int copy_data);
da_status da_data_load_col(da_datastore store, da_int n_rows, da_int n_cols,
                           const char **block, da_order order);
template <typename T>
da_status da_data_load_row(da_datastore store, da_int n_rows, da_int n_cols, T *block,
                           da_order order, da_int copy_data);
da_status da_data_load_row(da_datastore store, da_int n_rows, da_int n_cols,
                           const char **block, da_order order);
template <typename T>
da_status da_data_get_element(da_datastore store, da_int i, da_int j, T *elem);
template <typename T>
da_status da_data_set_element(da_datastore store, da_int i, da_int j, T elem);
template <typename T>
da_status da_data_extract_column(da_datastore store, da_int idx, da_int dim, T *col);
template <typename T>
da_status da_data_extract_selection(da_datastore store, const char *key, da_order order,
                                    T *data, da_int lddata);

/* PCA declarations */
template <typename T>
da_status da_pca_set_data(da_handle handle, da_int n_samples, da_int n_features,
                          const T *A, da_int lda);
template <typename T> da_status da_pca_compute(da_handle handle);
template <typename T>
da_status da_pca_transform(da_handle handle, da_int m_samples, da_int m_features,
                           const T *X, da_int ldx, T *X_transform, da_int ldx_transform);
template <typename T>
da_status da_pca_inverse_transform(da_handle handle, da_int k_samples, da_int k_features,
                                   const T *X, da_int ldx, T *X_inv_transform,
                                   da_int ldx_inv_transform);

/* k-means declarations */
template <typename T>
da_status da_kmeans_set_data(da_handle handle, da_int n_samples, da_int n_features,
                             const T *A, da_int lda);
template <typename T>
da_status da_kmeans_set_init_centres(da_handle handle, const T *C, da_int ldc);
template <typename T> da_status da_kmeans_compute(da_handle handle);
template <typename T>
da_status da_kmeans_transform(da_handle handle, da_int m_samples, da_int m_features,
                              const T *X, da_int ldx, T *X_transform,
                              da_int ldx_transform);
template <typename T>
da_status da_kmeans_predict(da_handle handle, da_int k_samples, da_int k_features,
                            const T *Y, da_int ldy, da_int *Y_labels);

/* DBSCAN declarations */
template <typename T>
da_status da_dbscan_set_data(da_handle handle, da_int n_samples, da_int n_features,
                             const T *A, da_int lda);
template <typename T> da_status da_dbscan_compute(da_handle handle);

/* Decision Forest declarations */
/* Decision tree */
template <typename T>
da_status da_tree_set_training_data(da_handle handle, da_int n_samples, da_int n_features,
                                    da_int n_class, const T *X, da_int ldx,
                                    const da_int *y,
                                    const da_int *categorical_features = nullptr);
template <typename T> da_status da_tree_fit(da_handle handle);
template <typename T>
da_status da_tree_predict(da_handle handle, da_int n_obs, da_int n_features,
                          const T *X_test, da_int ldx_test, da_int *y_pred);
template <typename T>
da_status da_tree_predict_proba(da_handle handle, da_int n_obs, da_int n_features,
                                const T *X_test, da_int ldx_test, T *y_pred,
                                da_int n_class, da_int ldy);
template <typename T>
da_status da_tree_predict_log_proba(da_handle handle, da_int n_obs, da_int n_features,
                                    const T *X_test, da_int ldx_test, T *y_pred,
                                    da_int n_class, da_int ldy);
template <typename T>
da_status da_tree_score(da_handle handle, da_int n_samples, da_int n_features,
                        const T *X_test, da_int ldx_test, const da_int *y_test,
                        T *mean_accuracy);

/* Random forest */
template <typename T>
da_status da_forest_set_training_data(da_handle handle, da_int n_samples,
                                      da_int n_features, da_int n_class, const T *X,
                                      da_int ldx, const da_int *y,
                                      const da_int *categorical_features = nullptr);
template <typename T> da_status da_forest_fit(da_handle handle);
template <typename T>
da_status da_forest_predict(da_handle handle, da_int n_samples, da_int n_features,
                            const T *X_test, da_int ldx_test, da_int *y_pred);
template <typename T>
da_status da_forest_predict_proba(da_handle handle, da_int n_samples, da_int n_features,
                                  const T *X_test, da_int ldx_test, T *y_pred,
                                  da_int n_class, da_int ldy);
template <typename T>
da_status da_forest_predict_log_proba(da_handle handle, da_int n_obs, da_int n_features,
                                      const T *X_test, da_int ldx_test, T *y_pred,
                                      da_int n_class, da_int ldy);
template <typename T>
da_status da_forest_score(da_handle handle, da_int n_samples, da_int n_features,
                          const T *X_test, da_int ldx_test, const da_int *y_test,
                          T *mean_accuracy);

/* NLLS declarations */
template <typename T>
using da_resfun_t =
    std::conditional_t<std::is_same_v<T, double>, da_resfun_t_d, da_resfun_t_s>;
template <typename T>
using da_resgrd_t =
    std::conditional_t<std::is_same_v<T, double>, da_resgrd_t_d, da_resgrd_t_s>;
template <typename T>
using da_reshes_t =
    std::conditional_t<std::is_same_v<T, double>, da_reshes_t_d, da_reshes_t_s>;
template <typename T>
using da_reshp_t =
    std::conditional_t<std::is_same_v<T, double>, da_reshp_t_d, da_reshp_t_s>;

template <typename T>
da_status da_nlls_define_residuals(da_handle handle, da_int n_coef, da_int n_res,
                                   da_resfun_t<T> *resfun, da_resgrd_t<T> *resgrd,
                                   da_reshes_t<T> *reshes, da_reshp_t<T> *reshp);
template <typename T>
da_status da_nlls_define_bounds(da_handle handle, da_int n_coef, T *lower, T *upper);
template <typename T>
da_status da_nlls_define_weights(da_handle handle, da_int n_coef, T *weights);
template <typename T>
da_status da_nlls_fit(da_handle handle, da_int n_coef, T *coef, void *udata);

/* Pairwise distances declarations */
template <typename T>
da_status da_pairwise_distances(da_order order, da_int m, da_int n, da_int k, const T *X,
                                da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd, T p,
                                da_metric metric);

/* k-NN declarations */
template <typename T>
da_status da_nn_set_data(da_handle handle, da_int n_samples, da_int n_features,
                         const T *X_train, da_int ldx_train);
template <typename T>
da_status da_nn_set_labels(da_handle handle, da_int n_samples, const da_int *y_train);
template <typename T>
da_status da_nn_set_targets(da_handle handle, da_int n_samples, const T *y_train);
template <typename T>
da_status da_nn_kneighbors(da_handle handle, da_int n_queries, da_int n_features,
                           const T *X_test, da_int ldx_test, da_int *n_ind, T *n_dist,
                           da_int k, da_int return_distance);
template <typename T>
da_status da_nn_classes(da_handle handle, da_int *n_classes, da_int *classes);
template <typename T>
da_status da_nn_classifier_predict_proba(da_handle handle, da_int n_queries,
                                         da_int n_features, const T *X_test,
                                         da_int ldx_test, T *proba,
                                         da_nn_search_mode search_mode);
template <typename T>
da_status da_nn_classifier_predict(da_handle handle, da_int n_queries, da_int n_features,
                                   const T *X_test, da_int ldx_test, da_int *y_test,
                                   da_nn_search_mode search_mode);
template <typename T>
da_status da_nn_regressor_predict(da_handle handle, da_int n_queries, da_int n_features,
                                  const T *X_test, da_int ldx_test, T *y_test,
                                  da_nn_search_mode search_mode);
template <typename T>
da_status da_nn_radius_neighbors(da_handle handle, da_int n_queries, da_int n_features,
                                 const T *X_test, da_int ldx_test, T radius,
                                 da_int return_distance, da_int sort_results);

/* Utility declarations */
template <typename T>
da_status da_check_data(da_order order, da_int n_rows, da_int n_cols, const T *X,
                        da_int ldx);
template <typename T>
da_status da_switch_order_copy(da_order order, da_int n_rows, da_int n_cols, const T *X,
                               da_int ldx, T *Y, da_int ldy);
template <typename T>
da_status da_switch_order_in_place(da_order order_X_in, da_int n_rows, da_int n_cols,
                                   T *X, da_int ldx_in, da_int ldx_out);
template <typename T>
da_status da_get_shuffled_indices(da_int m, da_int seed, da_int train_size,
                                  da_int test_size, da_int fp_precision, const T *classes,
                                  da_int *shuffle_array);
template <typename T>
da_status da_train_test_split(da_order order, da_int m, da_int n, const T *X, da_int ldx,
                              da_int train_size, da_int test_size,
                              const da_int *shuffle_array, T *X_train, da_int ldx_train,
                              T *X_test, da_int ldx_test);

da_status da_print_model_metadata(const std::vector<char> &file_data);

/* Kernel function declarations */
template <typename T>
da_status da_rbf_kernel(da_order order, da_int m, da_int n, da_int k, const T *X,
                        da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd, T gamma);
template <typename T>
da_status da_linear_kernel(da_order order, da_int m, da_int n, da_int k, const T *X,
                           da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd);
template <typename T>
da_status da_polynomial_kernel(da_order order, da_int m, da_int n, da_int k, const T *X,
                               da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd,
                               T gamma, da_int degree, T coef0);
template <typename T>
da_status da_sigmoid_kernel(da_order order, da_int m, da_int n, da_int k, const T *X,
                            da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd, T gamma,
                            T coef0);

/* SVM declarations */
template <typename T> da_status da_svm_select_model(da_handle handle, da_svm_model mod);
template <typename T>
da_status da_svm_set_data(da_handle handle, da_int n_samples, da_int n_features,
                          const T *X, da_int ldx_train, const T *y);
template <typename T> da_status da_svm_compute(da_handle handle);
template <typename T>
da_status da_svm_predict(da_handle handle, da_int n_samples, da_int n_features,
                         const T *X_test, da_int ldx_test, T *predictions);
template <typename T>
da_status da_svm_decision_function(da_handle handle, da_int n_samples, da_int n_features,
                                   const T *X_test, da_int ldx_test,
                                   da_svm_decision_function_shape shape,
                                   T *decision_values, da_int ldd);
template <typename T>
da_status da_svm_score(da_handle handle, da_int n_samples, da_int n_features,
                       const T *X_test, da_int ldx_test, const T *y_test, T *score);
template <typename T>
da_status da_svm_predict_proba(da_handle handle, da_int n_samples, da_int n_features,
                               const T *X_test, da_int ldx_test, T *y_proba, da_int ldy);
template <typename T>
da_status da_svm_predict_log_proba(da_handle handle, da_int n_samples, da_int n_features,
                                   const T *X_test, da_int ldx_test, T *y_log_proba,
                                   da_int ldy);

/* Interpolation declarations */
template <typename T>
da_status da_interpolation_select_model(da_handle handle, da_interpolation_model model);
template <typename T>
da_status da_interpolation_set_sites(da_handle handle, da_int n_sites, const T *x);
template <typename T>
da_status da_interpolation_set_sites_uniform(da_handle handle, da_int n_sites, T x_start,
                                             T x_end);
template <typename T>
da_status da_interpolation_set_values(da_handle handle, da_int n, da_int dim,
                                      const T *y_data, da_int ldy, da_int order);
template <typename T>
da_status da_interpolation_search_cells(da_handle handle, da_int n_eval, const T *x_eval,
                                        da_int *cells);
template <typename T> da_status da_interpolation_interpolate(da_handle handle);
template <typename T>
da_status
da_interpolation_set_boundary_conditions(da_handle handle, da_int dim, da_int left_order,
                                         const T *left_values, da_int right_order,
                                         const T *right_values);
template <typename T>
da_status da_interpolation_evaluate(da_handle handle, da_int n_eval, const T *x_eval,
                                    T *y_eval, da_int n_orders, da_int *orders);

/* Approximate Neighbors declarations */
template <typename T>
da_status da_approx_nn_set_training_data(da_handle handle, da_int n_samples,
                                         da_int n_features, const T *X_train,
                                         da_int ldx_train);
template <typename T> da_status da_approx_nn_train(da_handle handle);
template <typename T>
da_status da_approx_nn_add(da_handle handle, da_int n_samples_add, da_int n_features,
                           const T *X_add, da_int ldx_add);
template <typename T> da_status da_approx_nn_train_and_add(da_handle handle);
template <typename T>
da_status da_approx_nn_kneighbors(da_handle handle, da_int n_queries, da_int n_features,
                                  const T *X_test, da_int ldx_test, da_int *n_ind,
                                  T *n_dist, da_int k, bool return_distance);

/* Kernel PCA declarations */
template <typename T>
da_status da_kernel_pca_set_data(da_handle handle, da_int n_samples, da_int n_features,
                                 const T *A, da_int lda);
template <typename T> da_status da_kernel_pca_compute(da_handle handle);
template <typename T>
da_status da_kernel_pca_transform(da_handle handle, da_int m_samples, da_int m_features,
                                  const T *X, da_int ldx, T *X_transform,
                                  da_int ldx_transform);
template <typename T>
da_status da_kernel_pca_inverse_transform(da_handle handle, da_int k_samples,
                                          da_int k_components, const T *Y, da_int ldy,
                                          T *Y_inv_transform, da_int ldy_inv_transform);

/* t-SNE declarations */
template <typename T>
da_status da_tsne_set_data(da_handle handle, da_int n_samples, da_int n_features,
                           const T *X, da_int ldx);
template <typename T>
da_status da_tsne_set_init_embedding(da_handle handle, const T *Y, da_int ldy);
template <typename T> da_status da_tsne_compute(da_handle handle);

#endif // AOCLDA_CPP_OVERLOADS
