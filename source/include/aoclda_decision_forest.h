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

#ifndef AOCLDA_DF
#define AOCLDA_DF

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"

/**
 * \file
 */

/** \{
 * @brief Pass a data matrix and a label array to the \ref da_handle object
 * in preparation for fitting a decision tree.
 *
 * @param handle a @ref da_handle object, initialized with type @ref da_handle_decision_tree.
 * @param n_samples number of observations in \p X.
 * @param n_features number of features in \p X.
 * @param n_class number of distinct classes in \p y. Will be computed automatically if \p n_class is set to 0.
 * @param X array containing \p n_samples  @f$\times@f$ \p n_features data matrix, in column-major format.
 * @param ldx leading dimension of \p X.  Constraint: \p ldx @f$\ge@f$ \p n_samples.
 * @param y array containing the \p n_samples labels. The label values are expected to range from 0 to \p n_class - 1.
 * @return @ref da_status.  The function returns:
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using @ref da_handle_print_error_message.
 * - @ref da_status_memory_error - internal memory allocation encountered a problem.
*/
da_status da_tree_set_training_data_d(da_handle handle, da_int n_samples,
                                      da_int n_features, da_int n_class, double *X,
                                      da_int ldx, da_int *y);
da_status da_tree_set_training_data_s(da_handle handle, da_int n_samples,
                                      da_int n_features, da_int n_class, float *X,
                                      da_int ldx, da_int *y);
/** \} */

/** \{
 * @brief Pass a data matrix and a label array to the \ref da_handle object
 * in preparation for fitting a decision forest.
 *
 * @param handle a @ref da_handle object, initialized with type @ref da_handle_decision_forest.
 * @param n_samples number of observations in \p X.
 * @param n_features number of features in \p X.
 * @param n_class number of distinct classes in \p y. Will be computed automatically if \p n_class is set to 0.
 * @param X array containing \p n_samples  @f$\times@f$ \p n_features data matrix, in column-major format.
 * @param ldx leading dimension of \p X.  Constraint: \p ldx @f$\ge@f$ \p n_samples.
 * @param y array containing the \p n_samples labels. The label values are expected to range from 0 to \p n_class - 1.
 * @return @ref da_status.  The function returns:
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using @ref da_handle_print_error_message.
 */
da_status da_forest_set_training_data_d(da_handle handle, da_int n_samples,
                                        da_int n_features, da_int n_class, double *X,
                                        da_int ldx, da_int *y);
da_status da_forest_set_training_data_s(da_handle handle, da_int n_samples,
                                        da_int n_features, da_int n_class, float *X,
                                        da_int ldx, da_int *y);
/** \} */

/** \{
 * @brief Fit the decision tree defined in the @p handle.
 *
 * @rst
 * Compute the decision tree parameters given the data passed by :ref:`da_tree_set_training_data_? <da_tree_set_training_data>`.
 * Note that you can customize the model before using the fit function through the use of optional parameters,
 * see :ref:`this section <opts_decisionforests>` for a list of available options.
 * @endrst
 *
 * @param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_decision_tree.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle
 *   initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_incompatible_options - some of the options set are incompatible with the model defined in \p handle.
 *        You can obtain further information using @ref da_handle_print_error_message.
 * - @ref da_status_memory_error - internal memory allocation encountered a problem.
 * - @ref da_status_internal_error - an unexpected error occurred.
 *
 * \post
 * After successful execution, \ref da_handle_get_result_s "da_handle_get_result_?" can be queried with the following enum:
 * - \p da_rinfo - return an array of size 5 containing \p n_features, \p n_samples, the number of samples the tree was
 *   trained on, the value of the random seed used to fit the tree and the depth of the tree.
 */
da_status da_tree_fit_d(da_handle handle);
da_status da_tree_fit_s(da_handle handle);
/** \} */

/** \{
 * @brief Fit the decision forest defined in the @p handle.
 *
 * @rst
 * Compute the decision forest parameters given the data passed by :ref:`da_forest_set_training_data_? <da_forest_set_training_data>`.
 * Note that you can customize the model before using the fit function through the use of optional parameters,
 * see :ref:`this section <opts_decisiontrees>` for a list of available options.
 * @endrst
 *
 * @param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_decision_forest.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle
 *   initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_incompatible_options - some of the options set are incompatible with the model defined in \p handle.
 *        You can obtain further information using @ref da_handle_print_error_message.
 * - @ref da_status_memory_error - internal memory allocation encountered a problem.
 * - @ref da_status_internal_error - an unexpected error occurred.
 *
 * \post
 * After successful execution, \ref da_handle_get_result_s "da_handle_get_result_?" can be queried with the following enum:
 * - \p da_rinfo - return an array of size 5 containing \p n_features, \p n_samples, the number of samples the tree was
 *   trained on, the value of the random seed used by the RNG and \p n_tree, the total number of trees in the forest.
*/
da_status da_forest_fit_d(da_handle handle);
da_status da_forest_fit_s(da_handle handle);
/** \} */

/** \{
 * @brief Generate labels using fitted decision tree on a new set of data @p X_test.
 *
 * @rst
 * After a model has been fitted using :ref:`da_tree_fit_? <da_tree_fit>`, it can be used to generate predicted labels on new data. This
 * function returns the decision tree predictions in the array ``y_pred``.
 *
 * For each data point ``i``, ``y_pred[i]`` will contain the label of the most likely class
 * according to the decision tree,
 * ``X_test[i + j*ldx_test]`` should contain the feature ``j`` for observation ``i``.
 * @endrst
 *
 * @param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_decision_tree.
 * @param[in] n_samples - number of observations in \p X_test.
 * @param[in] n_features - number of features in \p X_test.
 * @param[in] X_test array containing \p n_samples  @f$\times@f$ \p n_features data matrix, in column-major format.
 * @param[in] ldx_test leading dimension of \p X_test.  Constraint: \p ldx_test @f$\ge@f$ \p n_samples.
 * @param[out] y_pred - array of size at least \p n_samples. On output, will contain the predicted class labels.
 * @return da_status
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle
 *   initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using
 *   @ref da_handle_print_error_message.
 * - @ref da_status_out_of_date - the model has not been trained yet.
 */
da_status da_tree_predict_d(da_handle handle, da_int n_samples, da_int n_features,
                            double *X_test, da_int ldx_test, da_int *y_pred);
da_status da_tree_predict_s(da_handle handle, da_int n_samples, da_int n_features,
                            float *X_test, da_int ldx_test, da_int *y_pred);
/** \} */

/** \{
 * @brief Generate class probabilities using fitted decision tree on a new set of data @p X_test.
 *
 * @rst
 * After a model has been fitted using :ref:`da_tree_fit_? <da_tree_fit>`, it can be used to generate predicted labels on new data. This
 * function returns the decision tree class probabilities in the array ``y_pred``.
 *
 * For each data point ``i``, and class ``j``, ``y_proba[i + j*ldy]`` will contain the class probability
 * according to the decision tree,
 * ``X_test[i + j*ldx_test]`` should contain the feature ``j`` for observation ``i``.
 * @endrst
 *
 * @param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_decision_tree.
 * @param[in] n_samples - number of observations in \p X_test.
 * @param[in] n_features - number of features in \p X_test.
 * @param[in] X_test array containing \p n_samples  @f$\times@f$ \p n_features data matrix, in column-major format.
 * @param[in] ldx_test leading dimension of \p X_test.  Constraint: \p ldx_test @f$\ge@f$ \p n_samples.
 * @param[out] y_proba - array of size at least \p n_samples @f$\times@f$ \p n_class . On output, will contain the predicted class probabilities.
 * @param[in] n_class - number of classes in \p y_proba.
 * @param[in] ldy leading dimension of \p y_proba.  Constraint: \p ldy @f$\ge@f$ \p n_samples.
 * @return da_status
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle
 *   initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using
 *   @ref da_handle_print_error_message.
 * - @ref da_status_out_of_date - the model has not been trained yet.
 */
da_status da_tree_predict_proba_d(da_handle handle, da_int n_samples, da_int n_features,
                                  double *X_test, da_int ldx_test, double *y_proba,
                                  da_int n_class, da_int ldy);
da_status da_tree_predict_proba_s(da_handle handle, da_int n_samples, da_int n_features,
                                  float *X_test, da_int ldx_test, float *y_proba,
                                  da_int n_class, da_int ldy);
/** \} */

/** \{
 * @brief Generate class log probabilities using fitted decision tree on a new set of data @p X_test.
 *
 * @rst
 * After a model has been fitted using :ref:`da_tree_fit_? <da_tree_fit>`, it can be used to generate predicted labels on new data. This
 * function returns the decision tree class log probabilities in the array ``y_pred``.
 *
 * For each data point ``i``, and class ``j``, ``y_proba[i + j*ldy]`` will contain the class log probability
 * according to the decision tree,
 * ``X_test[i + j*ldx_test]`` should contain the feature ``j`` for observation ``i``.
 * @endrst
 *
 * @param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_decision_tree.
 * @param[in] n_samples - number of observations in \p X_test.
 * @param[in] n_features - number of features in \p X_test.
 * @param[in] X_test array containing \p n_samples  @f$\times@f$ \p n_features data matrix, in column-major format.
 * @param[in] ldx_test leading dimension of \p X_test.  Constraint: \p ldx_test @f$\ge@f$ \p n_samples.
 * @param[out] y_log_proba - array of size at least \p n_samples @f$\times@f$ \p n_class . On output, will contain the predicted class log probabilities.
 * @param[in] n_class - number of classes in \p y_log_proba.
 * @param[in] ldy leading dimension of \p y_log_proba.  Constraint: \p ldy @f$\ge@f$ \p n_samples.
 * @return da_status
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle
 *   initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using
 *   @ref da_handle_print_error_message.
 * - @ref da_status_out_of_date - the model has not been trained yet.
 */
da_status da_tree_predict_log_proba_d(da_handle handle, da_int n_samples,
                                      da_int n_features, double *X_test, da_int ldx_test,
                                      double *y_log_proba, da_int n_class, da_int ldy);
da_status da_tree_predict_log_proba_s(da_handle handle, da_int n_samples,
                                      da_int n_features, float *X_test, da_int ldx_test,
                                      float *y_log_proba, da_int n_class, da_int ldy);
/** \} */

/** \{
 * @brief Generate labels using fitted decision forest on a new set of data @p X_test.
 *
 * @rst
 * After a model has been fitted using :ref:`forest_fit_? <da_forest_fit>`, it can be used to generate predicted labels on new data. This
 * function returns the decision forest predictions in the array ``y_pred``.
 *
 * For each data point ``i``, ``y_pred[i]`` will contain the label of the most likely class
 * according to the decision forest;
 * ``x[i + j*ldx_test]`` should contain the feature ``j`` for observation ``i``.
 * @endrst
 *
* @param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_decision_tree.
 * @param[in] n_samples - number of observations in \p X_test.
 * @param[in] n_features - number of features in \p X_test.
 * @param[in] X_test array containing \p n_samples  @f$\times@f$ \p n_features data matrix, in column-major format.
 * @param[in] ldx_test leading dimension of \p X_test.  Constraint: \p ldx_test @f$\ge@f$ \p n_samples.
 * @param[out] y_pred - array of size at least \p n_samples. On output, will contain the predicted class labels.
 * @return da_status
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle
 *   initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using
 *   @ref da_handle_print_error_message.
 * - @ref da_status_out_of_date - the model has not been trained yet.
 */
da_status da_forest_predict_d(da_handle handle, da_int n_samples, da_int n_features,
                              double *X_test, da_int ldx_test, da_int *y_pred);
da_status da_forest_predict_s(da_handle handle, da_int n_samples, da_int n_features,
                              float *X_test, da_int ldx_test, da_int *y_pred);
/** \} */

/** \{
 * @brief Generate class probabilities using fitted decision forest on a new set of data @p X_test.
 *
 * @rst
 * After a model has been fitted using :ref:`da_forest_fit_? <da_forest_fit>`, it can be used to generate predicted labels on new data. This
 * function returns the decision forest class probabilities in the array ``y_pred``.
 *
 * For each data point ``i``, and class ``j``, ``y_proba[i*n_class + j]`` will contain the class probability
 * according to the decision forest,
 * ``X_test[i + j*ldx_test]`` should contain the feature ``j`` for observation ``i``.
 * @endrst
 *
 * @param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_decision_forest.
 * @param[in] n_samples - number of observations in \p X_test.
 * @param[in] n_features - number of features in \p X_test.
 * @param[in] X_test array containing \p n_samples  @f$\times@f$ \p n_features data matrix, in column-major format.
 * @param[in] ldx_test leading dimension of \p X_test.  Constraint: \p ldx_test @f$\ge@f$ \p n_samples.
 * @param[out] y_proba - array of size at least \p n_samples @f$\times@f$ \p n_class . On output, will contain the predicted class probabilities.
 * @param[in] n_class - number of classes in \p y_proba.
 * @param[in] ldy leading dimension of \p y_proba.  Constraint: \p ldy @f$\ge@f$ \p n_samples.
 * @return da_status
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle
 *   initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using
 *   @ref da_handle_print_error_message.
 * - @ref da_status_out_of_date - the model has not been trained yet.
 */
da_status da_forest_predict_proba_d(da_handle handle, da_int n_samples, da_int n_features,
                                    double *X_test, da_int ldx_test, double *y_proba,
                                    da_int n_class, da_int ldy);
da_status da_forest_predict_proba_s(da_handle handle, da_int n_samples, da_int n_features,
                                    float *X_test, da_int ldx_test, float *y_proba,
                                    da_int n_class, da_int ldy);
/** \} */

/** \{
 * @brief Generate class log probabilities using fitted decision forest on a new set of data @p X_test.
 *
 * @rst
 * After a model has been fitted using :ref:`da_forest_fit_? <da_forest_fit>`, it can be used to generate predicted labels on new data. This
 * function returns the decision forest class log probabilities in the array ``y_pred``.
 *
 * For each data point ``i``, and class ``j``, ``y_proba[i*n_class + j]`` will contain the class log probability
 * according to the decision forest,
 * ``X_test[i + j*ldx_test]`` should contain the feature ``j`` for observation ``i``.
 * @endrst
 *
 * @param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_decision_forest.
 * @param[in] n_samples - number of observations in \p X_test.
 * @param[in] n_features - number of features in \p X_test.
 * @param[in] X_test array containing \p n_samples  @f$\times@f$ \p n_features data matrix, in column-major format.
 * @param[in] ldx_test leading dimension of \p X_test.  Constraint: \p ldx_test @f$\ge@f$ \p n_samples.
 * @param[out] y_log_proba - array of size at least \p n_samples @f$\times@f$ \p n_class . On output, will contain the predicted class log probabilities.
 * @param[in] n_class - number of classes in \p y_log_proba.
 * @param[in] ldy leading dimension of \p y_log_proba.  Constraint: \p ldy @f$\ge@f$ \p n_samples.
 * @return da_status
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle
 *   initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using
 *   @ref da_handle_print_error_message.
 * - @ref da_status_out_of_date - the model has not been trained yet.
 */
da_status da_forest_predict_log_proba_d(da_handle handle, da_int n_samples,
                                        da_int n_features, double *X_test,
                                        da_int ldx_test, double *y_log_proba,
                                        da_int n_class, da_int ldy);
da_status da_forest_predict_log_proba_s(da_handle handle, da_int n_samples,
                                        da_int n_features, float *X_test, da_int ldx_test,
                                        float *y_log_proba, da_int n_class, da_int ldy);
/** \} */

/** \{
 * @brief Calculate score (prediction accuracy) by comparing predicted labels and actual labels on a new set
 * of data @p X_test.
 *
 * @rst
 * To be used after a model has been fitted using :ref:`da_tree_fit_? <da_tree_fit>`.
 *
 * For each data point ``i``, ``y_test[i]`` will contain the label of the test data,
 * ``X_test[i + j*ldx_test]`` should contain the feature ``j`` for observation ``i``.
 * @endrst
 *
 * @param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_decision_tree.
 * @param[in] n_samples - number of observations in \p X_test.
 * @param[in] n_features - number of features in \p X_test. It must match the number of features from
 *                         the training data set.
 * @param[in] X_test array containing \p n_samples  @f$\times@f$ \p n_features data matrix, in column-major format.
 * @param[in] ldx_test leading dimension of \p X_test.  Constraint: \p ldx_test @f$\ge@f$ \p n_samples.
 * @param[in] y_test - actual class labels.
 * @param[out] mean_accuracy - proportion of observations where predicted label matches actual label.
 * @return da_status
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle
 *   initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using
 *   @ref da_handle_print_error_message.
 * - @ref da_status_out_of_date - the model has not been trained yet.
 */
da_status da_tree_score_d(da_handle handle, da_int n_samples, da_int n_features,
                          double *X_test, da_int ldx_test, da_int *y_test,
                          double *mean_accuracy);

da_status da_tree_score_s(da_handle handle, da_int n_samples, da_int n_features,
                          float *X_test, da_int ldx_test, da_int *y_test,
                          float *mean_accuracy);
/** \} */

/** \{
 * @brief Calculate score (prediction accuracy) by comparing predicted labels and actual labels on a new set
 * of data @p X_test.
 *
 * @rst
 * To be used after a model has been fitted using :ref:`da_forest_fit_? <da_forest_fit>`.
 *
 * For each data point ``i``, ``y_test[i]`` will contain the label of the test data,
 * ``X_test[i + j*ldx_test]`` should contain the feature ``j`` for observation ``i``.
 * @endrst
 *
 * @param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_decision_forest.
 * @param[in] n_samples - number of observations in \p X_test.
 * @param[in] n_features - number of features in \p X_test. It must match the number of features from
 *                         the training data set.
 * @param[in] X_test array containing \p n_samples  @f$\times@f$ \p n_features data matrix, in column-major format.
 * @param[in] ldx_test leading dimension of \p X_test.  Constraint: \p ldx_test @f$\ge@f$ \p n_samples.
 * @param[in] y_test - actual class labels.
 * @param[out] mean_accuracy - proportion of observations where predicted label matches actual label.
 * @return da_status
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle
 *   initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using
 *   @ref da_handle_print_error_message.
 * - @ref da_status_out_of_date - the model has not been trained yet.
 */
da_status da_forest_score_d(da_handle handle, da_int n_samples, da_int n_features,
                            double *X_test, da_int ldx_test, da_int *y_test,
                            double *mean_accuracy);
da_status da_forest_score_s(da_handle handle, da_int n_samples, da_int n_features,
                            float *X_test, da_int ldx_test, da_int *y_test,
                            float *mean_accuracy);

/** \} */

#endif
