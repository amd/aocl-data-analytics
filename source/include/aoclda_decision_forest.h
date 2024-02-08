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
 * @brief Pass a 2d feature matrix containing double precision data and a 1d label array to the \ref da_handle object
 * in preparation for fitting a decision forest.
 *
 * A copy of the training data is stored internally, to avoid overwriting the user's data during computation.
 *
 * @param handle a @ref da_handle object, initialized with type @ref da_handle_decision_forest.
 * @param n_obs number of observations in \p X
 * @param n_features number of features in \p X
 * @param X array containing \p n_obs  @f$\times@f$ \p n_features data matrix, in column-major format
 * @param ldx leading dimension of \p X.  Constraint: \p ldx @f$\ge@f$ \p n_obs.
 * @param y 1d array containing n_obs labels
 * @return @ref da_status.  The function returns:
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using @ref da_handle_print_error_message.
 */
da_status da_df_set_training_data_d(da_handle handle, da_int n_obs, da_int n_features,
                                    double *X, da_int ldx, uint8_t *y);

da_status da_df_set_training_data_s(da_handle handle, da_int n_obs, da_int n_features,
                                    float *X, da_int ldx, uint8_t *y);
/** \} */

/** \{
 * @brief Fit the decision forest defined in the @p handle.
 *
 * @rst
 * Compute the decision forest parameters given the data passed by :ref:`da_df_set_training_data_? <da_df_set_training_data>`.
 * Note that you can customize the model before using the fit function through the use of optional parameters,
 * see :ref:`this section <opts_decisionforest>` for a list of available options.
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
 * @rst
 * After successful execution, :ref:`da_handle_get_result_? <da_handle_get_result>` can be queried with the following enum:
 * @endrst
 * - \p da_rinfo - return an array of size 3 containing \p seed_val, \p n_obs and \p n_features.
 */
da_status da_df_fit_d(da_handle handle);

da_status da_df_fit_s(da_handle handle);
/** \} */

/** \{
 * @brief Generate labels using fitted decision forest on a new set of data @p x.
 *
 * @rst
 * After a model has been fit using :ref:`da_df_fit_? <da_df_fit>`, it can be used to generate predicted labels on new data. This
 * function returns the decision forest predictions in the array ``y_pred``.
 *
 * For each data point ``i``, ``y_pred[i]`` will contain the label of the most likely class
 * according to the decision forest,
 * ``x[i + j*ldx]`` should contain the feature ``j`` for observation ``i``.
 * @endrst
 *
 * @param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_decision_forest.
 * @param[in] n_obs - number of observations in \p X
 * @param[in] n_features - number of features in \p X
 * @param[in] X_test array containing \p n_obs  @f$\times@f$ \p n_features data matrix, in column-major format
 * @param[in] ldx_test leading dimension of \p X_test.  Constraint: \p ldx @f$\ge@f$ \p n_obs.
 * @param[out] y_pred - predicted class labels
 * @return da_status
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle
 *   initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using
 *   @ref da_handle_print_error_message.
 * - @ref da_status_out_of_date - the model has not been trained yet.
 */
da_status da_df_predict_d(da_handle handle, da_int n_obs, da_int n_features,
                          double *X_test, da_int ldx_test, uint8_t *y_pred);

da_status da_df_predict_s(da_handle handle, da_int n_obs, da_int n_features,
                          float *X_test, da_int ldx_test, uint8_t *y_pred);
/** \} */

/** \{
 * @brief Calculate score (prediction accuracy) by comparing predicted labels and actual labels on a new set
 * of data @p x_test.
 *
 * @rst
 * To be used after a model has been fit using :ref:`da_df_fit_? <da_df_fit>`.
 *
 * For each data point ``i``, ``y_test[i]`` will contain the actual label,
 * ``x[i + j*ldx]`` should contain the feature ``j`` for observation ``i``.
 * @endrst
 *
 * @param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_decision_forest.
 * @param[in] n_obs - number of observations in \p X
 * @param[in] n_features - number of features in \p X
 * @param[in] X_test array containing \p n_obs  @f$\times@f$ \p n_features data matrix, in column-major format
 * @param[in] ldx_test leading dimension of \p X_test.  Constraint: \p ldx @f$\ge@f$ \p n_obs.
 * @param[in] y_test - actual class labels
 * @param[out] score - proportion of observations where predicted label matches actual label
 * @return da_status
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle
 *   initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using
 *   @ref da_handle_print_error_message.
 * - @ref da_status_out_of_date - the model has not been trained yet.
 */
da_status da_df_score_d(da_handle handle, da_int n_obs, da_int n_features, double *X_test,
                        da_int ldx_test, uint8_t *y_test, double *score);

da_status da_df_score_s(da_handle handle, da_int n_obs, da_int n_features, float *X_test,
                        da_int ldx_test, uint8_t *y_test, float *score);
/** \} */

#endif
