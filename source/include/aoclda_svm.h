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

#ifndef AOCLDA_SVM
#define AOCLDA_SVM

/**
 * \file
 */

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"

/**
 * \brief Defines which SVM model is computed
 **/
typedef enum da_svm_model_ {
    svm_undefined = 0, ///< No svm model set.
    svc,               ///< C regularized classification.
    nusvc,             ///< Nu regularized classification.
    svr,               ///< Eps regularized regression.
    nusvr              ///< Nu regularized regression.
} da_svm_model;

/**
 * \brief Defines which shape of decision function is computed
 **/
typedef enum da_svm_decision_function_shape_ {
    ovr = 0, ///< One-vs-Rest.
    ovo      ///< One-vs-One.
} da_svm_decision_function_shape;

/** \{
 * @brief Select which SVM model to solve.
 * @rst
 * The model definition can be further enhanced by setting up optional parameters like regularization parameters and kernel functions.
 * See the :ref:`svm options section <svm_options>` for more information.
 * @endrst
 *
 * @param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_svm.
 * @param[in] mod a @ref da_svm_model enum type to select the SVM model.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_unknown_query - unknown @ref da_svm_model has been selected.
 */
da_status da_svm_select_model_d(da_handle handle, da_svm_model mod);
da_status da_svm_select_model_s(da_handle handle, da_svm_model mod);
/** \} */

/** \{
 * @brief Define the data to train an SVM model.

 * Pass pointers to a data matrix @p X containing @p n_samples observations (rows) over @p n_features features (columns)
 * and a response vector @p y of size @p n_samples.
 *
 * Only the pointers to @p X and @p y are stored; no internal copy is made.
 *
 * @param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_svm.
 * @param[in] n_samples the number of observations (rows) of the data matrix @p X. Constraint: @p n_samples @f$\ge@f$ 1.
 * @param[in] n_features the number of features (columns) of the data matrix, @p X. Constraint: @p n_features @f$\ge@f$ 1.
 * @param[in] X the @p n_samples @f$\times@f$ @p n_features data matrix. By default, it should be stored in column-major order, unless you have set the <em>storage order</em> option to <em>row-major</em>.
 * @param[in] ldx_train the leading dimension of @p X. Constraint: \p ldx_train @f$\ge@f$ \p n_samples if \p X is stored in column-major order, or \p ldx_train @f$\ge@f$ \p n_features if \p X is stored in row-major order.
 * @param[in] y the response vector, of size @p n_samples. The label values are expected to range from 0 to \p n_class - 1.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 * - @ref da_status_invalid_handle_type - the @p handle has not been correctly initialized.
 * - @ref da_status_unknown_query - unknown SVM model has been selected. You can obtain further information using @ref da_handle_print_error_message.
 * - \ref da_status_invalid_leading_dimension - the constraint on \p lda was violated.
 */
da_status da_svm_set_data_d(da_handle handle, da_int n_samples, da_int n_features,
                            const double *X, da_int ldx_train, const double *y);
da_status da_svm_set_data_s(da_handle handle, da_int n_samples, da_int n_features,
                            const float *X, da_int ldx_train, const float *y);
/** \} */

/** \{
 * @brief Fit the SVM model defined in the @p handle.
 *
 * Compute the SVM model defined by \ref da_svm_select_model_s "da_svm_select_model_?" on the data passed by the last call to the function \ref da_svm_set_data_s "da_svm_set_data_?".
 * 
 * @param[in,out] handle a @ref da_handle object, initialized with type @ref da_handle_svm.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successfully completed.
 * - @ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 * - @ref da_status_no_data - \ref da_svm_set_data_s "da_svm_set_data_?" has not been called prior to this function call.
 * - @ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - @ref da_status_invalid_option - some of the options set have invalid values.
 *        You can obtain further information using @ref da_handle_print_error_message.
 * - @ref da_status_internal_error - an unexpected error occurred.

 *
 * \post
 * \parblock
 * After successful execution, \ref da_handle_get_result_s "da_handle_get_result_?" can be queried with the following enum for floating-point output:
 * - \p da_rinfo - return an array of size 3 containing the values of \p n_samples, \p n_features, \p n_class.
 *
 * In addition \ref da_handle_get_result_int can be queried with the following enums:
 * - \p da_svm_dual_coef - return an array of size \p n_class-1 @f$\times@f$ \p n_support containing the dual coefficients of the support vectors.
 * - \p da_svm_support_vectors - return an array of size \p n_support @f$\times@f$ \p n_features containing the support vectors.
 * - \p da_svm_bias - return an array of size \p n_class-1 containing the bias terms.
 * - \p da_svm_n_support_vectors - return a total number of support vectors (integer).
 * - \p da_svm_n_support_vectors_per_class - return an array of size \p n_class containing the number of support vectors per class.
 * - \p da_svm_idx_support_vectors - return an array of size \p n_support containing the indexes of the support vectors.
 * \endparblock
 */
da_status da_svm_compute_d(da_handle handle);
da_status da_svm_compute_s(da_handle handle);
/** \} */

/** \{
 * @brief Predict labels (or outputs) using the previously fitted SVM model.
 *
 * Predicts the labels (classification) or values (regression) for the given data matrix @p X_test, storing results in @p predictions.
 *
 * @param[in,out] handle a @ref da_handle object, with type @ref da_handle_svm and a model already computed via \ref da_svm_compute_s "da_svm_compute_?".
 * @param[in] n_samples number of observations (rows) in the data matrix @p X_test. Constraint: @p n_samples @f$\ge@f$ 1.
 * @param[in] n_features number of features (columns) of the data matrix @p X_test. Constraint: @p n_features has to be the same as in \ref da_svm_set_data_s "da_svm_set_data_?".
 * @param[in] X_test the @p n_samples @f$\times@f$ @p n_features data matrix. By default, it should be stored in column-major order, unless you have set the <em>storage order</em> option to <em>row-major</em>.
 * @param[in] ldx_test the leading dimension of @p X_test. Constraint: \p ldx_test @f$\ge@f$ \p n_samples if \p X_test is stored in column-major order, or \p ldx_test @f$\ge@f$ \p n_features if \p X_test is stored in row-major order.
 * @param[out] predictions the array of size @p n_samples to store predicted outcomes.
 * @return @ref da_status. Possible returns:
 * - @ref da_status_success - operation was successfully completed.
 * - @ref da_status_wrong_type - floating point precision is incompatible with the handle initialization.
 * - @ref da_status_invalid_pointer - the handle was not properly initialized with SVM or one of the arrays is null.
 * - @ref da_status_out_of_date - the model was not computed prior to this function call.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value.
 * - @ref da_status_invalid_leading_dimension - the constraint on \p ldx_test was violated.
 */
da_status da_svm_predict_d(da_handle handle, da_int n_samples, da_int n_features,
                           const double *X_test, da_int ldx_test, double *predictions);

da_status da_svm_predict_s(da_handle handle, da_int n_samples, da_int n_features,
                           const float *X_test, da_int ldx_test, float *predictions);
/** \} */

/** \{
 * @brief Compute the decision function for each sample in @p X_test using the SVM model.
 *
 * The decision values are stored in @p decision_values, and its leading dimension is given by @p ldd. The parameter @p shape selects
 * the decision function shape for multi-class classification. Note that One-vs-Rest decision values are constructed from One-vs-One values.
 * The function is defined only for classification problems (for regression simply use \ref da_svm_predict_s "da_svm_predict_?").
 *
 * @param[in,out] handle a @ref da_handle object, with type @ref da_handle_svm and a model already computed via \ref da_svm_compute_s "da_svm_compute_?".
 * @param[in] n_samples number of observations (rows) in the data matrix @p X_test. Constraint: @p n_samples @f$\ge@f$ 1.
 * @param[in] n_features number of features (columns) of the data matrix @p X_test. Constraint: @p n_features has to be the same as in \ref da_svm_set_data_s "da_svm_set_data_?".
 * @param[in] X_test the @p n_samples @f$\times@f$ @p n_features data matrix. By default, it should be stored in column-major order, unless you have set the <em>storage order</em> option to <em>row-major</em>.
 * @param[in] ldx_test leading dimension of @p X_test. Constraint: \p ldx_test @f$\ge@f$ \p n_samples if \p X_test is stored in column-major order, or \p ldx_test @f$\ge@f$ \p n_features if \p X_test is stored in row-major order.
 * @param[out] decision_values array to store the decision function output. Must be of size at least @p n_samples @f$\times@f$ \p n_class if shape is one-vs-rest or \p n_class*(n_class-1)/2 if \p shape is one-vs-one.
 * @param[in] ldd leading dimension of @p decision_values. Constraint: \p ldd @f$\ge@f$ \p n_samples if \p decision_values is stored in column-major order, or if \p decision_values is stored in row-major order then \p ldd @f$\ge@f$ \p n_class if \p shape is one-vs-rest and \p ldd @f$\ge@f$ \p n_class*(n_class-1)/2 if \p shape is one-vs-one.
 * @param[in] shape a @ref da_svm_decision_function_shape enum specifying either one-vs-rest or one-vs-one, only used in multi-class classification.
 * @return @ref da_status. Possible returns:
 * - @ref da_status_success - operation was successfully completed.
 * - @ref da_status_wrong_type - floating point precision is incompatible with the handle initialization.
 * - @ref da_status_invalid_pointer - the handle was not properly initialized with SVM or one of the arrays is null.
 * - @ref da_status_out_of_date - the model was not computed prior to this function call.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value.
 * - @ref da_status_invalid_leading_dimension - the constraint on \p ldx_test or \p ldd was violated.
 */
da_status da_svm_decision_function_d(da_handle handle, da_int n_samples,
                                     da_int n_features, const double *X_test,
                                     da_int ldx_test, double *decision_values, da_int ldd,
                                     da_svm_decision_function_shape shape);

da_status da_svm_decision_function_s(da_handle handle, da_int n_samples,
                                     da_int n_features, const float *X_test,
                                     da_int ldx_test, float *decision_values, da_int ldd,
                                     da_svm_decision_function_shape shape);
/** \} */

/** \{
 * @brief Evaluate the quality of the predictions on the dataset @p X_test against the true labels @p y_test.
 *
 * Computes a scalar score (accuracy for classification, @f$R^2@f$ for regression) and stores it in @p score.
 *
 * @param[in,out] handle a @ref da_handle object, with type @ref da_handle_svm and a model already computed via \ref da_svm_compute_s "da_svm_compute_?".
 * @param[in] n_samples the number of observations (rows) in the data matrix @p X_test. Constraint: @p n_samples @f$\ge@f$ 1.
 * @param[in] n_features the number of features (columns) of the data matrix @p X_test. Constraint: @p n_features has to be the same as in \ref da_svm_set_data_s "da_svm_set_data_?".
 * @param[in] X_test the @p n_samples @f$\times@f$ @p n_features data matrix. By default, it should be stored in column-major order, unless you have set the <em>storage order</em> option to <em>row-major</em>.
 * @param[in] ldx_test the leading dimension of @p X_test. Constraint: \p ldx_test @f$\ge@f$ \p n_samples if \p X_test is stored in column-major order, or \p ldx_test @f$\ge@f$ \p n_features if \p X_test is stored in row-major order.
 * @param[in] y_test the true labels or responses, of size @p n_samples.
 * @param[out] score pointer to a single scalar value in which the score is stored.
 * @return @ref da_status. Possible returns:
 * - @ref da_status_success - operation was successfully completed.
 * - @ref da_status_wrong_type - floating point precision is incompatible with the handle initialization.
 * - @ref da_status_invalid_pointer - the handle was not properly initialized with SVM or one of the arrays is null.
 * - @ref da_status_out_of_date - the model was not computed prior to this function call.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value.
 * - @ref da_status_invalid_leading_dimension - the constraint on \p ldx_test was violated.
 */
da_status da_svm_score_d(da_handle handle, da_int n_samples, da_int n_features,
                         const double *X_test, da_int ldx_test, const double *y_test,
                         double *score);

da_status da_svm_score_s(da_handle handle, da_int n_samples, da_int n_features,
                         const float *X_test, da_int ldx_test, const float *y_test,
                         float *score);
/** \} */

#endif
