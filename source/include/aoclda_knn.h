/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef AOCLDA_kNN
#define AOCLDA_kNN

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * \file
 */

/**
 * \brief Defines which algorithm is used to compute the <i>k</i>-nearest neighbors.
 **/
enum da_knn_algorithm_ {
    da_brute_force ///< Use Brute Force.
};

/** @brief Alias for the \ref da_knn_algorithm_ enum. */
typedef enum da_knn_algorithm_ da_knn_algorithm;

/**
 * \brief Sets the weight function used to compute the <i>k</i>-nearest neighbors.
 **/
enum da_knn_weights_ {
    da_knn_uniform, ///< Use uniform weights.
    da_knn_distance ///< Weight points by the inverse of their distance.
};

/** @brief Alias for the \ref da_knn_weights_ enum. */
typedef enum da_knn_weights_ da_knn_weights;

/** \{
 * \brief Pass a data matrix and a label array to the \ref da_handle object
 * in preparation of computing a <i>k</i>-NN.
 *
 * The data itself is not copied; a pointer to the data matrix is stored instead.
 * @rst
 * After calling this function you may use the option setting APIs to set :ref:`options <knn_options>`.
 * @endrst

 * \param[in,out] handle a \ref da_handle object, initialized with type \ref da_handle_knn.
 * \param[in] n_samples number of observations in \p X_train.
 * \param[in] n_features number of features in \p X_train.
 * \param[in] X_train array containing \p n_samples  @f$\times@f$ \p n_features data matrix, in column-major format.
 * \param[in] ldx_train leading dimension of \p X_train.  Constraint: \p ldx_train @f$\ge@f$ \p n_samples.
 * \param[in] y_train array containing the \p n_samples labels.
 * \return \ref da_status.  The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 * - \ref da_status_invalid_pointer - the @p handle has not been correctly initialized, or \p X_train or \p y_train are invalid.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_memory_error - internal memory allocation encountered a problem.
*/
da_status da_knn_set_training_data_d(da_handle handle, da_int n_samples,
                                     da_int n_features, const double *X_train,
                                     da_int ldx_train, const da_int *y_train);
da_status da_knn_set_training_data_s(da_handle handle, da_int n_samples,
                                     da_int n_features, const float *X_train,
                                     da_int ldx_train, const da_int *y_train);
/** \} */

/** \{
 * \brief Compute <i>k</i>-Nearest Neighbors (<i>k</i>-NN)
 *
 * @rst
 * Compute the *k*-NN of a test data :math:`X_{test}` with respect to the data matrix
 * previously passed into the handle using :ref:`da_knn_set_training_data_? <da_knn_set_training_data>`.
 * @endrst
 *
 * \param[in,out] handle a \ref da_handle object, initialized with type \ref da_handle_knn.
 * \param[in] n_queries number of observations in \p X_test.
 * \param[in] n_features number of features in \p X_test. Constraint: \p n_features @f$=@f$ the number of features in the data matrix originally supplied to \ref da_knn_set_training_data_s "da_knn_set_training_data_?".
 * \param[in] X_test array containing \p n_queries  @f$\times@f$ \p n_features data matrix, in column-major format.
 * \param[in] ldx_test leading dimension of \p X_test.  Constraint: \p ldx_test @f$\ge@f$ \p n_queries.
 * \param[out] n_ind array containing the \p n_queries @f$\times@f$ \p k matrix, with the indices of the \p k - nearest neighbors of the test data \p X_test. If \p k @f$\le@f$ 0, the number of neighbors passed during the option setting will be used instead.
 * \param[out] n_dist array containing the corresponding distances to the neighbors whose indices are stored in \p n_ind, if \p return_distance is 1.
 * \param[in] k number of nearest neighbors requested. If \p k @f$\le@f$ 0, the number of neighbors passed during the option setting will be used instead. Constraint: If \p k @f$\le@f$ \p n_features.
 * \param[in] return_distance denotes if the distances to the <i>k</i>-NN need be computed. If \p return_distance is 1, the distances are returned.
 * \return \ref da_status.  The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 * - \ref da_status_invalid_pointer - the @p handle has not been correctly initialized, or \p X_test or \p n_ind or \p n_dist is invalid.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_memory_error - internal memory allocation encountered a problem.
*/
da_status da_knn_kneighbors_d(da_handle handle, da_int n_queries, da_int n_features,
                              const double *X_test, da_int ldx_test, da_int *n_ind,
                              double *n_dist, da_int k, da_int return_distance);
da_status da_knn_kneighbors_s(da_handle handle, da_int n_queries, da_int n_features,
                              const float *X_test, da_int ldx_test, da_int *n_ind,
                              float *n_dist, da_int k, da_int return_distance);
/** \} */

/** \{
 * \brief Get number of the distinct class labels
 *
 * @rst
 * Request the number of different class labels provided in the training data so that
 * memory is allocated correctly for computing the class probabilities of a test data set.
 * @endrst
 *
 * \param[in,out] handle a \ref da_handle object, initialized with type \ref da_handle_knn.
 * \param[in,out] n_classes the number of different classes. If \p n_classes @f$\le@f$ 0, the number of different classes will be returned and \p classes will not be referenced.
 * \param[out] classes ordered array that holds the different classes.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using the wrong precision.
 * - \ref da_status_invalid_pointer - the handle has not been initialized, or one of the arrays is null.
 * - \ref da_status_no_data - the <i>k</i>-nearest neighbors have not been computed prior to this function call.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
*/
da_status da_knn_classes_d(da_handle handle, da_int *n_classes, da_int *classes);
da_status da_knn_classes_s(da_handle handle, da_int *n_classes, da_int *classes);
/** \} */

/** \{
 * \brief Compute probability estimates using <i>k</i>-Nearest Neighbors
 *
 * @rst
 * Compute the probability estimates for the different classes based on the *k*-NN of a test data :math:`X_{test}` with respect to the data matrix
 * data matrix previously passed into the handle using :ref:`da_knn_set_training_data_? <da_knn_set_training_data>`.
 * @endrst
 *
 * \param[in,out] handle a \ref da_handle object, initialized with type \ref da_handle_knn.
 * \param[in] n_queries number of observations in \p X_test.
 * \param[in] n_features number of features in \p X_test. Constraint: \p n_features @f$=@f$ the number of features in the data matrix originally supplied to \ref da_knn_set_training_data_s "da_knn_set_training_data_?".
 * \param[in] X_test array containing \p n_queries  @f$\times@f$ \p n_features data matrix, in column-major format.
 * \param[in] ldx_test leading dimension of \p X_test.  Constraint: \p ldx_test @f$\ge@f$ \p n_queries.
 * \param[out] proba array of size \p n_queries  @f$\times@f$ \p n_classes containing the probability estimates for each of the available classes.
 * \return \ref da_status.  The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 * - \ref da_status_invalid_pointer - the @p handle has not been correctly initialized, or \p X_test or \p n_ind or \p n_dist is invalid.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_memory_error - internal memory allocation encountered a problem.
*/
da_status da_knn_predict_proba_d(da_handle handle, da_int n_queries, da_int n_features,
                                 const double *X_test, da_int ldx_test, double *proba);
da_status da_knn_predict_proba_s(da_handle handle, da_int n_queries, da_int n_features,
                                 const float *X_test, da_int ldx_test, float *proba);
/** \} */

/** \{
 * \brief Compute estimated labels of a data set using <i>k</i>-Nearest Neighbors
 *
 * @rst
 * Compute the estimated labels based on the *k*-NN of a test data :math:`X_{test}` with respect to the data matrix
 * data matrix previously passed into the handle using :ref:`da_knn_set_training_data_? <da_knn_set_training_data>`.
 * @endrst
 *
 * \param[in,out] handle a \ref da_handle object, initialized with type \ref da_handle_knn.
 * \param[in] n_queries number of observations in \p X_test.
 * \param[in] n_features number of features in \p X_test. Constraint: \p n_features @f$=@f$ the number of features in the data matrix originally supplied to \ref da_knn_set_training_data_s "da_knn_set_training_data_?".
 * \param[in] X_test array containing \p n_queries  @f$\times@f$ \p n_features data matrix, in column-major format.
 * \param[in] ldx_test leading dimension of \p X_test.  Constraint: \p ldx_test @f$\ge@f$ \p n_queries.
 * \param[out] y_test array of size \p n_queries containing the estimated label for each query.
 * \return \ref da_status.  The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 * - \ref da_status_invalid_pointer - the @p handle has not been correctly initialized, or \p X_test or \p n_ind or \p n_dist is invalid.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_memory_error - internal memory allocation encountered a problem.
*/
da_status da_knn_predict_d(da_handle handle, da_int n_queries, da_int n_features,
                           const double *X_test, da_int ldx_test, da_int *y_test);
da_status da_knn_predict_s(da_handle handle, da_int n_queries, da_int n_features,
                           const float *X_test, da_int ldx_test, da_int *y_test);
/** \} */

#ifdef __cplusplus
}
#endif

#endif
