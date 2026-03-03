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

#ifndef AOCLDA_NEAREST_NEIGHBORS_OPTIONS
#define AOCLDA_NEAREST_NEIGHBORS_OPTIONS

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
 * \brief Enumeration used to set the search mode for nearest neighbors.
 **/
enum da_nn_search_mode_ {
    knn_search_mode = 0,   ///< Use k-nearest neighbors for prediction.
    radius_search_mode = 1 ///< Use radius neighbors for prediction.
};

/** @brief Alias for the \ref da_nn_search_mode_ enum. */
typedef enum da_nn_search_mode_ da_nn_search_mode;

/** \{
 * \brief Pass a data matrix to the \ref da_handle object
 * in preparation for computing a <i>k</i>-NN.
 *
 * The data itself is not copied; a pointer to the data matrix is stored instead.
 * @rst
 * This function must be called after using the option setting APIs to set :ref:`options <nn_options>`, since the options are required in case of a k-d tree algorithm.
 * @endrst

 * \param[inout] handle a \ref da_handle object, initialized with type \ref da_handle_nn.
 * \param[in] n_samples number of observations in \p X_train.
 * \param[in] n_features number of features in \p X_train.
 * \param[in] X_train array containing \p n_samples  @f$\times@f$ \p n_features data matrix. By default, it should be stored in column-major order, unless you have set the <em>storage order</em> option to <em>row-major</em>.
 * \param[in] ldx_train leading dimension of \p X_train. Constraint: \p ldx_train @f$\ge@f$ \p n_samples if \p X_train is stored in column-major order, or \p ldx_train @f$\ge@f$ \p n_features if \p X_train is stored in row-major order.
 * \return \ref da_status.  The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 * - \ref da_status_invalid_pointer - the @p handle has not been correctly initialized, or \p X_train is invalid.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_memory_error - internal memory allocation encountered a problem.
 * - \ref da_status_invalid_leading_dimension - the constraint on \p ldx_train was violated.
*/
da_status da_nn_set_data_d(da_handle handle, da_int n_samples, da_int n_features,
                           const double *X_train, da_int ldx_train);
da_status da_nn_set_data_s(da_handle handle, da_int n_samples, da_int n_features,
                           const float *X_train, da_int ldx_train);
/** \} */

/** \{
 * \brief Pass classification labels to the \ref da_handle object
 * in preparation for computing a <i>k</i>-NN classification.
 *
 * The labels are not copied; a pointer to the label array is stored instead.

 * \param[inout] handle a \ref da_handle object, initialized with type \ref da_handle_nn.
 * \param[in] n_samples number of labels in \p y_train. Constraint: must match the n_samples from \ref da_nn_set_data_s "da_nn_set_data_?".
 * \param[in] y_train array containing the \p n_samples classification labels.
 * \return \ref da_status.  The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized with the wrong precision.
 * - \ref da_status_invalid_pointer - the @p handle has not been correctly initialized, or \p y_train is invalid.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
*/
da_status da_nn_set_labels_s(da_handle handle, da_int n_samples, const da_int *y_train);
da_status da_nn_set_labels_d(da_handle handle, da_int n_samples, const da_int *y_train);
/** \} */

/** \{
 * \brief Pass regression targets to the \ref da_handle object
 * in preparation for computing a <i>k</i>-NN regression.
 *
 * The targets are not copied; a pointer to the target array is stored instead.

 * \param[inout] handle a \ref da_handle object, initialized with type \ref da_handle_nn.
 * \param[in] n_samples number of targets in \p y_train. Constraint: must match the n_samples from \ref da_nn_set_data_s "da_nn_set_data_?".
 * \param[in] y_train array containing the \p n_samples regression targets.
 * \return \ref da_status.  The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized with the wrong precision.
 * - \ref da_status_invalid_pointer - the @p handle has not been correctly initialized, or \p y_train is invalid.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
*/
da_status da_nn_set_targets_d(da_handle handle, da_int n_samples, const double *y_train);
da_status da_nn_set_targets_s(da_handle handle, da_int n_samples, const float *y_train);
/** \} */

/** \{
 * \brief Compute <i>k</i>-Nearest Neighbors (<i>k</i>-NN)
 *
 * @rst
 * Compute the *k*-NN of a test data :math:`X_{test}` with respect to the data matrix
 * previously passed into the handle using :ref:`da_nn_set_data_? <da_nn_set_data>`.
 * @endrst
 *
 * \param[inout] handle a \ref da_handle object, initialized with type \ref da_handle_nn.
 * \param[in] n_queries number of observations in \p X_test.
 * \param[in] n_features number of features in \p X_test. Constraint: \p n_features @f$=@f$ the number of features in the data matrix originally supplied to \ref da_nn_set_data_s "da_nn_set_data_?".
 * \param[in] X_test array containing \p n_queries  @f$\times@f$ \p n_features data matrix, in the same storage format used to set the training data.
 * \param[in] ldx_test leading dimension of \p X_test. Constraint: \p ldx_test @f$\ge@f$ \p n_queries if \p X_test is stored in column-major order, or \p ldx_test @f$\ge@f$ \p n_features if \p X_test is stored in row-major order.
 * \param[out] n_ind array containing the \p n_queries @f$\times@f$ \p k matrix, with the indices of the \p k - nearest neighbors of the test data \p X_test. If \p k @f$\le@f$ 0, the number of neighbors passed during the option setting will be used instead.
 * \param[out] n_dist array containing the corresponding distances to the neighbors whose indices are stored in \p n_ind, if \p return_distance is 1.
 * \param[in] k number of nearest neighbors requested. If \p k @f$\le@f$ 0, the number of neighbors passed during the option setting will be used instead. Constraint: If \p k @f$\le@f$ \p n_features.
 * \param[in] return_distance denotes if the distances to the <i>k</i>-NN need be computed. If \p return_distance is 1, the distances are returned.
 * \return \ref da_status.  The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized with the wrong precision.
 * - \ref da_status_invalid_pointer - the @p handle has not been correctly initialized, or \p X_test or \p n_ind or \p n_dist is invalid.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_memory_error - internal memory allocation encountered a problem.
 * - \ref da_status_invalid_leading_dimension - the constraint on \p ldx_test was violated.
*/
da_status da_nn_kneighbors_d(da_handle handle, da_int n_queries, da_int n_features,
                             const double *X_test, da_int ldx_test, da_int *n_ind,
                             double *n_dist, da_int k, da_int return_distance);
da_status da_nn_kneighbors_s(da_handle handle, da_int n_queries, da_int n_features,
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
 * \param[inout] handle a \ref da_handle object, initialized with type \ref da_handle_nn.
 * \param[inout] n_classes the number of different classes. If \p n_classes @f$\le@f$ 0, the number of different classes will be returned and \p classes will not be referenced.
 * \param[out] classes ordered array that holds the different classes.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized with the wrong precision.
 * - \ref da_status_invalid_pointer - the handle has not been initialized, or one of the arrays is null.
 * - \ref da_status_no_data - the <i>k</i>-nearest neighbors have not been computed prior to this function call.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
*/
da_status da_nn_classes_d(da_handle handle, da_int *n_classes, da_int *classes);
da_status da_nn_classes_s(da_handle handle, da_int *n_classes, da_int *classes);
/** \} */

/** \{
 * \brief Compute probability estimates using <i>k</i>-Nearest Neighbors
 *
 * @rst
 * Compute the probability estimates for the different classes based on the *k*-NN of a test data :math:`X_{test}` with respect to the data matrix
 * data matrix previously passed into the handle using :ref:`da_nn_set_data_? <da_nn_set_data>`.
 * @endrst
 *
 * \param[inout] handle a \ref da_handle object, initialized with type \ref da_handle_nn.
 * \param[in] n_queries number of observations in \p X_test.
 * \param[in] n_features number of features in \p X_test. Constraint: \p n_features @f$=@f$ the number of features in the data matrix originally supplied to \ref da_nn_set_data_s "da_nn_set_data_?".
 * \param[in] X_test array containing \p n_queries  @f$\times@f$ \p n_features data matrix, in the same storage format used to fit the model.
 * \param[in] ldx_test leading dimension of \p X_test.  Constraint: \p ldx_test @f$\ge@f$ \p n_queries if \p X_test is stored in column-major order, or \p ldx_test @f$\ge@f$ \p n_features if \p X_test is stored in row-major order.
 * \param[out] proba array of size \p n_queries  @f$\times@f$ \p n_classes containing the probability estimates for each of the available classes.
 * \param[in] search_mode the search mode to use and is of \ref da_nn_search_mode enum type. E.g., \ref knn_search_mode or \ref radius_search_mode.
 * \return \ref da_status.  The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized with the wrong precision.
 * - \ref da_status_invalid_pointer - the @p handle has not been correctly initialized, or \p X_test or \p n_ind or \p n_dist is invalid.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_memory_error - internal memory allocation encountered a problem.
 * - \ref da_status_invalid_leading_dimension - the constraint on \p ldx_test was violated.
*/
da_status da_nn_classifier_predict_proba_d(da_handle handle, da_int n_queries,
                                           da_int n_features, const double *X_test,
                                           da_int ldx_test, double *proba,
                                           da_nn_search_mode search_mode);
da_status da_nn_classifier_predict_proba_s(da_handle handle, da_int n_queries,
                                           da_int n_features, const float *X_test,
                                           da_int ldx_test, float *proba,
                                           da_nn_search_mode search_mode);
/** \} */

/** \{
 * \brief Compute estimated labels of a data set using <i>k</i>-Nearest Neighbors
 *
 * @rst
 * Compute the estimated labels based on the *k*-NN of a test data :math:`X_{test}` with respect to the data matrix
 * data matrix previously passed into the handle using :ref:`da_nn_set_data_? <da_nn_set_data>`.
 * @endrst
 *
 * \param[inout] handle a \ref da_handle object, initialized with type \ref da_handle_nn.
 * \param[in] n_queries number of observations in \p X_test.
 * \param[in] n_features number of features in \p X_test. Constraint: \p n_features @f$=@f$ the number of features in the data matrix originally supplied to \ref da_nn_set_data_s "da_nn_set_data_?".
 * \param[in] X_test array containing \p n_queries  @f$\times@f$ \p n_features data matrix, in the same storage format used to fit the model.
 * \param[in] ldx_test leading dimension of \p X_test.  Constraint: \p ldx_test @f$\ge@f$ \p n_queries if \p X_test is stored in column-major order, or \p ldx_test @f$\ge@f$ \p n_features if \p X_test is stored in row-major order.
 * \param[out] y_test array of size \p n_queries containing the estimated label for each query.
 * \param[in] search_mode the search mode to use and is of \ref da_nn_search_mode enum type. E.g., \ref knn_search_mode or \ref radius_search_mode.
 * \return \ref da_status.  The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized with the wrong precision.
 * - \ref da_status_invalid_pointer - the @p handle has not been correctly initialized, or \p X_test or \p n_ind or \p n_dist is invalid.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_memory_error - internal memory allocation encountered a problem.
 * - \ref da_status_invalid_leading_dimension - the constraint on \p ldx_test was violated.
*/
da_status da_nn_classifier_predict_d(da_handle handle, da_int n_queries,
                                     da_int n_features, const double *X_test,
                                     da_int ldx_test, da_int *y_test,
                                     da_nn_search_mode search_mode);
da_status da_nn_classifier_predict_s(da_handle handle, da_int n_queries,
                                     da_int n_features, const float *X_test,
                                     da_int ldx_test, da_int *y_test,
                                     da_nn_search_mode search_mode);
/** \} */

/** \{
 * \brief Compute estimated target values of a data set using <i>k</i>-Nearest Neighbors
 *
 * @rst
 * Compute the estimated target values based on the *k*-NN of a test data :math:`X_{test}` with respect to the data matrix
 * data matrix previously passed into the handle using :ref:`da_nn_set_data_? <da_nn_set_data>`.
 * @endrst
 *
 * \param[inout] handle a \ref da_handle object, initialized with type \ref da_handle_nn.
 * \param[in] n_queries number of observations in \p X_test.
 * \param[in] n_features number of features in \p X_test. Constraint: \p n_features @f$=@f$ the number of features in the data matrix originally supplied to \ref da_nn_set_data_s "da_nn_set_data_?".
 * \param[in] X_test array containing \p n_queries  @f$\times@f$ \p n_features data matrix, in the same storage format used to fit the model.
 * \param[in] ldx_test leading dimension of \p X_test.  Constraint: \p ldx_test @f$\ge@f$ \p n_queries if \p X_test is stored in column-major order, or \p ldx_test @f$\ge@f$ \p n_features if \p X_test is stored in row-major order.
 * \param[out] y_test array of size \p n_queries containing the estimated target value for each query.
 * \param[in] search_mode the search mode to use and is of \ref da_nn_search_mode enum type. E.g., \ref knn_search_mode or \ref radius_search_mode.
 * \return \ref da_status.  The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized with the wrong precision.
 * - \ref da_status_invalid_pointer - the @p handle has not been correctly initialized, or \p X_test or \p n_ind or \p n_dist is invalid.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_memory_error - internal memory allocation encountered a problem.
 * - \ref da_status_invalid_leading_dimension - the constraint on \p ldx_test was violated.
*/
da_status da_nn_regressor_predict_d(da_handle handle, da_int n_queries, da_int n_features,
                                    const double *X_test, da_int ldx_test, double *y_test,
                                    da_nn_search_mode search_mode);
da_status da_nn_regressor_predict_s(da_handle handle, da_int n_queries, da_int n_features,
                                    const float *X_test, da_int ldx_test, float *y_test,
                                    da_nn_search_mode search_mode);
/** \} */

/** \{
 * \brief Compute Radius Neighbors
 *
 * @rst
 * Compute the radius neighbors of a test data :math:`X_{test}` with respect to the data matrix
 * previously passed into the handle using :ref:`da_nn_set_data_? <da_nn_set_data>`.
 * @endrst
 *
 * \param[inout] handle a \ref da_handle object, initialized with type \ref da_handle_nn.
 * \param[in] n_queries number of observations in \p X_test.
 * \param[in] n_features number of features in \p X_test. Constraint: \p n_features @f$=@f$ the number of features in the data matrix originally supplied to \ref da_nn_set_data_s "da_nn_set_data_?".
 * \param[in] X_test array containing \p n_queries  @f$\times@f$ \p n_features data matrix, in the same storage format used to set the training data.
 * \param[in] ldx_test leading dimension of \p X_test. Constraint: \p ldx_test @f$\ge@f$ \p n_queries if \p X_test is stored in column-major order, or \p ldx_test @f$\ge@f$ \p n_features if \p X_test is stored in row-major order.
 * \param[in] radius radius of the neighborhood. If \p radius < 0 , the radius passed during the option setting will be used instead.
 * \param[in] return_distance denotes if the distances to the radius neighbors need be computed. If \p return_distance is 1, the distances are computed.
 * \param[in] sort_results denotes if the results need to be sorted. If \p sort_results is 1, the indices and distances of the neighbors are sorted in ascending order. Constraint: If \p sort_results is 1, then \p return_distance must also be 1.
 * \return \ref da_status.  The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized with the wrong precision.
 * - \ref da_status_invalid_pointer - the @p handle has not been correctly initialized, or \p X_test or \p n_ind or \p n_dist is invalid.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_memory_error - internal memory allocation encountered a problem.
 * - \ref da_status_invalid_leading_dimension - the constraint on \p ldx_test was violated.
 *
 * \post
 * \parblock
 * After successful execution of this function,  \ref da_handle_get_result_s "da_handle_get_result_?" can be queried to retrieve the radius neighbors' indices and distances with the following enums for floating-point output:
 * - \p da_nn_radius_neighbors_distances - return a flattened array containing the distances to the radius neighbors. The size of the array is equal to the total number of radius neighbors found for all query points. To access the distances for each query point, use the offsets returned by \p da_nn_radius_neighbors_offsets.
 * - \p da_nn_radius_neighbors_distances_index - return a flattened array containing the indices of the radius neighbors for a specific query. The size of the array can be obtained by \p da_nn_radius_neighbors_count. On input the first element of the array should be set to the query index for which the radius neighbors' indices are requested.
 * In addition \ref da_handle_get_result_int can be queried with the following enums:
 * - \p da_nn_radius_neighbors_count - return an array of size \p n_queries + 1 containing the number of radius neighbors for each query point. The last element contains the total number of radius neighbors found for all query points.
 * - \p da_nn_radius_neighbors_offsets - return an array of size \p n_queries + 1 containing the offsets to locate the radius neighbors for each query point. For query points where no radius neighbors were found, the offsets will be -1. For example, the indices of the radius neighbors for query point \p i start at radius_ind[offsets[i]] where radius_ind has been extracted using \p da_nn_radius_neighbors_indices.
 * - \p da_nn_radius_neighbors_indices - return a flattened array containing the indices of the radius neighbors. The size of the array is equal to the total number of radius neighbors found for all query points. To access the indices for each query point, use the offsets returned by \p da_nn_radius_neighbors_offsets.
 * - \p da_nn_radius_neighbors_indices_index - return a flattened array containing the indices of the radius neighbors for a specific query. The size of the array can be obtained by \p da_nn_radius_neighbors_count. On input the first element of the array should be set to the query index for which the radius neighbors' indices are requested.
 * \endparblock
*/
da_status da_nn_radius_neighbors_d(da_handle handle, da_int n_queries, da_int n_features,
                                   const double *X_test, da_int ldx_test, double radius,
                                   da_int return_distance, da_int sort_results);
da_status da_nn_radius_neighbors_s(da_handle handle, da_int n_queries, da_int n_features,
                                   const float *X_test, da_int ldx_test, float radius,
                                   da_int return_distance, da_int sort_results);
/** \} */

#ifdef __cplusplus
}
#endif

#endif
