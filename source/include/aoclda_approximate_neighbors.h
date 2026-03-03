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

#ifndef AOCLDA_APPROX_NN
#define AOCLDA_APPROX_NN

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * \file
 */

/** \{
 * \brief Pass a data matrix to the \ref da_handle object in preparation for approximate nearest neighbor search.
 *
 * The data itself is not copied; a pointer to the data matrix is stored instead.
 * @rst
 * After calling this function you may use the option setting APIs to set :ref:`options <ann_options>`.
 * @endrst
 *
 * \param[inout] handle a \ref da_handle object, initialized with type \ref da_handle_approx_nn.
 * \param[in] n_samples the number of rows in the data matrix, \p X_train. Constraint: \p n_samples @f$\ge@f$ 1.
 * \param[in] n_features the number of columns in the data matrix, \p X_train. Constraint: \p n_features @f$\ge@f$ 1.
 * \param[in] X_train array containing \p n_samples @f$\times@f$ \p n_features data matrix. By default, it should be stored in column-major order, unless you have set the <em>storage order</em> option to <em>row-major</em>.
 * \param[in] ldx_train leading dimension of \p X_train. Constraint: \p ldx_train @f$\ge@f$ \p n_samples if \p X_train is stored in column-major order, or \p ldx_train @f$\ge@f$ \p n_features if \p X_train is stored in row-major order.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 * - \ref da_status_invalid_pointer - the @p handle has not been correctly initialized, or \p X_train is null.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_invalid_leading_dimension - the constraint on \p ldx_train was violated.
 * - \ref da_status_invalid_array_dimension - one of \p n_samples or \p n_features has an invalid value.
 * - \ref da_status_internal_error - an unexpected error occurred.
 */
da_status da_approx_nn_set_training_data_d(da_handle handle, da_int n_samples,
                                           da_int n_features, const double *X_train,
                                           da_int ldx_train);

da_status da_approx_nn_set_training_data_s(da_handle handle, da_int n_samples,
                                           da_int n_features, const float *X_train,
                                           da_int ldx_train);
/** \} */

/** \{
 * \brief Train the approximate nearest neighbor index.
 *
 * @rst
 * Trains the approximate nearest neighbor index using the data matrix previously passed into the handle using :ref:`da_approx_nn_set_training_data_? <da_approx_nn_set_training_data>`.
 * @endrst
 *
 * \param[inout] handle a \ref da_handle object, initialized with type \ref da_handle_approx_nn and with data passed in via \ref da_approx_nn_set_training_data_s "da_approx_nn_set_training_data_?".
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_no_data - \ref da_approx_nn_set_training_data_s "da_approx_nn_set_training_data_?" has not been called prior to this function call.
 * - \ref da_status_invalid_array_dimension - \p n_samples is less than the <em>n_list</em> option value.
 * - \ref da_status_memory_error - internal memory allocation encountered a problem.
 *
 * \post
 * \parblock
 * After successful execution, \ref da_handle_get_result_s "da_handle_get_result_?" can be queried with the following enums for floating-point output:
 * - \p da_rinfo - return an array of size 4 containing \p n_list (the number of lists in the index), \p n_index (the number of samples currently in the index), \p n_features (the number of features) and \p kmeans_iter (the number of <i>k</i>-means iterations performed during training). Note: \p n_index is zero until \ref da_approx_nn_add_s "da_approx_nn_add_?" has been called.
 * - \p da_approx_nn_cluster_centroids - return an array of size \p n_list @f$\times@f$ \p n_features containing the coordinates of the cluster centroids, in the same storage format as the input data.
 * In addition \ref da_handle_get_result_int can be queried with the following enum:
 * - \p da_approx_nn_list_sizes - return an array of size \p n_list containing the number of samples assigned to each list. Note: all entries of the array are zero until \ref da_approx_nn_add_s "da_approx_nn_add_?" has been called.
 * \endparblock
 */
da_status da_approx_nn_train_d(da_handle handle);

da_status da_approx_nn_train_s(da_handle handle);
/** \} */

/** \{
 * \brief Add data points to the approximate nearest neighbor index.
 *
 * @rst
 * Adds data points to an already trained approximate nearest neighbor index.
 * The index must have been previously trained using :ref:`da_approx_nn_train_? <da_approx_nn_train>`.
 * @endrst
 *
 * \param[inout] handle a \ref da_handle object, with the index previously trained via \ref da_approx_nn_train_s "da_approx_nn_train_?".
 * \param[in] n_samples_add the number of rows in the data matrix, \p X_add. Constraint: \p n_samples_add @f$\ge@f$ 1.
 * \param[in] n_features the number of columns in the data matrix, \p X_add. Constraint: \p n_features @f$=@f$ the number of columns in the data matrix originally supplied to \ref da_approx_nn_set_training_data_s "da_approx_nn_set_training_data_?".
 * \param[in] X_add array containing \p n_samples_add @f$\times@f$ \p n_features data matrix, in the same storage format used to set the training data.
 * \param[in] ldx_add leading dimension of \p X_add. Constraint: \p ldx_add @f$\ge@f$ \p n_samples_add if \p X_add is stored in column-major order, or \p ldx_add @f$\ge@f$ \p n_features if \p X_add is stored in row-major order.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 * - \ref da_status_invalid_pointer - the @p handle has not been correctly initialized, or \p X_add is null.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_no_data - the index has not been trained prior to this function call.
 * - \ref da_status_option_locked - an option that cannot be changed after training was modified.
 * - \ref da_status_memory_error - internal memory allocation encountered a problem.
 * - \ref da_status_invalid_leading_dimension - the constraint on \p ldx_add was violated.
 * - \ref da_status_invalid_array_dimension - one of \p n_samples_add or \p n_features has an invalid value.
 */
da_status da_approx_nn_add_d(da_handle handle, da_int n_samples_add, da_int n_features,
                             const double *X_add, da_int ldx_add);

da_status da_approx_nn_add_s(da_handle handle, da_int n_samples_add, da_int n_features,
                             const float *X_add, da_int ldx_add);
/** \} */

/** \{
 * \brief Train the approximate nearest neighbor index and add the training data.
 *
 * @rst
 * This is a convenience function that trains the approximate nearest neighbor index and adds the training data to the index in a single step.
 * It is equivalent to calling :ref:`da_approx_nn_train_? <da_approx_nn_train>` followed by :ref:`da_approx_nn_add_? <da_approx_nn_add>` with the same training data.
 * The training data must have been previously passed into the handle using :ref:`da_approx_nn_set_training_data_? <da_approx_nn_set_training_data>`.
 * @endrst
 *
 * \param[inout] handle a \ref da_handle object, initialized with type \ref da_handle_approx_nn and with data passed in via \ref da_approx_nn_set_training_data_s "da_approx_nn_set_training_data_?".
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 * - \ref da_status_invalid_pointer - the @p handle has not been correctly initialized.
 * - \ref da_status_no_data - \ref da_approx_nn_set_training_data_s "da_approx_nn_set_training_data_?" has not been called prior to this function call.
 * - \ref da_status_invalid_array_dimension - \p n_samples is less than the <em>n_list</em> option value.
 * - \ref da_status_memory_error - internal memory allocation encountered a problem.
 */
da_status da_approx_nn_train_and_add_d(da_handle handle);

da_status da_approx_nn_train_and_add_s(da_handle handle);
/** \} */

/** \{
 * \brief Compute approximate <i>k</i>-nearest neighbors.
 *
 * @rst
 * Computes the approximate *k*-nearest neighbors of a test data :math:`X_{test}` with respect to the data previously added to the index using :ref:`da_approx_nn_add_? <da_approx_nn_add>` or :ref:`da_approx_nn_train_and_add_? <da_approx_nn_train_and_add>`.
 * @endrst
 *
 * \param[inout] handle a \ref da_handle object, with data previously added to the index via \ref da_approx_nn_add_s "da_approx_nn_add_?" or \ref da_approx_nn_train_and_add_s "da_approx_nn_train_and_add_?".
 * \param[in] n_queries the number of rows in the data matrix, \p X_test. Constraint: \p n_queries @f$\ge@f$ 1.
 * \param[in] n_features the number of columns in the data matrix, \p X_test. Constraint: \p n_features @f$=@f$ the number of columns in the data matrix originally supplied to \ref da_approx_nn_set_training_data_s "da_approx_nn_set_training_data_?".
 * \param[in] X_test array containing \p n_queries @f$\times@f$ \p n_features data matrix, in the same storage format used to set the training data.
 * \param[in] ldx_test leading dimension of \p X_test. Constraint: \p ldx_test @f$\ge@f$ \p n_queries if \p X_test is stored in column-major order, or \p ldx_test @f$\ge@f$ \p n_features if \p X_test is stored in row-major order.
 * \param[out] n_ind array containing the \p n_queries @f$\times@f$ \p k matrix, with the indices of the \p k approximate nearest neighbors for each query point. The indices correspond to the order in which data points were added to the index.
 * \param[out] n_dist array containing the corresponding distances to the neighbors whose indices are stored in \p n_ind, if \p return_distance is 1.
 * \param[in] k number of nearest neighbors requested. If \p k @f$\le@f$ 0, the number of neighbors set via the options will be used instead. Constraint: \p k @f$\le@f$ the number of samples added to the index.
 * \param[in] return_distance denotes if the distances to the approximate nearest neighbors must be computed. If \p return_distance is 1, the distances are returned.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the floating point precision of the arguments is incompatible with the @p handle initialization.
 * - \ref da_status_invalid_pointer - the @p handle has not been correctly initialized, or \p X_test or \p n_ind is null, or \p n_dist is null when \p return_distance is 1.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_no_data - no data has been added to the index prior to this function call.
 * - \ref da_status_option_locked - an option that cannot be changed after training was modified.
 * - \ref da_status_memory_error - internal memory allocation encountered a problem.
 * - \ref da_status_invalid_leading_dimension - the constraint on \p ldx_test was violated.
 * - \ref da_status_invalid_array_dimension - one of \p n_queries or \p n_features has an invalid value.
 */
da_status da_approx_nn_kneighbors_d(da_handle handle, da_int n_queries, da_int n_features,
                                    const double *X_test, da_int ldx_test, da_int *n_ind,
                                    double *n_dist, da_int k, da_int return_distance);

da_status da_approx_nn_kneighbors_s(da_handle handle, da_int n_queries, da_int n_features,
                                    const float *X_test, da_int ldx_test, da_int *n_ind,
                                    float *n_dist, da_int k, da_int return_distance);
/** \} */

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // AOCLDA_APPROX_NN
