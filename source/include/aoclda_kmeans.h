/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef AOCLDA_KMEANS
#define AOCLDA_KMEANS

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
 * \brief Pass a data matrix to the \ref da_handle object in preparation for *k*-means clustering.
 *
 * A copy of the data matrix is stored internally, to avoid overwriting the user's data during the computation.
 * @rst
 * After calling this function you may use the option setting APIs to set :ref:`options <kmeans_options>`.
 * @endrst
 *
 * \param[in,out] handle a \ref da_handle object, initialized using \ref da_handle_init_d with type \ref da_handle_kmeans.
 * \param[in] n_samples the number of rows of the data matrix, \p A. Constraint: \p n_samples @f$\ge@f$ 1.
 * \param[in] n_features the number of columns of the data matrix, \p A. Constraint: \p n_features @f$\ge@f$ 1.
 * \param[in] A the \p n_samples @f$\times@f$ \p n_features data matrix, in column major format.
 * \param[in] lda the leading dimension of the data matrix. Constraint: \p lda @f$\ge@f$ \p n_samples.
  * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using \ref da_handle_init_s.
 * - \ref da_status_invalid_pointer - the handle has not been initialized, or \p A is null.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 */
da_status da_kmeans_set_data_d(da_handle handle, da_int n_samples, da_int n_features,
                               const double *A, da_int lda);
/** \} */

/** \{
 * \brief Pass a data matrix to the \ref da_handle object in preparation for *k*-means clustering.
 *
 * A copy of the data matrix is stored internally, to avoid overwriting the user's data during the computation.
 * @rst
 * After calling this function you may use the option setting APIs to set :ref:`options <kmeans_options>`.
 * @endrst
 *
 * \param[in,out] handle a \ref da_handle object, initialized using \ref da_handle_init_s
 * with type \ref da_handle_kmeans.
 * \param[in] n_samples the number of rows of the data matrix, \p A. Constraint: \p n_samples @f$\ge@f$ 1.
 * \param[in] n_features the number of columns of the data matrix, \p A. Constraint \p n_features @f$\ge@f$ 1.
 * \param[in] A the \p n_samples @f$\times@f$ \p n_features data matrix, in column major format.
 * \param[in] lda the leading dimension of the data matrix. Constraint: \p lda @f$\ge@f$ \p n_samples.
  * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using \ref da_handle_init_d.
 * - \ref da_status_invalid_pointer - the handle has not been initialized or \p A is null.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 */
da_status da_kmeans_set_data_s(da_handle handle, da_int n_samples, da_int n_features,
                               const float *A, da_int lda);
/** \} */

/** \{
 * \brief Pass a matrix of initial cluster centres to the \ref da_handle object in preparation for *k*-means clustering.
 *
 * A copy of the data matrix is stored internally, to avoid overwriting the user's data during the computation.
 * The matrix of initial clusters is not required if *k*-means++ or random initialization are used (which is specified using \ref da_options_set_string)
 *
 * Note, you must call \ref da_kmeans_set_data_s prior to this function.
 *
 * \param[in,out] handle a \ref da_handle object, initialized using \ref da_handle_init_d with type \ref da_handle_kmeans.
 * \param[in] C the \p n_clusters @f$\times@f$ \p n_features matrix of initial centres, in column major format.
 * \param[in] ldc the leading dimension of the data matrix. Constraint: \p ldc @f$\ge@f$ \p n_clusters so make sure you set \p n_clusters using \ref da_options_set_int first.
  * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using \ref da_handle_init_s.
 * - \ref da_status_invalid_pointer - the handle has not been initialized, or \p C is null.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 */
da_status da_kmeans_set_init_centres_d(da_handle handle, const double *C, da_int ldc);
/** \} */

/** \{
 * \brief Pass a matrix of initial cluster centres to the \ref da_handle object in preparation for *k*-means clustering.
 *
 * A copy of the data matrix is stored internally, to avoid overwriting the user's data during the computation.
 * The matrix of initial clusters is not required if *k*-means++ or random initialization are used (which is specified using \ref da_options_set_string)
 *
 * Note, you must call \ref da_kmeans_set_data_d prior to this function.
 *
 * \param[in,out] handle a \ref da_handle object, initialized using \ref da_handle_init_d with type \ref da_handle_kmeans.
 * \param[in] C the \p n_clusters @f$\times@f$ \p n_features matrix of initial centres, in column major format.
 * \param[in] ldc the leading dimension of the data matrix. Constraint: \p ldc @f$\ge@f$ \p n_clusters so make sure you set \p n_clusters using \ref da_options_set_int first.
  * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using \ref da_handle_init_s.
 * - \ref da_status_invalid_pointer - the handle has not been initialized, or \p C is null.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 */
da_status da_kmeans_set_init_centres_s(da_handle handle, const float *C, da_int ldc);
/** \} */

/** \{
 * \brief Compute *k*-means clustering
 *
 * Computes *k*-means clustering on the data matrix previously passed into the handle using \ref da_kmeans_set_data_d.
 *
 * \param[in,out] handle a \ref da_handle object, initialized using \ref da_handle_init_d
 *  with type \ref da_handle_kmeans and with data passed in via \ref da_kmeans_set_data_d.
  * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using \ref da_handle_init_s.
 * - \ref da_status_invalid_pointer - the handle has not been initialized.
 * - \ref da_status_no_data - \ref da_kmeans_set_data_d has not been called prior to this function call.
 * - \ref da_status_internal_error - this can occur if your data contains undefined values.
 * - \ref da_status_maxit - iteration limit reached without converging

 *
 * \post
 * After successful execution, \ref da_handle_get_result_d can be queried with the following enums:
 * - \p da_kmeans_cluster_centres - return an array of size \p n_clusters @f$\times@f$ \p n_features containing the coordinates of the cluster centres, in column major format.
 * - \p da_rinfo - return an array of size 5 containing \p n_samples, \p n_features, \p n_clusters, \p n_iter (the number of iterations performed) and \p inertia (the sum of the squared distance of each sample to its closest cluster centre).
 * and \ref da_handle_get_result_int can be queried with the following enum:
 * - \p da_kmeans_labels - return an array of size \p n_samples containing the label (i.e. which cluster it is in) of each sample point.
 */
da_status da_kmeans_compute_d(da_handle handle);
/** \} */

/** \{
 * \brief Compute *k*-means clustering
 *
 * Computes  *k*-means clustering on the data matrix previously passed into the handle using \ref da_kmeans_set_data_s.
 *
 * \param[in,out] handle a \ref da_handle object, initialized using \ref da_handle_init_s
 *  with type \ref da_handle_kmeans and with data passed in via \ref da_kmeans_set_data_s.
  * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using \ref da_handle_init_d.
 * - \ref da_status_invalid_pointer - the handle has not been initialized.
 * - \ref da_status_no_data - \ref da_kmeans_set_data_d has not been called prior to this function call.
 * - \ref da_status_internal_error - this can occur if your data contains undefined values.
 * - \ref da_status_maxit - iteration limit reached without converging

 *
 * \post
 * After succesful execution, \ref da_handle_get_result_s can be queried with the following enums:
 * - \p da_kmeans_cluster_centres - return an array of size \p n_clusters @f$\times@f$ \p n_features containing the coordinates of the cluster centres, in column major format.
 * - \p da_rinfo - return an array of size 5 containing \p n_samples, \p n_features, \p n_clusters, \p n_iter (the number of iterations performed) and \p inertia (the sum of the squared distance of each sample to its closest cluster centre).
 * and \ref da_handle_get_result_int can be queried with the following enum:
 * - \p da_kmeans_labels - return an array of size \p n_samples containing the label (i.e. which cluster it is in) of each sample point.
 */
da_status da_kmeans_compute_s(da_handle handle);
/** \} */

/** \{
 * \brief Transform a data matrix into the cluster distance space
 *
 * Transforms a data matrix \p X from the original coordinate system into the new coordinates in which each dimension is the distance to the cluster centres previously computed in \ref da_kmeans_compute_d.
 *
 * \param[in,out] handle a \ref da_handle object, with *k*-means clusters previously computed via \ref da_kmeans_compute_d.
 * \param[in] m_samples the number of rows of the data matrix, \p X. Constraint: \p m_samples @f$\ge@f$ 1.
 * \param[in] m_features the number of columns of the data matrix, \p X. Constraint: \p m_features @f$=@f$ \p n_features, the number of features in the data matrix originally supplied to \ref da_kmeans_set_data_d.
 * \param[in] X the \p m_samples @f$\times@f$ \p m_features data matrix, in column major format.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: \p ldx @f$\ge@f$ \p m_samples.
 * \param[out] X_transform an array of size at least \p m_samples @f$\times@f$ \p n_clusters, in which the transformed data will be stored (in column major format).
 * \param[in] ldx_transform the leading dimension of \p X_transform. Constraint: \p ldx_transform @f$\ge@f$ \p m_samples.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using \ref da_handle_init_s.
 * - \ref da_status_invalid_pointer - the handle has not been initialized, or one of the arrays is null.
 * - \ref da_status_no_data - the *k*-means clusters have not been computed prior to this function call.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 *
 */
da_status da_kmeans_transform_d(da_handle handle, da_int m_samples, da_int m_features,
                                const double *X, da_int ldx, double *X_transform,
                                da_int ldx_transform);
/** \} */

/** \{
 * \brief Transform a data matrix into the cluster distance space
 *
 * Transforms a data matrix \p X from the original coordinate system into the new coordinates in which each dimension is the distance to the cluster centres previously computed in \ref da_kmeans_compute_s.
 *
 * \param[in,out] handle a \ref da_handle object, with *k*-means clusters previously computed via \ref da_kmeans_compute_s.
 * \param[in] m_samples the number of rows of the data matrix, \p X. Constraint: \p m_samples @f$\ge@f$ 1.
 * \param[in] m_features the number of columns of the data matrix, \p X. Constraint: \p m_features @f$=@f$ \p n_features, the number of features in the data matrix originally supplied to \ref da_kmeans_set_data_s.
 * \param[in] X the \p m_samples @f$\times@f$ \p m_features data matrix, in column major format.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: \p ldx @f$\ge@f$ \p m_samples.
 * \param[out] X_transform an array of size at least \p m_samples @f$\times@f$ \p n_clusters, in which the transformed data will be stored (in column major format).
 * \param[in] ldx_transform the leading dimension of \p X_transform. Constraint: \p ldx_transform @f$\ge@f$ \p m_samples.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using \ref da_handle_init_d.
 * - \ref da_status_invalid_pointer - the handle has not been initialized, or one of the arrays is null.
 * - \ref da_status_no_data - the *k*-means clustering has not been computed prior to this function call.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 *
 */
da_status da_kmeans_transform_s(da_handle handle, da_int m_samples, da_int m_features,
                                const float *X, da_int ldx, float *X_transform,
                                da_int ldx_transform);
/** \} */

/** \{
 * \brief Predict the cluster each sample in a data matrix belongs to
 *
 * For each sample in the data matrix \p Y find the closest cluster centre out of the clusters previously computed in \ref da_kmeans_compute_d.
 *
 * \param[in,out] handle a \ref da_handle object, with *k*-means clusters previously computed via \ref da_kmeans_compute_d.
 * \param[in] k_samples the number of rows of the data matrix, \p Y. Constraint: \p k_samples @f$\ge@f$ 1.
 * \param[in] k_features the number of columns of the data matrix, \p Y. Constraint: \p k_features @f$=@f$ \p n_features, the number of features in the data matrix originally supplied to \ref da_kmeans_set_data_d.
 * \param[in] Y the \p k_samples @f$\times@f$ \p k_features data matrix, in column major format.
 * \param[in] ldy the leading dimension of the data matrix. Constraint: \p ldy @f$\ge@f$ \p k_samples.
 * \param[out] Y_labels an array of size at least \p k_samples, in which the labels will be stored.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using \ref da_handle_init_s.
 * - \ref da_status_invalid_pointer - the handle has not been initialized, or one of the arrays is null.
 * - \ref da_status_no_data - the *k*-means clustering has not been computed prior to this function call.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 *
 */
da_status da_kmeans_predict_d(da_handle handle, da_int k_samples, da_int k_features,
                              const double *Y, da_int ldy, da_int *Y_labels);
/** \} */

/** \{
 * \brief Predict the cluster each sample in a data matrix belongs to
 *
 * For each sample in the data matrix \p Y find the closest cluster centre out of the clusters previously computed in \ref da_kmeans_compute_s.
 *
 * \param[in,out] handle a \ref da_handle object, with *k*-means clusters previously computed via \ref da_kmeans_compute_s.
 * \param[in] k_samples the number of rows of the data matrix, \p Y. Constraint: \p k_samples @f$\ge@f$ 1.
 * \param[in] k_features the number of columns of the data matrix, \p Y. Constraint: \p k_features @f$=@f$ \p n_components, the number of features in the data matrix originally supplied to \ref da_kmeans_set_data_s.
 * \param[in] Y the \p k_samples @f$\times@f$ \p k_features data matrix, in column major format.
 * \param[in] ldy the leading dimension of the data matrix. Constraint: \p ldy @f$\ge@f$ \p k_samples.
 * \param[out] Y_labels an array of size at least \p k_samples, in which the labels will be stored.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using \ref da_handle_init_d.
 * - \ref da_status_invalid_pointer - the handle has not been initialized, or one of the arrays is null.
 * - \ref da_status_no_data - the *k*-means clustering has not been computed prior to this function call.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 *
 */
da_status da_kmeans_predict_s(da_handle handle, da_int k_samples, da_int k_features,
                              const float *Y, da_int ldy, da_int *Y_labels);
/** \} */

#ifdef __cplusplus
}
#endif

#endif
