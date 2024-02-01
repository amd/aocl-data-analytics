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

#ifndef AOCLDA_PCA
#define AOCLDA_PCA

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
 * \brief Pass a data matrix to the \ref da_handle object in preparation for computing the PCA.
 *
 * A copy of the data matrix is stored internally, to avoid overwriting the user's data during the computation.
 * @rst
 * After calling this function you may use the option setting APIs to set :ref:`options <pca_options>`.
 * @endrst
 *
 * \param[in,out] handle a \ref da_handle object, initialized using \ref da_handle_init_s "da_handle_init_?" with type \ref da_handle_pca.
 * \param[in] n_samples the number of rows of the data matrix, \p A. Constraint: \p n_samples @f$\ge@f$ 1.
 * \param[in] n_features the number of columns of the data matrix, \p A. Constraint: \p n_features @f$\ge@f$ 1.
 * \param[in] A the \p n_samples @f$\times@f$ \p n_features data matrix, in column major format.
 * \param[in] lda the leading dimension of the data matrix. Constraint: \p lda @f$\ge@f$ \p n_samples.
  * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using th wrong precision.
 * - \ref da_status_invalid_pointer - the handle has not been initialized, or \p A is null.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 */
da_status da_pca_set_data_d(da_handle handle, da_int n_samples, da_int n_features,
                            const double *A, da_int lda);

da_status da_pca_set_data_s(da_handle handle, da_int n_samples, da_int n_features,
                            const float *A, da_int lda);
/** \} */

/** \{
 * \brief Compute PCA
 *
 * Computes a principal component analysis on the data matrix previously passed into the handle using \ref da_pca_set_data_s "da_pca_set_data_?".
 *
 * \param[in,out] handle a \ref da_handle object, initialized using \ref da_handle_init_s "da_handle_init_?"
 *  with type \ref da_handle_pca and with data passed in via \ref da_pca_set_data_s "da_pca_set_data_?".
  * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using the wrong precision.
 * - \ref da_status_invalid_pointer - the handle has not been initialized.
 * - \ref da_status_no_data - \ref da_pca_set_data_s "da_pca_set_data_?" has not been called prior to this function call.
 * - \ref da_status_internal_error - this can occur if your data contains undefined values.
 *
 * \post
 * After successful execution, \ref da_handle_get_result_s "da_handle_get_result_?" can be queried with the following enums:
 * - \p da_rinfo - return an array of size 3 containing \p n_samples, \p n_features and \p n_components, the number of principal components found.
 * - \p da_pca_principal_components - return an array of size \p n_components @f$\times@f$ \p n_features containing the principal components in column major format in decreasing order of explained variance.
 * - \p da_pca_scores - return an array of size \p n_samples @f$\times@f$ \p n_components containing the scores in column major format.
 * - \p da_pca_variance - return an array of size \p n_components containing the amount of variance explained by each principal component.
 * - \p da_pca_total_variance - return an array of size \p 1 containing the total of variance within the dataset.
 * - \p da_pca_u - return an array of size \p n_samples @f$\times@f$ \p n_components containing the matrix @f$ U @f$ from the SVD of the standardized data matrix in column major format.
 * - \p da_pca_sigma - return an array of size \p n_components containing the singular values of the standardized data matrix.
 * - \p da_pca_vt - return an array of size \p n_components @f$\times@f$ \p n_features containing the matrix @f$ V^T @f$ from the SVD of the standardized data matrix in column major format.
 * - \p da_pca_column_means - return an array of size \p n_features containing the column means of the data matrix (note this is only available if the option <i> PCA method</i> was set to \a covariance or \a correlation).
 * - \p da_pca_column_sdevs - return an array of size \p n_features containing the column standard deviations (with \p n_samples @f$\ - 1 @f$ degrees of freedom) of the data matrix (note this is only available if the option <i>PCA method</i> was set to \a correlation).
 */
da_status da_pca_compute_d(da_handle handle);

da_status da_pca_compute_s(da_handle handle);
/** \} */

/** \{
 * \brief Transform a data matrix into new feature space
 *
 * Transforms a data matrix \p X from the original coordinate system into the new coordinates previously computed in \ref da_pca_compute_s "da_pca_compute_?".
 * The transformation is computed by applying any standardization used on the original data matrix to \p X, then projecting \p X into the previously computed principal components.
 *
 * \param[in,out] handle a \ref da_handle object, with a PCA previously computed via \ref da_pca_compute_s "da_pca_compute_?".
 * \param[in] m_samples the number of rows of the data matrix, \p X. Constraint: \p m_samples @f$\ge@f$ 1.
 * \param[in] m_features the number of columns of the data matrix, \p X. Constraint: \p m_features @f$=@f$ \p n_features, the number of features in the data matrix originally supplied to \ref da_pca_set_data_s "da_pca_set_data_?".
 * \param[in] X the \p m_samples @f$\times@f$ \p m_features data matrix, in column major format.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: \p ldx @f$\ge@f$ \p m_samples.
 * \param[out] X_transform an array of size at least \p m_samples @f$\times@f$ \p n_components, in which the transformed data will be stored (in column major format).
 * \param[in] ldx_transform the leading dimension of \p X_transform. Constraint: \p ldx_transform @f$\ge@f$ \p m_samples.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using the wrong precision.
 * - \ref da_status_invalid_pointer - the handle has not been initialized, or one of the arrays is null.
 * - \ref da_status_no_data - the PCA has not been computed prior to this function call.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 *
 */
da_status da_pca_transform_d(da_handle handle, da_int m_samples, da_int m_features,
                             const double *X, da_int ldx, double *X_transform,
                             da_int ldx_transform);

da_status da_pca_transform_s(da_handle handle, da_int m_samples, da_int m_features,
                             const float *X, da_int ldx, float *X_transform,
                             da_int ldx_transform);
/** \} */

/** \{
 * \brief Transform a data matrix into the original coordinate space
 *
 * Transforms a data matrix  \p Y in the new feature space back into the original coordinate space used by the matrix which was supplied to \ref da_pca_set_data_s "da_pca_set_data_?".
 * The transformation is computed by projecting \p Y into the original coordinate space, then inverting any standardization used on the original data matrix.
 *
 * \param[in,out] handle a \ref da_handle object, with a PCA previously computed via \ref da_pca_compute_s "da_pca_compute_?".
 * \param[in] k_samples the number of rows of the data matrix, \p Y. Constraint: \p k_samples @f$\ge@f$ 1.
 * \param[in] k_features the number of columns of the data matrix, \p Y. Constraint: \p k_features @f$=@f$ \p n_components, the number of PCA components computed by \ref da_pca_compute_s "da_pca_compute_?".
 * \param[in] Y the \p k_samples @f$\times@f$ \p k_features data matrix, in column major format.
 * \param[in] ldy the leading dimension of the data matrix. Constraint: \p ldy @f$\ge@f$ \p k_samples.
 * \param[out] Y_inv_transform an array of size at least \p k_samples @f$\times@f$ \p n_features, in which the transformed data will be stored (in column major format).
 * \param[in] ldy_inv_transform the leading dimension of \p Y_inv_transform. Constraint: \p ldy_inv_transform @f$\ge@f$ \p k_samples.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using the wrong precision.
 * - \ref da_status_invalid_pointer - the handle has not been initialized, or one of the arrays is null.
 * - \ref da_status_no_data - the PCA has not been computed prior to this function call.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 *
 */
da_status da_pca_inverse_transform_d(da_handle handle, da_int k_samples,
                                     da_int k_features, const double *Y, da_int ldy,
                                     double *Y_inv_transform, da_int ldy_inv_transform);

da_status da_pca_inverse_transform_s(da_handle handle, da_int k_samples,
                                     da_int k_features, const float *Y, da_int ldy,
                                     float *Y_inv_transform, da_int ldy_inv_transform);
/** \} */

#ifdef __cplusplus
}
#endif

#endif
