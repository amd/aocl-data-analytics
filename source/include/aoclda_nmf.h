/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

#ifndef AOCLDA_NMF
#define AOCLDA_NMF

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"

/**
 * \file
 * \brief Nonnegative Matrix Factorization (NMF) public C API.
 *
 * Currently only the multiplicative-update (MU) solver minimizing the
 * Frobenius norm reconstruction error is exposed.
 */

/** \{
 * \brief Pass a data matrix to the \ref da_handle object in preparation for computing the NMF.
 *
 * The data itself is not copied, a pointer to the data matrix is stored instead.
 * @rst
 * After calling this function you may use the option setting APIs to set :ref:`options <nmf_options>`.
 * @endrst
 *
 * \param[inout] handle a \ref da_handle object, initialized using \ref da_handle_init_s "da_handle_init_?" with type \ref da_handle_nmf.
 * \param[in] n_samples the number of rows of the data matrix, \p A. Constraint: \p n_samples @f$\ge@f$ 1.
 * \param[in] n_features the number of columns of the data matrix, \p A. Constraint: \p n_features @f$\ge@f$ 1.
 * \param[in] A the \p n_samples @f$\times@f$ \p n_features data matrix (must be nonnegative). By default, it should be stored in column-major order, unless you have set the <em>storage order</em> option to <em>row-major</em>.
 * \param[in] lda the leading dimension of the data matrix. Constraint: \p lda @f$\ge@f$ \p n_samples if \p A is stored in column-major order, or \p lda @f$\ge@f$ \p n_features if \p A is stored in row-major order.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using the wrong precision.
 * - \ref da_status_handle_not_initialized - \p handle is null.
 * - \ref da_status_invalid_pointer - \p A is null.
 * - \ref da_status_invalid_array_dimension - \p n_samples or \p n_features is less than 1.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value (e.g. \p A contains a negative entry). You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_invalid_leading_dimension - the constraint on \p lda was violated.
 */
da_status da_nmf_init_d(da_handle handle, da_int n_samples, da_int n_features,
                        const double *A, da_int lda);

da_status da_nmf_init_s(da_handle handle, da_int n_samples, da_int n_features,
                        const float *A, da_int lda);
/** \} */

/** \{
 * \brief Supply a custom nonnegative initialization @f$(W_0, H_0)@f$ to the NMF
 *        handle.
 *
 * When the <em>init</em> option is set to \c "custom", the multiplicative-update solver starts from the user-supplied factors.
 * The data matrix \c A must already have been uploaded with \ref da_nmf_init_s "da_nmf_init_?". Both factors must be nonnegative.
 * \p n_components must equal the value of the <em>n_components</em> option. The call supports partial upload: exactly one of
 * \p W or \p H may be null, in which case the provided factor is stored and a warning status is returned.
 * This allows uploading \p W and \p H in two separate calls, but compute requires both to be available.
 *
 * \param[inout] handle a \ref da_handle object, initialized using \ref da_handle_init_s "da_handle_init_?" with type \ref da_handle_nmf and with data passed in via \ref da_nmf_init_s "da_nmf_init_?".
 * \param[in] n_samples the number of rows of \p W. Constraint: \p n_samples must equal the value of \p n_samples passed to \ref da_nmf_init_s "da_nmf_init_?".
 * \param[in] n_components the number of components @f$k@f$. Constraint: must match the <em>n_components</em> option and satisfy @f$1 \le k \le \min(n_{\text{samples}}, n_{\text{features}})@f$.
 * \param[in] W the @f$n_{\text{samples}} \times k@f$ initial left factor, or null to leave \p W unchanged in this call.
 * \param[in] ldw the leading dimension of \p W. Constraint: \p ldw @f$\ge@f$ \p n_samples if \p W is stored in column-major order, or \p ldw @f$\ge@f$ \p n_components if \p W is stored in row-major order.
 * \param[in] n_features the number of columns of \p H. Constraint: \p n_features must equal the value of \p n_features passed to \ref da_nmf_init_s "da_nmf_init_?".
 * \param[in] H the @f$k \times n_{\text{features}}@f$ initial right factor, or null to leave \p H unchanged in this call.
 * \param[in] ldh the leading dimension of \p H. Constraint: \p ldh @f$\ge@f$ \p n_components if \p H is stored in column-major order, or \p ldh @f$\ge@f$ \p n_features if \p H is stored in row-major order.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using the wrong precision.
 * - \ref da_status_handle_not_initialized - \p handle is null.
 * - \ref da_status_no_data - \ref da_nmf_init_s "da_nmf_init_?" has not been called prior to this function call.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value (e.g. a factor contains negative entries, or a dimension does not match the data matrix). You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_invalid_leading_dimension - a constraint on \p ldw or \p ldh was violated.
 * - \ref da_status_memory_error - a memory allocation failed.
 * - \ref da_status_missing_block - only one of \p W or \p H was provided; supply the remaining factor before calling \ref da_nmf_compute_s "da_nmf_compute_?".
 */
da_status da_nmf_set_data_d(da_handle handle, da_int n_samples, da_int n_components,
                            const double *W, da_int ldw, da_int n_features,
                            const double *H, da_int ldh);

da_status da_nmf_set_data_s(da_handle handle, da_int n_samples, da_int n_components,
                            const float *W, da_int ldw, da_int n_features, const float *H,
                            da_int ldh);
/** \} */

/** \{
 * \brief Compute the NMF.
 *
 * Computes the nonnegative matrix factorization @f$A \approx W H@f$ of the data
 * matrix previously passed via \ref da_nmf_init_s "da_nmf_init_?", using
 * the multiplicative-update solver on the Frobenius-norm objective.
 *
 * \param[inout] handle a \ref da_handle object, initialized using \ref da_handle_init_s "da_handle_init_?" with type \ref da_handle_nmf and with data passed in via \ref da_nmf_init_s "da_nmf_init_?".
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using the wrong precision.
 * - \ref da_status_handle_not_initialized - \p handle is null.
 * - \ref da_status_no_data - \ref da_nmf_init_s "da_nmf_init_?" has not been called, or the <em>init</em> option is set to <em>custom</em> but \ref da_nmf_set_data_s "da_nmf_set_data_?" has not provided both factors.
 * - \ref da_status_invalid_input - an option had an invalid value, or the number of components in the custom initial factors does not match the <em>n_components</em> option. You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_memory_error - a memory allocation failed.
 *
 * \post
 * After successful execution, \ref da_handle_get_result_s "da_handle_get_result_?" can be queried with the following enums:
 * - \p da_nmf_w - return an array of size \p n_samples @f$\times@f$ \p n_components containing the left factor @f$W@f$, in the same storage order as the input data.
 * - \p da_nmf_h - return an array of size \p n_components @f$\times@f$ \p n_features containing the right factor @f$H@f$, in the same storage order as the input data.
 * - \p da_nmf_reconstruction_err - return an array of size 1 containing the Frobenius reconstruction error @f$\|A - WH\|_F@f$.
 * - \p da_rinfo - return an array of size 2 containing \p n_samples and \p n_features.
 *
 * In addition, \ref da_handle_get_result_int can be queried with the following enums:
 * - \p da_nmf_n_iter - return an array of size 1 containing the number of multiplicative-update iterations performed.
 * - \p da_nmf_n_components - return an array of size 1 containing the number of components @f$k@f$ used (equal to the minimum of \p n_samples and \p n_features when the <em>n_components</em> option was set to 0).
 */
da_status da_nmf_compute_d(da_handle handle);

da_status da_nmf_compute_s(da_handle handle);
/** \} */

/** \{
 * \brief Transform a data matrix into the NMF feature space.
 *
 * Transforms a data matrix \p X into its NMF representation using the components @f$H@f$ previously computed in \ref da_nmf_compute_s "da_nmf_compute_?".
 * The transformation holds @f$H@f$ fixed and finds a nonnegative coefficient matrix @f$W_{\text{new}}@f$ such that @f$X \approx W_{\text{new}} H@f$
 * by running multiplicative-update iterations on \p X with the same solver options used during fitting.
 *
 * \param[inout] handle a \ref da_handle object, with an NMF previously computed via \ref da_nmf_compute_s "da_nmf_compute_?".
 * \param[in] m_samples the number of rows of the data matrix, \p X. Constraint: \p m_samples @f$\ge@f$ 1.
 * \param[in] m_features the number of columns of the data matrix, \p X. Constraint: \p m_features @f$=@f$ \p n_features, the number of features in the data matrix originally supplied to \ref da_nmf_init_s "da_nmf_init_?".
 * \param[in] X the \p m_samples @f$\times@f$ \p m_features data matrix (must be nonnegative), in the same storage format used to fit the model.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: \p ldx @f$\ge@f$ \p m_samples if \p X is stored in column-major order, or \p ldx @f$\ge@f$ \p m_features if \p X is stored in row-major order.
 * \param[out] X_transform an array of size at least \p m_samples @f$\times@f$ \p n_components, in which the NMF coefficient matrix @f$W_{\text{new}}@f$ will be stored (in the same storage format used to fit the model).
 * \param[in] ldx_transform the leading dimension of \p X_transform. Constraint: \p ldx_transform @f$\ge@f$ \p m_samples if \p X_transform is stored in column-major order, or \p ldx_transform @f$\ge@f$ \p n_components if \p X_transform is stored in row-major order.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using the wrong precision.
 * - \ref da_status_handle_not_initialized - \p handle is null.
 * - \ref da_status_no_data - the NMF has not been computed prior to this function call.
 * - \ref da_status_invalid_pointer - \p X or \p X_transform is null.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value (e.g. \p m_features does not match, or \p X contains a negative entry). You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_invalid_array_dimension - \p m_samples was less than 1.
 * - \ref da_status_invalid_leading_dimension - a constraint on \p ldx or \p ldx_transform was violated.
 * - \ref da_status_memory_error - a memory allocation failed.
 */
da_status da_nmf_transform_d(da_handle handle, da_int m_samples, da_int m_features,
                             const double *X, da_int ldx, double *X_transform,
                             da_int ldx_transform);

da_status da_nmf_transform_s(da_handle handle, da_int m_samples, da_int m_features,
                             const float *X, da_int ldx, float *X_transform,
                             da_int ldx_transform);
/** \} */

/** \{
 * \brief Reconstruct a data matrix from NMF coefficients.
 *
 * Transforms a coefficient matrix \p Y from the NMF feature space back into the original feature space
 * using the components @f$H@f$ previously computed in \ref da_nmf_compute_s "da_nmf_compute_?".
 * The reconstruction is computed as @f$X = Y H@f$.
 *
 * \param[inout] handle a \ref da_handle object, with an NMF previously computed via \ref da_nmf_compute_s "da_nmf_compute_?".
 * \param[in] k_samples the number of rows of the data matrix, \p Y. Constraint: \p k_samples @f$\ge@f$ 1.
 * \param[in] k_components the number of columns of the data matrix, \p Y. Constraint: \p k_components @f$=@f$ \p n_components, the number of components computed by \ref da_nmf_compute_s "da_nmf_compute_?".
 * \param[in] Y the \p k_samples @f$\times@f$ \p k_components coefficient matrix, in the same storage format used to fit the model.
 * \param[in] ldy the leading dimension of \p Y. Constraint: \p ldy @f$\ge@f$ \p k_samples if \p Y is stored in column-major order, or \p ldy @f$\ge@f$ \p k_components if \p Y is stored in row-major order.
 * \param[out] Y_inv_transform an array of size at least \p k_samples @f$\times@f$ \p n_features, in which the reconstructed data will be stored (in the same storage format used to fit the model).
 * \param[in] ldy_inv_transform the leading dimension of \p Y_inv_transform. Constraint: \p ldy_inv_transform @f$\ge@f$ \p k_samples if \p Y_inv_transform is stored in column-major order, or \p ldy_inv_transform @f$\ge@f$ \p n_features if \p Y_inv_transform is stored in row-major order.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using the wrong precision.
 * - \ref da_status_handle_not_initialized - \p handle is null.
 * - \ref da_status_no_data - the NMF has not been computed prior to this function call.
 * - \ref da_status_invalid_pointer - \p Y or \p Y_inv_transform is null.
 * - \ref da_status_invalid_input - \p k_components does not equal \p n_components from the fitted model. You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_invalid_array_dimension - \p k_samples was less than 1.
 * - \ref da_status_invalid_leading_dimension - a constraint on \p ldy_inv_transform or \p ldy was violated.
 */
da_status da_nmf_inverse_transform_d(da_handle handle, da_int k_samples,
                                     da_int k_components, const double *Y, da_int ldy,
                                     double *Y_inv_transform, da_int ldy_inv_transform);

da_status da_nmf_inverse_transform_s(da_handle handle, da_int k_samples,
                                     da_int k_components, const float *Y, da_int ldy,
                                     float *Y_inv_transform, da_int ldy_inv_transform);
/** \} */

#endif
