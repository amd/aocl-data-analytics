/* ************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */
#ifndef AOCLDA_TSNE
#define AOCLDA_TSNE

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"

/**
 * \file
 */

/** \{
 * \brief Pass a data matrix to the \ref da_handle object in preparation for computing <i>t</i>-SNE.
 *
 * Depending on options and layout, the data may be referenced directly or copied
 * into internal storage (for example, when normalization or a layout conversion
 * is required).
 * @rst
 * After calling this function you may use the option setting APIs to set :ref:`options <tsne_options>`.
 * @endrst
 *
 * \param[inout] handle a \ref da_handle object, initialized using \ref da_handle_init_s "da_handle_init_?"
 *  with type \ref da_handle_tsne.
 * \param[in] n_samples the number of rows of the data matrix, \p X. Constraint: \p n_samples @f$\ge@f$ 2.
 * \param[in] n_features the number of columns of the data matrix, \p X. Constraint: \p n_features @f$\ge@f$ 1.
 * \param[in] X the \p n_samples @f$\times@f$ \p n_features data matrix. By default, it should be stored in column-major order, unless you have set the <em>storage order</em> option to <em>row-major</em>.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: \p ldx @f$\ge@f$ \p n_samples if \p X is stored in column-major order, or \p ldx @f$\ge@f$ \p n_features if \p X is stored in row-major order.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using the wrong precision.
 * - \ref da_status_handle_not_initialized - the handle has not been initialized.
 * - \ref da_status_invalid_pointer - \p X is null.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_invalid_leading_dimension - the constraint on \p ldx was violated.
 * - \ref da_status_incompatible_options - if you have already set the number of components or perplexity and it is too high, then it will be reduced accordingly, and this warning returned.
 */
da_status da_tsne_set_data_d(da_handle handle, da_int n_samples, da_int n_features,
                             const double *X, da_int ldx);

da_status da_tsne_set_data_s(da_handle handle, da_int n_samples, da_int n_features,
                             const float *X, da_int ldx);
/** \} */

/** \{
 * \brief Supply an explicit initial embedding for <i>t</i>-SNE.
 *
 * Use this API with the <em>init</em> option set to <em>supplied</em> to start optimization
 * from a user-provided embedding. \ref da_tsne_set_data_s "da_tsne_set_data_?" must be
 * called before this function so that \p n_samples is known.
 *
 * \param[inout] handle a \ref da_handle object, initialized using \ref da_handle_init_s "da_handle_init_?"
 *  with type \ref da_handle_tsne, and with data already passed in via
 *  \ref da_tsne_set_data_s "da_tsne_set_data_?".
 * \param[in] Y initial embedding matrix of shape \p n_samples @f$\times@f$ \p n_components, in the storage order specified by the <em>storage order</em> option.
 * \param[in] ldy leading dimension of \p Y. Constraint: \p ldy @f$\ge@f$ \p n_samples if \p Y is stored in column-major order, or \p ldy @f$\ge@f$ \p n_components if \p Y is stored in row-major order.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using the wrong precision.
 * - \ref da_status_handle_not_initialized - the handle has not been initialized.
 * - \ref da_status_invalid_handle_type - the handle was not initialized with type \ref da_handle_tsne.
 * - \ref da_status_no_data - \ref da_tsne_set_data_s "da_tsne_set_data_?" has not been called prior to this function call.
 * - \ref da_status_invalid_pointer - \p Y is null.
 * - \ref da_status_invalid_leading_dimension - the constraint on \p ldy was violated.
 * - \ref da_status_memory_error - a memory allocation error occurred.
 */
da_status da_tsne_set_init_embedding_d(da_handle handle, const double *Y, da_int ldy);

da_status da_tsne_set_init_embedding_s(da_handle handle, const float *Y, da_int ldy);
/** \} */

/** \{
 * \brief Compute <i>t</i>-SNE
 *
 * Computes <i>t</i>-SNE on the data matrix previously passed into the handle using
 * \ref da_tsne_set_data_s "da_tsne_set_data_?".
 *
 * \param[inout] handle a \ref da_handle object, initialized using \ref da_handle_init_s "da_handle_init_?"
 *  with type \ref da_handle_tsne and with data passed in via
 *  \ref da_tsne_set_data_s "da_tsne_set_data_?".
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using the wrong precision.
 * - \ref da_status_handle_not_initialized - the handle has not been initialized.
 * - \ref da_status_no_data - \ref da_tsne_set_data_s "da_tsne_set_data_?" has not been called prior to this function call.
 * - \ref da_status_internal_error - this can occur if your data contains undefined values.
 *
 * \post
 * \parblock
 * After successful execution, \ref da_handle_get_result_s "da_handle_get_result_?" can be queried with:
 * - \p da_tsne_embedding - return an array of size \p n_samples @f$\times@f$ \p n_components containing the low-dimensional embedding, in the same storage order as the input data.
 * - \p da_rinfo - return an array of size 6 containing \p n_samples, \p n_features, \p n_components, \p n_iter, the final KL divergence and the number of low precision iterations if mixed precision is enabled.
 * \endparblock
 */
da_status da_tsne_compute_d(da_handle handle);

da_status da_tsne_compute_s(da_handle handle);
/** \} */

#endif
