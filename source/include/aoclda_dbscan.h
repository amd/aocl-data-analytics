/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#ifndef AOCLDA_DBSCAN
#define AOCLDA_DBSCAN

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"

/**
 * \file
 */

/** \{
 * \brief Pass a data matrix to the \ref da_handle object in preparation for DBSCAN clustering.
 *
 * The data itself is not copied; a pointer to the data matrix is stored instead.
 * @rst
 * After calling this function you may use the option setting APIs to set :ref:`options <dbscan_options>`.
 * @endrst
 *
 * \param[inout] handle a \ref da_handle object, initialized with type \ref da_handle_dbscan.
 * \param[in] n_samples the number of rows of the data matrix, \p A. Constraint: \p n_samples @f$\ge@f$ 1.
 * \param[in] n_features the number of columns of the data matrix, \p A. Constraint: \p n_features @f$\ge@f$ 1.
 * \param[in] A the \p n_samples @f$\times@f$ \p n_features data matrix. By default, it should be stored in column-major order, unless you have set the <em>storage order</em> option to <em>row-major</em>.
 * \param[in] lda the leading dimension of the data matrix. Constraint: \p lda @f$\ge@f$ \p n_samples if \p A is stored in column-major order, or \p lda @f$\ge@f$ \p n_features if \p A is stored in row-major order.
  * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized with the wrong precision.
 * - \ref da_status_invalid_pointer - the handle has not been initialized, or \p A is null.
 * - \ref da_status_invalid_input - one of the arguments had an invalid value. You can obtain further information using \ref da_handle_print_error_message.
 * - \ref da_status_invalid_leading_dimension - the constraint on \p lda was violated.
 */
da_status da_dbscan_set_data_d(da_handle handle, da_int n_samples, da_int n_features,
                               const double *A, da_int lda);

da_status da_dbscan_set_data_s(da_handle handle, da_int n_samples, da_int n_features,
                               const float *A, da_int lda);
/** \} */

/** \{
 * \brief Compute DBSCAN clustering
 *
 * @rst
 * Computes DBSCAN clustering on the data matrix previously passed into the handle using :ref:`da_dbscan_set_data_? <da_dbscan_set_data>`.
 * @endrst
 *
 * \param[inout] handle a \ref da_handle object, initialized
 *  with type \ref da_handle_dbscan and with data passed in via \ref da_dbscan_set_data_s "da_dbscan_set_data_?".
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_wrong_type - the handle may have been initialized using the wrong precision.
 * - \ref da_status_invalid_pointer - the handle has not been initialized.
 * - \ref da_status_no_data - \ref da_dbscan_set_data_s "da_dbscan_set_data_?" has not been called prior to this function call.
 * - \ref da_status_internal_error - this can occur if your data contains undefined values.
 * - \ref da_status_incompatible_options - you can obtain further information using \ref da_handle_print_error_message.

 *
 * \post
 * \parblock
 * After successful execution, \ref da_handle_get_result_s "da_handle_get_result_?" can be queried with the following enum for floating-point output:
 * - \p da_rinfo - return an array of size 9 containing the values of \p n_samples, \p n_features, \p lda, \p eps, \p min_samples, \p leaf_size, \p p, \p n_core_samples and \p n_clusters.
 *
 * In addition \ref da_handle_get_result_int can be queried with the following enums:
 * - \p da_dbscan_n_clusters - return the number of clusters found.
 * - \p da_dbscan_n_core_samples - return the number of core samples found, \p n_core_samples.
 * - \p da_dbscan_labels - return an array of size \p n_samples containing the label (i.e. which cluster it is in) of each sample point. A label of -1 indicates that the point has been classified as noise and has not been assigned to a cluster.
 * - \p da_dbscan_core_sample_indices - return an array of size \p n_core_samples containing the indices of the core samples.
 * \endparblock
 */
da_status da_dbscan_compute_d(da_handle handle);

da_status da_dbscan_compute_s(da_handle handle);
/** \} */

#endif
