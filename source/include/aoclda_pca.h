/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

/**
 * \file
 * \anchor chapter_f
 * \brief Chapter F - Factorizatoin
 *
 * This section descripes about all factorization API in AOCL-DA
 *
 * \section chpc_intro Introduction
 * \section chc_fac Facrorization
 * \subsection chc_pca Principal Component Analysis
 * \subsection chc_qr QR Factorization
 * \subsection chc_chol Cholesky Factorization *
 */

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** \{
 * \brief enum to specify PCA compute method
 *  pca_method_svd=0 
 *  pca_method_corr=1
 */
typedef enum pca_comp_method_ { pca_method_svd = 0, pca_method_corr = 1 } pca_comp_method;

/** \{
 * \brief enum to get specific PCA results 
 */
typedef enum pca_results_flags_ {
    pca_components = 1,
    pca_scores = 2,
    pca_variance = 4,
    pca_total_variance = 8
} pca_results_flags;

/** \{
 * \brief Create and init PCA data structures
 *
 * Creates pca_d handle and allocates the memory/buffers required 
 * based on given inputs n and p to perform PCA on given data.
 * 
 * Initializze the algo parameters with default values
 * 
 * Copies the input data into temporary local buffer
 * 
 * \param[in,out] handle a \ref da_handle object, initialized using \ref da_handle_init 
 * with type \ref da_handle_pca.
 * \param[in] n the number of inputs
 * \param[in] p the number of features per input
 * \param[in] A the pointer to the input data
  * \return \ref da_status_. The function returns:
 * - \ref da_status_success - the operation was successfully completed
 * - \ref da_status_memory_error = 2 - When the handle is not created
 * - \ref da_status_wrong_type = 7 = when the input type is not matching with api
 * - \ref da_status_invalid_pointer = 3 - When the pca handle is not created
 */
da_status da_pca_d_init(da_handle handle, da_int n, da_int p, double *A);
da_status da_pca_s_init(da_handle handle, da_int n, da_int p, float *A);

da_status da_pca_set_method(da_handle, pca_comp_method method);
da_status da_pca_set_num_components(da_handle, da_int num_components);

/** \{
 * \brief Create and init PCA data structures
 *
 * Computes PCA on given input A and options set by user through init function through the handler
 * 
 * \param[in,out] handle a \ref da_handle object, initialized using \ref da_handle_init 
 *  with type \ref da_handle_pca.
  * \return \ref da_status_. The function returns:
 * - \ref da_status_success - the operation was successfully completed
 * - \ref da_status_invalid_input - When input parameters are not valid
 * - \ref da_status_memory_error = 2 - When the handle is not created
 * - \ref da_status_wrong_type = 7 = when the input type is not matching with api
 * - \ref da_status_invalid_pointer = 3 - When the pca handle is not initialized
 * - \ref da_status_internal_error =  1 - When lapack/internal routines error out
 * - \ref da_status_not_implemented = 5 - When requested features/method is not implemented/supported
 */
da_status da_pca_d_compute(da_handle);
da_status da_pca_s_compute(da_handle);

#ifdef __cplusplus
}
#endif

#endif
