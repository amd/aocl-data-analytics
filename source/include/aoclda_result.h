/*
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef AOCLDA_RESULT
#define AOCLDA_RESULT

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
 * \brief Enumeration defining which result
 * to extract from a handle
 **/
typedef enum da_result_ {
    // General purpose data 1..100
    da_rinfo =
        1, ///< General information array, containing a variety of metrics. See each solver's documentation for further information, since each solver stores different information in this array.
    // Linear models 101..200
    da_linmod_coeff =
        101, ///< Optimal fitted coefficients produced by the last call to a linear regression solver.
    // PCA 201..300
    da_pca_scores = 201, ///< Matrix of scores computed by the PCA API.
    da_pca_variance, ///< The variance explained by each component computed by the PCA API.
    da_pca_principal_components, ///< Matrix of principal components computed by the PCA API.
    da_pca_total_variance, ///< The toal variance of the data matrix supplied to the PCA API.
    da_pca_column_means, ///< Column means of the data matrix supplied to the PCA API.
    da_pca_column_sdevs, ///< Column standard deviations of the data matrix supplied to the PCA API.
    da_pca_transformed_data, ///< Transformed data matrix computed by projecting into principal component space following a PCA computation.
    da_pca_inverse_transformed_data, ///< Inverse transformed data matrix computed by projecting from principal component space into the original coordinate space used by a PCA computation.
    da_pca_u, ///< The matrix @f$U@f$ from the singular value decomposition @f$A = U\Sigma V^T@f$, computed as part of a PCA computation.
    da_pca_sigma, ///< The nonzero diagonal entries of @f$\Sigma@f$ from the singular value decomposition @f$A = U\Sigma V^T@f$, computed as part of a PCA computation.
    da_pca_vt, ///< The matrix @f$V^T@f$ from the singular value decomposition @f$A = U\Sigma V^T@f$, computed as part of a PCA computation.
    // Nonlinear Optimization 301..400
    // Random Forrests 401..500
    // ...
} da_result;

/** \{
 * @brief Get results stored in a \ref da_handle
 *
 * Some solvers will store relevant data in the handle. These functions provide a means to extract it.
 * To check the available data stored by a given API check its associated documentation.
 *
 * @param handle a valid handle used to call any of the solvers.
 * @param query the data of interest, see \ref da_result. If the result is not avaible or not found in the \p handle then the function returns \ref da_status_unknown_query.
 * @param dim the size of the array \p result. If \p dim is too small, then on exit it will be overwritten with the correct size and the function will return \ref da_status_invalid_array_dimension.
 * @param result location of the array in which to store the data.
 *
 * @return \ref da_status. The function returns:
 * - \ref da_status_success - the operation completed successfully.
 * - \ref da_status_unknown_query - the \p query
 *         is either not available or not found in the \p handle.
 *         This can happen if you try an extract a result before performing the operation (for example, 
 *         extracting the coefficient of a linear regression before actually performing a
 *         sucessful fit), or if the handle is of the wrong type (for example, a handle
 *         initialized for linear models cannot contain results about a principal component analysis).
 * - \ref da_status_wrong_type - the floating point precision used to initialize the
 *         \p handle does not match the precision of \p result.
 * - \ref da_status_handle_not_initialized - the \p handle has not been initialized or is corrupted.
 * - \ref da_status_invalid_array_dimension - the size \p dim of the \p result
 *         array is too small. After the call \p dim contains the correct size.
 * - \ref da_status_invalid_pointer - the pointer to \p handle is invalid.
 */
da_status da_handle_get_result_d(const da_handle handle, da_result query, da_int *dim,
                                 double *result);

da_status da_handle_get_result_s(const da_handle handle, da_result query, da_int *dim,
                                 float *result);
/** \} */

/** \{
 * @brief Get results stored in a \ref da_handle
 *
 * Some solvers will store relevant data in the handle. These functions provide a means to extract it.
 * To check the available data stored by a given API check its associated documentation.
 *
 * @param handle a valid handle used to call any of the solvers.
 * @param query  the data of interest, see \ref da_result. If the result is not avaible or not found in the \p handle then the function returns \ref da_status_unknown_query.
 * @param dim the size of the array \p result. If \p dim is too small, then on exit it will be overwritten with the correct size and the function will return \ref da_status_invalid_array_dimension.
 * @param result location of the array in which to store the data.
 *
 * @return \ref da_status. The function returns:
 * - \ref da_status_success - the operation completed successfully.
 * - \ref da_status_unknown_query - the \p query
 *         is either not available or not found in the \p handle.
 *         This can happen if you try an extract a result before performing the operation (for example, 
 *         extracting the coefficient of a linear regression before actually performing a
 *         sucessful fit), or if the handle is of the wrong type (for example, a handle
 *         initialized for linear models cannot contain results about a principal component analysis).
 *- \ref da_status_handle_not_initialized - the \p handle has not been initialized or is corrupted.
 * - \ref da_status_invalid_array_dimension - the size \p dim of the \p result
 *         array is too small. After the call \p dim contains the correct size.
 * - \ref da_status_invalid_pointer - the pointer to \p handle is invalid.
 */
da_status da_handle_get_result_int(const da_handle handle, da_result query, da_int *dim,
                                   da_int *result);

#ifdef __cplusplus
}
#endif

#endif
