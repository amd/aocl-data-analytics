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

#ifndef AOCLDA_METRICS
#define AOCLDA_METRICS

#include "aoclda_error.h"
#include "aoclda_types.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * \file
 */

/**
 * \brief Defines the metric used when calculating the pairwise distances.
 **/
enum da_metric_ {
    da_euclidean,   ///< Use euclidean distance.
    da_sqeuclidean, ///< Use squared euclidean distance.
    da_minkowski,   ///< Use Minkowski distance (used internally only).
    da_manhattan    ///< Use Manhattan distance (used internally only).
};

/** @brief Alias for the \ref da_metric_ enum. */
typedef enum da_metric_ da_metric;

/**
 * \brief Defines the type of data that is allowed as inputs when pairwise distances are computed.
 **/
enum da_data_types_ {
    da_all_finite,     ///< Force all values to be finite.
    da_allow_infinite, ///< Allow infinite and NaN values in input data.
    da_allow_NaN       ///< Allow NaN values in input data.
};

/** @brief Alias for the \ref da_data_types_ enum. */
typedef enum da_data_types_ da_data_types;

/** \{
 * \brief Compute the distance matrix for an \p m by \p k matrix \p X and optionally an \p n by \p k matrix \p Y.
 *
 * \param[in] order a \ref da_order enumerated type, specifying whether \p X, \p Y and \p D are stored in row-major order or column-major order.
 * \param[in] m the number of rows of matrix \p X.
 * \param[in] n the number of rows of matrix \p Y.
 * \param[in] k the number of columns of matrices \p X and \p Y.
 * \param[in] X the \p m @f$\times @f$ \p k matrix.
 * \param[in] ldx the leading dimension of the matrix \p X. Constraint: \p ldx @f$\ge@f$ \p m if \p order = \p column_major, or \p ldx @f$\ge@f$ \p k if \p order = \p row_major.
 * \param[in] Y the \p n @f$\times @f$ \p k matrix.
 * \param[in] ldy the leading dimension of the matrix \p Y. Constraint: \p ldy @f$\ge@f$ \p n if \p order = \p column_major, or \p ldy @f$\ge@f$ \p k if \p order = \p row_major.
 * \param[out] D if Y is nullptr, the \p m @f$\times @f$ \p m distance matrix, and the \p m @f$\times @f$ \p n distance matrix, otherwise.
 * \param[in] ldd the leading dimension of the matrix D. Constraint: \p ldd @f$\ge@f$ \p m, if \p Y is nullptr or \p order = \p column_major, and \p ldd @f$\ge@f$ \p n, otherwise.
 * \param[in] metric enum that specifies the metric to use to compute the distance matrix. The default value is \ref da_euclidean.
 * \param[in] force_all_finite enum that specifies whether to raise an error on infinite or NaN values.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - one of the constraints on \p ldx, \p ldy or \p ldd was violated.
 * - \ref da_status_invalid_pointer - one of the arrays \p X or \p D is null.
 * - \ref da_status_invalid_array_dimension - either \p m @f$< 1@f$, or \p k @f$< 1@f$, or \p n @f$< 1@f$, while \p Y is not nullptr.
 * - \ref da_status_not_implemented - an option that is currently not implemented was set.
 * - \ref da_status_memory_error - a memory allocation error occurred.
 */
da_status da_pairwise_distances_d(da_order order, da_int m, da_int n, da_int k,
                                  const double *X, da_int ldx, const double *Y,
                                  da_int ldy, double *D, da_int ldd, da_metric metric,
                                  da_data_types force_all_finite);

da_status da_pairwise_distances_s(da_order order, da_int m, da_int n, da_int k,
                                  const float *X, da_int ldx, const float *Y, da_int ldy,
                                  float *D, da_int ldd, da_metric metric,
                                  da_data_types force_all_finite);
/** \} */

#ifdef __cplusplus
}
#endif

#endif
