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
typedef enum da_metric_ {
    da_euclidean,   ///< Use euclidean distance.
    da_sqeuclidean, ///< Use squared euclidean distance.
    da_minkowski,   ///< Use Minkowski distance.
    da_manhattan    ///< Use Manhattan distance.
} da_metric;

/**
 * \brief Defines the type of data that is allowed as inputs when pairwise distances are computed.
 **/
typedef enum da_data_types_ {
    da_all_finite,     ///< Force all values to be finite.
    da_allow_infinite, ///< Allow infinite and NaN values in input data.
    da_allow_NaN       ///< Allow NaN values in input data.
} da_data_types;

/** \{
 * \brief Compute the distance matrix for an \p m by \p k matrix \p X and optionally an \p n by \p k matrix \p Y (both in column major order).
 *
 * \param[in] m the number of rows of matrix X.
 * \param[in] n the number of rows of matrix Y.
 * \param[in] k the number of columns of matrices X and Y.
 * \param[in] X the \p m @f$\times @f$ \p k matrix. Data is expected to be stored in column major order, so that the element in the @f$i@f$th row and @f$j@f$th column (indexed from 0, so that @f$0 \le i \le@f$ \p m @f$-1@f$ and @f$0 \le j \le@f$ \p k @f$-1@f$) is stored in the [<i>j</i> @f$\times@f$ \p ldx + <i>i</i>]th entry of \p X.
 * \param[in] ldx the leading dimension of the matrix X. Constraint: \p ldx @f$\ge@f$ \p m.
 * \param[in] Y the \p n @f$\times @f$ \p k matrix. Data is expected to be stored in column major order, so that the element in the @f$i@f$th row and @f$j@f$th column (indexed from 0, so that @f$0 \le i \le@f$ \p n @f$-1@f$ and @f$0 \le j \le@f$ \p k @f$-1@f$) is stored in the [<i>j</i> @f$\times@f$ \p ldy + <i>i</i>]th entry of \p Y.
 * \param[in] ldy the leading dimension of the matrix Y. Constraint: \p ldy @f$\ge@f$ \p n.
 * \param[out] D if Y is nullptr, the \p m @f$\times @f$ \p m distance matrix, and the \p m @f$\times @f$ \p n distance matrix, otherwise. Data is expected to be stored in column major order, so that the element in the @f$i@f$th row and @f$j@f$th column (indexed from 0, so that @f$0 \le i \le@f$ \p m @f$-1@f$ and @f$0 \le j \le@f$ \p n @f$-1@f$) is stored in the [<i>j</i> @f$\times@f$ \p ldd + <i>i</i>]th entry of \p D.
 * \param[in] ldd the leading dimension of the matrix D. Constraint: \p ldd @f$\ge@f$ \p m, if Y is nullptr, and \p ldd @f$\ge@f$ \p n, otherwise.
 * \param[in] metric enum that specifies the metric to use to compute the distance matrix. The default value is \ref da_euclidean.
 * \param[in] force_all_finite enum that specifies whether to raise an error on infinite or NaN values.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - one of the constraints \p ldx @f$\ge@f$ \p m, \p ldy @f$\ge@f$ \p n, or \p ldd @f$\ge@f$ \p m or \p n was violated.
 * - \ref da_status_invalid_pointer - one of the arrays \p X or \p D is null.
 * - \ref da_status_invalid_array_dimension - either \p m @f$< 1@f$, or \p k @f$< 1@f$, or \p n @f$< 1@f$, while Y is not nullptr.
 * - \ref da_status_not_implemented - an option that is currently not implemented was set.
 * - \ref da_status_memory_error - a memory allocation error occurred.
 */
da_status da_pairwise_distances_d(da_int m, da_int n, da_int k, const double *X,
                                  da_int ldx, const double *Y, da_int ldy, double *D,
                                  da_int ldd, da_metric metric,
                                  da_data_types force_all_finite);

da_status da_pairwise_distances_s(da_int m, da_int n, da_int k, const float *X,
                                  da_int ldx, const float *Y, da_int ldy, float *D,
                                  da_int ldd, da_metric metric,
                                  da_data_types force_all_finite);
/** \} */

#ifdef __cplusplus
}
#endif

#endif
