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

#ifndef AOCLDA_UTILS
#define AOCLDA_UTILS

#include "aoclda_error.h"
#include "aoclda_types.h"

/**
 * \file
 */

/** \{
 * \brief Check a data matrix for NaNs.
 *
 * Return an error if a data matrix is found to contain any NaNs.
 *
 * \param[in] order a \ref da_order enumerated type, specifying whether \p X is stored in row-major order or column-major order.
 * \param[in] n_rows the number of rows in the data matrix. Constraint: \p n_rows @f$\ge 1@f$.
 * \param[in] n_cols the number of columns in the data matrix. Constraint: \p n_cols @f$\ge 1@f$.
 * \param[in] X the \p n_rows @f$\times @f$ \p n_cols data matrix.
 * \param[in] ldx the leading dimension of the data matrix. Constraint: \p ldx @f$\ge@f$ \p n_rows if \p order = \p column_major, or \p ldx @f$\ge@f$ \p n_cols if \p order = \p row_major.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_input - a NaN was found in \p X.
 * - \ref da_status_invalid_leading_dimension - the constraint on \p ldx was violated.
 * - \ref da_status_invalid_array_dimension - either \p n_rows @f$< 1@f$ or \p n_cols @f$< 1@f$.
 */
da_status da_check_data_d(da_order order, da_int n_rows, da_int n_cols, const double *X,
                          da_int ldx);
da_status da_check_data_s(da_order order, da_int n_rows, da_int n_cols, const float *X,
                          da_int ldx);
/** \} */

/** \{
 * \brief Copy and convert an array from row-major order to column-major order or vice versa.
 *
 * Either copy a column-major array into a new array stored in row-major order or copy a row-major array into a new array stored in column-major order.
 *
 * \param[in] order_X a \ref da_order enumerated type, specifying whether \p X is stored in row-major order or column-major order. \p Y will then be returned with the opposite ordering scheme.
 * \param[in] n_rows the number of rows in \p X. Constraint: \p n_rows @f$\ge 1@f$.
 * \param[in] n_cols the number of columns in \p X. Constraint: \p n_cols @f$\ge 1@f$.
 * \param[in] X the \p n_rows @f$\times @f$ \p n_cols data matrix.
 * \param[in] ldx the leading dimension of \p X. Constraint: \p ldx @f$\ge@f$ \p n_rows if \p order_X = \p column_major, or \p ldx @f$\ge@f$ \p n_cols if \p order_X = \p row_major.
 * \param[out] Y the \p n_rows @f$\times @f$ \p n_cols output matrix containing the same values as \p X, but with the opposite ordering scheme.
 * \param[in] ldy the leading dimension of \p Y. Constraint: \p ldy @f$\ge@f$ \p n_cols if \p order_X = \p column_major, or \p ldy @f$\ge@f$ \p n_rows if \p order_X = \p row_major.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - one of the constraints on \p ldx or \p ldy was violated.
 * - \ref da_status_invalid_array_dimension - either \p n_rows @f$< 1@f$ or \p n_cols @f$< 1@f$.
 * - \ref da_status_invalid_pointer - one of the arrays \p X or \p Y was null.
 */
da_status da_switch_order_copy_d(da_order order, da_int n_rows, da_int n_cols,
                                 const double *X, da_int ldx, double *Y, da_int ldy);
da_status da_switch_order_copy_s(da_order order, da_int n_rows, da_int n_cols,
                                 const float *X, da_int ldx, float *Y, da_int ldy);
/** \} */

/** \{
 * \brief Convert an array from row-major order to column-major order or vice versa, in place.
 *
 * Either convert a column-major array into row-major order or convert a row-major array into column-major order, overwriting the input array with the converted output array.
 *
 * \param[in] order_X_in a \ref da_order enumerated type, specifying whether \p X is supplied in row-major order or column-major order. \p X will then be returned with the opposite ordering scheme.
 * \param[in] n_rows the number of rows in \p X. Constraint: \p n_rows @f$\ge 1@f$.
 * \param[in] n_cols the number of columns in \p X. Constraint: \p n_cols @f$\ge 1@f$.
 * \param[inout] X the \p n_rows @f$\times @f$ \p n_cols data matrix.
 * \param[in] ldx_in the leading dimension of \p X on entry. Constraint: \p ldx_in @f$\ge@f$ \p n_rows if \p order_X_in = \p column_major, or \p ldx_in @f$\ge@f$ \p n_cols if \p order_X_in = \p row_major.
 * \param[in] ldx_out the required leading dimension of \p X on exit. Constraint: \p ldx_out @f$\ge@f$ \p n_cols if \p order_X_in = \p column_major, or \p ldx_out @f$\ge@f$ \p n_rows if \p order_X_out = \p row_major.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - one of the constraints on \p ldx_in or \p ldx_out was violated.
 * - \ref da_status_invalid_array_dimension - either \p n_rows @f$< 1@f$ or \p n_cols @f$< 1@f$.
 * - \ref da_status_invalid_pointer - the array \p X was null.
 */
da_status da_switch_order_in_place_d(da_order order_X_in, da_int n_rows, da_int n_cols,
                                     double *X, da_int ldx_in, da_int ldx_out);
da_status da_switch_order_in_place_s(da_order order_X_in, da_int n_rows, da_int n_cols,
                                     float *X, da_int ldx_in, da_int ldx_out);
/** \} */

#endif
