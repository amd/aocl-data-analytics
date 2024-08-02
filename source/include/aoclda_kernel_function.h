/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef AOCLDA_KERNEL_FUNCTION
#define AOCLDA_KERNEL_FUNCTION

#include "aoclda_error.h"
#include "aoclda_types.h"
#include <stdbool.h>

/**
 * \file
 */

/** \{
 * @brief Compute the RBF (Radial Basis Function) kernel matrix for the matrices \p X and, optionally, \p Y.
 * @rst
 * The last suffix of the function name marks the floating point precision on which the handle operates (see :ref:`precision section <da_real_prec>`).
 * @endrst
 *
 * This function computes the RBF kernel between the matrix \p X (size \p m @f$\times@f$ \p k) and \p Y (size \p n @f$\times@f$ \p k) if provided.
 * If \p Y is null, it computes the kernel of \p X with itself (@f$XX^T@f$). The results are stored in \p D.
 *
 * @param[in] order @ref da_order enum specifying column-major or row-major layout.
 * @param[in] m the number of rows of matrix X. Constraint: @p m @f$\ge@f$ 1.
 * @param[in] n the number of rows of matrix Y. Constraint: @p n @f$\ge@f$ 1.
 * @param[in] k the number of columns of matrices X and Y. Constraint: @p k @f$\ge@f$ 1.
 * @param[in] X the matrix of size \p m @f$\times@f$ \p k, stored in column-major order by default.
 * @param[in] ldx the leading dimension of \p X. Constraint: \p ldx @f$\ge@f$ \p m if \p order = \p column_major, or \p ldx @f$\ge@f$ \p k if \p order = \p row_major.
 * @param[in] Y the matrix of size \p n @f$\times@f$ \p k, or null if computing the kernel of \p X with itself.
 * @param[in] ldy the leading dimension of \p Y. Constraint: \p ldy @f$\ge@f$ \p n if \p order = \p column_major, or \p ldy @f$\ge@f$ \p k if \p order = \p row_major.
 * @param[out] D the resulting kernel matrix of size \p m @f$\times@f$ \p n if \p Y is not null, or \p m @f$\times@f$ \p m otherwise.
 * @param[in] ldd the leading dimension of the matrix \p D. Constraint: \p ldd @f$\ge@f$ \p m, if \p Y is nullptr or \p order = \p column_major, and \p ldd @f$\ge@f$ \p n, otherwise.
 * @param[in] gamma the RBF kernel scale factor. Constraint: \p gamma @f$\ge@f$ 0.
 * @return @ref da_status
 * - @ref da_status_success - operation completed successfully.
 * - @ref da_status_invalid_leading_dimension - one of the constraints on \p ldx, \p ldy, or \p ldd was violated.
 * - @ref da_status_invalid_pointer - one of the input pointers is null.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value.
 * - @ref da_status_invalid_array_dimension - one of the dimensions \p m, \p n, or \p k is invalid.
 * - @ref da_status_memory_error - unable to allocate memory.
 */
da_status da_rbf_kernel_d(da_order order, da_int m, da_int n, da_int k, const double *X,
                          da_int ldx, const double *Y, da_int ldy, double *D, da_int ldd,
                          double gamma);

da_status da_rbf_kernel_s(da_order order, da_int m, da_int n, da_int k, const float *X,
                          da_int ldx, const float *Y, da_int ldy, float *D, da_int ldd,
                          float gamma);
/** \} */

/** \{
 * @brief Compute the linear kernel matrix for the matrices \p X and, optionally, \p Y.
 * @rst
 * The last suffix of the function name marks the floating point precision on which the handle operates (see :ref:`precision section <da_real_prec>`).
 * @endrst
 *
 * This function computes the linear kernel between the rows of \p X (size \p m @f$\times@f$ \p k) and \p Y (size \p n @f$\times@f$ \p k) if provided.
 * If \p Y is null, it computes the kernel of \p X with itself. The results are stored in \p D.
 *
 * @param[in] order @ref da_order enum specifying column-major or row-major layout.
 * @param[in] m the number of rows of matrix X. Constraint: @p m @f$\ge@f$ 1.
 * @param[in] n the number of rows of matrix Y. Constraint: @p n @f$\ge@f$ 1.
 * @param[in] k the number of columns of matrices X and Y. Constraint: @p k @f$\ge@f$ 1.
 * @param[in] X the matrix of size \p m @f$\times@f$ \p k, stored in column-major order by default.
 * @param[in] ldx the leading dimension of \p X. Constraint: \p ldx @f$\ge@f$ \p m if \p order = \p column_major, or \p ldx @f$\ge@f$ \p k if \p order = \p row_major.
 * @param[in] Y the matrix of size \p n @f$\times@f$ \p k, or null if computing the kernel of \p X with itself.
 * @param[in] ldy the leading dimension of \p Y. Constraint: \p ldy @f$\ge@f$ \p n if \p order = \p column_major, or \p ldy @f$\ge@f$ \p k if \p order = \p row_major.
 * @param[out] D the resulting kernel matrix of size \p m @f$\times@f$ \p n if \p Y is not null, or \p m @f$\times@f$ \p m otherwise.
 * @param[in] ldd the leading dimension of the matrix \p D. Constraint: \p ldd @f$\ge@f$ \p m, if \p Y is nullptr or \p order = \p column_major, and \p ldd @f$\ge@f$ \p n, otherwise.
 * @return @ref da_status
 * - @ref da_status_success - operation completed successfully.
 * - @ref da_status_invalid_leading_dimension - one of the constraints on \p ldx, \p ldy, or \p ldd was violated.
 * - @ref da_status_invalid_pointer - one of the input pointers is null.
 * - @ref da_status_invalid_array_dimension - one of the dimensions \p m, \p n, or \p k is invalid.
 * - @ref da_status_memory_error - unable to allocate memory.
 */
da_status da_linear_kernel_d(da_order order, da_int m, da_int n, da_int k,
                             const double *X, da_int ldx, const double *Y, da_int ldy,
                             double *D, da_int ldd);

da_status da_linear_kernel_s(da_order order, da_int m, da_int n, da_int k, const float *X,
                             da_int ldx, const float *Y, da_int ldy, float *D,
                             da_int ldd);
/** \} */

/** \{
 * @brief Compute the polynomial kernel matrix for the matrices \p X and, optionally, \p Y.
 * @rst
 * The last suffix of the function name marks the floating point precision on which the handle operates (see :ref:`precision section <da_real_prec>`).
 * @endrst
 *
 * This function computes the polynomial kernel between the rows of \p X (size \p m @f$\times@f$ \p k) and \p Y (size \p n @f$\times@f$ \p k) if provided.
 * If \p Y is null, it computes the kernel of \p X with itself. The results are stored in \p D.
 *
 * @param[in] order @ref da_order enum specifying column-major or row-major layout.
 * @param[in] m the number of rows of matrix X. Constraint: @p m @f$\ge@f$ 1.
 * @param[in] n the number of rows of matrix Y. Constraint: @p n @f$\ge@f$ 1.
 * @param[in] k the number of columns of matrices X and Y. Constraint: @p k @f$\ge@f$ 1.
 * @param[in] X the matrix of size \p m @f$\times@f$ \p k, stored in column-major order by default.
 * @param[in] ldx the leading dimension of \p X. Constraint: \p ldx @f$\ge@f$ \p m if \p order = \p column_major, or \p ldx @f$\ge@f$ \p k if \p order = \p row_major.
 * @param[in] Y the matrix of size \p n @f$\times@f$ \p k, or null if computing the kernel of \p X with itself.
 * @param[in] ldy the leading dimension of \p Y. Constraint: \p ldy @f$\ge@f$ \p n if \p order = \p column_major, or \p ldy @f$\ge@f$ \p k if \p order = \p row_major.
 * @param[out] D the resulting kernel matrix of size \p m @f$\times@f$ \p n if \p Y is not null, or \p m @f$\times@f$ \p m otherwise.
 * @param[in] ldd the leading dimension of the matrix \p D. Constraint: \p ldd @f$\ge@f$ \p m, if \p Y is nullptr or \p order = \p column_major, and \p ldd @f$\ge@f$ \p n, otherwise.
 * @param[in] gamma the scale factor used in polynomial kernel. Constraint: \p gamma @f$\ge@f$ 0.
 * @param[in] degree the degree of the polynomial kernel.  Constraint: \p degree @f$\ge@f$ 0.
 * @param[in] coef0 the independent term in the polynomial kernel.
 * @return @ref da_status
 * - @ref da_status_success - operation completed successfully.
 * - @ref da_status_invalid_leading_dimension - one of the constraints on \p ldx, \p ldy, or \p ldd was violated.
 * - @ref da_status_invalid_pointer - one of the input pointers is null.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value.
 * - @ref da_status_invalid_array_dimension - one of the dimensions \p m, \p n, or \p k is invalid.
 * - @ref da_status_memory_error - unable to allocate memory.
 */
da_status da_polynomial_kernel_d(da_order order, da_int m, da_int n, da_int k,
                                 const double *X, da_int ldx, const double *Y, da_int ldy,
                                 double *D, da_int ldd, double gamma, da_int degree,
                                 double coef0);

da_status da_polynomial_kernel_s(da_order order, da_int m, da_int n, da_int k,
                                 const float *X, da_int ldx, const float *Y, da_int ldy,
                                 float *D, da_int ldd, float gamma, da_int degree,
                                 float coef0);
/** \} */

/** \{
 * @brief Compute the sigmoid kernel matrix for the matrices \p X and, optionally, \p Y.
 * @rst
 * The last suffix of the function name marks the floating point precision on which the handle operates (see :ref:`precision section <da_real_prec>`).
 * @endrst
 *
 * This function computes the sigmoid kernel between the rows of \p X (size \p m @f$\times@f$ \p k) and \p Y (size \p n @f$\times@f$ \p k) if provided.
 * If \p Y is null, it computes the kernel of \p X with itself. The results are stored in \p D.
 *
 * @param[in] order @ref da_order enum specifying column-major or row-major layout.
 * @param[in] m the number of rows of matrix X. Constraint: @p m @f$\ge@f$ 1.
 * @param[in] n the number of rows of matrix Y. Constraint: @p n @f$\ge@f$ 1.
 * @param[in] k the number of columns of matrices X and Y. Constraint: @p k @f$\ge@f$ 1.
 * @param[in] X the matrix of size \p m @f$\times@f$ \p k, stored in column-major order by default.
 * @param[in] ldx the leading dimension of \p X. Constraint: \p ldx @f$\ge@f$ \p m if \p order = \p column_major, or \p ldx @f$\ge@f$ \p k if \p order = \p row_major.
 * @param[in] Y the matrix of size \p n @f$\times@f$ \p k, or null if computing the kernel of \p X with itself.
 * @param[in] ldy the leading dimension of \p Y. Constraint: \p ldy @f$\ge@f$ \p n if \p order = \p column_major, or \p ldy @f$\ge@f$ \p k if \p order = \p row_major.
 * @param[out] D the resulting kernel matrix of size \p m @f$\times@f$ \p n if \p Y is not null, or \p m @f$\times@f$ \p m otherwise.
 * @param[in] ldd the leading dimension of the matrix \p D. Constraint: \p ldd @f$\ge@f$ \p m, if \p Y is nullptr or \p order = \p column_major, and \p ldd @f$\ge@f$ \p n, otherwise.
 * @param[in] gamma the scale factor used in sigmoid kernel. Constraint: \p gamma @f$\ge@f$ 0.
 * @param[in] coef0 constant term in the sigmoid kernel.
 * @return @ref da_status
 * - @ref da_status_success - operation completed successfully.
 * - @ref da_status_invalid_leading_dimension - one of the constraints on \p ldx, \p ldy, or \p ldd was violated.
 * - @ref da_status_invalid_pointer - one of the input pointers is null.
 * - @ref da_status_invalid_input - one of the arguments had an invalid value.
 * - @ref da_status_invalid_array_dimension - one of the dimensions \p m, \p n, or \p k is invalid.
 * - @ref da_status_memory_error - unable to allocate memory.
 */
da_status da_sigmoid_kernel_d(da_order order, da_int m, da_int n, da_int k,
                              const double *X, da_int ldx, const double *Y, da_int ldy,
                              double *D, da_int ldd, double gamma, double coef0);

da_status da_sigmoid_kernel_s(da_order order, da_int m, da_int n, da_int k,
                              const float *X, da_int ldx, const float *Y, da_int ldy,
                              float *D, da_int ldd, float gamma, float coef0);
/** \} */

#endif
