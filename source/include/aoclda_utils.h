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
da_status da_switch_order_copy_d(da_order order_X, da_int n_rows, da_int n_cols,
                                 const double *X, da_int ldx, double *Y, da_int ldy);
da_status da_switch_order_copy_s(da_order order_X, da_int n_rows, da_int n_cols,
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

/** hidden doc {
 * \brief Get information about the selected CPU architecture.
 *
 * Returns or prints the architecture currently selected to be used by the library's API.
 *
 * Calling this function also refreshes the context for the dynamic dispatch querying the
 * value of the environmental variable \c AOCL_DA_ARCH.
 *
 * Note: This function is targeted to debugging the library.
 *
 * If \p arch, or \p ns are NULL or \p len point to a non-positive value, then print to standard output the architecture info.
 * If \p arch, and \p ns are provided and len is valid, fills buffer with architecture information strings.
 *
 * \param[inout] len length of the output buffer. If 0, returns required buffer size.
 * \param[out] arch buffer to store local architecture information string. Can be NULL if \p len is 0.
 * \param[out] ns buffer to store dispatched architecture information string. Can be NULL if \p len is 0.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success - operation completed successfully.
 * - \ref da_status_invalid_array_dimension - provided buffer length too small, returns in \p len the minimum size.
 */
da_status da_get_arch_info(da_int *len, char *arch, char *ns);

/** \{
 * \brief Splits input matrix into train and test matrices.
 * 
 * For a dataset of size \p m x \p n, forms two new datasets:
 * - a training matrix of size \p train_size by \p n
 * - a test matrix of size \p test_size by \p n
 * 
 *
 *
 * \param[in] order a \ref da_order enumerated type, specifying whether \p X is stored in row-major or column-major order.
 * \param[in] m the number of rows in the data matrix. Constraint: \p m @f$\ge 2@f$.
 * \param[in] n the number of columns in the data matrix. Constraint: \p n @f$\ge 1@f$.
 * \param[in] X the \p m @f$\times@f$ \p n data matrix.
 * \param[in] ldx the leading dimension of \p X. Constraint: \p ldx @f$\ge@f$ \p m if \p order = \p column_major, or \p ldx @f$\ge@f$ \p n if \p order = \p row_major.
 * \param[in] train_size the number of rows in the training matrix. Constraint: @f$1 \le@f$ \p train_size; \p train_size + \p test_size @f$\le@f$ \p m.
 * \param[in] test_size the number of rows in the test matrix. Constraint: @f$1 \le@f$ \p test_size; \p train_size + \p test_size @f$\le@f$ \p m.
 * \param[in] shuffle_array an array of indices specifying the order in which to select samples if shuffling is desired. The function uses the first \p train_size indices for the training split and the next \p test_size indices for the test split. A shuffled array can be generated using \ref da_get_shuffled_indices_s. If shuffling is not required, \p nullptr should be supplied. 
 * \param[out] X_train the array which will hold the train set.
 * \param[in] ldx_train the leading dimension of \p X_train. Constraint: \p ldx_train @f$\ge@f$ \p train_size if \p order = \p column_major, or \p ldx_train @f$\ge@f$ \p n if \p order = \p row_major.
 * \param[out] X_test the array which will hold the test set.
 * \param[in] ldx_test the leading dimension of \p X_test. Constraint: \p ldx_test @f$\ge@f$ \p test_size if \p order = \p column_major, or \p ldx_test @f$\ge@f$ \p n if \p order = \p row_major.
 * \return \ref da_status. the function returns:
 * - \ref da_status_success - The operation was successfully completed.
 * - \ref da_status_invalid_leading_dimension - The constraint on either \p ldx. \p ldx_train or \p ldx_test was violated.
 * - \ref da_status_invalid_pointer - One of the arrays \p X, \p X_train or \p X_test is \p null.
 * - \ref da_status_invalid_array_dimension - Either \p m @f$ < 2@f$ or \p n @f$< 1@f$.
 * - \ref da_status_invalid_input - Constraint on \p train_size or \p test_size was violated.
 */
da_status da_train_test_split_int(da_order order, da_int m, da_int n, const da_int *X,
                                  da_int ldx, da_int train_size, da_int test_size,
                                  const da_int *shuffle_array, da_int *X_train,
                                  da_int ldx_train, da_int *X_test, da_int ldx_test);
da_status da_train_test_split_s(da_order order, da_int m, da_int n, const float *X,
                                da_int ldx, da_int train_size, da_int test_size,
                                const da_int *shuffle_array, float *X_train,
                                da_int ldx_train, float *X_test, da_int ldx_test);
da_status da_train_test_split_d(da_order order, da_int m, da_int n, const double *X,
                                da_int ldx, da_int train_size, da_int test_size,
                                const da_int *shuffle_array, double *X_train,
                                da_int ldx_train, double *X_test, da_int ldx_test);
/** \} */

/** \{ */
/**
 * \brief Randomly shuffle an array of integers.
 * 
 * Returns an array of shuffled indices, generated by randomly permuting the integers from 0 to @f$ m-1 @f$.
 *  
 * \param[in] m the size of \p shuffle_array. Constraint: \p m @f$ > 1@f$.
 * \param[in] seed the seed to be used for the random engine. If \p seed = -1, it results in non-repeatable results. Constraint: \p > -1.
 * \param[in] train_size test size to be used for the stratified shuffling. If \p classes = \p nullptr, it will be ignored. Constraint: \p train_size must be greater than or equal to the number of samples in the smallest class, \p train_size + \p test_size @f$\le@f$ \p m.
 * \param[in] test_size test size to be used for the stratified shuffling. If \p classes = \p nullptr, it will be ignored. Constraint: \p test_size must be greater than or equal to the number of samples in the smallest class, \p train_size + \p test_size @f$\le@f$ \p m.
 * \param[in] fp_precision an integer specifying the scaling factor applied to floating-point class labels to determine their precision. When \p classes is of type float or double, each class label is multiplied by \p fp_precision and then floored to obtain an integer class label for stratified shuffling. Constraint: \p fp_precision @f$\ge 10@f$.
 * \param[in] classes an array containing the class of each sample. If supplied it will be used to do stratified shuffling. The stratified shuffle aims to preserve the frequency of the classes in the train and the test split, as they were in the original array. If stratified shuffling is not required, then a \p null pointer of type \p da_int, \p float or \p double should be supplied. Must be of size \p m. Constraint: The number of samples of each class must be @f$ \ge 2@f$; The number of unique classes should be at @f$ \ge 2@f$.
 * \param[out] shuffle_array the array which will store the shuffled indices. If \p classes != \p nullptr, then it would return \p m shuffled indices. Otherwise, it will return \p train_size + \p test_size shuffled indices from the range of \p 0 to \p m. Size must be @f$ \ge @f$ \p m.
 * \return \ref da_status. the function returns:
 * - \ref da_status_success - the operation was successfully completed.
 * - \ref da_status_invalid_array_dimension - \p m @f$ < 1@f$.
 * - \ref da_status_invalid_input - Either the constraint on \p seed was violated or if \p classes != \p nullptr the constraint on either \p train_size, \p test_size, \p m, \p classes or \p fp_precision was violated.
 * - \ref da_status_invalid_pointer - shuffle_array is \p null.
*/
da_status da_get_shuffled_indices_int(da_int m, da_int seed, da_int train_size,
                                      da_int test_size, da_int fp_precision,
                                      const da_int *classes, da_int *shuffle_array);
da_status da_get_shuffled_indices_s(da_int m, da_int seed, da_int train_size,
                                    da_int test_size, da_int fp_precision,
                                    const float *classes, da_int *shuffle_array);
da_status da_get_shuffled_indices_d(da_int m, da_int seed, da_int train_size,
                                    da_int test_size, da_int fp_precision,
                                    const double *classes, da_int *shuffle_array);

/**
 * @brief returns a char* array describing the da_int integer used by the library
 *
 * '32' if da_int = int32_t
 * '64' if da_int = int64_t
 * '?'  if da_int is set to an unexpected value
 *
 * @param len      The length of the buffer int_type. Will be set to the minimal value if too small.
 * @param int_type The character array containing the integer description on output.
 *
 * @return da_status_success on success,
 *         da_status_invalid_input if @p len or @p int_type are nullptr,
 *         da_status_invalid_array_dimension if @p len < 3,
 */
da_status da_get_int_info(size_t *len, char *int_type);

/**
 * @brief Sets option key-value pair in context settings.
 *
 * @param key   The debug option key as a null-terminated C string.
 * @param value The debug option value as a null-terminated C string.
 * @return da_status Returns da_status_success on success,
 *                  da_status_invalid_input if key or value is null,
 *                  or da_status_operation_failed on exception.
 */
da_status da_debug_set(const char *key, const char *value);

/**
 * @brief Gets option from key-value pair in context settings.
 *
 * This function looks up a hidden setting in the current context by its key.
 * If both @p key and @p value are null, it prints all registered context settings to stdout.
 *
 * @param key      The key of the setting to retrieve. If null and @p value is also null, prints all settings.
 * @param lvalue   The maximum number of characters to copy into @p value (including null terminator).
 *                 Must be at least 100.
 * @param value    The buffer to store the retrieved value. Must be non-null if @p key is non-null.
 *
 * @return da_status_success on success,
 *         da_status_invalid_input if @p lvalue < 100,
 *         da_status_option_not_found if the key is not found.
 */
da_status da_debug_get(const char *key, da_int lvalue, char *value);
/** \} */

#endif
