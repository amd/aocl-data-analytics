/*
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef BASIC_HANDLE_HPP
#define BASIC_HANDLE_HPP
#include "aoclda_error.h"
#include "aoclda_result.h"
#include "aoclda_types.h"
#include "da_error.hpp"
#include "da_utils.hpp"
#include "options.hpp"

/*
 * Base handle class (basic_handle) that contains members that
 * are common for all specialized handle types, pca, linear
 * models, etc.
 *
 * This handle is inherited by all specialized (internal) handles.
 */
template <typename T> class basic_handle {
  public:
    basic_handle(da_errors::da_error_t *err = nullptr);
    basic_handle(da_errors::da_error_t &err);
    virtual ~basic_handle();

    /*
     * Generic interface to extract all data stored
     * in handle via the da_get_result_X C API
     */
    virtual da_status get_result(da_result query, da_int *dim, T *result) = 0;
    virtual da_status get_result(da_result query, da_int *dim, da_int *result) = 0;

    /*
     * Function to inform that something related to the (sub)handle has
     * changed and to mark as update-required. E.g. options changed and potentially
     * the underlying model is different and a new call to fit is required.
     * Each (sub)handle is responsible to implement this function if required.
     */
    virtual void refresh();

    // user's data storage: row or column major order
    // all 2d arrays passed from/to user are in this order, internally
    // these arrays can and are converted. But when passing data back to user
    // this is the order used.
    da_order order{da_order::column_major};

    // Pointer to error trace
    da_errors::da_error_t *err{nullptr};

    // Options registry
    da_options::OptionRegistry opts;

    da_options::OptionRegistry &get_opts();

    // Argument checking for a 1D input array, including NaN check if option is set
    // @tparam T Type of the data array: float, double, da_int
    da_status check_1D_array(da_int n, const T *data, const std::string &n_name,
                             const std::string &data_name, da_int n_min = 1);
    da_status check_1D_array(da_int n, const da_int *data, const std::string &n_name,
                             const std::string &data_name, da_int n_min = 1);

    /**
     * @brief Stores a 2D array into the handle, performing necessary checks and transformations.
     *
     * This function stores/checks a 2D array into the handle, based on the specified storage order
     * and mode. It performs checks for invalid dimensions, null pointers, and NaN values (if enabled).
     * If the storage order is row-major and mode is 0, it transposes the data and stores it internally
     * in column-major format.
     *
     * @param n_rows The number of rows in the 2D array.
     * @param n_cols The number of columns in the 2D array.
     * @param data A pointer to the input data.
     * @param lddata The leading dimension of the input data.
     * @param temp_data A pointer to a temporary array where transposed data can be stored.  This memory
     *        is allocated within the function when row-major storage is requested and mode is 0, and must be
     *        freed by the caller.
     * @param data_internal A pointer to a const pointer where the address of the stored data will be written.
     * @param lddata_internal A reference to the leading dimension of the stored data.
     * @param n_rows_name The name of the variable representing the number of rows (for error messages).
     * @param n_cols_name The name of the variable representing the number of columns (for error messages).
     * @param data_name The name of the variable representing the data (for error messages).
     * @param lddata_name The name of the variable representing the leading dimension (for error messages).
     * @param mode An integer representing the mode of operation:
     *        mode = 0 will do the following:
     *                 1. If the user's data array is in column major format, point data_internal to the same data.
     *                 2. If the user's data array is in row major format, allocate memory for data_internal, copy
     *                    and transpose the data.
     *                 3. In each case, lddata_internal is updated appropriately and the `check data` option is read
     *                    and acted upon accordingly to check for NaNs.
     *                 4. The temp_data pointer is used to allocate memory to be deallocated later (it is not const,
     *                    but points to any allocated memory).
     *        mode = 1 same as mode = 0 but data is not checked for NaN nor is copied (use case: output array)
     *        mode = 2 updates internal data pointer and leading dimension argument, without argument or option checking
     *                 (for use when we already know data is usable and in column major format)
     *        mode = 3 only performs checks on the dimensions and values, does not do any copy or storage scheme changes
     * @param n_rows_min The minimum allowed value for the number of rows.
     * @param n_cols_min The minimum allowed value for the number of columns.
     *
     * @note If the storage order is row-major and mode is 0, the input data is transposed and stored internally in
     *       column-major format. In this case, memory is allocated for the transposed data, and the caller is responsible
     *       for freeing this memory.
     */
    da_status store_2D_array(da_int n_rows, da_int n_cols, const T *data, da_int lddata,
                             T **temp_data, const T **data_internal,
                             da_int &lddata_internal, const std::string &n_rows_name,
                             const std::string &n_cols_name, const std::string &data_name,
                             const std::string &lddata_name, da_int mode = 0,
                             da_int n_rows_min = 1, da_int n_cols_min = 1);

    // Copy a column major internal 2D results array into the user's buffer, converting to row-major ordering and checking for NaNs if necessary
    void copy_2D_results_array(da_int n_rows, da_int n_cols, T *data, da_int lddata,
                               T *results_arr);
};

#endif