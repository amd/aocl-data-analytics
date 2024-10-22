/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "basic_handle_options.hpp"
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
    basic_handle(){};
    basic_handle(da_errors::da_error_t &err) {
        // Assumes that err is valid
        this->err = &err;
        // Initialize the options registry with common options to all handles
        register_common_options<T>(this->opts, *this->err);
    };
    virtual ~basic_handle(){};

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
    virtual void refresh(){};

    // Is the user's data stored in row or column major order
    da_int order = column_major;

    // Pointer to error trace
    da_errors::da_error_t *err = nullptr;

    // Options registry
    da_options::OptionRegistry opts;

    da_options::OptionRegistry &get_opts() { return this->opts; };

    // Argument checking for a 1D input array, including NaN check if option is set
    da_status check_1D_array(da_int n, const T *data, const std::string &n_name,
                             const std::string &data_name, da_int n_min = 1);
    da_status check_1D_array(da_int n, const da_int *data, const std::string &n_name,
                             const std::string &data_name, da_int n_min = 1);

    // Store a pointer to a 2D array, converting to column major ordering if necessary, and optionally checking arguments
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

/*
Calling the function will do the following:
1. Point data_internal to the same data.
2. Argument checking on the data pointer and the size
3. Read the `check data` option and accordingly to check for NaNs.
*/
template <typename T>
da_status basic_handle<T>::check_1D_array(da_int n, const T *data,
                                          const std::string &n_name,
                                          const std::string &data_name, da_int n_min) {

    if (data == nullptr)
        return da_error(this->err, da_status_invalid_pointer,
                        "The array " + data_name + " is null.");

    // Check for illegal rows/columns arguments
    if (n < n_min)
        return da_error(this->err, da_status_invalid_array_dimension,
                        "The function was called with " + n_name + " = " +
                            std::to_string(n) + ". Constraint: " + n_name +
                            " >= " + std::to_string(n_min) + ".");

    // Check for NaNs if the `check data` option has been set
    da_int check_data = 0;
    std::string check_data_str;
    this->opts.get("check data", check_data_str, check_data);
    if (check_data) {
        da_status status = da_utils::check_data(column_major, n, 1, data, n);
        if (status == da_status_invalid_input)
            return da_error(this->err, da_status_invalid_input,
                            "The array " + data_name + " contains at least one NaN.");
    }

    return da_status_success;
}

template <typename T>
da_status basic_handle<T>::check_1D_array(da_int n, const da_int *data,
                                          const std::string &n_name,
                                          const std::string &data_name, da_int n_min) {

    if (data == nullptr)
        return da_error(this->err, da_status_invalid_pointer,
                        "The array " + data_name + " is null.");

    // Check for illegal rows/columns arguments
    if (n < n_min)
        return da_error(this->err, da_status_invalid_array_dimension,
                        "The function was called with " + n_name + " = " +
                            std::to_string(n) + ". Constraint: " + n_name +
                            " >= " + std::to_string(n_min) + ".");
    return da_status_success;
}

/*
Calling the function with mode = 0 will do the following:
1. If the user's data array is in column major format, point data_internal to the same data.
2. If the user's data array is in row major format, allocate memory for data_internal, copy and transpose the data.
3. In each case, lddata_internal is updated appropriately and the `check data` option is read and acted upon accordingly to check for NaNs.
4. The temp_data pointer is used to enable memory to be deallocated later (it is not const, but points to any allocated memory).
Calling the function with mode = 1 will do the same except that no copying or data checking occurs (use case: output array)
Calling the function with mode = 2 just copies the pointer and leading dimension argument, without argument
or option checking (for use when we already know data is usable and in column major format)
*/
template <typename T>
da_status basic_handle<T>::store_2D_array(
    da_int n_rows, da_int n_cols, const T *data, da_int lddata, T **temp_data,
    const T **data_internal, da_int &lddata_internal, const std::string &n_rows_name,
    const std::string &n_cols_name, const std::string &data_name,
    const std::string &lddata_name, da_int mode, da_int n_rows_min, da_int n_cols_min) {

    // Quick exit if mode == 2
    if (mode == 2) {
        *data_internal = data;
        lddata_internal = lddata;
        return da_status_success;
    }

    // Check for illegal rows/columns arguments
    if (n_rows < n_rows_min)
        return da_error(this->err, da_status_invalid_array_dimension,
                        "The function was called with " + n_rows_name + " = " +
                            std::to_string(n_rows) + ". Constraint: " + n_rows_name +
                            " >= " + std::to_string(n_rows_min) + ".");
    if (n_cols < n_cols_min)
        return da_error(this->err, da_status_invalid_array_dimension,
                        "The function was called with " + n_cols_name + " = " +
                            std::to_string(n_rows) + ". Constraint: " + n_cols_name +
                            " >= " + std::to_string(n_cols_min) + ".");

    if (data == nullptr)
        return da_error(this->err, da_status_invalid_pointer,
                        "The array " + data_name + " is null.");

    // Read in data storage option
    std::string opt_order;
    this->opts.get("storage order", opt_order, this->order);

    // Check for NaNs if the `check data` option has been set
    da_int check_data = 0;
    if (mode == 0) {
        std::string check_data_str;
        this->opts.get("check data", check_data_str, check_data);
        if (check_data) {
            da_status status =
                da_utils::check_data((da_order)this->order, n_rows, n_cols, data, lddata);
            if (status == da_status_invalid_input)
                return da_error(this->err, da_status_invalid_input,
                                "The array " + data_name + " contains at least one NaN.");
        }
    }

    std::string wrong_order = "";

    switch (order) {
    case column_major:

        if (lddata < n_rows) {
            if (lddata >= n_cols) {
                wrong_order = "The handle is set to expect column major data. Did "
                              "you mean to set "
                              "it to row major?";
            }
            return da_error(this->err, da_status_invalid_leading_dimension,
                            "The function was called with " + n_rows_name + " = " +
                                std::to_string(n_rows) + " and " + lddata_name + " = " +
                                std::to_string(lddata) + ". Constraint: " + lddata_name +
                                " >= " + n_rows_name + "." + wrong_order);
        }
        *data_internal = data;
        lddata_internal = lddata;
        break;
    case row_major: {
        if (lddata < n_cols) {
            if (lddata >= n_rows) {
                wrong_order =
                    "The handle is set to expect row major data. Did you mean to set "
                    "it to column major?";
            }
            return da_error(this->err, da_status_invalid_leading_dimension,
                            "The function was called with " + n_cols_name + "  = " +
                                std::to_string(n_cols) + " and " + lddata_name + " = " +
                                std::to_string(lddata) + ". Constraint: " + lddata_name +
                                " >= " + n_cols_name + "." + wrong_order);
        }

        // Allocate memory for transposed data
        try {
            *temp_data = new T[n_rows * n_cols];
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }

        // Transpose the data
        if (mode == 0)
            da_utils::copy_transpose_2D_array_row_to_column_major(
                n_rows, n_cols, data, lddata, *temp_data, n_rows);

        // Cast the non-const pointer and assign it to the const pointer
        *const_cast<T **>(data_internal) = *temp_data;
        lddata_internal = n_rows;
        break;
    }
    default:
        break;
    }

    return da_status_success;
}

template <typename T>
void basic_handle<T>::copy_2D_results_array(da_int n_rows, da_int n_cols, T *data,
                                            da_int lddata, T *results_arr) {
    if (order == column_major) {
        for (da_int j = 0; j < n_cols; ++j) {
            for (da_int i = 0; i < n_rows; ++i) {
                results_arr[i + j * n_rows] = data[i + lddata * j];
            }
        }
    } else {
        for (da_int i = 0; i < n_rows; ++i) {
            for (da_int j = 0; j < n_cols; ++j) {
                results_arr[i * n_cols + j] = data[i + lddata * j];
            }
        }
    }
}
#endif