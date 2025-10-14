/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "basic_handle.hpp"
#include "aoclda_error.h"
#include "aoclda_result.h"
#include "aoclda_types.h"
#include "basic_handle_options.hpp"
#include "da_error.hpp"
#include "options.hpp"

/*
 * Base handle class (basic_handle) that contains members that
 * are common for all specialized handle types, pca, linear
 * models, etc.
 *
 * This handle is inherited by all specialized (internal) handles.
 */

template <typename T> basic_handle<T>::basic_handle(da_errors::da_error_t *err) {
    this->err = err;
}

template <typename T> basic_handle<T>::basic_handle(da_errors::da_error_t &err) {
    // Assumes that err is valid
    this->err = &err;
    // Initialize the options registry with common options to all handles
    register_common_options<T>(this->opts, *this->err);
}

template <typename T> basic_handle<T>::~basic_handle() {}

template <typename T> da_options::OptionRegistry &basic_handle<T>::get_opts() {
    return this->opts;
}

template <typename T> void basic_handle<T>::refresh() {}

template <typename T>
da_status basic_handle<T>::check_1D_array(da_int n, const T *data,
                                          const std::string &n_name,
                                          const std::string &data_name, da_int n_min) {

    da_int check_data = 0;
    std::string check_data_str;
    opts.get("check data", check_data_str, check_data);
    return ARCH::da_utils::check_1D_array<T>(check_data != 0, this->err, n, data, n_name,
                                             data_name, n_min);
}

template <typename T>
da_status basic_handle<T>::check_1D_array(da_int n, const da_int *data,
                                          const std::string &n_name,
                                          const std::string &data_name, da_int n_min) {

    da_int check_data = 0;
    std::string check_data_str;
    opts.get("check data", check_data_str, check_data);
    return ARCH::da_utils::check_1D_array<da_int>(check_data != 0, this->err, n, data,
                                                  n_name, data_name, n_min);
}

template <typename T>
da_status basic_handle<T>::store_2D_array(
    da_int n_rows, da_int n_cols, const T *data, da_int lddata, T **temp_data,
    const T **data_internal, da_int &lddata_internal, const std::string &n_rows_name,
    const std::string &n_cols_name, const std::string &data_name,
    const std::string &lddata_name, da_int mode, da_int n_rows_min, da_int n_cols_min) {

    if (mode == 2) {
        // Quick exit
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

    // Read in data storage option (if handle has a valid options registry)
    std::string opt_order;
    da_int iorder;
    da_status status = this->opts.get("storage order", opt_order, iorder);
    bool aux = status != da_status_success;
    // aux = true => handle is an auxiliary handle with no options not error buffer

    // don't change the default if aux handle
    if (!aux)
        this->order = da_order(iorder);

    // Check for NaNs if the `check data` option has been set
    da_int check_data{!aux &&
                      (mode == 0 || mode == 3)}; // skip for mode 1 or if aux handle
    if (check_data) {
        std::string check_data_str;
        this->opts.get("check data", check_data_str, check_data);
        if (check_data) {
            status = ARCH::da_utils::check_data((da_order)this->order, n_rows, n_cols,
                                                data, lddata);
            if (status == da_status_invalid_input)
                return da_error(this->err, da_status_invalid_input,
                                "The array " + data_name + " contains at least one NaN.");
        }
    }

    std::string wrong_order = "";

    switch (this->order) {
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
        break;
    }
    default:
        return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpected storage scheme was requested.");
        break;
    }

    if (mode == 3) {
        // Mode 3: all done, exit
        return da_status_success;
    }

    // From here on only mode 0 and 1 remain
    if (this->order == column_major) {
        *data_internal = data;
        lddata_internal = lddata;
    } else { // order == row_major
        // Allocate memory for transposed data
        try {
            *temp_data = new T[n_rows * n_cols];
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }

        // Transpose the data
        if (mode == 0)
            ARCH::da_utils::copy_transpose_2D_array_row_to_column_major(
                n_rows, n_cols, data, lddata, *temp_data, n_rows);

        // Cast the non-const pointer and assign it to the const pointer
        *const_cast<T **>(data_internal) = *temp_data;
        lddata_internal = n_rows;
    }

    return da_status_success;
}

template <typename T>
void basic_handle<T>::copy_2D_results_array(da_int n_rows, da_int n_cols, T *data,
                                            da_int lddata, T *results_arr) {
    // Assumes input data is in column-major order and output is specified by this->order
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

template class basic_handle<double>;
template class basic_handle<float>;