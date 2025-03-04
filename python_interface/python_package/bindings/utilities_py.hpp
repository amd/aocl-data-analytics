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

#ifndef UTILITIES_PY_HPP
#define UTILITIES_PY_HPP

#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include <iostream>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;
using namespace pybind11::literals;

/* Helper enum to record whether we are using Fortran or C ordering */
enum numpy_order { f_contiguous = 0, c_contiguous, undetermined };

/*
 * Determine the size and ordering of a numpy array
 */
template <typename T>
void get_size(da_order &order, py::array_t<T> &X, da_int &m, da_int &n, da_int &ldx) {

    if (X.ndim() > 2) {
        throw std::length_error(
            "Function does not accept arrays with more than 2 dimensions.");
    }
    // If we are dealing with 1D array the shape attribute is stored as (n_samples, )
    // so accessing X.shape()[1] will cause an error when 1D array is passed
    if (X.ndim() == 1) {
        n = X.shape()[0];
        m = 1;
        order = column_major;
        ldx = m;
    } else {
        m = X.shape()[0], n = X.shape()[1];

        if (m == 1) {
            // Special case for single row which is both C and Fortran contiguous
            ldx = m;
            order = column_major;
        } else {

            auto X_buffer = X.request();
            bool X_c_style = (X_buffer.strides[X_buffer.ndim - 1] == sizeof(T));
            if (X_c_style) {
                // X is C-style array
                ldx = X.strides()[0] / sizeof(T);
                order = row_major;
            } else {
                // X is F-style array
                ldx = X.strides()[1] / sizeof(T);
                order = column_major;
            }
        }
    }
}

class pyda_handle {
  protected:
    da_handle handle = nullptr;
    da_precision precision = da_double;
    numpy_order order = undetermined;

  public:
    void print_error_message() { da_handle_print_error_message(handle); };
    void exception_check(da_status status, std::string mesg = "") {
        if (status == da_status_success)
            return;
        // If we got to here, there's an error to deal with

        // Override the handle message if provided alternative message
        if (mesg != "") {
            PyErr_SetString(PyExc_RuntimeError, mesg.c_str());
            throw py::error_already_set();
        }
        char *message;
        da_severity severity;
        da_handle_get_error_message(handle, &message);
        da_handle_get_error_severity(handle, &severity);
        if (severity == DA_ERROR) {
            PyErr_SetString(PyExc_RuntimeError, std::string(message).c_str());
            throw py::error_already_set();
        } else
            PyErr_WarnEx(PyExc_RuntimeWarning, std::string(message).c_str(), 1);

        free(message);
    }
    /* Extract the storage scheme of a numpy array and check if it matches the order stored in the class.
       If this is the first call to such a routine, then set the order accordingly.
    */
    template <typename T>
    void get_numpy_array_properties(py::array_t<T> &X, da_int &n_rows, da_int &n_cols,
                                    da_int &ldx) {

        if (X.ndim() == 1) {
            n_rows = X.shape()[0];
            n_cols = 1;
        } else {
            n_rows = X.shape()[0];
            n_cols = X.shape()[1];
        }

        // Special case for single row or column which can be either C or Fortran contiguous
        if (n_rows == 1 || n_cols == 1) {
            switch (order) {
            case f_contiguous:
                ldx = n_rows;
                break;
            case c_contiguous:
                ldx = n_cols;
                break;
            default:
                // Order hasn't been set yet so warn user and carry on with column major
                std::string warn_message =
                    "Cannot determine storage scheme; defaulting to "
                    "column-major.";
                PyErr_WarnEx(PyExc_RuntimeWarning, std::string(warn_message).c_str(), 1);
                ldx = n_rows;
                order = f_contiguous;
                break;
            }
            return;
        }

        auto X_buffer = X.request();
        bool X_c_style = (X_buffer.strides[X_buffer.ndim - 1] == sizeof(T));
        numpy_order X_order;
        if (X_c_style) {
            // X is C-style array
            ldx = X.strides()[0] / sizeof(T);
            X_order = c_contiguous;
        } else {
            // X is F-style array
            ldx = X.strides()[1] / sizeof(T);
            X_order = f_contiguous;
        }

        // Check if we match expected order
        if (order == undetermined) {
            // Order hasn't yet been se, so set it
            order = X_order;
            return;
        } else if (order != X_order) {
            std::string err_message = "Inconsistent use of C and Fortran ordering.";
            PyErr_SetString(PyExc_RuntimeError, std::string(err_message).c_str());
            throw py::error_already_set();
        }

        return;
    }
};

/*
 * Helper function to copy a numpy array, taking care that the array may be a slice and therefore might not be contiguous
 */
template <typename T> py::array_t<T> copy_numpy_array(py::array_t<T> &X) {

    da_int m, n, ldx;
    da_order order;

    // Query X for its size and ordering
    get_size(order, X, m, n, ldx);

    // Create the output array as a numpy array with contiguous data
    size_t shape[2]{(size_t)m, (size_t)n};
    size_t strides[2];
    if (order == column_major) {
        strides[0] = sizeof(T);
        strides[1] = sizeof(T) * m;
    } else {
        strides[0] = sizeof(T) * n;
        strides[1] = sizeof(T);
    }

    py::array_t<T> copy_X(shape, strides);

    // Determine the size of each contiguous block to copy
    size_t block_size = (order == column_major) ? m : n;
    size_t n_blocks = (order == column_major) ? n : m;

    for (size_t i = 0; i < n_blocks; i++) {
        memcpy(copy_X.mutable_data() + i * block_size, X.mutable_data() + i * ldx,
               block_size * sizeof(T));
    }

    return copy_X;
}

void status_to_exception(da_status status);

#endif
