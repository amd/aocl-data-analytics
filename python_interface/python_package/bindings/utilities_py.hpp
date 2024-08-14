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

/* Helper function to avoid redundancy
 * Determine the size of output array based on the axis
 */
template <typename T>
void get_size(std::string axis, da_axis &axis_enum, py::array_t<T, py::array::f_style> &X,
              size_t &size, da_int &m, da_int &n) {
    if (axis == "col") {
        axis_enum = da_axis_col;
    } else if (axis == "row") {
        axis_enum = da_axis_row;
    } else if (axis == "all") {
        axis_enum = da_axis_all;
    } else {
        throw std::invalid_argument(
            "Given axis does not exist. Available choices are: 'col', 'row', 'all'.");
    }
    if (X.ndim() > 2) {
        throw std::length_error(
            "Function does not accept arrays with more than 2 dimensions.");
    }
    // If we are dealing with 1D array the shape attribute is stored as (n_samples, )
    // so accessing X.shape()[1] is causing errors when 1D array is passed
    if (X.ndim() == 1) {
        n = X.shape()[0];
        m = 1;
        // If user provided 1D array then calculating for example mean over all elements
        // would mean calculation over that one row of data.
        if (axis_enum == da_axis_all) {
            axis_enum = da_axis_row;
        }
    } else {
        m = X.shape()[0], n = X.shape()[1];
    }
    switch (axis_enum) {
    case (da_axis_all):
        size = 1;
        break;

    case (da_axis_col):
        size = n;
        break;

    case (da_axis_row):
        size = m;
        break;
    }
}

class pyda_handle {
  protected:
    da_handle handle = nullptr;

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
};

void status_to_exception(da_status status);

#endif
