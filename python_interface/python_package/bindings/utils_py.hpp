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

/* Utilities
 *
 * All adjacent utilities functions that are not directly related to
 * a specific algorithm go here.
 */

#ifndef UTILS_PY_HPP
#define UTILS_PY_HPP

#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include "internal_utilities_py.hpp"
#include <iostream>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <tuple>

namespace py = pybind11;

/*
 * da_debug_get wrapper
 */
void py_debug_print_context_registry(void) noexcept { da_debug_get(nullptr, 0, nullptr); }

std::string py_da_debug_get(std::string key) {
    char value[100];
    da_status status = da_debug_get(key.c_str(), 100, value);
    if (status != da_status_success) {
        throw std::runtime_error("Failed to get value for key: ``" + key + "``.");
    }
    return std::string(value);
}

std::string py_da_get_int_info() {
    char value[10];
    size_t len = 10;
    da_status status = da_get_int_info(&len, value);
    if (status != da_status_success) {
        throw std::runtime_error("Failed to get integer info");
    }
    return std::string(value);
}

void py_da_debug_set(std::string key, std::string value) {
    da_status status = da_debug_set(key.c_str(), value.c_str());
    if (status != da_status_success) {
        throw std::runtime_error("Failed to set value for key: ``" + key +
                                 "`` and value: ``" + value + "''.");
    }
}

/*
 * Splits the data into train and test parts
*/
template <typename T>
std::tuple<py::array_t<T>, py::array_t<T>>
py_train_test_split(py::array_t<T> &X, da_int test_size, da_int train_size,
                    std::optional<py::array_t<da_int>> &shuffled_indices) {

    da_int m, n, ldx, ldx_train, ldx_test;
    da_order order;

    // Special case when 1D array is inputted
    // It would be treated as a single column rather than single row
    if (X.ndim() == 1) {
        n = 1;
        m = X.shape()[0];
        order = row_major;
        ldx = n;
    } else {
        get_size(order, X, m, n, ldx);
    }

    if (order == column_major) {
        ldx_train = train_size;
        ldx_test = test_size;
    } else {
        ldx_train = n;
        ldx_test = n;
    }

    size_t shape_X_train[2]{(size_t)train_size, (size_t)n};
    size_t shape_X_test[2]{(size_t)test_size, (size_t)n};
    size_t strides_X_train[2];
    size_t strides_X_test[2];
    if (order == column_major) {
        strides_X_train[0] = sizeof(T);
        strides_X_train[1] = sizeof(T) * train_size;
        strides_X_test[0] = sizeof(T);
        strides_X_test[1] = sizeof(T) * test_size;
    } else {
        strides_X_train[0] = sizeof(T) * n;
        strides_X_train[1] = sizeof(T);
        strides_X_test[0] = sizeof(T) * n;
        strides_X_test[1] = sizeof(T);
    }

    auto X_train = py::array_t<T>(shape_X_train, strides_X_train);
    auto X_test = py::array_t<T>(shape_X_test, strides_X_test);

    da_status status;
    if (shuffled_indices.has_value()) {
        status = da_train_test_split(order, m, n, X.data(), ldx, train_size, test_size,
                                     shuffled_indices->data(), X_train.mutable_data(),
                                     ldx_train, X_test.mutable_data(), ldx_test);

    } else {
        status = da_train_test_split(order, m, n, X.data(), ldx, train_size, test_size,
                                     nullptr, X_train.mutable_data(), ldx_train,
                                     X_test.mutable_data(), ldx_test);
    }

    status_to_exception(status);

    return std::tuple{X_train, X_test};
}

/*
 * Creates an array with shuffled indices from 0 to m-1
*/
template <typename T>
py::array_t<da_int> py_get_shuffled_indices(da_int size, da_int seed, da_int train_size,
                                            da_int test_size, da_int fp_precision,
                                            std::optional<py::array_t<T>> &classes) {

    size_t shape[2]{(size_t)size, (size_t)1};
    size_t strides[2]{sizeof(da_int), sizeof(da_int)};

    auto shuffled_indices = py::array_t<da_int>(shape, strides);

    da_status status;

    if (classes.has_value()) {
        status =
            da_get_shuffled_indices(size, seed, train_size, test_size, fp_precision,
                                    classes->data(), shuffled_indices.mutable_data());
    } else {
        status = da_get_shuffled_indices(size, seed, train_size, test_size, fp_precision,
                                         (T *)nullptr, shuffled_indices.mutable_data());
    }

    status_to_exception(status);

    return shuffled_indices;
}

#endif