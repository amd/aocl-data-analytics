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

#ifndef METRICS_PY_HPP
#define METRICS_PY_HPP

#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include "utilities_py.hpp"
#include <iostream>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

template <typename T>
py::array_t<T>
py_da_pairwise_distances(py::array_t<T, py::array::f_style> X,
                         std::optional<py::array_t<T, py::array::f_style>> Y,
                         std::string metric = "euclidean",
                         std::string force_all_finite = "allow_infinite") {
    da_status status;
    da_int m = X.shape()[0], k = X.shape()[1];
    // Initialize numbers of columns for distance matrix D.
    // If Y is provided, D is m-by-n, otherwise it's m-by-m.
    da_int ncols = m;
    T *dummy = nullptr;
    // Check if the user provided matrix Y and set n accordingly
    da_int n = 0;
    if (Y.has_value()) {
        n = Y->shape()[0];
        // Y is provided, set number of columns correctly.
        ncols = n;
        if (X.shape()[1] != Y->shape()[1])
            throw std::invalid_argument(
                "Incompatible dimension for X and Y matrices: X.shape[1]=" +
                std::to_string(X.shape()[1]) +
                " while Y.shape[1]=" + std::to_string(Y->shape()[1]) + ".");
    }

    // Translate strings to enums
    da_metric metric_enum;
    if (metric == "euclidean") {
        metric_enum = da_euclidean;
    } else if (metric == "sqeuclidean") {
        metric_enum = da_sqeuclidean;
    } else {
        throw std::invalid_argument(
            "Given metric does not exist. Available choices are: 'euclidean', 'sqeuclidean'.");
    }
    da_data_types force_all_finite_enum;
    if (force_all_finite == "allow_infinite") {
        force_all_finite_enum = da_allow_infinite;
    } else {
        throw std::invalid_argument("Given force_all_finite option does not exist. "
                                    "Available choice is: 'allow_infinite'.");
    }
    // Create the output distance matrix as a numpy array
    size_t shape[2]{(size_t)m, (size_t)ncols};
    auto D = py::array_t<T, py::array::f_style>(shape);

    if (Y.has_value()) {
        status = da_pairwise_distances(m, n, k, X.data(), m, Y->mutable_data(), n,
                                       D.mutable_data(), m, metric_enum,
                                       force_all_finite_enum);
    } else {
        status = da_pairwise_distances(m, n, k, X.data(), m, dummy, n, D.mutable_data(),
                                       m, metric_enum, force_all_finite_enum);
    }
    status_to_exception(status);

    return D;
}

#endif
