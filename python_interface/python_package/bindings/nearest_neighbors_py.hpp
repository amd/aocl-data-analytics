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

#ifndef kNN_PY_HPP
#define kNN_PY_HPP

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

class knn_classifier : public pyda_handle {
    da_precision precision = da_double;
    da_int internal_neigh;
    py::array_t<float, py::array::f_style> Xf_internal;
    py::array_t<double, py::array::f_style> Xd_internal;
    py::array_t<da_int> y_internal;

  public:
    knn_classifier(da_int n_neighbors = 5, std::string weights = "uniform",
                   std::string algorithm = "brute", std::string metric = "euclidean",
                   std::string prec = "double") {
        da_status status;
        if (prec == "double") {
            status = da_handle_init<double>(&handle, da_handle_knn);
        } else if (prec == "single") {
            status = da_handle_init<float>(&handle, da_handle_knn);
            precision = da_single;
        } /*else {
            status = da_status_handle_not_initialized;
        }*/
        exception_check(status);
        status = da_options_set(handle, "number of neighbors", n_neighbors);
        exception_check(status);
        internal_neigh = n_neighbors;
        status = da_options_set(handle, "weights", weights.c_str());
        exception_check(status);
        status = da_options_set(handle, "algorithm", algorithm.c_str());
        exception_check(status);
        status = da_options_set(handle, "metric", metric.c_str());
        exception_check(status);
    }
    ~knn_classifier() { da_handle_destroy(&handle); }

    template <typename T>
    void fit(py::array_t<T, py::array::f_style> X, py::array_t<da_int> y) {
        da_status status;

        da_int n_samples = X.shape()[0], n_features = X.shape()[1];
        status = da_knn_set_training_data(handle, n_samples, n_features, X.data(),
                                          n_samples, y.data());
        exception_check(status); // throw an exception if status is not success

        // Store the pointers since they can be temporary so that memory is not
        // free'd and reused later.
        if constexpr (std::is_same_v<T, double>) {
            Xd_internal = std::move(X);
        } else if (std::is_same_v<T, float>) {
            Xf_internal = std::move(X);
        }
        y_internal = std::move(y);
    }

    template <typename T>
    py::array_t<da_int> kneighbors_indices(py::array_t<T, py::array::f_style> X,
                                           da_int n_neighbors = (da_int)0) {
        da_status status;
        da_int n_queries = X.shape()[0], n_features = X.shape()[1], ldx = X.shape()[0];
        da_int req_neigh = internal_neigh;
        if (n_neighbors != 0)
            req_neigh = n_neighbors;
        size_t shape[2]{(size_t)n_queries, (size_t)req_neigh};
        size_t strides[2]{sizeof(da_int), sizeof(da_int) * n_queries};
        auto k_ind = py::array_t<da_int>(shape, strides);
        status = da_knn_kneighbors(handle, n_queries, n_features, X.data(), ldx,
                                   k_ind.mutable_data(), nullptr, req_neigh, 0);
        exception_check(status);
        return k_ind;
    }

    template <typename T>
    py::tuple kneighbors(py::array_t<T, py::array::f_style> X,
                         da_int n_neighbors = (da_int)0) {
        da_status status;
        da_int n_queries = X.shape()[0], n_features = X.shape()[1], ldx = X.shape()[0];
        da_int req_neigh = internal_neigh;
        if (n_neighbors != 0)
            req_neigh = n_neighbors;
        size_t shape[2]{(size_t)n_queries, (size_t)req_neigh};
        size_t strides_i[2]{sizeof(da_int), sizeof(da_int) * n_queries};
        auto k_ind = py::array_t<da_int>(shape, strides_i);
        size_t strides_f[2]{sizeof(T), sizeof(T) * n_queries};
        auto k_dist = py::array_t<T>(shape, strides_f);
        status =
            da_knn_kneighbors(handle, n_queries, n_features, X.data(), ldx,
                              k_ind.mutable_data(), k_dist.mutable_data(), req_neigh, 1);
        exception_check(status);
        py::tuple k_info = py::make_tuple(k_dist, k_ind);
        return k_info;
    }

    template <typename T>
    py::array_t<T> predict_proba(py::array_t<T, py::array::f_style> X) {
        da_status status;
        da_int n_queries = X.shape()[0], n_features = X.shape()[1], ldx = X.shape()[0];
        da_int num_classes = -3;
        status = da_knn_classes<T>(handle, &num_classes, nullptr);
        exception_check(status);

        size_t shape[2]{(size_t)n_queries, (size_t)num_classes};
        size_t strides[2]{sizeof(T), sizeof(T) * n_queries};
        auto proba = py::array_t<T>(shape, strides);
        status = da_knn_predict_proba(handle, n_queries, n_features, X.data(), ldx,
                                      proba.mutable_data());
        exception_check(status);
        return proba;
    }

    template <typename T>
    py::array_t<da_int> predict(py::array_t<T, py::array::f_style> X) {
        da_status status;
        da_int n_queries = X.shape()[0], n_features = X.shape()[1], ldx = X.shape()[0];

        size_t shape[1]{(size_t)n_queries};
        auto y_test = py::array_t<da_int>(shape);
        status = da_knn_predict(handle, n_queries, n_features, X.data(), ldx,
                                y_test.mutable_data());
        exception_check(status);
        return y_test;
    }
};

#endif