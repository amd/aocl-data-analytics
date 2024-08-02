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

#ifndef KMEANS_PY_HPP
#define KMEANS_PY_HPP

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

class kmeans : public pyda_handle {

  public:
    kmeans(da_int n_clusters = 1, std::string initialization_method = "k-means++",
           da_int n_init = 10, da_int max_iter = 300, da_int seed = -1,
           std::string algorithm = "elkan", std::string prec = "double",
           bool check_data = false) {
        if (prec == "double")
            da_handle_init<double>(&handle, da_handle_kmeans);
        else if (prec == "single") {
            da_handle_init<float>(&handle, da_handle_kmeans);
            precision = da_single;
        }
        da_status status;
        status = da_options_set_int(handle, "n_clusters", n_clusters);
        exception_check(status);
        status = da_options_set_string(handle, "algorithm", algorithm.c_str());
        exception_check(status);
        status = da_options_set_string(handle, "initialization method",
                                       initialization_method.c_str());
        exception_check(status);
        status = da_options_set_int(handle, "max_iter", max_iter);
        exception_check(status);
        status = da_options_set_int(handle, "seed", seed);
        exception_check(status);
        status = da_options_set_int(handle, "n_init", n_init);
        exception_check(status);
        if (check_data == true) {
            std::string yes_str = "yes";
            status = da_options_set(handle, "check data", yes_str.data());
            exception_check(status);
        }
    }
    ~kmeans() { da_handle_destroy(&handle); }

    template <typename T>
    void fit(py::array_t<T> A, std::optional<py::array_t<T>> C, T tol = 1.0e-4) {
        // floating point optional parameters are defined here since we cannot define those in the constructor (no template param)
        da_status status;
        status = da_options_set(handle, "convergence tolerance", tol);
        exception_check(status);
        da_int n_samples, n_features, lda, ldc, tmp1, tmp2;

        get_numpy_array_properties(A, n_samples, n_features, lda);

        if (order == c_contiguous) {
            status = da_options_set(handle, "storage order", "row-major");
        } else {
            status = da_options_set(handle, "storage order", "column-major");
        }
        exception_check(status);
        status = da_kmeans_set_data(handle, n_samples, n_features, A.data(), lda);

        exception_check(status);
        if (C.has_value()) {

            get_numpy_array_properties(C.value(), tmp1, tmp2, ldc);

            status = da_options_set_string(handle, "initialization method", "supplied");
            status = da_kmeans_set_init_centres(handle, C->data(), ldc);
            exception_check(status);
        }
        status = da_kmeans_compute<T>(handle);
        exception_check(status);
    }

    template <typename T> py::array_t<T> transform(py::array_t<T> X) {
        da_status status;
        da_int m_samples, m_features, ldx;
        get_numpy_array_properties(X, m_samples, m_features, ldx);

        T result[5];
        da_int dim = 5;

        status = da_handle_get_result(handle, da_rinfo, &dim, result);
        exception_check(status);

        // Define the output vector
        da_int n_clusters = (da_int)result[2], ldx_transform;
        size_t shape[2]{(size_t)m_samples, (size_t)n_clusters};
        size_t strides[2];
        if (order == c_contiguous) {
            ldx_transform = n_clusters;
            strides[0] = sizeof(T) * n_clusters;
            strides[1] = sizeof(T);
        } else {
            ldx_transform = m_samples;
            strides[0] = sizeof(T);
            strides[1] = sizeof(T) * m_samples;
        }
        auto X_transform = py::array_t<T>(shape, strides);

        status = da_kmeans_transform(handle, m_samples, m_features, X.data(), ldx,
                                     X_transform.mutable_data(), ldx_transform);
        exception_check(status);
        return X_transform;
    }

    template <typename T> py::array_t<da_int> predict(py::array_t<T> Y) {
        da_status status;
        da_int k_samples, k_features, ldy;
        get_numpy_array_properties(Y, k_samples, k_features, ldy);

        T result[5];
        da_int dim = 5;

        status = da_handle_get_result(handle, da_rinfo, &dim, result);
        exception_check(status);

        // define the output vector
        size_t shape[1]{(size_t)k_samples};
        size_t strides[1]{sizeof(da_int)};
        auto Y_labels = py::array_t<da_int>(shape, strides);

        status = da_kmeans_predict(handle, k_samples, k_features, Y.data(), ldy,
                                   Y_labels.mutable_data());
        exception_check(status);
        return Y_labels;
    }

    template <typename T>
    void get_rinfo(da_int *n_samples, da_int *n_features, da_int *n_clusters,
                   da_int *n_iter, T *inertia) {
        da_status status;

        da_int dim = 5;

        T rinfo[5];
        status = da_handle_get_result(handle, da_rinfo, &dim, rinfo);
        *n_samples = (da_int)rinfo[0];
        *n_features = (da_int)rinfo[1];
        *n_clusters = (da_int)rinfo[2];
        *n_iter = (da_int)rinfo[3];
        *inertia = rinfo[4];

        exception_check(status);
    }

    auto get_cluster_centres() {
        da_status status;

        size_t stride_size;
        da_int n_samples, n_features, n_clusters, n_iter;
        da_int dim, dim1, dim2;

        if (precision == da_single) {
            stride_size = sizeof(float);
            float inertia;
            get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);
        } else {
            stride_size = sizeof(double);
            double inertia;
            get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);
        }

        dim1 = n_clusters;
        dim2 = n_features;
        dim = dim1 * dim2;

        std::vector<size_t> shape, strides;
        shape.push_back(dim1);
        if (dim2 > 1) {
            shape.push_back(dim2);
        }

        if (order == c_contiguous) {
            if (dim2 > 1) {
                strides.push_back(stride_size * dim2);
            }
            strides.push_back(stride_size);
        } else {
            strides.push_back(stride_size);
            if (dim2 > 1) {
                strides.push_back(stride_size * dim1);
            }
        }

        if (precision == da_single) {

            // define the output vector
            auto res = py::array_t<float>(shape, strides);
            status = da_handle_get_result(handle, da_kmeans_cluster_centres, &dim,
                                          res.mutable_data());
            exception_check(status);
            py::array ret = py::reinterpret_borrow<py::array>(res);
            return ret;
        } else {

            // define the output vector
            auto res = py::array_t<double>(shape, strides);
            status = da_handle_get_result(handle, da_kmeans_cluster_centres, &dim,
                                          res.mutable_data());
            exception_check(status);
            py::array ret = py::reinterpret_borrow<py::array>(res);
            return ret;
        }
    }

    auto get_labels() {

        da_status status;

        size_t stride_size = sizeof(da_int);
        da_int n_samples, n_features, n_clusters, n_iter;
        da_int dim, dim1, dim2;

        if (precision == da_single) {
            float inertia;
            get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);
        } else {
            double inertia;
            get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);
        }

        dim1 = n_samples;
        dim2 = 1;
        dim = dim1 * dim2;

        std::vector<size_t> shape, strides;
        shape.push_back(dim1);
        strides.push_back(stride_size);
        // define the output vector
        auto res = py::array_t<da_int>(shape, strides);
        status = da_handle_get_result(handle, da_kmeans_labels, &dim, res.mutable_data());
        exception_check(status);
        py::array ret = py::reinterpret_borrow<py::array>(res);
        return ret;
    }

    auto get_inertia() {

        size_t stride_size;
        da_int n_samples, n_features, n_clusters, n_iter;

        if (precision == da_single) {
            stride_size = sizeof(float);
            float inertia;
            get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);
            std::vector<size_t> shape, strides;
            shape.push_back(1);
            strides.push_back(stride_size);
            // define the output vector
            auto res = py::array_t<float>(shape, strides);
            *(res.mutable_data(0)) = inertia;
            py::array ret = py::reinterpret_borrow<py::array>(res);
            return ret;
        } else {
            stride_size = sizeof(double);
            double inertia;
            get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);
            std::vector<size_t> shape, strides;
            shape.push_back(1);
            strides.push_back(stride_size);
            // define the output vector
            auto res = py::array_t<double>(shape, strides);
            *(res.mutable_data(0)) = inertia;
            py::array ret = py::reinterpret_borrow<py::array>(res);
            return ret;
        }
    }

    auto get_n_iter() {

        da_int n_samples, n_features, n_clusters, n_iter;

        if (precision == da_single) {
            float inertia;
            get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);
        } else {
            double inertia;
            get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);
        }

        return n_iter;
    }

    auto get_n_samples() {

        da_int n_samples, n_features, n_clusters, n_iter;

        if (precision == da_single) {
            float inertia;
            get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);
        } else {
            double inertia;
            get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);
        }

        return n_samples;
    }

    auto get_n_features() {

        da_int n_samples, n_features, n_clusters, n_iter;

        if (precision == da_single) {
            float inertia;
            get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);
        } else {
            double inertia;
            get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);
        }

        return n_features;
    }

    auto get_n_clusters() {

        da_int n_samples, n_features, n_clusters, n_iter;

        if (precision == da_single) {
            float inertia;
            get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);
        } else {
            double inertia;
            get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);
        }

        return n_clusters;
    }
};

#endif