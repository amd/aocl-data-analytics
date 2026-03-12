/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef APPROX_NN_PY_HPP
#define APPROX_NN_PY_HPP

#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include "internal_utilities_py.hpp"
#include <iostream>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

class approximate_neighbors : public pyda_handle {
    da_int internal_neigh;

  public:
    approximate_neighbors(da_int n_neighbors = 5, std::string algorithm = "ivfflat",
                          std::string metric = "sqeuclidean", da_int n_list = 1,
                          da_int n_probe = 1, da_int kmeans_iter = 10, da_int seed = 0,
                          std::string prec = "double", bool check_data = false) {
        da_status status;
        if (prec == "double") {
            status = da_handle_init<double>(&handle, da_handle_approx_nn);
        } else {
            status = da_handle_init<float>(&handle, da_handle_approx_nn);
            precision = da_single;
        }
        exception_check(status);
        status = da_options_set(handle, "number of neighbors", n_neighbors);
        exception_check(status);
        internal_neigh = n_neighbors;
        status = da_options_set(handle, "algorithm", algorithm.c_str());
        exception_check(status);
        status = da_options_set(handle, "metric", metric.c_str());
        exception_check(status);
        status = da_options_set(handle, "n_list", n_list);
        exception_check(status);
        status = da_options_set(handle, "n_probe", n_probe);
        exception_check(status);
        status = da_options_set(handle, "k-means_iter", kmeans_iter);
        exception_check(status);
        status = da_options_set(handle, "seed", seed);
        exception_check(status);

        if (check_data == true) {
            std::string yes_str = "yes";
            status = da_options_set(handle, "check data", yes_str.data());
            exception_check(status);
        }
    }
    ~approximate_neighbors() { da_handle_destroy(&handle); }

    void set_n_probe_opt(da_int n_probe) {
        da_status status;
        status = da_options_set(handle, "n_probe", n_probe);
        exception_check(status);
    }

    template <typename T> void train(py::array_t<T> X, T train_fraction = 1.0) {
        // define floating-point optional parameters here since they can't de defined in constructor
        da_status status;
        status = da_options_set(handle, "train fraction", T(train_fraction));
        exception_check(status);
        da_int n_samples, n_features, ldx;

        get_numpy_array_properties(X, n_samples, n_features, ldx);

        if (order == c_contiguous) {
            status = da_options_set(handle, "storage order", "row-major");
        } else {
            status = da_options_set(handle, "storage order", "column-major");
        }
        exception_check(status);

        status =
            da_approx_nn_set_training_data(handle, n_samples, n_features, X.data(), ldx);
        exception_check(status);

        status = da_approx_nn_train<T>(handle);
        exception_check(status);
    }

    template <typename T> void train_and_add(py::array_t<T> X, T train_fraction = 1.0) {
        // define floating-point optional parameters here since they can't de defined in constructor
        da_status status;
        status = da_options_set(handle, "train fraction", T(train_fraction));
        exception_check(status);
        da_int n_samples, n_features, ldx;

        get_numpy_array_properties(X, n_samples, n_features, ldx);

        if (order == c_contiguous) {
            status = da_options_set(handle, "storage order", "row-major");
        } else {
            status = da_options_set(handle, "storage order", "column-major");
        }
        exception_check(status);

        status =
            da_approx_nn_set_training_data(handle, n_samples, n_features, X.data(), ldx);
        exception_check(status);

        status = da_approx_nn_train_and_add<T>(handle);
        exception_check(status);
    }

    template <typename T> void add(py::array_t<T> X) {
        da_status status;

        da_int n_samples, n_features, ldx;

        get_numpy_array_properties(X, n_samples, n_features, ldx);

        status = da_approx_nn_add(handle, n_samples, n_features, X.data(), ldx);
        exception_check(status);
    }

    template <typename T>
    py::array_t<da_int> kneighbors_indices(py::array_t<T> X,
                                           da_int n_neighbors = (da_int)0) {
        da_status status;

        da_int n_queries, n_features, ldx;

        get_numpy_array_properties(X, n_queries, n_features, ldx);

        da_int req_neigh = internal_neigh;
        if (n_neighbors != 0)
            req_neigh = n_neighbors;

        size_t shape[2]{(size_t)n_queries, (size_t)req_neigh};

        size_t strides[2];
        if (order == c_contiguous) {
            strides[0] = sizeof(da_int) * req_neigh;
            strides[1] = sizeof(da_int);
        } else {
            strides[0] = sizeof(da_int);
            strides[1] = sizeof(da_int) * n_queries;
        }

        auto k_ind = py::array_t<da_int>(shape, strides);
        status = da_approx_nn_kneighbors(handle, n_queries, n_features, X.data(), ldx,
                                         k_ind.mutable_data(), nullptr, req_neigh, 0);
        exception_check(status);
        return k_ind;
    }

    template <typename T>
    py::tuple kneighbors(py::array_t<T> X, da_int n_neighbors = (da_int)0) {
        da_status status;
        da_int n_queries, n_features, ldx;

        get_numpy_array_properties(X, n_queries, n_features, ldx);

        da_int req_neigh = internal_neigh;
        if (n_neighbors != 0)
            req_neigh = n_neighbors;

        size_t shape[2]{(size_t)n_queries, (size_t)req_neigh};

        size_t strides_i[2], strides_f[2];
        if (order == c_contiguous) {
            strides_i[0] = sizeof(da_int) * req_neigh;
            strides_i[1] = sizeof(da_int);
            strides_f[0] = sizeof(T) * req_neigh;
            strides_f[1] = sizeof(T);
        } else {
            strides_i[0] = sizeof(da_int);
            strides_i[1] = sizeof(da_int) * n_queries;
            strides_f[0] = sizeof(T);
            strides_f[1] = sizeof(T) * n_queries;
        }

        auto k_ind = py::array_t<da_int>(shape, strides_i);
        auto k_dist = py::array_t<T>(shape, strides_f);

        status = da_approx_nn_kneighbors(handle, n_queries, n_features, X.data(), ldx,
                                         k_ind.mutable_data(), k_dist.mutable_data(),
                                         req_neigh, 1);
        exception_check(status);
        py::tuple k_info = py::make_tuple(k_dist, k_ind);
        return k_info;
    }

    void get_rinfo(da_int *n_list, da_int *n_index, da_int *n_features,
                   da_int *kmeans_iter, da_int *stride_size) {
        da_status status;

        da_int dim = 4;

        if (precision == da_single) {
            float rinfo[4];
            *stride_size = sizeof(float);
            status = da_handle_get_result(handle, da_rinfo, &dim, rinfo);
            *n_list = (da_int)rinfo[0];
            *n_index = (da_int)rinfo[1];
            *n_features = (da_int)rinfo[2];
            *kmeans_iter = (da_int)rinfo[3];
        } else {
            double rinfo[4];
            *stride_size = sizeof(double);
            status = da_handle_get_result(handle, da_rinfo, &dim, rinfo);
            *n_list = (da_int)rinfo[0];
            *n_index = (da_int)rinfo[1];
            *n_features = (da_int)rinfo[2];
            *kmeans_iter = (da_int)rinfo[3];
        }

        exception_check(status);
    }

    auto get_cluster_centroids() {
        da_status status = da_status_success;

        da_int n_list, n_index, n_features, kmeans_iter;
        da_int stride_size;

        get_rinfo(&n_list, &n_index, &n_features, &kmeans_iter, &stride_size);

        da_int dim = n_list * n_features;
        size_t shape[2]{(size_t)n_list, (size_t)n_features};
        size_t strides[2];

        if (order == c_contiguous) {
            strides[0] = stride_size * n_features;
            strides[1] = stride_size;
        } else {
            strides[0] = stride_size;
            strides[1] = stride_size * n_list;
        }

        if (precision == da_single) {
            auto res = py::array_t<float>(shape, strides);
            status = da_handle_get_result(handle, da_approx_nn_cluster_centroids, &dim,
                                          res.mutable_data());
            exception_check(status);
            py::array ret = py::reinterpret_borrow<py::array>(res);
            return ret;
        } else {
            auto res = py::array_t<double>(shape, strides);
            status = da_handle_get_result(handle, da_approx_nn_cluster_centroids, &dim,
                                          res.mutable_data());
            exception_check(status);
            py::array ret = py::reinterpret_borrow<py::array>(res);
            return ret;
        }
    }

    auto get_list_sizes() {
        da_status status = da_status_success;

        da_int n_list, n_index, n_features, kmeans_iter;
        da_int stride_size;

        get_rinfo(&n_list, &n_index, &n_features, &kmeans_iter, &stride_size);

        da_int dim = n_list;

        auto res = py::array_t<da_int>((size_t)n_list);
        status = da_handle_get_result(handle, da_approx_nn_list_sizes, &dim,
                                      res.mutable_data());
        exception_check(status);
        return res;
    }

    auto get_n_list() {
        da_int n_list, n_index, n_features, kmeans_iter;
        da_int stride_size;
        get_rinfo(&n_list, &n_index, &n_features, &kmeans_iter, &stride_size);
        return n_list;
    }

    auto get_n_index() {
        da_int n_list, n_index, n_features, kmeans_iter;
        da_int stride_size;
        get_rinfo(&n_list, &n_index, &n_features, &kmeans_iter, &stride_size);
        return n_index;
    }

    auto get_n_features() {
        da_int n_list, n_index, n_features, kmeans_iter;
        da_int stride_size;
        get_rinfo(&n_list, &n_index, &n_features, &kmeans_iter, &stride_size);
        return n_features;
    }

    auto get_kmeans_iter() {
        da_int n_list, n_index, n_features, kmeans_iter;
        da_int stride_size;
        get_rinfo(&n_list, &n_index, &n_features, &kmeans_iter, &stride_size);
        return kmeans_iter;
    }
};

#endif
