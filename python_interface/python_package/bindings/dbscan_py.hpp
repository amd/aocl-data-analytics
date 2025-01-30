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

#ifndef DBSCAN_PY_HPP
#define DBSCAN_PY_HPP

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

class DBSCAN : public pyda_handle {

  public:
    DBSCAN(da_int min_samples = 5, std::string metric = "euclidean",
           std::string algorithm = "brute", da_int leaf_size = 30,
           std::string prec = "double", bool check_data = false) {
        if (prec == "double")
            da_handle_init<double>(&handle, da_handle_dbscan);
        else if (prec == "single") {
            da_handle_init<float>(&handle, da_handle_dbscan);
            precision = da_single;
        }
        da_status status;
        status = da_options_set_int(handle, "leaf size", leaf_size);
        exception_check(status);
        status = da_options_set_int(handle, "min samples", min_samples);
        exception_check(status);
        status = da_options_set_string(handle, "algorithm", algorithm.c_str());
        exception_check(status);
        status = da_options_set_string(handle, "metric", metric.c_str());
        exception_check(status);
        if (check_data == true) {
            std::string yes_str = "yes";
            status = da_options_set(handle, "check data", yes_str.data());
            exception_check(status);
        }
    }
    ~DBSCAN() { da_handle_destroy(&handle); }

    template <typename T> void fit(py::array_t<T> A, T eps = 0.5, T power = 2.0) {
        // floating point optional parameters are defined here since we cannot define those in the constructor (no template param)
        da_status status;
        status = da_options_set(handle, "eps", eps);
        exception_check(status);
        status = da_options_set(handle, "power", power);
        exception_check(status);
        da_int n_samples, n_features, lda;

        get_numpy_array_properties(A, n_samples, n_features, lda);

        if (order == c_contiguous) {
            status = da_options_set(handle, "storage order", "row-major");
        } else {
            status = da_options_set(handle, "storage order", "column");
        }
        status = da_dbscan_set_data(handle, n_samples, n_features, A.data(), lda);

        exception_check(status);

        status = da_dbscan_compute<T>(handle);
        exception_check(status);
    }

    template <typename T>
    void get_rinfo(da_int *n_samples, da_int *n_features, da_int *lda, T *eps,
                   da_int *min_samples, da_int *leaf_size, T *power,
                   da_int *n_core_samples, da_int *n_clusters) {
        da_status status;

        da_int dim = 9;

        T rinfo[9];
        status = da_handle_get_result(handle, da_rinfo, &dim, rinfo);
        *n_samples = (da_int)rinfo[0];
        *n_features = (da_int)rinfo[1];
        *lda = (da_int)rinfo[2];
        *eps = rinfo[3];
        *min_samples = (da_int)rinfo[4];
        *leaf_size = (da_int)rinfo[5];
        *power = rinfo[6];
        *n_core_samples = (da_int)rinfo[7];
        *n_clusters = (da_int)rinfo[8];

        exception_check(status);
    }

    auto get_labels() {

        da_status status;

        size_t stride_size = sizeof(da_int);
        da_int n_samples, n_features, n_clusters, lda, min_samples, leaf_size,
            n_core_samples;
        da_int dim, dim1, dim2;

        if (precision == da_single) {
            float eps, power;
            get_rinfo(&n_samples, &n_features, &lda, &eps, &min_samples, &leaf_size,
                      &power, &n_core_samples, &n_clusters);
        } else {
            double eps, power;
            get_rinfo(&n_samples, &n_features, &lda, &eps, &min_samples, &leaf_size,
                      &power, &n_core_samples, &n_clusters);
        }

        dim1 = n_samples;
        dim2 = 1;
        dim = dim1 * dim2;

        std::vector<size_t> shape, strides;
        shape.push_back(dim1);
        strides.push_back(stride_size);
        // define the output vector
        auto res = py::array_t<da_int>(shape, strides);
        status = da_handle_get_result(handle, da_dbscan_labels, &dim, res.mutable_data());
        exception_check(status);
        py::array ret = py::reinterpret_borrow<py::array>(res);
        return ret;
    }

    auto get_core_sample_indices() {
        da_status status;

        size_t stride_size = sizeof(da_int);
        da_int n_samples, n_features, n_clusters, lda, min_samples, leaf_size,
            n_core_samples;
        da_int dim, dim1, dim2;

        if (precision == da_single) {
            float eps, power;
            get_rinfo(&n_samples, &n_features, &lda, &eps, &min_samples, &leaf_size,
                      &power, &n_core_samples, &n_clusters);
        } else {
            double eps, power;
            get_rinfo(&n_samples, &n_features, &lda, &eps, &min_samples, &leaf_size,
                      &power, &n_core_samples, &n_clusters);
        }

        dim1 = n_core_samples;
        dim2 = 1;
        dim = dim1 * dim2;

        std::vector<size_t> shape, strides;
        shape.push_back(dim1);
        strides.push_back(stride_size);
        // define the output vector
        auto res = py::array_t<da_int>(shape, strides);
        status = da_handle_get_result(handle, da_dbscan_core_sample_indices, &dim,
                                      res.mutable_data());
        exception_check(status);
        py::array ret = py::reinterpret_borrow<py::array>(res);
        return ret;
    }

    auto get_n_samples() {

        da_int n_samples, n_features, n_clusters, lda, min_samples, leaf_size,
            n_core_samples;

        if (precision == da_single) {
            float eps, power;
            get_rinfo(&n_samples, &n_features, &lda, &eps, &min_samples, &leaf_size,
                      &power, &n_core_samples, &n_clusters);
        } else {
            double eps, power;
            get_rinfo(&n_samples, &n_features, &lda, &eps, &min_samples, &leaf_size,
                      &power, &n_core_samples, &n_clusters);
        }

        return n_samples;
    }

    auto get_n_core_samples() {
        da_int n_samples, n_features, n_clusters, lda, min_samples, leaf_size,
            n_core_samples;

        if (precision == da_single) {
            float eps, power;
            get_rinfo(&n_samples, &n_features, &lda, &eps, &min_samples, &leaf_size,
                      &power, &n_core_samples, &n_clusters);
        } else {
            double eps, power;
            get_rinfo(&n_samples, &n_features, &lda, &eps, &min_samples, &leaf_size,
                      &power, &n_core_samples, &n_clusters);
        }

        return n_core_samples;
    }

    auto get_n_features() {

        da_int n_samples, n_features, n_clusters, lda, min_samples, leaf_size,
            n_core_samples;

        if (precision == da_single) {
            float eps, power;
            get_rinfo(&n_samples, &n_features, &lda, &eps, &min_samples, &leaf_size,
                      &power, &n_core_samples, &n_clusters);
        } else {
            double eps, power;
            get_rinfo(&n_samples, &n_features, &lda, &eps, &min_samples, &leaf_size,
                      &power, &n_core_samples, &n_clusters);
        }

        return n_features;
    }

    auto get_n_clusters() {

        da_int n_samples, n_features, n_clusters, lda, min_samples, leaf_size,
            n_core_samples;

        if (precision == da_single) {
            float eps, power;
            get_rinfo(&n_samples, &n_features, &lda, &eps, &min_samples, &leaf_size,
                      &power, &n_core_samples, &n_clusters);
        } else {
            double eps, power;
            get_rinfo(&n_samples, &n_features, &lda, &eps, &min_samples, &leaf_size,
                      &power, &n_core_samples, &n_clusters);
        }

        return n_clusters;
    }
};

#endif