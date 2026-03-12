/*
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "internal_utilities_py.hpp"
#include <iostream>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

class nearest_neighbors : public pyda_handle {
    da_int internal_neigh;
    double internal_radius_d;
    float internal_radius_s;
    std::string precision_str;

  public:
    nearest_neighbors(da_int n_neighbors = 5, std::string algorithm = "auto",
                      da_int leaf_size = 30, std::string metric = "euclidean",
                      std::string weights = "uniform", std::string prec = "double",
                      bool check_data = false) {
        da_status status;
        precision_str = prec;
        if (prec == "double") {
            status = da_handle_init<double>(&handle, da_handle_nn);
        } else {
            status = da_handle_init<float>(&handle, da_handle_nn);
            precision = da_single;
        }
        exception_check(status);
        status = da_options_set(handle, "number of neighbors", n_neighbors);
        exception_check(status);
        internal_neigh = n_neighbors;
        status = da_options_set(handle, "weights", weights.c_str());
        exception_check(status);
        std::string algo = algorithm;
        if (algorithm == "kd_tree")
            algo = "kd tree";
        if (algorithm == "ball_tree")
            algo = "ball tree";
        status = da_options_set(handle, "algorithm", algo.c_str());
        exception_check(status);
        status = da_options_set(handle, "metric", metric.c_str());
        exception_check(status);
        status = da_options_set(handle, "leaf size", leaf_size);
        exception_check(status);

        if (check_data == true) {
            std::string yes_str = "yes";
            status = da_options_set(handle, "check data", yes_str.data());
            exception_check(status);
        }
    }
    ~nearest_neighbors() { da_handle_destroy(&handle); }

    template <typename T> void fit(py::array_t<T> X, T p, T radius) {
        da_status status;

        da_int n_samples, n_features, ldx;

        get_numpy_array_properties(X, n_samples, n_features, ldx);

        if (order == c_contiguous) {
            status = da_options_set(handle, "storage order", "row-major");
        } else {
            status = da_options_set(handle, "storage order", "column-major");
        }
        exception_check(status);

        status = da_options_set(handle, "minkowski parameter", T(p));
        exception_check(status);

        status = da_options_set(handle, "radius", T(radius));
        exception_check(status);
        if (precision == da_double)
            internal_radius_d = (double)radius;
        else
            internal_radius_s = (float)radius;

        status = da_nn_set_data(handle, n_samples, n_features, X.data(), ldx);
        exception_check(status); // throw an exception if status is not success
    }

    template <typename T> void set_labels(py::array_t<da_int> y) {
        da_status status;

        da_int n_samples, n_features, ldx;

        get_numpy_array_properties(y, n_samples, n_features, ldx);
        if (precision_str == "double")
            status = da_nn_set_labels<double>(handle, n_samples, y.data());
        else
            status = da_nn_set_labels<float>(handle, n_samples, y.data());
        exception_check(status); // throw an exception if status is not success
    }

    template <typename T> void set_targets(py::array_t<T> y) {
        da_status status;

        da_int n_samples, n_features, ldx;

        get_numpy_array_properties(y, n_samples, n_features, ldx);
        status = da_nn_set_targets(handle, n_samples, y.data());
        exception_check(status); // throw an exception if status is not success
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
        status = da_nn_kneighbors(handle, n_queries, n_features, X.data(), ldx,
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

        status =
            da_nn_kneighbors(handle, n_queries, n_features, X.data(), ldx,
                             k_ind.mutable_data(), k_dist.mutable_data(), req_neigh, 1);
        exception_check(status);
        py::tuple k_info = py::make_tuple(k_dist, k_ind);
        return k_info;
    }

    template <typename T>
    py::array_t<T> classifier_predict_proba(py::array_t<T> X,
                                            std::string search_mode = "knn") {
        da_status status;

        da_int n_queries, n_features, ldx;

        get_numpy_array_properties(X, n_queries, n_features, ldx);

        da_int num_classes = -3;
        status = da_nn_classes<T>(handle, &num_classes, nullptr);
        exception_check(status);

        da_nn_search_mode search_mode_internal;
        if (search_mode == "knn")
            search_mode_internal = knn_search_mode;
        else if (search_mode == "radius_neighbors")
            search_mode_internal = radius_search_mode;
        else
            throw std::invalid_argument(
                "Invalid search_mode. Supported modes are 'knn' and 'radius_neighbors'.");

        size_t shape[2]{(size_t)n_queries, (size_t)num_classes};

        size_t strides[2];
        if (order == c_contiguous) {
            strides[0] = sizeof(T) * num_classes;
            strides[1] = sizeof(T);
        } else {
            strides[0] = sizeof(T);
            strides[1] = sizeof(T) * n_queries;
        }

        auto proba = py::array_t<T>(shape, strides);
        status =
            da_nn_classifier_predict_proba(handle, n_queries, n_features, X.data(), ldx,
                                           proba.mutable_data(), search_mode_internal);
        exception_check(status);
        return proba;
    }

    template <typename T>
    py::array_t<da_int> classifier_predict(py::array_t<T> X,
                                           std::string search_mode = "knn") {
        da_status status;
        da_int n_queries, n_features, ldx;

        get_numpy_array_properties(X, n_queries, n_features, ldx);

        da_nn_search_mode search_mode_internal;
        if (search_mode == "knn")
            search_mode_internal = knn_search_mode;
        else if (search_mode == "radius_neighbors")
            search_mode_internal = radius_search_mode;
        else
            throw std::invalid_argument(
                "Invalid search_mode. Supported modes are 'knn' and 'radius_neighbors'.");

        size_t shape[1]{(size_t)n_queries};
        auto y_test = py::array_t<da_int>(shape);
        status = da_nn_classifier_predict(handle, n_queries, n_features, X.data(), ldx,
                                          y_test.mutable_data(), search_mode_internal);
        exception_check(status);
        return y_test;
    }

    template <typename T>
    py::array_t<T> regressor_predict(py::array_t<T> X, std::string search_mode = "knn") {
        da_status status;
        da_int n_queries, n_features, ldx;

        get_numpy_array_properties(X, n_queries, n_features, ldx);

        da_nn_search_mode search_mode_internal;
        if (search_mode == "knn")
            search_mode_internal = knn_search_mode;
        else if (search_mode == "radius_neighbors")
            search_mode_internal = radius_search_mode;
        else
            throw std::invalid_argument(
                "Invalid search_mode. Supported modes are 'knn' and 'radius_neighbors'.");

        size_t shape[1]{(size_t)n_queries};
        auto y_test = py::array_t<T>(shape);
        status = da_nn_regressor_predict(handle, n_queries, n_features, X.data(), ldx,
                                         y_test.mutable_data(), search_mode_internal);
        exception_check(status);
        return y_test;
    }

    template <typename T>
    py::list radius_neighbors_indices(py::array_t<T> X, T radius = (T)0.0) {
        da_status status;

        da_int n_queries, n_features, ldx;

        get_numpy_array_properties(X, n_queries, n_features, ldx);

        T req_radius = -1.0;
        if (precision == da_double)
            req_radius = internal_radius_d;
        else
            req_radius = internal_radius_s;
        if (radius != (T)0.0)
            req_radius = radius;

        // Compute the radius neighbors
        status = da_nn_radius_neighbors(handle, n_queries, n_features, X.data(), ldx,
                                        req_radius, 0, 0);
        exception_check(status);

        // Extract the number of neighbors for each query point
        da_int array_size = n_queries + 1;
        da_int *n_radius_neighbors = new da_int[array_size];
        status = da_handle_get_result(handle, da_nn_radius_neighbors_count, &array_size,
                                      n_radius_neighbors);
        exception_check(status);

        // This will hold the final results in a list of arrays
        py::list my_temp;
        for (auto i = 0; i < n_queries; i++) {
            size_t shape[1] = {(size_t)n_radius_neighbors[i]};
            size_t strides[1] = {sizeof(da_int)};
            auto r_neigh_i = py::array_t<da_int>(py::array::ShapeContainer(shape),
                                                 py::array::StridesContainer(strides));
            if (n_radius_neighbors[i] > 0) {
                da_int *data_ptr = r_neigh_i.mutable_data();
                data_ptr[0] = i;
                status = da_handle_get_result(
                    handle, da_nn_radius_neighbors_indices_index, &n_radius_neighbors[i],
                    r_neigh_i.mutable_data());
                exception_check(status);
            }
            my_temp.append(r_neigh_i);
        }

        // Clean up the allocated memory for n_radius_neighbors
        delete[] n_radius_neighbors;

        return my_temp;
    }

    template <typename T>
    py::tuple radius_neighbors(py::array_t<T> X, T radius = (T)0.0,
                               bool sort_results = false) {
        da_status status;
        da_int n_queries, n_features, ldx;

        get_numpy_array_properties(X, n_queries, n_features, ldx);

        T req_radius = -1.0;
        if (precision == da_double)
            req_radius = internal_radius_d;
        else
            req_radius = internal_radius_s;
        if (radius != (T)0.0)
            req_radius = radius;

        da_int sort_res = 0;
        if (sort_results)
            sort_res = 1;

        // Compute the radius neighbors
        status = da_nn_radius_neighbors(handle, n_queries, n_features, X.data(), ldx,
                                        req_radius, 1, sort_res);
        exception_check(status);

        // Extract the number of neighbors for each query point
        da_int array_size = n_queries + 1;
        da_int *n_radius_neighbors = new da_int[array_size];
        status = da_handle_get_result(handle, da_nn_radius_neighbors_count, &array_size,
                                      n_radius_neighbors);
        exception_check(status);

        // This will hold the final results in a list of arrays
        py::list my_temp_ind;
        py::list my_temp_dist;
        for (auto i = 0; i < n_queries; i++) {
            size_t shape[1] = {(size_t)n_radius_neighbors[i]};
            size_t strides[1] = {sizeof(da_int)};
            auto r_neigh_i = py::array_t<da_int>(py::array::ShapeContainer(shape),
                                                 py::array::StridesContainer(strides));
            auto r_neigh_d = py::array_t<T>(py::array::ShapeContainer(shape),
                                            py::array::StridesContainer({sizeof(T)}));
            if (n_radius_neighbors[i] > 0) {
                da_int *data_ptr_i = r_neigh_i.mutable_data();
                data_ptr_i[0] = i;
                status = da_handle_get_result(
                    handle, da_nn_radius_neighbors_indices_index, &n_radius_neighbors[i],
                    r_neigh_i.mutable_data());
                exception_check(status);
                T *data_ptr_d = r_neigh_d.mutable_data();
                data_ptr_d[0] = T(i);
                status = da_handle_get_result(
                    handle, da_nn_radius_neighbors_distances_index,
                    &n_radius_neighbors[i], r_neigh_d.mutable_data());
                exception_check(status);
            }
            my_temp_ind.append(r_neigh_i);
            my_temp_dist.append(r_neigh_d);
        }

        // Clean up the allocated memory for n_radius_neighbors
        delete[] n_radius_neighbors;

        py::tuple k_info = py::make_tuple(my_temp_dist, my_temp_ind);
        return k_info;
    }
};

#endif
