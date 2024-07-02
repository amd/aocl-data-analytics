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

#ifndef DECISION_FOREST_PY_HPP
#define DECISION_FOREST_PY_HPP

#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include "utilities_py.hpp"
#include <iostream>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

class decision_tree : public pyda_handle {
    da_precision precision = da_double;

  public:
    decision_tree(da_int seed = -1, da_int max_depth = 10, da_int max_features = 0,
                  std::string criterion = "gini", da_int min_samples_split = 2,
                  std::string build_order = "breadth first",
                  std::string prec = "double") {
        da_status status;
        if (prec == "double") {
            da_handle_init<double>(&handle, da_handle_decision_tree);
        } else {
            da_handle_init<float>(&handle, da_handle_decision_tree);
            precision = da_single;
        }

        status = da_options_set(handle, "seed", seed);
        exception_check(status);
        status = da_options_set(handle, "maximum depth", max_depth);
        exception_check(status);
        status = da_options_set(handle, "maximum features", max_features);
        exception_check(status);
        status = da_options_set(handle, "scoring function", criterion.data());
        exception_check(status);
        status = da_options_set(handle, "node minimum samples", min_samples_split);
        exception_check(status);
        status = da_options_set(handle, "tree building order", build_order.data());
        exception_check(status);
    }
    ~decision_tree() { da_handle_destroy(&handle); }

    template <typename T>
    void fit(py::array_t<T, py::array::f_style> X, py::array_t<da_int> y,
             T min_impurity_decrease = 0.03, T min_split_score = 0.03,
             T feat_thresh = 0.0) {
        da_status status;

        // TODO pass real options to constructor
        status =
            da_options_set(handle, "minimum split improvement", min_impurity_decrease);
        exception_check(status);
        status = da_options_set(handle, "minimum split score", min_split_score);
        exception_check(status);
        status = da_options_set(handle, "feature threshold", feat_thresh);
        exception_check(status);

        da_int n_samples = X.shape()[0], n_features = X.shape()[1];
        status = da_tree_set_training_data(handle, n_samples, n_features, 0,
                                           X.mutable_data(), n_samples, y.mutable_data());
        exception_check(status); // throw an exception if status is not success

        status = da_tree_fit<T>(handle);
        exception_check(status);
    }

    template <typename T>
    T score(py::array_t<T, py::array::f_style> X_test, py::array_t<da_int> y_test) {
        da_status status;
        T score_val = 0.0;
        da_int n_obs = X_test.shape()[0], d = X_test.shape()[1];
        status = da_tree_score(handle, n_obs, d, X_test.mutable_data(), n_obs,
                               y_test.mutable_data(), &score_val);
        exception_check(status);
        return score_val;
    }

    template <typename T>
    py::array_t<da_int> predict(py::array_t<T, py::array::f_style> X) {

        da_status status;
        da_int n_samples = X.shape()[0], n_features = X.shape()[1];
        size_t shape[1]{(size_t)n_samples};
        size_t strides[1]{sizeof(T)};
        auto predictions = py::array_t<da_int>(shape, strides);
        status = da_tree_predict(handle, n_samples, n_features, X.mutable_data(),
                                 n_samples, predictions.mutable_data());
        exception_check(status);
        return predictions;
    }

    void get_rinfo(da_int &n_samples, da_int &n_features, da_int &n_obs, da_int &seed,
                   da_int &depth, da_int &n_nodes, da_int &n_leaves) {
        da_status status;

        da_int dim = 10;

        if (precision == da_single) {
            float rinfo[10];
            status = da_handle_get_result(handle, da_rinfo, &dim, rinfo);
            n_features = (da_int)rinfo[0];
            n_samples = (da_int)rinfo[1];
            n_obs = (da_int)rinfo[2];
            seed = (da_int)rinfo[3];
            depth = (da_int)rinfo[4];
            n_nodes = (da_int)rinfo[5];
            n_leaves = (da_int)rinfo[6];
        } else {
            double rinfo[10];
            status = da_handle_get_result(handle, da_rinfo, &dim, rinfo);
            n_features = (da_int)rinfo[0];
            n_samples = (da_int)rinfo[1];
            n_obs = (da_int)rinfo[2];
            seed = (da_int)rinfo[3];
            depth = (da_int)rinfo[4];
            n_nodes = (da_int)rinfo[5];
            n_leaves = (da_int)rinfo[6];
        }

        exception_check(status);
    }

    auto get_n_nodes() {

        da_int n_samples, n_features, n_obs, seed, depth, n_nodes, n_leaves;
        size_t stride_size;

        if (precision == da_single) {
            get_rinfo(n_samples, n_features, n_obs, seed, depth, n_nodes, n_leaves);
        } else {
            get_rinfo(n_samples, n_features, n_obs, seed, depth, n_nodes, n_leaves);
        }

        return n_nodes;
    }
    auto get_n_leaves() {

        da_int n_samples, n_features, n_obs, seed, depth, n_nodes, n_leaves;
        size_t stride_size;

        if (precision == da_single) {
            get_rinfo(n_samples, n_features, n_obs, seed, depth, n_nodes, n_leaves);
        } else {
            get_rinfo(n_samples, n_features, n_obs, seed, depth, n_nodes, n_leaves);
        }

        return n_leaves;
    }
};

class decision_forest : public pyda_handle {

    da_precision precision = da_double;

  public:
    decision_forest(da_int n_trees = 100, std::string criterion = "gini",
                    da_int seed = -1, da_int max_depth = 10, da_int min_samples_split = 2,
                    std::string build_order = "breadth first", bool bootstrap = true,
                    std::string features_selection = "sqrt", da_int max_features = 0,
                    std::string prec = "double") {
        da_status status;
        if (prec == "double") {
            da_handle_init<double>(&handle, da_handle_decision_forest);
        } else {
            da_handle_init<float>(&handle, da_handle_decision_forest);
            precision = da_single;
        }
        std::string bootstrap_str = "yes";
        if (!bootstrap)
            bootstrap_str = "no";

        status = da_options_set(handle, "seed", seed);
        exception_check(status);
        status = da_options_set(handle, "number of trees", n_trees);
        exception_check(status);
        status = da_options_set_string(handle, "scoring function", criterion.data());
        exception_check(status);
        status = da_options_set(handle, "maximum depth", max_depth);
        exception_check(status);
        status = da_options_set(handle, "node minimum samples", min_samples_split);
        exception_check(status);
        status = da_options_set(handle, "tree building order", build_order.data());
        exception_check(status);
        status = da_options_set(handle, "bootstrap", bootstrap_str.data());
        exception_check(status);
        status = da_options_set(handle, "features selection", features_selection.data());
        exception_check(status);
        status = da_options_set(handle, "maximum features", max_features);
        exception_check(status);
    }
    ~decision_forest() { da_handle_destroy(&handle); }

    template <typename T>
    void fit(py::array_t<T, py::array::f_style> X, py::array_t<da_int> y,
             T samples_factor = 0.8, T min_impurity_decrease = 0.03,
             T min_split_score = 0.03, T feat_thresh = 0.03) {
        da_status status;

        // TODO options to constructor
        status = da_options_set(handle, "bootstrap samples factor", samples_factor);
        exception_check(status);
        status =
            da_options_set(handle, "minimum split improvement", min_impurity_decrease);
        exception_check(status);
        status = da_options_set(handle, "minimum split score", min_split_score);
        exception_check(status);
        status = da_options_set(handle, "feature threshold", feat_thresh);
        exception_check(status);

        da_int n_obs = X.shape()[0], n_features = X.shape()[1];
        status = da_forest_set_training_data(handle, n_obs, n_features, 2,
                                             X.mutable_data(), n_obs, y.mutable_data());
        exception_check(status); // throw an exception if status is not success

        status = da_forest_fit<T>(handle);
        exception_check(status);
    }

    template <typename T>
    T score(py::array_t<T, py::array::f_style> X_test, py::array_t<da_int> y_test) {
        da_status status;
        T score_val = 0.0;
        da_int n_obs = X_test.shape()[0], d = X_test.shape()[1];
        status = da_forest_score(handle, n_obs, d, X_test.mutable_data(), n_obs,
                                 y_test.mutable_data(), &score_val);
        exception_check(status);
        return score_val;
    }

    template <typename T>
    py::array_t<da_int> predict(py::array_t<T, py::array::f_style> X) {

        da_status status;
        da_int n_samples = X.shape()[0], n_features = X.shape()[1];
        size_t shape[1]{(size_t)n_samples};
        size_t strides[1]{sizeof(T)};
        auto predictions = py::array_t<da_int>(shape, strides);
        status = da_forest_predict(handle, n_samples, n_features, X.mutable_data(),
                                   n_samples, predictions.mutable_data());
        exception_check(status);
        return predictions;
    }
};

#endif
