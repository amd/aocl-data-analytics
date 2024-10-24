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

#ifndef LINMOD_PY_HPP
#define LINMOD_PY_HPP

#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include "utilities_py.hpp"
#include <iostream>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

class linmod : public pyda_handle {
    da_int n_samples, n_feat, n_class;
    bool intercept;
    linmod_model mod_enum;
    std::string logreg_constraint_str;

  public:
    linmod(std::string mod, std::optional<da_int> max_iter, bool intercept = false,
           std::string solver = "auto", std::string scaling = "auto",
           std::string constraint = "ssc", std::string prec = "double",
           bool check_data = false)
        : intercept(intercept), logreg_constraint_str(constraint) {
        da_status status;
        if (mod == "mse") {
            mod_enum = linmod_model_mse;
        } else if (mod == "logistic") {
            mod_enum = linmod_model_logistic;
        } else {
            mod_enum = linmod_model_undefined;
        }
        if (prec == "double") {
            da_handle_init<double>(&handle, da_handle_linmod);
            status = da_linmod_select_model<double>(handle, mod_enum);
        } else if (prec == "single") {
            da_handle_init<float>(&handle, da_handle_linmod);
            status = da_linmod_select_model<float>(handle, mod_enum);
            precision = da_single;
        }
        exception_check(status);
        // Set optional parameters
        if (intercept)
            da_options_set_int(handle, "intercept", 1);
        status = da_options_set_string(handle, "optim method", solver.c_str());
        exception_check(status);
        status = da_options_set_string(handle, "scaling", scaling.c_str());
        exception_check(status);
        status = da_options_set_string(handle, "logistic constraint", constraint.c_str());
        exception_check(status);
        if (max_iter.has_value()) {
            status =
                da_options_set_int(handle, "optim iteration limit", max_iter.value());
            exception_check(status);
        }
        if (check_data == true) {
            std::string yes_str = "yes";
            status = da_options_set(handle, "check data", yes_str.data());
            exception_check(status);
        }
    }
    ~linmod() { da_handle_destroy(&handle); }

    template <typename T>
    void fit(py::array_t<T> X, py::array_t<T> y, std::optional<py::array_t<T>> x0,
             std::optional<T> progress_factor, T reg_lambda = 0.0, T reg_alpha = 0.0,
             T tol = 0.0001) {
        // floating point optional parameters are defined here since we cannot define those in the constructor (no template param)

        da_status status;
        da_int ldx;

        get_numpy_array_properties(X, n_samples, n_feat, ldx);

        if (order == c_contiguous) {
            status = da_options_set(handle, "storage order", "row-major");
        } else {
            status = da_options_set(handle, "storage order", "column-major");
        }
        // y rhs is assumed to only contain values from 0 to K-1 (K being the number of classes)
        n_class =
            (da_int)(std::round(*std::max_element(y.data(), y.data() + n_samples)) + 1);
        status = da_linmod_define_features(handle, n_samples, n_feat, X.data(), y.data());

        exception_check(status); // throw an exception if status is not success

        // Set the real optional parameters

        status = da_options_set(handle, "lambda", reg_lambda);

        exception_check(status);
        status = da_options_set(handle, "alpha", reg_alpha);

        exception_check(status);
        status = da_options_set(handle, "optim convergence tol", tol);
        exception_check(status);

        if (progress_factor.has_value()) {

            status =
                da_options_set(handle, "optim progress factor", progress_factor.value());
            exception_check(status);
        }
        if (x0.has_value()) {
            if (precision == da_double) {

                da_int ncoef = x0->shape()[0];
                status = da_linmod_fit_start<T>(handle, ncoef, x0->data());
            } else {
                da_int ncoef = x0->shape()[0];
                status = da_linmod_fit_start<T>(handle, ncoef, x0->data());
            }
        } else {
            if (precision == da_double) {

                status = da_linmod_fit<double>(handle);
            } else
                status = da_linmod_fit<float>(handle);
        }

        exception_check(status);
    }

    template <typename T> py::array_t<T> predict(py::array_t<T> X) {

        da_status status;

        da_int n_samples, n_features, ldx;

        get_numpy_array_properties(X, n_samples, n_features, ldx);

        size_t shape[1]{(size_t)n_samples};
        size_t strides[1]{sizeof(T)};

        auto predictions = py::array_t<T>(shape, strides);

        status = da_linmod_evaluate_model(handle, n_samples, n_features, X.data(),
                                          predictions.mutable_data());
        exception_check(status);
        return predictions;
    }

    auto get_coef() {
        da_status status = da_status_success;
        da_int dim;
        switch (mod_enum) {
        case linmod_model_mse:
            dim = intercept ? n_feat + 1 : n_feat;
            break;
        case linmod_model_logistic:
            if (logreg_constraint_str == "rsc" ||
                logreg_constraint_str == "reference category" || n_class == 2) {
                dim = (n_class - 1) * n_feat;
                if (intercept)
                    dim += n_class - 1;
            } else if (logreg_constraint_str == "ssc" ||
                       logreg_constraint_str == "symmetric side" ||
                       logreg_constraint_str == "symmetric") {
                dim = n_class * n_feat;
                if (intercept)
                    dim += n_class;
            }
            break;
        default:
            status = da_status_internal_error;
            break;
        }
        exception_check(status, "Model type was not correctly defined.");

        // define the output vector
        size_t shape[1]{(size_t)dim};
        if (precision == da_single) {
            size_t strides[1]{sizeof(float)};
            auto coef = py::array_t<float>(shape, strides);
            status =
                da_handle_get_result(handle, da_linmod_coef, &dim, coef.mutable_data());
            exception_check(status);
            py::array ret = py::reinterpret_borrow<py::array>(coef);
            return ret;
        } else {
            size_t strides[1]{sizeof(double)};
            auto coef = py::array_t<double>(shape, strides);
            status =
                da_handle_get_result(handle, da_linmod_coef, &dim, coef.mutable_data());
            exception_check(status);
            py::array ret = py::reinterpret_borrow<py::array>(coef);
            return ret;
        }
    }

    template <typename T>
    void get_rinfo(T *loss, T *nrm_gradient_loss, da_int *n_iter, T *time) {
        da_status status;

        da_int dim = 100;
        T rinfo[100];
        status = da_handle_get_result(handle, da_rinfo, &dim, rinfo);
        *loss = (T)rinfo[0];
        *nrm_gradient_loss = (T)rinfo[1];
        *n_iter = (da_int)rinfo[2];
        *time = (T)rinfo[3];

        exception_check(status);
    }

    auto get_loss() {
        // This is a bit tricky due to return variable being either float or double
        // Followed what was done for kmeans's get_inertia()
        da_int n_iter;
        size_t stride_size;

        if (precision == da_single) {
            stride_size = sizeof(float);
            float loss, nrm_gradient_loss, time;
            get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time);
            std::vector<size_t> shape, strides;
            shape.push_back(1);
            strides.push_back(stride_size);
            // define the output vector
            auto res = py::array_t<float>(shape, strides);
            *(res.mutable_data(0)) = loss;
            py::array ret = py::reinterpret_borrow<py::array>(res);
            return ret;
        } else {
            stride_size = sizeof(double);
            double loss, nrm_gradient_loss, time;
            get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time);
            std::vector<size_t> shape, strides;
            shape.push_back(1);
            strides.push_back(stride_size);
            // define the output vector
            auto res = py::array_t<double>(shape, strides);
            *(res.mutable_data(0)) = loss;
            py::array ret = py::reinterpret_borrow<py::array>(res);
            return ret;
        }
    }

    auto get_norm_gradient_loss() {
        // This is a bit tricky due to return variable being either float or double
        // Followed what was done for kmeans's get_inertia()
        da_int n_iter;
        size_t stride_size;

        if (precision == da_single) {
            stride_size = sizeof(float);
            float loss, nrm_gradient_loss, time;
            get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time);
            std::vector<size_t> shape, strides;
            shape.push_back(1);
            strides.push_back(stride_size);
            // define the output vector
            auto res = py::array_t<float>(shape, strides);
            *(res.mutable_data(0)) = nrm_gradient_loss;
            py::array ret = py::reinterpret_borrow<py::array>(res);
            return ret;
        } else {
            stride_size = sizeof(double);
            double loss, nrm_gradient_loss, time;
            get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time);
            std::vector<size_t> shape, strides;
            shape.push_back(1);
            strides.push_back(stride_size);
            // define the output vector
            auto res = py::array_t<double>(shape, strides);
            *(res.mutable_data(0)) = nrm_gradient_loss;
            py::array ret = py::reinterpret_borrow<py::array>(res);
            return ret;
        }
    }

    auto get_time() {
        // This is a bit tricky due to return variable being either float or double
        // Followed what was done for kmeans's get_inertia()
        da_int n_iter;
        size_t stride_size;

        if (precision == da_single) {
            stride_size = sizeof(float);
            float loss, nrm_gradient_loss, time;
            get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time);
            std::vector<size_t> shape, strides;
            shape.push_back(1);
            strides.push_back(stride_size);
            // define the output vector
            auto res = py::array_t<float>(shape, strides);
            *(res.mutable_data(0)) = time;
            py::array ret = py::reinterpret_borrow<py::array>(res);
            return ret;
        } else {
            stride_size = sizeof(double);
            double loss, nrm_gradient_loss, time;
            get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time);
            std::vector<size_t> shape, strides;
            shape.push_back(1);
            strides.push_back(stride_size);
            // define the output vector
            auto res = py::array_t<double>(shape, strides);
            *(res.mutable_data(0)) = time;
            py::array ret = py::reinterpret_borrow<py::array>(res);
            return ret;
        }
    }

    auto get_n_iter() {

        da_int n_iter;

        if (precision == da_single) {
            float loss, nrm_gradient_loss, time;
            get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time);
        } else {
            double loss, nrm_gradient_loss, time;
            get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time);
        }

        return n_iter;
    }
};

#endif