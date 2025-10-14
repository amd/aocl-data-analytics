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

#ifndef LINMOD_PY_HPP
#define LINMOD_PY_HPP

#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include "internal_utilities_py.hpp"
#include <iostream>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

class linmod : public pyda_handle {
    da_int n_samples, n_feat;
    bool intercept, warm_start;
    linmod_model mod_enum;
    std::string logreg_constraint_str, solver_str;
    bool fitted = false;

  public:
    linmod(std::string mod, std::optional<da_int> max_iter, bool intercept = false,
           std::string solver = "auto", std::string scaling = "auto",
           std::string constraint = "ssc", bool warm_start = false,
           std::string prec = "double", bool check_data = false)
        : intercept(intercept), warm_start(warm_start), logreg_constraint_str(constraint),
          solver_str(solver) {
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
        } else {
            da_handle_init<float>(&handle, da_handle_linmod);
            status = da_linmod_select_model<float>(handle, mod_enum);
            precision = da_single;
        }
        exception_check(status);
        // Set optional parameters
        if (intercept) {
            status = da_options_set_int(handle, "intercept", 1);
            exception_check(status);
        }
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

        da_status status = da_status_success;
        da_int ldx, dim;
        std::vector<T> warm_coefficients;

        get_numpy_array_properties(X, n_samples, n_feat, ldx);

        if (this->fitted && warm_start) {
            da_int n_iter, nrow_prev, ncol_prev, n_samples_prev, n_feat_prev,
                n_class_prev;
            bool well_determined_prev;
            T loss, time, nrm_gradient_loss;
            status = get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time, &n_samples_prev,
                               &n_feat_prev, &n_class_prev, &nrow_prev, &ncol_prev,
                               &well_determined_prev);
            exception_check(status);

            da_int n_class =
                (da_int)(std::round(*std::max_element(y.data(), y.data() + n_samples)) +
                         1);
            da_int ncol = intercept ? n_feat + 1 : n_feat;
            bool is_well_determined = n_samples >= n_feat + (da_int)intercept;

            if (n_class != n_class_prev && mod_enum == linmod_model_logistic)
                exception_check(
                    da_status_invalid_input,
                    "The number of classes has changed since the last fit. Cannot cast "
                    "previous coefficients. Try again with warm_start=False.");
            if (ncol != ncol_prev)
                exception_check(
                    da_status_invalid_input,
                    "The number of features has changed since the last fit. Cannot cast "
                    "previous coefficients. Try again with warm_start=False.");
            if (!is_well_determined && (solver_str == "cg" || solver_str == "sparse_cg"))
                if (well_determined_prev)
                    exception_check(
                        da_status_invalid_input,
                        "Previously fit data was well determined, cannot obtain dual "
                        "coefficients to warm start. Try again with warm_start=False.");
                else {
                    dim = n_samples;
                    warm_coefficients.resize(dim);
                    status = da_handle_get_result(handle, da_linmod_dual_coef, &dim,
                                                  warm_coefficients.data());
                }
            else {
                dim = ncol_prev * nrow_prev;
                warm_coefficients.resize(dim);
                status = da_handle_get_result(handle, da_linmod_coef, &dim,
                                              warm_coefficients.data());
            }
            exception_check(status); // throw an exception if status is not success
        }
        if (order == c_contiguous) {
            status = da_options_set(handle, "storage order", "row-major");
        } else {
            status = da_options_set(handle, "storage order", "column-major");
        }
        status =
            da_linmod_define_features(handle, n_samples, n_feat, X.data(), ldx, y.data());

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
            da_int ncoef = x0->shape()[0];
            status = da_linmod_fit_start<T>(handle, ncoef, x0->data());
        } else {
            if (this->fitted && warm_start) {
                status = da_linmod_fit_start<T>(handle, dim, warm_coefficients.data());
            } else {
                if (precision == da_double)
                    status = da_linmod_fit<double>(handle);
                else
                    status = da_linmod_fit<float>(handle);
            }
        }

        this->fitted = status == da_status_success ? true : false;

        exception_check(status);
    }

    template <typename T> py::array_t<T> predict(py::array_t<T> X) {

        da_status status;

        da_int n_samples, n_features, ldx;

        get_numpy_array_properties(X, n_samples, n_features, ldx);

        size_t shape[1]{(size_t)n_samples};
        size_t strides[1]{sizeof(T)};

        auto predictions = py::array_t<T>(shape, strides);

        status = da_linmod_evaluate_model(handle, n_samples, n_features, X.data(), ldx,
                                          predictions.mutable_data());
        exception_check(status);
        return predictions;
    }

    auto get_coef() {
        da_status status = da_status_success;
        da_int dim = 0;
        // For linear models that are not logistic, the coef is a 1D array
        // For logistic models, the coef is a 2D array to match scikitlearn output
        switch (mod_enum) {
        case linmod_model_mse: {
            dim = intercept ? n_feat + 1 : n_feat;
            if (precision == da_single) {
                size_t shape[1]{(size_t)dim};
                size_t strides[1]{sizeof(float)};
                auto coef = py::array_t<float>(shape, strides);
                status = da_handle_get_result(handle, da_linmod_coef, &dim,
                                              coef.mutable_data());
                exception_check(status);
                py::array ret = py::reinterpret_borrow<py::array>(coef);
                return ret;
            } else {
                size_t shape[1]{(size_t)dim};
                size_t strides[1]{sizeof(double)};
                auto coef = py::array_t<double>(shape, strides);
                status = da_handle_get_result(handle, da_linmod_coef, &dim,
                                              coef.mutable_data());
                exception_check(status);
                py::array ret = py::reinterpret_borrow<py::array>(coef);
                return ret;
            }
        } break;
        case linmod_model_logistic: {
            da_int n_iter, n_class, nrow_coef, ncol_coef, n_samples, n_feat;
            bool is_well_determined;
            if (precision == da_single) {
                float loss, time, nrm_gradient_loss;
                status = get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time, &n_samples,
                                   &n_feat, &n_class, &nrow_coef, &ncol_coef,
                                   &is_well_determined);
                exception_check(status);
                dim = ncol_coef * nrow_coef;
                size_t shape[2]{(size_t)nrow_coef, (size_t)ncol_coef};
                size_t strides[2];
                if (order == c_contiguous) {
                    strides[0] = sizeof(float) * ncol_coef;
                    strides[1] = sizeof(float);
                } else {
                    strides[0] = sizeof(float);
                    strides[1] = sizeof(float) * nrow_coef;
                }
                auto coef = py::array_t<float>(shape, strides);
                status = da_handle_get_result(handle, da_linmod_coef, &dim,
                                              coef.mutable_data());
                exception_check(status);
                py::array ret = py::reinterpret_borrow<py::array>(coef);
                return ret;
            } else {
                double loss, time, nrm_gradient_loss;
                status = get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time, &n_samples,
                                   &n_feat, &n_class, &nrow_coef, &ncol_coef,
                                   &is_well_determined);
                exception_check(status);
                dim = ncol_coef * nrow_coef;
                size_t shape[2]{(size_t)nrow_coef, (size_t)ncol_coef};
                size_t strides[2];
                if (order == c_contiguous) {
                    strides[0] = sizeof(double) * ncol_coef;
                    strides[1] = sizeof(double);
                } else {
                    strides[0] = sizeof(double);
                    strides[1] = sizeof(double) * nrow_coef;
                }
                auto coef = py::array_t<double>(shape, strides);
                status = da_handle_get_result(handle, da_linmod_coef, &dim,
                                              coef.mutable_data());
                exception_check(status);
                py::array ret = py::reinterpret_borrow<py::array>(coef);
                return ret;
            }
        } break;
        default:
            exception_check(da_status_internal_error,
                            "Model type was not correctly defined.");
            return py::array();
            break;
        }
    }

    template <typename T>
    da_status get_rinfo(T *loss, T *nrm_gradient_loss, da_int *n_iter, T *time,
                        da_int *n_samples, da_int *n_feat, da_int *n_class,
                        da_int *nrow_coef, da_int *ncol_coef, bool *is_well_determined) {
        da_status status;

        da_int dim = 100;
        T rinfo[100];
        status = da_handle_get_result(handle, da_rinfo, &dim, rinfo);
        if (status != da_status_success) {
            return status;
        }
        *loss = (T)(rinfo[da_linmod_info_t_::linmod_info_objective]);
        *nrm_gradient_loss = (T)(rinfo[da_linmod_info_t_::linmod_info_grad_norm]);
        *n_iter = (da_int)(rinfo[da_linmod_info_t_::linmod_info_iter]);
        *time = (T)(rinfo[da_linmod_info_t_::linmod_info_time]);
        *n_samples = (da_int)(rinfo[da_linmod_info_t_::linmod_info_nsamples]);
        *n_feat = (da_int)(rinfo[da_linmod_info_t_::linmod_info_nfeat]);
        *n_class = (da_int)(rinfo[da_linmod_info_t_::linmod_info_nclass]);
        *nrow_coef = (da_int)(rinfo[da_linmod_info_t_::linmod_info_nrow_coef]);
        *ncol_coef = (da_int)(rinfo[da_linmod_info_t_::linmod_info_ncol_coef]);
        *is_well_determined =
            (bool)(rinfo[da_linmod_info_t_::linmod_info_well_determined]);
        return da_status_success;
    }

    auto get_dual_coef() {
        da_status status;

        da_int dim = n_samples;
        size_t shape[1]{(size_t)dim};
        if (precision == da_single) {
            size_t strides[1]{sizeof(float)};
            auto dual_coef = py::array_t<float>(shape, strides);
            status = da_handle_get_result(handle, da_linmod_dual_coef, &dim,
                                          dual_coef.mutable_data());
            exception_check(status);
            py::array ret = py::reinterpret_borrow<py::array>(dual_coef);
            return ret;
        } else {
            size_t strides[1]{sizeof(double)};
            auto dual_coef = py::array_t<double>(shape, strides);
            status = da_handle_get_result(handle, da_linmod_dual_coef, &dim,
                                          dual_coef.mutable_data());
            exception_check(status);
            py::array ret = py::reinterpret_borrow<py::array>(dual_coef);
            return ret;
        }

        exception_check(status);
    }

    auto get_loss() {
        // This is a bit tricky due to return variable being either float or double
        // Followed what was done for kmeans's get_inertia()
        da_int n_iter, n_class, nrow_coef, ncol_coef, n_samples, n_feat;
        bool is_well_determined;
        size_t stride_size;
        da_status status;

        if (precision == da_single) {
            stride_size = sizeof(float);
            float loss, nrm_gradient_loss, time;
            status =
                get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time, &n_samples, &n_feat,
                          &n_class, &nrow_coef, &ncol_coef, &is_well_determined);
            exception_check(status);
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
            status =
                get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time, &n_samples, &n_feat,
                          &n_class, &nrow_coef, &ncol_coef, &is_well_determined);
            exception_check(status);
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
        da_int n_iter, n_class, nrow_coef, ncol_coef, n_samples, n_feat;
        bool is_well_determined;
        size_t stride_size;
        da_status status;

        if (precision == da_single) {
            stride_size = sizeof(float);
            float loss, nrm_gradient_loss, time;
            status =
                get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time, &n_samples, &n_feat,
                          &n_class, &nrow_coef, &ncol_coef, &is_well_determined);
            exception_check(status);
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
            status =
                get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time, &n_samples, &n_feat,
                          &n_class, &nrow_coef, &ncol_coef, &is_well_determined);
            exception_check(status);
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
        da_int n_iter, n_class, nrow_coef, ncol_coef, n_samples, n_feat;
        bool is_well_determined;
        size_t stride_size;
        da_status status;

        if (precision == da_single) {
            stride_size = sizeof(float);
            float loss, nrm_gradient_loss, time;
            status =
                get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time, &n_samples, &n_feat,
                          &n_class, &nrow_coef, &ncol_coef, &is_well_determined);
            exception_check(status);
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
            status =
                get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time, &n_samples, &n_feat,
                          &n_class, &nrow_coef, &ncol_coef, &is_well_determined);
            exception_check(status);
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

        da_int n_iter, n_class, nrow_coef, ncol_coef, n_samples, n_feat;
        bool is_well_determined;
        da_status status;

        if (precision == da_single) {
            float loss, nrm_gradient_loss, time;
            status =
                get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time, &n_samples, &n_feat,
                          &n_class, &nrow_coef, &ncol_coef, &is_well_determined);
            exception_check(status);
        } else {
            double loss, nrm_gradient_loss, time;
            status =
                get_rinfo(&loss, &nrm_gradient_loss, &n_iter, &time, &n_samples, &n_feat,
                          &n_class, &nrow_coef, &ncol_coef, &is_well_determined);
            exception_check(status);
        }

        return n_iter;
    }
};

#endif