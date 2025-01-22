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

#ifndef SVM_PY_HPP
#define SVM_PY_HPP

#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include "utilities_py.hpp"
#include <iostream>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

class py_svm : public pyda_handle {
  protected:
    da_precision precision = da_double;
    da_int n_samples, n_feat;

  public:
    py_svm(da_svm_model model, std::string kernel = "rbf", da_int degree = 3,
           da_int max_iter = -1, std::string prec = "double", bool check_data = false) {
        da_status status;
        if (prec == "double") {
            da_handle_init<double>(&handle, da_handle_svm);
            status = da_svm_select_model<double>(handle, model);
        } else {
            da_handle_init<float>(&handle, da_handle_svm);
            status = da_svm_select_model<float>(handle, model);
            precision = da_single;
        }
        exception_check(status);
        // Set optional parameters
        status = da_options_set(handle, "kernel", kernel.c_str());
        exception_check(status);
        status = da_options_set(handle, "degree", degree);
        exception_check(status);
        status = da_options_set(handle, "max_iter", max_iter);
        exception_check(status);
        if (check_data == true) {
            std::string yes_str = "yes";
            status = da_options_set(handle, "check data", yes_str.data());
            exception_check(status);
        }
    }
    ~py_svm() { da_handle_destroy(&handle); }

    template <typename T>
    void common_fit(py::array_t<T> &X, py::array_t<T> &y, T gamma, T coef0, T tol,
                    std::optional<T> tau) {
        // floating point optional parameters are defined here since we cannot define those in the constructor (no template param)
        da_status status;
        da_int ldx;

        status = da_options_set(handle, "gamma", gamma);
        exception_check(status);
        status = da_options_set(handle, "coef0", coef0);
        exception_check(status);
        status = da_options_set(handle, "tolerance", tol);
        exception_check(status);
        if (tau.has_value()) {
            status = da_options_set(handle, "tau", tau.value());
            exception_check(status);
        }

        get_numpy_array_properties(X, n_samples, n_feat, ldx);

        if (order == c_contiguous) {
            status = da_options_set(handle, "storage order", "row-major");
        } else {
            status = da_options_set(handle, "storage order", "column-major");
        }
        exception_check(status);

        status = da_svm_set_data(handle, n_samples, n_feat, X.mutable_data(), ldx,
                                 y.mutable_data());
        exception_check(status);
        status = da_svm_compute<T>(handle);
        exception_check(status);
    }

    template <typename T> py::array_t<T> predict(py::array_t<T> X) {

        da_status status;
        da_int n_samples, n_features, ldx;
        get_numpy_array_properties(X, n_samples, n_features, ldx);
        size_t shape[1]{(size_t)n_samples};
        size_t strides[1]{sizeof(T)};
        auto predictions = py::array_t<T>(shape, strides);
        status = da_svm_predict(handle, n_samples, n_features, X.mutable_data(), ldx,
                                predictions.mutable_data());
        exception_check(status);
        return predictions;
    }

    template <typename T>
    py::array_t<T> decision_function(py::array_t<T> X, std::string shape = "ovr") {
        da_status status;
        da_int n_samples, n_features, ldx, ldd;
        get_numpy_array_properties(X, n_samples, n_features, ldx);
        da_int nclass = get_n_classes();
        da_int nclassifiers = nclass * (nclass - 1) / 2;
        da_svm_decision_function_shape shape_enum;
        if (shape == "ovo") {
            shape_enum = ovo;
        } else if (shape == "ovr") {
            shape_enum = ovr;
        } else {
            throw std::invalid_argument("Given decision function shape does not exist. "
                                        "Available choices are: 'ovo', 'ovr'.");
        }
        if (nclass > 2) {
            da_int n_col = shape_enum == ovo ? nclassifiers : nclass;
            size_t shape[2]{(size_t)n_samples, (size_t)n_col};
            size_t strides[2];
            if (order == c_contiguous) {
                ldd = n_col;
                strides[0] = sizeof(T) * n_col;
                strides[1] = sizeof(T);
            } else {
                ldd = n_samples;
                strides[0] = sizeof(T);
                strides[1] = sizeof(T) * n_samples;
            }
            auto decision_values = py::array_t<T>(shape, strides);
            status =
                da_svm_decision_function(handle, n_samples, n_features, X.data(), ldx,
                                         shape_enum, decision_values.mutable_data(), ldd);
            exception_check(status);
            return decision_values;
        } else {
            size_t shape[1]{(size_t)n_samples};
            size_t strides[1]{sizeof(T)};

            auto decision_values = py::array_t<T>(shape, strides);
            status = da_svm_decision_function(handle, n_samples, n_features, X.data(),
                                              ldx, shape_enum,
                                              decision_values.mutable_data(), n_samples);
            exception_check(status);
            return decision_values;
        }
    }

    template <typename T> T score(py::array_t<T> X, py::array_t<T> y) {
        da_status status;
        da_int n_samples, n_features, ldx;
        get_numpy_array_properties(X, n_samples, n_features, ldx);

        T score_val = 0.0;

        status = da_svm_score(handle, n_samples, n_features, X.data(), ldx, y.data(),
                              &score_val);
        exception_check(status);
        return score_val;
    }

    void get_rinfo(da_int *n_samples, da_int *n_features, da_int *n_classes) {
        da_status status;

        da_int dim = 100;

        if (precision == da_single) {
            float rinfo[100];
            status = da_handle_get_result(handle, da_rinfo, &dim, rinfo);
            *n_samples = (da_int)rinfo[0];
            *n_features = (da_int)rinfo[1];
            *n_classes = (da_int)rinfo[2];
        } else {
            double rinfo[100];
            status = da_handle_get_result(handle, da_rinfo, &dim, rinfo);
            *n_samples = (da_int)rinfo[0];
            *n_features = (da_int)rinfo[1];
            *n_classes = (da_int)rinfo[2];
        }

        exception_check(status);
    }

    da_int get_n_samples() {
        da_int n_samples, n_features, n_classes;
        get_rinfo(&n_samples, &n_features, &n_classes);
        return n_samples;
    }

    da_int get_n_features() {
        da_int n_samples, n_features, n_classes;
        get_rinfo(&n_samples, &n_features, &n_classes);
        return n_features;
    }

    da_int get_n_classes() {
        da_int n_samples, n_features, n_classes;
        get_rinfo(&n_samples, &n_features, &n_classes);
        return n_classes;
    }

    da_int get_n_sv() {
        da_status status = da_status_success;
        da_int one = 1;
        da_int n_sv;
        status = da_handle_get_result(handle, da_svm_n_support_vectors, &one, &n_sv);
        exception_check(status);
        return n_sv;
    }

    auto get_n_sv_per_class() {
        da_status status = da_status_success;
        da_int n_samples, n_features, n_classes;
        get_rinfo(&n_samples, &n_features, &n_classes);
        size_t shape[1]{(size_t)n_classes};
        size_t strides[1]{sizeof(da_int)};
        auto n_sv_per_class = py::array_t<da_int>(shape, strides);
        status = da_handle_get_result(handle, da_svm_n_support_vectors_per_class,
                                      &n_classes, n_sv_per_class.mutable_data());
        exception_check(status);
        py::array ret = py::reinterpret_borrow<py::array>(n_sv_per_class);
        return ret;
    }

    auto get_bias() {
        da_status status = da_status_success;
        da_int n_samples, n_features, n_classes;
        get_rinfo(&n_samples, &n_features, &n_classes);
        da_int n_classifiers = n_classes * (n_classes - 1) / 2;
        size_t shape[1]{(size_t)n_classifiers};
        if (precision == da_double) {
            size_t strides[1]{sizeof(double)};
            auto bias = py::array_t<double>(shape, strides);
            status = da_handle_get_result(handle, da_svm_bias, &n_classifiers,
                                          bias.mutable_data());
            exception_check(status);
            py::array ret = py::reinterpret_borrow<py::array>(bias);
            return ret;
        } else {
            size_t strides[1]{sizeof(float)};
            auto bias = py::array_t<float>(shape, strides);
            status = da_handle_get_result(handle, da_svm_bias, &n_classifiers,
                                          bias.mutable_data());
            exception_check(status);
            auto res = py::array_t<float>(shape, strides);
            py::array ret = py::reinterpret_borrow<py::array>(bias);
            return ret;
        }
    }

    auto get_n_iterations() {
        da_status status = da_status_success;
        da_int n_samples, n_features, n_classes;
        get_rinfo(&n_samples, &n_features, &n_classes);
        da_int n_classifiers = n_classes * (n_classes - 1) / 2;
        size_t shape[1]{(size_t)n_classifiers};
        size_t strides[1]{sizeof(da_int)};
        auto n_iteration = py::array_t<da_int>(shape, strides);
        status = da_handle_get_result(handle, da_svm_n_iterations, &n_classifiers,
                                      n_iteration.mutable_data());
        exception_check(status);
        py::array ret = py::reinterpret_borrow<py::array>(n_iteration);
        return ret;
    }

    auto get_dual_coef() {
        da_status status = da_status_success;
        da_int one = 1;
        da_int n_sv;
        status = da_handle_get_result(handle, da_svm_n_support_vectors, &one, &n_sv);
        da_int n_samples, n_features, n_classes;
        get_rinfo(&n_samples, &n_features, &n_classes);
        da_int dim = (n_classes - 1) * n_sv;
        size_t shape[2]{(size_t)(n_classes - 1), (size_t)n_sv};
        size_t strides[2];
        if (precision == da_double) {
            if (order == c_contiguous) {
                strides[0] = sizeof(double) * n_sv;
                strides[1] = sizeof(double);
            } else {
                strides[0] = sizeof(double);
                strides[1] = sizeof(double) * (n_classes - 1);
            }
            auto dual_coef = py::array_t<double>(shape, strides);
            status = da_handle_get_result(handle, da_svm_dual_coef, &dim,
                                          dual_coef.mutable_data());
            exception_check(status);
            py::array ret = py::reinterpret_borrow<py::array>(dual_coef);
            return ret;
        } else {
            if (order == c_contiguous) {
                strides[0] = sizeof(float) * n_sv;
                strides[1] = sizeof(float);
            } else {
                strides[0] = sizeof(float);
                strides[1] = sizeof(float) * (n_classes - 1);
            }
            auto dual_coef = py::array_t<float>(shape, strides);
            status = da_handle_get_result(handle, da_svm_dual_coef, &dim,
                                          dual_coef.mutable_data());
            exception_check(status);
            py::array ret = py::reinterpret_borrow<py::array>(dual_coef);
            return ret;
        }
    }

    auto get_sv() {
        da_status status = da_status_success;
        da_int one = 1;
        da_int n_sv;
        status = da_handle_get_result(handle, da_svm_n_support_vectors, &one, &n_sv);
        exception_check(status);
        da_int n_samples, n_features, n_classes;
        get_rinfo(&n_samples, &n_features, &n_classes);
        da_int dim = n_sv * n_features;
        size_t shape[2]{(size_t)n_sv, (size_t)n_features};
        size_t strides[2];
        if (precision == da_double) {
            if (order == c_contiguous) {
                strides[0] = sizeof(double) * n_features;
                strides[1] = sizeof(double);
            } else {
                strides[0] = sizeof(double);
                strides[1] = sizeof(double) * n_sv;
            }
            auto support_vectors = py::array_t<double>(shape, strides);
            status = da_handle_get_result(handle, da_svm_support_vectors, &dim,
                                          support_vectors.mutable_data());
            exception_check(status);
            py::array ret = py::reinterpret_borrow<py::array>(support_vectors);
            return ret;
        } else {
            if (order == c_contiguous) {
                strides[0] = sizeof(float) * n_features;
                strides[1] = sizeof(float);
            } else {
                strides[0] = sizeof(float);
                strides[1] = sizeof(float) * n_sv;
            }
            auto support_vectors = py::array_t<float>(shape, strides);
            status = da_handle_get_result(handle, da_svm_support_vectors, &dim,
                                          support_vectors.mutable_data());
            exception_check(status);
            py::array ret = py::reinterpret_borrow<py::array>(support_vectors);
            return ret;
        }
    }

    auto get_support_vectors_idx() {
        da_status status = da_status_success;
        da_int one = 1;
        da_int n_sv;
        status = da_handle_get_result(handle, da_svm_n_support_vectors, &one, &n_sv);
        exception_check(status);
        da_int dim = n_sv;
        size_t shape[1]{(size_t)dim};
        size_t strides[1]{sizeof(da_int)};
        auto support_vectors_idx = py::array_t<da_int>(shape, strides);
        status = da_handle_get_result(handle, da_svm_idx_support_vectors, &dim,
                                      support_vectors_idx.mutable_data());
        exception_check(status);
        py::array ret = py::reinterpret_borrow<py::array>(support_vectors_idx);
        return ret;
    }
};

/*******************/
/*      SVC        */
/*******************/
class py_svc : public py_svm {

  public:
    py_svc(std::string kernel = "rbf", da_int degree = 3, da_int max_iter = 100000,
           std::string prec = "double", bool check_data = false)
        : py_svm(svc, kernel, degree, max_iter, prec, check_data) {}
    ~py_svc() {}

    template <typename T>
    void fit(py::array_t<T> X, py::array_t<T> y, std::optional<T> tau, T C = 1.0,
             T gamma = 1, T coef0 = 0.0, T tol = 0.001) {
        // floating point optional parameters are defined here since we cannot define those in the constructor (no template param)
        da_status status;

        status = da_options_set(handle, "C", C);
        exception_check(status);
        common_fit(X, y, gamma, coef0, tol, tau);
    }
};

/*******************/
/*      SVR        */
/*******************/
class py_svr : public py_svm {

  public:
    py_svr(std::string kernel = "rbf", da_int degree = 3, da_int max_iter = 100000,
           std::string prec = "double", bool check_data = false)
        : py_svm(svr, kernel, degree, max_iter, prec, check_data) {}
    ~py_svr() {}

    template <typename T>
    void fit(py::array_t<T> X, py::array_t<T> y, std::optional<T> tau, T C = 1.0,
             T epsilon = 0.1, T gamma = 1, T coef0 = 0.0, T tol = 0.001) {
        // floating point optional parameters are defined here since we cannot define those in the constructor (no template param)
        da_status status;

        status = da_options_set(handle, "C", C);
        exception_check(status);
        status = da_options_set(handle, "epsilon", epsilon);
        exception_check(status);
        common_fit(X, y, gamma, coef0, tol, tau);
    }
};

/*******************/
/*      nuSVC      */
/*******************/
class py_nusvc : public py_svm {

  public:
    py_nusvc(std::string kernel = "rbf", da_int degree = 3, da_int max_iter = 100000,
             std::string prec = "double", bool check_data = false)
        : py_svm(nusvc, kernel, degree, max_iter, prec, check_data) {}
    ~py_nusvc() {}

    template <typename T>
    void fit(py::array_t<T> X, py::array_t<T> y, std::optional<T> tau, T nu = 0.5,
             T gamma = 1, T coef0 = 0.0, T tol = 0.001) {
        // floating point optional parameters are defined here since we cannot define those in the constructor (no template param)
        da_status status;

        status = da_options_set(handle, "nu", nu);
        exception_check(status);
        common_fit(X, y, gamma, coef0, tol, tau);
    }
};

/*******************/
/*      nuSVR      */
/*******************/
class py_nusvr : public py_svm {

  public:
    py_nusvr(std::string kernel = "rbf", da_int degree = 3, da_int max_iter = 100000,
             std::string prec = "double", bool check_data = false)
        : py_svm(nusvr, kernel, degree, max_iter, prec, check_data) {}
    ~py_nusvr() {}

    template <typename T>
    void fit(py::array_t<T> X, py::array_t<T> y, std::optional<T> tau, T nu = 0.5,
             T C = 1.0, T gamma = 1, T coef0 = 0.0, T tol = 0.001) {
        // floating point optional parameters are defined here since we cannot define those in the constructor (no template param)
        da_status status;

        status = da_options_set(handle, "nu", nu);
        exception_check(status);
        status = da_options_set(handle, "C", C);
        exception_check(status);
        common_fit(X, y, gamma, coef0, tol, tau);
    }
};
#endif