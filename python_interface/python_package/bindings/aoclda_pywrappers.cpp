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

#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include <iostream>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;
using namespace pybind11::literals;

/* Parsing error codes
 * Translate codes into standard C++ exceptions that are translated automatically in Python
 * doc: https://pybind11.readthedocs.io/en/stable/advanced/exceptions.html
 * This function is intended for use with basic stats, where there are no handles and no warning exits
 */
void status_to_exception(da_status status) {
    switch (status) {
    case (da_status_success):
        return;
    case (da_status_memory_error):
        throw std::bad_alloc();
    case (da_status_invalid_array_dimension):
        throw std::length_error("One of the input arrays was too small.");
    case (da_status_invalid_input):
    case (da_status_negative_data):
        throw std::invalid_argument(
            "One of the options passed to the function had an invalid value.");
    default:
        std::cout
            << "An internal error occurred. This could be a memory corruption issue."
            << std::endl;
        throw std::exception();
    }
}

/* Helper function to avoid redundancy
 * Determine the size of output array based on the axis
 */
template <typename T>
void get_size(std::string axis, da_axis &axis_enum, py::array_t<T, py::array::f_style> &X,
              size_t &size, da_int &m, da_int &n) {
    if (axis == "col") {
        axis_enum = da_axis_col;
    } else if (axis == "row") {
        axis_enum = da_axis_row;
    } else if (axis == "all") {
        axis_enum = da_axis_all;
    } else {
        throw std::invalid_argument(
            "Given axis does not exist. Available choices are: 'col', 'row', 'all'.");
    }
    if (X.ndim() > 2) {
        throw std::length_error(
            "Function does not accept arrays with more than 2 dimensions.");
    }
    // If we are dealing with 1D array the shape attribute is stored as (n_samples, )
    // so accessing X.shape()[1] is causing errors when 1D array is passed
    if (X.ndim() == 1) {
        n = X.shape()[0];
        m = 1;
        // If user provided 1D array then calculating for example mean over all elements
        // would mean calculation over that one row of data.
        if (axis_enum == da_axis_all) {
            axis_enum = da_axis_row;
        }
    } else {
        m = X.shape()[0], n = X.shape()[1];
    }
    switch (axis_enum) {
    case (da_axis_all):
        size = 1;
        break;

    case (da_axis_col):
        size = n;
        break;

    case (da_axis_row):
        size = m;
        break;
    }
}

template <typename T>
py::array_t<T> py_da_mean(py::array_t<T, py::array::f_style> X,
                          std::string axis = "col") {
    da_status status;
    da_int m, n;
    size_t mean_sz;
    da_axis axis_enum;

    get_size(axis, axis_enum, X, mean_sz, m, n);

    // Create the output mean array as a numpy array
    size_t shape[1]{mean_sz};
    size_t strides[1]{sizeof(T)};
    auto mean = py::array_t<T>(shape, strides);

    status = da_mean(axis_enum, m, n, X.data(), m, mean.mutable_data());

    status_to_exception(status);

    return mean;
}

template <typename T>
py::array_t<T> py_da_harmonic_mean(py::array_t<T, py::array::f_style> X,
                                   std::string axis = "col") {
    da_status status;
    da_int m, n;
    size_t harmonic_mean_sz;
    da_axis axis_enum;

    get_size(axis, axis_enum, X, harmonic_mean_sz, m, n);

    // Create the output mean array as a numpy array
    size_t shape[1]{harmonic_mean_sz};
    size_t strides[1]{sizeof(T)};
    auto harmonic_mean = py::array_t<T>(shape, strides);

    status = da_harmonic_mean(axis_enum, m, n, X.data(), m, harmonic_mean.mutable_data());

    status_to_exception(status);

    return harmonic_mean;
}

template <typename T>
py::array_t<T> py_da_geometric_mean(py::array_t<T, py::array::f_style> X,
                                    std::string axis = "col") {
    da_status status;
    da_int m, n;
    size_t geometric_mean_sz;
    da_axis axis_enum;

    get_size(axis, axis_enum, X, geometric_mean_sz, m, n);

    // Create the output mean array as a numpy array
    size_t shape[1]{geometric_mean_sz};
    size_t strides[1]{sizeof(T)};
    auto geometric_mean = py::array_t<T>(shape, strides);

    status =
        da_geometric_mean(axis_enum, m, n, X.data(), m, geometric_mean.mutable_data());

    status_to_exception(status);

    return geometric_mean;
}

template <typename T>
py::array_t<T> py_da_variance(py::array_t<T, py::array::f_style> X, da_int dof = 0,
                              std::string axis = "col") {
    da_status status;
    da_int m, n;
    size_t variance_sz;
    da_axis axis_enum;

    get_size(axis, axis_enum, X, variance_sz, m, n);

    // Create the output variance array as a numpy array
    size_t shape[1]{variance_sz};
    size_t strides[1]{sizeof(T)};
    auto mean = py::array_t<T>(shape, strides);
    auto variance = py::array_t<T>(shape, strides);

    status = da_variance(axis_enum, m, n, X.data(), m, dof, mean.mutable_data(),
                         variance.mutable_data());

    status_to_exception(status);

    return variance;
}

template <typename T>
py::array_t<T> py_da_skewness(py::array_t<T, py::array::f_style> X,
                              std::string axis = "col") {
    da_status status;
    da_int m, n;
    size_t skewness_sz;
    da_axis axis_enum;

    get_size(axis, axis_enum, X, skewness_sz, m, n);

    // Create the output skewness array as a numpy array as well as other arrays to store auxilary output
    size_t shape[1]{skewness_sz};
    size_t strides[1]{sizeof(T)};
    auto mean = py::array_t<T>(shape, strides);
    auto variance = py::array_t<T>(shape, strides);
    auto skewness = py::array_t<T>(shape, strides);

    status = da_skewness(axis_enum, m, n, X.data(), m, mean.mutable_data(),
                         variance.mutable_data(), skewness.mutable_data());

    status_to_exception(status);

    return skewness;
}

template <typename T>
py::array_t<T> py_da_kurtosis(py::array_t<T, py::array::f_style> X,
                              std::string axis = "col") {
    da_status status;
    da_int m, n;
    size_t kurtosis_sz;
    da_axis axis_enum;

    get_size(axis, axis_enum, X, kurtosis_sz, m, n);

    // Create the output kurtosis array as a numpy array as well as other arrays to store auxilary output
    size_t shape[1]{kurtosis_sz};
    size_t strides[1]{sizeof(T)};
    auto mean = py::array_t<T>(shape, strides);
    auto variance = py::array_t<T>(shape, strides);
    auto kurtosis = py::array_t<T>(shape, strides);

    status = da_kurtosis(axis_enum, m, n, X.data(), m, mean.mutable_data(),
                         variance.mutable_data(), kurtosis.mutable_data());

    status_to_exception(status);

    return kurtosis;
}

template <typename T>
py::array_t<T> py_da_moment(py::array_t<T, py::array::f_style> X, da_int k,
                            std::optional<py::array_t<T, py::array::f_style>> mean,
                            std::string axis = "col") {
    da_status status;
    da_int m, n;
    size_t moment_sz;
    da_axis axis_enum;

    get_size(axis, axis_enum, X, moment_sz, m, n);

    // Create the output moment array as a numpy array
    size_t shape[1]{moment_sz};
    size_t strides[1]{sizeof(T)};
    auto moment = py::array_t<T>(shape, strides);

    // Check if user provided precalculated mean
    if (mean.has_value()) {
        // Check if provided means have correct size
        if ((size_t)mean->shape()[0] != moment_sz || mean->ndim() != 1) {
            throw std::length_error("The size of mean array does not match data size.");
        }
        status = da_moment(axis_enum, m, n, X.data(), m, k, 1, mean->mutable_data(),
                           moment.mutable_data());
    } else {
        auto mean_aux = py::array_t<T>(shape, strides);
        status = da_moment(axis_enum, m, n, X.data(), m, k, 0, mean_aux.mutable_data(),
                           moment.mutable_data());
    }

    status_to_exception(status);

    return moment;
}

template <typename T>
py::array_t<T> py_da_quantile(py::array_t<T, py::array::f_style> X, T q,
                              std::string method = "linear", std::string axis = "col") {
    da_status status;
    da_int m, n;
    size_t quantile_sz;
    da_axis axis_enum;

    get_size(axis, axis_enum, X, quantile_sz, m, n);

    // Create the output quantile array as a numpy array
    size_t shape[1]{quantile_sz};
    size_t strides[1]{sizeof(T)};
    auto quantiles = py::array_t<T>(shape, strides);

    if (method == "inverted_cdf") {
        status = da_quantile(axis_enum, m, n, X.data(), m, q, quantiles.mutable_data(),
                             da_quantile_type_1);
    } else if (method == "averaged_inverted_cdf") {
        status = da_quantile(axis_enum, m, n, X.data(), m, q, quantiles.mutable_data(),
                             da_quantile_type_2);
    } else if (method == "closest_observation") {
        status = da_quantile(axis_enum, m, n, X.data(), m, q, quantiles.mutable_data(),
                             da_quantile_type_3);
    } else if (method == "interpolated_inverted_cdf") {
        status = da_quantile(axis_enum, m, n, X.data(), m, q, quantiles.mutable_data(),
                             da_quantile_type_4);
    } else if (method == "hazen") {
        status = da_quantile(axis_enum, m, n, X.data(), m, q, quantiles.mutable_data(),
                             da_quantile_type_5);
    } else if (method == "weibull") {
        status = da_quantile(axis_enum, m, n, X.data(), m, q, quantiles.mutable_data(),
                             da_quantile_type_6);
    } else if (method == "linear") {
        status = da_quantile(axis_enum, m, n, X.data(), m, q, quantiles.mutable_data(),
                             da_quantile_type_7);
    } else if (method == "median_unbiased") {
        status = da_quantile(axis_enum, m, n, X.data(), m, q, quantiles.mutable_data(),
                             da_quantile_type_8);
    } else if (method == "normal_unbiased") {
        status = da_quantile(axis_enum, m, n, X.data(), m, q, quantiles.mutable_data(),
                             da_quantile_type_9);
    } else {
        throw std::invalid_argument("Provided method does not exist.");
    }

    status_to_exception(status);

    return quantiles;
}

template <typename T>
std::tuple<py::array_t<T>, py::array_t<T>, py::array_t<T>, py::array_t<T>, py::array_t<T>>
py_da_five_point_summary(py::array_t<T, py::array::f_style> X, std::string axis = "col") {
    da_status status;
    da_int m, n;
    size_t fps_sz;
    da_axis axis_enum;

    get_size(axis, axis_enum, X, fps_sz, m, n);

    // Create the output arrays to store five point summary as a numpy array
    size_t shape[1]{fps_sz};
    size_t strides[1]{sizeof(T)};
    auto min = py::array_t<T>(shape, strides);
    auto lq = py::array_t<T>(shape, strides);
    auto med = py::array_t<T>(shape, strides);
    auto uq = py::array_t<T>(shape, strides);
    auto max = py::array_t<T>(shape, strides);

    status = da_five_point_summary(axis_enum, m, n, X.data(), m, min.mutable_data(),
                                   lq.mutable_data(), med.mutable_data(),
                                   uq.mutable_data(), max.mutable_data());

    status_to_exception(status);

    return std::make_tuple(min, lq, med, uq, max);
}

template <typename T>
py::array_t<T> py_da_standardize(py::array_t<T, py::array::f_style> X,
                                 std::optional<py::array_t<T, py::array::f_style>> shift,
                                 std::optional<py::array_t<T, py::array::f_style>> scale,
                                 da_int dof = 0, bool reverse = false,
                                 bool inplace = false, std::string axis = "col") {
    da_status status;
    da_int m, n;
    size_t standardize_sz;
    da_axis axis_enum;

    get_size(axis, axis_enum, X, standardize_sz, m, n);

    // Create parameters for potential copy_X of original numpy array
    size_t shape[2]{(size_t)m, (size_t)n};
    size_t strides[2]{sizeof(T), sizeof(T) * m};
    T *dummy = nullptr;

    if (shift.has_value() && scale.has_value()) {
        if ((size_t)shift->shape()[0] != standardize_sz ||
            (size_t)scale->shape()[0] != standardize_sz) {
            throw std::length_error(
                "The size of shift or scale array does not match the expected size.");
        }
        if (reverse) {
            if (inplace) {
                status = da_standardize(axis_enum, m, n, X.mutable_data(), m, dof, 1,
                                        shift->mutable_data(), scale->mutable_data());
                status_to_exception(status);
                return X;
            } else {
                py::array_t<T, py::array::f_style> copy_X(shape, strides);
                memcpy(copy_X.mutable_data(), X.mutable_data(), sizeof(T) * X.size());
                status = da_standardize(axis_enum, m, n, copy_X.mutable_data(), m, dof, 1,
                                        shift->mutable_data(), scale->mutable_data());
                status_to_exception(status);
                return copy_X;
            }
        } else {
            if (inplace) {
                status = da_standardize(axis_enum, m, n, X.mutable_data(), m, dof, 0,
                                        shift->mutable_data(), scale->mutable_data());
                status_to_exception(status);
                return X;
            } else {
                py::array_t<T, py::array::f_style> copy_X(shape, strides);
                memcpy(copy_X.mutable_data(), X.mutable_data(), sizeof(T) * X.size());
                status = da_standardize(axis_enum, m, n, copy_X.mutable_data(), m, dof, 0,
                                        shift->mutable_data(), scale->mutable_data());
                status_to_exception(status);
                return copy_X;
            }
        }
    } else if (shift.has_value()) {
        if ((size_t)shift->shape()[0] != standardize_sz) {
            throw std::length_error(
                "The size of shift array does not match the expected size.");
        }
        if (reverse) {
            if (inplace) {
                status = da_standardize(axis_enum, m, n, X.mutable_data(), m, dof, 1,
                                        shift->mutable_data(), dummy);
                status_to_exception(status);
                return X;
            } else {
                py::array_t<T, py::array::f_style> copy_X(shape, strides);
                memcpy(copy_X.mutable_data(), X.mutable_data(), sizeof(T) * X.size());
                status = da_standardize(axis_enum, m, n, copy_X.mutable_data(), m, dof, 1,
                                        shift->mutable_data(), dummy);
                status_to_exception(status);
                return copy_X;
            }
        } else {
            if (inplace) {
                status = da_standardize(axis_enum, m, n, X.mutable_data(), m, dof, 0,
                                        shift->mutable_data(), dummy);
                status_to_exception(status);
                return X;
            } else {
                py::array_t<T, py::array::f_style> copy_X(shape, strides);
                memcpy(copy_X.mutable_data(), X.mutable_data(), sizeof(T) * X.size());
                status = da_standardize(axis_enum, m, n, copy_X.mutable_data(), m, dof, 0,
                                        shift->mutable_data(), dummy);
                status_to_exception(status);
                return copy_X;
            }
        }
    } else if (scale.has_value()) {
        if ((size_t)scale->shape()[0] != standardize_sz) {
            throw std::length_error(
                "The size of scale array does not match the expected size.");
        }
        if (reverse) {
            if (inplace) {
                status = da_standardize(axis_enum, m, n, X.mutable_data(), m, dof, 1,
                                        dummy, scale->mutable_data());
                status_to_exception(status);
                return X;
            } else {
                py::array_t<T, py::array::f_style> copy_X(shape, strides);
                memcpy(copy_X.mutable_data(), X.mutable_data(), sizeof(T) * X.size());
                status = da_standardize(axis_enum, m, n, copy_X.mutable_data(), m, dof, 1,
                                        dummy, scale->mutable_data());
                status_to_exception(status);
                return copy_X;
            }
        } else {
            if (inplace) {
                status = da_standardize(axis_enum, m, n, X.mutable_data(), m, dof, 0,
                                        dummy, scale->mutable_data());
                status_to_exception(status);
                return X;
            } else {
                py::array_t<T, py::array::f_style> copy_X(shape, strides);
                memcpy(copy_X.mutable_data(), X.mutable_data(), sizeof(T) * X.size());
                status = da_standardize(axis_enum, m, n, copy_X.mutable_data(), m, dof, 0,
                                        dummy, scale->mutable_data());
                status_to_exception(status);
                return copy_X;
            }
        }
    } else {
        if (reverse) {
            throw std::invalid_argument(
                "Reverse standardization only works with supplied both shift and scale.");
        } else {
            if (inplace) {
                status = da_standardize(axis_enum, m, n, X.mutable_data(), m, dof, 0,
                                        dummy, dummy);
                status_to_exception(status);
                return X;
            } else {
                py::array_t<T, py::array::f_style> copy_X(shape, strides);
                memcpy(copy_X.mutable_data(), X.mutable_data(), sizeof(T) * X.size());
                status = da_standardize(axis_enum, m, n, copy_X.mutable_data(), m, dof, 0,
                                        dummy, dummy);
                status_to_exception(status);
                return copy_X;
            }
        }
    }
}

template <typename T>
py::array_t<T> py_da_covariance(py::array_t<T, py::array::f_style> X, da_int dof = 0) {
    da_status status;
    da_int m, n;

    // Guard for 1D arrays passed
    if (X.ndim() == 1) {
        n = X.shape()[0];
        m = 1;
    } else {
        m = X.shape()[0], n = X.shape()[1];
    }

    // Create the output covariance array as a numpy array
    size_t shape[2]{(size_t)n, (size_t)n};
    size_t strides[2]{sizeof(T), sizeof(T) * n};
    py::array_t<T, py::array::f_style> cov(shape, strides);

    status = da_covariance_matrix(m, n, X.data(), m, dof, cov.mutable_data(), n);

    status_to_exception(status);

    return cov;
}

template <typename T>
py::array_t<T> py_da_correlation(py::array_t<T, py::array::f_style> X) {
    da_status status;
    da_int m, n;

    // Guard for 1D arrays passed
    if (X.ndim() == 1) {
        n = X.shape()[0];
        m = 1;
    } else {
        m = X.shape()[0], n = X.shape()[1];
    }

    // Create the output correlation array as a numpy array
    size_t shape[2]{(size_t)n, (size_t)n};
    size_t strides[2]{sizeof(T), sizeof(T) * n};
    py::array_t<T, py::array::f_style> corr(shape, strides);

    status = da_correlation_matrix(m, n, X.data(), m, corr.mutable_data(), n);

    status_to_exception(status);

    return corr;
}

class pyda_handle {
  protected:
    da_handle handle = nullptr;

  public:
    void print_error_message() { da_handle_print_error_message(handle); };
    void exception_check(da_status status) {
        if (status == da_status_success) {
            return;
        }

        // If we got to here, there's an error to deal with
        char *message;
        da_severity severity;
        da_handle_get_error_message(handle, &message);
        da_handle_get_error_severity(handle, &severity);
        std::string mesg = message;
        if (severity == DA_ERROR) {
            PyErr_SetString(PyExc_RuntimeError, mesg.c_str());
            throw py::error_already_set();
        } else
            PyErr_WarnEx(PyExc_RuntimeWarning, mesg.c_str(), 1);

        free(message);
    }
};

class linmod : public pyda_handle {
    //da_handle handle = nullptr;
    da_precision precision = da_double;
    da_int n_samples, n_feat;
    bool intercept;

  public:
    linmod(std::string mod, std::optional<da_int> max_iter, bool intercept = false,
           std::string solver = "auto", std::string scaling = "auto",
           std::string prec = "double")
        : intercept(intercept) {
        da_status status;
        linmod_model mod_enum;
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
        if (max_iter.has_value()) {
            status =
                da_options_set_int(handle, "optim iteration limit", max_iter.value());
            exception_check(status);
        }
    }
    ~linmod() { da_handle_destroy(&handle); }

    template <typename T>
    void fit(py::array_t<T, py::array::f_style> X, py::array_t<T> y, T reg_lambda = 0.0,
             T reg_alpha = 0.0, T tol = 0.0001) {
        // floating point optional parameters are defined here since we cannot define those in the constructor (no template param)
        // TODO Should it be a separate function call like in C with the "define_features" function

        da_status status;
        n_samples = X.shape()[0];
        n_feat = X.shape()[1];
        status = da_linmod_define_features(handle, n_samples, n_feat, X.mutable_data(),
                                           y.mutable_data());
        exception_check(status); // throw an exception if status is not success

        // Set the real optional parameters
        status = da_options_set(handle, "lambda", reg_lambda);
        exception_check(status);
        status = da_options_set(handle, "alpha", reg_alpha);
        exception_check(status);
        status = da_options_set(handle, "optim convergence tol", tol);
        exception_check(status);

        if (precision == da_double)
            status = da_linmod_fit<double>(handle);
        else
            status = da_linmod_fit<float>(handle);

        exception_check(status);
    }

    template <typename T> py::array_t<T> predict(py::array_t<T, py::array::f_style> X) {

        da_status status;
        da_int n_samples = X.shape()[0], n_features = X.shape()[1];
        size_t shape[1]{(size_t)n_samples};
        size_t strides[1]{sizeof(T)};
        auto predictions = py::array_t<T>(shape, strides);
        status = da_linmod_evaluate_model(handle, n_samples, n_features, X.mutable_data(),
                                          predictions.mutable_data());
        exception_check(status);
        return predictions;
    }

    auto get_coef() {
        da_status status;
        da_int dim = intercept ? n_feat + 1 : n_feat;
        // define the output vector
        size_t shape[1]{(size_t)dim};
        if (precision == da_single) {
            size_t strides[1]{sizeof(float)};
            auto coef = py::array_t<float>(shape, strides);
            status =
                da_handle_get_result(handle, da_linmod_coef, &dim, coef.mutable_data());
            py::array ret = py::reinterpret_borrow<py::array>(coef);
            return ret;
        } else {
            size_t strides[1]{sizeof(double)};
            auto coef = py::array_t<double>(shape, strides);
            status =
                da_handle_get_result(handle, da_linmod_coef, &dim, coef.mutable_data());
            py::array ret = py::reinterpret_borrow<py::array>(coef);
            return ret;
        }
    }
};

class pca : public pyda_handle {
    //da_handle handle = nullptr;
    da_precision precision = da_double;

  public:
    pca(da_int n_components = 1, std::string bias = "unbiased",
        std::string method = "covariance", std::string solver = "gesdd",
        std::string prec = "double") {
        if (prec == "double")
            da_handle_init<double>(&handle, da_handle_pca);
        else if (prec == "single") {
            da_handle_init<float>(&handle, da_handle_pca);
            precision = da_single;
        }
        da_status status;
        status = da_options_set_int(handle, "n_components", n_components);
        exception_check(status);
        status = da_options_set_string(handle, "PCA method", method.c_str());
        exception_check(status);
        status = da_options_set_string(handle, "degrees of freedom", bias.c_str());
        exception_check(status);
        status = da_options_set_string(handle, "svd solver", solver.c_str());
        exception_check(status);
    }
    ~pca() { da_handle_destroy(&handle); }

    template <typename T> void fit(py::array_t<T, py::array::f_style> A) {
        da_status status;
        da_int n_samples = A.shape()[0], n_features = A.shape()[1], lda = A.shape()[0];
        status = da_pca_set_data(handle, n_samples, n_features, A.data(), lda);
        exception_check(status);
        status = da_pca_compute<T>(handle);
        exception_check(status);
    }

    template <typename T> py::array_t<T> transform(py::array_t<T, py::array::f_style> X) {
        da_status status;
        da_int m_samples = X.shape()[0], m_features = X.shape()[1], ldx = X.shape()[0];

        T result[3];
        da_int dim = 3;

        status = da_handle_get_result(handle, da_rinfo, &dim, result);
        exception_check(status);

        // define the output vector
        da_int n_components = (da_int)result[2];
        da_int ldx_transform = m_samples;
        size_t shape[2]{(size_t)m_samples, (size_t)n_components};
        size_t strides[2]{sizeof(T), sizeof(T) * m_samples};
        auto X_transform = py::array_t<T>(shape, strides);

        status = da_pca_transform(handle, m_samples, m_features, X.data(), ldx,
                                  X_transform.mutable_data(), ldx_transform);
        exception_check(status);
        return X_transform;
    }

    template <typename T>
    py::array_t<T> inverse_transform(py::array_t<T, py::array::f_style> Y) {
        da_status status;
        da_int k_samples = Y.shape()[0], k_features = Y.shape()[1], ldy = Y.shape()[0];

        T result[3];
        da_int dim = 3;

        status = da_handle_get_result(handle, da_rinfo, &dim, result);
        exception_check(status);

        // define the output vector
        da_int n_features = (da_int)result[1];
        da_int ldy_inv_transform = k_samples;
        size_t shape[2]{(size_t)k_samples, (size_t)n_features};
        size_t strides[2]{sizeof(T), sizeof(T) * k_samples};
        auto Y_inv_transform = py::array_t<T>(shape, strides);

        status =
            da_pca_inverse_transform(handle, k_samples, k_features, Y.data(), ldy,
                                     Y_inv_transform.mutable_data(), ldy_inv_transform);
        exception_check(status);
        return Y_inv_transform;
    }

    void get_rinfo(da_int *n_samples, da_int *n_features, da_int *n_components,
                   size_t *stride_size) {
        da_status status;

        da_int dim = 3;

        if (precision == da_single) {
            float rinfo[3];
            *stride_size = sizeof(float);
            status = da_handle_get_result(handle, da_rinfo, &dim, rinfo);
            *n_samples = (da_int)rinfo[0];
            *n_features = (da_int)rinfo[1];
            *n_components = (da_int)rinfo[2];
        } else {
            double rinfo[3];
            *stride_size = sizeof(double);
            status = da_handle_get_result(handle, da_rinfo, &dim, rinfo);
            *n_samples = (da_int)rinfo[0];
            *n_features = (da_int)rinfo[1];
            *n_components = (da_int)rinfo[2];
        }

        exception_check(status);
    }

    auto get_result(da_result result) {
        da_status status;

        da_int n_samples, n_features, n_components;
        da_int dim = 3, dim1, dim2;
        size_t stride_size;

        get_rinfo(&n_samples, &n_features, &n_components, &stride_size);

        switch (result) {
        case da_pca_principal_components:
            dim1 = n_components;
            dim2 = n_features;
            break;
        case da_pca_scores:
            dim1 = n_samples;
            dim2 = n_components;
            break;
        case da_pca_variance:
            dim1 = n_components;
            dim2 = 1;
            break;
        case da_pca_total_variance:
            dim1 = 1;
            dim2 = 1;
            break;
        case da_pca_u:
            dim1 = n_samples;
            dim2 = n_components;
            break;
        case da_pca_sigma:
            dim1 = n_components;
            dim2 = 1;
            break;
        case da_pca_vt:
            dim1 = n_components;
            dim2 = n_features;
            break;
        case da_pca_column_means:
            dim1 = n_features;
            dim2 = 1;
            break;
        case da_pca_column_sdevs:
            dim1 = n_features;
            dim2 = 1;
            break;
        case da_rinfo:
            dim1 = 3;
            dim2 = 1;
            break;
        }

        dim = dim1 * dim2;
        std::vector<size_t> shape, strides;
        shape.push_back(dim1);
        strides.push_back(stride_size);
        if (dim2 > 1) {
            shape.push_back(dim2);
            strides.push_back(stride_size * dim1);
        }

        if (precision == da_single) {

            // define the output vector
            auto res = py::array_t<float>(shape, strides);
            status = da_handle_get_result(handle, result, &dim, res.mutable_data());
            exception_check(status);
            py::array ret = py::reinterpret_borrow<py::array>(res);
            return ret;
        } else {

            // define the output vector
            auto res = py::array_t<double>(shape, strides);
            status = da_handle_get_result(handle, result, &dim, res.mutable_data());
            exception_check(status);
            py::array ret = py::reinterpret_borrow<py::array>(res);
            return ret;
        }
    }
    auto get_principal_components() { return get_result(da_pca_principal_components); }
    auto get_scores() { return get_result(da_pca_scores); }
    auto get_variance() { return get_result(da_pca_variance); }
    auto get_total_variance() { return get_result(da_pca_total_variance); }
    auto get_u() { return get_result(da_pca_u); }
    auto get_sigma() { return get_result(da_pca_sigma); }
    auto get_vt() { return get_result(da_pca_vt); }
    auto get_column_means() { return get_result(da_pca_column_means); }
    auto get_column_sdevs() { return get_result(da_pca_column_sdevs); }
    auto get_n_samples() {

        da_int n_samples, n_features, n_components;
        size_t stride_size;

        if (precision == da_single) {
            float inertia;
            get_rinfo(&n_samples, &n_features, &n_components, &stride_size);
        } else {
            double inertia;
            get_rinfo(&n_samples, &n_features, &n_components, &stride_size);
        }

        return n_samples;
    }
    auto get_n_components() {

        da_int n_samples, n_features, n_components;
        size_t stride_size;

        if (precision == da_single) {
            float inertia;
            get_rinfo(&n_samples, &n_features, &n_components, &stride_size);
        } else {
            double inertia;
            get_rinfo(&n_samples, &n_features, &n_components, &stride_size);
        }

        return n_components;
    }
    auto get_n_features() {

        da_int n_samples, n_features, n_components;
        size_t stride_size;

        if (precision == da_single) {
            float inertia;
            get_rinfo(&n_samples, &n_features, &n_components, &stride_size);
        } else {
            double inertia;
            get_rinfo(&n_samples, &n_features, &n_components, &stride_size);
        }

        return n_features;
    }
};

class kmeans : public pyda_handle {
    da_precision precision = da_double;

  public:
    kmeans(da_int n_clusters = 1, std::string initialization_method = "k-means++",
           da_int n_init = 10, da_int max_iter = 300, da_int seed = -1,
           std::string algorithm = "elkan", std::string prec = "double") {
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
    }
    ~kmeans() { da_handle_destroy(&handle); }

    template <typename T>
    void fit(py::array_t<T, py::array::f_style> A,
             std::optional<py::array_t<T, py::array::f_style>> C, T tol = 1.0e-4) {
        // floating point optional parameters are defined here since we cannot define those in the constructor (no template param)
        da_status status;
        status = da_options_set(handle, "convergence tolerance", tol);
        exception_check(status);
        da_int n_samples = A.shape()[0], n_features = A.shape()[1], lda = A.shape()[0];
        status = da_kmeans_set_data(handle, n_samples, n_features, A.data(), lda);
        exception_check(status);
        if (C.has_value()) {
            status = da_options_set_string(handle, "initialization method", "supplied");
            exception_check(status);
            da_int ldc = C->shape()[0];
            status = da_kmeans_set_init_centres(handle, C->data(), ldc);
            exception_check(status);
        }
        status = da_kmeans_compute<T>(handle);
        exception_check(status);
    }

    template <typename T> py::array_t<T> transform(py::array_t<T, py::array::f_style> X) {
        da_status status;
        da_int m_samples = X.shape()[0], m_features = X.shape()[1], ldx = X.shape()[0];

        T result[5];
        da_int dim = 5;

        status = da_handle_get_result(handle, da_rinfo, &dim, result);
        exception_check(status);

        // define the output vector
        da_int n_clusters = (da_int)result[2];
        da_int ldx_transform = m_samples;
        size_t shape[2]{(size_t)m_samples, (size_t)n_clusters};
        size_t strides[2]{sizeof(T), sizeof(T) * m_samples};
        auto X_transform = py::array_t<T>(shape, strides);

        status = da_kmeans_transform(handle, m_samples, m_features, X.data(), ldx,
                                     X_transform.mutable_data(), ldx_transform);
        exception_check(status);
        return X_transform;
    }

    template <typename T>
    py::array_t<da_int> predict(py::array_t<T, py::array::f_style> Y) {
        da_status status;
        da_int k_samples = Y.shape()[0], k_features = Y.shape()[1], ldy = Y.shape()[0];

        T result[5];
        da_int dim = 5;

        status = da_handle_get_result(handle, da_rinfo, &dim, result);
        exception_check(status);

        // define the output vector
        da_int n_clusters = (da_int)result[2];
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
        strides.push_back(stride_size);
        if (dim2 > 1) {
            shape.push_back(dim2);
            strides.push_back(stride_size * dim1);
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

        da_status status;

        size_t stride_size;
        da_int n_samples, n_features, n_clusters, n_iter;
        da_int dim, dim1, dim2;

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

PYBIND11_MODULE(_aoclda, m) {
    m.doc() = "Python wrappers for the AOCL-DA library";

    /**********************************/
    /*         Basic statistics       */
    /**********************************/
    auto m_stats = m.def_submodule("basic_stats", "Basic statistics.");
    m_stats.def("pybind_mean", &py_da_mean<float>, "X"_a, "axis"_a = "col");
    m_stats.def("pybind_mean", &py_da_mean<double>, "X"_a, "axis"_a = "col");
    m_stats.def("pybind_harmonic_mean", &py_da_harmonic_mean<float>, "X"_a,
                "axis"_a = "col");
    m_stats.def("pybind_harmonic_mean", &py_da_harmonic_mean<double>, "X"_a,
                "axis"_a = "col");
    m_stats.def("pybind_geometric_mean", &py_da_geometric_mean<float>, "X"_a,
                "axis"_a = "col");
    m_stats.def("pybind_geometric_mean", &py_da_geometric_mean<double>, "X"_a,
                "axis"_a = "col");
    m_stats.def("pybind_variance", &py_da_variance<float>, "X"_a, "dof"_a = 0,
                "axis"_a = "col");
    m_stats.def("pybind_variance", &py_da_variance<double>, "X"_a, "dof"_a = 0,
                "axis"_a = "col");
    m_stats.def("pybind_skewness", &py_da_skewness<float>, "X"_a, "axis"_a = "col");
    m_stats.def("pybind_skewness", &py_da_skewness<double>, "X"_a, "axis"_a = "col");
    m_stats.def("pybind_kurtosis", &py_da_kurtosis<float>, "X"_a, "axis"_a = "col");
    m_stats.def("pybind_kurtosis", &py_da_kurtosis<double>, "X"_a, "axis"_a = "col");
    m_stats.def("pybind_moment", &py_da_moment<float>, "X"_a, "k"_a,
                "mean"_a = py::none(), "axis"_a = "col");
    m_stats.def("pybind_moment", &py_da_moment<double>, "X"_a, "k"_a,
                "mean"_a = py::none(), "axis"_a = "col");
    m_stats.def("pybind_quantile", &py_da_quantile<float>, "X"_a, "q"_a,
                "method"_a = "linear", "axis"_a = "col");
    m_stats.def("pybind_quantile", &py_da_quantile<double>, "X"_a, "q"_a,
                "method"_a = "linear", "axis"_a = "col");
    m_stats.def("pybind_five_point_summary", &py_da_five_point_summary<float>, "X"_a,
                "axis"_a = "col");
    m_stats.def("pybind_five_point_summary", &py_da_five_point_summary<double>, "X"_a,
                "axis"_a = "col");
    m_stats.def("pybind_standardize", &py_da_standardize<float>, "X"_a,
                "shift"_a = py::none(), "scale"_a = py::none(), "dof"_a = 0,
                "reverse"_a = false, "inplace"_a = false, "axis"_a = "col");
    m_stats.def("pybind_standardize", &py_da_standardize<double>, "X"_a,
                "shift"_a = py::none(), "scale"_a = py::none(), "dof"_a = 0,
                "reverse"_a = false, "inplace"_a = false, "axis"_a = "col");
    m_stats.def("pybind_covariance_matrix", &py_da_covariance<float>, "X"_a, "dof"_a = 0);
    m_stats.def("pybind_covariance_matrix", &py_da_covariance<double>, "X"_a,
                "dof"_a = 0);
    m_stats.def("pybind_correlation_matrix", &py_da_correlation<float>);
    m_stats.def("pybind_correlation_matrix", &py_da_correlation<double>);

    /**********************************/
    /*          Main handle           */
    /**********************************/
    py::class_<pyda_handle>(m, "handle")
        .def(py::init<>())
        .def("print_error_message", &pyda_handle::print_error_message);

    /**********************************/
    /*         Linear Models          */
    /**********************************/
    auto m_linmod = m.def_submodule("linear_model", "Linear models.");
    py::class_<linmod, pyda_handle>(m_linmod, "pybind_linmod")
        .def(py::init<std::string, std::optional<da_int>, bool, std::string, std::string,
                      std::string &>(),
             py::arg("mod"), py::arg("max_iter") = py::none(),
             py::arg("intercept") = false, py::arg("solver") = "auto",
             py::arg("scaling") = "auto", py::arg("precision") = "double")
        .def("pybind_fit", &linmod::fit<float>, "Computes the model", "X"_a, "y"_a,
             py::arg("reg_lambda") = (float)0.0, py::arg("reg_alpha") = (float)0.0,
             py::arg("tol") = (float)0.0001)
        .def("pybind_fit", &linmod::fit<double>, "Computes the model", "X"_a, "y"_a,
             py::arg("reg_lambda") = (double)0.0, py::arg("reg_alpha") = (double)0.0,
             py::arg("tol") = (double)0.0001)
        .def("pybind_predict", &linmod::predict<double>, "Evaluate the model on X", "X"_a)
        .def("pybind_predict", &linmod::predict<float>, "Evaluate the model on X", "X"_a)
        .def("get_coef", &linmod::get_coef);

    /**********************************/
    /*  Principal component analysis  */
    /**********************************/
    auto m_factorization = m.def_submodule("factorization", "Matrix factorizations.");
    py::class_<pca, pyda_handle>(m_factorization, "pybind_PCA")
        .def(py::init<da_int, std::string, std::string, std::string, std::string &>(),
             py::arg("n_components") = 1, py::arg("bias") = "unbiased",
             py::arg("method") = "covariance", py::arg("solver") = "gesdd",
             py::arg("precision") = "double")
        .def("pybind_fit", &pca::fit<float>, "Fit the principal component analysis",
             "A"_a)
        .def("pybind_fit", &pca::fit<double>, "Fit the principal component analysis",
             "A"_a)
        .def("pybind_transform", &pca::transform<float>, "Transform using computed PCA",
             "X"_a)
        .def("pybind_transform", &pca::transform<double>, "Transform using computed PCA",
             "X"_a)
        .def("pybind_inverse_transform", &pca::inverse_transform<float>,
             "Inverse transform using computed PCA", "Y"_a)
        .def("pybind_inverse_transform", &pca::inverse_transform<double>,
             "Inverse transform using computed PCA", "Y"_a)
        .def("get_principal_components", &pca::get_principal_components)
        .def("get_scores", &pca::get_scores)
        .def("get_variance", &pca::get_variance)
        .def("get_total_variance", &pca::get_total_variance)
        .def("get_u", &pca::get_u)
        .def("get_sigma", &pca::get_sigma)
        .def("get_vt", &pca::get_vt)
        .def("get_column_means", &pca::get_column_means)
        .def("get_column_sdevs", &pca::get_column_sdevs)
        .def("get_n_samples", &pca::get_n_samples)
        .def("get_n_features", &pca::get_n_features)
        .def("get_n_components", &pca::get_n_components);
    /**********************************/
    /*       k-means clustering       */
    /**********************************/
    auto m_clustering = m.def_submodule("clustering", "Clustering algorithms.");
    py::class_<kmeans, pyda_handle>(m_clustering, "pybind_kmeans")
        .def(py::init<da_int, std::string, da_int, da_int, da_int, std::string,
                      std::string &>(),
             py::arg("n_clusters") = 1, py::arg("initialization_method") = "k-means++",
             py::arg("n_init") = 10, py::arg("max_iter") = 300, py::arg("seed") = -1,
             py::arg("algorithm") = "elkan", py::arg("precision") = "double")
        .def("pybind_fit", &kmeans::fit<float>, "Fit the k-means clusters", "A"_a,
             "C"_a = py::none(), py::arg("convergence_tolerance") = (float)1.0e-4)
        .def("pybind_fit", &kmeans::fit<double>, "Fit the k-means clusters", "A"_a,
             "C"_a = py::none(), py::arg("convergence_tolerance") = (double)1.0e-4)
        .def("pybind_transform", &kmeans::transform<float>,
             "Transform using computed k-means clusters", "X"_a)
        .def("pybind_transform", &kmeans::transform<double>,
             "Transform using computed k-means clusters", "X"_a)
        .def("pybind_predict", &kmeans::predict<float>,
             "Predict labels using computed k-means clusters", "Y"_a)
        .def("pybind_predict", &kmeans::predict<double>,
             "Predict labels using computed k-means clusters", "Y"_a)
        .def("get_cluster_centres", &kmeans::get_cluster_centres)
        .def("get_labels", &kmeans::get_labels)
        .def("get_inertia", &kmeans::get_inertia)
        .def("get_n_samples", &kmeans::get_n_samples)
        .def("get_n_features", &kmeans::get_n_features)
        .def("get_n_clusters", &kmeans::get_n_clusters)
        .def("get_n_iter", &kmeans::get_n_iter);
}