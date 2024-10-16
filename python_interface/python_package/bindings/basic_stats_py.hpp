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

#ifndef BASIC_STATS_PY_HPP
#define BASIC_STATS_PY_HPP

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

/*
 * Determine the size and ordering of a numpy array and set the axis enum
 */
template <typename T>
void get_size_and_axis(std::string axis, da_order &order, da_axis &axis_enum,
                       py::array_t<T> &X, size_t &size, da_int &m, da_int &n,
                       da_int &ldx) {
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

    if (X.ndim() == 1) {
        // If user provided 1D array then calculating for example mean over all elements
        // would mean calculation over that one row of data.
        if (axis_enum == da_axis_all) {
            axis_enum = da_axis_row;
        }
    }

    get_size(order, X, m, n, ldx);

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
py::array_t<T> py_da_mean(py::array_t<T> X, std::string axis = "col") {
    da_status status;
    da_int m, n, ldx;
    da_order order;
    size_t mean_sz;
    da_axis axis_enum;

    get_size_and_axis(axis, order, axis_enum, X, mean_sz, m, n, ldx);

    // Create the output mean array as a numpy array
    size_t shape[1]{mean_sz};
    size_t strides[1]{sizeof(T)};
    auto mean = py::array_t<T>(shape, strides);

    status = da_mean(order, axis_enum, m, n, X.data(), ldx, mean.mutable_data());

    status_to_exception(status);

    return mean;
}

template <typename T>
py::array_t<T> py_da_harmonic_mean(py::array_t<T> X, std::string axis = "col") {
    da_status status;
    da_int m, n, ldx;
    da_order order;
    size_t harmonic_mean_sz;
    da_axis axis_enum;

    get_size_and_axis(axis, order, axis_enum, X, harmonic_mean_sz, m, n, ldx);

    // Create the output mean array as a numpy array
    size_t shape[1]{harmonic_mean_sz};
    size_t strides[1]{sizeof(T)};
    auto harmonic_mean = py::array_t<T>(shape, strides);

    status = da_harmonic_mean(order, axis_enum, m, n, X.data(), ldx,
                              harmonic_mean.mutable_data());

    status_to_exception(status);

    return harmonic_mean;
}

template <typename T>
py::array_t<T> py_da_geometric_mean(py::array_t<T> X, std::string axis = "col") {
    da_status status;
    da_int m, n, ldx;
    da_order order;
    size_t geometric_mean_sz;
    da_axis axis_enum;

    get_size_and_axis(axis, order, axis_enum, X, geometric_mean_sz, m, n, ldx);

    // Create the output mean array as a numpy array
    size_t shape[1]{geometric_mean_sz};
    size_t strides[1]{sizeof(T)};
    auto geometric_mean = py::array_t<T>(shape, strides);

    status = da_geometric_mean(order, axis_enum, m, n, X.data(), ldx,
                               geometric_mean.mutable_data());

    status_to_exception(status);

    return geometric_mean;
}

template <typename T>
py::array_t<T> py_da_variance(py::array_t<T> X, da_int dof = 0,
                              std::string axis = "col") {
    da_status status;
    da_int m, n, ldx;
    da_order order;
    size_t variance_sz;
    da_axis axis_enum;

    get_size_and_axis(axis, order, axis_enum, X, variance_sz, m, n, ldx);

    // Create the output variance array as a numpy array
    size_t shape[1]{variance_sz};
    size_t strides[1]{sizeof(T)};
    auto mean = py::array_t<T>(shape, strides);
    auto variance = py::array_t<T>(shape, strides);

    status = da_variance(order, axis_enum, m, n, X.data(), ldx, dof, mean.mutable_data(),
                         variance.mutable_data());

    status_to_exception(status);

    return variance;
}

template <typename T>
py::array_t<T> py_da_skewness(py::array_t<T> X, std::string axis = "col") {
    da_status status;
    da_int m, n, ldx;
    da_order order;
    size_t skewness_sz;
    da_axis axis_enum;

    get_size_and_axis(axis, order, axis_enum, X, skewness_sz, m, n, ldx);

    // Create the output skewness array as a numpy array, as well as other arrays to store auxiliary output
    size_t shape[1]{skewness_sz};
    size_t strides[1]{sizeof(T)};
    auto mean = py::array_t<T>(shape, strides);
    auto variance = py::array_t<T>(shape, strides);
    auto skewness = py::array_t<T>(shape, strides);

    status = da_skewness(order, axis_enum, m, n, X.data(), ldx, mean.mutable_data(),
                         variance.mutable_data(), skewness.mutable_data());

    status_to_exception(status);

    return skewness;
}

template <typename T>
py::array_t<T> py_da_kurtosis(py::array_t<T> X, std::string axis = "col") {
    da_status status;
    da_int m, n, ldx;
    da_order order;
    size_t kurtosis_sz;
    da_axis axis_enum;

    get_size_and_axis(axis, order, axis_enum, X, kurtosis_sz, m, n, ldx);

    // Create the output kurtosis array as a numpy array, as well as other arrays to store auxiliary output
    size_t shape[1]{kurtosis_sz};
    size_t strides[1]{sizeof(T)};
    auto mean = py::array_t<T>(shape, strides);
    auto variance = py::array_t<T>(shape, strides);
    auto kurtosis = py::array_t<T>(shape, strides);

    status = da_kurtosis(order, axis_enum, m, n, X.data(), ldx, mean.mutable_data(),
                         variance.mutable_data(), kurtosis.mutable_data());

    status_to_exception(status);

    return kurtosis;
}

template <typename T>
py::array_t<T> py_da_moment(py::array_t<T> X, da_int k,
                            std::optional<py::array_t<T>> mean,
                            std::string axis = "col") {
    da_status status;
    da_int m, n, ldx;
    da_order order;
    size_t moment_sz;
    da_axis axis_enum;

    get_size_and_axis(axis, order, axis_enum, X, moment_sz, m, n, ldx);

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
        status = da_moment(order, axis_enum, m, n, X.data(), ldx, k, 1,
                           mean->mutable_data(), moment.mutable_data());
    } else {
        auto mean_aux = py::array_t<T>(shape, strides);
        status = da_moment(order, axis_enum, m, n, X.data(), ldx, k, 0,
                           mean_aux.mutable_data(), moment.mutable_data());
    }

    status_to_exception(status);

    return moment;
}

template <typename T>
py::array_t<T> py_da_quantile(py::array_t<T> X, T q, std::string method = "linear",
                              std::string axis = "col") {
    da_status status;
    da_int m, n, ldx;
    da_order order;
    size_t quantile_sz;
    da_axis axis_enum;

    get_size_and_axis(axis, order, axis_enum, X, quantile_sz, m, n, ldx);

    // Create the output quantile array as a numpy array
    size_t shape[1]{quantile_sz};
    size_t strides[1]{sizeof(T)};
    auto quantiles = py::array_t<T>(shape, strides);

    if (method == "inverted_cdf") {
        status = da_quantile(order, axis_enum, m, n, X.data(), ldx, q,
                             quantiles.mutable_data(), da_quantile_type_1);
    } else if (method == "averaged_inverted_cdf") {
        status = da_quantile(order, axis_enum, m, n, X.data(), ldx, q,
                             quantiles.mutable_data(), da_quantile_type_2);
    } else if (method == "closest_observation") {
        status = da_quantile(order, axis_enum, m, n, X.data(), ldx, q,
                             quantiles.mutable_data(), da_quantile_type_3);
    } else if (method == "interpolated_inverted_cdf") {
        status = da_quantile(order, axis_enum, m, n, X.data(), ldx, q,
                             quantiles.mutable_data(), da_quantile_type_4);
    } else if (method == "hazen") {
        status = da_quantile(order, axis_enum, m, n, X.data(), ldx, q,
                             quantiles.mutable_data(), da_quantile_type_5);
    } else if (method == "weibull") {
        status = da_quantile(order, axis_enum, m, n, X.data(), ldx, q,
                             quantiles.mutable_data(), da_quantile_type_6);
    } else if (method == "linear") {
        status = da_quantile(order, axis_enum, m, n, X.data(), ldx, q,
                             quantiles.mutable_data(), da_quantile_type_7);
    } else if (method == "median_unbiased") {
        status = da_quantile(order, axis_enum, m, n, X.data(), ldx, q,
                             quantiles.mutable_data(), da_quantile_type_8);
    } else if (method == "normal_unbiased") {
        status = da_quantile(order, axis_enum, m, n, X.data(), ldx, q,
                             quantiles.mutable_data(), da_quantile_type_9);
    } else {
        throw std::invalid_argument("Provided method does not exist.");
    }

    status_to_exception(status);

    return quantiles;
}

template <typename T>
std::tuple<py::array_t<T>, py::array_t<T>, py::array_t<T>, py::array_t<T>, py::array_t<T>>
py_da_five_point_summary(py::array_t<T> X, std::string axis = "col") {
    da_status status;
    da_int m, n, ldx;
    da_order order;
    size_t fps_sz;
    da_axis axis_enum;

    get_size_and_axis(axis, order, axis_enum, X, fps_sz, m, n, ldx);

    // Create the output arrays to store five point summary as a numpy array
    size_t shape[1]{fps_sz};
    size_t strides[1]{sizeof(T)};
    auto min = py::array_t<T>(shape, strides);
    auto lq = py::array_t<T>(shape, strides);
    auto med = py::array_t<T>(shape, strides);
    auto uq = py::array_t<T>(shape, strides);
    auto max = py::array_t<T>(shape, strides);

    status = da_five_point_summary(
        order, axis_enum, m, n, X.data(), ldx, min.mutable_data(), lq.mutable_data(),
        med.mutable_data(), uq.mutable_data(), max.mutable_data());

    status_to_exception(status);

    return std::make_tuple(min, lq, med, uq, max);
}

template <typename T>
py::array_t<T> py_da_standardize(py::array_t<T> X, std::optional<py::array_t<T>> shift,
                                 std::optional<py::array_t<T>> scale, da_int dof = 0,
                                 bool reverse = false, bool inplace = false,
                                 std::string axis = "col") {
    da_status status;
    da_int m, n, ldx;
    da_order order;
    size_t standardize_sz;
    da_axis axis_enum;

    get_size_and_axis(axis, order, axis_enum, X, standardize_sz, m, n, ldx);

    // Create parameters for potential copy_X of original numpy array
    size_t shape[2]{(size_t)m, (size_t)n};
    size_t strides[2];
    if (order == column_major) {
        strides[0] = sizeof(T);
        strides[1] = sizeof(T) * m;
    } else {
        strides[0] = sizeof(T) * n;
        strides[1] = sizeof(T);
    }
    T *dummy = nullptr;

    if (shift.has_value() && scale.has_value()) {
        if ((size_t)shift->shape()[0] != standardize_sz ||
            (size_t)scale->shape()[0] != standardize_sz) {
            throw std::length_error(
                "The size of shift or scale array does not match the expected size.");
        }
        if (reverse) {
            if (inplace) {
                status =
                    da_standardize(order, axis_enum, m, n, X.mutable_data(), ldx, dof, 1,
                                   shift->mutable_data(), scale->mutable_data());
                status_to_exception(status);
                return X;
            } else {
                py::array_t<T> copy_X(shape, strides);
                memcpy(copy_X.mutable_data(), X.mutable_data(), sizeof(T) * X.size());
                status =
                    da_standardize(order, axis_enum, m, n, copy_X.mutable_data(), ldx,
                                   dof, 1, shift->mutable_data(), scale->mutable_data());
                status_to_exception(status);
                return copy_X;
            }
        } else {
            if (inplace) {
                status =
                    da_standardize(order, axis_enum, m, n, X.mutable_data(), ldx, dof, 0,
                                   shift->mutable_data(), scale->mutable_data());
                status_to_exception(status);
                return X;
            } else {
                py::array_t<T> copy_X(shape, strides);
                memcpy(copy_X.mutable_data(), X.mutable_data(), sizeof(T) * X.size());
                status =
                    da_standardize(order, axis_enum, m, n, copy_X.mutable_data(), ldx,
                                   dof, 0, shift->mutable_data(), scale->mutable_data());
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
                status = da_standardize(order, axis_enum, m, n, X.mutable_data(), ldx,
                                        dof, 1, shift->mutable_data(), dummy);
                status_to_exception(status);
                return X;
            } else {
                py::array_t<T> copy_X(shape, strides);
                memcpy(copy_X.mutable_data(), X.mutable_data(), sizeof(T) * X.size());
                status = da_standardize(order, axis_enum, m, n, copy_X.mutable_data(),
                                        ldx, dof, 1, shift->mutable_data(), dummy);
                status_to_exception(status);
                return copy_X;
            }
        } else {
            if (inplace) {
                status = da_standardize(order, axis_enum, m, n, X.mutable_data(), ldx,
                                        dof, 0, shift->mutable_data(), dummy);
                status_to_exception(status);
                return X;
            } else {
                py::array_t<T> copy_X(shape, strides);
                memcpy(copy_X.mutable_data(), X.mutable_data(), sizeof(T) * X.size());
                status = da_standardize(order, axis_enum, m, n, copy_X.mutable_data(),
                                        ldx, dof, 0, shift->mutable_data(), dummy);
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
                status = da_standardize(order, axis_enum, m, n, X.mutable_data(), m, dof,
                                        1, dummy, scale->mutable_data());
                status_to_exception(status);
                return X;
            } else {
                py::array_t<T> copy_X(shape, strides);
                memcpy(copy_X.mutable_data(), X.mutable_data(), sizeof(T) * X.size());
                status = da_standardize(order, axis_enum, m, n, copy_X.mutable_data(),
                                        ldx, dof, 1, dummy, scale->mutable_data());
                status_to_exception(status);
                return copy_X;
            }
        } else {
            if (inplace) {
                status = da_standardize(order, axis_enum, m, n, X.mutable_data(), ldx,
                                        dof, 0, dummy, scale->mutable_data());
                status_to_exception(status);
                return X;
            } else {
                py::array_t<T> copy_X(shape, strides);
                memcpy(copy_X.mutable_data(), X.mutable_data(), sizeof(T) * X.size());
                status = da_standardize(order, axis_enum, m, n, copy_X.mutable_data(),
                                        ldx, dof, 0, dummy, scale->mutable_data());
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
                status = da_standardize(order, axis_enum, m, n, X.mutable_data(), ldx,
                                        dof, 0, dummy, dummy);
                status_to_exception(status);
                return X;
            } else {
                py::array_t<T> copy_X(shape, strides);
                memcpy(copy_X.mutable_data(), X.mutable_data(), sizeof(T) * X.size());
                status = da_standardize(order, axis_enum, m, n, copy_X.mutable_data(),
                                        ldx, dof, 0, dummy, dummy);
                status_to_exception(status);
                return copy_X;
            }
        }
    }
}

template <typename T> py::array_t<T> py_da_covariance(py::array_t<T> X, da_int dof = 0) {
    da_status status;
    da_int m, n, ldx;
    da_order order;

    get_size(order, X, m, n, ldx);

    // Create the output covariance array as a numpy array
    size_t shape[2]{(size_t)n, (size_t)n};
    size_t strides[2];
    if (order == column_major) {
        strides[0] = sizeof(T);
        strides[1] = sizeof(T) * n;
    } else {
        strides[0] = sizeof(T) * n;
        strides[1] = sizeof(T);
    }

    py::array_t<T> cov(shape, strides);

    status = da_covariance_matrix(order, m, n, X.data(), ldx, dof, cov.mutable_data(), n);

    status_to_exception(status);

    return cov;
}

template <typename T> py::array_t<T> py_da_correlation(py::array_t<T> X) {
    da_status status;
    da_int m, n, ldx;
    da_order order;

    get_size(order, X, m, n, ldx);

    // Create the output correlation array as a numpy array
    size_t shape[2]{(size_t)n, (size_t)n};
    size_t strides[2];
    if (order == column_major) {
        strides[0] = sizeof(T);
        strides[1] = sizeof(T) * n;
    } else {
        strides[0] = sizeof(T) * n;
        strides[1] = sizeof(T);
    }

    py::array_t<T> corr(shape, strides);

    status = da_correlation_matrix(order, m, n, X.data(), ldx, corr.mutable_data(), n);

    status_to_exception(status);

    return corr;
}

#endif
