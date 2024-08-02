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

#ifndef FACTORIZATION_PY_HPP
#define FACTORIZATION_PY_HPP

#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include "utilities_py.hpp"
#include <iostream>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

class pca : public pyda_handle {

  public:
    pca(da_int n_components = 1, std::string bias = "unbiased",
        std::string method = "covariance", std::string solver = "gesdd",
        bool store_U = false, std::string prec = "double", bool check_data = false) {
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
        if (store_U == true) {
            status = da_options_set_int(handle, "store U", 1);
            exception_check(status);
        }
        if (check_data == true) {
            std::string yes_str = "yes";
            status = da_options_set(handle, "check data", yes_str.data());
            exception_check(status);
        }
    }
    ~pca() { da_handle_destroy(&handle); }

    template <typename T> void fit(py::array_t<T> A) {
        da_status status;
        da_int n_samples, n_features, lda;

        get_numpy_array_properties(A, n_samples, n_features, lda);

        if (order == c_contiguous) {
            status = da_options_set(handle, "storage order", "row-major");
        } else {
            status = da_options_set(handle, "storage order", "column-major");
        }
        exception_check(status);

        status = da_pca_set_data(handle, n_samples, n_features, A.data(), lda);
        exception_check(status);
        status = da_pca_compute<T>(handle);
        exception_check(status);
    }

    template <typename T> py::array_t<T> transform(py::array_t<T> X) {
        da_status status;
        da_int m_samples, m_features, ldx;

        get_numpy_array_properties(X, m_samples, m_features, ldx);

        T result[3];
        da_int dim = 3;

        status = da_handle_get_result(handle, da_rinfo, &dim, result);
        exception_check(status);

        // Define the output vector
        da_int n_components = (da_int)result[2], ldx_transform;
        size_t shape[2]{(size_t)m_samples, (size_t)n_components};
        size_t strides[2];
        if (order == c_contiguous) {
            ldx_transform = n_components;
            strides[0] = sizeof(T) * n_components;
            strides[1] = sizeof(T);
        } else {
            ldx_transform = m_samples;
            strides[0] = sizeof(T);
            strides[1] = sizeof(T) * m_samples;
        }
        auto X_transform = py::array_t<T>(shape, strides);

        status = da_pca_transform(handle, m_samples, m_features, X.data(), ldx,
                                  X_transform.mutable_data(), ldx_transform);
        exception_check(status);
        return X_transform;
    }

    template <typename T> py::array_t<T> inverse_transform(py::array_t<T> Y) {
        da_status status;
        da_int k_samples, k_features, ldy;

        get_numpy_array_properties(Y, k_samples, k_features, ldy);

        T result[3];
        da_int dim = 3;

        status = da_handle_get_result(handle, da_rinfo, &dim, result);
        exception_check(status);

        // Define the output vector
        da_int n_features = (da_int)result[1], ldy_inv_transform;
        size_t shape[2]{(size_t)k_samples, (size_t)n_features};
        size_t strides[2];
        if (order == c_contiguous) {
            ldy_inv_transform = n_features;
            strides[0] = sizeof(T) * n_features;
            strides[1] = sizeof(T);
        } else {
            ldy_inv_transform = k_samples;
            strides[0] = sizeof(T);
            strides[1] = sizeof(T) * k_samples;
        }
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
        da_status status = da_status_success;

        da_int n_samples, n_features, n_components;
        da_int dim = 3, dim1 = 0, dim2 = 0;
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
        default:
            status = da_status_invalid_input;
            break;
        }
        exception_check(status, "Unexpected result input");

        dim = dim1 * dim2;
        std::vector<size_t> shape, strides;

        shape.push_back(dim1);
        if (dim2 > 1)
            shape.push_back(dim2);

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
            get_rinfo(&n_samples, &n_features, &n_components, &stride_size);
        } else {
            get_rinfo(&n_samples, &n_features, &n_components, &stride_size);
        }

        return n_samples;
    }
    auto get_n_components() {

        da_int n_samples, n_features, n_components;
        size_t stride_size;

        if (precision == da_single) {
            get_rinfo(&n_samples, &n_features, &n_components, &stride_size);
        } else {
            get_rinfo(&n_samples, &n_features, &n_components, &stride_size);
        }

        return n_components;
    }
    auto get_n_features() {

        da_int n_samples, n_features, n_components;
        size_t stride_size;

        if (precision == da_single) {
            get_rinfo(&n_samples, &n_features, &n_components, &stride_size);
        } else {
            get_rinfo(&n_samples, &n_features, &n_components, &stride_size);
        }

        return n_features;
    }
};

#endif