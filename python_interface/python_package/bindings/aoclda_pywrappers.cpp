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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
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
        throw std::invalid_argument(
            "One of the options passed to the function had an invalid value.");
    default:
        std::cout
            << "An internal error occurred. This could be a memory corruption issue."
            << std::endl;
        throw std::exception();
    }
}

template <typename T>
py::array_t<T> py_da_mean(da_axis axis, py::array_t<T, py::array::f_style> X) {
    da_status status;
    da_int m = X.shape()[0], n = X.shape()[1];

    size_t mean_sz;
    switch (axis) {
    case (da_axis_all):
        mean_sz = 1;
        break;

    case (da_axis_col):
        mean_sz = n;
        break;

    case (da_axis_row):
        mean_sz = m;
        break;
    }

    // Create the output mean array as a numpy array
    size_t shape[1]{mean_sz};
    size_t strides[1]{sizeof(T)};
    auto mean = py::array_t<T>(shape, strides);

    status = da_mean(axis, m, n, X.data(), m, mean.mutable_data());

    status_to_exception(status);

    return mean;
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
        if (severity == DA_ERROR) {
            PyErr_SetString(PyExc_RuntimeError, message);
            throw py::error_already_set();
        }
        else
            PyErr_WarnEx(PyExc_RuntimeWarning, message, 1);

        free(message);
    }
};

class linmod : public pyda_handle {
    //da_handle handle = nullptr;
    da_precision precision = da_double;

  public:
    linmod(linmod_model mod, bool intercept = false, da_precision prec = da_double) {
        da_status status;
        if (prec == da_double) {
            da_handle_init<double>(&handle, da_handle_linmod);
            status = da_linmod_select_model<double>(handle, mod);
        } else {
            da_handle_init<float>(&handle, da_handle_linmod);
            status = da_linmod_select_model<float>(handle, mod);
            precision = da_single;
        }
        exception_check(status);
        // Set optional parameters
        if (intercept)
            da_options_set_int(handle, "linmod intercept", 1);
    }
    ~linmod() { da_handle_destroy(&handle); }

    template <typename T>
    void fit(py::array_t<T, py::array::f_style> X, py::array_t<T> y, T reg_lambda = 0.0,
             T reg_alpha = 0.0) {
        // floating point optional parameters are defined here since we cannot define those in the constructor (no template param)
        // TODO Should it be a separate function call like in C with the "define_features" function

        da_status status;
        da_int n_samples = X.shape()[0], n_feat = X.shape()[1];
        status = da_linmod_define_features(handle, n_samples, n_feat, X.mutable_data(),
                                           y.mutable_data());
        exception_check(status); // throw an exception if status is not success

        // Set the real optional parameters
        if (precision == da_double) {
            status = da_options_set_real_d(handle, "linmod lambda", reg_lambda);
            std::cout << "STATUS " << status << std::endl;
            exception_check(status);
            status = da_options_set_real_d(handle, "linmod alpha", reg_alpha);
            std::cout << "STATUS " << status << std::endl;
            exception_check(status);
        } else {
            status = da_options_set_real_d(handle, "linmod lambda", reg_lambda);
            exception_check(status);
            status = da_options_set_real_d(handle, "linmod alpha", reg_alpha);
            exception_check(status);
        }

        if (precision == da_double)
            status = da_linmod_fit<double>(handle);
        else
            status = da_linmod_fit<float>(handle);

        exception_check(status);
    }

    auto get_coef() {
        da_status status;
        da_int dim = -1;
        if (precision == da_single) {
            float result_s = 1;
            // First call to get dim right
            status = da_handle_get_result(handle, da_linmod_coeff, &dim, &result_s);
            if (status != da_status_invalid_array_dimension)
                status_to_exception(status);

            // define the output vector
            size_t shape[1]{(size_t)dim};
            size_t strides[1]{sizeof(float)};
            auto coef = py::array_t<float>(shape, strides);
            status =
                da_handle_get_result(handle, da_linmod_coeff, &dim, coef.mutable_data());
            exception_check(status);
            py::array ret = py::reinterpret_borrow<py::array>(coef);
            return ret;
        } else {
            double result_d = 1;
            // First call to get dim right
            status = da_handle_get_result(handle, da_linmod_coeff, &dim, &result_d);
            if (status != da_status_invalid_array_dimension)
                exception_check(status);

            // define the output vector
            size_t shape[1]{(size_t)dim};
            size_t strides[1]{sizeof(double)};
            auto coef = py::array_t<double>(shape, strides);
            status =
                da_handle_get_result(handle, da_linmod_coeff, &dim, coef.mutable_data());
            exception_check(status);
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
        da_precision prec = da_double) {
        if (prec == da_double)
            da_handle_init<double>(&handle, da_handle_pca);
        else {
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

    auto get_result(da_result result) {
        da_status status;

        da_int n_samples, n_features, n_components;
        da_int dim = 3, dim1, dim2;
        size_t stride_size;

        if (precision == da_single) {
            float rinfo[3];
            stride_size = sizeof(float);
            status = da_handle_get_result(handle, da_rinfo, &dim, rinfo);
            n_samples = (da_int)rinfo[0];
            n_features = (da_int)rinfo[1];
            n_components = (da_int)rinfo[2];
        } else {
            double rinfo[3];
            stride_size = sizeof(double);
            status = da_handle_get_result(handle, da_rinfo, &dim, rinfo);
            n_samples = (da_int)rinfo[0];
            n_features = (da_int)rinfo[1];
            n_components = (da_int)rinfo[2];
        }
        exception_check(status);

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
};

PYBIND11_MODULE(_aoclda, m) {
    m.doc() = "Python wrappers for the AOCL-DA library";

    /* Higher level types */
    py::enum_<da_precision>(m, "precision")
        .value("single", da_single)
        .value("double", da_double)
        .export_values();

    /**********************************/
    /*         Basic statistics       */
    /**********************************/
    auto m_stats = m.def_submodule("basic_stats", "Basic statistics.");
    /* enum types */
    py::enum_<da_axis_>(m_stats, "axis")
        .value("col", da_axis::da_axis_col)
        .value("row", da_axis::da_axis_row)
        .value("all", da_axis::da_axis_all)
        .export_values();
    m_stats.def("mean", &py_da_mean<float>);
    m_stats.def("mean", &py_da_mean<double>);

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
    /* enum */
    py::enum_<linmod_model_>(m_linmod, "linmod_model")
        .value("mse", linmod_model::linmod_model_mse)
        .value("logistic", linmod_model::linmod_model_logistic)
        .export_values();
    py::class_<linmod, pyda_handle>(m_linmod, "pybind_linmod")
        .def(py::init<linmod_model_, bool, da_precision &>(), "mod"_a,
             py::arg("intercept") = false, py::arg("precision") = da_double)
        .def("pybind_fit", &linmod::fit<float>, "Computes the model", "X"_a, "y"_a,
             py::arg("reg_lambda") = (float)0.0, py::arg("reg_alpha") = (float)0.0)
        .def("pybind_fit", &linmod::fit<double>, "Computes the model", "X"_a, "y"_a,
             py::arg("reg_lambda") = (double)0.0, py::arg("reg_alpha") = (double)0.0)
        .def("get_coef", &linmod::get_coef);

    /**********************************/
    /*  Principal component analysis  */
    /**********************************/
    auto m_factorization = m.def_submodule("factorization", "Matrix factorizations.");
    py::class_<pca, pyda_handle>(m_factorization, "pybind_PCA")
        .def(py::init<da_int, std::string, std::string, std::string, da_precision &>(),
             py::arg("n_components") = 1, py::arg("bias") = "unbiased",
             py::arg("method") = "covariance", py::arg("solver") = "gesdd",
             py::arg("precision") = da_double)
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
        .def("get_column_sdevs", &pca::get_column_sdevs);
}