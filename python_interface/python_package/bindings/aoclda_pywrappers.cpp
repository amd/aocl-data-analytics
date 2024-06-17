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
#include "basic_stats_py.hpp"
#include "decision_forest_py.hpp"
#include "factorization_py.hpp"
#include "linmod_py.hpp"
#include "utilities_py.hpp"
#include <iostream>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;
using namespace pybind11::literals;

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
             py::arg("x0") = py::none(), py::arg("reg_lambda") = (float)0.0,
             py::arg("reg_alpha") = (float)0.0, py::arg("tol") = (float)0.0001)
        .def("pybind_fit", &linmod::fit<double>, "Computes the model", "X"_a, "y"_a,
             py::arg("x0") = py::none(), py::arg("reg_lambda") = (double)0.0,
             py::arg("reg_alpha") = (double)0.0, py::arg("tol") = (double)0.0001)
        .def("pybind_predict", &linmod::predict<double>, "Evaluate the model on X", "X"_a)
        .def("pybind_predict", &linmod::predict<float>, "Evaluate the model on X", "X"_a)
        .def("get_coef", &linmod::get_coef)
        .def("get_loss", &linmod::get_loss)
        .def("get_norm_gradient_loss", &linmod::get_norm_gradient_loss)
        .def("get_n_iter", &linmod::get_n_iter)
        .def("get_time", &linmod::get_time);

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

    /**********************************/
    /*  Decision Trees                */
    /**********************************/
    auto m_decision_tree = m.def_submodule("decision_tree", "Decision trees.");
    py::class_<decision_tree, pyda_handle>(m_decision_tree, "pybind_decision_tree")
        .def(py::init<da_int, da_int, da_int, std::string, da_int, std::string,
                      std::string &>(),
             py::arg("seed") = -1, py::arg("max_depth") = 10, py::arg("max_features") = 0,
             py::arg("criterion") = "gini", py::arg("min_samples_split") = 2,
             py::arg("build_order") = "breadth first", py::arg("precision") = "double")
        .def("pybind_fit", &decision_tree::fit<float>, "Fit the decision tree", "X"_a,
             "y"_a, py::arg("min_impurity_decrease") = 0.03,
             py::arg("min_split_score") = 0.03, py::arg("feat_thresh") = 0.0)
        .def("pybind_fit", &decision_tree::fit<double>, "Fit the decision tree", "X"_a,
             "y"_a, py::arg("min_impurity_decrease") = 0.03,
             py::arg("min_split_score") = 0.03, py::arg("feat_thresh") = 0.0)
        .def("pybind_score", &decision_tree::score<float>, "Score the decision tree",
             "X_test"_a, "y_test"_a)
        .def("pybind_score", &decision_tree::score<double>, "Score the decision tree",
             "X_test"_a, "y_test"_a)
        .def("pybind_predict", &decision_tree::predict<double>, "Evaluate the model on X",
             "X"_a)
        .def("pybind_predict", &decision_tree::predict<float>, "Evaluate the model on X",
             "X"_a);

    /**********************************/
    /*  Decision Forests              */
    /**********************************/
    auto m_decision_forest = m.def_submodule("decision_forest", "Decision forests.");
    py::class_<decision_forest, pyda_handle>(m_decision_forest, "pybind_decision_forest")
        .def(py::init<da_int, std::string, da_int, da_int, da_int, std::string,
                      std::string, std::string, da_int, std::string &>(),
             py::arg("n_trees") = 100, py::arg("criterion") = "gini",
             py::arg("seed") = -1, py::arg("max_depth") = 10,
             py::arg("min_samples_split") = 2, py::arg("build_order") = "breadth first",
             py::arg("bootstrap") = "yes", py::arg("features_selection") = "sqrt",
             py::arg("max_features") = 0, py::arg("precision") = "double")
        .def("pybind_fit", &decision_forest::fit<float>, "Fit the decision forest", "X"_a,
             "y"_a, py::arg("samples factor") = 0.8,
             py::arg("min_impurity_decrease") = 0.03, py::arg("min_split_score") = 0.03,
             py::arg("feat_thresh") = 0.0)
        .def("pybind_fit", &decision_forest::fit<double>, "Fit the decision forest",
             "X"_a, "y"_a, py::arg("samples factor") = 0.8,
             py::arg("min_impurity_decrease") = 0.03, py::arg("min_split_score") = 0.03,
             py::arg("feat_thresh") = 0.0)
        .def("pybind_score", &decision_forest::score<float>, "Score the decision forest",
             "X_test"_a, "y_test"_a)
        .def("pybind_score", &decision_forest::score<double>, "Score the decision forest",
             "X_test"_a, "y_test"_a)
        .def("pybind_predict", &decision_forest::predict<double>,
             "Evaluate the model on X", "X"_a)
        .def("pybind_predict", &decision_forest::predict<float>,
             "Evaluate the model on X", "X"_a);
}
