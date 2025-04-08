/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "dbscan_py.hpp"
#include "decision_forest_py.hpp"
#include "factorization_py.hpp"
#include "kernel_functions_py.hpp"
#include "kmeans_py.hpp"
#include "linmod_py.hpp"
#include "metrics_py.hpp"
#include "nearest_neighbors_py.hpp"
#include "nlls_py.hpp"
#include "svm_py.hpp"
#include "utilities_py.hpp"
#include <iostream>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;
using namespace pybind11::literals;

// -----------------------------------------------------------------------------
// Call backs
// NLLS
// C++ side call-backs, these will instantiate a template that calls the python user-function
da_int py_wrapper_resfun_d(da_int n_coef, da_int n_res, void *data, const double *x,
                           double *r) {
    return nlls_cb::py_wrapper_resfun_t(n_coef, n_res, data, x, r);
}
da_int py_wrapper_resfun_s(da_int n_coef, da_int n_res, void *data, const float *x,
                           float *r) {
    return nlls_cb::py_wrapper_resfun_t(n_coef, n_res, data, x, r);
}
da_int py_wrapper_resgrd_d(da_int n_coef, da_int n_res, void *data, const double *x,
                           double *J) {
    return nlls_cb::py_wrapper_resgrd_t(n_coef, n_res, data, x, J);
}
da_int py_wrapper_resgrd_s(da_int n_coef, da_int n_res, void *data, const float *x,
                           float *J) {
    return nlls_cb::py_wrapper_resgrd_t(n_coef, n_res, data, x, J);
}
da_int py_wrapper_reshes_d(da_int n_coef, da_int n_res, void *data, const double *x,
                           const double *r, double *HF) {
    return nlls_cb::py_wrapper_reshes_t(n_coef, n_res, data, x, r, HF);
}
da_int py_wrapper_reshes_s(da_int n_coef, da_int n_res, void *data, const float *x,
                           const float *r, float *HF) {
    return nlls_cb::py_wrapper_reshes_t(n_coef, n_res, data, x, r, HF);
}
da_int py_wrapper_reshp_d(da_int n_coef, da_int n_res, const double *x, const double *y,
                          double *HP, void *data) {
    return nlls_cb::py_wrapper_reshp_t(n_coef, n_res, x, y, HP, data);
}
da_int py_wrapper_reshp_s(da_int n_coef, da_int n_res, const float *x, const float *y,
                          float *HP, void *data) {
    return nlls_cb::py_wrapper_reshp_t(n_coef, n_res, x, y, HP, data);
}
// -----------------------------------------------------------------------------

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
                      std::string, std::string &, bool>(),
             py::arg("mod"), py::arg("max_iter") = py::none(),
             py::arg("intercept") = false, py::arg("solver") = "auto",
             py::arg("scaling") = "auto", py::arg("constraint") = "ssc",
             py::arg("precision") = "double", py::arg("check_data") = false)
        .def("pybind_fit", &linmod::fit<float>, "Computes the model", "X"_a, "y"_a,
             py::arg("x0") = py::none(), py::arg("progress_factor") = py::none(),
             py::arg("reg_lambda") = (float)0.0, py::arg("reg_alpha") = (float)0.0,
             py::arg("tol") = (float)0.0001)
        .def("pybind_fit", &linmod::fit<double>, "Computes the model", "X"_a, "y"_a,
             py::arg("x0") = py::none(), py::arg("progress_factor") = py::none(),
             py::arg("reg_lambda") = (double)0.0, py::arg("reg_alpha") = (double)0.0,
             py::arg("tol") = (double)0.0001)
        .def("pybind_predict", &linmod::predict<double>, "Evaluate the model on X", "X"_a)
        .def("pybind_predict", &linmod::predict<float>, "Evaluate the model on X", "X"_a)
        .def("get_coef", &linmod::get_coef)
        .def("get_loss", &linmod::get_loss)
        .def("get_norm_gradient_loss", &linmod::get_norm_gradient_loss)
        .def("get_n_iter", &linmod::get_n_iter)
        .def("get_time", &linmod::get_time);

    /**********************************/
    /*  Principal Component Analysis  */
    /**********************************/
    auto m_factorization = m.def_submodule("factorization", "Matrix factorizations.");
    py::class_<pca, pyda_handle>(m_factorization, "pybind_PCA")
        .def(py::init<da_int, std::string, std::string, std::string, bool, std::string &,
                      bool>(),
             py::arg("n_components") = 1, py::arg("bias") = "unbiased",
             py::arg("method") = "covariance", py::arg("solver") = "gesdd",
             py::arg("store_U") = false, py::arg("precision") = "double",
             py::arg("check_data") = false)
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
                      std::string &, bool>(),
             py::arg("n_clusters") = 1, py::arg("initialization_method") = "k-means++",
             py::arg("n_init") = 10, py::arg("max_iter") = 300, py::arg("seed") = -1,
             py::arg("algorithm") = "elkan", py::arg("precision") = "double",
             py::arg("check_data") = false)
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
    /*       DBSCAN clustering        */
    /**********************************/

    py::class_<DBSCAN, pyda_handle>(m_clustering, "pybind_DBSCAN")
        .def(py::init<da_int, std::string, std::string, da_int, std::string &, bool>(),
             py::arg("min_samples") = 5, py::arg("metric") = "euclidean",
             py::arg("algorithm") = "brute", py::arg("leaf_size") = 30,
             py::arg("precision") = "double", py::arg("check_data") = false)
        .def("pybind_fit", &DBSCAN::fit<float>, "Fit the DBSCAN clusters", "A"_a,
             py::arg("eps") = (float)0.5, py::arg("power") = (float)2.0)
        .def("pybind_fit", &DBSCAN::fit<double>, "Fit the DBSCAN clusters", "A"_a,
             py::arg("eps") = (double)0.5, py::arg("power") = (double)2.0)
        .def("get_labels", &DBSCAN::get_labels)
        .def("get_n_samples", &DBSCAN::get_n_samples)
        .def("get_n_features", &DBSCAN::get_n_features)
        .def("get_n_core_samples", &DBSCAN::get_n_core_samples)
        .def("get_core_sample_indices", &DBSCAN::get_core_sample_indices)
        .def("get_n_clusters", &DBSCAN::get_n_clusters);

    /**********************************/
    /*        Decision Trees          */
    /**********************************/
    auto m_decision_tree = m.def_submodule("decision_tree", "Decision trees.");
    py::class_<decision_tree, pyda_handle>(m_decision_tree, "pybind_decision_tree")
        .def(py::init<da_int, da_int, da_int, std::string, da_int, std::string,
                      std::string, std::string &, bool>(),
             py::arg("seed") = -1, py::arg("max_depth") = 29, py::arg("max_features") = 0,
             py::arg("criterion") = "gini", py::arg("min_samples_split") = 2,
             py::arg("build_order") = "breadth first", py::arg("sort_method") = "boost",
             py::arg("precision") = "double", py::arg("check_data") = false)
        .def("pybind_fit", &decision_tree::fit<float>, "Fit the decision tree", "X"_a,
             "y"_a, py::arg("min_impurity_decrease") = 0.0,
             py::arg("min_split_score") = 0.0, py::arg("feat_thresh") = 0.0)
        .def("pybind_fit", &decision_tree::fit<double>, "Fit the decision tree", "X"_a,
             "y"_a, py::arg("min_impurity_decrease") = 0.0,
             py::arg("min_split_score") = 0.0, py::arg("feat_thresh") = 0.0)
        .def("pybind_score", &decision_tree::score<float>, "Score the decision tree",
             "X_test"_a, "y_test"_a)
        .def("pybind_score", &decision_tree::score<double>, "Score the decision tree",
             "X_test"_a, "y_test"_a)
        .def("pybind_predict", &decision_tree::predict<double>, "Evaluate the model on X",
             "X"_a)
        .def("pybind_predict", &decision_tree::predict<float>, "Evaluate the model on X",
             "X"_a)
        .def("pybind_predict_proba", &decision_tree::predict_proba<double>,
             "Evaluate the model on X", "X"_a)
        .def("pybind_predict_proba", &decision_tree::predict_proba<float>,
             "Evaluate the model on X", "X"_a)
        .def("pybind_predict_log_proba", &decision_tree::predict_log_proba<double>,
             "Evaluate the model on X", "X"_a)
        .def("pybind_predict_log_proba", &decision_tree::predict_log_proba<float>,
             "Evaluate the model on X", "X"_a)
        .def("get_n_nodes", &decision_tree::get_n_nodes)
        .def("get_n_leaves", &decision_tree::get_n_leaves)
        .def("set_max_features_opt", &decision_tree::set_max_features_opt,
             "Set options for feature selection", py::arg("max_features") = 0);

    /**********************************/
    /*       Decision Forests         */
    /**********************************/
    auto m_decision_forest = m.def_submodule("decision_forest", "Decision forests.");
    py::class_<decision_forest, pyda_handle>(m_decision_forest, "pybind_decision_forest")
        .def(py::init<da_int, std::string, da_int, da_int, da_int, std::string, bool,
                      std::string, da_int, std::string &, bool>(),
             py::arg("n_trees") = 100, py::arg("criterion") = "gini",
             py::arg("seed") = -1, py::arg("max_depth") = 29,
             py::arg("min_samples_split") = 2, py::arg("build_order") = "breadth first",
             py::arg("bootstrap") = true, py::arg("features_selection") = "sqrt",
             py::arg("max_features") = 0, py::arg("precision") = "double",
             py::arg("check_data") = false)
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
             "Evaluate the model on X", "X"_a)
        .def("pybind_predict_proba", &decision_forest::predict_proba<double>,
             "Evaluate the model on X", "X"_a)
        .def("pybind_predict_proba", &decision_forest::predict_proba<float>,
             "Evaluate the model on X", "X"_a)
        .def("pybind_predict_log_proba", &decision_forest::predict_log_proba<double>,
             "Evaluate the model on X", "X"_a)
        .def("pybind_predict_log_proba", &decision_forest::predict_log_proba<float>,
             "Evaluate the model on X", "X"_a)
        .def("set_max_features_opt", &decision_forest::set_max_features_opt,
             "Set options for feature selection", py::arg("max_features") = 0)
        .def("set_features_selection_opt", &decision_forest::set_features_selection_opt,
             "Set options for feature selection", py::arg("features_selection") = "sqrt");

    /**********************************/
    /*     Nonlinear Data Fitting     */
    /**********************************/
    auto m_nlls = m.def_submodule("nlls", "Nonlinear data fitting.");
    py::class_<nlls, pyda_handle>(m_nlls, "pybind_nlls")
        .def(py::init<da_int, da_int, std::optional<py::array>, std::optional<py::array>,
                      std::optional<py::array>, std::string, std::string, std::string,
                      std::string, std::string, std::string, std::string, da_int, bool>(),
             py::arg("n_coef"), py::arg("n_res"), py::arg("weights") = py::none(),
             py::arg("lower_bounds") = py::none(), py::arg("upper_bounds") = py::none(),
             py::arg("order") = "c", py::arg("prec") = "double",
             py::arg("model") = "hybrid", py::arg("method") = "galahad",
             py::arg("glob_strategy") = "tr", py::arg("reg_power") = "quadratic",
             py::arg("check_derivatives") = "no", py::arg("verbose") = (da_int)0,
             py::arg("check_data") = false)
        .def("fit_d", &nlls::fit<double>, "Fit data and train the model", "x"_a, "fun"_a,
             "jac"_a, "hes"_a = py::none(), "hep"_a = py::none(), "data"_a = py::none(),
             py::arg("ftol") = (double)1.0e-8, py::arg("abs_ftol") = (double)1.0e-8,
             py::arg("gtol") = (double)1.0e-8, py::arg("abs_gtol") = (double)1.0e-5,
             py::arg("xtol") = (double)2.22e-16, py::arg("reg_term") = (double)0.0,
             py::arg("maxit") = (da_int)100, py::arg("fd_step") = (double)1.0e-7,
             py::arg("fd_ttol") = (double)1.0e-4)
        .def("fit_s", &nlls::fit<float>, "Fit data and train the model", "x"_a, "fun"_a,
             "jac"_a, "hes"_a = py::none(), "hep"_a = py::none(), "data"_a = py::none(),
             py::arg("ftol") = (float)1.0e-8, py::arg("abs_ftol") = (float)1.0e-8,
             py::arg("gtol") = (float)1.0e-8, py::arg("abs_gtol") = (float)1.0e-5,
             py::arg("xtol") = (float)2.22e-16, py::arg("reg_term") = (float)0.0,
             py::arg("maxit") = (da_int)100, py::arg("fd_step") = (float)1.0e-7,
             py::arg("fd_ttol") = (float)1.0e-4)
        // hidden @properties
        .def("_get_precision", &nlls::get_precision) // -> string
        // @properties
        // info[da_optim_info_t::info_iter] = T(inform.iter);
        .def("get_info_iter", &nlls::get_info_iter) // -> int
        // info[da_optim_info_t::info_nevalf] = T(inform.f_eval);
        // info[da_optim_info_t::info_nevalg] = T(inform.g_eval);
        // info[da_optim_info_t::info_nevalh] = T(inform.h_eval);
        // info[da_optim_info_t::info_nevalhp] = T(inform.hp_eval);
        // info[da_optim_info_t::info_nevalfd] = T(inform.fd_f_eval);
        .def("get_info_evals", &nlls::get_info_evals) // -> dict
        // info[da_optim_info_t::info_objective] = T(inform.obj);
        // info[da_optim_info_t::info_grad_norm] = T(inform.norm_g);
        // info[da_optim_info_t::info_scl_grad_norm] = T(inform.scaled_g);
        .def("get_info_optim", &nlls::get_info_optim); // -> dict

    /**********************************/
    /*         kNN Classifier         */
    /**********************************/
    auto m_knn_classifier =
        m.def_submodule("nearest_neighbors", "k-Nearest Neighbors for classification");
    py::class_<knn_classifier, pyda_handle>(m_knn_classifier, "pybind_knn_classifier")
        .def(py::init<da_int, std::string, std::string, std::string, std::string &,
                      bool>(),
             py::arg("n_neighbors") = (da_int)5, py::arg("weights") = "uniform",
             py::arg("algorithm") = "brute", py::arg("metric") = "euclidean",
             py::arg("precision") = "double", py::arg("check_data") = false)
        .def("pybind_fit", &knn_classifier::fit<float>, "Fit the knn classifier", "X"_a,
             "y"_a, py::arg("p") = (float)2.0)
        .def("pybind_fit", &knn_classifier::fit<double>, "Fit the knn classifier", "X"_a,
             "y"_a, py::arg("p") = (double)2.0)
        .def("pybind_kneighbors_indices", &knn_classifier::kneighbors_indices<float>,
             "Compute the indices of the k-nearest neighbors", "X"_a,
             py::arg("n_neighbors") = (da_int)0)
        .def("pybind_kneighbors_indices", &knn_classifier::kneighbors_indices<double>,
             "Compute the indices of the k-nearest neighbors", "X"_a,
             py::arg("n_neighbors") = (da_int)0)
        .def("pybind_kneighbors", &knn_classifier::kneighbors<float>,
             "Compute the indices of the k-nearest neighbors and the corresponding "
             "distances",
             "X"_a, py::arg("n_neighbors") = (da_int)0)
        .def("pybind_kneighbors", &knn_classifier::kneighbors<double>,
             "Compute the indices of the k-nearest neighbors and the corresponding "
             "distances",
             "X"_a, py::arg("n_neighbors") = (da_int)0)
        // Definition for float type needs to be before double type, otherwise this gives
        // "RuntimeError: The handle was initialized with a different precision type than double."
        .def("pybind_predict_proba", &knn_classifier::predict_proba<float>,
             "Compute the probabilities estimates for the test data", "X"_a)
        .def("pybind_predict_proba", &knn_classifier::predict_proba<double>,
             "Compute the probabilities estimates for the test data", "X"_a)
        .def("pybind_predict", &knn_classifier::predict<float>,
             "Compute the predicted labels for the test data", "X"_a)
        .def("pybind_predict", &knn_classifier::predict<double>,
             "Compute the predicted labels for the test data", "X"_a);

    /**********************************/
    /*         Pairwise Distances     */
    /**********************************/
    auto m_pairwise = m.def_submodule("metrics", "Distance Metrics.");
    m_pairwise.def("pybind_pairwise_distances", &py_da_pairwise_distances<float>, "X"_a,
                   "Y"_a = py::none(), "metric"_a = "euclidean",
                   py::arg("p") = (float)2.0);
    m_pairwise.def("pybind_pairwise_distances", &py_da_pairwise_distances<double>, "X"_a,
                   "Y"_a = py::none(), "metric"_a = "euclidean",
                   py::arg("p") = (double)2.0);
    /**********************************/
    /*         Kernel Functions       */
    /**********************************/
    auto m_kernel_functions = m.def_submodule("kernel_functions", "Kernel Functions.");
    m_kernel_functions.def("pybind_rbf_kernel", &py_da_rbf_kernel<float>, "X"_a,
                           "Y"_a = py::none(), "gamma"_a = (float)1.0);
    m_kernel_functions.def("pybind_rbf_kernel", &py_da_rbf_kernel<double>, "X"_a,
                           "Y"_a = py::none(), "gamma"_a = (double)1.0);
    m_kernel_functions.def("pybind_linear_kernel", &py_da_linear_kernel<float>, "X"_a,
                           "Y"_a = py::none());
    m_kernel_functions.def("pybind_linear_kernel", &py_da_linear_kernel<double>, "X"_a,
                           "Y"_a = py::none());
    m_kernel_functions.def("pybind_polynomial_kernel", &py_da_polynomial_kernel<float>,
                           "X"_a, "Y"_a = py::none(), "degree"_a = (da_int)3,
                           "gamma"_a = (float)1.0, "coef0"_a = (float)1.0);
    m_kernel_functions.def("pybind_polynomial_kernel", &py_da_polynomial_kernel<double>,
                           "X"_a, "Y"_a = py::none(), "degree"_a = (da_int)3,
                           "gamma"_a = (double)1.0, "coef0"_a = (double)1.0);
    m_kernel_functions.def("pybind_sigmoid_kernel", &py_da_sigmoid_kernel<float>, "X"_a,
                           "Y"_a = py::none(), "gamma"_a = (float)1.0,
                           "coef0"_a = (float)1.0);
    m_kernel_functions.def("pybind_sigmoid_kernel", &py_da_sigmoid_kernel<double>, "X"_a,
                           "Y"_a = py::none(), "gamma"_a = (double)1.0,
                           "coef0"_a = (double)1.0);

    /*******************/
    /*      SVM        */
    /*******************/
    auto m_svm = m.def_submodule("svm", "Support Vector Machine");
    // SVC
    py::class_<py_svc, pyda_handle>(m_svm, "pybind_svc")
        .def(py::init<std::string, da_int, da_int, std::string, bool &>(),
             py::arg("kernel") = "rbf", py::arg("degree") = 3, py::arg("max_iter") = -1,
             py::arg("precision") = "double", py::arg("check_data") = false)
        .def("pybind_fit", &py_svc::fit<float>, "Fit the SVC model", "X"_a, "y"_a,
             py::arg("tau") = py::none(), py::arg("C") = 1.0, py::arg("gamma") = 1,
             py::arg("coef0") = 0.0, py::arg("tol") = 0.001)
        .def("pybind_fit", &py_svc::fit<double>, "Fit the SVC model", "X"_a, "y"_a,
             py::arg("tau") = py::none(), py::arg("C") = 1.0, py::arg("gamma") = 1,
             py::arg("coef0") = 0.0, py::arg("tol") = 0.001)
        .def("pybind_predict", &py_svm::predict<float>,
             "Compute the predicted labels for the test data", "X"_a)
        .def("pybind_predict", &py_svm::predict<double>,
             "Compute the predicted labels for the test data", "X"_a)
        .def("pybind_decision_function", &py_svm::decision_function<float>,
             "Compute the decision function for the test data", "X"_a,
             py::arg("shape") = "ovr")
        .def("pybind_decision_function", &py_svm::decision_function<double>,
             "Compute the decision function for the test data", "X"_a,
             py::arg("shape") = "ovr")
        .def("pybind_score", &py_svm::score<float>, "Compute the score for the test data",
             "X"_a, "y"_a)
        .def("pybind_score", &py_svm::score<double>,
             "Compute the score for the test data", "X"_a, "y"_a)
        .def("get_n_samples", &py_svm::get_n_samples)
        .def("get_n_features", &py_svm::get_n_features)
        .def("get_n_classes", &py_svm::get_n_classes)
        .def("get_n_iterations", &py_svm::get_n_iterations)
        .def("get_n_sv", &py_svm::get_n_sv)
        .def("get_n_sv_per_class", &py_svm::get_n_sv_per_class)
        .def("get_dual_coef", &py_svm::get_dual_coef)
        .def("get_bias", &py_svm::get_bias)
        .def("get_support_vectors_idx", &py_svm::get_support_vectors_idx)
        .def("get_sv", &py_svm::get_sv);
    // SVR
    py::class_<py_svr, pyda_handle>(m_svm, "pybind_svr")
        .def(py::init<std::string, da_int, da_int, std::string, bool &>(),
             py::arg("kernel") = "rbf", py::arg("degree") = 3, py::arg("max_iter") = -1,
             py::arg("precision") = "double", py::arg("check_data") = false)
        .def("pybind_fit", &py_svr::fit<float>, "Fit the SVC model", "X"_a, "y"_a,
             py::arg("tau") = py::none(), py::arg("C") = 1.0, py::arg("epsilon") = 0.1,
             py::arg("gamma") = 1, py::arg("coef0") = 0.0, py::arg("tol") = 0.001)
        .def("pybind_fit", &py_svr::fit<double>, "Fit the SVC model", "X"_a, "y"_a,
             py::arg("tau") = py::none(), py::arg("C") = 1.0, py::arg("epsilon") = 0.1,
             py::arg("gamma") = 1, py::arg("coef0") = 0.0, py::arg("tol") = 0.001)
        .def("pybind_predict", &py_svm::predict<float>,
             "Compute the predicted labels for the test data", "X"_a)
        .def("pybind_predict", &py_svm::predict<double>,
             "Compute the predicted labels for the test data", "X"_a)
        .def("pybind_score", &py_svm::score<float>, "Compute the score for the test data",
             "X"_a, "y"_a)
        .def("pybind_score", &py_svm::score<double>,
             "Compute the score for the test data", "X"_a, "y"_a)
        .def("get_n_samples", &py_svm::get_n_samples)
        .def("get_n_features", &py_svm::get_n_features)
        .def("get_n_classes", &py_svm::get_n_classes)
        .def("get_n_iterations", &py_svm::get_n_iterations)
        .def("get_n_sv", &py_svm::get_n_sv)
        .def("get_dual_coef", &py_svm::get_dual_coef)
        .def("get_bias", &py_svm::get_bias)
        .def("get_support_vectors_idx", &py_svm::get_support_vectors_idx)
        .def("get_sv", &py_svm::get_sv);
    // nuSVC
    py::class_<py_nusvc, pyda_handle>(m_svm, "pybind_nusvc")
        .def(py::init<std::string, da_int, da_int, std::string, bool &>(),
             py::arg("kernel") = "rbf", py::arg("degree") = 3, py::arg("max_iter") = -1,
             py::arg("precision") = "double", py::arg("check_data") = false)
        .def("pybind_fit", &py_nusvc::fit<float>, "Fit the SVC model", "X"_a, "y"_a,
             py::arg("tau") = py::none(), py::arg("nu") = 0.5, py::arg("gamma") = 1,
             py::arg("coef0") = 0.0, py::arg("tol") = 0.001)
        .def("pybind_fit", &py_nusvc::fit<double>, "Fit the SVC model", "X"_a, "y"_a,
             py::arg("tau") = py::none(), py::arg("nu") = 0.5, py::arg("gamma") = 1,
             py::arg("coef0") = 0.0, py::arg("tol") = 0.001)
        .def("pybind_predict", &py_svm::predict<float>,
             "Compute the predicted labels for the test data", "X"_a)
        .def("pybind_predict", &py_svm::predict<double>,
             "Compute the predicted labels for the test data", "X"_a)
        .def("pybind_decision_function", &py_svm::decision_function<float>,
             "Compute the decision function for the test data", "X"_a,
             py::arg("shape") = "ovr")
        .def("pybind_decision_function", &py_svm::decision_function<double>,
             "Compute the decision function for the test data", "X"_a,
             py::arg("shape") = "ovr")
        .def("pybind_score", &py_svm::score<float>, "Compute the score for the test data",
             "X"_a, "y"_a)
        .def("pybind_score", &py_svm::score<double>,
             "Compute the score for the test data", "X"_a, "y"_a)
        .def("get_n_samples", &py_svm::get_n_samples)
        .def("get_n_features", &py_svm::get_n_features)
        .def("get_n_classes", &py_svm::get_n_classes)
        .def("get_n_iterations", &py_svm::get_n_iterations)
        .def("get_n_sv", &py_svm::get_n_sv)
        .def("get_n_sv_per_class", &py_svm::get_n_sv_per_class)
        .def("get_dual_coef", &py_svm::get_dual_coef)
        .def("get_bias", &py_svm::get_bias)
        .def("get_support_vectors_idx", &py_svm::get_support_vectors_idx)
        .def("get_sv", &py_svm::get_sv);
    // nuSVR
    py::class_<py_nusvr, pyda_handle>(m_svm, "pybind_nusvr")
        .def(py::init<std::string, da_int, da_int, std::string, bool &>(),
             py::arg("kernel") = "rbf", py::arg("degree") = 3, py::arg("max_iter") = -1,
             py::arg("precision") = "double", py::arg("check_data") = false)
        .def("pybind_fit", &py_nusvr::fit<float>, "Fit the SVC model", "X"_a, "y"_a,
             py::arg("tau") = py::none(), py::arg("nu") = 0.5, py::arg("C") = 1.0,
             py::arg("gamma") = 1, py::arg("coef0") = 0.0, py::arg("tol") = 0.001)
        .def("pybind_fit", &py_nusvr::fit<double>, "Fit the SVC model", "X"_a, "y"_a,
             py::arg("tau") = py::none(), py::arg("nu") = 0.5, py::arg("C") = 1.0,
             py::arg("gamma") = 1, py::arg("coef0") = 0.0, py::arg("tol") = 0.001)
        .def("pybind_predict", &py_svm::predict<float>,
             "Compute the predicted labels for the test data", "X"_a)
        .def("pybind_predict", &py_svm::predict<double>,
             "Compute the predicted labels for the test data", "X"_a)
        .def("pybind_score", &py_svm::score<float>, "Compute the score for the test data",
             "X"_a, "y"_a)
        .def("pybind_score", &py_svm::score<double>,
             "Compute the score for the test data", "X"_a, "y"_a)
        .def("get_n_samples", &py_svm::get_n_samples)
        .def("get_n_features", &py_svm::get_n_features)
        .def("get_n_classes", &py_svm::get_n_classes)
        .def("get_n_iterations", &py_svm::get_n_iterations)
        .def("get_n_sv", &py_svm::get_n_sv)
        .def("get_dual_coef", &py_svm::get_dual_coef)
        .def("get_bias", &py_svm::get_bias)
        .def("get_support_vectors_idx", &py_svm::get_support_vectors_idx)
        .def("get_sv", &py_svm::get_sv);
}
