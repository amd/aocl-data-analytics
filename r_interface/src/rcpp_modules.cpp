/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Single file collecting all RCPP_MODULE exports for the AOCL-DA R interface.
// Analogous to python_interface/python_package/bindings/aoclda_pywrappers.cpp.
//
// Each *_r.hpp header contains the wrapper class/function implementations.
// This file includes all headers and registers every module so R can load them.

#include "approximate_neighbors_r.hpp"
#include "basic_stats_r.hpp"
#include "dbscan_r.hpp"
#include "decision_forest_r.hpp"
#include "factorization_r.hpp"
#include "interpolation_r.hpp"
#include "kernel_functions_r.hpp"
#include "kmeans_r.hpp"
#include "linmod_r.hpp"
#include "metrics_r.hpp"
#include "nearest_neighbors_r.hpp"
#include "nlls_r.hpp"
#include "svm_r.hpp"
#include "tsne_r.hpp"
#include "utils_r.hpp"

using namespace Rcpp;

// ---------------------------------------------------------------------------
// Basic statistics
// ---------------------------------------------------------------------------
RCPP_MODULE(basic_stats_module) {
    function("rcpp_mean", &da_mean_wrapper, "Compute mean along specified axis");
    function("rcpp_harmonic_mean", &da_harmonic_mean_wrapper,
             "Compute harmonic mean along specified axis");
    function("rcpp_geometric_mean", &da_geometric_mean_wrapper,
             "Compute geometric mean along specified axis");
    function("rcpp_variance", &da_variance_wrapper,
             "Compute variance along specified axis");
    function("rcpp_skewness", &da_skewness_wrapper,
             "Compute skewness along specified axis");
    function("rcpp_kurtosis", &da_kurtosis_wrapper,
             "Compute kurtosis along specified axis");
    function("rcpp_moment", &da_moment_wrapper,
             "Compute k-th moment along specified axis");
    function("rcpp_quantile", &da_quantile_wrapper,
             "Compute quantile along specified axis");
    function("rcpp_five_point_summary", &da_five_point_summary_wrapper,
             "Compute five-point summary along specified axis");
    function("rcpp_standardize", &da_standardize_wrapper,
             "Standardize data along specified axis");
    function("rcpp_covariance", &da_covariance_wrapper, "Compute covariance matrix");
    function("rcpp_correlation", &da_correlation_wrapper, "Compute correlation matrix");
}

// ---------------------------------------------------------------------------
// Approximate nearest neighbors
// ---------------------------------------------------------------------------
RCPP_MODULE(approx_nn_module) {
    class_<ApproximateNeighbors>("rccp_ApproximateNeighbors")
        .constructor<>("Initialize ApproximateNeighbors with defaults")
        .constructor<da_int, std::string, std::string, da_int, da_int, da_int, da_int,
                     std::string, std::string, bool, bool>(
            "Initialize ApproximateNeighbors")
        .method("set_n_probe_opt", &ApproximateNeighbors::set_n_probe_opt,
                "Update n_probe without re-training")
        .method("train", &ApproximateNeighbors::train,
                "Train the ANN index (centroids only)")
        .method("add", &ApproximateNeighbors::add, "Add data to an already-trained index")
        .method("train_and_add", &ApproximateNeighbors::train_and_add,
                "Train and add data in one step")
        .method("kneighbors_indices", &ApproximateNeighbors::kneighbors_indices,
                "Query k approximate nearest neighbor indices only")
        .method("kneighbors", &ApproximateNeighbors::kneighbors,
                "Query k approximate nearest neighbors")
        .method("get_cluster_centroids", &ApproximateNeighbors::get_cluster_centroids,
                "Get cluster centroids after training")
        .method("get_list_sizes", &ApproximateNeighbors::get_list_sizes,
                "Get sizes of each inverted list")
        .method("get_n_list", &ApproximateNeighbors::get_n_list,
                "Get number of inverted lists")
        .method("get_n_index", &ApproximateNeighbors::get_n_index,
                "Get number of indexed points")
        .method("get_n_features", &ApproximateNeighbors::get_n_features,
                "Get number of features")
        .method("get_kmeans_iter", &ApproximateNeighbors::get_kmeans_iter,
                "Get number of k-means iterations used")
        .method("print_model_versions",
                static_cast<void (ApproximateNeighbors::*)()>(
                    &rda_handle::print_model_versions),
                "Print model version metadata")
        .method("save_model",
                static_cast<Rcpp::List (ApproximateNeighbors::*)()>(
                    &ApproximateNeighbors::save_model),
                "Save model state")
        .method("load_model",
                static_cast<void (ApproximateNeighbors::*)(Rcpp::List)>(
                    &ApproximateNeighbors::load_model),
                "Load model state");
}

// ---------------------------------------------------------------------------
// DBSCAN
// ---------------------------------------------------------------------------
RCPP_MODULE(dbscan_module) {
    class_<DBSCAN>("rccp_DBSCAN")
        .constructor<>("Initialize DBSCAN with defaults")
        .constructor<da_int, std::string, std::string, da_int, bool>("Initialize DBSCAN")
        .method("fit", &DBSCAN::fit, "Fit DBSCAN clustering")
        .method("get_labels", &DBSCAN::get_labels, "Get cluster labels")
        .method("get_core_sample_indices", &DBSCAN::get_core_sample_indices,
                "Get core sample indices")
        .method("get_n_samples", &DBSCAN::get_n_samples, "Get number of samples")
        .method("get_n_features", &DBSCAN::get_n_features, "Get number of features")
        .method("get_n_clusters", &DBSCAN::get_n_clusters, "Get number of clusters")
        .method("get_n_core_samples", &DBSCAN::get_n_core_samples,
                "Get number of core samples")
        .method("print_model_versions",
                static_cast<void (DBSCAN::*)()>(&rda_handle::print_model_versions),
                "Print model version metadata")
        .method("save_model", static_cast<Rcpp::List (DBSCAN::*)()>(&DBSCAN::save_model),
                "Save model state")
        .method("load_model",
                static_cast<void (DBSCAN::*)(Rcpp::List)>(&DBSCAN::load_model),
                "Load model state");
}

// ---------------------------------------------------------------------------
// Decision forest
// ---------------------------------------------------------------------------
RCPP_MODULE(decision_forest_module) {
    class_<DecisionForest>("rccp_DecisionForest")
        .constructor<>("Initialize Decision Forest with defaults")
        .constructor<da_int, std::string, da_int, da_int, da_int, bool, std::string,
                     da_int, bool, da_int, da_int, std::string, da_int, bool>(
            "Initialize Decision Forest")
        .method("fit", &DecisionForest::fit, "Fit decision forest")
        .method("predict", &DecisionForest::predict,
                "Predict using fitted decision forest")
        .method("predict_proba", &DecisionForest::predict_proba,
                "Predict class probabilities")
        .method("predict_log_proba", &DecisionForest::predict_log_proba,
                "Predict log class probabilities")
        .method("score", &DecisionForest::score, "Score the model")
        .method("set_features_selection_opt", &DecisionForest::set_features_selection_opt,
                "Set features selection option")
        .method("set_max_features_opt", &DecisionForest::set_max_features_opt,
                "Set maximum features option")
        .method("get_max_features_opt", &DecisionForest::get_max_features_opt,
                "Get maximum features option")
        .method("get_features_selection_opt", &DecisionForest::get_features_selection_opt,
                "Get features selection option")
        .method(
            "print_model_versions",
            static_cast<void (DecisionForest::*)()>(&rda_handle::print_model_versions),
            "Print model version metadata")
        .method(
            "save_model",
            static_cast<Rcpp::List (DecisionForest::*)()>(&DecisionForest::save_model),
            "Save model state")
        .method("load_model",
                static_cast<void (DecisionForest::*)(Rcpp::List)>(
                    &DecisionForest::load_model),
                "Load model state");
}

// ---------------------------------------------------------------------------
// PCA / factorization
// ---------------------------------------------------------------------------
RCPP_MODULE(pca_module) {
    class_<PCADecomposition>("rccp_PCADecomposition")
        .constructor<>("Initialize PCA with defaults")
        .constructor<da_int, std::string, std::string, std::string, bool, bool, da_int,
                     da_int, std::string, da_int, bool>("Initialize PCA")
        .method("fit", &PCADecomposition::fit, "Fit PCA model")
        .method("transform", &PCADecomposition::transform,
                "Transform data using fitted PCA")
        .method("inverse_transform", &PCADecomposition::inverse_transform,
                "Inverse transform data")
        .method("get_principal_components", &PCADecomposition::get_principal_components,
                "Get principal components")
        .method("get_scores", &PCADecomposition::get_scores,
                "Get principal component scores")
        .method("get_variance", &PCADecomposition::get_variance,
                "Get variance explained by each component")
        .method("get_total_variance", &PCADecomposition::get_total_variance,
                "Get total variance")
        .method("get_u", &PCADecomposition::get_u, "Get U matrix from SVD")
        .method("get_sigma", &PCADecomposition::get_sigma, "Get sigma values from SVD")
        .method("get_vt", &PCADecomposition::get_vt, "Get Vt matrix from SVD")
        .method("get_column_means", &PCADecomposition::get_column_means,
                "Get column means")
        .method("get_column_sdevs", &PCADecomposition::get_column_sdevs,
                "Get column standard deviations")
        .method("get_n_samples", &PCADecomposition::get_n_samples,
                "Get number of samples")
        .method("get_n_features", &PCADecomposition::get_n_features,
                "Get number of features")
        .method("get_n_components", &PCADecomposition::get_n_components,
                "Get number of components")
        .method(
            "print_model_versions",
            static_cast<void (PCADecomposition::*)()>(&rda_handle::print_model_versions),
            "Print model version metadata")
        .method("save_model",
                static_cast<Rcpp::List (PCADecomposition::*)()>(
                    &PCADecomposition::save_model),
                "Save model state")
        .method("load_model",
                static_cast<void (PCADecomposition::*)(Rcpp::List)>(
                    &PCADecomposition::load_model),
                "Load model state");
}

// ---------------------------------------------------------------------------
// Kernel PCA
// ---------------------------------------------------------------------------
RCPP_MODULE(kernel_pca_module) {
    class_<KernelPCADecomposition>("rccp_KernelPCADecomposition")
        .constructor<>("Initialize Kernel PCA with defaults")
        .constructor<da_int, std::string, std::string, da_int, bool, bool, bool, da_int,
                     da_int, std::string, da_int, bool>("Initialize Kernel PCA")
        .method("fit", &KernelPCADecomposition::fit, "Fit Kernel PCA model")
        .method("transform", &KernelPCADecomposition::transform,
                "Transform data using fitted Kernel PCA")
        .method("inverse_transform", &KernelPCADecomposition::inverse_transform,
                "Inverse transform data")
        .method("get_eigenvalues", &KernelPCADecomposition::get_eigenvalues,
                "Get eigenvalues")
        .method("get_eigenvectors", &KernelPCADecomposition::get_eigenvectors,
                "Get eigenvectors")
        .method("get_scores", &KernelPCADecomposition::get_scores, "Get scores")
        .method("get_dual_coef", &KernelPCADecomposition::get_dual_coef,
                "Get dual coefficients")
        .method("get_X_fit", &KernelPCADecomposition::get_X_fit, "Get training data")
        .method("get_gamma", &KernelPCADecomposition::get_gamma, "Get resolved gamma")
        .method("get_n_samples", &KernelPCADecomposition::get_n_samples,
                "Get number of samples")
        .method("get_n_features", &KernelPCADecomposition::get_n_features,
                "Get number of features")
        .method("get_n_components", &KernelPCADecomposition::get_n_components,
                "Get number of components")
        .method("print_model_versions",
                static_cast<void (KernelPCADecomposition::*)()>(
                    &rda_handle::print_model_versions),
                "Print model version metadata")
        .method("save_model",
                static_cast<Rcpp::List (KernelPCADecomposition::*)()>(
                    &KernelPCADecomposition::save_model),
                "Save model state")
        .method("load_model",
                static_cast<void (KernelPCADecomposition::*)(Rcpp::List)>(
                    &KernelPCADecomposition::load_model),
                "Load model state");
}

// ---------------------------------------------------------------------------
// Nonnegative matrix factorization
// ---------------------------------------------------------------------------
RCPP_MODULE(nmf_module) {
    class_<NMFDecomposition>("rccp_NMFDecomposition")
        .constructor<>("Initialize NMF with defaults")
        .constructor<da_int, std::string, std::string, da_int, double, std::string,
                     da_int, da_int, bool>("Initialize NMF")
        .method("fit", &NMFDecomposition::fit, "Fit NMF model")
        .method("fit_custom", &NMFDecomposition::fit_custom,
                "Fit NMF model with user-supplied (W, H) for init='custom'")
        .method("transform", &NMFDecomposition::transform,
                "Transform data using fitted NMF")
        .method("inverse_transform", &NMFDecomposition::inverse_transform,
                "Inverse transform data")
        .method("set_n_components", &NMFDecomposition::set_n_components,
                "Set the number of components option")
        .method("get_components", &NMFDecomposition::get_components,
                "Get the components matrix H")
        .method("get_transform", &NMFDecomposition::get_transform,
                "Get the transform matrix W")
        .method("get_reconstruction_error", &NMFDecomposition::get_reconstruction_error,
                "Get the Frobenius reconstruction error")
        .method("get_n_iter", &NMFDecomposition::get_n_iter,
                "Get the number of iterations performed")
        .method("get_n_samples", &NMFDecomposition::get_n_samples,
                "Get number of samples")
        .method("get_n_features", &NMFDecomposition::get_n_features,
                "Get number of features")
        .method("get_n_components", &NMFDecomposition::get_n_components,
                "Get number of components")
        .method(
            "print_model_versions",
            static_cast<void (NMFDecomposition::*)()>(&rda_handle::print_model_versions),
            "Print model version metadata")
        .method("save_model",
                static_cast<Rcpp::List (NMFDecomposition::*)()>(
                    &NMFDecomposition::save_model),
                "Save model state")
        .method("load_model",
                static_cast<void (NMFDecomposition::*)(Rcpp::List)>(
                    &NMFDecomposition::load_model),
                "Load model state");
}

// ---------------------------------------------------------------------------
// t-SNE
// ---------------------------------------------------------------------------
RCPP_MODULE(tsne_module) {
    class_<TSNE>("rccp_TSNE")
        .constructor<>("Initialize t-SNE with defaults")
        .constructor<da_int, da_int, std::string, da_int, bool, bool, da_int>(
            "Initialize t-SNE")
        .method("fit", &TSNE::fit, "Fit t-SNE model")
        .method("fit_transform", &TSNE::fit_transform,
                "Fit t-SNE model and return embedding")
        .method("get_embedding", &TSNE::get_embedding, "Get embedding")
        .method("get_n_samples", &TSNE::get_n_samples, "Get number of samples")
        .method("get_n_features", &TSNE::get_n_features, "Get number of features")
        .method("get_n_components", &TSNE::get_n_components, "Get number of components")
        .method("get_n_iter", &TSNE::get_n_iter, "Get number of iterations")
        .method("get_lp_n_iter", &TSNE::get_lp_n_iter,
                "Get number of low precision iterations")
        .method("get_kl_divergence", &TSNE::get_kl_divergence, "Get final KL divergence")
        .method("print_model_versions",
                static_cast<void (TSNE::*)()>(&rda_handle::print_model_versions),
                "Print model version metadata")
        .method("save_model", static_cast<Rcpp::List (TSNE::*)()>(&TSNE::save_model),
                "Save model state")
        .method("load_model", static_cast<void (TSNE::*)(Rcpp::List)>(&TSNE::load_model),
                "Load model state");
}

// ---------------------------------------------------------------------------
// Kernel functions
// ---------------------------------------------------------------------------
RCPP_MODULE(kernel_module) {
    function("rcpp_rbf_kernel", &da_rbf_kernel_wrapper, "Compute RBF kernel matrix");
    function("rcpp_linear_kernel", &da_linear_kernel_wrapper,
             "Compute linear kernel matrix");
    function("rcpp_polynomial_kernel", &da_polynomial_kernel_wrapper,
             "Compute polynomial kernel matrix");
    function("rcpp_sigmoid_kernel", &da_sigmoid_kernel_wrapper,
             "Compute sigmoid kernel matrix");
}

// ---------------------------------------------------------------------------
// k-means
// ---------------------------------------------------------------------------
RCPP_MODULE(kmeans_module) {
    class_<KMeans>("rccp_kmeans")
        .constructor<>("Initialize KMeans with defaults")
        .constructor<da_int, std::string, da_int, da_int, da_int, std::string,
                     std::string, bool, std::string, da_int, bool, da_int, bool>(
            "Initialize KMeans")
        .method("fit", &KMeans::fit, "Fit the k-means model")
        .method("predict", &KMeans::predict, "Predict cluster assignments")
        .method("transform", &KMeans::transform,
                "Transform data to distances from cluster centers")
        .method("get_cluster_centers", &KMeans::get_cluster_centers,
                "Get cluster centers")
        .method("get_inertia", &KMeans::get_inertia, "Get within-cluster sum of squares")
        .method("get_n_iter", &KMeans::get_n_iter, "Get number of iterations")
        .method("get_lp_n_iter", &KMeans::get_lp_n_iter,
                "Get number of low precision iterations")
        .method("get_n_features", &KMeans::get_n_features, "Get number of features")
        .method("get_n_samples", &KMeans::get_n_samples, "Get number of samples")
        .method("get_labels", &KMeans::get_labels, "Get cluster labels")
        .method("get_n_clusters", &KMeans::get_n_clusters, "Get number of clusters")
        .method("print_model_versions",
                static_cast<void (KMeans::*)()>(&rda_handle::print_model_versions),
                "Print model version metadata")
        .method("save_model", static_cast<Rcpp::List (KMeans::*)()>(&KMeans::save_model),
                "Save model state")
        .method("load_model",
                static_cast<void (KMeans::*)(Rcpp::List)>(&KMeans::load_model),
                "Load model state");
}

// ---------------------------------------------------------------------------
// Cubic spline interpolation
// ---------------------------------------------------------------------------
RCPP_MODULE(interpolation_module) {
    class_<CubicSpline>("rccp_cubic_spline")
        .constructor<std::string, bool>("Initialize cubic spline interpolation")
        .method("fit", &CubicSpline::fit, "Fit the cubic spline")
        .method("evaluate", &CubicSpline::evaluate,
                "Evaluate the spline or its derivatives")
        .method("search_cells", &CubicSpline::search_cells,
                "Find the cell containing each query point")
        .method("get_coefficients", &CubicSpline::get_coefficients,
                "Get piecewise polynomial coefficients")
        .method("get_n_sites", &CubicSpline::get_n_sites,
                "Get number of interpolation sites")
        .method("get_dim", &CubicSpline::get_dim, "Get dimensionality of the values");
}

// ---------------------------------------------------------------------------
// Linear models
// ---------------------------------------------------------------------------
RCPP_MODULE(linmod_module) {
    class_<LinearModel>("rccp_LinearModel")
        .constructor<std::string, Nullable<da_int>, bool, std::string, std::string,
                     std::string, bool, bool, Nullable<da_int>, bool, Nullable<da_int>,
                     Nullable<da_int>, Nullable<da_int>>("Initialize LinearModel")
        .method("fit", &LinearModel::fit, "Fit the linear model")
        .method("predict", &LinearModel::predict, "Predict using the fitted model")
        .method("get_model_trained", &LinearModel::get_model_trained,
                "Check if model is trained")
        .method("get_coef", &LinearModel::get_coef, "Get model coefficients")
        .method("get_dual_coef", &LinearModel::get_dual_coef, "Get dual coefficients")
        .method("get_loss", &LinearModel::get_loss, "Get final loss value")
        .method("get_time", &LinearModel::get_time, "Get computation time")
        .method("get_n_iter", &LinearModel::get_n_iter, "Get number of iterations")
        .method("get_lp_n_iter", &LinearModel::get_lp_n_iter,
                "Get number of low precision iterations")
        .method("get_norm_gradient_loss", &LinearModel::get_norm_gradient_loss,
                "Get norm gradient loss")
        .method("print_model_versions",
                static_cast<void (LinearModel::*)()>(&rda_handle::print_model_versions),
                "Print model version metadata")
        .method("save_model",
                static_cast<Rcpp::List (LinearModel::*)()>(&LinearModel::save_model),
                "Save model state")
        .method("load_model",
                static_cast<void (LinearModel::*)(Rcpp::List)>(&LinearModel::load_model),
                "Load model state");
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------
RCPP_MODULE(metrics_module) {
    function("rcpp_pairwise_distances", &da_pairwise_distances_wrapper,
             "Compute pairwise distances between rows of X (and optionally Y)");
}

// ---------------------------------------------------------------------------
// Nearest neighbors
// ---------------------------------------------------------------------------
RCPP_MODULE(knn_module) {
    class_<NearestNeighbors>("rccp_NearestNeighbors")
        .constructor<>("Initialize NearestNeighbors with defaults")
        .constructor<da_int, std::string, da_int, std::string, std::string, bool>(
            "Initialize NearestNeighbors")
        .method("fit", &NearestNeighbors::fit, "Fit the nearest neighbors model")
        .method("set_labels", &NearestNeighbors::set_labels,
                "Set labels for classification")
        .method("set_targets", &NearestNeighbors::set_targets,
                "Set targets for regression")
        .method("set_outlier_info", &NearestNeighbors::set_outlier_info,
                "Set outlier handling options for radius search")
        .method("kneighbors", &NearestNeighbors::kneighbors, "Find k nearest neighbors")
        .method("kneighbors_indices", &NearestNeighbors::kneighbors_indices,
                "Find k nearest neighbor indices")
        .method("classifier_predict", &NearestNeighbors::classifier_predict,
                "Predict class labels")
        .method("classifier_predict_proba", &NearestNeighbors::classifier_predict_proba,
                "Predict class probabilities")
        .method("regressor_predict", &NearestNeighbors::regressor_predict,
                "Predict target values")
        .method("radius_neighbors", &NearestNeighbors::radius_neighbors,
                "Find all neighbors within a given radius")
        .method("radius_neighbors_indices", &NearestNeighbors::radius_neighbors_indices,
                "Find indices of all neighbors within a given radius")
        .method(
            "print_model_versions",
            static_cast<void (NearestNeighbors::*)()>(&rda_handle::print_model_versions),
            "Print model version metadata")
        .method("save_model",
                static_cast<Rcpp::List (NearestNeighbors::*)()>(
                    &NearestNeighbors::save_model),
                "Save model state")
        .method("load_model",
                static_cast<void (NearestNeighbors::*)(Rcpp::List)>(
                    &NearestNeighbors::load_model),
                "Load model state");
}

// ---------------------------------------------------------------------------
// Nonlinear least squares
// ---------------------------------------------------------------------------
RCPP_MODULE(nlls_module) {
    class_<NLLSModel>("rcpp_NLLSModel")
        .constructor<da_int, da_int, Nullable<NumericVector>, Nullable<NumericVector>,
                     Nullable<NumericVector>, std::string, std::string, std::string,
                     std::string, std::string, int, bool>("Initialize NLLSModel")
        .method("fit", &NLLSModel::fit, "Fit nonlinear least squares model")
        .method("get_info", &NLLSModel::get_info, "Retrieve all solver info after fit")
        .method("get_info_iter", &NLLSModel::get_info_iter,
                "Retrieve iteration count after fit")
        .method("get_info_evals", &NLLSModel::get_info_evals,
                "Retrieve function evaluation counts after fit")
        .method("get_info_optim", &NLLSModel::get_info_optim,
                "Retrieve optimality metrics after fit");
}

// ---------------------------------------------------------------------------
// SVMs
// ---------------------------------------------------------------------------
RCPP_MODULE(svm_module) {
    class_<SVC>("rccp_SVC")
        .constructor<>("Initialize SVC with defaults")
        .constructor<std::string, da_int, da_int, da_int, da_int, da_int, bool, da_int,
                     bool>("Initialize SVC")
        .method("fit", &SVC::fit, "Fit the SVC model")
        .method("predict",
                static_cast<NumericVector (SVC::*)(NumericMatrix)>(&BaseSVM::predict),
                "Predict using the SVC model")
        .method(
            "score",
            static_cast<double (SVC::*)(NumericMatrix, NumericVector)>(&BaseSVM::score),
            "Score the SVC model")
        .method("decision_function",
                static_cast<NumericMatrix (SVC::*)(NumericMatrix, std::string)>(
                    &BaseSVM::decision_function),
                "Decision function of the SVC model")
        .method(
            "predict_proba",
            static_cast<NumericMatrix (SVC::*)(NumericMatrix)>(&BaseSVM::predict_proba),
            "Predict class probabilities")
        .method("predict_log_proba",
                static_cast<NumericMatrix (SVC::*)(NumericMatrix)>(
                    &BaseSVM::predict_log_proba),
                "Predict log class probabilities")
        .method("get_n_classes", static_cast<da_int (SVC::*)()>(&BaseSVM::get_n_classes),
                "Get number of classes")
        .method("get_n_samples", static_cast<da_int (SVC::*)()>(&BaseSVM::get_n_samples),
                "Get number of samples")
        .method("get_n_features",
                static_cast<da_int (SVC::*)()>(&BaseSVM::get_n_features),
                "Get number of features")
        .method("get_n_sv", static_cast<da_int (SVC::*)()>(&BaseSVM::get_n_sv),
                "Get number of support vectors")
        .method("get_n_sv_per_class",
                static_cast<IntegerVector (SVC::*)()>(&BaseSVM::get_n_sv_per_class),
                "Get support vectors per class")
        .method("get_bias", static_cast<NumericVector (SVC::*)()>(&BaseSVM::get_bias),
                "Get bias/intercept")
        .method("get_probA", static_cast<NumericVector (SVC::*)()>(&BaseSVM::get_probA),
                "Get probability parameter A")
        .method("get_probB", static_cast<NumericVector (SVC::*)()>(&BaseSVM::get_probB),
                "Get probability parameter B")
        .method("get_n_iterations",
                static_cast<IntegerVector (SVC::*)()>(&BaseSVM::get_n_iterations),
                "Get number of iterations")
        .method("get_lp_n_iterations",
                static_cast<IntegerVector (SVC::*)()>(&BaseSVM::get_lp_n_iterations),
                "Get number of low precision iterations")
        .method("get_dual_coefs",
                static_cast<NumericMatrix (SVC::*)()>(&BaseSVM::get_dual_coefs),
                "Get dual coefficients")
        .method("get_sv", static_cast<NumericMatrix (SVC::*)()>(&BaseSVM::get_sv),
                "Get support vectors")
        .method("get_support_vectors_idx",
                static_cast<IntegerVector (SVC::*)()>(&BaseSVM::get_support_vectors_idx),
                "Get support vector indices")
        .method("print_model_versions",
                static_cast<void (SVC::*)()>(&rda_handle::print_model_versions),
                "Print model version metadata")
        .method("save_model", static_cast<Rcpp::List (SVC::*)()>(&SVC::save_model),
                "Save model state")
        .method("load_model", static_cast<void (SVC::*)(Rcpp::List)>(&SVC::load_model),
                "Load model state");

    class_<SVR>("rccp_SVR")
        .constructor<>("Initialize SVR with defaults")
        .constructor<std::string, da_int, da_int, da_int, bool, da_int, bool>(
            "Initialize SVR")
        .method("fit", &SVR::fit, "Fit the SVR model")
        .method("predict",
                static_cast<NumericVector (SVR::*)(NumericMatrix)>(&BaseSVM::predict),
                "Predict using the SVR model")
        .method(
            "score",
            static_cast<double (SVR::*)(NumericMatrix, NumericVector)>(&BaseSVM::score),
            "Score the SVR model")
        .method("get_n_classes", static_cast<da_int (SVR::*)()>(&BaseSVM::get_n_classes),
                "Get number of classes")
        .method("get_n_samples", static_cast<da_int (SVR::*)()>(&BaseSVM::get_n_samples),
                "Get number of samples")
        .method("get_n_features",
                static_cast<da_int (SVR::*)()>(&BaseSVM::get_n_features),
                "Get number of features")
        .method("get_n_sv", static_cast<da_int (SVR::*)()>(&BaseSVM::get_n_sv),
                "Get number of support vectors")
        .method("get_n_sv_per_class",
                static_cast<IntegerVector (SVR::*)()>(&BaseSVM::get_n_sv_per_class),
                "Get support vectors per class")
        .method("get_bias", static_cast<NumericVector (SVR::*)()>(&BaseSVM::get_bias),
                "Get bias/intercept")
        .method("get_n_iterations",
                static_cast<IntegerVector (SVR::*)()>(&BaseSVM::get_n_iterations),
                "Get number of iterations")
        .method("get_lp_n_iterations",
                static_cast<IntegerVector (SVR::*)()>(&BaseSVM::get_lp_n_iterations),
                "Get number of low precision iterations")
        .method("get_dual_coefs",
                static_cast<NumericMatrix (SVR::*)()>(&BaseSVM::get_dual_coefs),
                "Get dual coefficients")
        .method("get_sv", static_cast<NumericMatrix (SVR::*)()>(&BaseSVM::get_sv),
                "Get support vectors")
        .method("get_support_vectors_idx",
                static_cast<IntegerVector (SVR::*)()>(&BaseSVM::get_support_vectors_idx),
                "Get support vector indices")
        .method("decision_function",
                static_cast<NumericMatrix (SVR::*)(NumericMatrix, std::string)>(
                    &BaseSVM::decision_function),
                "Decision function of the SVR model")
        .method("print_model_versions",
                static_cast<void (SVR::*)()>(&rda_handle::print_model_versions),
                "Print model version metadata")
        .method("save_model", static_cast<Rcpp::List (SVR::*)()>(&SVR::save_model),
                "Save model state")
        .method("load_model", static_cast<void (SVR::*)(Rcpp::List)>(&SVR::load_model),
                "Load model state");

    class_<NuSVC>("rccp_NuSVC")
        .constructor<>("Initialize NuSVC with defaults")
        .constructor<std::string, da_int, da_int, da_int, da_int, da_int, bool, da_int,
                     bool>("Initialize NuSVC")
        .method("fit", &NuSVC::fit, "Fit the NuSVC model")
        .method("predict",
                static_cast<NumericVector (NuSVC::*)(NumericMatrix)>(&BaseSVM::predict),
                "Predict using the NuSVC model")
        .method(
            "score",
            static_cast<double (NuSVC::*)(NumericMatrix, NumericVector)>(&BaseSVM::score),
            "Score the NuSVC model")
        .method("decision_function",
                static_cast<NumericMatrix (NuSVC::*)(NumericMatrix, std::string)>(
                    &BaseSVM::decision_function),
                "Decision function of the NuSVC model")
        .method(
            "predict_proba",
            static_cast<NumericMatrix (NuSVC::*)(NumericMatrix)>(&BaseSVM::predict_proba),
            "Predict class probabilities")
        .method("predict_log_proba",
                static_cast<NumericMatrix (NuSVC::*)(NumericMatrix)>(
                    &BaseSVM::predict_log_proba),
                "Predict log class probabilities")
        .method("get_n_classes",
                static_cast<da_int (NuSVC::*)()>(&BaseSVM::get_n_classes),
                "Get number of classes")
        .method("get_n_samples",
                static_cast<da_int (NuSVC::*)()>(&BaseSVM::get_n_samples),
                "Get number of samples")
        .method("get_n_features",
                static_cast<da_int (NuSVC::*)()>(&BaseSVM::get_n_features),
                "Get number of features")
        .method("get_n_sv", static_cast<da_int (NuSVC::*)()>(&BaseSVM::get_n_sv),
                "Get number of support vectors")
        .method("get_n_sv_per_class",
                static_cast<IntegerVector (NuSVC::*)()>(&BaseSVM::get_n_sv_per_class),
                "Get support vectors per class")
        .method("get_bias", static_cast<NumericVector (NuSVC::*)()>(&BaseSVM::get_bias),
                "Get bias/intercept")
        .method("get_probA", static_cast<NumericVector (NuSVC::*)()>(&BaseSVM::get_probA),
                "Get probability parameter A")
        .method("get_probB", static_cast<NumericVector (NuSVC::*)()>(&BaseSVM::get_probB),
                "Get probability parameter B")
        .method("get_n_iterations",
                static_cast<IntegerVector (NuSVC::*)()>(&BaseSVM::get_n_iterations),
                "Get number of iterations")
        .method("get_lp_n_iterations",
                static_cast<IntegerVector (NuSVC::*)()>(&BaseSVM::get_lp_n_iterations),
                "Get number of low precision iterations")
        .method("get_dual_coefs",
                static_cast<NumericMatrix (NuSVC::*)()>(&BaseSVM::get_dual_coefs),
                "Get dual coefficients")
        .method("get_sv", static_cast<NumericMatrix (NuSVC::*)()>(&BaseSVM::get_sv),
                "Get support vectors")
        .method(
            "get_support_vectors_idx",
            static_cast<IntegerVector (NuSVC::*)()>(&BaseSVM::get_support_vectors_idx),
            "Get support vector indices")
        .method("print_model_versions",
                static_cast<void (NuSVC::*)()>(&rda_handle::print_model_versions),
                "Print model version metadata")
        .method("save_model", static_cast<Rcpp::List (NuSVC::*)()>(&NuSVC::save_model),
                "Save model state")
        .method("load_model",
                static_cast<void (NuSVC::*)(Rcpp::List)>(&NuSVC::load_model),
                "Load model state");

    class_<NuSVR>("rccp_NuSVR")
        .constructor<>("Initialize NuSVR with defaults")
        .constructor<std::string, da_int, da_int, da_int, bool, da_int, bool>(
            "Initialize NuSVR")
        .method("fit", &NuSVR::fit, "Fit the NuSVR model")
        .method("predict",
                static_cast<NumericVector (NuSVR::*)(NumericMatrix)>(&BaseSVM::predict),
                "Predict using the NuSVR model")
        .method(
            "score",
            static_cast<double (NuSVR::*)(NumericMatrix, NumericVector)>(&BaseSVM::score),
            "Score the NuSVR model")
        .method("get_n_classes",
                static_cast<da_int (NuSVR::*)()>(&BaseSVM::get_n_classes),
                "Get number of classes")
        .method("get_n_samples",
                static_cast<da_int (NuSVR::*)()>(&BaseSVM::get_n_samples),
                "Get number of samples")
        .method("get_n_features",
                static_cast<da_int (NuSVR::*)()>(&BaseSVM::get_n_features),
                "Get number of features")
        .method("get_n_sv", static_cast<da_int (NuSVR::*)()>(&BaseSVM::get_n_sv),
                "Get number of support vectors")
        .method("get_n_sv_per_class",
                static_cast<IntegerVector (NuSVR::*)()>(&BaseSVM::get_n_sv_per_class),
                "Get support vectors per class")
        .method("get_bias", static_cast<NumericVector (NuSVR::*)()>(&BaseSVM::get_bias),
                "Get bias/intercept")
        .method("get_n_iterations",
                static_cast<IntegerVector (NuSVR::*)()>(&BaseSVM::get_n_iterations),
                "Get number of iterations")
        .method("get_lp_n_iterations",
                static_cast<IntegerVector (NuSVR::*)()>(&BaseSVM::get_lp_n_iterations),
                "Get number of low precision iterations")
        .method("get_dual_coefs",
                static_cast<NumericMatrix (NuSVR::*)()>(&BaseSVM::get_dual_coefs),
                "Get dual coefficients")
        .method("get_sv", static_cast<NumericMatrix (NuSVR::*)()>(&BaseSVM::get_sv),
                "Get support vectors")
        .method(
            "get_support_vectors_idx",
            static_cast<IntegerVector (NuSVR::*)()>(&BaseSVM::get_support_vectors_idx),
            "Get support vector indices")
        .method("decision_function",
                static_cast<NumericMatrix (NuSVR::*)(NumericMatrix, std::string)>(
                    &BaseSVM::decision_function),
                "Decision function of the NuSVR model")
        .method("print_model_versions",
                static_cast<void (NuSVR::*)()>(&rda_handle::print_model_versions),
                "Print model version metadata")
        .method("save_model", static_cast<Rcpp::List (NuSVR::*)()>(&NuSVR::save_model),
                "Save model state")
        .method("load_model",
                static_cast<void (NuSVR::*)(Rcpp::List)>(&NuSVR::load_model),
                "Load model state");
}

// ---------------------------------------------------------------------------
// Decision tree
// ---------------------------------------------------------------------------
RCPP_MODULE(decision_tree_module) {
    class_<DecisionTree>("rccp_DecisionTree")
        .constructor<>("Initialize Decision Tree with defaults")
        .constructor<int, int, int, std::string, int, std::string, bool, int, bool, bool>(
            "Initialize Decision Tree")
        .method("fit", &DecisionTree::fit, "Fit decision tree")
        .method("set_max_features_opt", &DecisionTree::set_max_features_opt,
                "Set maximum features")
        .method("get_max_features_opt", &DecisionTree::get_max_features_opt,
                "Get maximum features option")
        .method("predict", &DecisionTree::predict, "Predict using fitted decision tree")
        .method("predict_proba", &DecisionTree::predict_proba,
                "Predict class probabilities")
        .method("predict_log_proba", &DecisionTree::predict_log_proba,
                "Predict log class probabilities")
        .method("score", &DecisionTree::score, "Score the model")
        .method("get_model_info", &DecisionTree::get_model_info, "Get model information")
        .method("print_model_versions",
                static_cast<void (DecisionTree::*)()>(&rda_handle::print_model_versions),
                "Print model version metadata")
        .method("save_model",
                static_cast<Rcpp::List (DecisionTree::*)()>(&DecisionTree::save_model),
                "Save model state")
        .method(
            "load_model",
            static_cast<void (DecisionTree::*)(Rcpp::List)>(&DecisionTree::load_model),
            "Load model state");
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------
RCPP_MODULE(utils_module) {
    function("rcpp_train_test_split", &train_test_split,
             "Unified train-test split for both numeric and integer data");
    function("rcpp_get_shuffled_indices", &get_shuffled_indices,
             "Unified shuffled indices with precision parameter");
    function("rcpp_print_model_metadata", &print_model_metadata_from_raw,
             "Print model metadata from raw serialized buffer");
}
