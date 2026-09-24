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

#ifndef kNN_R_HPP
#define kNN_R_HPP

#include "internal_utilities_r.hpp"
#include <Rcpp.h>
#include <aoclda.h>

using namespace Rcpp;

// Helper: parse search mode string to C API enum
inline da_nn_search_mode parse_search_mode(const std::string &mode) {
    if (mode == "knn")
        return knn_search_mode;
    if (mode == "radius_neighbors")
        return radius_search_mode;
    Rcpp::stop("Invalid search_mode. Supported modes are 'knn' and 'radius_neighbors'.");
}

class NearestNeighbors : public rda_handle {
  private:
    da_int internal_neigh;
    double internal_radius;

  public:
    NearestNeighbors(da_int n_neighbors = 5, std::string algorithm = "auto",
                     da_int leaf_size = 30, std::string metric = "euclidean",
                     std::string weights = "uniform", bool check_data = false) {

        da_status status = da_handle_init_d(&handle, da_handle_nn);
        exception_check(status);

        status = da_options_set(handle, "number of neighbors", n_neighbors);
        exception_check(status);
        internal_neigh = n_neighbors;

        std::string algo = algorithm;
        if (algorithm == "kd_tree")
            algo = "kd tree";
        if (algorithm == "ball_tree")
            algo = "ball tree";
        status = da_options_set(handle, "algorithm", algo.c_str());
        exception_check(status);
        std::string met = metric;
        if (metric == "inner_product")
            met = "inner product";
        status = da_options_set(handle, "metric", met.c_str());
        exception_check(status);
        status = da_options_set(handle, "weights", weights.c_str());
        exception_check(status);
        status = da_options_set(handle, "leaf size", leaf_size);
        exception_check(status);

        if (check_data) {
            status = da_options_set(handle, "check data", "yes");
            exception_check(status);
        }
    }

    void fit(NumericMatrix X, double p, double radius) {
        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        da_status status = da_options_set(handle, "storage order", "column-major");
        exception_check(status);

        status = da_options_set(handle, "minkowski parameter", p);
        exception_check(status);

        status = da_options_set(handle, "radius", radius);
        exception_check(status);

        this->internal_radius = radius;

        status = da_nn_set_data_d(handle, n_samples, n_features, X.begin(), n_samples);
        exception_check(status);
    }

    void set_labels(IntegerVector y) {
        da_int n_samples = y.length();
        da_status status = da_nn_set_labels_d(handle, n_samples, y.begin());
        exception_check(status);
    }

    void set_outlier_info(const std::string &outlier_handling = "none",
                          da_int outlier_label = 0, double outlier_target = 0.0) {
        std::string outlier_handling_str = outlier_handling;
        if (outlier_handling == "most_frequent")
            outlier_handling_str = "most frequent";
        da_status status =
            da_options_set(handle, "outlier handling", outlier_handling_str.c_str());
        exception_check(status);
        status = da_options_set(handle, "outlier label", outlier_label);
        exception_check(status);
        status = da_options_set(handle, "outlier target", outlier_target);
        exception_check(status);
    }

    void set_targets(NumericVector y) {
        da_int n_samples = y.length();
        da_status status = da_nn_set_targets_d(handle, n_samples, y.begin());
        exception_check(status);
    }

    IntegerMatrix kneighbors_indices(NumericMatrix X, da_int n_neighbors = 0) {
        da_int n_queries = X.nrow();
        da_int n_features = X.ncol();
        da_int req_neigh = (n_neighbors != 0) ? n_neighbors : internal_neigh;

        IntegerMatrix indices(n_queries, req_neigh);

        da_status status =
            da_nn_kneighbors_d(handle, n_queries, n_features, X.begin(), n_queries,
                               indices.begin(), nullptr, req_neigh, 0);
        exception_check(status);

        return indices;
    }

    List kneighbors(NumericMatrix X, da_int n_neighbors = 0) {
        da_int n_queries = X.nrow();
        da_int n_features = X.ncol();
        da_int req_neigh = (n_neighbors != 0) ? n_neighbors : internal_neigh;

        NumericMatrix distances(n_queries, req_neigh);
        IntegerMatrix indices(n_queries, req_neigh);

        da_status status =
            da_nn_kneighbors_d(handle, n_queries, n_features, X.begin(), n_queries,
                               indices.begin(), distances.begin(), req_neigh, 1);
        exception_check(status);

        return List::create(Named("distances") = distances, Named("indices") = indices);
    }

    NumericMatrix classifier_predict_proba(NumericMatrix X,
                                           std::string search_mode = "knn") {
        da_int n_queries = X.nrow();
        da_int n_features = X.ncol();
        da_int num_classes = -3;
        da_status status = da_nn_classes_d(handle, &num_classes, nullptr);
        exception_check(status);
        NumericMatrix probabilities(n_queries, num_classes);
        status = da_nn_classifier_predict_proba_d(
            handle, n_queries, n_features, X.begin(), n_queries, probabilities.begin(),
            parse_search_mode(search_mode));
        exception_check(status);
        return probabilities;
    }

    IntegerVector classifier_predict(NumericMatrix X, std::string search_mode = "knn") {
        da_int n_queries = X.nrow();
        da_int n_features = X.ncol();
        IntegerVector predictions(n_queries);
        da_status status = da_nn_classifier_predict_d(
            handle, n_queries, n_features, X.begin(), n_queries, predictions.begin(),
            parse_search_mode(search_mode));
        exception_check(status);
        return predictions;
    }

    NumericVector regressor_predict(NumericMatrix X, std::string search_mode = "knn") {
        da_int n_queries = X.nrow();
        da_int n_features = X.ncol();
        NumericVector predictions(n_queries);
        da_status status = da_nn_regressor_predict_d(
            handle, n_queries, n_features, X.begin(), n_queries, predictions.begin(),
            parse_search_mode(search_mode));
        exception_check(status);
        return predictions;
    }

    List radius_neighbors_indices(NumericMatrix X, double radius = 0.0) {
        da_int n_queries = X.nrow();
        da_int n_features = X.ncol();
        double req_radius = (radius != 0.0) ? radius : internal_radius;
        da_status status = da_nn_radius_neighbors_d(
            handle, n_queries, n_features, X.begin(), n_queries, req_radius, 0, 0);
        exception_check(status);

        da_int array_size = n_queries + 1;
        std::vector<da_int> n_radius_neighbors(array_size);
        status = da_handle_get_result_int(handle, da_nn_radius_neighbors_count,
                                          &array_size, n_radius_neighbors.data());
        exception_check(status);

        List result(n_queries);
        for (da_int i = 0; i < n_queries; ++i) {
            IntegerVector r_neigh_i(n_radius_neighbors[i]);
            if (n_radius_neighbors[i] > 0) {
                r_neigh_i[0] = i;
                status =
                    da_handle_get_result_int(handle, da_nn_radius_neighbors_indices_index,
                                             &n_radius_neighbors[i], r_neigh_i.begin());
                exception_check(status);
            }
            result[i] = r_neigh_i;
        }
        return result;
    }

    List radius_neighbors(NumericMatrix X, double radius = 0.0,
                          bool sort_results = false) {
        da_int n_queries = X.nrow();
        da_int n_features = X.ncol();
        double req_radius = (radius != 0.0) ? radius : internal_radius;

        da_int sort_res = 0;
        if (sort_results)
            sort_res = 1;

        da_status status = da_nn_radius_neighbors_d(
            handle, n_queries, n_features, X.begin(), n_queries, req_radius, 1, sort_res);
        exception_check(status);

        da_int array_size = n_queries + 1;
        std::vector<da_int> n_radius_neighbors(array_size);
        status = da_handle_get_result_int(handle, da_nn_radius_neighbors_count,
                                          &array_size, n_radius_neighbors.data());
        exception_check(status);

        List result(n_queries);
        for (da_int i = 0; i < n_queries; ++i) {
            IntegerVector r_neigh_i(n_radius_neighbors[i]);
            NumericVector r_neigh_d(n_radius_neighbors[i]);
            if (n_radius_neighbors[i] > 0) {
                r_neigh_i[0] = i;
                status =
                    da_handle_get_result_int(handle, da_nn_radius_neighbors_indices_index,
                                             &n_radius_neighbors[i], r_neigh_i.begin());
                exception_check(status);
                r_neigh_d[0] = (double)i;
                status =
                    da_handle_get_result_d(handle, da_nn_radius_neighbors_distances_index,
                                           &n_radius_neighbors[i], r_neigh_d.begin());
                exception_check(status);
            }
            result[i] = List::create(Named("distances") = r_neigh_d,
                                     Named("indices") = r_neigh_i);
        }
        return result;
    }

    void save_data(Rcpp::List &state) override {
        state["internal_neigh"] = (int)internal_neigh;
        state["internal_radius"] = internal_radius;
    }

    void load_data(Rcpp::List &state) override {
        internal_neigh = Rcpp::as<da_int>(state["internal_neigh"]);
        internal_radius = Rcpp::as<double>(state["internal_radius"]);
    }
};

#endif // kNN_R_HPP
