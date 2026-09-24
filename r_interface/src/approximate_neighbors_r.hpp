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

#ifndef APPROX_NN_R_HPP
#define APPROX_NN_R_HPP

#include <Rcpp.h>
#include <vector>

#include "aoclda.h"
#include "internal_utilities_r.hpp"

using namespace Rcpp;

class ApproximateNeighbors : public rda_handle {
  private:
    da_int internal_neigh;

  public:
    ApproximateNeighbors(da_int n_neighbors = 5, std::string algorithm = "auto",
                         std::string metric = "sqeuclidean", da_int n_list = 1,
                         da_int n_probe = 1, da_int kmeans_iter = 10, da_int seed = 0,
                         std::string quantization = "bfloat16",
                         std::string quantization_scheme = "per_dimension",
                         bool quantization_residual = true, bool check_data = false)
        : internal_neigh(n_neighbors) {

        da_status status = da_handle_init_d(&handle, da_handle_approx_nn);
        exception_check(status);

        status = da_options_set(handle, "number of neighbors", n_neighbors);
        exception_check(status);
        status = da_options_set(handle, "algorithm", algorithm.c_str());
        exception_check(status);
        status = da_options_set(handle, "metric", metric.c_str());
        exception_check(status);
        status = da_options_set(handle, "n_list", n_list);
        exception_check(status);
        status = da_options_set(handle, "n_probe", n_probe);
        exception_check(status);
        status = da_options_set(handle, "k-means_iter", kmeans_iter);
        exception_check(status);
        status = da_options_set(handle, "seed", seed);
        exception_check(status);
        status = da_options_set(handle, "quantization", quantization.c_str());
        exception_check(status);
        status =
            da_options_set(handle, "quantization_scheme", quantization_scheme.c_str());
        exception_check(status);
        status = da_options_set(handle, "quantization_residual",
                                quantization_residual ? "yes" : "no");
        exception_check(status);

        if (check_data) {
            status = da_options_set(handle, "check data", "yes");
            exception_check(status);
        }
    }

    void set_n_probe_opt(da_int n_probe) {
        da_status status = da_options_set(handle, "n_probe", n_probe);
        exception_check(status);
    }

    void train(NumericMatrix X, double train_fraction = 1.0) {
        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        da_status status = da_options_set(handle, "train fraction", train_fraction);
        exception_check(status);

        status = da_options_set(handle, "storage order", "column-major");
        exception_check(status);

        status = da_approx_nn_set_training_data_d(handle, n_samples, n_features,
                                                  X.begin(), n_samples);
        exception_check(status);

        status = da_approx_nn_train_d(handle);
        exception_check(status);
    }

    void train_and_add(NumericMatrix X, double train_fraction = 1.0) {
        da_status status = da_options_set(handle, "train fraction", train_fraction);
        exception_check(status);

        status = da_options_set(handle, "storage order", "column-major");
        exception_check(status);

        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        status = da_approx_nn_set_training_data_d(handle, n_samples, n_features,
                                                  X.begin(), n_samples);
        exception_check(status);

        status = da_approx_nn_train_and_add_d(handle);
        exception_check(status);
    }

    void add(NumericMatrix X) {
        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        da_status status =
            da_approx_nn_add_d(handle, n_samples, n_features, X.begin(), n_samples);
        exception_check(status);
    }

    IntegerMatrix kneighbors_indices(NumericMatrix X, da_int n_neighbors = 0) {
        da_int n_queries = X.nrow();
        da_int n_features = X.ncol();
        da_int req_neigh = (n_neighbors != 0) ? n_neighbors : internal_neigh;

        IntegerMatrix k_ind(n_queries, req_neigh);
        da_status status =
            da_approx_nn_kneighbors_d(handle, n_queries, n_features, X.begin(), n_queries,
                                      k_ind.begin(), nullptr, req_neigh, 0);
        exception_check(status);
        return k_ind;
    }

    List kneighbors(NumericMatrix X, da_int n_neighbors = 0) {
        da_int n_queries = X.nrow();
        da_int n_features = X.ncol();
        da_int req_neigh = (n_neighbors != 0) ? n_neighbors : internal_neigh;

        IntegerMatrix k_ind(n_queries, req_neigh);
        NumericMatrix k_dist(n_queries, req_neigh);

        da_status status;
        status =
            da_approx_nn_kneighbors_d(handle, n_queries, n_features, X.begin(), n_queries,
                                      k_ind.begin(), k_dist.begin(), req_neigh, 1);
        exception_check(status);

        return List::create(Named("distances") = k_dist, Named("indices") = k_ind);
    }

    void get_rinfo(da_int *n_list, da_int *n_index, da_int *n_features,
                   da_int *kmeans_iter) {
        da_int dim = 4;
        double rinfo[4];
        da_status status = da_handle_get_result_d(handle, da_rinfo, &dim, rinfo);
        exception_check(status);
        *n_list = (da_int)rinfo[0];
        *n_index = (da_int)rinfo[1];
        *n_features = (da_int)rinfo[2];
        *kmeans_iter = (da_int)rinfo[3];
    }

    NumericMatrix get_cluster_centroids() {
        da_int n_list, n_index, n_features, kmeans_iter;
        get_rinfo(&n_list, &n_index, &n_features, &kmeans_iter);

        da_int dim = n_list * n_features;
        NumericMatrix res(n_list, n_features);
        da_status status = da_handle_get_result_d(handle, da_approx_nn_cluster_centroids,
                                                  &dim, res.begin());
        exception_check(status);
        return res;
    }

    IntegerVector get_list_sizes() {
        da_int n_list, n_index, n_features, kmeans_iter;
        get_rinfo(&n_list, &n_index, &n_features, &kmeans_iter);

        da_int dim = n_list;
        IntegerVector res(n_list);
        da_status status =
            da_handle_get_result_int(handle, da_approx_nn_list_sizes, &dim, res.begin());
        exception_check(status);
        return res;
    }

    da_int get_n_list() {
        da_int n_list, n_index, n_features, kmeans_iter;
        get_rinfo(&n_list, &n_index, &n_features, &kmeans_iter);
        return n_list;
    }

    da_int get_n_index() {
        da_int n_list, n_index, n_features, kmeans_iter;
        get_rinfo(&n_list, &n_index, &n_features, &kmeans_iter);
        return n_index;
    }

    da_int get_n_features() {
        da_int n_list, n_index, n_features, kmeans_iter;
        get_rinfo(&n_list, &n_index, &n_features, &kmeans_iter);
        return n_features;
    }

    da_int get_kmeans_iter() {
        da_int n_list, n_index, n_features, kmeans_iter;
        get_rinfo(&n_list, &n_index, &n_features, &kmeans_iter);
        return kmeans_iter;
    }

    void save_data(Rcpp::List &state) override {
        state["internal_neigh"] = this->internal_neigh;
    }

    void load_data(Rcpp::List &state) override {
        this->internal_neigh = Rcpp::as<da_int>(state["internal_neigh"]);
    }
};

#endif // APPROX_NN_R_HPP
