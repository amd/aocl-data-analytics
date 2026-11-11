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

#ifndef KMEANS_R_HPP
#define KMEANS_R_HPP

#include <Rcpp.h>
#include <vector>

#include "aoclda.h"
#include "internal_utilities_r.hpp"

using namespace Rcpp;

class KMeans : public rda_handle {
  public:
    // Constructor
    KMeans(da_int n_clusters = 1, std::string initialization_method = "k-means++",
           da_int n_init = 10, da_int max_iter = 300, da_int seed = -1,
           std::string algorithm = "lloyd", std::string distance = "euclidean",
           bool normalize_data = false, std::string empty_clusters = "ignore",
           da_int afk_mcmc_samples = 50, bool mixed_precision = false,
           da_int low_precision_max_iter = 200, bool check_data = false) {

        da_status status = da_handle_init_d(&handle, da_handle_kmeans);
        exception_check(status);
        status = da_options_set(handle, "n_clusters", n_clusters);
        exception_check(status);
        status = da_options_set(handle, "algorithm", algorithm.c_str());
        exception_check(status);
        status = da_options_set(handle, "distance", distance.c_str());
        exception_check(status);
        if (normalize_data) {
            status = da_options_set(handle, "normalize data", "yes");
            exception_check(status);
        }
        status = da_options_set(handle, "empty clusters", empty_clusters.c_str());
        exception_check(status);
        status = da_options_set(handle, "afk-mc2 samples", afk_mcmc_samples);
        exception_check(status);
        status = da_options_set(handle, "initialization method",
                                initialization_method.c_str());
        exception_check(status);
        status = da_options_set(handle, "max_iter", max_iter);
        exception_check(status);
        status = da_options_set(handle, "low precision max_iter", low_precision_max_iter);
        exception_check(status);
        status = da_options_set(handle, "seed", seed);
        exception_check(status);
        status = da_options_set(handle, "n_init", n_init);
        exception_check(status);
        if (mixed_precision) {
            status = da_options_set(handle, "mixed precision", "yes");
            exception_check(status);
        }
        if (check_data) {
            status = da_options_set(handle, "check data", "yes");
            exception_check(status);
        }
    }

    void fit(NumericMatrix A, Nullable<NumericMatrix> C, double tol = 1.0e-4,
             double low_prec_tol = 1.0e-2) {
        da_status status = da_options_set(handle, "convergence tolerance", tol);
        exception_check(status);
        status =
            da_options_set(handle, "low precision convergence tolerance", low_prec_tol);
        exception_check(status);

        da_int n_samples = A.nrow();
        da_int n_features = A.ncol();

        status = da_options_set(handle, "storage order", "column-major");
        exception_check(status);

        status =
            da_kmeans_set_data_d(handle, n_samples, n_features, A.begin(), n_samples);
        exception_check(status);

        if (C.isNotNull()) {
            const NumericMatrix &centers_matrix = C.get();
            status = da_options_set(handle, "initialization method", "supplied");
            exception_check(status);
            status = da_kmeans_set_init_centres_d(handle, centers_matrix.begin(),
                                                  centers_matrix.nrow());
            exception_check(status);
        }

        status = da_kmeans_compute_d(handle);
        exception_check(status);
    }

    NumericMatrix transform(NumericMatrix X) {
        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        da_int n_clusters = get_n_clusters();

        NumericMatrix X_transform(n_samples, n_clusters);
        da_status status =
            da_kmeans_transform_d(handle, n_samples, n_features, X.begin(), n_samples,
                                  X_transform.begin(), n_samples);
        exception_check(status);

        return X_transform;
    }

    IntegerVector predict(NumericMatrix Y) {
        da_int n_samples = Y.nrow();
        da_int n_features = Y.ncol();

        IntegerVector Y_labels(n_samples);
        da_status status = da_kmeans_predict_d(handle, n_samples, n_features, Y.begin(),
                                               n_samples, Y_labels.begin());
        exception_check(status);

        return Y_labels;
    }

    // Get results method - mimicking Python bindings exactly
    void get_rinfo(da_int *n_samples, da_int *n_features, da_int *n_clusters,
                   da_int *n_iter, double *inertia, da_int *lp_n_iter = nullptr) {
        da_status status;
        da_int dim = 6;
        double rinfo[6];
        status = da_handle_get_result_d(handle, da_rinfo, &dim, rinfo);
        exception_check(status);

        *n_samples = static_cast<da_int>(rinfo[0]);
        *n_features = static_cast<da_int>(rinfo[1]);
        *n_clusters = static_cast<da_int>(rinfo[2]);
        *n_iter = static_cast<da_int>(rinfo[3]);
        *inertia = rinfo[4];
        if (lp_n_iter)
            *lp_n_iter = static_cast<da_int>(rinfo[5]);
    }

    NumericMatrix get_cluster_centers() {
        da_int n_samples, n_features, n_clusters, n_iter;
        double inertia;
        get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);

        da_int dim = n_clusters * n_features;
        NumericMatrix centers(n_clusters, n_features);
        da_status status = da_handle_get_result_d(handle, da_kmeans_cluster_centres, &dim,
                                                  centers.begin());
        exception_check(status);
        return centers;
    }

    IntegerVector get_labels() {
        da_int n_samples, n_features, n_clusters, n_iter;
        double inertia;
        get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);

        da_int dim = n_samples;
        IntegerVector labels(n_samples);
        da_status status =
            da_handle_get_result_int(handle, da_kmeans_labels, &dim, labels.begin());
        exception_check(status);

        return labels;
    }

    double get_inertia() {
        da_int n_samples, n_features, n_clusters, n_iter;
        double inertia;
        get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);
        return inertia;
    }

    da_int get_n_iter() {
        da_int n_samples, n_features, n_clusters, n_iter;
        double inertia;
        get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);
        return n_iter;
    }

    da_int get_lp_n_iter() {
        da_int n_samples, n_features, n_clusters, n_iter, lp_n_iter;
        double inertia;
        get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia, &lp_n_iter);
        return lp_n_iter;
    }

    da_int get_n_samples() {
        da_int n_samples, n_features, n_clusters, n_iter;
        double inertia;
        get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);
        return n_samples;
    }

    da_int get_n_features() {
        da_int n_samples, n_features, n_clusters, n_iter;
        double inertia;
        get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);
        return n_features;
    }

    da_int get_n_clusters() {
        da_int n_samples, n_features, n_clusters, n_iter;
        double inertia;
        get_rinfo(&n_samples, &n_features, &n_clusters, &n_iter, &inertia);
        return n_clusters;
    }
};

#endif // KMEANS_R_HPP
