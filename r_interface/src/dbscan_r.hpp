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

#ifndef DBSCAN_R_HPP
#define DBSCAN_R_HPP

#include <Rcpp.h>
#include <vector>

#include "aoclda.h"
#include "internal_utilities_r.hpp"

using namespace Rcpp;

class DBSCAN : public rda_handle {
  public:
    DBSCAN(da_int min_samples = 5, std::string metric = "euclidean",
           std::string algorithm = "brute", da_int leaf_size = 30,
           bool check_data = false) {

        da_status status = da_handle_init_d(&handle, da_handle_dbscan);
        exception_check(status);

        status = da_options_set(handle, "leaf size", leaf_size);
        exception_check(status);
        status = da_options_set(handle, "min samples", min_samples);
        exception_check(status);

        std::string algo = algorithm;
        if (algorithm == "kd_tree") {
            algo = "kd tree";
        } else if (algorithm == "ball_tree") {
            algo = "ball tree";
        }

        status = da_options_set(handle, "algorithm", algo.c_str());
        exception_check(status);
        status = da_options_set(handle, "metric", metric.c_str());
        exception_check(status);

        if (check_data) {
            status = da_options_set(handle, "check data", "yes");
            exception_check(status);
        }
    }

    void fit(NumericMatrix X, double eps = 0.5, double power = 2.0) {

        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        da_status status;

        status = da_options_set(handle, "eps", eps);
        exception_check(status);
        status = da_options_set(handle, "power", power);
        exception_check(status);
        status = da_options_set(handle, "storage order", "column-major");
        exception_check(status);

        status =
            da_dbscan_set_data_d(handle, n_samples, n_features, X.begin(), n_samples);
        exception_check(status);

        status = da_dbscan_compute_d(handle);
        exception_check(status);
    }

    void get_rinfo(da_int *n_samples, da_int *n_features, da_int *lda, double *eps,
                   da_int *min_samples, da_int *leaf_size, double *power,
                   da_int *n_core_samples, da_int *n_clusters) {

        da_status status;
        da_int dim = 9;
        double rinfo[9];
        status = da_handle_get_result_d(handle, da_rinfo, &dim, rinfo);
        exception_check(status);

        *n_samples = static_cast<da_int>(rinfo[0]);
        *n_features = static_cast<da_int>(rinfo[1]);
        *lda = static_cast<da_int>(rinfo[2]);
        *eps = rinfo[3];
        *min_samples = static_cast<da_int>(rinfo[4]);
        *leaf_size = static_cast<da_int>(rinfo[5]);
        *power = rinfo[6];
        *n_core_samples = static_cast<da_int>(rinfo[7]);
        *n_clusters = static_cast<da_int>(rinfo[8]);
    }

    IntegerVector get_labels() {
        da_status status;
        da_int n_samples, n_features, lda, min_samples, leaf_size, n_core_samples,
            n_clusters;
        double eps, power;

        get_rinfo(&n_samples, &n_features, &lda, &eps, &min_samples, &leaf_size, &power,
                  &n_core_samples, &n_clusters);

        da_int dim = n_samples;
        IntegerVector labels(dim);
        status = da_handle_get_result_int(handle, da_dbscan_labels, &dim, labels.begin());
        exception_check(status);

        return labels;
    }

    IntegerVector get_core_sample_indices() {

        da_status status;
        da_int n_samples, n_features, lda, min_samples, leaf_size, n_core_samples,
            n_clusters;
        double eps, power;

        get_rinfo(&n_samples, &n_features, &lda, &eps, &min_samples, &leaf_size, &power,
                  &n_core_samples, &n_clusters);

        da_int dim = n_core_samples;
        IntegerVector core_indices(dim);
        status = da_handle_get_result_int(handle, da_dbscan_core_sample_indices, &dim,
                                          core_indices.begin());
        exception_check(status);

        return core_indices;
    }

    da_int get_n_samples() {
        da_int n_samples, n_features, lda, min_samples, leaf_size, n_core_samples,
            n_clusters;
        double eps, power;
        get_rinfo(&n_samples, &n_features, &lda, &eps, &min_samples, &leaf_size, &power,
                  &n_core_samples, &n_clusters);
        return n_samples;
    }

    da_int get_n_core_samples() {
        da_int n_samples, n_features, lda, min_samples, leaf_size, n_core_samples,
            n_clusters;
        double eps, power;
        get_rinfo(&n_samples, &n_features, &lda, &eps, &min_samples, &leaf_size, &power,
                  &n_core_samples, &n_clusters);
        return n_core_samples;
    }

    da_int get_n_features() {
        da_int n_samples, n_features, lda, min_samples, leaf_size, n_core_samples,
            n_clusters;
        double eps, power;
        get_rinfo(&n_samples, &n_features, &lda, &eps, &min_samples, &leaf_size, &power,
                  &n_core_samples, &n_clusters);
        return n_features;
    }

    da_int get_n_clusters() {
        da_int n_samples, n_features, lda, min_samples, leaf_size, n_core_samples,
            n_clusters;
        double eps, power;
        get_rinfo(&n_samples, &n_features, &lda, &eps, &min_samples, &leaf_size, &power,
                  &n_core_samples, &n_clusters);
        return n_clusters;
    }
};

#endif // DBSCAN_R_HPP
