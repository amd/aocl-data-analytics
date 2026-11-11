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

#ifndef TSNE_R_HPP
#define TSNE_R_HPP

#include <Rcpp.h>
#include <vector>

#include "aoclda.h"
#include "internal_utilities_r.hpp"

using namespace Rcpp;

class TSNE : public rda_handle {
  public:
    TSNE(da_int n_components = 2, da_int max_iter = 1000, std::string init = "pca",
         da_int seed = -1, bool check_data = false, bool mixed_precision = false,
         da_int low_precision_max_iter = 200) {

        da_status status = da_handle_init_d(&handle, da_handle_tsne);
        exception_check(status);

        status = da_options_set_int(handle, "n_components", n_components);
        exception_check(status);
        status = da_options_set_int(handle, "max_iter", max_iter);
        exception_check(status);
        status = da_options_set_string(handle, "init", init.c_str());
        exception_check(status);
        status = da_options_set_int(handle, "seed", seed);
        exception_check(status);
        if (mixed_precision) {
            status = da_options_set(handle, "mixed precision", "yes");
            exception_check(status);
        }
        status = da_options_set_int(handle, "low precision max_iter",
                                    low_precision_max_iter);
        exception_check(status);
        if (check_data) {
            status = da_options_set(handle, "check data", "yes");
            exception_check(status);
        }
    }

    void fit(NumericMatrix X, double perplexity = 30.0, double learning_rate = -1.0,
             double early_exaggeration = 12.0, double theta = 0.5,
             da_int n_iter_without_progress = 300, double min_grad_norm = 1e-7,
             Nullable<NumericMatrix> init_embedding = R_NilValue,
             double low_precision_min_grad_norm = 1e-4) {

        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        da_status status;
        status = da_options_set(handle, "perplexity", perplexity);
        exception_check(status);
        status = da_options_set(handle, "learning rate", learning_rate);
        exception_check(status);
        status = da_options_set(handle, "early exaggeration", early_exaggeration);
        exception_check(status);
        status = da_options_set(handle, "theta", theta);
        exception_check(status);
        status = da_options_set_int(handle, "n_iter_without_progress",
                                    n_iter_without_progress);
        exception_check(status);
        status = da_options_set(handle, "min_grad_norm", min_grad_norm);
        exception_check(status);
        status = da_options_set(handle, "low precision min_grad_norm",
                                low_precision_min_grad_norm);
        exception_check(status);
        status = da_options_set(handle, "storage order", "column-major");
        exception_check(status);

        status = da_tsne_set_data_d(handle, n_samples, n_features, X.begin(), n_samples);
        exception_check(status);

        if (init_embedding.isNotNull()) {
            const NumericMatrix &Y = init_embedding.get();
            status = da_tsne_set_init_embedding_d(handle, Y.begin(), n_samples);
            exception_check(status);
        }

        status = da_tsne_compute_d(handle);
        exception_check(status);
    }

    NumericMatrix fit_transform(NumericMatrix X, double perplexity = 30.0,
                                double learning_rate = -1.0,
                                double early_exaggeration = 12.0,
                                double theta = 0.5,
                                da_int n_iter_without_progress = 300,
                                double min_grad_norm = 1e-7,
                                Nullable<NumericMatrix> init_embedding = R_NilValue,
                                double low_precision_min_grad_norm = 1e-4) {
        fit(X, perplexity, learning_rate, early_exaggeration, theta,
            n_iter_without_progress, min_grad_norm, init_embedding,
            low_precision_min_grad_norm);
        return get_embedding();
    }

    NumericMatrix get_embedding() {
        da_int n_samples, n_features, n_components, n_iter;
        double kl_div;
        da_status status =
            get_rinfo(&n_samples, &n_features, &n_components, &n_iter, &kl_div);
        exception_check(status);

        da_int dim = n_samples * n_components;
        NumericMatrix result(n_samples, n_components);
        status =
            da_handle_get_result_d(handle, da_tsne_embedding, &dim, result.begin());
        exception_check(status);
        return result;
    }

    da_int get_n_samples() {
        da_int n_samples, n_features, n_components, n_iter;
        double kl_div;
        da_status status =
            get_rinfo(&n_samples, &n_features, &n_components, &n_iter, &kl_div);
        exception_check(status);
        return n_samples;
    }

    da_int get_n_features() {
        da_int n_samples, n_features, n_components, n_iter;
        double kl_div;
        da_status status =
            get_rinfo(&n_samples, &n_features, &n_components, &n_iter, &kl_div);
        exception_check(status);
        return n_features;
    }

    da_int get_n_components() {
        da_int n_samples, n_features, n_components, n_iter;
        double kl_div;
        da_status status =
            get_rinfo(&n_samples, &n_features, &n_components, &n_iter, &kl_div);
        exception_check(status);
        return n_components;
    }

    da_int get_n_iter() {
        da_int n_samples, n_features, n_components, n_iter;
        double kl_div;
        da_status status =
            get_rinfo(&n_samples, &n_features, &n_components, &n_iter, &kl_div);
        exception_check(status);
        return n_iter;
    }

    da_int get_lp_n_iter() {
        da_int n_samples, n_features, n_components, n_iter, lp_n_iter;
        double kl_div;
        da_status status =
            get_rinfo(&n_samples, &n_features, &n_components, &n_iter, &kl_div,
                      &lp_n_iter);
        exception_check(status);
        return lp_n_iter;
    }

    double get_kl_divergence() {
        da_int n_samples, n_features, n_components, n_iter;
        double kl_div;
        da_status status =
            get_rinfo(&n_samples, &n_features, &n_components, &n_iter, &kl_div);
        exception_check(status);
        return kl_div;
    }

  private:
    da_status get_rinfo(da_int *n_samples, da_int *n_features, da_int *n_components,
                        da_int *n_iter, double *kl_divergence,
                        da_int *lp_n_iter = nullptr) {
        da_int dim = 6;
        double rinfo[6];
        da_status status = da_handle_get_result_d(handle, da_rinfo, &dim, rinfo);
        if (status != da_status_success)
            return status;
        *n_samples = static_cast<da_int>(rinfo[0]);
        *n_features = static_cast<da_int>(rinfo[1]);
        *n_components = static_cast<da_int>(rinfo[2]);
        *n_iter = static_cast<da_int>(rinfo[3]);
        *kl_divergence = rinfo[4];
        if (lp_n_iter)
            *lp_n_iter = static_cast<da_int>(rinfo[5]);
        return da_status_success;
    }
};

#endif // TSNE_R_HPP
