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

#ifndef DECISION_FOREST_R_HPP
#define DECISION_FOREST_R_HPP

#include <Rcpp.h>
#include <vector>

#include "aoclda.h"
#include "internal_utilities_r.hpp"

using namespace Rcpp;

class DecisionTree : public rda_handle {
  private:
    da_int n_class;

  public:
    DecisionTree(da_int seed = -1, da_int max_depth = 29, da_int max_features = 0,
                 std::string criterion = "gini", da_int min_samples_split = 2,
                 std::string category_split_strategy = "ordered", bool histogram = false,
                 da_int maximum_bins = 256, bool predict_proba = true,
                 bool check_data = false) {

        da_status status = da_handle_init_d(&handle, da_handle_decision_tree);
        exception_check(status);

        status = da_options_set(handle, "seed", seed);
        exception_check(status);
        status = da_options_set(handle, "maximum depth", max_depth);
        exception_check(status);
        status = da_options_set(handle, "maximum features", max_features);
        exception_check(status);
        status = da_options_set(handle, "scoring function", criterion.c_str());
        exception_check(status);
        status = da_options_set(handle, "node minimum samples", min_samples_split);
        exception_check(status);
        status = da_options_set(handle, "histogram", histogram ? "yes" : "no");
        exception_check(status);
        status = da_options_set(handle, "maximum bins", maximum_bins);
        exception_check(status);
        status =
            da_options_set(handle, "predict probabilities", predict_proba ? "yes" : "no");
        exception_check(status);
        status = da_options_set(handle, "category split strategy",
                                category_split_strategy.c_str());
        exception_check(status);

        if (check_data) {
            status = da_options_set(handle, "check data", "yes");
            exception_check(status);
        }
    }

    void set_max_features_opt(da_int max_features = 0) {
        da_status status = da_options_set(handle, "maximum features", max_features);
        exception_check(status);
    }

    da_int get_max_features_opt() {
        da_int max_features;
        da_status status = da_options_get(handle, "maximum features", &max_features);
        exception_check(status);
        return max_features;
    }

    void fit(NumericMatrix X, IntegerVector y, double min_impurity_decrease = 0.0,
             double min_split_score = 0.0, double feat_thresh = 1.0e-05,
             Nullable<IntegerVector> categorical_features = R_NilValue) {

        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        da_status status;

        status =
            da_options_set(handle, "minimum impurity decrease", min_impurity_decrease);
        exception_check(status);
        status = da_options_set(handle, "minimum split score", min_split_score);
        exception_check(status);
        status = da_options_set(handle, "feature threshold", feat_thresh);
        exception_check(status);
        status = da_options_set(handle, "storage order", "column-major");
        exception_check(status);

        this->n_class =
            (da_int)(std::round(*std::max_element(y.begin(), y.begin() + n_samples)) + 1);

        const da_int *cat_data = nullptr;
        if (categorical_features.isNotNull()) {
            cat_data = INTEGER(categorical_features.get());
        }

        status = da_tree_set_training_data_d(handle, n_samples, n_features, this->n_class,
                                             X.begin(), n_samples, y.begin(), cat_data);
        exception_check(status);

        status = da_tree_fit_d(handle);
        exception_check(status);
    }

    double score(NumericMatrix X, IntegerVector y) {
        da_int n_obs = X.nrow();
        da_int d = X.ncol();

        double score_val = 0.0;

        da_status status =
            da_tree_score_d(handle, n_obs, d, X.begin(), n_obs, y.begin(), &score_val);
        exception_check(status);

        return score_val;
    }

    IntegerVector predict(NumericMatrix X) {
        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        IntegerVector predictions(n_samples);

        da_status status = da_tree_predict_d(handle, n_samples, n_features, X.begin(),
                                             n_samples, predictions.begin());
        exception_check(status);

        return predictions;
    }

    NumericMatrix predict_proba(NumericMatrix X) {
        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        NumericMatrix probabilities(n_samples, n_class);

        da_status status =
            da_tree_predict_proba_d(handle, n_samples, n_features, X.begin(), n_samples,
                                    probabilities.begin(), n_class, n_samples);
        exception_check(status);

        return probabilities;
    }

    NumericMatrix predict_log_proba(NumericMatrix X) {
        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        NumericMatrix log_probabilities(n_samples, n_class);

        da_status status = da_tree_predict_log_proba_d(
            handle, n_samples, n_features, X.begin(), n_samples,
            log_probabilities.begin(), n_class, n_samples);
        exception_check(status);

        return log_probabilities;
    }

    List get_model_info() {
        da_int dim = 10;

        double rinfo[10];
        da_status status = da_handle_get_result_d(handle, da_rinfo, &dim, rinfo);
        exception_check(status);

        return List::create(Named("n_features") = static_cast<da_int>(rinfo[0]),
                            Named("n_samples") = static_cast<da_int>(rinfo[1]),
                            Named("n_obs") = static_cast<da_int>(rinfo[2]),
                            Named("seed") = static_cast<da_int>(rinfo[3]),
                            Named("depth") = static_cast<da_int>(rinfo[4]),
                            Named("n_nodes") = static_cast<da_int>(rinfo[5]),
                            Named("n_leaves") = static_cast<da_int>(rinfo[6]),
                            Named("n_threads") = static_cast<da_int>(rinfo[7]));
    }

    void save_data(Rcpp::List &state) override { state["n_class"] = n_class; }

    void load_data(Rcpp::List &state) override {
        n_class = Rcpp::as<da_int>(state["n_class"]);
    }
};

class DecisionForest : public rda_handle {
  private:
    da_int n_class = 0;

  public:
    DecisionForest(da_int n_trees = 100, std::string criterion = "gini", da_int seed = -1,
                   da_int max_depth = 29, da_int min_samples_split = 2,
                   bool bootstrap = true, std::string features_selection = "sqrt",
                   da_int max_features = 0, bool histogram = false,
                   da_int maximum_bins = 256, da_int block_size = 256,
                   std::string category_split_strategy = "ordered",
                   da_int max_tree_threads = 0, bool check_data = false) {

        da_status status = da_handle_init_d(&handle, da_handle_decision_forest);
        exception_check(status);

        status = da_options_set(handle, "seed", seed);
        exception_check(status);
        status = da_options_set(handle, "number of trees", n_trees);
        exception_check(status);
        status = da_options_set(handle, "scoring function", criterion.c_str());
        exception_check(status);
        status = da_options_set(handle, "maximum depth", max_depth);
        exception_check(status);
        status = da_options_set(handle, "node minimum samples", min_samples_split);
        exception_check(status);
        status = da_options_set(handle, "bootstrap", bootstrap ? "yes" : "no");
        exception_check(status);
        status = da_options_set(handle, "features selection", features_selection.c_str());
        exception_check(status);
        status = da_options_set(handle, "maximum features", max_features);
        exception_check(status);
        status = da_options_set(handle, "histogram", histogram ? "yes" : "no");
        exception_check(status);
        status = da_options_set(handle, "maximum bins", maximum_bins);
        exception_check(status);
        status = da_options_set(handle, "block size", block_size);
        exception_check(status);
        status = da_options_set(handle, "category split strategy",
                                category_split_strategy.c_str());
        exception_check(status);
        status = da_options_set(handle, "maximum tree threads", max_tree_threads);
        exception_check(status);

        if (check_data) {
            status = da_options_set(handle, "check data", "yes");
            exception_check(status);
        }
    }

    void set_features_selection_opt(std::string features_selection = "sqrt") {
        da_status status;
        status = da_options_set(handle, "features selection", features_selection.c_str());
        exception_check(status);
    }

    void set_max_features_opt(da_int max_features = 0) {
        da_status status;
        status = da_options_set(handle, "maximum features", max_features);
        exception_check(status);
    }

    da_int get_max_features_opt() {
        da_int max_features = 0;
        da_status status = da_options_get(handle, "maximum features", &max_features);
        exception_check(status);
        return max_features;
    }

    std::string get_features_selection_opt() {
        char feat_sel[20];
        da_int lval = 20;
        da_status status = da_options_get(handle, "features selection", feat_sel, &lval);
        exception_check(status);
        return std::string(feat_sel);
    }

    void fit(NumericMatrix X, IntegerVector y, double samples_factor = 1.0,
             double min_impurity_decrease = 0.0, double min_split_score = 0.0,
             double feat_thresh = 1.0e-05, double proportion_features = 0.1,
             Nullable<IntegerVector> categorical_features = R_NilValue) {

        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        da_status status;
        status = da_options_set(handle, "bootstrap samples factor", samples_factor);
        exception_check(status);
        status =
            da_options_set(handle, "minimum impurity decrease", min_impurity_decrease);
        exception_check(status);
        status = da_options_set(handle, "minimum split score", min_split_score);
        exception_check(status);
        status = da_options_set(handle, "feature threshold", feat_thresh);
        exception_check(status);
        status = da_options_set(handle, "proportion features", proportion_features);
        exception_check(status);
        status = da_options_set(handle, "storage order", "column-major");
        exception_check(status);

        this->n_class =
            (da_int)(std::round(*std::max_element(y.begin(), y.begin() + n_samples)) + 1);

        const da_int *cat_data = nullptr;
        if (categorical_features.isNotNull()) {
            cat_data = INTEGER(categorical_features.get());
        }

        status =
            da_forest_set_training_data_d(handle, n_samples, n_features, this->n_class,
                                          X.begin(), n_samples, y.begin(), cat_data);
        exception_check(status);

        status = da_forest_fit_d(handle);
        exception_check(status);
    }

    double score(NumericMatrix X, IntegerVector y) {
        da_int n_obs = X.nrow();
        da_int d = X.ncol();

        double score_val = 0.0;

        da_status status =
            da_forest_score_d(handle, n_obs, d, X.begin(), n_obs, y.begin(), &score_val);
        exception_check(status);

        return score_val;
    }

    IntegerVector predict(NumericMatrix X) {
        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        IntegerVector predictions(n_samples);

        da_status status = da_forest_predict_d(handle, n_samples, n_features, X.begin(),
                                               n_samples, predictions.begin());
        exception_check(status);

        return predictions;
    }

    NumericMatrix predict_proba(NumericMatrix X) {
        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        NumericMatrix probabilities(n_samples, n_class);

        da_status status =
            da_forest_predict_proba_d(handle, n_samples, n_features, X.begin(), n_samples,
                                      probabilities.begin(), n_class, n_samples);
        exception_check(status);

        return probabilities;
    }

    NumericMatrix predict_log_proba(NumericMatrix X) {
        da_int n_samples = X.nrow();
        da_int n_features = X.ncol();

        NumericMatrix log_probabilities(n_samples, n_class);

        da_status status = da_forest_predict_log_proba_d(
            handle, n_samples, n_features, X.begin(), n_samples,
            log_probabilities.begin(), n_class, n_samples);
        exception_check(status);

        return log_probabilities;
    }

    void save_data(Rcpp::List &state) override { state["n_class"] = n_class; }

    void load_data(Rcpp::List &state) override {
        n_class = Rcpp::as<da_int>(state["n_class"]);
    }
};

#endif // DECISION_FOREST_R_HPP
