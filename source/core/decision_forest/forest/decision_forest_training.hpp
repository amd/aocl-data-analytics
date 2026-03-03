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

#ifndef FOREST_TRAINING_HPP
#define FOREST_TRAINING_HPP

namespace ARCH {

namespace da_decision_forest {

template <typename T> da_status decision_forest<T>::fit() {

    // Read optional parameters
    bool opt_pass = true, bootstrap;
    da_int max_depth, min_node_sample, method, nfeat_split, bootstrap_opt, feat_select,
        cat_split_strat;
    T feat_thresh, min_split_score, min_improvement, prop, feat_proportion;
    std::string opt_val;
    opt_pass &= this->opts.get("number of trees", n_tree) == da_status_success;
    opt_pass &= this->opts.get("maximum depth", max_depth) == da_status_success;
    opt_pass &= this->opts.get("seed", seed) == da_status_success;
    opt_pass &=
        this->opts.get("node minimum samples", min_node_sample) == da_status_success;
    opt_pass &=
        this->opts.get("node minimum samples", min_node_sample) == da_status_success;
    opt_pass &= this->opts.get("scoring function", opt_val, method) == da_status_success;
    opt_pass &=
        this->opts.get("features selection", opt_val, feat_select) == da_status_success;
    opt_pass &= this->opts.get("maximum features", nfeat_split) == da_status_success;
    opt_pass &=
        this->opts.get("proportion features", feat_proportion) == da_status_success;
    opt_pass &= this->opts.get("feature threshold", feat_thresh) == da_status_success;
    opt_pass &=
        this->opts.get("minimum split score", min_split_score) == da_status_success;
    opt_pass &=
        this->opts.get("minimum impurity decrease", min_improvement) == da_status_success;
    opt_pass &= this->opts.get("bootstrap", opt_val, bootstrap_opt) == da_status_success;
    opt_pass &= this->opts.get("bootstrap samples factor", prop) == da_status_success;
    opt_pass &= this->opts.get("block size", block_size) == da_status_success;
    // Histogram options
    opt_pass &= this->opts.get("histogram", opt_val, use_hist) == da_status_success;
    opt_pass &= this->opts.get("maximum bins", usr_max_bins) == da_status_success;
    opt_pass &= this->opts.get("category split strategy", opt_val, cat_split_strat) ==
                da_status_success;

    if (!opt_pass)
        return da_error_trace(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                              "Unexpected error while reading the optional parameters.");

    std::vector<da_int> seed_tree;
    try {
        forest.resize(n_tree);
        seed_tree.resize(n_tree);
        if (use_hist) {
            if (X_binned != nullptr) {
                delete X_binned;
                X_binned = nullptr;
            }
            X_binned = new bins<T>(usr_max_bins, n_samples, n_features);
        }
    } catch (std::bad_alloc &) {                           // LCOV_EXCL_LINE
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    } catch (std::invalid_argument &e) {
        return da_error(this->err, da_status_invalid_option,
                        std::string("Exception: ") + e.what());
    }
    if (use_hist)
        X_binned->compute_histograms(X, n_samples, n_features, ldx);

    // Initialize the seeds of all the trees to be able to reproduce results if required
    std::mt19937 mt_engine;
    if (seed == -1) {
        std::random_device r;
        seed = std::abs((da_int)r());
    }
    mt_engine.seed(seed);
    std::uniform_int_distribution<da_int> uniform_dist(0, 1000000);
    std::generate(seed_tree.begin(), seed_tree.end(),
                  [&uniform_dist, &mt_engine]() { return uniform_dist(mt_engine); });

    // Initialize remaining tree optional parameters
    switch (feat_select) {
    case feat_selection::all:
        nfeat_split = n_features;
        break;

    case feat_selection::proportion:
        nfeat_split = (da_int)std::ceil(feat_proportion * n_features);
        break;

    case feat_selection::sqrt:
        nfeat_split = (da_int)std::ceil(std::sqrt(n_features));
        break;

    case feat_selection::log2:
        nfeat_split = (da_int)std::ceil(std::log2(n_features));
        break;

    case feat_selection::custom:
        // nfeat_split was already loaded with the maximum features option value
    default:
        break;
    }
    bootstrap = bootstrap_opt == 1;

    n_obs = n_samples;
    if (bootstrap) {
        n_obs = std::max((da_int)std::round(n_samples * prop), (da_int)1);
    }
    da_int n_failed_tree = 0;
    // Train all the trees in parallel
#pragma omp parallel for shared(                                                         \
        n_failed_tree, forest, n_tree, max_depth, min_node_sample, method, seed_tree,    \
            min_split_score, feat_thresh, min_improvement, n_samples, n_features, X,     \
            ldx, y, n_class, n_obs, nfeat_split, bootstrap, use_hist, usr_max_bins,      \
            X_binned, cat_split_strat) default(none) schedule(dynamic)
    for (da_int i = 0; i < n_tree; i++) {
        // Set tree optional parameters
        bool check_categorical_data = false;
        da_int opt_max_cat = 10;
        T cat_tol = (T)0.0;
        try {
            forest[i] = std::make_unique<decision_tree<T>>(
                decision_tree(X_binned, max_depth, min_node_sample, method, nfeat_split,
                              seed_tree[i], min_split_score, feat_thresh, min_improvement,
                              bootstrap, check_categorical_data, opt_max_cat, use_hist,
                              usr_max_bins, cat_tol, cat_split_strat));
        } catch (std::bad_alloc &) {
#pragma omp atomic
            n_failed_tree++;
            continue;
        }
        da_status tree_status;
        tree_status =
            forest[i]->set_training_data(n_samples, n_features, X, ldx, y, n_class, n_obs,
                                         nullptr, usr_categorical_feat);
        tree_status = forest[i]->fit();
        forest[i]->clear_working_memory();
        if (tree_status != da_status_success) {
#pragma omp atomic
            n_failed_tree++;
        }
    }

    if (n_failed_tree != 0)
        return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                        std::to_string(n_failed_tree) +
                            " trees failed training unexpectedly.");

    model_trained = true;
    return da_status_success;
}

} // namespace da_decision_forest
} // namespace ARCH

#endif