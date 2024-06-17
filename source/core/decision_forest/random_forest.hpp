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

#ifndef FOREST_HPP
#define FOREST_HPP

#include "aoclda.h"
#include "da_error.hpp"
#include "da_omp.hpp"
#include "decision_tree.hpp"
#include "random_forest_options.hpp"
// #include <memory>

namespace da_random_forest {
using namespace da_decision_tree;
using namespace da_errors;

template <typename T> class random_forest : public basic_handle<T> {

    da_errors::da_error_t *err = nullptr;

    bool model_trained = false;

    // user data. Never modified by the classifier
    // X[n_samples X n_features]: features -- floating point matrix, column major
    // y[n_samples]: labels -- integer array, 0,...,n_classes-1 values
    const T *X = nullptr;
    const da_int *y = nullptr;
    da_int n_samples = 0;
    da_int ldx = 0;
    da_int n_features = 0;
    da_int n_class = 0;

    // options
    da_int n_tree = 0;
    da_int seed, n_obs;

    // model data
    std::vector<std::unique_ptr<decision_tree<T>>> forest;

  public:
    da_options::OptionRegistry opts;
    random_forest(da_errors::da_error_t &err) {
        // assumes that err is valid
        this->err = &err;
        register_forest_options<T>(opts);
    }
    da_status set_training_data(da_int n_samples, da_int n_features, const T *X,
                                da_int ldx, const da_int *y, da_int n_class = 0);
    da_status fit();
    da_status predict(da_int nn_samples, da_int n_features, const T *X, da_int ldx,
                      da_int *y_pred);
    da_status score(da_int nsamp, da_int nfeat, const T *X_test, da_int ldx_test,
                    da_int *y_test, T *score);

    da_status get_result(da_result query, da_int *dim, T *result);
    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] da_int *result) {
        return da_warn(err, da_status_unknown_query,
                       "There are no integer results available for this API.");
    };
};

template <typename T>
da_status random_forest<T>::get_result(da_result query, da_int *dim, T *result) {

    if (!model_trained)
        return da_warn_bypass(
            this->err, da_status_unknown_query,
            "Handle does not contain data relevant to this query. Was the "
            "last call to the solver successful?");
    // Pointers were already tested in the generic get_result

    da_int rinfo_size = 100;
    switch (query) {
    case da_result::da_rinfo:
        if (*dim < rinfo_size) {
            *dim = rinfo_size;
            return da_warn(err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(rinfo_size) + ".");
        }
        result[0] = (T)n_features;
        result[1] = (T)n_samples;
        result[2] = (T)n_obs;
        result[3] = (T)seed;
        result[4] = (T)n_tree;
        break;
    default:
        return da_warn_bypass(err, da_status_unknown_query,
                              "The requested result could not be found.");
    }
    return da_status_success;
}

template <typename T>
da_status random_forest<T>::set_training_data(da_int n_samples, da_int n_features,
                                              const T *X, da_int ldx, const da_int *y,
                                              da_int n_class) {
    da_status status;
    if (X == nullptr || y == nullptr)
        return da_error(this->err, da_status_invalid_input,
                        "Either X, or y are not valid pointers.");
    if (n_samples <= 0 || n_features <= 0) {
        return da_error(
            this->err, da_status_invalid_input,
            "n_samples = " + std::to_string(n_samples) +
                ", n_features = " + std::to_string(n_features) +
                ", the values of n_samples and n_features need to be greater than 0");
    }
    if (ldx < n_samples) {
        return da_error(this->err, da_status_invalid_input,
                        "n_samples = " + std::to_string(n_samples) +
                            ", ldx = " + std::to_string(ldx) +
                            ", the value of ldx needs to be at least as big as the value "
                            "of n_samples");
    }

    this->refresh();
    this->X = X;
    this->y = y;
    this->n_samples = n_samples;
    this->n_features = n_features;
    this->n_class = n_class;
    this->ldx = ldx;
    if (n_class <= 0)
        this->n_class = *std::max_element(y, y + n_samples) + 1;

    return da_status_success;
}

template <typename T> da_status random_forest<T>::fit() {

    // Read optional parameters
    bool opt_pass = true, bootstrap;
    da_int max_depth, min_node_sample, method, build_order, nfeat_split, bootstrap_opt,
        feat_select;
    T feat_thresh, min_split_score, min_improvement, prop;
    std::string opt_val;
    opt_pass &= opts.get("number of trees", n_tree) == da_status_success;
    opt_pass &= opts.get("maximum depth", max_depth) == da_status_success;
    opt_pass &= opts.get("seed", seed) == da_status_success;
    opt_pass &= opts.get("node minimum samples", min_node_sample) == da_status_success;
    opt_pass &= opts.get("node minimum samples", min_node_sample) == da_status_success;
    opt_pass &= opts.get("scoring function", opt_val, method) == da_status_success;
    opt_pass &=
        opts.get("tree building order", opt_val, build_order) == da_status_success;
    opt_pass &= opts.get("features selection", opt_val, feat_select) == da_status_success;
    opt_pass &= opts.get("maximum features", nfeat_split) == da_status_success;
    opt_pass &= opts.get("feature threshold", feat_thresh) == da_status_success;
    opt_pass &= opts.get("minimum split score", min_split_score) == da_status_success;
    opt_pass &=
        opts.get("Minimum split improvement", min_improvement) == da_status_success;
    opt_pass &= opts.get("bootstrap", opt_val, bootstrap_opt) == da_status_success;
    opt_pass &= opts.get("bootstrap samples factor", prop) == da_status_success;
    if (!opt_pass)
        return da_error_trace(err, da_status_internal_error, // LCOV_EXCL_LINE
                              "Unexpected error while reading the optional parameters.");

    std::vector<da_int> seed_tree;
    try {
        forest.resize(n_tree);
        seed_tree.resize(n_tree);
    } catch (std::bad_alloc &) {                     // LCOV_EXCL_LINE
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation error");
    }

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

    case feat_selection::sqrt:
        nfeat_split = (da_int)std::ceil(std::sqrt(n_features));
        break;

    case feat_selection::log2:
        nfeat_split = (da_int)std::ceil(std::sqrt(n_features));
        break;

    case feat_selection::custom:
        // nfeat_split was already loaded with the maximum features option value
    default:
        break;
    }
    bootstrap = bootstrap_opt == 1;

    n_obs = n_samples;
    if (bootstrap && prop < 1.0) {
        n_obs = std::max((da_int)std::round(n_samples * prop), (da_int)1);
    }
    da_int prn_times = 0;
    da_status tree_status;
    da_int n_failed_tree = 0;

    // Train all the trees in parallel
#pragma omp parallel shared(n_failed_tree, forest, n_tree, max_depth, min_node_sample,   \
                                method, prn_times, build_order, seed_tree,               \
                                min_split_score, feat_thresh, min_improvement,           \
                                n_samples, n_features, X, ldx, y, n_class, n_obs,        \
                                tree_status, nfeat_split, bootstrap) default(none)
    {
#pragma omp for schedule(dynamic)
        for (da_int i = 0; i < n_tree; i++) {
            // Set tree optional parameters
            try {
                forest[i] = std::make_unique<decision_tree<T>>(
                    decision_tree(max_depth, min_node_sample, method, prn_times,
                                  build_order, nfeat_split, seed_tree[i], min_split_score,
                                  feat_thresh, min_improvement, bootstrap));
            } catch (std::bad_alloc &) {
#pragma omp critical
                { n_failed_tree += 1; }
                continue;
            }
            tree_status = forest[i]->set_training_data(n_samples, n_features, X, ldx, y,
                                                       n_class, n_obs, nullptr);
            tree_status = forest[i]->fit();
            forest[i]->clear_working_memory();
            if (tree_status != da_status_success) {
#pragma omp critical
                { n_failed_tree += 1; }
            }
        }
    }
    if (n_failed_tree != 0)
        return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                        std::to_string(n_failed_tree) +
                            " trees failed training unexpectedly.");

    model_trained = true;
    return da_status_success;
}

template <typename T>
da_status random_forest<T>::predict(da_int nsamp, da_int nfeat, const T *X_test,
                                    da_int ldx_test, da_int *y_pred) {
    if (X_test == nullptr || y_pred == nullptr) {
        return da_error(this->err, da_status_invalid_input,
                        "Either X_test, or y_pred are not valid pointers.");
    }
    if (nsamp <= 0) {
        return da_error(this->err, da_status_invalid_input,
                        "nsamp = " + std::to_string(nsamp) +
                            ", it must be greater than 0.");
    }
    if (nfeat != n_features) {
        return da_error(this->err, da_status_invalid_input,
                        "nfeat = " + std::to_string(nfeat) +
                            " doesn't match the expected value " +
                            std::to_string(n_features) + ".");
    }
    if (ldx_test < nsamp) {
        return da_error(this->err, da_status_invalid_input,
                        "nsamp = " + std::to_string(nsamp) +
                            ", ldx = " + std::to_string(ldx_test) +
                            ", the value of ldx needs to be at least as big as the value "
                            "of nsamp");
    }

    if (!model_trained) {
        return da_error(this->err, da_status_out_of_date,
                        "The model has not yet been trained or the data it is "
                        "associated with is out of date.");
    }

    std::vector<da_int> count_classes, y_pred_tree;
    try {
        count_classes.resize(n_class * nsamp);
        y_pred_tree.resize(nsamp);
    } catch (std::bad_alloc const &) {               // LCOV_EXCL_LINE
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    // #pragma omp parallel shared(count_classes, nsamp, nfeat, X_test, ldx_test) default(none)
    {
        // #pragma omp for schedule(dynamic)
        for (auto &tree : forest) {
            tree->predict(nsamp, nfeat, X_test, ldx_test, y_pred_tree.data());
            for (da_int i = 0; i < nsamp; i++) {
                da_int c = y_pred_tree[i];
                // #pragma omp critical
                { count_classes[i * n_class + c] += 1; }
            }
        }
    }

#pragma omp parallel shared(nsamp, n_class, y_pred, count_classes) default(none)
#pragma omp for
    for (da_int i = 0; i < nsamp; i++) {
        da_int max_count = -1, class_i = -1;
        for (da_int c = 0; c < n_class; c++) {
            if (count_classes[i * n_class + c] > max_count) {
                max_count = count_classes[i * n_class + c];
                class_i = c;
            }
        }
        y_pred[i] = class_i;
    }

    return da_status_success;
} // namespace da_random_forest

template <typename T>
da_status random_forest<T>::score(da_int nsamp, da_int nfeat, const T *X_test,
                                  da_int ldx_test, da_int *y_test, T *score) {
    if (X_test == nullptr || y_test == nullptr || score == nullptr) {
        return da_error(this->err, da_status_invalid_input,
                        "Either X_test, y_pred or mean_accuracy are not valid pointers.");
    }
    if (nsamp <= 0) {
        return da_error(this->err, da_status_invalid_input,
                        "nsamp = " + std::to_string(nsamp) +
                            ", it must be greater than 0.");
    }
    if (nfeat != n_features) {
        return da_error(this->err, da_status_invalid_input,
                        "nfeat = " + std::to_string(nfeat) +
                            " doesn't match the expected value " +
                            std::to_string(n_features) + ".");
    }
    if (ldx_test < nsamp) {
        return da_error(this->err, da_status_invalid_input,
                        "nsamp = " + std::to_string(nsamp) +
                            ", ldx = " + std::to_string(ldx_test) +
                            ", the value of ldx needs to be at least as big as the value "
                            "of nsamp");
    }

    if (!model_trained) {
        return da_error(this->err, da_status_out_of_date,
                        "The model has not yet been trained or the data it is "
                        "associated with is out of date.");
    }
    std::vector<da_int> count_classes, y_pred_tree;
    try {
        count_classes.resize(n_class * nsamp);
        y_pred_tree.resize(nsamp);
    } catch (std::bad_alloc const &) {               // LCOV_EXCL_LINE
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    // Count the class predictions of each sample
    // #pragma omp parallel shared(count_classes, nsamp, nfeat, X_test, ldx_test) default(none)
    {
        // #pragma omp for schedule(dynamic)
        for (auto &tree : forest) {
            tree->predict(nsamp, nfeat, X_test, ldx_test, y_pred_tree.data());
            for (da_int i = 0; i < nsamp; i++) {
                da_int c = y_pred_tree[i];
                // #pragma omp critical
                { count_classes[i * n_class + c] += 1; }
            }
        }
    }

    // For each sample
    *score = 0;
#pragma omp parallel shared(nsamp, n_class, y_test, count_classes, score) default(none)
#pragma omp for
    for (da_int i = 0; i < nsamp; i++) {
        da_int max_count = -1, class_i = -1;
        for (da_int c = 0; c < n_class; c++) {
            if (count_classes[i * n_class + c] > max_count) {
                max_count = count_classes[i * n_class + c];
                class_i = c;
            }
        }
        if (class_i == y_test[i]) {
#pragma omp critical
            { *score += 1; }
        }
    }
    *score /= (T)nsamp;

    return da_status_success;
}

} // namespace da_random_forest
#endif
