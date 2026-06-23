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

#ifndef TREE_TRAINING_MEMORY_HPP
#define TREE_TRAINING_MEMORY_HPP

#include "aoclda.h"
#include "common/idx_sorting.hpp"
#include "decision_tree.hpp"
#include "macros.h"

namespace ARCH {
namespace da_decision_forest {

template <typename T> da_status decision_tree<T>::init_working_memory_raw() {
    bool init_cat_data = usr_categorical_feat != nullptr || check_cat_data;

    try {
        cat_feat.resize(this->n_features);
    } catch (std::bad_alloc &) {                                  // LCOV_EXCL_LINE
        return da_error_bypass(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                               "Memory allocation error");
    }

    if (usr_categorical_feat != 0)
        memcpy(cat_feat.data(), usr_categorical_feat, n_features * sizeof(da_int));
    else if (check_cat_data) {
        for (da_int j = 0; j < n_features; j++) {
            da_utils::check_categorical_data(n_samples, &X[j * ldx], cat_feat[j],
                                             opt_max_cat, cat_tol);
        }
    }
    if (init_cat_data) {
        max_cat =
            std::max((da_int)0, *std::max_element(cat_feat.begin(), cat_feat.end()));
    } else
        da_std::fill(cat_feat.begin(), cat_feat.end(), -1);

    // Allocate per-thread workspace buffers for raw data path
    try {
        for (da_int t = 0; t < n_threads_split; t++) {
            split_workspace<T> &ws = thread_workspaces[t];
            ws.feature_values.resize(n_obs);
            ws.cat_feat_table.resize(max_cat * n_class);
            ws.samples_idx_local.resize(n_obs);
        }
    } catch (std::bad_alloc &) {                                  // LCOV_EXCL_LINE
        return da_error_bypass(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                               "Memory allocation error");
    }

    return da_status_success;
}

template <typename T> da_status decision_tree<T>::init_working_memory_hist() {
    if (X_binned != nullptr && internal_bins) {
        delete X_binned;
        X_binned = nullptr;
    }
    try {
        if (X_binned == nullptr) {
            X_binned = new bins<T>(usr_max_bins, n_samples, n_features);
            internal_bins = true;
        }
    } catch (std::bad_alloc &) {                                  // LCOV_EXCL_LINE
        return da_error_bypass(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                               "Memory allocation error");
    } catch (std::invalid_argument &e) {
        return da_error_bypass(this->err, da_status_invalid_option, e.what());
    }

    // Allocate per-thread workspace buffers for histogram path
    try {
        for (da_int t = 0; t < n_threads_split; t++) {
            split_workspace<T> &ws = thread_workspaces[t];
            ws.node_hist.resize(n_class * X_binned->max_bin);
            ws.hist_count_samples.resize(X_binned->max_bin);
        }
    } catch (std::bad_alloc &) {                                  // LCOV_EXCL_LINE
        return da_error_bypass(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                               "Memory allocation error");
    }

    return da_status_success;
}

template <typename T> da_status decision_tree<T>::init_working_memory() {
    da_status status = da_status_success;

    // Initialize common memory
    try {
        samples_idx.resize(this->n_obs);
        count_classes.resize(this->n_class);
        features_idx.resize(this->n_features);
        selected_features.reserve(this->n_features);
    } catch (std::bad_alloc &) {                                  // LCOV_EXCL_LINE
        return da_error_bypass(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                               "Memory allocation error");
    }
    da_std::iota(features_idx.begin(), features_idx.end(), 0);

    if (bootstrap) {
        try {
            bootstrap_sample_frequency.resize(this->n_samples);
        } catch (std::bad_alloc &) {                                  // LCOV_EXCL_LINE
            return da_error_bypass(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                                   "Memory allocation error");
        }
    }

    // Allocate per-thread workspaces for parallel feature evaluation
    // heuristic for parallelization, at least 4 features per threads
    n_threads_split = std::max(da_utils::get_n_threads_loop(nfeat_split / 4), (da_int)1);
    if (max_threads > 0)
        n_threads_split = std::min(n_threads_split, max_threads);
    try {
        thread_workspaces.resize(n_threads_split);
        for (da_int t = 0; t < n_threads_split; t++) {
            split_workspace<T> &ws = thread_workspaces[t];
            ws.count_left_classes.resize(n_class);
            ws.count_right_classes.resize(n_class);
            ws.const_feats.resize(nfeat_split);
        }
    } catch (std::bad_alloc &) {                                  // LCOV_EXCL_LINE
        return da_error_bypass(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                               "Memory allocation error");
    }

    if (use_hist) {
        status = init_working_memory_hist();
    } else
        status = init_working_memory_raw();

    return status;
}

template <typename T> da_status decision_tree<T>::resize_tree(size_t new_size) {
    try {
        tree.resize(new_size);
        class_props.resize(new_size * this->n_class);
        return da_status_success;
    } catch (std::bad_alloc &) {                                  // LCOV_EXCL_LINE
        return da_error_bypass(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                               "Memory allocation error");
    }
}

} // namespace da_decision_forest
} // namespace ARCH

#endif