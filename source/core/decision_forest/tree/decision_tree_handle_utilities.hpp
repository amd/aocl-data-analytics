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

#ifndef TREE_UTILITIES_HPP
#define TREE_UTILITIES_HPP

#include "aoclda.h"
#include "da_std.hpp"
#include "decision_tree_options.hpp"
#include "macros.h"

namespace ARCH {

namespace da_decision_forest {

/**************************************************************
                Initialiazation & destruction
 **************************************************************/
template <typename T>
da_status decision_tree<T>::set_training_data(da_int n_samples, da_int n_features,
                                              const T *X, da_int ldx, const da_int *y,
                                              da_int n_class, da_int n_obs,
                                              da_int *samples_subset,
                                              const da_int *usr_cat_feat) {

    // Guard against errors due to multiple calls using the same class instantiation
    if (X_temp) {
        delete[] (X_temp);
        X_temp = nullptr;
    }

    da_status status =
        this->store_2D_array(n_samples, n_features, X, ldx, &X_temp, &this->X, this->ldx,
                             "n_samples", "n_features", "X", "ldx");
    if (status != da_status_success)
        return status;

    status = this->check_1D_array(n_samples, y, "n_samples", "y", 1);
    if (status != da_status_success)
        return status;

    if (n_obs > n_samples || n_obs < 0) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "n_obs = " + std::to_string(n_obs) +
                                   ", it must be set between 0 and n_samples = " +
                                   std::to_string(n_samples));
    }

    this->refresh();
    this->y = y;
    this->n_samples = n_samples;
    this->n_features = n_features;
    this->n_class = n_class;
    if (n_class <= 0)
        this->n_class = *std::max_element(y, y + n_samples) + 1;
    this->n_obs = n_obs;
    if (this->n_obs == 0)
        this->n_obs = this->n_samples;
    this->samples_subset = samples_subset;

    // Store pointer to the user defined categorical features array
    usr_categorical_feat = usr_cat_feat;

    return da_status_success;
}

template <typename T>
decision_tree<T>::decision_tree(da_errors::da_error_t &err) : basic_handle<T>(err) {
    // Initialize the options registry
    // Any error is stored err->status[.] and this needs to be checked
    // by the caller.
    register_decision_tree_options<T>(this->opts, *this->err);
}

// Constructor bypassing the optional parameters for internal forest use
// Values will NOT be checked
template <typename T>
decision_tree<T>::decision_tree(bins<T> *X_binned, da_int max_depth,
                                da_int min_node_sample, da_int method, da_int nfeat_split,
                                da_int seed, T min_split_score, T feat_thresh,
                                T min_improvement, bool bootstrap, da_int check_cat_data,
                                da_int opt_max_cat, da_int use_hist, da_int usr_max_bins,
                                T cat_tol, da_int cat_split_strat)
    : X_binned(X_binned), max_depth(max_depth), min_node_sample(min_node_sample),
      method(method), nfeat_split(nfeat_split), seed(seed),
      min_split_score(min_split_score), feat_thresh(feat_thresh),
      min_improvement(min_improvement), bootstrap(bootstrap),
      check_cat_data(check_cat_data), opt_max_cat(opt_max_cat), use_hist(use_hist),
      usr_max_bins(usr_max_bins), cat_tol(cat_tol), cat_split_strat(cat_split_strat) {
    this->err = nullptr;
    read_public_options = false;
}

template <typename T> decision_tree<T>::~decision_tree() {
    // Destructor needs to handle arrays that were allocated due to row major storage of input data
    if (X_temp)
        delete[] (X_temp);
    if (X_binned && internal_bins)
        delete X_binned;
}

template <typename T> void decision_tree<T>::refresh() {
    model_trained = false;
    if (tree.capacity() > 0)
        tree = std::vector<node<T>>();
}

template <typename T> void decision_tree<T>::clear_working_memory() {
    samples_idx = std::vector<da_int>();
    count_classes = std::vector<da_int>();
    feature_values = std::vector<T>();
    count_left_classes = std::vector<da_int>();
    count_right_classes = std::vector<da_int>();
    cat_feat = std::vector<da_int>();
    features_idx = std::vector<da_int>();
    cat_feat_table = std::vector<da_int>();
    node_hist = std::vector<da_int>();
    hist_count_samples = std::vector<da_int>();
    hist_feat_values = std::vector<da_int>();
    if (X_temp) {
        delete[] (X_temp);
        X = nullptr;
    }
    if (X_binned && internal_bins) {
        delete X_binned;
        X_binned = nullptr;
    }
}

/*************************************************************************
                        Stat extraction
 *************************************************************************/

template <typename T>
da_status decision_tree<T>::get_result([[maybe_unused]] da_result query,
                                       [[maybe_unused]] da_int *dim,
                                       [[maybe_unused]] da_int *result) {

    return da_warn_bypass(this->err, da_status_unknown_query,
                          "There are no integer results available for this API.");
};

template <typename T>
da_status decision_tree<T>::get_result(da_result query, da_int *dim, T *result) {

    if (!model_trained)
        return da_error_bypass(
            this->err, da_status_unknown_query,
            "Handle does not contain data relevant to this query. Was the "
            "last call to the solver successful?");
    // Pointers were already tested in the generic get_result

    da_int rinfo_size = 7;
    switch (query) {
    case da_result::da_rinfo:
        if (*dim < rinfo_size) {
            *dim = rinfo_size;
            return da_warn_bypass(this->err, da_status_invalid_array_dimension,
                                  "The array is too small. Please provide an array of at "
                                  "least size: " +
                                      std::to_string(rinfo_size) + ".");
        }
        result[0] = (T)n_features;
        result[1] = (T)n_samples;
        result[2] = (T)n_obs;
        result[3] = (T)seed;
        result[4] = (T)depth;
        result[5] = (T)n_nodes;
        result[6] = (T)n_leaves;
        break;
    default:
        return da_warn_bypass(this->err, da_status_unknown_query,
                              "The requested result could not be found.");
    }
    return da_status_success;
}

// Getters for testing purposes
template <typename T> std::vector<da_int> const &decision_tree<T>::get_samples_idx() {
    return samples_idx;
}
template <typename T> void decision_tree<T>::init_samples_idx() {
    samples_idx.resize(n_samples);
    da_std::iota(samples_idx.begin(), samples_idx.end(), 0);
}
template <typename T> void decision_tree<T>::init_feature_values(da_int feat_idx) {
    da_int col_idx = ldx * feat_idx;
    feature_values.resize(n_samples);
    for (da_int i = 0; i < n_samples; i++) {
        feature_values[i] = X[col_idx + samples_idx[i]];
    }
}
template <typename T> std::vector<T> const &decision_tree<T>::get_features_values() {
    return feature_values;
}
template <typename T> std::vector<da_int> const &decision_tree<T>::get_count_classes() {
    return count_classes;
}
template <typename T>
std::vector<da_int> const &decision_tree<T>::get_count_left_classes() {
    return count_left_classes;
}
template <typename T>
std::vector<da_int> const &decision_tree<T>::get_count_right_classes() {
    return count_right_classes;
}
template <typename T> std::vector<da_int> const &decision_tree<T>::get_features_idx() {
    return features_idx;
}
template <typename T> bool decision_tree<T>::model_is_trained() { return model_trained; }
template <typename T> std::vector<node<T>> const &decision_tree<T>::get_tree() {
    return tree;
}

// Setters for testing purposes
template <typename T> void decision_tree<T>::set_bootstrap(bool bs) {
    this->bootstrap = bs;
}

} // namespace da_decision_forest
} // namespace ARCH

#endif