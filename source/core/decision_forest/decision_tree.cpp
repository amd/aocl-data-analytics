/*
        * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclda.h"
#include "da_error.hpp"
#include "da_omp.hpp"
#include "da_std.hpp"
#include "decision_forest.hpp"
#include "decision_tree_misc.hpp"
#include "decision_tree_options.hpp"
#include "macros.h"
#include "options.hpp"

#include <chrono>
#include <deque>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

namespace ARCH {

namespace da_decision_forest {

using namespace da_decision_tree_types;

template <typename T> void node<T>::dummy() {}

template <typename T> void split<T>::copy(split const &sp) {
    prop = sp.prop;
    feat_idx = sp.feat_idx;
    samp_idx = sp.samp_idx;
    score = sp.score;
    threshold = sp.threshold;
    left_score = sp.left_score;
    right_score = sp.right_score;
    category = sp.category;
}

/* Compute the impurity of a node containing n_samples samples.
* On input, count_classes[i] is assumed to contain the number of
* occurrences of class i within the node samples. */
template <class T>
using score_fun_t = typename std::function<T(da_int, da_int, std::vector<da_int> &)>;

template <class T>
T gini_score(da_int n_samples, da_int n_class, std::vector<da_int> &count_classes) {
    T score = 0.0;
    for (da_int c = 0; c < n_class; c++) {
        score += (T)count_classes[c] * (T)count_classes[c];
    }
    score = (T)1.0 - score / ((T)n_samples * (T)n_samples);
    return score;
}

template <class T>
T entropy_score(da_int n_samples, da_int n_class, std::vector<da_int> &count_classes) {
    T score = 0.0;
    for (da_int c = 0; c < n_class; c++) {
        T prob_c = (T)count_classes[c] / (T)n_samples;
        if (prob_c > (T)1.0e-5)
            score -= prob_c * std::log2(prob_c);
    }
    return score;
}

template <class T>
T misclassification_score(da_int n_samples, [[maybe_unused]] da_int n_class,
                          std::vector<da_int> &count_classes) {
    T score =
        (T)1.0 -
        ((T)*std::max_element(count_classes.begin(), count_classes.end())) / (T)n_samples;
    return score;
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

template <typename T> da_status decision_tree<T>::read_options() {
    if (read_public_options) {

        std::string opt_val;
        bool opt_pass = true;
        opt_pass &= this->opts.get("predict probabilities", opt_val, predict_proba_opt) ==
                    da_status_success;
        opt_pass &= this->opts.get("maximum depth", max_depth) == da_status_success;
        opt_pass &=
            this->opts.get("scoring function", opt_val, method) == da_status_success;
        opt_pass &=
            this->opts.get("node minimum samples", min_node_sample) == da_status_success;
        opt_pass &=
            this->opts.get("minimum split score", min_split_score) == da_status_success;
        opt_pass &= this->opts.get("maximum features", nfeat_split) == da_status_success;
        opt_pass &= this->opts.get("seed", seed) == da_status_success;
        opt_pass &= this->opts.get("feature threshold", feat_thresh) == da_status_success;
        opt_pass &= this->opts.get("minimum impurity decrease", min_improvement) ==
                    da_status_success;

        // Raw data options
        opt_pass &= this->opts.get("detect categorical data", opt_val, check_cat_data) ==
                    da_status_success;
        opt_pass &=
            this->opts.get("maximum categories", opt_max_cat) == da_status_success;
        opt_pass &= this->opts.get("category tolerance", cat_tol) == da_status_success;
        opt_pass &= this->opts.get("category split strategy", opt_val, cat_split_strat) ==
                    da_status_success;

        // Histogram options
        opt_pass &= this->opts.get("histogram", opt_val, use_hist) == da_status_success;
        opt_pass &= this->opts.get("maximum bins", usr_max_bins) == da_status_success;

        if (!opt_pass)
            return da_error_bypass(
                this->err, da_status_internal_error, // LCOV_EXCL_LINE
                "Unexpected error while reading the optional parameters.");
    }
    return da_status_success;
}

template <typename T> da_status decision_tree<T>::init_working_memory_raw() {
    bool init_cat_data = usr_categorical_feat != nullptr || check_cat_data;

    try {
        feature_values.resize(this->n_obs);
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
        try {
            cat_feat_table.resize(this->max_cat * this->n_class);
        } catch (std::bad_alloc &) {                                  // LCOV_EXCL_LINE
            return da_error_bypass(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                                   "Memory allocation error");
        }
    } else
        da_std::fill(cat_feat.begin(), cat_feat.end(), -1);

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
        node_hist.resize(n_class * X_binned->max_bin);
        if (DF_SAMPLES_SORT_HIST > 0 &&
            (split_property)cat_split_strat == categorical_ordered) {
            hist_feat_values.resize(this->n_obs); // LCOV_EXCL_LINE
        }
        hist_count_samples.resize(X_binned->max_bin);

    } catch (std::bad_alloc &) {                                  // LCOV_EXCL_LINE
        return da_error_bypass(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                               "Memory allocation error");
    } catch (std::invalid_argument &e) {
        return da_error_bypass(this->err, da_status_invalid_option, e.what());
    }

    return da_status_success;
}

template <typename T> da_status decision_tree<T>::init_working_memory() {
    da_status status = da_status_success;

    // Initialize common memory
    try {
        samples_idx.resize(this->n_obs);
        count_classes.resize(this->n_class);
        count_left_classes.resize(this->n_class);
        count_right_classes.resize(this->n_class);
        features_idx.resize(this->n_features);
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

    if (use_hist) {
        status = init_working_memory_hist();
    } else
        status = init_working_memory_raw();

    return status;
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

template <typename T>
da_status decision_tree<T>::get_result([[maybe_unused]] da_result query,
                                       [[maybe_unused]] da_int *dim,
                                       [[maybe_unused]] da_int *result) {

    return da_warn_bypass(this->err, da_status_unknown_query,
                          "There are no integer results available for this API.");
};

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

/****************************************************************************************
                                    Main tree functions
***************************************************************************************/
template <class T>
void decision_tree<T>::count_class_occurences(std::vector<da_int> &class_occ,
                                              da_int start_idx, da_int end_idx) {
    /* Count the number of occurence of each response class from y in the samples marked in
     * samples_idx[start_idx, end_idx] */
    da_std::fill(class_occ.begin(), class_occ.end(), 0);
    for (da_int i = start_idx; i <= end_idx; i++) {
        da_int idx = samples_idx[i];
        da_int c = y[idx];
        class_occ[c] += 1;
    }
}
template <class T>
void decision_tree<T>::count_class_occurences(std::vector<da_int> &class_occ,
                                              da_int start_idx, da_int end_idx,
                                              std::vector<da_int> &weights) {
    /* Same as above with weights for each sample index */
    da_std::fill(class_occ.begin(), class_occ.end(), 0);
    for (da_int i = start_idx; i <= end_idx; i++) {
        da_int idx = samples_idx[i];
        da_int w = weights[idx];
        da_int c = y[idx];
        class_occ[c] += w;
    }
}

template <class T>
da_status decision_tree<T>::add_node(da_int parent_idx, bool is_left, T score,
                                     da_int split_idx) {

    da_status status = da_status_success;
    if (tree.size() <= (size_t)n_nodes) {
        size_t new_size = 2 * tree.size() + 1;
        // Resize the tree and class_props arrays
        status = resize_tree(new_size);
        if (status != da_status_success)
            return status;
    }
    node<T> &new_node = tree[n_nodes];
    new_node.parent_idx = parent_idx;
    node<T> &parent_node = tree[parent_idx];
    if (is_left) {
        parent_node.left_child_idx = n_nodes;
        new_node.start_idx = parent_node.start_idx;
        new_node.end_idx = split_idx;
    } else {
        parent_node.right_child_idx = n_nodes;
        new_node.start_idx = split_idx + 1;
        new_node.end_idx = parent_node.end_idx;
    }
    new_node.depth = parent_node.depth + 1;
    if (new_node.depth > this->depth)
        this->depth = new_node.depth;
    new_node.score = score;
    new_node.n_samples = 0;
    // Prediction: most represented class in the samples subset
    if (bootstrap) {
        count_class_occurences(count_classes, new_node.start_idx, new_node.end_idx,
                               bootstrap_sample_frequency);
        for (da_int c = 0; c < n_class; c++) {
            new_node.n_samples += count_classes[c];
        }
    } else {
        count_class_occurences(count_classes, new_node.start_idx, new_node.end_idx);
        new_node.n_samples = new_node.end_idx - new_node.start_idx + 1;
    }
    new_node.y_pred = (da_int)std::distance(
        count_classes.begin(),
        std::max_element(count_classes.begin(), count_classes.end()));
    // Prediction probability
    if (predict_proba_opt) {
        for (da_int i = 0; i < n_class; i++) {
            T p = (T)count_classes[i] / (T)new_node.n_samples;
            class_props[n_nodes * n_class + i] = p;
        }
    }
    new_node.const_feat_idx = parent_node.const_feat_idx;
    n_nodes += 1;

    return status;
}

template <class T> da_int decision_tree<T>::get_next_node_idx() {
    // Get the next node index to treat.
    // LIFO: depth-first
    da_int node_idx = nodes_to_treat.back();
    nodes_to_treat.pop_back();

    return node_idx;
}

/****************************************************************************************
 *                                Split functions
 ***************************************************************************************/

/* Utility functions */
/* Update_count_* functions: add to the left count all the values of samples_idx between start and end_idx */
template <typename T>
void decision_tree<T>::update_count_left(da_int start_idx, da_int end_idx,
                                         da_int &ns_left) {
    for (da_int i = start_idx; i <= end_idx; i++) {
        da_int c = y[samples_idx[i]];
        count_left_classes[c]++;
        ns_left++;
    }
}
template <typename T>
void decision_tree<T>::update_count_left(da_int start_idx, da_int end_idx,
                                         da_int &ns_left, std::vector<da_int> &weights) {
    for (da_int i = start_idx; i <= end_idx; i++) {
        da_int idx = samples_idx[i];
        da_int c = y[idx];
        da_int w = weights[idx];
        count_left_classes[c] += w;
        ns_left += w;
    }
}
template <typename T>
void decision_tree<T>::update_count_right(da_int start_idx, da_int end_idx,
                                          da_int &ns_right) {
    for (da_int i = end_idx; i > start_idx; i--) {
        da_int c = y[samples_idx[i]];
        count_right_classes[c]++;
        ns_right++;
    }
}
template <typename T>
void decision_tree<T>::update_count_right(da_int start_idx, da_int end_idx,
                                          da_int &ns_right,
                                          std::vector<da_int> &weights) {
    for (da_int i = end_idx; i > start_idx; i--) {
        da_int idx = samples_idx[i];
        da_int c = y[idx];
        da_int w = weights[idx];
        count_right_classes[c] += w;
        ns_right += w;
    }
}

template <typename T>
template <typename U>
bool decision_tree<T>::update_split_sorted(da_int sidx, da_int &next_idx, da_int end_idx,
                                           da_int ns, da_int &ns_left, da_int &ns_right,
                                           T &left_score, T &right_score, T &split_score,
                                           std::vector<U> &fv) {
    /* update the values of count_[left|right]_occurences up to the next split value for continuous
     * sorted feature values. */
    bool end_split_search = false;
    next_idx = sidx;
    while (next_idx + 1 <= end_idx &&
           std::abs(fv[next_idx + 1] - fv[sidx]) < feat_thresh) {
        next_idx++;
    }
    if (next_idx >= end_idx)
        return true;
    // update from the left or right based on which side has fewer samples
    // The right side would typically be used for features with unbalanced data
    if (next_idx - sidx + 1 <= end_idx - next_idx + 1) {
        if (bootstrap)
            update_count_left(sidx, next_idx, ns_left, bootstrap_sample_frequency);
        else
            update_count_left(sidx, next_idx, ns_left);
        ns_right = ns - ns_left;
        for (da_int i = 0; i < n_class; i++)
            count_right_classes[i] = count_classes[i] - count_left_classes[i];
    } else {
        da_std::fill(count_right_classes.begin(), count_right_classes.end(), 0);
        ns_right = 0;
        if (bootstrap)
            update_count_right(next_idx, end_idx, ns_right, bootstrap_sample_frequency);
        else
            update_count_right(next_idx, end_idx, ns_right);
        ns_left = ns - ns_right;
        for (da_int i = 0; i < n_class; i++)
            count_left_classes[i] = count_classes[i] - count_right_classes[i];
    }

    left_score = score_function(ns_left, n_class, count_left_classes);
    right_score = score_function(ns_right, n_class, count_right_classes);
    split_score = (left_score * ns_left + right_score * ns_right) / ns;

    return end_split_search;
}

/* Binned data */
template <typename T>
bool decision_tree<T>::update_node_histogram(const node<T> &nd, da_int feat_idx) {
    /* Auxialiary function for histogram based split computation.
     *
     * for a  given feature feat_idx and node nd, count the number of each response class
     * occurence fo each possible bin value.
     * On output, node_hist[bin, c] will contain the number of samples from the node nd
     * that have both feature value equal to bin and class label equal to c.
     */
    da_int start_idx = feat_idx * n_samples;
    memset(node_hist.data(), 0, node_hist.size() * sizeof(da_int));
    memset(hist_count_samples.data(), 0, hist_count_samples.size() * sizeof(da_int));
    da_int const_cat_val = -1;
    bool const_feat = true;
    for (da_int i = nd.start_idx; i <= nd.end_idx; i++) {
        da_int idx = samples_idx[i];
        da_int c = y[idx];
        uint16_t cat = X_binned->binned_data[start_idx + idx];
        node_hist[cat * n_class + c]++;
        hist_count_samples[cat] += 1;
        if (const_feat) {
            if (const_cat_val == -1)
                const_cat_val = (da_int)cat;
            else if (const_cat_val != cat)
                const_feat = false;
        }
    }
    return const_feat;
}

template <typename T>
bool decision_tree<T>::update_node_histogram(const node<T> &nd, da_int feat_idx,
                                             std::vector<da_int> &weights) {
    /* Same as above, with weights for each sample index */
    da_int start_idx = feat_idx * n_samples;
    memset(node_hist.data(), 0, node_hist.size() * sizeof(da_int));
    memset(hist_count_samples.data(), 0, hist_count_samples.size() * sizeof(da_int));
    da_int const_cat_val = -1;
    bool const_feat = true;
    for (da_int i = nd.start_idx; i <= nd.end_idx; i++) {
        da_int idx = samples_idx[i];
        da_int w = weights[idx];
        da_int c = y[idx];
        uint16_t cat = X_binned->binned_data[start_idx + idx];
        node_hist[cat * n_class + c] += w;
        hist_count_samples[cat] += 1;
        if (const_feat) {
            if (const_cat_val == -1)
                const_cat_val = (da_int)cat;
            else if (const_cat_val != cat)
                const_feat = false;
        }
    }
    return const_feat;
}

template <typename T>
void decision_tree<T>::split_hist_onevall(const node<T> &nd, da_int &ns_left,
                                          da_int &ns_right, da_int cat_start_idx) {
    /* split strategy one vs all: all samples for the split bin value are set in the left child
     * while others are set in the right child.
     *
     * cat_start_idx: index of the split bin value column in node_hist (e.g., bin * n_class)
     * Compute the number of occurrences of each response class in a potential split
     * On output, count_[left|right]_classes will contain the number of samples in the
     * corresponding left and right split nodes.
     * node_hist is expected to have been updated beforehand.
     */
    for (da_int c = 0; c < n_class; c++) {
        count_left_classes[c] = node_hist[cat_start_idx + c];
        count_right_classes[c] = count_classes[c] - count_left_classes[c];
        ns_left += count_left_classes[c];
    }

    ns_right = nd.n_samples - ns_left;
}

template <typename T>
void decision_tree<T>::split_hist_ordered(const node<T> &nd, da_int &ns_left,
                                          da_int &ns_right, da_int cat_start_idx) {
    /* split strategy ordered: samples with a split bin value smaller than the split threshold
     * are set in the left child while others are set in the right child.
     */
    for (da_int c = 0; c < n_class; c++) {
        count_left_classes[c] += node_hist[cat_start_idx + c];
        count_right_classes[c] = count_classes[c] - count_left_classes[c];
        ns_left += count_left_classes[c];
    }
    ns_right = nd.n_samples - ns_left;
}

template <typename T>
bool decision_tree<T>::compute_best_split_hist(const node<T> &nd, da_int feat_idx,
                                               split<T> &sp) {
    /* Main function for histograms based splits.
     * loop through all the bin values of feature feat_idx and update the split properties of sp
     * if a good split is found. */
    bool const_feat = false;
    if (bootstrap)
        const_feat = update_node_histogram(nd, feat_idx, bootstrap_sample_frequency);
    else
        const_feat = update_node_histogram(nd, feat_idx);

    if (const_feat)
        return const_feat;

    memset(count_left_classes.data(), 0, n_class * sizeof(da_int));

    split_property prop = categorical_ordered;
    da_int n_cat = X_binned->nbins[feat_idx];
    if (n_cat < X_binned->max_bin)
        prop = (split_property)cat_split_strat;

    da_int old_ns_left = 0;
    da_int thresh_start_idx = feat_idx * (X_binned->max_bin - 1);
    for (da_int cat = 0; cat < n_cat; ++cat) {
        if (hist_count_samples[cat] == 0)
            continue;
        da_int ns_left = 0, ns_right = 0;
        da_int cat_start_idx = cat * n_class;

        if (prop == categorical_onevall)
            split_hist_onevall(nd, ns_left, ns_right, cat_start_idx);
        else
            split_hist_ordered(nd, ns_left, ns_right, cat_start_idx);

        if (ns_left < min_node_sample || ns_left <= old_ns_left)
            continue;
        if (ns_right < min_node_sample)
            break;
        old_ns_left = ns_left;

        T left_score = score_function(ns_left, n_class, count_left_classes);
        T right_score = score_function(ns_right, n_class, count_right_classes);
        T split_score = (left_score * ns_left + right_score * ns_right) / nd.n_samples;
        T split_improvement = (T)nd.n_samples / (T)n_obs_total * (nd.score - split_score);
        if (split_score < sp.score && split_improvement > min_improvement) {
            sp.score = split_score;
            sp.right_score = right_score;
            sp.left_score = left_score;
            sp.category = cat;
            sp.prop = prop;
            sp.feat_idx = feat_idx;
            if (prop == categorical_ordered)
                sp.threshold = X_binned->thresholds[thresh_start_idx + cat];
        }
    }
    return const_feat;
}

template <typename T>
bool decision_tree<T>::compute_best_split_hist_sort(const node<T> &nd, da_int feat_idx,
                                                    split<T> &sp) {
    /* Alternative to the previous function to compute the best split using histograms.
     * Here, similarly to continuous raw data, we sort the samples_idx array according to the
     * binned feature values and look at all possible splits values.
     *
     * (More benchmarks is needed to determine when each strategy is better suited).
     */
    bool const_feat = false;
    for (da_int i = nd.start_idx; i <= nd.end_idx; i++)
        hist_feat_values[i] =
            X_binned->binned_data[n_samples * feat_idx + samples_idx[i]];

    da_int node_obs = nd.end_idx - nd.start_idx + 1;
    multi_range_intro_sort(samples_idx, hist_feat_values, nd.start_idx, node_obs,
                           (da_int)(2 * std::log2(node_obs) + 2));

    if (hist_feat_values[nd.start_idx] == hist_feat_values[nd.end_idx])
        return true;
    sp.feat_idx = feat_idx;

    std::copy(count_classes.begin(), count_classes.end(), count_right_classes.begin());
    da_std::fill(count_left_classes.begin(), count_left_classes.end(), 0);
    T right_score = nd.score, left_score = 0.0;
    da_int ns_left = 0;
    da_int ns_right = nd.n_samples;
    sp.score = nd.score;

    T split_score = 0.;
    da_int sidx = nd.start_idx;
    da_int next_idx;
    da_int thresh_start_idx = feat_idx * (X_binned->max_bin - 1);
    while (sidx <= nd.end_idx - 1) {
        update_split_sorted(sidx, next_idx, nd.end_idx, nd.n_samples, ns_left, ns_right,
                            left_score, right_score, split_score, hist_feat_values);

        // Consider the split only if it brings at least minimum improvement
        // compared to the parent node
        T split_improvement = (T)nd.n_samples / (T)n_obs_total * (nd.score - split_score);
        if (split_score < sp.score && split_improvement > min_improvement) {
            sp.score = split_score;
            uint16_t cat = hist_feat_values[next_idx];
            sp.threshold = X_binned->thresholds[thresh_start_idx + cat];
            sp.category = cat;
            sp.right_score = right_score;
            sp.left_score = left_score;
            sp.prop = categorical_ordered;
        }

        sidx = next_idx + 1;
    }
    return const_feat;
}

/* Raw data */
template <typename T>
void decision_tree<T>::update_feature_values(da_int start_idx, da_int end_idx,
                                             da_int feat_idx) {
    /* fill the feature_values array with the values of X marked by the samples_idx and feat_idx */
    da_int col_idx = ldx * feat_idx;
    for (da_int i = start_idx; i <= end_idx; i++)
        feature_values[i] = X[col_idx + samples_idx[i]];
}

template <typename T>
void decision_tree<T>::split_raw_onevall(const node<T> &current_node, da_int feat_idx,
                                         split<T> &sp) {
    sp.score = current_node.score;

    // fill cat_feat_table, counting for each possible category of feat_idx
    // the number of occurrences of each response class in the samples
    // After the loop, column j of cat_feat_table will countain the count of
    // each class (from response vector y) in the samples contained in the current node
    da_std::fill(cat_feat_table.begin(), cat_feat_table.end(), 0);
    for (da_int i = current_node.start_idx; i <= current_node.end_idx; i++) {
        da_int idx = samples_idx[i];
        da_int c = y[idx];
        da_int cat = std::round(feature_values[i]);
        cat_feat_table[cat * n_class + c]++;
    }

    for (da_int cat = 0; cat < cat_feat[feat_idx]; cat++) {
        da_int ns_left = 0, ns_right = 0;
        for (da_int c = 0; c < n_class; c++) {
            count_left_classes[c] = cat_feat_table[cat * n_class + c];
            ns_left += count_left_classes[c];
            count_right_classes[c] = count_classes[c] - count_left_classes[c];
        }
        ns_right = current_node.n_samples - ns_left;
        if (ns_left < min_node_sample)
            continue;
        if (ns_right < min_node_sample)
            continue;

        T left_score = score_function(ns_left, n_class, count_left_classes);
        T right_score = score_function(ns_right, n_class, count_right_classes);
        T split_score =
            (left_score * ns_left + right_score * ns_right) / current_node.n_samples;
        T split_improvement = (T)current_node.n_samples / (T)n_obs_total *
                              (current_node.score - split_score);

        if (split_score < sp.score && split_improvement > min_improvement) {
            sp.score = split_score;
            sp.right_score = right_score;
            sp.left_score = left_score;
            sp.category = cat;
            sp.prop = categorical_onevall;
            sp.feat_idx = feat_idx;
        }
    }
}

template <typename T>
bool decision_tree<T>::compute_best_split_raw(const node<T> &nd, da_int feat_idx,
                                              split<T> &sp) {
    /* Main function for raw data splits. */
    update_feature_values(nd.start_idx, nd.end_idx, feat_idx);

    if ((split_property)cat_split_strat == categorical_ordered ||
        cat_feat[feat_idx] <= 0) {
        /* continuous data: samples_idx is sorted according to the values of feat_idx
         * column in X and all possible split threshold values are checked. */
        if (cat_feat[feat_idx] != 1) {
            da_int node_obs = nd.end_idx - nd.start_idx + 1;
            multi_range_intro_sort(samples_idx, feature_values, nd.start_idx, node_obs,
                                   (da_int)(2 * std::log2(node_obs) + 2));
        }

        // sort_samples(nd, feat_idx);
        if (std::abs(feature_values[nd.start_idx] - feature_values[nd.end_idx]) <
            (T)1.0e-05)
            // feature is constant, mark to skip it
            return true;
        sp.feat_idx = feat_idx;
        split_raw_continuous(nd, sp);

    } else {
        /* categorical data in raw matrix: similar strategy used in binned data. */
        sp.feat_idx = feat_idx;
        split_raw_onevall(nd, feat_idx, sp);
    }

    return false;
}

template <typename T>
bool decision_tree<T>::compute_best_split(const node<T> &nd, da_int feat_idx,
                                          split<T> &sp) {
    /* Main entry point for the split computation, dispatches to the correct function based
     * on split properties and strategies */
    bool const_feat;
    if (!use_hist)
        const_feat = compute_best_split_raw(nd, feat_idx, sp);
    else {
        if (nd.n_samples < DF_SAMPLES_SORT_HIST &&
            (split_property)cat_split_strat == categorical_ordered &&
            X_binned->nbins[feat_idx] < X_binned->max_bin)
            const_feat = compute_best_split_hist_sort(nd, feat_idx, sp);
        else
            const_feat = compute_best_split_hist(nd, feat_idx, sp);
    }
    return const_feat;
}

/****************************************************************************************
                                Partition functions
 ***************************************************************************************/
/* These functions partition the samples_idx array based on the node's feature and threshold
 * (depending on the strategy adopted for splitting)
 * All partition functions are expected to return the index of the split in samples_idx
 * (e.g., all values in samples_idx below the split index will be set in the left node)
 */
template <typename T>
da_int decision_tree<T>::partition_samples_raw_continuous(const node<T> &nd) {
    /* raw data: look in main feature matrix directly
     * continuous feature: all samples below the node threshold are first
     */
    da_int head_idx = nd.start_idx, tail_idx = nd.end_idx;
    da_int start_col = ldx * nd.feature;
    while (head_idx < tail_idx) {
        da_int h_sidx = samples_idx[head_idx];
        da_int t_sidx = samples_idx[tail_idx];
        T head_val = X[start_col + h_sidx];
        T tail_val = X[start_col + t_sidx];
        if (head_val <= nd.x_threshold)
            head_idx += 1;
        else if (tail_val > nd.x_threshold)
            tail_idx -= 1;
        else {
            std::swap(samples_idx[head_idx], samples_idx[tail_idx]);
        }
    }
    return head_idx - 1;
}

template <typename T>
da_int decision_tree<T>::partition_samples_raw_categorical(const node<T> &nd) {
    /* raw data: look in main feature matrix directly
     * categorical feature: all samples corresponding to the node category are first
     */
    da_int head_idx = nd.start_idx;
    da_int tail_idx = nd.end_idx;
    da_int col_idx = ldx * nd.feature;
    while (head_idx < tail_idx) {
        da_int head = samples_idx[head_idx];
        da_int tail = samples_idx[tail_idx];
        if (std::round(X[col_idx + head]) == nd.category)
            head_idx++;
        else if (std::round(X[col_idx + tail]) != nd.category)
            tail_idx--;
        else
            std::swap(samples_idx[head_idx], samples_idx[tail_idx]);
    }
    return head_idx - 1;
}

template <typename T>
da_int decision_tree<T>::partition_samples_hist_ordered(const node<T> &nd) {
    /* hist: use the binned data X_binned
     * ordered feature: all samples below the node threshold are first
     */
    da_int head_idx = nd.start_idx;
    da_int tail_idx = nd.end_idx;
    uint16_t cat_thresh = nd.category;
    da_int col_idx = ldx * nd.feature;
    while (head_idx < tail_idx) {
        da_int head = samples_idx[head_idx];
        da_int tail = samples_idx[tail_idx];
        uint16_t cat_head = X_binned->binned_data[col_idx + head];
        uint16_t cat_tail = X_binned->binned_data[col_idx + tail];
        if (cat_head <= cat_thresh)
            head_idx++;
        else if (cat_tail > cat_thresh)
            tail_idx--;
        else
            std::swap(samples_idx[head_idx], samples_idx[tail_idx]);
    }
    return head_idx - 1;
}

template <typename T>
da_int decision_tree<T>::partition_samples_hist_onevall(const node<T> &nd) {
    /* hist: use the binned data X_binned
     * categorical feature: all samples corresponding to the node category are first
     */
    da_int head_idx = nd.start_idx;
    da_int tail_idx = nd.end_idx;
    uint16_t cat_thresh = nd.category;
    da_int col_idx = ldx * nd.feature;
    while (head_idx < tail_idx) {
        da_int head = samples_idx[head_idx];
        da_int tail = samples_idx[tail_idx];
        uint16_t cat_head = X_binned->binned_data[col_idx + head];
        uint16_t cat_tail = X_binned->binned_data[col_idx + tail];
        if (cat_head == cat_thresh)
            head_idx++;
        else if (cat_tail != cat_thresh)
            tail_idx--;
        else
            std::swap(samples_idx[head_idx], samples_idx[tail_idx]);
    }
    return head_idx - 1;
}

template <typename T>
void decision_tree<T>::split_raw_continuous(const node<T> &current_node, split<T> &sp) {
    // Initialize the split, all nodes to the right child.
    // count_class, samples_idx and feature_values are required to be up to date
    std::copy(count_classes.begin(), count_classes.end(), count_right_classes.begin());
    da_std::fill(count_left_classes.begin(), count_left_classes.end(), 0);
    T right_score = current_node.score, left_score = 0.0;
    da_int ns_left = 0;
    da_int ns_right = current_node.n_samples;
    sp.score = current_node.score;
    // sp.samp_idx = -1;

    T split_score = 0.;
    da_int sidx = current_node.start_idx;
    da_int next_idx;
    while (sidx <= current_node.end_idx - 1) {
        update_split_sorted(sidx, next_idx, current_node.end_idx, current_node.n_samples,
                            ns_left, ns_right, left_score, right_score, split_score,
                            feature_values);
        if (ns_left < min_node_sample) {
            ++sidx;
            continue;
        }
        if (ns_right < min_node_sample)
            break;

        // Consider the split only if it brings at least minimum improvement
        // compared to the parent node
        T split_improvement = (T)current_node.n_samples / (T)n_obs_total *
                              (current_node.score - split_score);
        if (split_score < sp.score && split_improvement > min_improvement) {
            sp.score = split_score;
            sp.threshold = (feature_values[next_idx] + feature_values[next_idx + 1]) / 2;
            sp.right_score = right_score;
            sp.left_score = left_score;
            sp.prop = continuous;
        }

        sidx = next_idx + 1;
    }
}

template <typename T> da_status decision_tree<T>::fit() {
    da_status status = da_status_success;

    if (model_trained)
        // Nothing to do, exit
        return da_status_success;

    status = read_options();
    if (status != da_status_success)
        return status; // Error message already filled

    status = init_working_memory();
    if (status != da_status_success)
        return status; // Error message already filled

    switch (method) {
    case gini:
        score_function = gini_score<T>;
        break;

    case cross_entropy:
        score_function = entropy_score<T>;
        break;

    case misclassification:
        score_function = misclassification_score<T>;
        break;
    }

    if (nfeat_split == 0 || nfeat_split > n_features) {
        // All the features are to be considered in splitting a node
        nfeat_split = n_features;
    }

    // Initialize random number generator
    if (seed == -1) {
        std::random_device r;
        seed = std::abs((da_int)r());
    }
    mt_engine.seed(seed);

    // Compute the bins if needed
    if (use_hist && internal_bins) {
        X_binned->compute_histograms(X, n_samples, n_features, ldx);
    }

    // Allocate the tree and class_props arrays
    // accounting for a full binary tree of depth 10 (or maximum depth)
    size_t init_capacity = (da_int)1 << (std::min(max_depth, (da_int)10) + (da_int)1);
    status = resize_tree(init_capacity);
    if (status != da_status_success)
        return status;

    n_obs_total = n_obs;
    if (!bootstrap) {
        // Take all the samples
        da_std::iota(samples_idx.begin(), samples_idx.end(), 0);
    } else {
        if (samples_subset == nullptr) {
            // Fill the index vector with a random selection with replacement
            std::uniform_int_distribution<da_int> uniform_dist(0, n_samples - 1);
            std::generate(samples_idx.begin(), samples_idx.end(),
                          [&uniform_dist, &mt_engine = this->mt_engine]() {
                              return uniform_dist(mt_engine);
                          });
        } else {
            // Copy the input from the samples_subset array.
            // As it is intended mainly for testing, samples_subset is NOT validated.
            for (da_int i = 0; i < n_obs; i++)
                samples_idx[i] = samples_subset[i];
        }
        status = compress_count_occurences(samples_idx, bootstrap_sample_frequency);
        // only memory error can be raised
        if (status != da_status_success)
            return da_error_bypass(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                                   "Memory allocation error");
        n_obs = samples_idx.size();
    }

    // Reset number of leaves (if calling fit multiple times)
    n_leaves = 0;

    // Initialize the root node
    n_nodes = 1;
    tree[0].start_idx = 0;
    tree[0].end_idx = n_obs - 1;
    tree[0].depth = 0;
    tree[0].n_samples = n_obs_total;
    if (bootstrap)
        count_class_occurences(count_classes, 0, n_obs - 1, bootstrap_sample_frequency);
    else
        count_class_occurences(count_classes, 0, n_obs - 1);
    tree[0].score = score_function(n_obs_total, n_class, count_classes);
    tree[0].y_pred = (da_int)std::distance(
        count_classes.begin(),
        std::max_element(count_classes.begin(), count_classes.end()));
    tree[0].const_feat_idx = n_features;
    // Prediction probability
    if (predict_proba_opt) {
        for (da_int i = 0; i < n_class; i++) {
            T p = (T)count_classes[i] / (T)n_obs_total;
            class_props[i] = p;
        }
    }

    // Insert the root node in the queue if the maximum depth is big enough
    if (max_depth > 0)
        nodes_to_treat.push_back(0);

    split<T> sp, best_split;
    while (!nodes_to_treat.empty()) {
        da_int node_idx = get_next_node_idx();
        node<T> &current_node = tree[node_idx];

        // update the constant features discovered by the sibling node if necessary
        if (node_idx > 0 &&
            tree[current_node.parent_idx].const_feat_idx < current_node.const_feat_idx)
            current_node.const_feat_idx = tree[current_node.parent_idx].const_feat_idx;

        best_split.score = current_node.score;
        best_split.feat_idx = -1;
        if (node_idx > 0) {
            if (bootstrap)
                count_class_occurences(count_classes, current_node.start_idx,
                                       current_node.end_idx, bootstrap_sample_frequency);
            else
                count_class_occurences(count_classes, current_node.start_idx,
                                       current_node.end_idx);
        }

        da_int fvar_idx = 0;
        da_int feat_to_draw = nfeat_split;
        while (fvar_idx < feat_to_draw && fvar_idx < current_node.const_feat_idx) {
            if (nfeat_split < n_features) {
                std::uniform_int_distribution<da_int> uniform_dist(fvar_idx,
                                                                   n_features - 1);
                da_int random_feat_idx = uniform_dist(mt_engine);
                if (random_feat_idx >= current_node.const_feat_idx) {
                    feat_to_draw--;
                    continue;
                }
                std::swap(features_idx[fvar_idx], features_idx[random_feat_idx]);
            }

            da_int feat_idx = features_idx[fvar_idx];
            bool const_feat;
            sp.score = current_node.score;
            const_feat = compute_best_split(current_node, feat_idx, sp);

            if (const_feat) {
                current_node.const_feat_idx -= 1;
                features_idx[fvar_idx] = features_idx[current_node.const_feat_idx];
                features_idx[current_node.const_feat_idx] = feat_idx;
                continue;
            } else
                fvar_idx++;

            if (sp.score < best_split.score)
                best_split.copy(sp);
        }

        // project back the discovered constant features to the parent
        if (node_idx > 0 &&
            tree[current_node.parent_idx].const_feat_idx > current_node.const_feat_idx)
            tree[current_node.parent_idx].const_feat_idx = current_node.const_feat_idx;

        // Split the node and add the 2 children
        if (best_split.feat_idx != -1) {
            current_node.is_leaf = false;
            current_node.feature = best_split.feat_idx;

            // partition the samples according to the chosen feature and its threshold
            if (!use_hist) {
                if (best_split.prop == continuous) {
                    current_node.prop = continuous;
                    current_node.x_threshold = best_split.threshold;
                    best_split.samp_idx = partition_samples_raw_continuous(current_node);
                } else {
                    current_node.category = best_split.category;
                    current_node.prop = categorical_onevall;
                    best_split.samp_idx = partition_samples_raw_categorical(current_node);
                }
            } else {
                current_node.prop = best_split.prop;
                current_node.category = best_split.category;
                if (best_split.prop == categorical_ordered) {
                    best_split.samp_idx = partition_samples_hist_ordered(current_node);
                    current_node.x_threshold = best_split.threshold;
                } else if (best_split.prop == categorical_onevall)
                    best_split.samp_idx = partition_samples_hist_onevall(current_node);
                else
                    return da_error_bypass( // LCOV_EXCL_LINE
                        this->err, da_status_internal_error,
                        "continuous data requested with histograms unexpectedly.");
            }

            // Add children nodes and push them into the queue
            // if potential for further improvements is still high enough
            add_node(node_idx, false, best_split.right_score, best_split.samp_idx);
            if (best_split.right_score > min_split_score &&
                tree[n_nodes - 1].n_samples >= 2 * min_node_sample &&
                tree[n_nodes - 1].depth < max_depth)
                nodes_to_treat.push_back(n_nodes - 1);
            else
                n_leaves += 1;
            add_node(node_idx, true, best_split.left_score, best_split.samp_idx);
            if (best_split.left_score > min_split_score &&
                tree[n_nodes - 1].n_samples >= 2 * min_node_sample &&
                tree[n_nodes - 1].depth < max_depth)
                nodes_to_treat.push_back(n_nodes - 1);
            else
                n_leaves += 1;
        } else
            n_leaves += 1;
    }
    model_trained = true;
    return status;
}

/****************************************************************************************
                                Inference functions
 ***************************************************************************************/
template <typename T>
da_status decision_tree<T>::predict(da_int nsamp, da_int nfeat, const T *X_test,
                                    da_int ldx_test, da_int *y_pred, da_int mode) {
    if (y_pred == nullptr) {
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "y_pred is not a valid pointer.");
    }

    const T *X_test_temp;
    T *utility_ptr1 = nullptr;
    da_int ldx_test_temp;

    if (nfeat != n_features) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "n_features = " + std::to_string(nfeat) +
                                   " doesn't match the expected value " +
                                   std::to_string(n_features) + ".");
    }

    if (!model_trained) {
        return da_error_bypass(this->err, da_status_out_of_date,
                               "The model has not yet been trained or the data it is "
                               "associated with is out of date.");
    }

    da_status status = this->store_2D_array(nsamp, nfeat, X_test, ldx_test, &utility_ptr1,
                                            &X_test_temp, ldx_test_temp, "n_samples",
                                            "n_features", "X_test", "ldx_test", mode);
    if (status != da_status_success)
        return status;

    // Fill y_pred with the values of all the requested samples
    node<T> *current_node;
    for (da_int i = 0; i < nsamp; i++) {
        current_node = &tree[0];
        while (!current_node->is_leaf) {
            if (current_node->prop == continuous ||
                current_node->prop == categorical_ordered) {
                T feat_val = X_test_temp[ldx_test_temp * current_node->feature + i];
                if (feat_val < current_node->x_threshold)
                    current_node = &tree[current_node->left_child_idx];
                else
                    current_node = &tree[current_node->right_child_idx];
            } else {
                da_int cat_val =
                    std::round(X_test_temp[ldx_test_temp * current_node->feature + i]);
                if (cat_val == current_node->category)
                    current_node = &tree[current_node->left_child_idx];
                else
                    current_node = &tree[current_node->right_child_idx];
            }
        }
        y_pred[i] = current_node->y_pred;
    }
    if (utility_ptr1)
        delete[] (utility_ptr1);
    return da_status_success;
}

template <typename T>
da_status decision_tree<T>::predict_proba(da_int nsamp, da_int nfeat, const T *X_test,
                                          da_int ldx_test, T *y_proba_pred, da_int nclass,
                                          da_int ldy, da_int mode) {

    const T *X_test_temp;
    T *utility_ptr1;
    T *utility_ptr2;
    da_int ldx_test_temp;
    T *y_proba_pred_temp;
    da_int ldy_proba_pred_temp;

    if (!predict_proba_opt) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "predict_proba must be set to 1");
    }

    if (nfeat != n_features) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "n_features = " + std::to_string(nfeat) +
                                   " doesn't match the expected value " +
                                   std::to_string(n_features) + ".");
    }

    if (nclass != n_class) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "n_features = " + std::to_string(nclass) +
                                   " doesn't match the expected value " +
                                   std::to_string(nclass) + ".");
    }

    if (!model_trained) {
        return da_error_bypass(this->err, da_status_out_of_date,
                               "The model has not yet been trained or the data it is "
                               "associated with is out of date.");
    }

    da_status status = this->store_2D_array(nsamp, nfeat, X_test, ldx_test, &utility_ptr1,
                                            &X_test_temp, ldx_test_temp, "n_samples",
                                            "n_features", "X_test", "ldx_test", mode);
    if (status != da_status_success)
        return status;

    da_int mode_output = (mode == 0) ? 1 : mode;
    status = this->store_2D_array(nsamp, nclass, y_proba_pred, ldy, &utility_ptr2,
                                  const_cast<const T **>(&y_proba_pred_temp),
                                  ldy_proba_pred_temp, "n_samples", "n_class", "y_proba",
                                  "ldy", mode_output);
    if (status != da_status_success)
        return status;

    // Fill y_proba_pred with the values of all the requested samples
    node<T> *current_node;
    for (da_int i = 0; i < nsamp; i++) {
        current_node = &tree[0];
        da_int current_node_idx = 0;
        while (!current_node->is_leaf) {
            T feat_val = X_test_temp[ldx_test_temp * current_node->feature + i];
            if (feat_val < current_node->x_threshold) {
                current_node_idx = current_node->left_child_idx;
                current_node = &tree[current_node_idx];
            } else {
                current_node_idx = current_node->right_child_idx;
                current_node = &tree[current_node_idx];
            }
        }
        for (da_int j = 0; j < n_class; j++)
            y_proba_pred_temp[ldy_proba_pred_temp * j + i] =
                class_props[n_class * current_node_idx + j];
    }

    if (this->order == row_major) {

        da_utils::copy_transpose_2D_array_column_to_row_major(
            nsamp, n_class, y_proba_pred_temp, ldy_proba_pred_temp, y_proba_pred, ldy);
        if (utility_ptr1)
            delete[] (utility_ptr1);
        if (utility_ptr2)
            delete[] (utility_ptr2);
    }
    return da_status_success;
}

template <typename T>
da_status decision_tree<T>::predict_log_proba(da_int nsamp, da_int nfeat, const T *X_test,
                                              da_int ldx_test, T *y_log_proba,
                                              da_int nclass, da_int ldy) {
    da_status status = da_status_success;

    status = predict_proba(nsamp, nfeat, X_test, ldx_test, y_log_proba, n_class, ldy);
    if (status != da_status_success)
        return status;

    if (this->order == column_major) {
        for (da_int j = 0; j < nclass; j++) {
            for (da_int i = 0; i < nsamp; i++) {
                y_log_proba[ldy * j + i] = log(y_log_proba[ldy * j + i]);
            }
        }
    } else {
        for (da_int j = 0; j < nsamp; j++) {
            for (da_int i = 0; i < nclass; i++) {
                y_log_proba[j * ldy + i] = log(y_log_proba[j * ldy + i]);
            }
        }
    }
    return status;
}

template <typename T>
da_status decision_tree<T>::score(da_int nsamp, da_int nfeat, const T *X_test,
                                  da_int ldx_test, const da_int *y_test, T *accuracy) {

    const T *X_test_temp;
    T *utility_ptr1 = nullptr;
    da_int ldx_test_temp;

    if (accuracy == nullptr) {
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "mean_accuracy is not valid pointers.");
    }

    if (nfeat != n_features) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "nfeat = " + std::to_string(nfeat) +
                                   " doesn't match the expected value " +
                                   std::to_string(n_features) + ".");
    }

    if (!model_trained) {
        return da_error_bypass(this->err, da_status_out_of_date,
                               "The model has not yet been trained or the data it is "
                               "associated with is out of date.");
    }

    da_status status = this->store_2D_array(nsamp, nfeat, X_test, ldx_test, &utility_ptr1,
                                            &X_test_temp, ldx_test_temp, "n_samples",
                                            "n_features", "X_test", "ldx_test");
    if (status != da_status_success)
        return status;

    status = this->check_1D_array(nsamp, y_test, "n_samples", "y_test", 1);
    if (status != da_status_success)
        return status;

    node<T> *current_node;
    *accuracy = 0.;
    for (da_int i = 0; i < nsamp; i++) {
        current_node = &tree[0];
        while (!current_node->is_leaf) {
            if (current_node->prop == continuous ||
                current_node->prop == categorical_ordered) {
                T feat_val = X_test_temp[ldx_test_temp * current_node->feature + i];
                if (feat_val < current_node->x_threshold)
                    current_node = &tree[current_node->left_child_idx];
                else
                    current_node = &tree[current_node->right_child_idx];
            } else {
                da_int cat_val =
                    std::round(X_test_temp[ldx_test_temp * current_node->feature + i]);
                if (cat_val == current_node->category)
                    current_node = &tree[current_node->left_child_idx];
                else
                    current_node = &tree[current_node->right_child_idx];
            }
        }

        if (current_node->y_pred == y_test[i])
            *accuracy += (T)1.0;
    }
    *accuracy = *accuracy / (T)nsamp;
    if (utility_ptr1)
        delete[] (utility_ptr1);

    return da_status_success;
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

template <class T>
using score_fun_t = typename std::function<T(da_int, da_int, std::vector<da_int> &)>;

template double gini_score<double>(da_int n_samples, da_int n_class,
                                   std::vector<da_int> &count_classes);

template float gini_score<float>(da_int n_samples, da_int n_class,
                                 std::vector<da_int> &count_classes);

template double entropy_score<double>(da_int n_samples, da_int n_class,
                                      std::vector<da_int> &count_classes);

template float entropy_score<float>(da_int n_samples, da_int n_class,
                                    std::vector<da_int> &count_classes);

template double misclassification_score<double>(da_int n_samples,
                                                [[maybe_unused]] da_int n_class,
                                                std::vector<da_int> &count_classes);

template float misclassification_score<float>(da_int n_samples,
                                              [[maybe_unused]] da_int n_class,
                                              std::vector<da_int> &count_classes);
template class decision_tree<double>;
template class decision_tree<float>;
} // namespace da_decision_forest

} // namespace ARCH
