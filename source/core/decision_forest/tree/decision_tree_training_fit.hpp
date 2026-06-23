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

#ifndef TREE_TRAINING_FIT_HPP
#define TREE_TRAINING_FIT_HPP

#include "aoclda.h"
#include "common/idx_sorting.hpp"
#include "decision_tree.hpp"
#include "macros.h"

namespace ARCH {
namespace da_decision_forest {

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

template <typename T>
da_status decision_tree<T>::split_node_and_add_children(da_int node_idx,
                                                        split<T> &best_split) {
    node<T> &current_node = tree[node_idx];

    // Project back the discovered constant features to the parent
    if (node_idx > 0 &&
        tree[current_node.parent_idx].const_feat_idx > current_node.const_feat_idx)
        tree[current_node.parent_idx].const_feat_idx = current_node.const_feat_idx;

    // Split the node and add the 2 children
    if (best_split.feat_idx != -1) {
        current_node.is_leaf = false;
        current_node.feature = best_split.feat_idx;

        // Partition the samples according to the chosen feature and its threshold
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

    return da_status_success;
}

template <typename T> da_status decision_tree<T>::fit_serial() {
    da_status status = da_status_success;
    split<T> best_split;
    split<T> sp;
    split_workspace<T> &ws = thread_workspaces[0];

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

        // Find the best split across selected features
        sp.score = current_node.score;
        sp.feat_idx = -1;
        da_int feat_to_draw = nfeat_split;
        da_int fvar_idx = 0;
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

            bool const_feat =
                compute_best_split(current_node, feat_idx, sp, ws, samples_idx);

            if (const_feat) {
                current_node.const_feat_idx -= 1;
                features_idx[fvar_idx] = features_idx[current_node.const_feat_idx];
                features_idx[current_node.const_feat_idx] = feat_idx;
                continue;
            } else
                fvar_idx++;

            if (sp.feat_idx != -1 && sp.score < best_split.score)
                best_split.copy(sp);
        }

        status = split_node_and_add_children(node_idx, best_split);
        if (status != da_status_success)
            return status;
    }
    return status;
}

template <typename T> da_status decision_tree<T>::fit_parallel() {
    /* Fit the tree in a single persistent parallel region
     * One fork/join per tree instead of per node: reduce thread synchronization overhead in nested parallel regions from forests.
     * Sequential parts use omp single, parallel feature evaluation uses omp for.
     */
    da_status status = da_status_success;
    bool done = false;
    da_int node_idx = 0;
    da_int feat_to_draw = 0;
    da_int n_selected = 0;
    split<T> best_split;

#pragma omp parallel num_threads(n_threads_split)
    {
        bool local_done = false;
        while (!local_done) {
            // Sequential section: node selection, class counting, feature pre-selection
#pragma omp single
            {
                if (nodes_to_treat.empty()) {
                    done = true;
                    n_selected = 0;
                } else {
                    node_idx = get_next_node_idx();
                    node<T> &current_node = tree[node_idx];

                    if (node_idx > 0 && tree[current_node.parent_idx].const_feat_idx <
                                            current_node.const_feat_idx)
                        current_node.const_feat_idx =
                            tree[current_node.parent_idx].const_feat_idx;

                    best_split.score = current_node.score;
                    best_split.feat_idx = -1;
                    if (node_idx > 0) {
                        if (bootstrap)
                            count_class_occurences(count_classes, current_node.start_idx,
                                                   current_node.end_idx,
                                                   bootstrap_sample_frequency);
                        else
                            count_class_occurences(count_classes, current_node.start_idx,
                                                   current_node.end_idx);
                    }

                    feat_to_draw = nfeat_split;

                    // Pre-select features
                    selected_features.clear();
                    da_int fvar_idx = 0;
                    while (fvar_idx < feat_to_draw &&
                           fvar_idx < current_node.const_feat_idx) {
                        if (nfeat_split < n_features) {
                            std::uniform_int_distribution<da_int> uniform_dist(
                                fvar_idx, n_features - 1);
                            da_int random_feat_idx = uniform_dist(mt_engine);
                            if (random_feat_idx >= current_node.const_feat_idx) {
                                feat_to_draw--;
                                continue;
                            }
                            std::swap(features_idx[fvar_idx],
                                      features_idx[random_feat_idx]);
                        }
                        selected_features.push_back(features_idx[fvar_idx]);
                        fvar_idx++;
                    }

                    n_selected = fvar_idx;

                    // Initialize thread workspaces
                    for (da_int t = 0; t < n_threads_split; t++) {
                        thread_workspaces[t].best_split.score = current_node.score;
                        thread_workspaces[t].best_split.feat_idx = -1;
                        thread_workspaces[t].const_feats.clear();
                    }
                }
            }
            // implicit barrier + flush: all threads see updated shared state.
            // Copy done flag to thread-local variable to avoid compiler
            // caching the shared bool across iterations.
            local_done = done;

            // Parallel feature evaluation
#pragma omp for schedule(dynamic)
            for (da_int i = 0; i < n_selected; i++) {
                node<T> &current_node = tree[node_idx];
                da_int tid = omp_get_thread_num();
                split_workspace<T> &ws = thread_workspaces[tid];
                split<T> local_sp;
                local_sp.score = current_node.score;
                da_int feat_idx = selected_features[i];

                auto samp = std::ref(samples_idx);
                if (!use_hist) {
                    // the continuous data path sorts the values of samples_idx
                    // a local copy of the relevant node range is needed for each thread
                    std::copy(samples_idx.begin() + current_node.start_idx,
                              samples_idx.begin() + current_node.end_idx + 1,
                              ws.samples_idx_local.begin() + current_node.start_idx);
                    samp = std::ref(ws.samples_idx_local);
                }

                bool const_feat =
                    compute_best_split(current_node, feat_idx, local_sp, ws, samp);
                if (const_feat) {
                    ws.const_feats.push_back(feat_idx);
                } else if (local_sp.score < ws.best_split.score) {
                    ws.best_split.copy(local_sp);
                }
            }

            // Sequential section: reduce results, partition, add children
#pragma omp single
            {
                if (!done) {
                    node<T> &current_node = tree[node_idx];
                    // Reduce best splits and constant features
                    for (da_int t = 0; t < n_threads_split; t++) {
                        if (thread_workspaces[t].best_split.feat_idx != -1 &&
                            thread_workspaces[t].best_split.score < best_split.score)
                            best_split.copy(thread_workspaces[t].best_split);

                        for (da_int cf : thread_workspaces[t].const_feats) {
                            for (da_int j = 0; j < current_node.const_feat_idx; j++) {
                                if (features_idx[j] == cf) {
                                    current_node.const_feat_idx -= 1;
                                    features_idx[j] =
                                        features_idx[current_node.const_feat_idx];
                                    features_idx[current_node.const_feat_idx] = cf;
                                    break;
                                }
                            }
                        }
                    }
                    status = split_node_and_add_children(node_idx, best_split);
                }
            }
        }
    }
    return status;
}

template <typename T> da_status decision_tree<T>::fit() {
    da_status status = da_status_success;

    if (!this->init_done)
        return da_error_bypass(this->err, da_status_no_data,
                               "No data has been passed to the handle.");

    if (this->model_trained)
        // Nothing to do, exit
        return da_status_success;

    status = read_options();
    if (status != da_status_success)
        return status; // Error message already filled

    if (nfeat_split == 0 || nfeat_split > n_features) {
        // All the features are to be considered in splitting a node
        nfeat_split = n_features;
    }

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

    if (n_threads_split > 1) {
        status = fit_parallel();
    } else {
        status = fit_serial();
    }
    this->model_trained = true;
    return status;
}

} // namespace da_decision_forest
} // namespace ARCH

#endif
