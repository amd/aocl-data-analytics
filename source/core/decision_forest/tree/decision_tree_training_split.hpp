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

#ifndef TREE_TRAINING_SPLIT_HPP
#define TREE_TRAINING_SPLIT_HPP

#include "aoclda.h"
#include "common/idx_sorting.hpp"
#include "decision_tree.hpp"
#include "macros.h"

namespace ARCH {
namespace da_decision_forest {

/* Utility functions */
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

/* Update_count_* functions: add to the left count all the values of samples_idx between start and end_idx */
template <typename T>
void decision_tree<T>::update_count_left(da_int start_idx, da_int end_idx,
                                         da_int &ns_left, split_workspace<T> &ws,
                                         std::vector<da_int> &samp) {
    for (da_int i = start_idx; i <= end_idx; i++) {
        da_int c = y[samp[i]];
        ws.count_left_classes[c]++;
        ns_left++;
    }
}
template <typename T>
void decision_tree<T>::update_count_left(da_int start_idx, da_int end_idx,
                                         da_int &ns_left, std::vector<da_int> &weights,
                                         split_workspace<T> &ws,
                                         std::vector<da_int> &samp) {
    for (da_int i = start_idx; i <= end_idx; i++) {
        da_int idx = samp[i];
        da_int c = y[idx];
        da_int w = weights[idx];
        ws.count_left_classes[c] += w;
        ns_left += w;
    }
}
template <typename T>
void decision_tree<T>::update_count_right(da_int start_idx, da_int end_idx,
                                          da_int &ns_right, split_workspace<T> &ws,
                                          std::vector<da_int> &samp) {
    for (da_int i = end_idx; i > start_idx; i--) {
        da_int c = y[samp[i]];
        ws.count_right_classes[c]++;
        ns_right++;
    }
}
template <typename T>
void decision_tree<T>::update_count_right(da_int start_idx, da_int end_idx,
                                          da_int &ns_right, std::vector<da_int> &weights,
                                          split_workspace<T> &ws,
                                          std::vector<da_int> &samp) {
    for (da_int i = end_idx; i > start_idx; i--) {
        da_int idx = samp[i];
        da_int c = y[idx];
        da_int w = weights[idx];
        ws.count_right_classes[c] += w;
        ns_right += w;
    }
}

template <typename T>
template <typename U>
bool decision_tree<T>::update_split_sorted(da_int sidx, da_int &next_idx, da_int end_idx,
                                           da_int ns, da_int &ns_left, da_int &ns_right,
                                           T &left_score, T &right_score, T &split_score,
                                           std::vector<U> &fv, split_workspace<T> &ws,
                                           std::vector<da_int> &samp) {
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
            update_count_left(sidx, next_idx, ns_left, bootstrap_sample_frequency, ws,
                              samp);
        else
            update_count_left(sidx, next_idx, ns_left, ws, samp);
        ns_right = ns - ns_left;
        for (da_int i = 0; i < n_class; i++)
            ws.count_right_classes[i] = count_classes[i] - ws.count_left_classes[i];
    } else {
        da_std::fill(ws.count_right_classes.begin(), ws.count_right_classes.end(), 0);
        ns_right = 0;
        if (bootstrap)
            update_count_right(next_idx, end_idx, ns_right, bootstrap_sample_frequency,
                               ws, samp);
        else
            update_count_right(next_idx, end_idx, ns_right, ws, samp);
        ns_left = ns - ns_right;
        for (da_int i = 0; i < n_class; i++)
            ws.count_left_classes[i] = count_classes[i] - ws.count_right_classes[i];
    }

    left_score = score_function(ns_left, n_class, ws.count_left_classes);
    right_score = score_function(ns_right, n_class, ws.count_right_classes);
    split_score = (left_score * ns_left + right_score * ns_right) / ns;

    return end_split_search;
}

/* Binned data */
template <typename T>
bool decision_tree<T>::update_node_histogram(const node<T> &nd, da_int feat_idx,
                                             split_workspace<T> &ws) {
    /* Auxialiary function for histogram based split computation.
     *
     * for a  given feature feat_idx and node nd, count the number of each response class
     * occurence fo each possible bin value.
     * On output, ws.node_hist[bin, c] will contain the number of samples from the node nd
     * that have both feature value equal to bin and class label equal to c.
     */
    da_int start_idx = feat_idx * n_samples;
    memset(ws.node_hist.data(), 0, ws.node_hist.size() * sizeof(da_int));
    memset(ws.hist_count_samples.data(), 0,
           ws.hist_count_samples.size() * sizeof(da_int));
    da_int const_cat_val = -1;
    bool const_feat = true;
    for (da_int i = nd.start_idx; i <= nd.end_idx; i++) {
        da_int idx = samples_idx[i];
        da_int c = y[idx];
        uint16_t cat = X_binned->binned_data[start_idx + idx];
        ws.node_hist[cat * n_class + c]++;
        ws.hist_count_samples[cat] += 1;
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
                                             std::vector<da_int> &weights,
                                             split_workspace<T> &ws) {
    /* Same as above, with weights for each sample index */
    da_int start_idx = feat_idx * n_samples;
    memset(ws.node_hist.data(), 0, ws.node_hist.size() * sizeof(da_int));
    memset(ws.hist_count_samples.data(), 0,
           ws.hist_count_samples.size() * sizeof(da_int));
    da_int const_cat_val = -1;
    bool const_feat = true;
    for (da_int i = nd.start_idx; i <= nd.end_idx; i++) {
        da_int idx = samples_idx[i];
        da_int w = weights[idx];
        da_int c = y[idx];
        uint16_t cat = X_binned->binned_data[start_idx + idx];
        ws.node_hist[cat * n_class + c] += w;
        ws.hist_count_samples[cat] += 1;
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
                                          da_int &ns_right, da_int cat_start_idx,
                                          split_workspace<T> &ws) {
    /* split strategy one vs all: all samples for the split bin value are set in the left child
     * while others are set in the right child.
     *
     * cat_start_idx: index of the split bin value column in ws.node_hist (e.g., bin * n_class)
     * Compute the number of occurrences of each response class in a potential split
     * On output, ws.count_[left|right]_classes will contain the number of samples in the
     * corresponding left and right split nodes.
     * ws.node_hist is expected to have been updated beforehand.
     */
    for (da_int c = 0; c < n_class; c++) {
        ws.count_left_classes[c] = ws.node_hist[cat_start_idx + c];
        ws.count_right_classes[c] = count_classes[c] - ws.count_left_classes[c];
        ns_left += ws.count_left_classes[c];
    }

    ns_right = nd.n_samples - ns_left;
}

template <typename T>
void decision_tree<T>::split_hist_ordered(const node<T> &nd, da_int &ns_left,
                                          da_int &ns_right, da_int cat_start_idx,
                                          split_workspace<T> &ws) {
    /* split strategy ordered: samples with a split bin value smaller than the split threshold
     * are set in the left child while others are set in the right child.
     */
    for (da_int c = 0; c < n_class; c++) {
        ws.count_left_classes[c] += ws.node_hist[cat_start_idx + c];
        ws.count_right_classes[c] = count_classes[c] - ws.count_left_classes[c];
        ns_left += ws.count_left_classes[c];
    }
    ns_right = nd.n_samples - ns_left;
}

template <typename T>
bool decision_tree<T>::compute_best_split_hist(const node<T> &nd, da_int feat_idx,
                                               split<T> &sp, split_workspace<T> &ws) {
    /* Main function for histograms based splits.
     * loop through all the bin values of feature feat_idx and update the split properties of sp
     * if a good split is found. */
    bool const_feat = false;
    if (bootstrap)
        const_feat = update_node_histogram(nd, feat_idx, bootstrap_sample_frequency, ws);
    else
        const_feat = update_node_histogram(nd, feat_idx, ws);

    if (const_feat)
        return const_feat;

    memset(ws.count_left_classes.data(), 0, n_class * sizeof(da_int));

    split_property prop = categorical_ordered;
    da_int n_cat = X_binned->nbins[feat_idx];
    if (n_cat < X_binned->max_bin)
        prop = (split_property)cat_split_strat;

    da_int old_ns_left = 0;
    da_int thresh_start_idx = feat_idx * (X_binned->max_bin - 1);
    for (da_int cat = 0; cat < n_cat; ++cat) {
        if (ws.hist_count_samples[cat] == 0)
            continue;
        da_int ns_left = 0, ns_right = 0;
        da_int cat_start_idx = cat * n_class;

        if (prop == categorical_onevall)
            split_hist_onevall(nd, ns_left, ns_right, cat_start_idx, ws);
        else
            split_hist_ordered(nd, ns_left, ns_right, cat_start_idx, ws);

        if (ns_left < min_node_sample || ns_left <= old_ns_left)
            continue;
        if (ns_right < min_node_sample)
            break;
        old_ns_left = ns_left;

        T left_score = score_function(ns_left, n_class, ws.count_left_classes);
        T right_score = score_function(ns_right, n_class, ws.count_right_classes);
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

/* Raw data */
template <typename T>
void decision_tree<T>::update_feature_values(da_int start_idx, da_int end_idx,
                                             da_int feat_idx, split_workspace<T> &ws,
                                             std::vector<da_int> &samp) {
    /* fill the ws.feature_values array with the values of X marked by the samples_idx and feat_idx */
    da_int col_idx = ldx * feat_idx;
    for (da_int i = start_idx; i <= end_idx; i++)
        ws.feature_values[i] = X[col_idx + samp[i]];
}

template <typename T>
void decision_tree<T>::split_raw_onevall(const node<T> &current_node, da_int feat_idx,
                                         split<T> &sp, split_workspace<T> &ws,
                                         std::vector<da_int> &samp) {
    sp.score = current_node.score;

    // fill ws.cat_feat_table, counting for each possible category of feat_idx
    // the number of occurrences of each response class in the samples
    // After the loop, column j of ws.cat_feat_table will countain the count of
    // each class (from response vector y) in the samples contained in the current node
    da_std::fill(ws.cat_feat_table.begin(), ws.cat_feat_table.end(), 0);
    for (da_int i = current_node.start_idx; i <= current_node.end_idx; i++) {
        da_int idx = samp[i];
        da_int c = y[idx];
        da_int cat = std::round(ws.feature_values[i]);
        ws.cat_feat_table[cat * n_class + c]++;
    }

    for (da_int cat = 0; cat < cat_feat[feat_idx]; cat++) {
        da_int ns_left = 0, ns_right = 0;
        for (da_int c = 0; c < n_class; c++) {
            ws.count_left_classes[c] = ws.cat_feat_table[cat * n_class + c];
            ns_left += ws.count_left_classes[c];
            ws.count_right_classes[c] = count_classes[c] - ws.count_left_classes[c];
        }
        ns_right = current_node.n_samples - ns_left;
        if (ns_left < min_node_sample)
            continue;
        if (ns_right < min_node_sample)
            continue;

        T left_score = score_function(ns_left, n_class, ws.count_left_classes);
        T right_score = score_function(ns_right, n_class, ws.count_right_classes);
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
void decision_tree<T>::split_raw_continuous(const node<T> &current_node, split<T> &sp,
                                            split_workspace<T> &ws,
                                            std::vector<da_int> &samp) {
    // Initialize the split, all nodes to the right child.
    // count_class, ws.samples_idx_local and ws.feature_values are required to be up to date
    std::copy(count_classes.begin(), count_classes.end(), ws.count_right_classes.begin());
    da_std::fill(ws.count_left_classes.begin(), ws.count_left_classes.end(), 0);
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
                            ws.feature_values, ws, samp);
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
            sp.threshold =
                (ws.feature_values[next_idx] + ws.feature_values[next_idx + 1]) / 2;
            sp.right_score = right_score;
            sp.left_score = left_score;
            sp.prop = continuous;
        }

        sidx = next_idx + 1;
    }
}

template <typename T>
bool decision_tree<T>::compute_best_split_raw(const node<T> &nd, da_int feat_idx,
                                              split<T> &sp, split_workspace<T> &ws,
                                              std::vector<da_int> &samp) {
    /* Main function for raw data splits. */
    update_feature_values(nd.start_idx, nd.end_idx, feat_idx, ws, samp);

    if ((split_property)cat_split_strat == categorical_ordered ||
        cat_feat[feat_idx] <= 0) {
        /* continuous data: samp is sorted according to the values of feat_idx
         * column in X and all possible split threshold values are checked. */
        if (cat_feat[feat_idx] != 1) {
            da_int node_obs = nd.end_idx - nd.start_idx + 1;
            multi_range_intro_sort(samp, ws.feature_values, nd.start_idx, node_obs,
                                   (da_int)(2 * std::log2(node_obs) + 2));
        }

        if (std::abs(ws.feature_values[nd.start_idx] - ws.feature_values[nd.end_idx]) <
            (T)1.0e-05)
            // feature is constant, mark to skip it
            return true;
        sp.feat_idx = feat_idx;
        split_raw_continuous(nd, sp, ws, samp);

    } else {
        /* categorical data in raw matrix: similar strategy used in binned data. */
        sp.feat_idx = feat_idx;
        split_raw_onevall(nd, feat_idx, sp, ws, samp);
    }

    return false;
}

template <typename T>
bool decision_tree<T>::compute_best_split(const node<T> &nd, da_int feat_idx,
                                          split<T> &sp, split_workspace<T> &ws,
                                          std::vector<da_int> &samp) {
    /* Main entry point for the split computation, dispatches to the correct function based
     * on split properties and strategies */
    bool const_feat;
    if (!use_hist)
        const_feat = compute_best_split_raw(nd, feat_idx, sp, ws, samp);
    else {
        const_feat = compute_best_split_hist(nd, feat_idx, sp, ws);
    }
    return const_feat;
}

} // namespace da_decision_forest
} // namespace ARCH

#endif
