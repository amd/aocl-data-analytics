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

#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include "aoclda.h"
#include "basic_handle.hpp"
#include "da_error.hpp"
#include "decision_tree_options.hpp"
#include "decision_tree_types.hpp"
#include "options.hpp"

#include <chrono>
#include <deque>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace da_decision_tree {

// Commenting out the timing utilities until we figure out a better way to do it
// using std::chrono::duration, std::chrono::seconds, std::chrono::duration_cast;
// using hr_clock = std::chrono::high_resolution_clock;

template <typename T> class node {
  public:
    // Tree data
    // left|right_child_idx: index in the tree (vector of Node) of the 2 children
    // If the node is a leaf, the children indices are ignored
    bool is_leaf = true;
    da_int left_child_idx = -1;
    da_int right_child_idx = -1;
    da_int depth = 0;
    T score = std::numeric_limits<T>::max();

    // prediction data
    // y_pred: contains the predicted class of the data if all children were pruned
    // feature: Index of the feature the node is branching on, ignore if leaf
    // x_threshold: branch to the left child if x[feature] < threashold, right otherwise
    da_int y_pred = 0;
    da_int feature = -1;
    T x_threshold = 0.0;

    // start|end_idx: all the sample indices in the node and its children are stored in
    // samples_idx[start_idx:end_idx]
    da_int start_idx = -1, end_idx = -1;
    da_int n_samples = 0;
};

template <typename T> struct split {
    /* Contains split information
     * feat_idx: index of the features we are splitting on
     * samp_idx: index in the sorted samples_idx array of the sample before the split
     * threshold: threshold for the feat_idx split feature
     * score: average score of the 2 children if created
     * left_score: score of the left child if created
     * right_score: score of the right child if created
     */
    da_int feat_idx, samp_idx;
    T score, threshold, left_score, right_score;

    void copy(split const &sp) {
        feat_idx = sp.feat_idx;
        samp_idx = sp.samp_idx;
        score = sp.score;
        threshold = sp.threshold;
        left_score = sp.left_score;
        right_score = sp.right_score;
    }
};

/* Compute the impurity of a node containing n_samples samples
 * In input, count_classes[i] is assumed to contain the number of occurences of class i within
 * the node samples */
template <class T>
using score_fun_t = typename std::function<T(da_int, da_int, std::vector<da_int> &)>;

template <class T>
T gini_score(da_int n_samples, da_int n_class, std::vector<da_int> &count_classes) {
    T score = 0.0;
    for (da_int c = 0; c < n_class; c++) {
        score += count_classes[c] * count_classes[c];
    }
    score = (T)1.0 - score / (n_samples * n_samples);
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
T misclassification_score(da_int n_samples, da_int n_class,
                          std::vector<da_int> &count_classes) {
    T score =
        (T)1.0 -
        ((T)*std::max_element(count_classes.begin(), count_classes.end())) / (T)n_samples;
    return score;
}

template <typename T> class decision_tree : public basic_handle<T> {

    // pointer to error trace
    da_errors::da_error_t *err = nullptr;

    bool model_trained = false;

    // user data. Never modified by the classifier
    // X[n_samples x n_features]: features -- floating point matrix, column major
    // y[n_samples]: labels -- integer array, 0,...,n_classes-1 values
    // n_obs: the number of observation to pick randomly from the total samples.
    //        After call to set_training_data, 0 < n_obs <= n_samples
    // depth: the depth of the tree once trained
    // ldx: X leading dimension
    const T *X = nullptr;
    const da_int *y = nullptr;
    da_int ldx;
    da_int n_samples = 0;
    da_int n_features = 0;
    da_int n_class = 0;
    da_int n_obs;
    da_int depth = 0;

    // tree structure
    // tree (vector): contains the all the nodes, each node stores the indices of its children
    // nodes_to_treat: double ended queue containing the indices of the nodes yet to be treated
    da_int n_nodes = 0;
    std::vector<node<T>> tree;
    std::deque<da_int> nodes_to_treat;

    // All memory to compute scores
    // samples_idx: size n_obs. used to store the indices covered by a given node.
    //              after a node is inserted in a tree, samples_idx[start_idx:end_idx] contains
    //              the indices of samples covered by a node and its children
    // samples_subset: optional array of samples index containing a subset samples_idx
    //                 (with potential repetition). Used mainly to get repeatable sequences
    //                 for testing purposes
    // count_classes: size n_class. Used to count the number of occurences of all classes in a set
    //                of samples
    // count_left|right_classes: same as count classes for potential children
    // feature_values: size n_samples. used to copy and sort the feature values while computing
    //                 the score of a node
    std::vector<da_int> samples_idx;
    da_int *samples_subset = nullptr;
    std::vector<da_int> count_classes;
    std::vector<da_int> count_left_classes, count_right_classes;
    std::vector<T> feature_values;

    // features_idx: size n_features. Vector containing all the indices of the features.
    //               primarily used to pick a random subselection of indices to consider
    //               for splitting a node
    std::vector<da_int> features_idx;

    // random number generation
    da_int seed;
    std::mt19937 mt_engine;

    // Scoring function
    score_fun_t<T> score_function;

    // Profiling info
    // duration<float> fit_time = seconds(0), sort_time = seconds(0),
    //                 split_time = seconds(0), count_class_time = seconds(0),
    //                 setup_time = seconds(0);

    // Optional parameter values.
    // set by reading the option registry if used by external user.
    // Set by the alternate constructor if used by a forest
    bool read_public_options = true;
    da_int max_depth, min_node_sample, method, prn_times, build_order, nfeat_split;
    T min_split_score, feat_thresh, min_improvement;
    bool bootstrap = false;

  public:
    da_options::OptionRegistry opts;
    // Constructor for public interfaces
    decision_tree(da_errors::da_error_t &err) {
        // assumes that err is valid
        this->err = &err;
        register_decision_tree_options<T>(opts);
    }
    // constructor bypassing the optional parameters for internal forest use
    // Values will NOT be checked
    decision_tree(da_int max_depth, da_int min_node_sample, da_int method,
                  da_int prn_times, da_int build_order, da_int nfeat_split, da_int seed,
                  T min_split_score, T feat_thresh, T min_improvement, bool bootstrap)
        : max_depth(max_depth), min_node_sample(min_node_sample), method(method),
          prn_times(prn_times), build_order(build_order), nfeat_split(nfeat_split),
          seed(seed), min_split_score(min_split_score), feat_thresh(feat_thresh),
          min_improvement(min_improvement), bootstrap(bootstrap) {
        this->err = nullptr;
        read_public_options = false;
    }

    da_status set_training_data(da_int n_samples, da_int n_features, const T *X,
                                da_int ldx, const da_int *y, da_int n_class = 0,
                                da_int n_obs = 0, da_int *samples_subset = nullptr);
    void count_class_occurences(std::vector<da_int> &class_occ, da_int start_idx,
                                da_int end_idx);
    void sort_samples(node<T> &node, da_int feat_idx);
    da_status add_node(da_int parent_idx, bool is_left, T score, da_int split_idx);
    da_int get_next_node_idx(da_int build_order);
    void find_best_split(node<T> &current_node, T feat_thresh, T maximum_split_score,
                         split<T> &sp);
    da_status fit();
    da_status predict(da_int nsamp, da_int n_features, const T *X_test, da_int ldx,
                      da_int *y_pred);
    da_status score(da_int nsamp, da_int nfeat, const T *X_test, da_int ldx,
                    da_int *y_test, T *accuracy);
    void clear_working_memory();

    void refresh() {
        model_trained = false;
        if (tree.capacity() > 0)
            tree = std::vector<node<T>>();
    }

    da_status get_result(da_result query, da_int *dim, T *result);
    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] da_int *result) {

        return da_warn_bypass(err, da_status_unknown_query,
                              "There are no integer results available for this API.");
    };

    // void print_timings();

    // getters for testing purposes
    std::vector<da_int> const &get_samples_idx() { return samples_idx; }
    std::vector<T> const &get_features_values() { return feature_values; }
    std::vector<da_int> const &get_count_classes() { return count_classes; }
    std::vector<da_int> const &get_count_left_classes() { return count_left_classes; }
    std::vector<da_int> const &get_count_right_classes() { return count_right_classes; }
    std::vector<da_int> const &get_features_idx() { return features_idx; }
    bool model_is_trained() { return model_trained; }
    std::vector<node<T>> const &get_tree() { return tree; }

    // setters for testing purposes
    void set_bootstrap(bool bs) { this->bootstrap = bs; }
};

// template <typename T> void decision_tree<T>::print_timings() {
//     std::cout << "Decision tree timings\n";
//     std::cout << "   Total fit time:   " << fit_time.count() << "s\n";
//     std::cout << "   Setup time:       " << setup_time.count() << "s\n";
//     std::cout << "   Sort time:        " << sort_time.count() << "s\n";
//     std::cout << "   Split time:       " << split_time.count() << "s\n";
//     std::cout << std::endl;
// }

template <typename T>
da_status decision_tree<T>::get_result(da_result query, da_int *dim, T *result) {

    if (!model_trained)
        return da_warn_bypass(
            this->err, da_status_unknown_query,
            "Handle does not contain data relevant to this query. Was the "
            "last call to the solver successful?");
    // Pointers were already tested in the generic get_result

    da_int rinfo_size = 5;
    switch (query) {
    case da_result::da_rinfo:
        if (*dim < rinfo_size) {
            *dim = rinfo_size;
            return da_warn_bypass(err, da_status_invalid_array_dimension,
                                  "The array is too small. Please provide an array of at "
                                  "least size: " +
                                      std::to_string(rinfo_size) + ".");
        }
        result[0] = (T)n_features;
        result[1] = (T)n_samples;
        result[2] = (T)n_obs;
        result[3] = (T)seed;
        result[4] = (T)depth;
        break;
    default:
        return da_warn_bypass(err, da_status_unknown_query,
                              "The requested result could not be found.");
    }
    return da_status_success;
}

/* */
template <typename T>
da_status decision_tree<T>::set_training_data(da_int n_samples, da_int n_features,
                                              const T *X, da_int ldx, const da_int *y,
                                              da_int n_class, da_int n_obs,
                                              da_int *samples_subset) {
    if (X == nullptr || y == nullptr)
        return da_error_bypass(this->err, da_status_invalid_input,
                               "Either X, or y are not valid pointers.");
    if (n_samples <= 0 || n_features <= 0) {
        return da_error_bypass(
            this->err, da_status_invalid_input,
            "n_samples = " + std::to_string(n_samples) +
                ", n_features = " + std::to_string(n_features) +
                ", the values of n_samples and n_features need to be greater than 0");
    }
    if (ldx < n_samples) {
        return da_error_bypass(
            this->err, da_status_invalid_input,
            "n_samples = " + std::to_string(n_samples) +
                ", ldx = " + std::to_string(ldx) +
                ", the value of ldx needs to be at least as big as the value "
                "of n_samples");
    }
    if (n_obs > n_samples || n_obs < 0) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "n_obs = " + std::to_string(n_obs) +
                                   ", it must be set between 0 and n_samples = " +
                                   std::to_string(n_samples));
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
    this->n_obs = n_obs;
    if (this->n_obs == 0)
        this->n_obs = this->n_samples;
    this->samples_subset = samples_subset;

    // initialize working memory
    // samples_idx contains all the indices in order [0:n_samples-1]
    try {
        samples_idx.resize(this->n_obs);
        count_classes.resize(this->n_class);
        feature_values.resize(this->n_obs);
        count_left_classes.resize(this->n_class);
        count_right_classes.resize(this->n_class);
        features_idx.resize(this->n_features);
    } catch (std::bad_alloc &) {                            // LCOV_EXCL_LINE
        return da_error_bypass(err, da_status_memory_error, // LCOV_EXCL_LINE
                               "Memory allocation error");
    }
    std::iota(features_idx.begin(), features_idx.end(), 0);

    return da_status_success;
}

template <class T>
void decision_tree<T>::count_class_occurences(std::vector<da_int> &class_occ,
                                              da_int start_idx, da_int end_idx) {
    std::fill(class_occ.begin(), class_occ.end(), 0);
    for (da_int i = start_idx; i <= end_idx; i++) {
        da_int idx = samples_idx[i];
        da_int c = y[idx];
        class_occ[c] += 1;
    }
}

/* Possible errors:
 * - memory
 */
template <class T>
da_status decision_tree<T>::add_node(da_int parent_idx, bool is_left, T score,

                                     da_int split_idx) {

    if (tree.size() <= n_nodes) {
        try {
            tree.resize(2 * tree.size() + 1);
        } catch (std::bad_alloc &) {                            // LCOV_EXCL_LINE
            return da_error_bypass(err, da_status_memory_error, // LCOV_EXCL_LINE
                                   "Memory allocation error");
        }
    }

    if (is_left) {
        tree[parent_idx].left_child_idx = n_nodes;
        tree[n_nodes].start_idx = tree[parent_idx].start_idx;
        tree[n_nodes].end_idx = split_idx;
    } else {
        tree[parent_idx].right_child_idx = n_nodes;
        tree[n_nodes].start_idx = split_idx + 1;
        tree[n_nodes].end_idx = tree[parent_idx].end_idx;
    }
    tree[n_nodes].depth = tree[parent_idx].depth + 1;
    if (tree[n_nodes].depth > this->depth)
        this->depth = tree[n_nodes].depth;
    tree[n_nodes].score = score;
    tree[n_nodes].n_samples = tree[n_nodes].end_idx - tree[n_nodes].start_idx + 1;
    // prediction: most represented class in the samples subset
    // TODO Should this be pre-computed ??
    count_class_occurences(count_classes, tree[n_nodes].start_idx, tree[n_nodes].end_idx);
    tree[n_nodes].y_pred = (da_int)std::distance(
        count_classes.begin(),
        std::max_element(count_classes.begin(), count_classes.end()));
    n_nodes += 1;

    return da_status_success;
}

template <class T> void decision_tree<T>::sort_samples(node<T> &nd, da_int feat_idx) {
    /* Sort samples_idx according to the values of a given feature.
     * On output:
     * - the values of samples_idx will be sorted between the start and end indices
     *   of the node nd
     * - feature_values[nd.start_idx:nd.end_idx] will contain the values of the feat_idx feature
     *   corresponding to the indices in samples_idx
     */
    // hr_clock::time_point start_clock_sort = hr_clock::now();

    std::vector<da_int>::iterator start = samples_idx.begin() + nd.start_idx;
    std::vector<da_int>::iterator stop =
        samples_idx.begin() + nd.start_idx + nd.n_samples;

    std::sort(start, stop, [&](const da_int &i1, const da_int &i2) {
        return X[ldx * feat_idx + i1] < X[ldx * feat_idx + i2];
    });
    for (da_int i = nd.start_idx; i <= nd.end_idx; i++)
        feature_values[i] = X[ldx * feat_idx + samples_idx[i]];
    // hr_clock::time_point stop_clock_sort = hr_clock::now();
    // sort_time += duration_cast<duration<float>>(stop_clock_sort - start_clock_sort);
}

template <class T> da_int decision_tree<T>::get_next_node_idx(da_int build_order) {
    // Get the next node index to treat in function of the building order selected.
    // LIFO: depth-first
    // FIFO: breadth-first
    da_int node_idx = -1;
    switch (build_order) {
    case depth_first:
        node_idx = nodes_to_treat.back();
        nodes_to_treat.pop_back();
        break;
    case breadth_first:
        node_idx = nodes_to_treat.front();
        nodes_to_treat.pop_front();
        break;
    }

    return node_idx;
}

/* Test all the possible splits and return the best one */
template <typename T>
void decision_tree<T>::find_best_split(node<T> &current_node, T feat_thresh,
                                       T maximum_split_score, split<T> &sp) {

    // hr_clock::time_point start_clock_split = hr_clock::now();

    // Initialize the split, all nodes to the right child.
    // count_class, samples_idx and feature_values are required to be up to date
    std::copy(count_classes.begin(), count_classes.end(), count_right_classes.begin());
    std::fill(count_left_classes.begin(), count_left_classes.end(), 0);
    T right_score = current_node.score, left_score = 0.0;
    da_int ns_left = 0;
    da_int ns_right = current_node.n_samples;
    sp.score = current_node.score;
    sp.samp_idx = -1;

    T split_score;
    da_int sidx = current_node.start_idx;
    while (sidx <= current_node.end_idx - 1) {
        da_int c = y[samples_idx[sidx]];
        count_left_classes[c] += 1;
        count_right_classes[c] -= 1;
        ns_left += 1;
        ns_right -= 1;

        // skip testing splits where feature values are too close
        while (sidx + 1 <= current_node.end_idx &&
               std::abs(feature_values[sidx + 1] - feature_values[sidx]) < feat_thresh) {
            c = y[samples_idx[sidx + 1]];
            count_left_classes[c]++;
            count_right_classes[c]--;
            ns_left += 1;
            ns_right -= 1;
            sidx++;
        }
        if (sidx == current_node.end_idx)
            // All samples are in the left child. Do not check the split
            break;

        // TODO: cheaper score update? Does it matter?
        left_score = score_function(ns_left, n_class, count_left_classes);
        right_score = score_function(ns_right, n_class, count_right_classes);
        split_score =
            (left_score * ns_left + right_score * ns_right) / current_node.n_samples;
        // Consider the split only if it brings at least minimum improvement
        // compared to the parent node
        if (split_score < sp.score && split_score < maximum_split_score) {
            sp.score = split_score;
            sp.samp_idx = sidx;
            sp.threshold = (feature_values[sidx] + feature_values[sidx + 1]) / 2;
            sp.right_score = right_score;
            sp.left_score = left_score;
        }

        sidx++;
    }
    // hr_clock::time_point stop_clock_split = hr_clock::now();
    // split_time += duration_cast<duration<float>>(stop_clock_split - start_clock_split);
}

template <typename T> da_status decision_tree<T>::fit() {

    if (model_trained)
        // Nothing to do, exit
        return da_status_success;

    // hr_clock::time_point start_clock_fit = hr_clock::now();

    // Extract options
    if (read_public_options) {
        std::string opt_val;
        bool opt_pass = true;
        opt_pass &= opts.get("maximum depth", max_depth) == da_status_success;
        opt_pass &= opts.get("scoring function", opt_val, method) == da_status_success;
        opt_pass &=
            opts.get("Node minimum samples", min_node_sample) == da_status_success;
        opt_pass &= opts.get("Minimum split score", min_split_score) == da_status_success;
        opt_pass &=
            opts.get("tree building order", opt_val, build_order) == da_status_success;
        opt_pass &= opts.get("maximum features", nfeat_split) == da_status_success;
        opt_pass &= opts.get("seed", seed) == da_status_success;
        opt_pass &= opts.get("feature threshold", feat_thresh) == da_status_success;
        opt_pass &=
            opts.get("minimum split improvement", min_improvement) == da_status_success;
        opt_pass &= opts.get("print timings", opt_val, prn_times) == da_status_success;
        if (!opt_pass)
            return da_error_bypass(
                err, da_status_internal_error, // LCOV_EXCL_LINE
                "Unexpected error while reading the optional parameters.");
    }

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

    // Allocate the tree accounting for a full binary tree of depth 10 (or maximum depth)
    try {
        size_t init_capacity = ((da_int)1 << std::min(max_depth, (da_int)9)) + (da_int)1;
        tree.resize(init_capacity);
    } catch (std::bad_alloc &) {                            // LCOV_EXCL_LINE
        return da_error_bypass(err, da_status_memory_error, // LCOV_EXCL_LINE
                               "Memory allocation error");
    }

    if (!bootstrap) {
        // take all the samples
        std::iota(samples_idx.begin(), samples_idx.end(), 0);
    } else {
        if (samples_subset == nullptr) {
            // Fill the index vector with a random selection with replacement
            std::uniform_int_distribution<da_int> uniform_dist(0, n_samples - 1);
            std::generate(samples_idx.begin(), samples_idx.end(),
                          [&uniform_dist, &mt_engine = this->mt_engine]() {
                              return uniform_dist(mt_engine);
                          });
        } else {
            // copy the input from the samples_subset array.
            // As it is intended mainly for testing, samples_subset is NOT validated.
            for (da_int i = 0; i < n_obs; i++)
                samples_idx[i] = samples_subset[i];
        }
    }

    // Initialize the root node
    n_nodes = 1;
    tree[0].start_idx = 0;
    tree[0].end_idx = n_obs - 1;
    tree[0].depth = 1;
    tree[0].n_samples = n_obs;
    count_class_occurences(count_classes, 0, n_obs - 1);
    tree[0].score = score_function(n_obs, n_class, count_classes);
    tree[0].y_pred = (da_int)std::distance(
        count_classes.begin(),
        std::max_element(count_classes.begin(), count_classes.end()));

    // Insert the root node in the queue if the maximum depth is big enough
    // TODO/Discuss should we check for all memory reallocation??
    if (max_depth > 1)
        nodes_to_treat.push_back(0);

    // hr_clock::time_point stop_clock_setup = hr_clock::now();
    // setup_time = duration_cast<duration<float>>(stop_clock_setup - start_clock_fit);
    split<T> sp, best_split;
    while (!nodes_to_treat.empty()) {
        da_int node_idx = get_next_node_idx(build_order);
        node<T> &current_node = tree[node_idx];
        T maximum_split_score = current_node.score - min_improvement;

        // explore the candidate features for splitting
        // Randomly shuffle the index array and explore the first nfeat_split
        if (nfeat_split < n_features)
            std::shuffle(features_idx.begin(), features_idx.end(), mt_engine);
        best_split.score = current_node.score;
        best_split.feat_idx = -1;
        count_class_occurences(count_classes, current_node.start_idx,
                               current_node.end_idx);
        for (da_int j = 0; j < nfeat_split; j++) {
            da_int feat_idx = features_idx[j];
            sort_samples(current_node, feat_idx);
            sp.feat_idx = feat_idx;
            find_best_split(current_node, feat_thresh, maximum_split_score, sp);

            if (sp.score < best_split.score) {
                best_split.copy(sp);
            }
        }

        // Split the node and add the 2 children
        if (best_split.feat_idx != -1) {
            current_node.is_leaf = false;
            current_node.feature = best_split.feat_idx;
            current_node.x_threshold = best_split.threshold;

            // sort again the samples according to the chosen feature
            sort_samples(current_node, current_node.feature);

            // add chilren nodes and push them into the queue
            // if potential for further improvements is still high enough
            add_node(node_idx, false, best_split.right_score, best_split.samp_idx);
            if (best_split.right_score > min_split_score &&
                tree[n_nodes - 1].n_samples >= min_node_sample &&
                current_node.depth < max_depth - 1)
                nodes_to_treat.push_back(n_nodes - 1);
            add_node(node_idx, true, best_split.left_score, best_split.samp_idx);
            if (best_split.left_score > min_split_score &&
                tree[n_nodes - 1].n_samples >= min_node_sample &&
                current_node.depth < max_depth - 1)
                nodes_to_treat.push_back(n_nodes - 1);
        }
    }

    model_trained = true;
    // hr_clock::time_point stop_clock_fit = hr_clock::now();
    // fit_time = duration_cast<duration<float>>(stop_clock_fit - start_clock_fit);
    // if (prn_times)
    //     print_timings();
    return da_status_success;
}

template <typename T>
da_status decision_tree<T>::predict(da_int nsamp, da_int nfeat, const T *X_test,
                                    da_int ldx_test, da_int *y_pred) {
    if (X_test == nullptr || y_pred == nullptr) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "Either X_test, or y_pred are not valid pointers.");
    }
    if (nsamp <= 0) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "n_samples = " + std::to_string(nsamp) +
                                   ", it must be greater than 0.");
    }
    if (nfeat != n_features) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "n_features = " + std::to_string(nfeat) +
                                   " doesn't match the expected value " +
                                   std::to_string(n_features) + ".");
    }
    if (ldx_test < nsamp) {
        return da_error_bypass(
            this->err, da_status_invalid_input,
            "nsamp = " + std::to_string(nsamp) + ", ldx = " + std::to_string(ldx_test) +
                ", the value of ldx needs to be at least as big as the value "
                "of nsamp");
    }

    if (!model_trained) {
        return da_error_bypass(this->err, da_status_out_of_date,
                               "The model has not yet been trained or the data it is "
                               "associated with is out of date.");
    }

    // fill y_pred with the values of all the requested samples
    node<T> *current_node;
    for (da_int i = 0; i < nsamp; i++) {
        current_node = &tree[0];
        while (!current_node->is_leaf) {
            T feat_val = X_test[ldx_test * current_node->feature + i];
            if (feat_val < current_node->x_threshold)
                current_node = &tree[current_node->left_child_idx];
            else
                current_node = &tree[current_node->right_child_idx];
        }
        y_pred[i] = current_node->y_pred;
    }

    return da_status_success;
}

template <typename T>
da_status decision_tree<T>::score(da_int nsamp, da_int nfeat, const T *X_test,
                                  da_int ldx_test, da_int *y_test, T *accuracy) {
    if (X_test == nullptr || y_test == nullptr || accuracy == nullptr) {
        return da_error_bypass(
            this->err, da_status_invalid_input,
            "Either X_test, y_test or accuracy are not valid pointers.");
    }
    if (nsamp <= 0) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "nsamp = " + std::to_string(nsamp) +
                                   ", it must be greater than 0.");
    }
    if (nfeat != n_features) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "nfeat = " + std::to_string(nfeat) +
                                   " doesn't match the expected value " +
                                   std::to_string(n_features) + ".");
    }
    if (ldx_test < nsamp) {
        return da_error_bypass(
            this->err, da_status_invalid_input,
            "nsamp = " + std::to_string(nsamp) + ", ldx = " + std::to_string(ldx_test) +
                ", the value of ldx needs to be at least as big as the value "
                "of nsamp");
    }
    if (!model_trained) {
        return da_error_bypass(this->err, da_status_out_of_date,
                               "The model has not yet been trained or the data it is "
                               "associated with is out of date.");
    }

    node<T> *current_node;
    *accuracy = 0.;
    for (da_int i = 0; i < nsamp; i++) {
        current_node = &tree[0];
        while (!current_node->is_leaf) {
            T feat_val = X_test[ldx_test * current_node->feature + i];
            if (feat_val < current_node->x_threshold)
                current_node = &tree[current_node->left_child_idx];
            else
                current_node = &tree[current_node->right_child_idx];
        }
        if (current_node->y_pred == y_test[i])
            *accuracy += (T)1.0;
    }
    *accuracy = *accuracy / (T)nsamp;

    return da_status_success;
}

template <typename T> void decision_tree<T>::clear_working_memory() {
    // TODO Should we switch all vectors to raw pointers instead?
    samples_idx = std::vector<da_int>();
    count_classes = std::vector<da_int>();
    feature_values = std::vector<T>();
    count_left_classes = std::vector<da_int>();
    count_right_classes = std::vector<da_int>();
    features_idx = std::vector<da_int>();
}

} // namespace da_decision_tree

#endif