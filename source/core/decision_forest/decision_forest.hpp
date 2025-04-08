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
#include "basic_handle.hpp"
#include "da_omp.hpp"
#include "decision_tree_types.hpp"
#include "macros.h"
#include "options.hpp"

#include <deque>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

namespace ARCH {

namespace da_decision_forest {

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
    // x_threshold: branch to the left child if x[feature] < threshold, right otherwise
    da_int y_pred = 0;
    da_int feature = -1;
    T x_threshold = 0.0;

    // start|end_idx: all the sample indices in the node and its children are stored in
    // samples_idx[start_idx:end_idx]
    da_int start_idx = -1, end_idx = -1;
    da_int n_samples = 0;
    void dummy();
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

    void copy(split const &sp);
};

/* Compute the impurity of a node containing n_samples samples.
 * On input, count_classes[i] is assumed to contain the number of
 * occurrences of class i within the node samples. */
template <class T>
using score_fun_t = typename std::function<T(da_int, da_int, std::vector<da_int> &)>;

template <typename T> class decision_tree : public basic_handle<T> {

    bool model_trained = false;
    da_int predict_proba_opt = true;

    // user data. Never modified by the classifier
    // X[n_samples x n_features]: features -- floating point matrix, column major
    // y[n_samples]: labels -- integer array, 0,...,n_classes-1 values
    // n_obs: the number of observations to pick randomly from the total samples.
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

    //Utility pointer to column major allocated copy of user's data
    T *X_temp = nullptr;

    // Tree structure
    // tree (vector): contains the all the nodes, each node stores the indices of its children
    // class_props (vector): contains the proportions in each class for each node
    // nodes_to_treat: double ended queue containing the indices of the nodes yet to be treated
    da_int n_nodes = 0, n_leaves = 0;
    std::vector<node<T>> tree;
    std::vector<T> class_props;
    std::deque<da_int> nodes_to_treat;

    // All memory to compute scores
    // samples_idx: size n_obs. used to store the indices covered by a given node.
    //              after a node is inserted in a tree, samples_idx[start_idx:end_idx] contains
    //              the indices of samples covered by a node and its children
    // samples_subset: optional array of samples index containing a subset samples_idx
    //                 (with potential repetition). Used mainly to get repeatable sequences
    //                 for testing purposes
    // count_classes: size n_class. Used to count the number of occurrences of all classes in a set
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

    // Random number generation
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
    da_int max_depth, min_node_sample, method, prn_times, build_order, nfeat_split, seed,
        sort_method;
    T min_split_score, feat_thresh, min_improvement;
    bool bootstrap = false;

  public:
    // Constructor for public interfaces
    decision_tree(da_errors::da_error_t &err);
    // Constructor bypassing the optional parameters for internal forest use
    // Values will NOT be checked
    decision_tree(da_int max_depth, da_int min_node_sample, da_int method,
                  da_int prn_times, da_int build_order, da_int nfeat_split, da_int seed,
                  da_int sort_method, T min_split_score, T feat_thresh, T min_improvement,
                  bool bootstrap);
    ~decision_tree();
    da_status set_training_data(da_int n_samples, da_int n_features, const T *X,
                                da_int ldx, const da_int *y, da_int n_class = 0,
                                da_int n_obs = 0, da_int *samples_subset = nullptr);
    void count_class_occurences(std::vector<da_int> &class_occ, da_int start_idx,
                                da_int end_idx);
    void sort_samples(node<T> &nd, da_int feat_idx);

    void partition_samples(const node<T> &nd);
    da_status add_node(da_int parent_idx, bool is_left, T score, da_int split_idx);
    da_int get_next_node_idx(da_int build_order);
    void find_best_split(node<T> &current_node, T feat_thresh, T maximum_split_score,
                         split<T> &sp);
    da_status fit();
    da_status predict(da_int nsamp, da_int n_features, const T *X_test, da_int ldx,
                      da_int *y_pred, da_int mode = 0);
    da_status predict_proba(da_int nsamp, da_int n_features, const T *X_test, da_int ldx,
                            T *y_pred, da_int n_class, da_int ldy, da_int mode = 0);
    da_status predict_log_proba(da_int nsamp, da_int n_features, const T *X_test,
                                da_int ldx, T *y_pred, da_int n_class, da_int ldy);
    da_status score(da_int nsamp, da_int nfeat, const T *X_test, da_int ldx,
                    const da_int *y_test, T *accuracy);
    void clear_working_memory();

    void refresh();

    da_status resize_tree(size_t new_size);

    da_status get_result(da_result query, da_int *dim, T *result);
    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] da_int *result);

    // Getters for testing purposes
    std::vector<da_int> const &get_samples_idx();
    std::vector<T> const &get_features_values();
    std::vector<da_int> const &get_count_classes();
    std::vector<da_int> const &get_count_left_classes();
    std::vector<da_int> const &get_count_right_classes();
    std::vector<da_int> const &get_features_idx();
    bool model_is_trained();
    std::vector<node<T>> const &get_tree();

    // Setters for testing purposes
    void set_bootstrap(bool bs);
};

using namespace da_errors;

template <typename T> class random_forest : public basic_handle<T> {

    bool model_trained = false;

    // User data. Never modified by the classifier
    // X[n_samples X n_features]: features -- floating point matrix, column major
    // y[n_samples]: labels -- integer array, 0,...,n_classes-1 values
    const T *X = nullptr;
    const da_int *y = nullptr;
    da_int n_samples = 0;
    da_int ldx = 0;
    da_int n_features = 0;
    da_int n_class = 0;

    //Utility pointer to column major allocated copy of user's data
    T *X_temp = nullptr;

    // Options
    da_int n_tree = 0;
    da_int seed, n_obs;
    da_int block_size;

    // Model data
    std::vector<std::unique_ptr<decision_tree<T>>> forest;

  public:
    random_forest(da_errors::da_error_t &err);
    ~random_forest();
    da_status set_training_data(da_int n_samples, da_int n_features, const T *X,
                                da_int ldx, const da_int *y, da_int n_class = 0);
    da_status fit();
    void parallel_count_classes(const T *X_test, da_int ldx_test, const da_int &n_blocks,
                                const da_int &block_size, const da_int &block_rem,
                                const da_int &n_threads,
                                std::vector<da_int> &count_classes,
                                std::vector<da_int> &y_pred_tree);
    da_status predict(da_int nn_samples, da_int n_features, const T *X, da_int ldx,
                      da_int *y_pred);
    da_status predict_proba(da_int nsamp, da_int nfeat, const T *X_test, da_int ldx_test,
                            T *y_proba, da_int nclass, da_int ldy);
    da_status predict_log_proba(da_int nsamp, da_int n_features, const T *X_test,
                                da_int ldx, T *y_pred, da_int n_class, da_int ldy);
    da_status score(da_int nsamp, da_int nfeat, const T *X_test, da_int ldx_test,
                    const da_int *y_test, T *score);

    da_status get_result(da_result query, da_int *dim, T *result);
    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] da_int *result);
};

template <class T>
using score_fun_t = typename std::function<T(da_int, da_int, std::vector<da_int> &)>;

template <class T>
T gini_score(da_int n_samples, da_int n_class, std::vector<da_int> &count_classes);

template <class T>
T entropy_score(da_int n_samples, da_int n_class, std::vector<da_int> &count_classes);

template <class T>
T misclassification_score(da_int n_samples, [[maybe_unused]] da_int n_class,
                          std::vector<da_int> &count_classes);

} // namespace da_decision_forest

} // namespace ARCH
