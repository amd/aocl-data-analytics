/*
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "common/histogram.hpp"
#include "common/scoring.hpp"
#include "common/tree_options_types.hpp"
#include "da_omp.hpp"
#include "da_utils.hpp"
#include "macros.h"
#include "model_persistence.hpp"
#include "options.hpp"

#include <deque>
#include <random>

namespace ARCH {

namespace da_decision_forest {
using namespace da_tree_options_types;

template <typename T> class node {
  public:
    // Tree data
    // parent_idx: Index of the parent in the vector of node
    // left|right_child_idx: index in the tree (vector of Node) of the 2 children
    // If the node is a leaf, the children indices are ignored
    bool is_leaf = true;
    da_int parent_idx = -1;
    da_int left_child_idx = -1;
    da_int right_child_idx = -1;
    da_int depth = 0;
    T score = std::numeric_limits<T>::max();

    // prediction data
    // prop: wether the split is on continuous or categorical data
    // y_pred: contains the predicted class of the data if all children were pruned
    // feature: Index of the feature the node is branching on, ignore if leaf
    // x_threshold: [if prop is continuous]
    //               branch to the left child if x[feature] < threshold, right otherwise
    // category: [if prop is categorical]
    //            elements of this category are in the left node, rest in right node
    split_property prop = continuous;
    da_int y_pred = 0;
    da_int feature = -1;
    T x_threshold = 0.0;
    da_int category = -1;

    // start|end_idx: all the sample indices in the node and its children are stored in
    // samples_idx[start_idx:end_idx]
    da_int start_idx = -1, end_idx = -1;
    da_int n_samples = 0;

    // const_feat_idx: features indices in [const_feat_idx, n_features] are constant for
    //                 the node and all its children. Only used if the strategy is 'depth first'
    da_int const_feat_idx = std::numeric_limits<da_int>::max();
    da_int children_const_idx = std::numeric_limits<da_int>::max();
    void dummy();

    da_status serialize(da_model_persistence::serialization_buffer &buffer);
};

template <typename T> struct split {
    /* Contains split information
     * prop: is the split made on a continuous or categorical feature?
     * feat_idx: index of the features we are splitting on
     * score: average score of the 2 children if created
     * left_score: score of the left child if created
     * right_score: score of the right child if created
     * samp_idx: index in the sorted samples_idx array of the sample before the split
     *
     * Continuous split info:
     * threshold: threshold for the feat_idx split feature
     *
     * Categorical split info:
     * category: index of the category we are splitting on,
     *           elements of this category are in the left node, rest in right node
     */

    split_property prop = continuous;
    da_int feat_idx;
    T score, left_score = 0.0, right_score = 0.0;
    da_int samp_idx = -1;
    // Continuous split
    T threshold = 0.0;
    // categorical split
    da_int category = -1;

    void copy(split const &sp);
};

/* Per-thread workspace for split computation */
template <typename T> struct split_workspace {
    // Class count buffers for left/right children
    std::vector<da_int> count_left_classes;
    std::vector<da_int> count_right_classes;

    // Raw data buffers
    std::vector<T> feature_values;
    std::vector<da_int> cat_feat_table;

    // Histogram buffers
    std::vector<da_int> node_hist;
    std::vector<da_int> hist_count_samples;

    // Local copy of samples_idx node range
    // (only allocated for sorting based splits)
    std::vector<da_int> samples_idx_local;

    // Results
    split<T> best_split;
    std::vector<da_int> const_feats;
};

/* Compute the impurity of a node containing n_samples samples.
* On input, count_classes[i] is assumed to contain the number of
* occurrences of class i within the node samples. */
template <class T>
using score_fun_t = typename std::function<T(da_int, da_int, std::vector<da_int> &)>;

template <typename T> class decision_tree : public basic_handle<T> {

    bool init_done = false;
    da_int predict_proba_opt = true;

    // user data. Never modified by the classifier
    // X[n_samples x n_features]: features -- floating point matrix, column major
    // y[n_samples]: labels -- integer array, 0,...,n_classes-1 values
    // n_obs: the number of unique observations to pick randomly from the total samples.
    //        After call to set_training_data, 0 < n_obs <= n_samples
    // n_obs_total: total number of observations used for training, including duplicates
    // depth: the depth of the tree once trained
    // ldx: X leading dimension
    // usr_categorical_feat[n_features]: usr_categorical_feat[i] contains the number of categories for feature i if it is categorical.
    //                               <= 0 if feature i is continuous
    //                               if usr_categorical_feat == nullptr, all features are continuous
    const T *X = nullptr;
    const da_int *y = nullptr;
    const da_int *usr_categorical_feat = nullptr;
    da_int ldx;
    da_int n_samples = 0;
    da_int n_features = 0;
    da_int n_class = 0;
    da_int n_obs, n_obs_total;
    da_int depth = 0;

    // Utility pointer to column major allocated copy of user's data
    T *X_temp = nullptr;

    // Tree structure
    // tree (vector): contains all the nodes, each node stores the indices of its children
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
    // bootstrap_sample_frequency: size n_samples. Used to store the frequency of each sample if bootstrap
    //                             is selected.
    std::vector<da_int> samples_idx;
    da_int *samples_subset = nullptr;
    std::vector<da_int> count_classes;
    std::vector<da_int> bootstrap_sample_frequency;

    // Used when splits are computed on raw data (no histograms)
    // max_cat: The maximum number of different categories if categorical variables are present in X
    std::vector<da_int> cat_feat;
    da_int max_cat = 0;

    // Histogram: data is quantized into max_bin categories
    // X_binned: class containing the binned X and auxialliary routines
    // internal_bins: true if the bins are computed inside the tree
    bool internal_bins = false;
    bins<T> *X_binned = nullptr;

    // features_idx: size n_features. Vector containing all the indices of the features.
    //               primarily used to pick a random subselection of indices to consider
    //               for splitting a node
    std::vector<da_int> features_idx;

    // thread_workspace(n_thrads_split): per-thread workspaces for parallel feature evaluation
    //        When executed in serial, thread_workspaces[0] is used.
    // selected_features (n_features): pre-selected features for parallel split evaluation
    da_int n_threads_split = 1;
    std::vector<split_workspace<T>> thread_workspaces;
    std::vector<da_int> selected_features;

    // Random number generation
    std::mt19937 mt_engine;

    // Scoring function
    score_fun_t<T> score_function;

    // Optional parameter values.
    // set by reading the option registry if used by external user.
    // Set by the alternate constructor if used by a forest
    bool read_public_options = true;
    da_int max_depth, min_node_sample, method, nfeat_split, seed;
    T min_split_score, feat_thresh, min_improvement;
    bool bootstrap = false;
    da_int check_cat_data = 0, opt_max_cat;
    da_int use_hist = false;
    da_int usr_max_bins;
    T cat_tol;
    da_int cat_split_strat;
    da_int max_threads = 0;

    da_status tree_serialization(da_model_persistence::serialization_buffer &buffer);

  public:
    // Constructor for public interfaces
    decision_tree(da_errors::da_error_t &err);
    // Constructor for forest model loading
    decision_tree() = default;
    // Constructor bypassing the optional parameters for internal forest use
    // Values will NOT be checked
    decision_tree(bins<T> *X_binned, da_int max_depth, da_int min_node_sample,
                  da_int method, da_int nfeat_split, da_int seed, T min_split_score,
                  T feat_thresh, T min_improvement, bool bootstrap, da_int check_cat_data,
                  da_int opt_max_cat, da_int use_hist, da_int usr_max_bins, T cat_tol,
                  da_int cat_split_strat, da_int max_threads);
    ~decision_tree();

    // Memory management
    da_status read_options();
    da_status init_working_memory();
    da_status init_working_memory_raw();
    da_status init_working_memory_hist();
    da_status resize_tree(size_t new_size);
    void clear_working_memory();
    void refresh() override;

    // Public training
    da_status set_training_data(da_int n_samples, da_int n_features, const T *X,
                                da_int ldx, const da_int *y, da_int n_class = 0,
                                da_int n_obs = 0, da_int *samples_subset = nullptr,
                                const da_int *usr_cat_feat = nullptr);
    da_status fit();

    // Scoring utilities
    void count_class_occurences(std::vector<da_int> &class_occ, da_int start_idx,
                                da_int end_idx);
    void count_class_occurences(std::vector<da_int> &class_occ, da_int start_idx,
                                da_int end_idx, std::vector<da_int> &weights);

    // Splitting functions
    bool compute_best_split(const node<T> &nd, da_int feat_idx, split<T> &sp,
                            split_workspace<T> &ws, std::vector<da_int> &samp);
    template <typename U>
    bool update_split_sorted(da_int sidx, da_int &next_idx, da_int end_idx, da_int ns,
                             da_int &ns_left, da_int &ns_right, T &left_score,
                             T &right_score, T &split_score, std::vector<U> &fv,
                             split_workspace<T> &ws, std::vector<da_int> &samp);

    // Split: raw data
    void update_feature_values(da_int start_idx, da_int end_idx, da_int feat_idx,
                               split_workspace<T> &ws, std::vector<da_int> &samp);
    void split_raw_onevall(const node<T> &current_node, da_int feat_idx, split<T> &sp,
                           split_workspace<T> &ws, std::vector<da_int> &samp);
    void update_count_left(da_int start_idx, da_int end_idx, da_int &ns_left,
                           split_workspace<T> &ws, std::vector<da_int> &samp);
    void update_count_left(da_int start_idx, da_int end_idx, da_int &ns_left,
                           std::vector<da_int> &weights, split_workspace<T> &ws,
                           std::vector<da_int> &samp);
    void update_count_right(da_int start_idx, da_int end_idx, da_int &ns_right,
                            split_workspace<T> &ws, std::vector<da_int> &samp);
    void update_count_right(da_int start_idx, da_int end_idx, da_int &ns_right,
                            std::vector<da_int> &weights, split_workspace<T> &ws,
                            std::vector<da_int> &samp);
    void split_raw_continuous(const node<T> &current_node, split<T> &sp,
                              split_workspace<T> &ws, std::vector<da_int> &samp);
    bool compute_best_split_raw(const node<T> &nd, da_int feat_idx, split<T> &sp,
                                split_workspace<T> &ws, std::vector<da_int> &samp);
    // Split: binned data
    bool update_node_histogram(const node<T> &nd, da_int feat_idx,
                               split_workspace<T> &ws);
    bool update_node_histogram(const node<T> &nd, da_int feat_idx,
                               std::vector<da_int> &weights, split_workspace<T> &ws);
    bool compute_best_split_hist(const node<T> &nd, da_int feat_idx, split<T> &sp,
                                 split_workspace<T> &ws);
    void split_hist_onevall(const node<T> &nd, da_int &ns_left, da_int &ns_ritgh,
                            da_int cat_start_idx, split_workspace<T> &ws);
    void split_hist_ordered(const node<T> &nd, da_int &ns_left, da_int &ns_ritgh,
                            da_int cat_start_idx, split_workspace<T> &ws);

    // partition functions
    // raw data
    da_int partition_samples_raw_continuous(const node<T> &nd);
    da_int partition_samples_raw_categorical(const node<T> &nd);
    // binned data
    da_int partition_samples_hist_ordered(const node<T> &nd);
    da_int partition_samples_hist_onevall(const node<T> &nd);

    // Tree training strategies
    da_status fit_serial();
    da_status fit_parallel();

    // tree building
    da_status split_node_and_add_children(da_int node_idx, split<T> &best_split);
    da_status add_node(da_int parent_idx, bool is_left, T score, da_int split_idx);
    da_int get_next_node_idx();

    // Inference
    da_status predict(da_int nsamp, da_int n_features, const T *X_test, da_int ldx,
                      da_int *y_pred, da_int mode = 0);
    da_status predict_proba(da_int nsamp, da_int n_features, const T *X_test, da_int ldx,
                            T *y_pred, da_int n_class, da_int ldy, da_int mode = 0);
    da_status predict_log_proba(da_int nsamp, da_int n_features, const T *X_test,
                                da_int ldx, T *y_pred, da_int n_class, da_int ldy);
    da_status score(da_int nsamp, da_int nfeat, const T *X_test, da_int ldx,
                    const da_int *y_test, T *accuracy);

    da_status get_result(da_result query, da_int *dim, T *result) override;
    da_status get_result(da_result query, da_int *dim, da_int *result) override;

    // Getters for testing purposes
    std::vector<da_int> const &get_samples_idx();
    void init_samples_idx();
    void init_feature_values(da_int feat_idx);
    std::vector<T> const &get_features_values();
    std::vector<da_int> const &get_count_classes();
    std::vector<da_int> const &get_count_left_classes();
    std::vector<da_int> const &get_count_right_classes();
    std::vector<da_int> const &get_features_idx();
    std::vector<split_workspace<T>> const &get_thread_workspaces();
    bool model_is_trained();
    std::vector<node<T>> const &get_tree();
    da_int get_n_leaves() { return n_leaves; }
    // Setters for testing purposes
    void set_bootstrap(bool bs);

    da_status serialize(da_model_persistence::serialization_buffer &buffer) override;
    da_status save_model(da_model_persistence::serialization_buffer &buffer) override;
    da_status load_model(da_model_persistence::serialization_buffer &buffer) override;
};

} // namespace da_decision_forest
} // namespace ARCH
#endif