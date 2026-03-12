/*
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "binary_tree.hpp"
#include "da_error.hpp"
#include "da_std.hpp"
#include "macros.h"
#include "nearest_neighbors_options.hpp"

namespace ARCH {

namespace da_neighbors {

/* nearest neighbors class */
template <typename T> class neighbors : public basic_handle<T> {
  private:
    // Set true when initialization is complete by set_params() function
    bool is_up_to_date = false;
    // Set true if training data has been provided via set_data()
    bool istrained_Xtrain = false;
    bool istrained_labels = false;
    bool istrained_targets = false;
    // Set true if the available classes have been computed via a call to available_classes()
    bool classes_computed = false;
    bool radius_neighbors_computed = false;
    bool sort_results = false;
    bool rnn_return_distances = false;

    // Number of neighbors to be considered
    da_int n_neighbors = 5;
    // Algorithm to be used for the knn computation
    da_int algo = da_neighbors_types::nn_algorithm::automatic;
    da_int working_algo = da_neighbors_types::nn_algorithm::automatic;
    // Metric to be used for the distance computation
    da_int metric = da_euclidean;
    // Internal metric to be used for the distance computation
    // We want to avoid squaring the distance unless it's necessary
    da_metric internal_metric = da_sqeuclidean;
    // Leaf size for the k-d tree algorithm
    da_int leaf_size = 30;
    // Denote if squaring of the internal metric is required
    bool get_squares = false;
    // Minkowski parameter used for the minkowski distance conputation
    T p = 2.0;
    // Weight function used to compute the k-nearest neighbors
    da_int weights = ::da_neighbors_types::nn_weights::uniform;
    // User's data
    da_int n_samples = 0, n_features = 0, ldx_train = 0;
    const T *X_train = nullptr /*n_samples-by-n_features*/;
    const da_int *y_train_class = nullptr /*n_samples*/;
    const T *y_train_reg = nullptr /*n_samples*/;
    // Utility pointer to column major allocated copy of user's data
    T *X_train_temp = nullptr;
    // Internal tree objects to be initialized only when that options is requested
    std::unique_ptr<ARCH::da_binary_tree::kd_tree<T>> internal_kd_tree = nullptr;
    std::unique_ptr<ARCH::da_binary_tree::ball_tree<T>> internal_ball_tree = nullptr;
    // For radius neighbors
    T radius = 1.0;
    std::vector<da_int> radius_neighbors_count;
    std::vector<da_vector::da_vector<da_int>> radius_neighbors_indices;
    std::vector<da_vector::da_vector<T>> radius_neighbors_distances;

  public:
    std::vector<da_int> classes;
    da_int n_classes = -1;

    ~neighbors();

    neighbors(da_errors::da_error_t &err);

    /* get_result (required to be defined by basic_handle) */
    da_status get_result(da_result query, da_int *dim, T *result);
    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] da_int *result);
    // Set input parameters
    da_status set_params();
    // Chose the appropriate algorithm for kNN if auto is selected
    void set_neighbors_algorithm();
    // Initialize the k-d tree
    da_status init_kd_tree();
    // Initialize the ball tree
    da_status init_ball_tree();
    // Check if the options have been updated between calls
    da_status check_options_update();
    // Set the training data (features)
    da_status set_data(da_int n_samples, da_int n_features, const T *X_train,
                       da_int ldx_train);
    // Set the training labels for classification
    da_status set_labels(da_int n_samples, const da_int *y_train_class);
    // Set the training targets for regression
    da_status set_targets(da_int n_samples, const T *y_train_reg);
    // Compute the k-nearest neighbors and optionally the corresponding distances
    // Includes the appropriate checks for input arguments
    da_status kneighbors(da_int n_queries, da_int n_features, const T *X_test,
                         da_int ldx_test, da_int *n_ind, T *n_dist, da_int k = 0,
                         bool return_distance = 0);
    // Compute kernel for the k-nearest neighbors and optionally the corresponding distances
    // so that all neighbours of each observation lies contiguously in memory, same for the distances.
    // Assumes column-major order.
    da_status kneighbors_compute(da_int n_queries, da_int n_features, const T *X_test,
                                 da_int ldx_test, da_int *n_ind, T *n_dist,
                                 da_int n_neigh, bool return_distance);
    // Compute kernel for brute force algorithm
    da_status kneighbors_compute_brute_force(da_int n_queries, da_int n_features,
                                             const T *X_test, da_int ldx_test,
                                             da_int *n_ind, T *n_dist, da_int n_neigh,
                                             bool return_distance);
    // Compute kernel for k-d tree algorithm
    da_status kneighbors_compute_kd_tree(da_int n_queries, da_int n_features,
                                         const T *X_test, da_int ldx_test, da_int *n_ind,
                                         T *n_dist, da_int n_neigh, bool return_distance);
    // Compute kernel for ball tree algorithm
    da_status kneighbors_compute_ball_tree(da_int n_queries, da_int n_features,
                                           const T *X_test, da_int ldx_test,
                                           da_int *n_ind, T *n_dist, da_int n_neigh,
                                           bool return_distance);

    // Computational kernel that computes kneighbors using blocking on Xtest for overall algorithm.
    // In addition, it uses blocking for Xtrain only for the distance computation.
    template <da_int XTRAIN_BLOCK, da_int XTEST_BLOCK>
    inline da_status kneighbors_brute_force_Xtest(da_int n_queries, da_int n_features,
                                                  const T *X_test, da_int ldx_test,
                                                  da_int *n_ind, T *n_dist,
                                                  da_int n_neigh, bool return_distance);
    // Compute the k-nearest neighbors and optionally the corresponding distances
    // Inlining is performance-critical for this function
    template <da_int XTRAIN_BLOCK>
    inline da_status kneighbors_brute_force_Xtest_kernel(
        da_int xtrain_block_size, da_int n_blocks_train, da_int block_rem_train,
        da_int n_queries, da_int n_features, const T *X_test, da_int ldx_test, T *D,
        da_int *n_ind, T *n_dist, da_int k, bool return_distance);
    // Compute probability estimates for provided test data
    da_status predict_proba(da_int n_queries, da_int n_features, const T *X_test,
                            da_int ldx_test, T *proba, da_nn_search_mode search_mode);
    // Compute probability estimates for the provided test data so that the probabilities
    // for each observation lie contiguously in memory.
    // Assumes column-major order.
    da_status predict_proba_compute_knn(da_int n_queries, da_int n_features,
                                        const T *X_test, da_int ldx_test, T *proba);
    // Predict the classification labels for provided test data
    da_status predict(da_int n_queries, da_int n_features, const T *X_test,
                      da_int ldx_test, da_int *y_test, da_nn_search_mode search_mode);
    // Predict the regression targets for provided test data
    da_status predict(da_int n_queries, da_int n_features, const T *X_test,
                      da_int ldx_test, T *y_test, da_nn_search_mode search_mode);
    // Predict the regression targets for provided test data using k-nearest neighbors
    // Assumes column-major order.
    da_status predict_targets_knn(da_int n_queries, da_int n_features, const T *X_test,
                                  da_int ldx_test, T *y_test);
    // Internal function used to compute the std::vector that holds the available classes
    da_status available_classes();

    // Implementing refresh
    void refresh();

    // Radius neighbors functions
    // Compute the radius nearest neighbors and optionally the corresponding distances
    // Includes the appropriate checks for input arguments
    da_status radius_neighbors(da_int n_queries, da_int n_features, const T *X_test,
                               da_int ldx_test, T radius = 1.0,
                               bool return_distance = false, bool sort_results = false);
    // Compute kernel for the radius nearest neighbors and optionally the corresponding distances
    // so that all neighbours of each observation lies contiguously in memory, same for the distances.
    // Assumes column-major order.
    da_status
    radius_neighbors_compute(da_int n_queries, da_int n_features, const T *X_test,
                             da_int ldx_test, T radius, std::vector<da_int> &rnn_count,
                             std::vector<da_vector::da_vector<da_int>> &rnn_indices,
                             std::vector<da_vector::da_vector<T>> &rnn_distances,
                             bool return_distance, bool sort_results, bool is_temp);
    // Compute kernel for brute force algorithm
    da_status radius_neighbors_compute_brute_force(
        da_int n_queries, da_int n_features, const T *X_test, da_int ldx_test, T radius,
        std::vector<da_vector::da_vector<da_int>> &rnn_indices,
        std::vector<da_vector::da_vector<T>> &rnn_distances, bool return_distance);
    // Compute kernel for k-d tree algorithm
    da_status radius_neighbors_compute_kd_tree(
        da_int n_queries, da_int n_features, const T *X_test, da_int ldx_test, T radius,
        std::vector<da_vector::da_vector<da_int>> &rnn_indices,
        std::vector<da_vector::da_vector<T>> &rnn_distances, bool return_distance);
    // Compute kernel for ball tree algorithm
    da_status radius_neighbors_compute_ball_tree(
        da_int n_queries, da_int n_features, const T *X_test, da_int ldx_test, T radius,
        std::vector<da_vector::da_vector<da_int>> &rnn_indices,
        std::vector<da_vector::da_vector<T>> &rnn_distances, bool return_distance);

    // Return the number of radius neighbors for each query point
    da_status radius_neighbors_count_internal(da_int n_queries,
                                              da_int *n_radius_neighbors);

    // Extract the radius neighbors for the sample point query_index
    da_status extract_radius_neighbors(da_int query_index, da_int n_neighbors,
                                       da_int *neighbors_indices, T *neighbors_distances);

    // Extract the radius neighbors for the sample point query_index
    da_status extract_radius_neighbors_indices(da_int query_index, da_int n_neighbors,
                                               da_int *neighbors_indices);
    // Extract the radius neighbors for the sample point query_index
    da_status extract_radius_neighbors_distances(da_int query_index, da_int n_neighbors,
                                                 T *neighbors_distances);
    // Compute probability estimates based on radius neighbors.
    // Assumes column-major order.
    da_status predict_proba_compute_rnn(da_int n_queries, da_int n_features,
                                        const T *X_test, da_int ldx_test, T *proba);
    // Predict the regression targets for provided test data using radius nearest neighbors
    // Assumes column-major order.
    da_status predict_targets_rnn(da_int n_queries, da_int n_features, const T *X_test,
                                  da_int ldx_test, T *y_test);
};

} // namespace da_neighbors
} // namespace ARCH