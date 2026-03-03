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

#include "nearest_neighbors.hpp"
#include "basic_statistics.hpp"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "da_omp.hpp"
#include "macros.h"
#include "nearest_neighbors_options.hpp"
#include "nearest_neighbors_utils.hpp"
#include "pairwise_distances.hpp"
#include <numeric>

namespace ARCH {

namespace da_neighbors {

#define KNN_BLOCK_FLOAT 2048
#define KNN_BLOCK_DOUBLE 1024
#define KNN_BLOCK_SMALL 16
#define KNN_BLOCK_MEDIUM 128

#define RNN_BLOCK_FLOAT 2048
#define RNN_BLOCK_DOUBLE 1024

#define XTRAIN_RNN_BLOCK_SIZE da_int(256)
#define XTEST_RNN_BLOCK_SIZE da_int(256)

// Inline helper function to validate and store X_test data
template <typename T>
da_status validate_and_store_X_test(neighbors<T> *self, da_int n_queries,
                                    da_int n_features, const T *X_test, da_int ldx_test,
                                    T **utility_ptr1, const T **X_test_temp,
                                    da_int &ldx_test_temp, da_errors::da_error_t *err,
                                    da_int n_features_train) {

    // Check X_test pointer first
    if (X_test == nullptr) {
        return da_error_bypass(err, da_status_invalid_pointer,
                               "X_test is not a valid pointer.");
    }

    // Data matrix X must have the same number of columns as X_train.
    if (n_features != n_features_train) {
        return da_error_bypass(err, da_status_invalid_array_dimension,
                               "n_features = " + std::to_string(n_features) +
                                   " doesn't match the expected value " +
                                   std::to_string(n_features_train) + ".");
    }

    ldx_test_temp = ldx_test;
    da_status status = da_status_success;
    // Store and validate the 2D array
    status = self->store_2D_array(n_queries, n_features, X_test, ldx_test, utility_ptr1,
                                  X_test_temp, ldx_test_temp, "n_queries", "n_features",
                                  "X_test", "ldx_test");
    return status;
}

template <typename T> neighbors<T>::~neighbors() {
    // Destructor needs to handle arrays that were allocated due to row major storage of input data
    if (X_train_temp)
        delete[] (X_train_temp);
}

template <typename T>
neighbors<T>::neighbors(da_errors::da_error_t &err) : basic_handle<T>(err) {
    // Initialize the options registry
    // Any error is stored err->status[.] and this NEEDS to be checked
    // by the caller.
    register_neighbors_options<T>(this->opts, *this->err);
}

template <typename T>
da_status neighbors<T>::get_result(da_result query, da_int *dim, T *result) {
    da_int n_count = *dim;

    switch (query) {
    case da_result::da_nn_radius_neighbors_distances_index: {
        if (!this->radius_neighbors_computed) {
            return da_warn(this->err, da_status_no_data,
                           "Radius neighbors have not been computed. Please call "
                           "da_nn_radius_neighbors_s or da_nn_radius_neighbors_d first.");
        }
        if (this->rnn_return_distances == false) {
            return da_warn(this->err, da_status_no_data,
                           "Distances were not requested during radius neighbors "
                           "computation. Please set return_distance to true in "
                           "da_nn_radius_neighbors_s or da_nn_radius_neighbors_d call.");
        }
        da_int index = da_int(result[0]);
        if (index < 0 || index >= (da_int)this->radius_neighbors_distances.size()) {
            return da_warn(
                this->err, da_status_invalid_input,
                "The provided index is out of bounds. It should be in the "
                "range [0, " +
                    std::to_string(this->radius_neighbors_distances.size() - 1) + "].");
        }
        da_int n_neighbors = (da_int)this->radius_neighbors_count[index];
        if (n_neighbors > n_count) {
            // Set dim to the correct size needed
            *dim = n_neighbors;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(n_neighbors) + ".");
        }

        return neighbors<T>::extract_radius_neighbors_distances(index, n_neighbors,
                                                                result);
        break;
    }
    case da_result::da_nn_radius_neighbors_distances: {
        if (!this->radius_neighbors_computed) {
            return da_warn(this->err, da_status_no_data,
                           "Radius neighbors have not been computed. Please call "
                           "da_nn_radius_neighbors_s or da_nn_radius_neighbors_d first.");
        }
        if (this->rnn_return_distances == false) {
            return da_warn(this->err, da_status_no_data,
                           "Distances were not requested during radius neighbors "
                           "computation. Please set return_distance to true in "
                           "da_nn_radius_neighbors_s or da_nn_radius_neighbors_d call.");
        }
        da_int total_neighbors = 0;
        da_int array_index = 0;
        for (da_int i = 0; i < (da_int)this->radius_neighbors_distances.size(); i++) {
            total_neighbors += this->radius_neighbors_count[i];
        }
        if (total_neighbors > n_count) {
            // Set dim to the correct size needed
            *dim = total_neighbors;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(total_neighbors) + ".");
        }
        for (da_int i = 0; i < (da_int)this->radius_neighbors_distances.size(); i++) {
            da_int temp_size = (da_int)this->radius_neighbors_count[i];
            neighbors<T>::extract_radius_neighbors_distances(i, temp_size,
                                                             result + array_index);
            array_index += temp_size;
        }

        break;
    }
    default:
        return da_warn(this->err, da_status_unknown_query,
                       "The requested result could not be found.");
    }
    return da_status_success;
}

template <typename T>
da_status neighbors<T>::get_result(da_result query, da_int *dim, da_int *result) {
    da_int n_count = *dim;
    switch (query) {
    case da_result::da_nn_radius_neighbors_count: {
        if (!this->radius_neighbors_computed) {
            return da_warn(this->err, da_status_no_data,
                           "Radius neighbors have not been computed. Please call "
                           "da_nn_radius_neighbors_s or da_nn_radius_neighbors_d first.");
        }

        da_int n_queries = (da_int)this->radius_neighbors_count.size();
        if (n_queries + 1 > n_count) {
            // Set dim to the correct size needed
            *dim = n_queries + 1;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(n_queries + 1) + ".");
        }
        return neighbors<T>::radius_neighbors_count_internal(n_count, result);
        break;
    }
    case da_result::da_nn_radius_neighbors_offsets: {
        if (!this->radius_neighbors_computed) {
            return da_warn(this->err, da_status_no_data,
                           "Radius neighbors have not been computed. Please call "
                           "radius_neighbors first.");
        }

        da_int offset = 0;
        da_int n_queries = (da_int)this->radius_neighbors_count.size();
        if (n_queries + 1 > n_count) {
            // Set dim to the correct size needed
            *dim = n_queries + 1;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(n_queries + 1) + ".");
        }
        for (da_int i = 0; i < n_queries; i++) {
            if ((da_int)this->radius_neighbors_count[i] > 0) {
                result[i] = offset;
                offset += (da_int)this->radius_neighbors_count[i];
            } else
                result[i] = -1;
        }
        result[this->radius_neighbors_indices.size()] = offset;
        break;
    }
    case da_result::da_nn_radius_neighbors_indices_index: {
        if (!this->radius_neighbors_computed) {
            return da_warn(this->err, da_status_no_data,
                           "Radius neighbors have not been computed. Please call "
                           "da_nn_radius_neighbors_s or da_nn_radius_neighbors_d first.");
        }
        da_int index = da_int(result[0]);
        if (index < 0 || index >= (da_int)this->radius_neighbors_indices.size()) {
            return da_warn(this->err, da_status_invalid_input,
                           "The provided index is out of bounds. It should be in the "
                           "range [0, " +
                               std::to_string(this->radius_neighbors_indices.size() - 1) +
                               "].");
        }
        da_int n_neighbors = (da_int)this->radius_neighbors_count[index];
        if (n_neighbors > n_count) {
            // Set dim to the correct size needed
            *dim = n_neighbors;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(n_neighbors) + ".");
        }
        return neighbors<T>::extract_radius_neighbors_indices(index, n_neighbors, result);
        break;
    }
    case da_result::da_nn_radius_neighbors_indices: {
        if (!this->radius_neighbors_computed) {
            return da_warn(this->err, da_status_no_data,
                           "Radius neighbors have not been computed. Please call "
                           "da_nn_radius_neighbors_s or da_nn_radius_neighbors_d first.");
        }

        da_int total_neighbors = 0;
        da_int array_index = 0;
        for (da_int i = 0; i < (da_int)this->radius_neighbors_indices.size(); i++) {
            total_neighbors += this->radius_neighbors_count[i];
        }
        if (total_neighbors > n_count) {
            // Set dim to the correct size needed
            *dim = total_neighbors;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(total_neighbors) + ".");
        }
        for (da_int i = 0; i < (da_int)this->radius_neighbors_indices.size(); i++) {
            for (da_int j = 0; j < (da_int)this->radius_neighbors_count[i]; j++) {
                result[array_index++] = this->radius_neighbors_indices[i][j];
            }
        }
        break;
    }
    default:
        return da_warn(this->err, da_status_unknown_query,
                       "The requested result could not be found.");
    }
    return da_status_success;
}

template <typename T> da_status neighbors<T>::set_params() {
    // Extract options
    std::string opt_val;
    bool opt_pass = true;
    opt_pass &= this->opts.get("number of neighbors", n_neighbors) == da_status_success;
    opt_pass &= this->opts.get("algorithm", opt_val, algo) == da_status_success;
    opt_pass &= this->opts.get("metric", opt_val, metric) == da_status_success;
    opt_pass &= this->opts.get("weights", opt_val, weights) == da_status_success;
    opt_pass &= this->opts.get("minkowski parameter", p) == da_status_success;
    opt_pass &= this->opts.get("leaf size", leaf_size) == da_status_success;
    opt_pass &= this->opts.get("radius", radius) == da_status_success;

    if (!opt_pass)
        return da_error_bypass(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                               "Unexpected error while reading the optional parameters.");
    internal_metric = da_metric(metric);

    working_algo = algo;
    // If auto is chosen, calculate the correct algorithm depending on the other options
    if (this->algo == da_neighbors_types::nn_algorithm::automatic)
        set_neighbors_algorithm();
    // Check for incompatible options
    else if (this->working_algo == da_neighbors_types::nn_algorithm::kd_tree ||
             this->working_algo == da_neighbors_types::nn_algorithm::ball_tree) {
        if (metric == da_cosine || metric == da_sqeuclidean ||
            metric == da_sqeuclidean_gemm) {
            return da_error(this->err, da_status_incompatible_options,
                            "The tree algorithms are not compatible with the cosine or "
                            "squared Euclidean distances.");
        } else if (metric == da_minkowski && p < (T)1.0) {
            // Minkowski distance with p<1 does not satisfy the triangle inequality,
            // so it is not a metric.
            return da_error(this->err, da_status_incompatible_options,
                            "Tree algorithms are not compatible with the Minkowski "
                            "metric when 0 < p < 1.");
        }
    }

    if (metric == da_euclidean || (metric == da_minkowski && p == T(2.0)) ||
        metric == da_euclidean_gemm) {
        this->get_squares = true;
        if (this->working_algo == brute) {
            // If the algorithm is brute force, we need to use the squared Euclidean distance
            // to avoid computing the square root.
            if (metric == da_euclidean_gemm)
                internal_metric = da_sqeuclidean_gemm;
            else
                internal_metric = da_sqeuclidean;
        }
    }

    this->is_up_to_date = true;
    return da_status_success;
}

// Chose the appropriate algorithm if auto is selected
template <typename T> void neighbors<T>::set_neighbors_algorithm() {
    if ((this->metric == da_cosine) || (this->metric == da_sqeuclidean) ||
        (this->metric == da_minkowski && this->p < (T)1.0) ||
        (this->metric == da_sqeuclidean_gemm)) { // LCOV_EXCL_LINE
        this->working_algo = da_neighbors_types::nn_algorithm::brute;
    } else {
        // If the number of features is small and the number of samples is large, use k-d tree
        if (this->n_features < 10 && this->n_samples > 100000) { // LCOV_EXCL_LINE
            this->working_algo = da_neighbors_types::nn_algorithm::kd_tree;
        } else {
            this->working_algo = da_neighbors_types::nn_algorithm::brute;
        }
    }
}

// Initialize the k-d tree
template <typename T> da_status neighbors<T>::init_kd_tree() {
    try {
        this->internal_kd_tree = std::make_unique<ARCH::da_binary_tree::kd_tree<T>>(
            n_samples, n_features, X_train, ldx_train, this->leaf_size,
            da_metric(this->internal_metric), this->p);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    return da_status_success;
}

// Initialize the k-d tree
template <typename T> da_status neighbors<T>::init_ball_tree() {
    try {
        this->internal_ball_tree = std::make_unique<ARCH::da_binary_tree::ball_tree<T>>(
            n_samples, n_features, X_train, ldx_train, this->leaf_size,
            da_metric(this->internal_metric), this->p);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    return da_status_success;
}

// Check if the options have been updated between calls
template <typename T> da_status neighbors<T>::check_options_update() {
    // Check if the parameters are updated and if so, recompute the tree.
    // This is needed in case the user has changed the parameters after the training data was set.
    std::string opt_val;
    bool opt_pass = true;
    da_int local_algo, local_metric, local_leaf_size;
    T local_p;
    opt_pass &= this->opts.get("algorithm", opt_val, local_algo) == da_status_success;
    opt_pass &= this->opts.get("radius", radius) == da_status_success;
    if (!opt_pass)
        return da_error_bypass(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                               "Unexpected error while reading the optional parameters.");
    // If the algorithm is auto or k-d tree, we would need to recompute the tree
    if (local_algo == da_neighbors_types::nn_algorithm::automatic ||
        local_algo == da_neighbors_types::nn_algorithm::kd_tree ||
        local_algo == da_neighbors_types::nn_algorithm::ball_tree) {
        if (this->algo != local_algo) {
            return da_error_bypass(
                this->err, da_status_option_locked,
                "Options need to be set before calling set_training_data().");
        }
        // If the algorithm did not change, check if the other options in which
        // we depend on have changed.
        opt_pass &= this->opts.get("leaf size", local_leaf_size) == da_status_success;
        if (!opt_pass)
            return da_error_bypass(
                this->err, da_status_internal_error, // LCOV_EXCL_LINE
                "Unexpected error while reading the optional parameters.");
        if (this->leaf_size != local_leaf_size) {
            return da_error_bypass(
                this->err, da_status_option_locked,
                "Options need to be set before calling set_training_data().");
        }
        if (local_algo == da_neighbors_types::nn_algorithm::automatic) {
            opt_pass &=
                this->opts.get("metric", opt_val, local_metric) == da_status_success;
            opt_pass &=
                this->opts.get("minkowski parameter", local_p) == da_status_success;
            if (!opt_pass)
                return da_error_bypass(
                    this->err, da_status_internal_error, // LCOV_EXCL_LINE
                    "Unexpected error while reading the optional parameters.");
            if (this->metric != local_metric || this->p != local_p) {
                return da_error_bypass(
                    this->err, da_status_option_locked,
                    "Options need to be set before calling set_training_data().");
            }
        }
    }
    return da_status_success;
}

// Set the training data (features)
template <typename T>
da_status neighbors<T>::set_data(da_int n_samples, da_int n_features, const T *X_train,
                                 da_int ldx_train) {
    // Verify n_samples matches if already set from set_labels() or set_targets()
    if ((this->n_samples > 0) && (n_samples != this->n_samples)) {
        return da_error_bypass(this->err, da_status_invalid_array_dimension,
                               "n_samples = " + std::to_string(n_samples) +
                                   " doesn't match the training data size " +
                                   std::to_string(this->n_samples) + ".");
    }

    // Guard against errors due to multiple calls using the same class instantiation
    if (X_train_temp) {
        delete[] (X_train_temp);
        X_train_temp = nullptr;
    }

    da_status status = this->store_2D_array(
        n_samples, n_features, X_train, ldx_train, &X_train_temp, &this->X_train,
        this->ldx_train, "n_samples", "n_features", "X_train", "ldx_train");
    if (status != da_status_success)
        return status;

    // Set internal parameters
    this->n_samples = n_samples;
    this->n_features = n_features;

    // Check if the option for k-d tree is set, in which case we need to initialize the
    // internal kd_tree object.
    if (!is_up_to_date)
        status = neighbors<T>::set_params();
    if (status != da_status_success)
        return status;

    if (this->working_algo == da_neighbors_types::nn_algorithm::kd_tree) {
        status = neighbors<T>::init_kd_tree();
        if (status != da_status_success)
            return status;
    } else if (this->working_algo == da_neighbors_types::nn_algorithm::ball_tree) {
        status = neighbors<T>::init_ball_tree();
        if (status != da_status_success)
            return status;
    }
    this->istrained_Xtrain = true;
    return da_status_success;
}

// Set the training labels for classification
template <typename T>
da_status neighbors<T>::set_labels(da_int n_samples, const da_int *y_train_class) {
    // Verify n_samples matches the training data size, or set it if not yet set
    if ((this->n_samples > 0) && (n_samples != this->n_samples)) {
        return da_error_bypass(this->err, da_status_invalid_array_dimension,
                               "n_samples = " + std::to_string(n_samples) +
                                   " doesn't match the training data size " +
                                   std::to_string(this->n_samples) + ".");
    }

    da_status status =
        this->check_1D_array(n_samples, y_train_class, "n_samples", "y_train_class", 1);
    if (status != da_status_success)
        return status;

    // Set n_samples if not yet set from set_data()
    this->n_samples = n_samples;

    // Set internal pointer to user data
    this->y_train_class = y_train_class;
    this->istrained_labels = true;
    return da_status_success;
}

// Set the training targets for regression
template <typename T>
da_status neighbors<T>::set_targets(da_int n_samples, const T *y_train_reg) {
    // Verify n_samples matches the training data size, or set it if not yet set
    if ((this->n_samples > 0) && (n_samples != this->n_samples)) {
        return da_error_bypass(this->err, da_status_invalid_array_dimension,
                               "n_samples = " + std::to_string(n_samples) +
                                   " doesn't match the training data size " +
                                   std::to_string(this->n_samples) + ".");
    }

    da_status status =
        this->check_1D_array(n_samples, y_train_reg, "n_samples", "y_train_reg", 1);
    if (status != da_status_success)
        return status;

    // Set n_samples if not yet set from set_data()
    this->n_samples = n_samples;

    // Set internal pointer to user data
    this->y_train_reg = y_train_reg;
    this->istrained_targets = true;
    return da_status_success;
}

// Given a vector D of length n and an integer k, this function returns in the first k positions
// of a vector k_dist, the k smaller values of D (unordered) and in the first k positions of a vector
// k_ind, the corresponding indices of the original vector D, where initial indices are
// init_index, init_index+1, ...
template <typename T>
inline void smaller_values_and_indices_cblas(da_int n, T *D, da_int k, da_int *k_ind,
                                             T *k_dist, da_int init_index,
                                             bool init = true) {
    // Initialize the first k values of k_ind with init_index, init_index+1, ..., init_index+k-1
    if (init)
        da_std::iota(k_ind, k_ind + k, init_index);
    // Find the index of the maximum element and the corresponding maximum value.
    da_int max_index = da_blas::cblas_iamax(k, k_dist, 1);
    T max_val = k_dist[max_index];

    for (da_int i = k; i < n; i++) {
        // Check if an element of D is smaller than the maximum value. If it is,
        // we need to replace it's index in k_ind and replace the corresponding D[i] in k_dist.
        if (D[i] <= max_val) {
            // We know D[i] is smaller than Dmax. So we update k_ind[max_index] and D[max_index]
            // so that they hold the new value.
            k_ind[max_index] = i;
            k_dist[max_index] = D[i];
            // Now we need to find the new maximum so that we compare against that in the next iteration.
            max_index = da_blas::cblas_iamax(k, k_dist, 1);
            max_val = k_dist[max_index];
        }
    }
}

template <typename T>
template <da_int XTRAIN_BLOCK>
inline __attribute__((__always_inline__)) da_status
neighbors<T>::kneighbors_brute_force_Xtest_kernel(
    da_int xtrain_block_size, da_int n_blocks_train, da_int block_rem_train,
    da_int n_queries, da_int n_features, const T *X_test, da_int ldx_test, T *D,
    da_int *n_ind, T *n_dist, da_int n_neigh, bool return_distance) {

    da_status status = da_status_success;
    // Set blocking of X_train depending on the block size
    constexpr bool block_xtrain = XTRAIN_BLOCK != 1;
    if constexpr (block_xtrain) {
        da_int xtrain_subblock = xtrain_block_size;
        for (da_int iblock = 0; iblock < n_blocks_train; iblock++) {
            if (iblock == n_blocks_train - 1 && block_rem_train > 0)
                xtrain_subblock = block_rem_train;

            status = da_metrics::pairwise_distances::pairwise_distance_kernel(
                column_major, xtrain_subblock, n_queries, n_features,
                X_train + iblock * xtrain_block_size, ldx_train, X_test, ldx_test,
                D + iblock * xtrain_block_size, n_samples, this->p,
                this->internal_metric);
        }
    } else {
        status = da_metrics::pairwise_distances::pairwise_distance_kernel(
            column_major, this->n_samples, n_queries, n_features, this->X_train,
            this->ldx_train, X_test, ldx_test, D, this->n_samples, this->p,
            this->internal_metric);
    }
    if (status != da_status_success) {
        return status;
    }
    // Get the first n_neigh smaller values of D on k_dist and the correspondind distances on n_dist.
    // Here we can use D and replace the smaller values in place since D is not used later on.
    T *k_dist = D;
    std::vector<da_int> perm_vector;
    std::vector<da_int> k_ind;
    try {
        perm_vector.resize(n_neigh);
        k_ind.resize(n_queries * n_neigh);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    for (da_int k = 0; k < n_queries; k++) {
        smaller_values_and_indices_cblas(this->n_samples, D + k * this->n_samples,
                                         n_neigh, k_ind.data() + k * n_neigh,
                                         k_dist + k * this->n_samples, 0);
        sorted_n_dist_n_ind(n_neigh, k_dist + k * this->n_samples,
                            k_ind.data() + k * n_neigh, n_dist + k * n_neigh,
                            n_ind + k * n_neigh, perm_vector.data(), return_distance,
                            this->get_squares);
    }

    return status;
}

// Computational kernel that computes kneighbors using blocking on Xtest for overall algorithm.
// In addition, it uses blocking for Xtrain only for the distance computation.
template <typename T>
template <da_int XTRAIN_BLOCK, da_int XTEST_BLOCK>
inline __attribute__((__always_inline__)) da_status
neighbors<T>::kneighbors_brute_force_Xtest(da_int n_queries, da_int n_features,
                                           const T *X_test, da_int ldx_test,
                                           da_int *n_ind, T *n_dist, da_int n_neigh,
                                           bool return_distance) {
    da_int xtest_block_size = std::min(XTEST_BLOCK, n_queries);
    da_int n_blocks_test = 0, block_rem_test = 0;
    da_utils::blocking_scheme(n_queries, xtest_block_size, n_blocks_test, block_rem_test);
    [[maybe_unused]] da_int n_threads =
        da_utils::get_n_threads_loop(std::max(n_blocks_test, (da_int)1));

    da_int threading_error = 0;
    da_int xtrain_block_size = std::min(XTRAIN_BLOCK, n_samples);
    da_int n_blocks_train = 0, block_rem_train = 0;
    da_utils::blocking_scheme(n_samples, xtrain_block_size, n_blocks_train,
                              block_rem_train);
    da_status private_status = da_status_success;
    da_int xtest_subblock = xtest_block_size;
    da_int samplex_x_xtest_block = this->n_samples * xtest_block_size;
    da_int xtest_block_x_n_neigh = xtest_block_size * n_neigh;

    // Iterate through the number of blocks
    if (block_rem_test > 0)
        n_blocks_test = n_blocks_test - 1;

#pragma omp parallel default(none)                                                       \
    shared(xtest_block_size, n_blocks_test, block_rem_test, n_features, X_test,          \
               n_queries, n_ind, n_dist, n_neigh, ldx_test, return_distance,             \
               threading_error, xtrain_block_size, n_blocks_train, block_rem_train,      \
               xtest_subblock, samplex_x_xtest_block,                                    \
               xtest_block_x_n_neigh) private(private_status) num_threads(n_threads)
    {
        std::vector<T> thread_local_d;
        try {
            thread_local_d.resize(samplex_x_xtest_block);
        } catch (std::bad_alloc const &) {
#pragma omp atomic write
            threading_error = 1;
        }

#pragma omp for schedule(dynamic)
        for (da_int jblock = 0; jblock < n_blocks_test; jblock++) {
            if (threading_error == 0) {
                // Use thread-local buffer
                private_status =
                    neighbors<T>::kneighbors_brute_force_Xtest_kernel<XTRAIN_BLOCK>(
                        xtrain_block_size, n_blocks_train, block_rem_train,
                        xtest_subblock, n_features, X_test + jblock * xtest_block_size,
                        ldx_test, thread_local_d.data(),
                        n_ind + jblock * xtest_block_x_n_neigh,
                        n_dist + jblock * xtest_block_x_n_neigh, n_neigh,
                        return_distance);
                if (private_status != da_status_success)
#pragma omp atomic write
                    threading_error = 1;
            }
        }
    }

    if (threading_error == 1)
        return da_error(this->err, da_status_memory_error, "Memory allocation failed.");

    // Do the remainder
    if (block_rem_test > 0) {
        std::vector<T> thread_local_d;
        try {
            thread_local_d.resize(this->n_samples * block_rem_test);
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error,
                            "Memory allocation failed.");
        }

        private_status = neighbors<T>::kneighbors_brute_force_Xtest_kernel<XTRAIN_BLOCK>(
            xtrain_block_size, n_blocks_train, block_rem_train, block_rem_test,
            n_features, X_test + n_blocks_test * xtest_block_size, ldx_test,
            thread_local_d.data(), n_ind + n_blocks_test * xtest_block_x_n_neigh,
            n_dist + n_blocks_test * xtest_block_x_n_neigh, n_neigh, return_distance);

        if (private_status != da_status_success)
            return da_status_memory_error;
    }

    return da_status_success;
}

template <typename T> struct knn_block_sizes {
    static constexpr da_int XTRAIN_BLOCK =
        std::is_same<T, float>::value ? KNN_BLOCK_FLOAT : KNN_BLOCK_DOUBLE;
    static constexpr da_int XTEST_BLOCK =
        std::is_same<T, float>::value ? KNN_BLOCK_FLOAT : KNN_BLOCK_DOUBLE;

    static constexpr da_int XTEST_BLOCK_SMALL = KNN_BLOCK_SMALL;
    static constexpr da_int XTEST_BLOCK_MEDIUM = KNN_BLOCK_MEDIUM;
};

// Compute kernel for brute force algorithm
template <typename T>
da_status neighbors<T>::kneighbors_compute_brute_force(da_int n_queries,
                                                       da_int n_features, const T *X_test,
                                                       da_int ldx_test, da_int *n_ind,
                                                       T *n_dist, da_int n_neigh,
                                                       bool return_distance) {
    // Get blocking parameters for knn
    constexpr da_int XTRAIN_BLOCK = knn_block_sizes<T>::XTRAIN_BLOCK;
    constexpr da_int XTEST_BLOCK = knn_block_sizes<T>::XTEST_BLOCK;
    constexpr da_int XTEST_BLOCK_SMALL = knn_block_sizes<T>::XTEST_BLOCK_SMALL;
    constexpr da_int XTEST_BLOCK_MEDIUM = knn_block_sizes<T>::XTEST_BLOCK_MEDIUM;

    if (n_features <= XTEST_BLOCK_SMALL) {
        return neighbors<T>::kneighbors_brute_force_Xtest<XTRAIN_BLOCK,
                                                          XTEST_BLOCK_SMALL>(
            n_queries, n_features, X_test, ldx_test, n_ind, n_dist, n_neigh,
            return_distance);

    } else if (n_features <= XTEST_BLOCK_MEDIUM) {
        return neighbors<T>::kneighbors_brute_force_Xtest<XTRAIN_BLOCK,
                                                          XTEST_BLOCK_MEDIUM>(
            n_queries, n_features, X_test, ldx_test, n_ind, n_dist, n_neigh,
            return_distance);
    } else {
        return neighbors<T>::kneighbors_brute_force_Xtest<XTRAIN_BLOCK, XTEST_BLOCK>(
            n_queries, n_features, X_test, ldx_test, n_ind, n_dist, n_neigh,
            return_distance);
    }
}

// Compute kernel for kd-tree algorithm
template <typename T>
da_status neighbors<T>::kneighbors_compute_kd_tree(da_int n_queries, da_int n_features,
                                                   const T *X_test, da_int ldx_test,
                                                   da_int *n_ind, T *n_dist,
                                                   da_int n_neigh, bool return_distance) {
    // Call the knn_neighbors member function of the k-d tree object
    if (!this->internal_kd_tree) {
        return da_error_bypass(
            this->err, da_status_no_data,
            "k-d tree is not initialized. Please set the training data first.");
    }
    std::vector<da_int> perm_vector;
    std::vector<da_int> k_ind;
    std::vector<T> k_dist;
    try {
        perm_vector.resize(n_neigh);
        k_ind.resize(n_queries * n_neigh);
        k_dist.resize(n_queries * n_neigh);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    this->internal_kd_tree->k_neighbors(n_queries, n_features, X_test, ldx_test, n_neigh,
                                        k_ind.data(), k_dist.data(), this->err);

    // k_neighbors() does not sort the indices and distances, so we need to do it here.
    for (da_int k = 0; k < n_queries; k++) {
        sorted_n_dist_n_ind(n_neigh, k_dist.data() + k * n_neigh,
                            k_ind.data() + k * n_neigh, n_dist + k * n_neigh,
                            n_ind + k * n_neigh, perm_vector.data(), return_distance,
                            this->get_squares);
    }
    return da_status_success;
}

// Compute kernel for ball tree algorithm
template <typename T>
da_status neighbors<T>::kneighbors_compute_ball_tree(da_int n_queries, da_int n_features,
                                                     const T *X_test, da_int ldx_test,
                                                     da_int *n_ind, T *n_dist,
                                                     da_int n_neigh,
                                                     bool return_distance) {
    // Call the knn_neighbors member function of the ball tree object
    if (!this->internal_ball_tree) {
        return da_error_bypass(
            this->err, da_status_no_data,
            "ball tree is not initialized. Please set the training data first.");
    }
    std::vector<da_int> perm_vector;
    std::vector<da_int> k_ind;
    std::vector<T> k_dist;
    try {
        perm_vector.resize(n_neigh);
        k_ind.resize(n_queries * n_neigh);
        k_dist.resize(n_queries * n_neigh);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    this->internal_ball_tree->k_neighbors(n_queries, n_features, X_test, ldx_test,
                                          n_neigh, k_ind.data(), k_dist.data(),
                                          this->err);

    // k_neighbors() does not sort the indices and distances, so we need to do it here.
    for (da_int k = 0; k < n_queries; k++) {
        sorted_n_dist_n_ind(n_neigh, k_dist.data() + k * n_neigh,
                            k_ind.data() + k * n_neigh, n_dist + k * n_neigh,
                            n_ind + k * n_neigh, perm_vector.data(), return_distance,
                            this->get_squares);
    }
    return da_status_success;
}

/**
 * Returns the indices of the k-nearest neighbors for each point in a test data set and, optionally, the
 * corresponding distances to each neighbor.
 *
 * This algorithm has the following steps:
 * - If X_test is nullptr, compute the distance matrix D(X_train, X_train). Otherwise, compute D(X_train, X).
 * - Create a matrix so that its j-th column holds the indices of each point in X_train in ascending order
 *   to the distance, where j is each point in X_test (or X_train when X_test is nullptr).
 * - Return in n_ind only the first k indices for each column (those would be the k-nearest neighbors).
 * - If return_distance is true, return the corresponding distances between each test point and
 *   its neighbors.
 */
template <typename T>
inline __attribute__((__always_inline__)) da_status
neighbors<T>::kneighbors_compute(da_int n_queries, da_int n_features, const T *X_test,
                                 da_int ldx_test, da_int *n_ind, T *n_dist,
                                 da_int n_neigh, bool return_distance) {

    if (this->working_algo == da_neighbors_types::nn_algorithm::brute) {
        return neighbors<T>::kneighbors_compute_brute_force(n_queries, n_features, X_test,
                                                            ldx_test, n_ind, n_dist,
                                                            n_neigh, return_distance);
    } else if (this->working_algo == da_neighbors_types::nn_algorithm::kd_tree) {
        return neighbors<T>::kneighbors_compute_kd_tree(n_queries, n_features, X_test,
                                                        ldx_test, n_ind, n_dist, n_neigh,
                                                        return_distance);
    } else if (this->working_algo == da_neighbors_types::nn_algorithm::ball_tree) {
        return neighbors<T>::kneighbors_compute_ball_tree(n_queries, n_features, X_test,
                                                          ldx_test, n_ind, n_dist,
                                                          n_neigh, return_distance);
    } else {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "Unknown algorithm: " + std::to_string(working_algo) +
                                   ".");
    }
}

/**
 * Returns the indices of the k-nearest neighbors for each point in a test data set and, optionally, the
 * corresponding distances to each neighbor.
 *
 * - If X_test is a nullptr, then throw an error
 * and compute the k-nearest neighbors of the training data matrix provided via set_training_data(),
 * not considering itself as a neighbor.
 * - If X_test is not nullptr, then X_test is the test data matrix of size m-by-n, and for each of its points
 * kneighbors() computes its neighbors in the training data matrix using kneighbors_compute().
 */
template <typename T>
da_status neighbors<T>::kneighbors(da_int n_queries, da_int n_features, const T *X_test,
                                   da_int ldx_test, da_int *n_ind, T *n_dist,
                                   da_int n_neigh, bool return_distance) {
    da_status status = da_status_success;

    // Return if set_data() has not been called
    if (!istrained_Xtrain)
        return da_error_bypass(this->err, da_status_no_data,
                               "No training data have been set. Please call "
                               "da_nn_set_data_s or da_nn_set_data_d.");
    // Check if the parameters are updated and if so, throw an error.
    status = this->check_options_update();
    if (status != da_status_success)
        return status;

    if (n_ind == nullptr) {
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "n_ind is not a valid pointer.");
    }

    const T *X_test_temp = nullptr;
    T *utility_ptr1 = nullptr;
    da_int ldx_test_temp = ldx_test;
    status = validate_and_store_X_test(this, n_queries, n_features, X_test, ldx_test,
                                       &utility_ptr1, &X_test_temp, ldx_test_temp,
                                       this->err, this->n_features);
    if (status != da_status_success)
        return status;

    // Check number of requested neighbors
    if ((n_neigh <= 0 && this->n_neighbors <= 0)) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "Number of requested neighbors must be positive.");
    }
    // If n_neigh is <= 0, use the default value in n_neighbors.
    if (n_neigh <= 0)
        n_neigh = this->n_neighbors;

    // Effective number of neighbors needs to be at most the size of features.
    if (n_neigh > this->n_samples) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "Number of requested neighbors must be at least as big as "
                               "the number of samples.");
    }

    // If distances are requested, check the pointer for outputs is valid.
    if (return_distance) {
        if (n_dist == nullptr) {
            return da_error_bypass(this->err, da_status_invalid_pointer,
                                   "n_dist is not a valid pointer.");
        }
    }

    status = neighbors<T>::kneighbors_compute(n_queries, n_features, X_test_temp,
                                              ldx_test_temp, n_ind, n_dist, n_neigh,
                                              return_distance);

    if (this->order == column_major) {
// If da_int is 64 bit, cast to double
#if defined(AOCLDA_ILP64)
        da_blas::imatcopy('T', n_neigh, n_queries, 1.0, reinterpret_cast<double *>(n_ind),
                          n_neigh, n_queries);
#else // da_int is 32 bit, cast to float
        da_blas::imatcopy('T', n_neigh, n_queries, 1.0, reinterpret_cast<float *>(n_ind),
                          n_neigh, n_queries);
#endif
        // transpose distances
        if (return_distance) {
            da_blas::imatcopy('T', n_neigh, n_queries, 1.0, n_dist, n_neigh, n_queries);
        }
    } else {
        delete[] (utility_ptr1);
    }

    return status;
}

/*
 * From a given distances matrix and a weighting description, compute the
 * corresponding weights to be used for the estimation of the labels.
 */
template <typename T> void get_weights(std::vector<T> &weights, da_int weight_desc) {
    // Potentially avoid a call here by checking for uniformity at a higher level
    if (weight_desc == ::da_neighbors_types::nn_weights::uniform) {
        return;
    } else { // ::da_neighbors_types::nn_weights::distance
        for (da_int i = 0; i < da_int(weights.size()); i++) {
            // If weights=distance is zero then the weight must be one since it's the closest element.
            weights[i] = (weights[i] <= std::numeric_limits<T>::epsilon())
                             ? 1.0
                             : 1.0 / weights[i];
        }
    }
}

template <typename T> da_status neighbors<T>::available_classes() {
    // Return if set_data() has not been called
    if (!istrained_Xtrain)
        return da_error_bypass(this->err, da_status_no_data,
                               "No training data have been set. Please call "
                               "da_nn_set_data_s or da_nn_set_data_d.");
    // Return if set_labels() has not been called
    if (!istrained_labels)
        return da_error_bypass(this->err, da_status_no_data,
                               "No classification labels have been set. Please call "
                               "da_nn_set_labels_s or da_nn_set_labels_d.");
    // From the input data y_train_class, find the available classes.
    try {
        std::vector<da_int> temp_classes(this->y_train_class,
                                         this->y_train_class + this->n_samples);
        std::sort(temp_classes.begin(), temp_classes.end());
        std::vector<da_int>::iterator ip;
        ip = std::unique(temp_classes.begin(), temp_classes.end());
        temp_classes.resize(std::distance(temp_classes.begin(), ip));
        this->classes = std::move(temp_classes);
        this->n_classes = da_int(this->classes.size());
        this->classes_computed = true;
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    return da_status_success;
}

/*
 * Get test data matrix X_test and compute the probability estimates for the test samples.
 * proba is a n_queries-by-n_classes matrix.
 * For each query of the matrix, compute the probability estimate for each of the
 * available classes presented in classes.
 */
template <typename T>
da_status neighbors<T>::predict_proba(da_int n_queries, da_int n_features,
                                      const T *X_test, da_int ldx_test, T *proba,
                                      da_nn_search_mode search_mode) {
    da_status status = da_status_success;
    // Return if set_data() has not been called
    if (!istrained_Xtrain)
        return da_error_bypass(this->err, da_status_no_data,
                               "No training data have been set. Please call "
                               "da_nn_set_data_s or da_nn_set_data_d.");
    // Return if set_labels() has not been called
    if (!istrained_labels)
        return da_error_bypass(this->err, da_status_no_data,
                               "No classification labels have been set. Please call "
                               "da_nn_set_labels_s or da_nn_set_labels_d.");

    const T *X_test_temp = nullptr;
    T *utility_ptr1 = nullptr;
    da_int ldx_test_temp = ldx_test;

    // Check if the parameters are updated and if so, throw an error.
    status = this->check_options_update();
    if (status != da_status_success)
        return status;

    if (proba == nullptr)
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "proba is not a valid pointer.");

    // Most checks occur lower in the call tree, but we need this one to prevent illegal allocation
    if (n_queries < 1)
        return da_error(this->err, da_status_invalid_array_dimension,
                        "Number of queries must be greater than zero.");

    if (n_features < 1)
        return da_error(this->err, da_status_invalid_array_dimension,
                        "Number of features must be greater than zero.");

    status = validate_and_store_X_test(this, n_queries, n_features, X_test, ldx_test,
                                       &utility_ptr1, &X_test_temp, ldx_test_temp,
                                       this->err, this->n_features);
    if (status != da_status_success)
        return status;

    // Call the compute that assumes column-major order
    if (search_mode == knn_search_mode) {
        status = predict_proba_compute_knn(n_queries, n_features, X_test_temp,
                                           ldx_test_temp, proba);
    } else if (search_mode == radius_search_mode) {
        status = predict_proba_compute_rnn(n_queries, n_features, X_test_temp,
                                           ldx_test_temp, proba);
    } else {
        status =
            da_error_bypass(this->err, da_status_invalid_input,
                            "Unknown search mode: " + std::to_string(search_mode) + ".");
    }

    if (this->order == column_major) {
        da_blas::imatcopy('T', n_classes, n_queries, 1.0, proba, n_classes, n_queries);
    } else {
        delete[] (utility_ptr1);
    }

    return status;
}

/*
 * Compute probability estimates for the provided test data so that the probabilities
 * for each observation lie contiguously in memory.
 * Assumes column-major order.
 */
template <typename T>
da_status neighbors<T>::predict_proba_compute_knn(da_int n_queries, da_int n_features,
                                                  const T *X_test, da_int ldx_test,
                                                  T *proba) {
    da_status status = da_status_success;

    if (!this->classes_computed) {
        // From the input data y_train, find the available classes.
        status = neighbors<T>::available_classes();
    }
    if (status != da_status_success)
        return da_error_bypass(this->err, status,
                               "Failed to compute probabilities due to an internal error "
                               "of the available classes computation.");

    // Allocate memory to set neighbors' indices and corresponding distances.
    // If n_ind and n_dist were returned in row order, then we need to transpose them
    try {
        std::vector<da_int> n_ind(n_queries * this->n_neighbors);
        std::vector<T> n_dist;
        // Call kneighbors_compute() so that all neighbours of each observation
        // lies contiguously in memory, same for the distances.
        // kneighbors() returns first all the first neighbors, then all second
        // neighbors and so on.
        if (this->weights == da_neighbors_types::nn_weights::uniform) {
            // Call kneighbors to compute the indices and distances.
            status = kneighbors_compute(n_queries, n_features, X_test, ldx_test,
                                        n_ind.data(), nullptr, this->n_neighbors, false);
        } else if (this->weights == da_neighbors_types::nn_weights::distance) {
            n_dist.resize(n_queries * this->n_neighbors);
            // Call kneighbors to compute the indices and distances.
            status =
                kneighbors_compute(n_queries, n_features, X_test, ldx_test, n_ind.data(),
                                   n_dist.data(), this->n_neighbors, true);
        }

        if (status != da_status_success)
            return da_error_bypass(
                this->err, status,
                "Failed to compute probabilities due to an internal error "
                "of the k-nearest neighbors computation.");
        // Compute the predicted labels.
        // Depending on the indices of the neighbors, for each test data point return the
        // label of each of the neighbors.

        std::vector<da_int> pred_labels(n_queries * this->n_neighbors);

        for (da_int j = 0; j < n_queries; j++)
            for (da_int i = 0; i < this->n_neighbors; i++)
                pred_labels[i + j * this->n_neighbors] =
                    this->y_train_class[n_ind[i + j * this->n_neighbors]];

        da_int num_classes = (da_int)this->classes.size();

        if (this->weights == ::da_neighbors_types::nn_weights::uniform) {
            T denominator;
            // Now that we computed the predicted labels for each neighbor,
            // we use this info to compute the probability for each of the class labels.
            for (da_int j = 0; j < n_queries; j++) {
                denominator = 0.0;
                for (da_int i = 0; i < num_classes; i++) {
                    proba[i + j * num_classes] = 0.0;
                    for (da_int neig = 0; neig < this->n_neighbors; neig++) {
                        if (classes[i] == pred_labels[neig + j * this->n_neighbors])
                            proba[i + j * num_classes]++;
                    }
                    denominator += proba[i + j * num_classes];
                }
                for (da_int i = 0; i < num_classes; i++) {
                    proba[i + j * num_classes] = proba[i + j * num_classes] / denominator;
                }
            }
        } else if (this->weights == ::da_neighbors_types::nn_weights::distance) {
            // Compute the most common value of y_test between the neighbors of each element of X_test.
            // Distance matrix of neighbors has dimensionality of n_queries-by-n_neighbors, so the weight
            // vector should be the same.
            std::vector<T> weight_vector(n_dist);
            get_weights(weight_vector, this->weights);
            T denominator;
            for (da_int j = 0; j < n_queries; j++) {
                denominator = 0.0;
                for (da_int i = 0; i < (da_int)this->classes.size(); i++) {
                    proba[i + j * num_classes] = 0.0;
                    for (da_int neig = 0; neig < this->n_neighbors; neig++)
                        if (classes[i] == pred_labels[neig + j * this->n_neighbors])
                            proba[i + j * num_classes] +=
                                weight_vector[neig + j * this->n_neighbors];
                    denominator += proba[i + j * num_classes];
                }
                for (da_int i = 0; i < num_classes; i++)
                    proba[i + j * num_classes] = proba[i + j * num_classes] / denominator;
            }
        }
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    return status;
}

/*
 * Compute probability estimates for the provided test data based on radius neighbors,
 * so that the probabilities for each observation lie contiguously in memory.
 * Assumes column-major order.
 */
template <typename T>
da_status neighbors<T>::predict_proba_compute_rnn(da_int n_queries, da_int n_features,
                                                  const T *X_test, da_int ldx_test,
                                                  T *proba) {
    da_status status = da_status_success;

    if (!this->classes_computed) {
        // From the input data y_train, find the available classes.
        status = neighbors<T>::available_classes();
    }
    if (status != da_status_success)
        return da_error_bypass(this->err, status,
                               "Failed to compute probabilities due to an internal error "
                               "of the available classes computation.");

    std::vector<da_int> temp_radius_neighbors_count;
    std::vector<da_vector::da_vector<da_int>> temp_radius_neighbors_indices;
    std::vector<da_vector::da_vector<T>> temp_radius_neighbors_distances;

    // Allocate memory to store radius neighbors results for each query
    try {
        bool compute_distances = false;
        if (this->weights == da_neighbors_types::nn_weights::distance)
            compute_distances = true;

        status = radius_neighbors_compute(
            n_queries, n_features, X_test, ldx_test, this->radius,
            temp_radius_neighbors_count, temp_radius_neighbors_indices,
            temp_radius_neighbors_distances, compute_distances, false, true);

        if (status != da_status_success)
            return da_error_bypass(this->err, status,
                                   "Failed to compute probabilities due to an internal "
                                   "error of the radius neighbors computation.");
        // Compute the predicted labels.
        // Access copied data from local vectors
        // Compute predicted labels for neighbors
        da_int num_classes = (da_int)this->classes.size();

        // Compute probabilities for each query
        if (this->weights == da_neighbors_types::nn_weights::uniform) {
            T denominator;
            // First compute the predicted labels for each neighbor, then
            // use this info to compute the probability for each of the class labels.
            for (da_int j = 0; j < n_queries; j++) {
                da_int n_neigh = temp_radius_neighbors_count[j];
                if (n_neigh == 0) {
                    return da_error_bypass(this->err, da_status_operation_failed,
                                           "Failed to compute probabilities for query " +
                                               std::to_string(j) +
                                               " since it does not have any neighbors "
                                               "within the specified radius.");
                }
                denominator = 0.0;
                // Count neighbors for each class
                da_int j_local = j * num_classes;
                for (da_int i = 0; i < num_classes; i++) {
                    proba[i + j_local] = 0.0;
                    for (da_int neig = 0; neig < n_neigh; neig++) {
                        da_int neighbor_idx = temp_radius_neighbors_indices[j][neig];
                        if (this->classes[i] == this->y_train_class[neighbor_idx]) {
                            proba[i + j_local]++;
                        }
                    }
                    denominator += proba[i + j_local];
                }
                for (da_int i = 0; i < num_classes; i++) {
                    proba[i + j_local] = proba[i + j_local] / denominator;
                }
            }
        } else if (this->weights == da_neighbors_types::nn_weights::distance) {
            T denominator;
            // Distance-weighted probabilities
            for (da_int j = 0; j < n_queries; j++) {
                denominator = 0.0;
                da_int n_neigh = temp_radius_neighbors_count[j];
                if (n_neigh == 0) {
                    return da_error_bypass(this->err, da_status_operation_failed,
                                           "Failed to compute probabilities for query " +
                                               std::to_string(j) +
                                               " since it does not have any neighbors "
                                               "within the specified radius.");
                }
                // Copy distances to weight vector, converting from squared to actual distances if needed
                std::vector<T> weight_vector(n_neigh);
                if (this->get_squares) {
                    for (da_int neig = 0; neig < n_neigh; neig++) {
                        weight_vector[neig] =
                            std::sqrt(temp_radius_neighbors_distances[j][neig]);
                    }
                } else {
                    for (da_int neig = 0; neig < n_neigh; neig++) {
                        weight_vector[neig] = temp_radius_neighbors_distances[j][neig];
                    }
                }

                get_weights(weight_vector, this->weights);

                da_int j_local = j * num_classes;
                for (da_int i = 0; i < num_classes; i++) {
                    proba[i + j_local] = 0.0;
                    for (da_int neig = 0; neig < n_neigh; neig++) {
                        da_int neighbor_idx = temp_radius_neighbors_indices[j][neig];
                        if (this->classes[i] == this->y_train_class[neighbor_idx]) {
                            proba[i + j_local] += weight_vector[neig];
                        }
                    }
                    denominator += proba[i + j_local];
                }

                for (da_int i = 0; i < num_classes; i++)
                    proba[i + j_local] = proba[i + j_local] / denominator;
            }
        }

    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    return status;
}

/*
 * Predict the class labels for the provided test data using either k-nearest neighbors
 * or radius neighbors.
 * Computes the probability estimates for each class based on neighbors 
 * and returns the class with the highest probability.
 */
template <typename T>
da_status neighbors<T>::predict(da_int n_queries, da_int n_features, const T *X_test,
                                da_int ldx_test, da_int *y_test,
                                da_nn_search_mode search_mode) {
    da_status status = da_status_success;

    // Check if the parameters are updated and if so, throw an error.
    status = this->check_options_update();
    if (status != da_status_success)
        return status;

    // Return if set_data() has not been called
    if (!istrained_Xtrain)
        return da_error_bypass(this->err, da_status_no_data,
                               "No training data have been set. Please call "
                               "da_nn_set_data_s or da_nn_set_data_d.");
    // Return if set_labels() has not been called
    if (!istrained_labels)
        return da_error_bypass(this->err, da_status_no_data,
                               "No classification labels have been set. Please call "
                               "da_nn_set_labels_s or da_nn_set_labels_d.");

    if (y_test == nullptr)
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "y_test is not a valid pointer.");

    // Only test n_queries before memory allocation since the rest will be tested
    // in predict_proba.
    if (n_queries < 1) {
        return da_error_bypass(this->err, da_status_invalid_array_dimension,
                               "n_queries must be greater than 0.");
    }

    const T *X_test_temp = nullptr;
    T *utility_ptr1 = nullptr;
    da_int ldx_test_temp = ldx_test;
    status = validate_and_store_X_test(this, n_queries, n_features, X_test, ldx_test,
                                       &utility_ptr1, &X_test_temp, ldx_test_temp,
                                       this->err, this->n_features);
    if (status != da_status_success)
        return status;

    if (!this->classes_computed) {
        // From the input data y_train, find the available classes.
        status = neighbors<T>::available_classes();
    }
    if (status != da_status_success)
        return da_error_bypass(this->err, status,
                               "Failed to compute probabilities due to an internal error "
                               "of the available classes computation.");

    std::vector<T> proba;
    try {
        proba.resize(n_queries * this->n_classes);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    // Call the compute that assumes column-major order
    if (search_mode == knn_search_mode) {
        status = neighbors<T>::predict_proba_compute_knn(
            n_queries, n_features, X_test_temp, ldx_test_temp, proba.data());
    } else if (search_mode == radius_search_mode) {
        status = neighbors<T>::predict_proba_compute_rnn(
            n_queries, n_features, X_test_temp, ldx_test_temp, proba.data());
    } else {
        status =
            da_error_bypass(this->err, da_status_invalid_input,
                            "Unknown search mode: " + std::to_string(search_mode) + ".");
    }
    if (status != da_status_success)
        return da_error_bypass(this->err, status,
                               "Failed to compute predicted labels due to an internal "
                               "error of predicting the probabilities.");

    // For each column of proba, check which label appears the most times.
    // In case of a tie, return the first label.
    da_int max_index;
    for (da_int i = 0; i < n_queries; i++) {
        max_index =
            da_blas::cblas_iamax(this->n_classes, proba.data() + i * this->n_classes, 1);
        y_test[i] = this->classes[max_index];
    }

    if (this->order == row_major) {
        delete[] (utility_ptr1);
    }

    return status;
}

/*
 * Predict the targets y_test for the provided test data.
 * Compute the nearest neighbors and return the corresponding target according to the target of the neighbors.
 */
template <typename T>
da_status neighbors<T>::predict(da_int n_queries, da_int n_features, const T *X_test,
                                da_int ldx_test, T *y_test,
                                da_nn_search_mode search_mode) {

    da_status status = da_status_success;
    // Return if set_data() has not been called
    if (!istrained_Xtrain)
        return da_error_bypass(this->err, da_status_no_data,
                               "No training data have been set. Please call "
                               "da_nn_set_data_s or da_nn_set_data_d.");
    // Return if set_targets() has not been called
    if (!istrained_targets)
        return da_error_bypass(this->err, da_status_no_data,
                               "No regression targets have been set. Please call "
                               "da_nn_set_targets_s or da_nn_set_targets_d.");

    // Check if the parameters are updated and if so, throw an error.
    status = this->check_options_update();
    if (status != da_status_success)
        return status;

    if (y_test == nullptr)
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "y_test is not a valid pointer.");

    // Test n_queries before memory allocation
    if (n_queries < 1) {
        return da_error_bypass(this->err, da_status_invalid_array_dimension,
                               "n_queries must be greater than 0.");
    }

    const T *X_test_temp = nullptr;
    T *utility_ptr1 = nullptr;
    da_int ldx_test_temp = ldx_test;

    status = validate_and_store_X_test(this, n_queries, n_features, X_test, ldx_test,
                                       &utility_ptr1, &X_test_temp, ldx_test_temp,
                                       this->err, this->n_features);
    if (status != da_status_success)
        return status;

    if (search_mode == knn_search_mode) {
        status = predict_targets_knn(n_queries, n_features, X_test_temp, ldx_test_temp,
                                     y_test);
    } else if (search_mode == radius_search_mode) {
        status = predict_targets_rnn(n_queries, n_features, X_test_temp, ldx_test_temp,
                                     y_test);
    } else {
        status =
            da_error_bypass(this->err, da_status_invalid_input,
                            "Unknown search mode: " + std::to_string(search_mode) + ".");
    }

    if (this->order == row_major) {
        delete[] (utility_ptr1);
    }

    return status;
}

/*
 * Predict the targets y_test for the provided test data.
 * Compute the nearest neighbors and return the corresponding target according to the target of the neighbors.
 */
template <typename T>
inline __attribute__((__always_inline__)) da_status
neighbors<T>::predict_targets_knn(da_int n_queries, da_int n_features, const T *X_test,
                                  da_int ldx_test, T *y_test) {

    da_status status = da_status_success;

    // Allocate memory to set neighbors' indices and corresponding distances.
    // If n_ind and n_dist were returned in row order, then we need to transpose them
    try {
        std::vector<da_int> n_ind(n_queries * this->n_neighbors);
        std::vector<T> n_dist;
        if (this->weights == da_neighbors_types::nn_weights::uniform) {
            // Call kneighbors to compute the indices and distances.
            status = kneighbors_compute(n_queries, n_features, X_test, ldx_test,
                                        n_ind.data(), nullptr, this->n_neighbors, false);
        } else if (this->weights == da_neighbors_types::nn_weights::distance) {
            n_dist.resize(n_queries * this->n_neighbors);
            // Call kneighbors to compute the indices and distances.
            status =
                kneighbors_compute(n_queries, n_features, X_test, ldx_test, n_ind.data(),
                                   n_dist.data(), this->n_neighbors, true);
        }
        if (status != da_status_success)
            return da_error_bypass(
                this->err, status,
                "Failed to compute probabilities due to an internal error "
                "of the k-nearest neighbors computation.");

        // Depending on the weights, compute the predicted target for each test data point
        // using the targets of the neighbors.
        if (this->weights == da_neighbors_types::nn_weights::uniform) {
            // Compute the predicted targets.
            // Depending on the indices of the neighbors, for each test data point return the
            // target of each of the neighbors.
            std::vector<T> pred_targets(n_queries * this->n_neighbors);

            for (da_int j = 0; j < n_queries; j++)
                for (da_int i = 0; i < this->n_neighbors; i++)
                    pred_targets[i + j * this->n_neighbors] =
                        this->y_train_reg[n_ind[i + j * this->n_neighbors]];

            for (da_int j = 0; j < n_queries; j++) {
                status = da_basic_statistics::mean(
                    column_major, da_axis_col, this->n_neighbors, 1,
                    pred_targets.data() + j * this->n_neighbors, this->n_neighbors,
                    y_test + j);
            }
        } else if (this->weights == da_neighbors_types::nn_weights::distance) {
            // Compute the most common value of y_test between the neighbors of each element of X_test.
            // Distance matrix of neighbors has dimensionality of n_queries-by-n_neighbors, so the weight
            // vector should be the same.
            std::vector<T> weight_vector(n_dist);
            get_weights(weight_vector, this->weights);
            T denominator;
            for (da_int j = 0; j < n_queries; j++) {
                denominator = 0.0;
                y_test[j] = 0.0; // Initialize the output to zero.
                for (da_int i = 0; i < this->n_neighbors; i++) {
                    y_test[j] += this->y_train_reg[n_ind[i + j * this->n_neighbors]] *
                                 weight_vector[i + j * this->n_neighbors];
                    denominator += weight_vector[i + j * this->n_neighbors];
                }
                y_test[j] = y_test[j] / denominator;
            }
        }

    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    return da_status_success; // LCOV_EXCL_LINE
}

/*
 * Predict the targets y_test for the provided test data.
 * Compute the nearest neighbors and return the corresponding target according to the target of the neighbors.
 */
template <typename T>
inline __attribute__((__always_inline__)) da_status
neighbors<T>::predict_targets_rnn(da_int n_queries, da_int n_features, const T *X_test,
                                  da_int ldx_test, T *y_test) {

    da_status status = da_status_success;

    std::vector<da_int> temp_radius_neighbors_count;
    std::vector<da_vector::da_vector<da_int>> temp_radius_neighbors_indices;
    std::vector<da_vector::da_vector<T>> temp_radius_neighbors_distances;

    try {
        bool compute_distances = false;
        if (this->weights == da_neighbors_types::nn_weights::distance)
            compute_distances = true;

        status = radius_neighbors_compute(
            n_queries, n_features, X_test, ldx_test, this->radius,
            temp_radius_neighbors_count, temp_radius_neighbors_indices,
            temp_radius_neighbors_distances, compute_distances, false, true);

        if (status != da_status_success)
            return da_error_bypass(this->err, status,
                                   "Failed to compute probabilities due to an internal "
                                   "error of the radius neighbors computation.");

        // Depending on the weights, compute the predicted target for each test data point
        // using the targets of the neighbors.
        if (this->weights == da_neighbors_types::nn_weights::uniform) {
            // The size of the maximum number of neighbors for any query
            da_int max_n_neighbors = 0;
            max_n_neighbors = *std::max_element(temp_radius_neighbors_count.begin(),
                                                temp_radius_neighbors_count.end() - 1);
            // Compute the predicted targets.
            // Depending on the indices of the neighbors, for each test data point return the
            // target of each of the neighbors.
            std::vector<T> pred_targets(max_n_neighbors, 0.0);

            for (da_int j = 0; j < n_queries; j++) {
                da_int n_neigh = temp_radius_neighbors_count[j];
                if (n_neigh == 0) {
                    return da_error_bypass(this->err, da_status_operation_failed,
                                           "Failed to compute probabilities for query " +
                                               std::to_string(j) +
                                               " since it does not have any neighbors "
                                               "within the specified radius.");
                }
                for (da_int i = 0; i < n_neigh; i++) {
                    da_int neighbor_idx = temp_radius_neighbors_indices[j][i];
                    pred_targets[i] = this->y_train_reg[neighbor_idx];
                }
                status =
                    da_basic_statistics::mean(column_major, da_axis_col, n_neigh, 1,
                                              pred_targets.data(), n_neigh, y_test + j);
            }

        } else if (this->weights == da_neighbors_types::nn_weights::distance) {
            T denominator;
            for (da_int j = 0; j < n_queries; j++) {
                denominator = 0.0;
                da_int n_neigh = temp_radius_neighbors_count[j];
                if (n_neigh == 0) {
                    return da_error_bypass(this->err, da_status_operation_failed,
                                           "Failed to compute probabilities for query " +
                                               std::to_string(j) +
                                               " since it does not have any neighbors "
                                               "within the specified radius.");
                }
                // Copy distances to weight vector, converting from squared to actual distances if needed
                std::vector<T> weight_vector(n_neigh);
                if (this->get_squares) {
                    for (da_int neig = 0; neig < n_neigh; neig++) {
                        weight_vector[neig] =
                            std::sqrt(temp_radius_neighbors_distances[j][neig]);
                    }
                } else {
                    for (da_int neig = 0; neig < n_neigh; neig++) {
                        weight_vector[neig] = temp_radius_neighbors_distances[j][neig];
                    }
                }

                get_weights(weight_vector, this->weights);

                y_test[j] = 0.0; // Initialize the output to zero.
                for (da_int i = 0; i < n_neigh; i++) {
                    da_int neighbor_idx = temp_radius_neighbors_indices[j][i];
                    y_test[j] += this->y_train_reg[neighbor_idx] * weight_vector[i];
                    denominator += weight_vector[i];
                }
                y_test[j] = y_test[j] / denominator;
            }
        }

    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    return status;
}

// Implementing refresh
template <typename T> void neighbors<T>::refresh() { is_up_to_date = false; }

// Compute the radius nearest neighbors and optionally the corresponding distances
// Includes the appropriate checks for input arguments
template <typename T>
da_status neighbors<T>::radius_neighbors(da_int n_queries, da_int n_features,
                                         const T *X_test, da_int ldx_test, T r,
                                         bool return_distance, bool sort_results) {
    da_status status = da_status_success;
    if ((!return_distance) && sort_results)
        return da_error_bypass(this->err, da_status_invalid_input,
                               "Cannot sort results if distances are not returned.");

    this->sort_results = sort_results;
    this->rnn_return_distances = return_distance;

    // Return if set_data() has not been called
    if (!istrained_Xtrain)
        return da_error_bypass(this->err, da_status_no_data,
                               "No training data have been set. Please call "
                               "da_nn_set_data_s or da_nn_set_data_d.");

    // Check if the parameters are updated and if so, throw an error.
    status = this->check_options_update();
    if (status != da_status_success)
        return status;

    const T *X_test_temp = nullptr;
    T *utility_ptr1 = nullptr;
    da_int ldx_test_temp = ldx_test;
    status = validate_and_store_X_test(this, n_queries, n_features, X_test, ldx_test,
                                       &utility_ptr1, &X_test_temp, ldx_test_temp,
                                       this->err, this->n_features);
    if (status != da_status_success)
        return status;

    // Check radius of requested neighbors
    if (r < 0.0 && this->radius < 0.0) {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "Radius of requested neighbors must be non-negative.");
    }
    // If radius is < 0, use the default value in n_neighbors.
    if (r < 0) {
        r = this->radius;
    }

    status = neighbors<T>::radius_neighbors_compute(
        n_queries, n_features, X_test_temp, ldx_test_temp, r,
        this->radius_neighbors_count, this->radius_neighbors_indices,
        this->radius_neighbors_distances, return_distance, sort_results, false);

    if (this->order == row_major) {
        delete[] (utility_ptr1);
    }

    return status;
}

// Compute kernel for the radius nearest neighbors and optionally the corresponding distances
// so that all neighbours of each observation lies contiguously in memory, same for the distances.
// Assumes column-major order.
template <typename T>
da_status neighbors<T>::radius_neighbors_compute(
    da_int n_queries, da_int n_features, const T *X_test, da_int ldx_test, T radius,
    std::vector<da_int> &rnn_count,
    std::vector<da_vector::da_vector<da_int>> &rnn_indices,
    std::vector<da_vector::da_vector<T>> &rnn_distances, bool return_distances,
    bool sort_results, bool is_temp) {
    if (!is_temp) {
        // If radius neighbors were already computed, clean up memory of radius neighbors and (optionally) distances
        if (this->radius_neighbors_computed) {
            this->radius_neighbors_count.clear();
            this->radius_neighbors_indices.clear();
            this->radius_neighbors_distances.clear();
            this->radius_neighbors_computed = false;
        }
    }
    // Allocate memory
    try {
        rnn_indices.resize(n_queries);
        rnn_count.resize(n_queries);
        if (return_distances)
            rnn_distances.resize(n_queries);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    da_status status = da_status_success;
    if (this->working_algo == da_neighbors_types::nn_algorithm::brute) {
        status = neighbors<T>::radius_neighbors_compute_brute_force(
            n_queries, n_features, X_test, ldx_test, radius, rnn_indices, rnn_distances,
            return_distances);
    } else if (this->working_algo == da_neighbors_types::nn_algorithm::kd_tree) {
        status = neighbors<T>::radius_neighbors_compute_kd_tree(
            n_queries, n_features, X_test, ldx_test, radius, rnn_indices, rnn_distances,
            return_distances);
    } else if (this->working_algo == da_neighbors_types::nn_algorithm::ball_tree) {
        status = neighbors<T>::radius_neighbors_compute_ball_tree(
            n_queries, n_features, X_test, ldx_test, radius, rnn_indices, rnn_distances,
            return_distances);
    } else {
        return da_error_bypass(this->err, da_status_invalid_input, // LCOV_EXCL_LINE
                               "Unknown algorithm: " + std::to_string(working_algo) +
                                   ".");
    }

    for (da_int j = 0; j < n_queries; j++) {
        rnn_count[j] = rnn_indices[j].size();
    }

    // Sort results only if distances are computed (would fail in radius_neighbors() otherwise)
    // This will be called from prediction with sort_results as false but ensure robustness
    // we add another condition.
    if (sort_results && (!is_temp)) {
        // Use std::max_element to get an iterator to the maximum element
        auto max_it = std::max_element(rnn_count.begin(), rnn_count.end());
        // Dereference the iterator to get the actual maximum value
        da_int max_value = *max_it;
        std::vector<da_int> perm_vector;
        try {
            perm_vector.resize(max_value);
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }
        for (da_int query_index = 0; query_index < n_queries; query_index++) {
            da_int n_neighbors = rnn_count[query_index];
            da_vector::da_vector<da_int> temp_ind;
            da_vector::da_vector<T> temp_dist;
            try {
                temp_ind = rnn_indices[query_index];
                temp_dist = rnn_distances[query_index];
            } catch (std::bad_alloc const &) {
                return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                                "Memory allocation failed.");
            }
            sorted_n_dist_n_ind(n_neighbors, temp_dist.data(), temp_ind.data(),
                                rnn_distances[query_index].data(),
                                rnn_indices[query_index].data(), perm_vector.data(),
                                return_distances, this->get_squares);
        }
    }

    if (!is_temp) {
        this->radius_neighbors_computed = true;
    }
    return status;
}

template <typename T> struct rnn_block_sizes {
    static constexpr da_int XTEST_BLOCK =
        std::is_same<T, float>::value ? RNN_BLOCK_FLOAT : RNN_BLOCK_DOUBLE;
};

/*
Compute the radius neighbors: for each sample point, the indices of the samples within a given
radius are returned. The brute-force method is used.
*/
template <typename T>
da_status neighbors<T>::radius_neighbors_compute_brute_force(
    da_int n_queries, da_int n_features, const T *X_test, da_int ldx_test, T radius,
    std::vector<da_vector::da_vector<da_int>> &rnn_indices,
    std::vector<da_vector::da_vector<T>> &rnn_distances, bool return_distances) {
    // Set the working radius to brute force
    T working_radius = radius;
    if (get_squares)
        working_radius = radius * radius;
    // 2D blocking scheme and threading scheme
    // Blocking X_test on the n_queries dimension
    da_int xtest_block_size = std::min(XTEST_RNN_BLOCK_SIZE, n_queries);
    da_int xtest_block_rem, xtest_n_blocks;
    ARCH::da_utils::blocking_scheme(n_queries, xtest_block_size, xtest_n_blocks,
                                    xtest_block_rem);
    // Blocking X_train on the n_samples dimension
    da_int xtrain_block_size = std::min(XTRAIN_RNN_BLOCK_SIZE, this->n_samples);
    da_int xtrain_block_rem, xtrain_n_blocks;
    ARCH::da_utils::blocking_scheme(this->n_samples, xtrain_block_size, xtrain_n_blocks,
                                    xtrain_block_rem);
    // In total we have xtest_n_blocks*xtrain_n_blocks blocks to process with regard to D
    da_int n_threads =
        ARCH::da_utils::get_n_threads_loop(xtest_n_blocks * xtrain_n_blocks);
    // Will be used to store the distance computations
    // One D matrix per thread
    std::vector<std::vector<T>> D;

    try {
        D.resize(n_threads);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    // Each matrix D will be of size at most (xtrain_block_size x xtest_block_size)
    // so set the leading dimension accordingly
    da_int ldd = xtrain_block_size;

    da_int threading_error = 0;

    // Local storage for neighbors to help avoid thread contention
    std::vector<std::vector<da_vector::da_vector<da_int>>> neighbors_local_indices(
        n_threads);
    std::vector<std::vector<da_vector::da_vector<T>>> neighbors_local_distances(
        n_threads);

#pragma omp parallel num_threads(n_threads) default(none) shared(                        \
        threading_error, rnn_indices, rnn_distances, xtrain_block_size,                  \
            xtrain_block_rem, xtrain_n_blocks, xtest_block_size, xtest_block_rem,        \
            xtest_n_blocks, n_samples, n_queries, D, ldd, working_radius, n_features,    \
            X_test, ldx_test, X_train, ldx_train, neighbors_local_indices,               \
            neighbors_local_distances, internal_metric, p, n_threads, return_distances)
    {
        // Thread 0 can write to neighbors; all other threads need to use neighbors_local_indices
        da_int this_thread = omp_get_thread_num();
        da_int local_error = 0;
        auto &this_D = D[this_thread];

        try {
            if (this_thread > 0) {
                neighbors_local_indices[this_thread].resize(n_queries);
                if (return_distances)
                    neighbors_local_distances[this_thread].resize(n_queries);
            }
            this_D.resize(xtrain_block_size * xtest_block_size);
        } catch (std::bad_alloc const &) {
#pragma omp atomic write
            threading_error = 1;
        }
#pragma omp for collapse(2) schedule(guided) nowait
        for (da_int j = 0; j < xtest_n_blocks; j++) {
            for (da_int i = 0; i < xtrain_n_blocks; i++) {
#pragma omp atomic read
                local_error = threading_error;
                if (local_error == 0) {
                    da_int local_xtest_block_size = xtest_block_size;
                    if (j == xtest_n_blocks - 1 && xtest_block_rem > 0)
                        local_xtest_block_size = xtest_block_rem;
                    da_int local_xtrain_block_size = xtrain_block_size;
                    if (i == xtrain_n_blocks - 1 && xtrain_block_rem > 0)
                        local_xtrain_block_size = xtrain_block_rem;
                    // Compute the distance matrix using the specified metric
                    da_status thd_status =
                        ARCH::da_metrics::pairwise_distances::pairwise_distance_kernel(
                            da_order::column_major, local_xtrain_block_size,
                            local_xtest_block_size, n_features,
                            X_train + i * xtrain_block_size, ldx_train,
                            X_test + j * xtest_block_size, ldx_test, this_D.data(), ldd,
                            p, this->internal_metric);
                    if (thd_status != da_status_success) {
#pragma omp atomic write
                        threading_error = 1;
                    }

                    // Iterate through the distance matrix and store the indices of the samples within the radius
                    for (da_int jj = 0; jj < local_xtest_block_size; jj++) {
                        for (da_int ii = 0; ii < local_xtrain_block_size; ii++) {
                            // i_local and j_local correspond to the actual sample point indices we are considering
                            da_int i_local = ii + i * xtrain_block_size;
                            da_int j_local = jj + j * xtest_block_size;
                            if (this_D[ii + jj * ldd] <= working_radius) {
                                try {
                                    if (this_thread == 0) {
                                        rnn_indices[j_local].push_back(i_local);
                                        if (return_distances) {
                                            rnn_distances[j_local].push_back(
                                                this_D[ii + jj * ldd]);
                                        }
                                    } else {
                                        neighbors_local_indices[this_thread][j_local]
                                            .push_back(i_local);
                                        if (return_distances) {
                                            neighbors_local_distances
                                                [this_thread][j_local]
                                                    .push_back(this_D[ii + jj * ldd]);
                                        }
                                    }
                                } catch (std::bad_alloc const &) {
#pragma omp atomic write
                                    threading_error = 1;
                                }
                            }
                        }
                    } // End of distance matrix iteration to compute local neighbors
                }     // end of local_error check
            }         // End of xtrain blocks
        }             // End of xtest blocks

        this_D = std::vector<T>{};

#pragma omp barrier

#pragma omp atomic read
        local_error = threading_error;
        if (local_error == 0) {
#pragma omp for schedule(guided)
            // Merge the local neighbors into the global radius_neighbors_indices
            for (da_int i = 0; i < n_queries; i++) {
                for (da_int t = 1; t < n_threads; t++) {
                    rnn_indices[i].append(neighbors_local_indices[t][i]);
                }
            }

            if (return_distances) {
#pragma omp for schedule(guided)
                // Merge the local distances into the global radius_neighbors_distances
                for (da_int i = 0; i < n_queries; i++) {
                    for (da_int t = 1; t < n_threads; t++) {
                        rnn_distances[i].append(neighbors_local_distances[t][i]);
                    }
                }
            }
        }
        neighbors_local_indices[this_thread] =
            std::vector<da_vector::da_vector<da_int>>{};
        neighbors_local_distances[this_thread] = std::vector<da_vector::da_vector<T>>{};

    } // End of parallel region

    if (threading_error != 0)
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    return da_status_success;
}

// Compute kernel for k-d tree algorithm
template <typename T>
da_status neighbors<T>::radius_neighbors_compute_kd_tree(
    da_int n_queries, da_int n_features, const T *X_test, da_int ldx_test, T radius,
    std::vector<da_vector::da_vector<da_int>> &rnn_indices,
    std::vector<da_vector::da_vector<T>> &rnn_distances, bool return_distances) {
    // Call the knn_neighbors member function of the k-d tree object
    if (!this->internal_kd_tree) {
        return da_error_bypass(
            this->err, da_status_no_data,
            "k-d tree is not initialized. Please set the training data first.");
    }
    return this->internal_kd_tree->radius_neighbors(
        n_queries, n_features, X_test, ldx_test, radius, rnn_indices, rnn_distances,
        return_distances, this->err);
}

// Compute kernel for ball tree algorithm
template <typename T>
da_status neighbors<T>::radius_neighbors_compute_ball_tree(
    da_int n_queries, da_int n_features, const T *X_test, da_int ldx_test, T radius,
    std::vector<da_vector::da_vector<da_int>> &rnn_indices,
    std::vector<da_vector::da_vector<T>> &rnn_distances, bool return_distances) {
    // Call the knn_neighbors member function of the ball tree object
    if (!this->internal_ball_tree) {
        return da_error_bypass(
            this->err, da_status_no_data,
            "ball tree is not initialized. Please set the training data first.");
    }

    return this->internal_ball_tree->radius_neighbors(
        n_queries, n_features, X_test, ldx_test, radius, rnn_indices, rnn_distances,
        return_distances, this->err);
}

// Return the number of radius neighbors for each query point
template <typename T>
da_status neighbors<T>::radius_neighbors_count_internal(da_int n_count,
                                                        da_int *n_radius_neighbors) {
    da_int count = 0;
    for (da_int i = 0; i < n_count - 1; i++) {
        n_radius_neighbors[i] = radius_neighbors_count[i];
        count += n_radius_neighbors[i];
    }
    n_radius_neighbors[n_count - 1] = count;
    return da_status_success;
}

// Extract the radius neighbors for the sample point query_index
template <typename T>
da_status neighbors<T>::extract_radius_neighbors_indices(da_int query_index,
                                                         da_int n_neighbors,
                                                         da_int *neighbors_indices) {
    // Copy the indices of the neighbors
    for (da_int i = 0; i < n_neighbors; i++) {
        neighbors_indices[i] = this->radius_neighbors_indices[query_index][i];
    }

    return da_status_success;
}

// Extract the radius neighbors for the sample point query_index
template <typename T>
da_status neighbors<T>::extract_radius_neighbors_distances(da_int query_index,
                                                           da_int n_neighbors,
                                                           T *neighbors_distances) {
    // Copy the distances of the neighbors
    if ((!this->sort_results) && this->get_squares) {
        // If results were sorted, distances are stored in radius_neighbors_distances
        for (da_int i = 0; i < n_neighbors; i++) {
            neighbors_distances[i] =
                std::sqrt(this->radius_neighbors_distances[query_index][i]);
        }
    } else {
        for (da_int i = 0; i < n_neighbors; i++) {
            neighbors_distances[i] = this->radius_neighbors_distances[query_index][i];
        }
    }

    return da_status_success;
}

// Force specific template instantiations
template da_status
neighbors<float>::kneighbors_brute_force_Xtest<KNN_BLOCK_FLOAT, KNN_BLOCK_FLOAT>(
    da_int, da_int, const float *, da_int, da_int *, float *, da_int, bool);

template da_status
neighbors<double>::kneighbors_brute_force_Xtest<KNN_BLOCK_DOUBLE, KNN_BLOCK_DOUBLE>(
    da_int, da_int, const double *, da_int, da_int *, double *, da_int, bool);

template da_status
neighbors<float>::kneighbors_brute_force_Xtest<KNN_BLOCK_FLOAT, KNN_BLOCK_SMALL>(
    da_int, da_int, const float *, da_int, da_int *, float *, da_int, bool);

template da_status
neighbors<double>::kneighbors_brute_force_Xtest<KNN_BLOCK_DOUBLE, KNN_BLOCK_SMALL>(
    da_int, da_int, const double *, da_int, da_int *, double *, da_int, bool);

template da_status
neighbors<float>::kneighbors_brute_force_Xtest<KNN_BLOCK_FLOAT, KNN_BLOCK_MEDIUM>(
    da_int, da_int, const float *, da_int, da_int *, float *, da_int, bool);

template da_status
neighbors<double>::kneighbors_brute_force_Xtest<KNN_BLOCK_DOUBLE, KNN_BLOCK_MEDIUM>(
    da_int, da_int, const double *, da_int, da_int *, double *, da_int, bool);

template class neighbors<double>;
template class neighbors<float>;
} // namespace da_neighbors

} // namespace ARCH
