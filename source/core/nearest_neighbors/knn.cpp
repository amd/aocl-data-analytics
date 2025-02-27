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

#include "knn.hpp"
#include "aoclda.h"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "da_omp.hpp"
#include "da_std.hpp"
#include "knn_options.hpp"
#include "macros.h"
#include "pairwise_distances.hpp"
#include <numeric>

namespace ARCH {

namespace da_knn {

template <typename T> knn<T>::~knn() {
    // Destructor needs to handle arrays that were allocated due to row major storage of input data
    if (X_train_temp)
        delete[] (X_train_temp);
}

template <typename T> knn<T>::knn(da_errors::da_error_t &err) : basic_handle<T>(err) {
    // Initialize the options registry
    // Any error is stored err->status[.] and this NEEDS to be checked
    // by the caller.
    register_knn_options<T>(this->opts, *this->err);
}

template <typename T>
da_status knn<T>::get_result([[maybe_unused]] da_result query,
                             [[maybe_unused]] da_int *dim, [[maybe_unused]] T *result) {
    return da_warn_bypass(this->err, da_status_unknown_query,
                          "There are no floating-point results available for this API.");
}

template <typename T>
da_status knn<T>::get_result([[maybe_unused]] da_result query,
                             [[maybe_unused]] da_int *dim,
                             [[maybe_unused]] da_int *result) {
    da_status status = da_status_success;
    if (!istrained)
        return da_warn_bypass(this->err, da_status_unknown_query,
                              "Handle does not contain data relevant to this query. "
                              "Model needs to be trained.");
    if (!is_up_to_date)
        status = knn<T>::set_params();
    // Pointers were already tested in the generic get_result
    da_int knn_info_size = 6;
    switch (query) {
    case da_result::da_knn_model_params:
        if (*dim < knn_info_size) {
            *dim = knn_info_size;
            return da_warn_bypass(this->err, da_status_invalid_array_dimension,
                                  "The array is too small. Please provide an array of at "
                                  "least size: " +
                                      std::to_string(knn_info_size) + ".");
        }
        result[0] = da_int(n_neighbors);
        result[1] = da_int(algo);
        result[2] = da_int(metric);
        result[3] = da_int(weights);
        result[4] = da_int(n_features);
        result[5] = da_int(n_samples);
        break;
    default:
        return da_warn_bypass(this->err, da_status_unknown_query,
                              "The requested result could not be found.");
    }
    return status;
}

template <typename T> da_status knn<T>::set_params() {
    // Extract options
    std::string opt_val;
    bool opt_pass = true;
    opt_pass &= this->opts.get("number of neighbors", n_neighbors) == da_status_success;
    opt_pass &= this->opts.get("algorithm", opt_val, algo) == da_status_success;
    opt_pass &= this->opts.get("metric", opt_val, metric) == da_status_success;
    opt_pass &= this->opts.get("weights", opt_val, weights) == da_status_success;
    opt_pass &= this->opts.get("minkowski parameter", p) == da_status_success;

    if (!opt_pass)
        return da_error_bypass(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                               "Unexpected error while reading the optional parameters.");
    internal_metric = da_metric(metric);

    if (metric == da_euclidean) {
        this->get_squares = true;
        internal_metric = da_sqeuclidean;
    }
    this->is_up_to_date = true;
    return da_status_success;
}

template <typename T>
da_status knn<T>::set_training_data(da_int n_samples, da_int n_features, const T *X_train,
                                    da_int ldx_train, const da_int *y_train) {

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

    status = this->check_1D_array(n_samples, y_train, "n_samples", "y_train", 1);
    if (status != da_status_success)
        return status;

    // Set internal pointers to user data
    this->y_train = y_train;
    this->n_samples = n_samples;
    this->n_features = n_features;
    this->istrained = true;

    return da_status_success;
}

// Given a vector D of length n and an integer k, this function returns in the first k positions
// of a vector k_dist, the k smaller values of D (unordered) and in the first k positions of a vector
// k_ind, the corresponding indices of the original vector D, where initial indices are
// init_index, init_index+1, ...
template <typename T>
inline void smaller_values_and_indices(da_int n, T *D, da_int k, da_int *k_ind, T *k_dist,
                                       da_int init_index, bool init = true) {
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

// Given a vector k_ind of length n, and a vector that holds the corresponding indices, return into two arrays n_dist and n_ind
// the sorted distances of D and the sorted indices, respectively.
template <typename T>
inline void sorted_n_dist_n_ind(da_int n, T *k_dist, da_int *k_ind, T *n_dist,
                                da_int *n_ind, da_int *perm_vector, bool return_distance,
                                bool get_squares) {
    // We sort with respect to partial distances and then we use the sorted array to reorder the array of indices.
    da_std::iota(perm_vector, perm_vector + n, 0);

    std::stable_sort(perm_vector, perm_vector + n,
                     [&](da_int i, da_int j) { return k_dist[i] < k_dist[j]; });

    for (da_int i = 0; i < n; i++)
        n_ind[i] = k_ind[perm_vector[i]];

    if (return_distance) {
        if (get_squares) {
            for (da_int i = 0; i < n; i++)
                n_dist[i] = std::sqrt(k_dist[perm_vector[i]]);
        } else {
            for (da_int i = 0; i < n; i++)
                n_dist[i] = k_dist[perm_vector[i]];
        }
    }
}

template <typename T>
template <da_int XTRAIN_BLOCK>
da_status knn<T>::kneighbors_kernel(da_int xtrain_block_size, da_int n_blocks_train,
                                    da_int block_rem_train, da_int n_queries,
                                    da_int n_features, const T *X_test, da_int ldx_test,
                                    T *D, da_int *n_ind, T *n_dist, da_int n_neigh,
                                    bool return_distance) {
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
        smaller_values_and_indices(this->n_samples, D + k * this->n_samples, n_neigh,
                                   k_ind.data() + k * n_neigh,
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
da_status knn<T>::kneighbors_blocked_Xtest(da_int n_queries, da_int n_features,
                                           const T *X_test, da_int ldx_test,
                                           da_int *n_ind, T *n_dist, da_int n_neigh,
                                           bool return_distance) {
    std::vector<T> D;
    da_int xtest_block_size = std::min(XTEST_BLOCK, n_queries);
    da_int n_blocks_test = 0, block_rem_test = 0;
    da_utils::blocking_scheme(n_queries, xtest_block_size, n_blocks_test, block_rem_test);
    da_int n_threads = da_utils::get_n_threads_loop(std::max(n_blocks_test, (da_int)1));
    try {
        D.resize(this->n_samples * xtest_block_size * n_threads);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    da_int threading_error = 0;
    da_int xtrain_block_size = std::min(XTRAIN_BLOCK, n_samples);
    da_int n_blocks_train = 0, block_rem_train = 0;
    da_utils::blocking_scheme(n_samples, xtrain_block_size, n_blocks_train,
                              block_rem_train);
    da_status private_status = da_status_success;
    // Iterate through the number of blocks
#pragma omp parallel default(none)                                                       \
    shared(xtest_block_size, n_blocks_test, block_rem_test, n_features, X_test,          \
               n_queries, n_ind, n_dist, n_neigh, ldx_test, return_distance, D,          \
               threading_error, xtrain_block_size, n_blocks_train,                       \
               block_rem_train) private(private_status) num_threads(n_threads)
    {
        da_int task_thread = (da_int)omp_get_thread_num();
        da_int D_index = task_thread * this->n_samples * xtest_block_size;
        da_int xtest_subblock = xtest_block_size;
#pragma omp for schedule(dynamic)
        for (da_int jblock = 0; jblock < n_blocks_test; jblock++) {
            if (jblock == n_blocks_test - 1 && block_rem_test > 0)
                xtest_subblock = block_rem_test;
            // If the remaining numbers of queries is ge than the block size
            // the size of the submatrix we are computing is xtest_block_size
            private_status = kneighbors_kernel<XTRAIN_BLOCK>(
                xtrain_block_size, n_blocks_train, block_rem_train, xtest_subblock,
                n_features, X_test + jblock * xtest_block_size, ldx_test, &D[D_index],
                n_ind + jblock * xtest_block_size * n_neigh,
                n_dist + jblock * xtest_block_size * n_neigh, n_neigh, return_distance);
            if (private_status != da_status_success)
#pragma omp atomic write
                threading_error = 1;
        }
    }
    if (threading_error == 1)
        return da_status_memory_error;

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
da_status knn<T>::kneighbors_compute(da_int n_queries, da_int n_features, const T *X_test,
                                     da_int ldx_test, da_int *n_ind, T *n_dist,
                                     da_int n_neigh, bool return_distance) {

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
    };
    bool is_threaded = omp_get_max_threads() != 1;
    if (is_threaded)
        return kneighbors_blocked_Xtest<1, 16>(n_queries, n_features, X_test, ldx_test,
                                               n_ind, n_dist, n_neigh, return_distance);
    else
        return kneighbors_blocked_Xtest<2048, 16>(n_queries, n_features, X_test, ldx_test,
                                                  n_ind, n_dist, n_neigh,
                                                  return_distance);
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
da_status knn<T>::kneighbors(da_int n_queries, da_int n_features, const T *X_test,
                             da_int ldx_test, da_int *n_ind, T *n_dist, da_int n_neigh,
                             bool return_distance) {
    da_status status = da_status_success;

    // Return if there are no training data
    if (!istrained)
        return da_error_bypass(this->err, da_status_no_data,
                               "No data has been passed to the handle. Please call "
                               "da_knn_set_data_s or da_knn_set_data_d.");
    const T *X_test_temp = nullptr;
    T *utility_ptr1 = nullptr;
    da_int ldx_test_temp = ldx_test;

    if (!is_up_to_date)
        status = knn<T>::set_params();
    if (status != da_status_success)
        return da_error_bypass(this->err, status,
                               "Error while setting the parameters in kneighbors().");

    if (n_ind == nullptr) {
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "n_ind is not a valid pointer.");
    }
    // This check is added based on the functionality that will be added in the future.
    // X can be nullptr. Only check parameters related to X, if X is not nullptr.
    if (X_test != nullptr) {
        status = this->store_2D_array(n_queries, n_features, X_test, ldx_test,
                                      &utility_ptr1, &X_test_temp, ldx_test_temp,
                                      "n_queries", "n_features", "X_test", "ldx_test");
        if (status != da_status_success)
            return status;
        // Data matrix X must have the same number of columns as X_train.
        if (n_features != this->n_features) {
            return da_error_bypass(this->err, da_status_invalid_array_dimension,
                                   "n_features = " + std::to_string(n_features) +
                                       " doesn't match the expected value " +
                                       std::to_string(this->n_features) + ".");
        }
    }

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

    status = knn<T>::kneighbors_compute(n_queries, n_features, X_test_temp, ldx_test_temp,
                                        n_ind, n_dist, n_neigh, return_distance);

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
template <typename T>
void get_weights(const std::vector<T> &D, da_int weight_desrc, std::vector<T> &weights) {
    // Potentially avoid a call here by checking for uniformity at a higher level
    if (weight_desrc == da_knn_uniform) {
        return;
    } else { // da_knn_distance
        for (da_int i = 0; i < da_int(D.size()); i++) {
            // If weights=distance is zero then the weight must be one since it's the closest element.
            weights[i] = (weights[i] <= std::numeric_limits<T>::epsilon())
                             ? 1.0
                             : 1.0 / weights[i];
        }
    }
}

template <typename T> da_status knn<T>::available_classes() {
    // Return if there are no training data
    if (!istrained)
        return da_error_bypass(this->err, da_status_no_data,
                               "No data has been passed to the handle. Please call "
                               "da_knn_set_data_s or da_knn_set_data_d.");
    // From the input data y_train, find the available classes.
    try {
        std::vector<da_int> temp_classes(y_train, y_train + this->n_samples);
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
da_status knn<T>::predict_proba(da_int n_queries, da_int n_features, const T *X_test,
                                da_int ldx_test, T *proba) {
    da_status status = da_status_success;
    // Return if there are no training data
    if (!istrained)
        return da_error_bypass(this->err, da_status_no_data,
                               "No data has been passed to the handle. Please call "
                               "da_knn_set_data_s or da_knn_set_data_d.");

    if (!is_up_to_date)
        status = knn<T>::set_params();

    if (!this->classes_computed) {
        // From the input data y_train, find the available classes.
        status = knn<T>::available_classes();
    }

    if (status != da_status_success)
        return da_error_bypass(this->err, status,
                               "Failed to compute probabilities due to an internal error "
                               "of the available classes computation.");

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

    // This check is added based on the functionality that will be added in the future.
    // X can be nullptr. Only check parameters related to X, if X is not nullptr.
    if (X_test != nullptr) {
        if (status != da_status_success)
            return status;
        // Data matrix X must have the same number of columns as X_train.
        if (n_features != this->n_features) {
            return da_error_bypass(this->err, da_status_invalid_array_dimension,
                                   "n_features = " + std::to_string(n_features) +
                                       " doesn't match the expected value " +
                                       std::to_string(this->n_features) + ".");
        }
    }
    // Allocate memory to set neighbors' indices and corresponding distances.
    // If n_ind and n_dist were returned in row order, then we need to transpose them
    try {
        std::vector<da_int> n_ind(n_queries * this->n_neighbors);

        std::vector<T> n_dist; //(n_queries * this->n_neighbors);
        if (this->weights == da_knn_uniform) {
            // Call kneighbors to compute the indices and distances.
            status = kneighbors(n_queries, n_features, X_test, ldx_test, n_ind.data(),
                                nullptr, this->n_neighbors, false);
        } else if (this->weights == da_knn_distance) {
            n_dist.resize(n_queries * this->n_neighbors);
            // Call kneighbors to compute the indices and distances.
            status = kneighbors(n_queries, n_features, X_test, ldx_test, n_ind.data(),
                                n_dist.data(), this->n_neighbors, true);
            if (this->order == row_major)
                da_blas::imatcopy('T', this->n_neighbors, n_queries, 1.0, n_dist.data(),
                                  this->n_neighbors, n_queries);
        }
#if defined AOCLDA_ILP64
        if (this->order == row_major)
            da_blas::imatcopy('T', this->n_neighbors, n_queries, 1.0,
                              reinterpret_cast<double *>(n_ind.data()), this->n_neighbors,
                              n_queries);
#else
        if (this->order == row_major)
            da_blas::imatcopy('T', this->n_neighbors, n_queries, 1.0,
                              reinterpret_cast<float *>(n_ind.data()), this->n_neighbors,
                              n_queries);
#endif
        if (status != da_status_success)
            return da_error_bypass(
                this->err, status,
                "Failed to compute probabilities due to an internal error "
                "of the k-nearest neighbors computation.");
        // Compute the predicted labels.
        // Depending on the indices of the neighbors, for each test data point return the
        // label of each of the neighbors.

        std::vector<da_int> pred_labels(n_queries * this->n_neighbors);

        for (da_int j = 0; j < this->n_neighbors; j++)
            for (da_int i = 0; i < n_queries; i++)
                pred_labels[i + j * n_queries] = y_train[n_ind[i + j * n_queries]];

        if (this->weights == da_knn_uniform) {
            T denominator;
            // Now that we computed the predicted labels for each neighbor,
            // we use this info to compute the probability for each of the class labels.
            for (da_int i = 0; i < n_queries; i++) {
                denominator = 0.0;
                for (da_int j = 0; j < (da_int)this->classes.size(); j++) {
                    proba[i + j * n_queries] = 0.0;
                    for (da_int neig = 0; neig < this->n_neighbors; neig++)
                        if (classes[j] == pred_labels[i + neig * n_queries])
                            proba[i + j * n_queries]++;
                    denominator += proba[i + j * n_queries];
                }
                for (da_int j = 0; j < (da_int)this->classes.size(); j++)
                    proba[i + j * n_queries] = proba[i + j * n_queries] / denominator;
            }
        } else if (this->weights == da_knn_distance) {
            // Compute the most common value of y_test between the neighbors of each element of X_test.
            // Distance matrix of neighbors has dimensionality of n_queries-by-n_neighbors, so the weight
            // vector should be the same.
            std::vector<T> weight_vector(n_dist);
            get_weights(n_dist, this->weights, weight_vector);
            T denominator;
            for (da_int i = 0; i < n_queries; i++) {
                denominator = 0.0;
                for (da_int j = 0; j < (da_int)this->classes.size(); j++) {
                    proba[i + j * n_queries] = 0.0;
                    for (da_int neig = 0; neig < this->n_neighbors; neig++)
                        if (classes[j] == pred_labels[i + neig * n_queries])
                            proba[i + j * n_queries] +=
                                weight_vector[i + neig * n_queries];
                    denominator += proba[i + j * n_queries];
                }
                for (da_int j = 0; j < (da_int)this->classes.size(); j++)
                    proba[i + j * n_queries] = proba[i + j * n_queries] / denominator;
            }
        }

    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    if (this->order == row_major) {

        da_blas::imatcopy('T', n_queries, n_classes, 1.0, proba, n_queries, n_classes);
    }

    return status;
}

/*
 * Predict the labels y_test for the provided test data.
 * Compute the probabilities for the labels and return the corresponding label according to that.
 */
template <typename T>
da_status knn<T>::predict(da_int n_queries, da_int n_features, const T *X_test,
                          da_int ldx_test, da_int *y_test) {

    da_status status = da_status_success;
    if (!is_up_to_date)
        status = knn<T>::set_params();

    // Return if there are no training data
    if (!istrained)
        return da_error_bypass(this->err, da_status_no_data,
                               "No data has been passed to the handle. Please call "
                               "da_knn_set_data_s or da_knn_set_data_d.");

    if (!this->classes_computed) {
        // From the input data y_train, find the available classes.
        status = knn<T>::available_classes();
    }
    if (status != da_status_success)
        return da_error_bypass(this->err, status,
                               "Failed to compute probabilities due to an internal error "
                               "of the available classes computation.");
    if (y_test == nullptr)
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "y_test is not a valid pointer.");

    // Only test n_queries before memory allocation since the rest will be tested
    // in predict_proba.
    if (n_queries < 1) {
        return da_error_bypass(this->err, da_status_invalid_array_dimension,
                               "n_queries must be greater than 0.");
    }

    try {
        std::vector<T> proba(n_queries * this->n_classes);

        status =
            knn<T>::predict_proba(n_queries, n_features, X_test, ldx_test, proba.data());
        if (status != da_status_success)
            return da_error_bypass(
                this->err, status,
                "Failed to compute predicted labels due to an internal "
                "error of predicting the probabilities.");

        // For each row in n_queries, check which label appears the most times.
        // In case of a tie, return the first label.
        da_int max_index;
        if (this->order == row_major)
            da_blas::imatcopy('T', this->n_classes, n_queries, 1.0, proba.data(),
                              this->n_classes, n_queries);
        for (da_int i = 0; i < n_queries; i++) {
            max_index =
                da_blas::cblas_iamax(this->n_classes, proba.data() + i, n_queries);
            y_test[i] = this->classes[max_index];
        }
        if (this->order == row_major)
            da_blas::imatcopy('T', n_queries, this->n_classes, 1.0, proba.data(),
                              n_queries, this->n_classes);

    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    return status;
}

// Implementing refresh
template <typename T> void knn<T>::refresh() { is_up_to_date = false; }

template class knn<double>;
template class knn<float>;

} // namespace da_knn

} // namespace ARCH
