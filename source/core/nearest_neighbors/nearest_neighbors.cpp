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
#include "da_simd_math.hpp"
#include "kt.hpp"
#include "macros.h"
#include "model_persistence.hpp"
#include "nearest_neighbors_options.hpp"
#include "nearest_neighbors_utils.hpp"
#include "pairwise_distances.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <set>

namespace ARCH {

namespace da_neighbors {

using namespace da_model_persistence;

#define KNN_BLOCK_FLOAT 512
#define KNN_BLOCK_DOUBLE 256

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

    // Validate the 2D array (dimensions, null, NaN checks)
    da_status status =
        self->check_2D_array(self->order, n_queries, n_features, X_test, ldx_test,
                             "n_queries", "n_features", "X_test", "ldx_test");
    if (status != da_status_success)
        return status;

    // Handle storage: column-major points directly, row-major needs transpose
    if (self->order == column_major) {
        *X_test_temp = X_test;
        ldx_test_temp = ldx_test;
    } else {
        try {
            *utility_ptr1 = new T[n_queries * n_features];
        } catch (std::bad_alloc const &) {
            return da_error_bypass(err, da_status_memory_error,
                                   "Memory allocation failed.");
        }
        ARCH::da_utils::copy_transpose_2D_array_row_to_column_major(
            n_queries, n_features, X_test, ldx_test, *utility_ptr1, n_queries);
        *const_cast<T **>(X_test_temp) = *utility_ptr1;
        ldx_test_temp = n_queries;
    }
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

    if (!this->model_trained) {
        return da_warn(this->err, da_status_no_data,
                       "Radius neighbors have not been computed. Please call "
                       "da_nn_radius_neighbors_s or da_nn_radius_neighbors_d first.");
    }
    switch (query) {
    case da_result::da_nn_radius_neighbors_distances_index: {
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
    // check to see if user needs common stuff from the basic handle first
    da_status status = this->get_result_common(query, dim, result);
    if (status != da_status_unknown_query) {
        return status; // either got requested info or error
    }
    if (!this->model_trained) {
        return da_warn(this->err, da_status_no_data,
                       "Radius neighbors have not been computed. Please call "
                       "da_nn_radius_neighbors first.");
    }
    da_int n_count = *dim;
    switch (query) {
    case da_result::da_nn_radius_neighbors_count: {

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
    opt_pass &= this->opts.get("outlier handling", opt_val, outlier_handling) ==
                da_status_success;
    opt_pass &= this->opts.get("outlier label", manual_label) == da_status_success;
    opt_pass &= this->opts.get("outlier target", manual_target) == da_status_success;
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
    opt_pass &= this->opts.get("outlier handling", opt_val, outlier_handling) ==
                da_status_success;
    if (outlier_handling == da_neighbors_types::nn_outlier_handling::manual) {
        da_int local_manual_label;
        T local_manual_target;
        opt_pass &=
            this->opts.get("outlier label", local_manual_label) == da_status_success;
        opt_pass &=
            this->opts.get("outlier target", local_manual_target) == da_status_success;
        // In case the manual outlier values change, update the booleans so that they get recomputed.
        if (local_manual_label != this->manual_label) {
            this->manual_label = local_manual_label;
            manual_outlier_label_checked = false;
        }
        if (local_manual_target != this->manual_target) {
            this->manual_target = local_manual_target;
        }
    }
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

using namespace kernel_templates;

// Inline max-finding. Distances are non-negative so absolute value is not needed.
template <typename T> inline da_int inline_iamax(da_int n, const T *x) {
    // set index and value to first element
    da_int idx = 0;
    T maxval = x[0];
    for (da_int i = 1; i < n; i++) {
        if (x[i] > maxval) {
            maxval = x[i];
            idx = i;
        }
    }
    return idx;
}

template <bsz BSZ, typename T>
inline auto compare_less_equal_mask(avxvector_t<BSZ, T> a, avxvector_t<BSZ, T> b) {
#ifdef __AVX512F__
    if constexpr (BSZ == kernel_templates::bsz::b512) {
        if constexpr (std::is_same_v<T, float>) {
            return _mm512_cmp_ps_mask(a, b, _CMP_LE_OS);
        } else if constexpr (std::is_same_v<T, double>) {
            return _mm512_cmp_pd_mask(a, b, _CMP_LE_OS);
        }
    } else
#endif
        if constexpr (BSZ == kernel_templates::bsz::b256) {
        if constexpr (std::is_same_v<T, float>) {
            auto cmp = _mm256_cmp_ps(a, b, _CMP_LE_OS);
            return _mm256_movemask_ps(cmp);
        } else if constexpr (std::is_same_v<T, double>) {
            auto cmp = _mm256_cmp_pd(a, b, _CMP_LE_OS);
            return _mm256_movemask_pd(cmp);
        }
    } else {
        static_assert(BSZ == kernel_templates::bsz::b256 ||
                          BSZ == kernel_templates::bsz::b512,
                      "Unsupported bit size");
    }
}

// Overload for incremental updates: scans D[0..n-1] against existing top-k in
// k_ind/k_dist, using global_offset for stored indices. k_ind and k_dist must
// already be fully populated with n_neigh candidates.
template <bsz BSZ, typename T>
inline __attribute__((always_inline)) void smaller_values_and_indices_vectorized_kernel(
    da_int n, const T *D, da_int k, da_int *k_ind, T *k_dist, da_int global_offset) {
    constexpr da_int VSIZE = da_int(tsz_v<BSZ, T>);
    // Find the max val of the first k values and its index.
    da_int max_index = inline_iamax(k, k_dist);
    T max_val = k_dist[max_index];
    // Starting from k, since we currently assume the first k values of D are the smaller values,
    // and the corresponding indices have been initialized in k_ind.
    da_int i = 0;
    auto k_dist_max = kt_set1_p<BSZ, T>(max_val);
    // For the rest of the values in D, we need to compare repeatedly with k_dist_vec
    // and replace the values that are smaller.
    for (; i + VSIZE <= n; i += VSIZE) {
        // Load the next VSIZE values of D
        auto k_dist_vec = kt_loadu_p<BSZ, T>(D + i);
        // Compare each lane and set the mask bit if D[i+lane] < max_val
        auto mask = compare_less_equal_mask<BSZ, T>(k_dist_vec, k_dist_max);
        // If no lane beats the max value, continue to the next iteration.
        if (mask == 0)
            continue;
        // If there is at least one lane that beats the max value, we need to find it and replace it
        // in k_ind and k_dist.
        // We could have multiple lanes so we need to iterate through them
        while (mask) {
            // __builtin_ctz returns the index of the lowest set bit,
            // i.e. the position of the next candidate within the chunk.
            da_int lane = __builtin_ctz(mask);
            // Now we go back to scalar code to do the replacements.
            T dist_candidate = D[i + lane];
            // Need to check again in case a previous iteration in this while-loop
            // has already replaced the max value.
            if (dist_candidate <= max_val) {
                // Replace the max value with the new smaller value and update the index.
                k_dist[max_index] = dist_candidate;
                k_ind[max_index] = i + lane + global_offset;
                // Find the new max value and index.
                max_index = inline_iamax(k, k_dist);
                max_val = k_dist[max_index];
            }
            // Unset the lowest set bit to find the next candidate in this chunk.
            mask = mask & (mask - 1);
        }
        // Update the broadcast vector with the new max value for the next comparisons.
        k_dist_max = kt_set1_p<BSZ, T>(max_val);
    }
    // In case we have a leftover chunk that is smaller than VSIZE, we need to do a scalar loop
    for (; i < n; i++) {
        if (D[i] <= max_val) {
            k_ind[max_index] = i + global_offset;
            k_dist[max_index] = D[i];
            max_index = inline_iamax(k, k_dist);
            max_val = k_dist[max_index];
        }
    }
}

// Overload for incremental updates with global index offset.
// k_ind and k_dist must already be fully populated with k candidates.
template <typename T>
void smaller_values_and_indices_vectorized(da_int n, const T *D, da_int k, da_int *k_ind,
                                           T *k_dist, da_int global_offset) {
#ifdef __AVX512F__
    smaller_values_and_indices_vectorized_kernel<bsz::b512, T>(n, D, k, k_ind, k_dist,
                                                               global_offset);
#elif defined(__AVX2__)
    smaller_values_and_indices_vectorized_kernel<bsz::b256, T>(n, D, k, k_ind, k_dist,
                                                               global_offset);
#else
    static_assert(false,
                  "smaller_values_and_indices_vectorized requires AVX2 or AVX512F");
#endif
}

// Compute kernel for brute force algorithm
template <typename T>
da_status neighbors<T>::kneighbors_compute_brute_force(da_int n_queries,
                                                       da_int n_features, const T *X_test,
                                                       da_int ldx_test, da_int *n_ind,
                                                       T *n_dist, da_int n_neigh,
                                                       bool return_distance) {
    // Block sizes matching knn_improvements for benchmarking
    da_int xtrain_block_max, xtest_block_max;
    if constexpr (std::is_same_v<T, float>) {
        xtrain_block_max = KNN_BLOCK_FLOAT;
        xtest_block_max = 1024;
    } else {
        xtrain_block_max = KNN_BLOCK_DOUBLE;
        xtest_block_max = 1024;
    }

    // 2D blocking scheme
    da_int xtest_block_size = std::min(xtest_block_max, n_queries);
    da_int xtest_n_blocks = 0, xtest_block_rem = 0;
    da_utils::blocking_scheme(n_queries, xtest_block_size, xtest_n_blocks,
                              xtest_block_rem);

    da_int xtrain_block_size = std::min(xtrain_block_max, n_samples);
    da_int xtrain_n_blocks = 0, xtrain_block_rem = 0;
    da_utils::blocking_scheme(n_samples, xtrain_block_size, xtrain_n_blocks,
                              xtrain_block_rem);

    // Thread count based on total number of 2D block pairs
    da_int n_threads = da_utils::get_n_threads_loop(xtest_n_blocks * xtrain_n_blocks);
    da_int ldd = xtrain_block_size;
    da_int threading_error = 0;

    // Per-thread storage for D matrices, k-nearest indices/distances, and counts
    std::vector<std::vector<T>> thread_D;
    std::vector<std::vector<da_int>> thread_k_ind;
    std::vector<std::vector<T>> thread_k_dist;
    std::vector<std::vector<da_int>> thread_query_count;
    try {
        thread_D.resize(n_threads);
        thread_k_ind.resize(n_threads);
        thread_k_dist.resize(n_threads);
        thread_query_count.resize(n_threads);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, "Memory allocation failed.");
    }

#pragma omp parallel num_threads(n_threads) default(none) shared(                        \
        threading_error, xtrain_block_size, xtrain_block_rem, xtrain_n_blocks,           \
            xtest_block_size, xtest_block_rem, xtest_n_blocks, n_samples, n_queries,     \
            ldd, n_features, X_test, ldx_test, n_ind, n_dist, n_neigh, return_distance,  \
            n_threads, thread_D, thread_k_ind, thread_k_dist, thread_query_count)
    {
        da_int this_thread = omp_get_thread_num();
        da_int local_error = 0;
        auto &this_D = thread_D[this_thread];

        try {
            this_D.resize(xtrain_block_size * xtest_block_size);
            thread_k_ind[this_thread].resize(n_queries * n_neigh);
            thread_k_dist[this_thread].resize(n_queries * n_neigh);
            thread_query_count[this_thread].resize(n_queries, 0);
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
                    da_int local_xtest_size = xtest_block_size;
                    if (j == xtest_n_blocks - 1 && xtest_block_rem > 0)
                        local_xtest_size = xtest_block_rem;
                    da_int local_xtrain_size = xtrain_block_size;
                    if (i == xtrain_n_blocks - 1 && xtrain_block_rem > 0)
                        local_xtrain_size = xtrain_block_rem;

                    // Compute pairwise distances for this block pair
                    da_status thd_status =
                        da_metrics::pairwise_distances::pairwise_distance_kernel(
                            column_major, local_xtrain_size, local_xtest_size, n_features,
                            X_train + i * xtrain_block_size, ldx_train,
                            X_test + j * xtest_block_size, ldx_test, this_D.data(), ldd,
                            this->p, this->internal_metric);
                    if (thd_status != da_status_success) {
#pragma omp atomic write
                        threading_error = 1;
                    }

                    // Clamp small negative values to zero for squared Euclidean GEMM
                    // distances. The GEMM-based computation (||x||^2 - 2<x,y> + ||y||^2)
                    // can produce small negatives from floating-point cancellation;
                    // these would cause NaN after sqrt.
                    if (this->internal_metric == da_sqeuclidean_gemm) {
                        da_simd_math::clamp_nonneg_matrix(
                            local_xtrain_size, local_xtest_size, this_D.data(), ldd);
                    }

                    // For each query in this X_test block, update the thread-local k-nearest
                    da_int *my_k_ind = thread_k_ind[this_thread].data();
                    T *my_k_dist = thread_k_dist[this_thread].data();

                    for (da_int jj = 0; jj < local_xtest_size; jj++) {
                        da_int j_local = jj + j * xtest_block_size;
                        da_int *query_k_ind = my_k_ind + j_local * n_neigh;
                        T *query_k_dist = my_k_dist + j_local * n_neigh;
                        da_int &count = thread_query_count[this_thread][j_local];

                        da_int ii = 0;
                        // Phase 1: fill up initial k candidates
                        for (; ii < local_xtrain_size && count < n_neigh; ii++) {
                            query_k_ind[count] = ii + i * xtrain_block_size;
                            query_k_dist[count] = this_D[ii + jj * ldd];
                            count++;
                        }
                        // Phase 2: update k-nearest with remaining distances
                        if (ii < local_xtrain_size) {
                            smaller_values_and_indices_vectorized(
                                local_xtrain_size - ii, this_D.data() + ii + jj * ldd,
                                n_neigh, query_k_ind, query_k_dist,
                                ii + i * xtrain_block_size);
                        }
                    } // End of per-query update
                }     // End of error check
            }         // End of xtrain blocks
        }             // End of xtest blocks

        this_D = std::vector<T>{};

        if (n_threads == 1) {
            // Single-thread fast path: no merge needed, sort directly from thread 0
#pragma omp atomic read
            local_error = threading_error;
            if (local_error == 0) {
                std::vector<da_int> perm_vector;
                try {
                    perm_vector.resize(n_neigh);
                } catch (std::bad_alloc const &) {
#pragma omp atomic write
                    threading_error = 1;
                }
                for (da_int q = 0; q < n_queries; q++) {
                    sorted_n_dist_n_ind(n_neigh, thread_k_dist[0].data() + q * n_neigh,
                                        thread_k_ind[0].data() + q * n_neigh,
                                        return_distance ? n_dist + q * n_neigh : nullptr,
                                        n_ind + q * n_neigh, perm_vector.data(),
                                        return_distance, get_squares);
                }
            }
        } else {
            // Multi-thread path: barrier, merge thread-local results, then sort
#pragma omp barrier

#pragma omp atomic read
            local_error = threading_error;
            if (local_error == 0) {
                std::vector<da_int> perm_vector;
                try {
                    perm_vector.resize(n_neigh);
                } catch (std::bad_alloc const &) {
#pragma omp atomic write
                    threading_error = 1;
                }

                // Merge thread-local k-nearest into thread 0's arrays, then sort
#pragma omp for schedule(guided)
                for (da_int q = 0; q < n_queries; q++) {
                    da_int *q_k_ind = thread_k_ind[0].data() + q * n_neigh;
                    T *q_k_dist = thread_k_dist[0].data() + q * n_neigh;
                    da_int count = thread_query_count[0][q];

                    for (da_int t = 1; t < n_threads; t++) {
                        da_int other_count = thread_query_count[t][q];
                        if (other_count == 0)
                            continue;

                        da_int *other_k_ind = thread_k_ind[t].data() + q * n_neigh;
                        T *other_k_dist = thread_k_dist[t].data() + q * n_neigh;

                        da_int oi = 0;
                        // Fill phase: if thread 0 doesn't have n_neigh candidates yet
                        for (; oi < other_count && count < n_neigh; oi++) {
                            q_k_ind[count] = other_k_ind[oi];
                            q_k_dist[count] = other_k_dist[oi];
                            count++;
                        }
                        // Update phase: merge remaining candidates into top-k
                        if (oi < other_count && count >= n_neigh) {
                            da_int max_idx = inline_iamax(n_neigh, q_k_dist);
                            T max_val = q_k_dist[max_idx];
                            for (; oi < other_count; oi++) {
                                if (other_k_dist[oi] <= max_val) {
                                    q_k_ind[max_idx] = other_k_ind[oi];
                                    q_k_dist[max_idx] = other_k_dist[oi];
                                    max_idx = inline_iamax(n_neigh, q_k_dist);
                                    max_val = q_k_dist[max_idx];
                                }
                            }
                        }
                    }

                    // Sort and copy to output
                    sorted_n_dist_n_ind(n_neigh, q_k_dist, q_k_ind,
                                        return_distance ? n_dist + q * n_neigh : nullptr,
                                        n_ind + q * n_neigh, perm_vector.data(),
                                        return_distance, get_squares);
                }
            }
        }

        // Barrier ensures all threads finish merging before any thread frees its buffers.
        // Without this, a thread that finishes early could free thread_k_ind[t] /
        // thread_k_dist[t] while another thread is still reading them during the merge.
#pragma omp barrier

        thread_k_ind[this_thread] = std::vector<da_int>{};
        thread_k_dist[this_thread] = std::vector<T>{};
        thread_query_count[this_thread] = std::vector<da_int>{};
    } // End of parallel region

    if (threading_error != 0)
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");

    return da_status_success;
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

// Given a vector x of length n, this function returns
// the most frequent element in x. In case of ties, it returns the smallest one.
inline __attribute__((__always_inline__)) da_int most_frequent_element(da_int n,
                                                                       const da_int *x) {
    // Insert all elements of x into the map and count their frequencies.
    // Using std::map for deterministic iteration order.
    std::map<da_int, da_int> freq_map;
    for (da_int i = 0; i < n; i++) {
        freq_map[x[i]]++;
    }
    // Find the maximum frequency and the corresponding element(s).
    da_int max_freq = 0;
    da_int most_freq_element = x[0];
    for (const auto &freq : freq_map) {
        da_int val = freq.first;
        da_int count = freq.second;
        // Update if frequency is higher, or if equal frequency but smaller value (tie-breaking)
        if (count > max_freq || (count == max_freq && val < most_freq_element)) {
            max_freq = count;
            most_freq_element = val;
        }
    }
    return most_freq_element;
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
        // std::set will automatically sort and remove duplicates.
        std::set<da_int> temp_classes_set(this->y_train_class,
                                          this->y_train_class + this->n_samples);
        std::vector<da_int> temp_classes(temp_classes_set.begin(),
                                         temp_classes_set.end());
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
    if ((this->outlier_handling ==
         da_neighbors_types::nn_outlier_handling::most_frequent) &&
        (!this->most_frequent_label_computed)) {
        // Compute the most frequent label if needed
        this->most_frequent_label =
            most_frequent_element(this->n_samples, this->y_train_class);
        this->most_frequent_label_computed = true;
    }

    if ((this->outlier_handling == da_neighbors_types::nn_outlier_handling::manual) &&
        (!this->manual_outlier_label_checked)) {
        // Check if the manually set outlier label is present in the training labels.
        auto it =
            std::find(this->classes.begin(), this->classes.end(), this->manual_label);
        this->manual_label_index =
            (it != this->classes.end()) ? std::distance(this->classes.begin(), it) : -1;
        this->manual_outlier_label_checked = true;
    }

    std::vector<da_int> temp_radius_neighbors_count;
    std::vector<da_vector::da_vector<da_int>> temp_radius_neighbors_indices;
    std::vector<da_vector::da_vector<T>> temp_radius_neighbors_distances;

    bool has_outliers = false;
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
                da_int j_local = j * num_classes;
                if (n_neigh == 0) {
                    if (this->outlier_handling ==
                        da_neighbors_types::nn_outlier_handling::none) {
                        return da_error_bypass(
                            this->err, da_status_operation_failed,
                            "Failed to compute probabilities for query " +
                                std::to_string(j) +
                                " since it does not have any neighbors "
                                "within the specified radius.");
                    } else if (this->outlier_handling ==
                               da_neighbors_types::nn_outlier_handling::manual) {
                        has_outliers = true;
                        // Fill probabilities with 0.0 for samples with no neighbors
                        da_std::fill(proba + j_local, proba + j_local + num_classes, 0.0);
                        // If the manually set outlier label is present in the training labels, set its probability to 1.0
                        if (this->manual_label_index != -1) {
                            proba[classes[this->manual_label_index] + j_local] = 1.0;
                        }
                    } else if (this->outlier_handling ==
                               da_neighbors_types::nn_outlier_handling::most_frequent) {
                        // Fill probabilities with 1.0 for most frequent class, 0.0 for others
                        for (da_int i = 0; i < num_classes; i++) {
                            proba[i + j_local] =
                                (this->classes[i] == this->most_frequent_label) ? 1.0
                                                                                : 0.0;
                        }
                    }
                    continue;
                }
                denominator = 0.0;
                // Count neighbors for each class
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
                da_int j_local = j * num_classes;
                if (n_neigh == 0) {
                    if (this->outlier_handling ==
                        da_neighbors_types::nn_outlier_handling::none) {
                        return da_error_bypass(
                            this->err, da_status_operation_failed,
                            "Failed to compute probabilities for query " +
                                std::to_string(j) +
                                " since it does not have any neighbors "
                                "within the specified radius.");
                    } else if (this->outlier_handling ==
                               da_neighbors_types::nn_outlier_handling::manual) {
                        has_outliers = true;
                        // Fill probabilities with 0.0 for samples with no neighbors
                        da_std::fill(proba + j_local, proba + j_local + num_classes, 0.0);
                        // If the manually set outlier label is present in the training labels, set its probability to 1.0
                        if (this->manual_label_index != -1) {
                            proba[classes[this->manual_label_index] + j_local] = 1.0;
                        }
                    } else if (this->outlier_handling ==
                               da_neighbors_types::nn_outlier_handling::most_frequent) {
                        // Fill probabilities with 1.0 for most frequent class, 0.0 for others
                        for (da_int i = 0; i < num_classes; i++) {
                            proba[i + j_local] =
                                (this->classes[i] == this->most_frequent_label) ? 1.0
                                                                                : 0.0;
                        }
                    }
                    continue;
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

    if (this->outlier_handling == da_neighbors_types::nn_outlier_handling::manual) {
        if (has_outliers && this->manual_label_index == -1) {
            return da_warn(
                this->err, da_status_outlier_warning,
                "The manually set outlier label is not present in the training labels. "
                "Setting probabilities to 0.0.");
        }
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
    if (status != da_status_success && status != da_status_outlier_warning)
        return da_error_bypass(this->err, status,
                               "Failed to compute predicted labels due to an internal "
                               "error of predicting the probabilities.");

    // For each column of proba, check which label appears the most times.
    // In case of a tie, return the first label.
    da_int max_index;
    if (search_mode == knn_search_mode ||
        this->outlier_handling == da_neighbors_types::nn_outlier_handling::none) {
        for (da_int i = 0; i < n_queries; i++) {
            max_index = da_blas::cblas_iamax(this->n_classes,
                                             proba.data() + i * this->n_classes, 1);
            y_test[i] = this->classes[max_index];
        }
    } else if (this->outlier_handling ==
               da_neighbors_types::nn_outlier_handling::manual) {
        for (da_int i = 0; i < n_queries; i++) {
            max_index = da_blas::cblas_iamax(this->n_classes,
                                             proba.data() + i * this->n_classes, 1);
            if (proba[max_index + i * this->n_classes] == 0.0) {
                y_test[i] = this->manual_label;
            } else {
                y_test[i] = this->classes[max_index];
            }
        }
    } else if (this->outlier_handling ==
               da_neighbors_types::nn_outlier_handling::most_frequent) {
        for (da_int i = 0; i < n_queries; i++) {
            max_index = da_blas::cblas_iamax(this->n_classes,
                                             proba.data() + i * this->n_classes, 1);
            if (proba[max_index + i * this->n_classes] == 0.0) {
                y_test[i] = this->most_frequent_label;
            } else {
                y_test[i] = this->classes[max_index];
            }
        }
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

    if (this->outlier_handling ==
            da_neighbors_types::nn_outlier_handling::most_frequent &&
        !this->mean_target_computed) {
        status = da_basic_statistics::mean(column_major, da_axis_col, this->n_samples, 1,
                                           this->y_train_reg, this->n_samples,
                                           &this->mean_target);
        if (status != da_status_success)
            return da_error_bypass(
                this->err, status,
                "Failed to compute mean target due to an internal error "
                "of the mean computation.");
        this->mean_target_computed = true;
    }

    std::vector<da_int> temp_radius_neighbors_count;
    std::vector<da_vector::da_vector<da_int>> temp_radius_neighbors_indices;
    std::vector<da_vector::da_vector<T>> temp_radius_neighbors_distances;

    bool has_outliers = false;
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
                                   "Failed to predict targets due to an internal "
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
                    if (this->outlier_handling ==
                        da_neighbors_types::nn_outlier_handling::none) {
                        return da_error_bypass(
                            this->err, da_status_operation_failed,
                            "Failed to compute targets for query " + std::to_string(j) +
                                " since it does not have any neighbors "
                                "within the specified radius.");
                    } else if (this->outlier_handling ==
                               da_neighbors_types::nn_outlier_handling::most_frequent) {
                        y_test[j] = this->mean_target;
                    } else {
                        has_outliers = true;
                        y_test[j] = this->manual_target;
                    }
                    continue;
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
                    if (this->outlier_handling ==
                        da_neighbors_types::nn_outlier_handling::none) {
                        return da_error_bypass(
                            this->err, da_status_operation_failed,
                            "Failed to predict targets for query " + std::to_string(j) +
                                " since it does not have any neighbors "
                                "within the specified radius.");
                    } else if (this->outlier_handling ==
                               da_neighbors_types::nn_outlier_handling::most_frequent) {
                        y_test[j] = this->mean_target;
                    } else {
                        has_outliers = true;
                        y_test[j] = this->manual_target;
                    }
                    continue;
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

    if ((this->outlier_handling == da_neighbors_types::nn_outlier_handling::manual) &&
        has_outliers) {
        return da_warn(
            this->err, da_status_outlier_warning,
            "One or more samples have no neighbors within the specified radius. "
            "Setting the corresponding targets to " +
                std::to_string(this->manual_target) + ".");
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
        if (this->model_trained) {
            this->radius_neighbors_count.clear();
            this->radius_neighbors_indices.clear();
            this->radius_neighbors_distances.clear();
            this->model_trained = false;
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
        this->model_trained = true;
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
    std::vector<std::vector<da_vector::da_vector<da_int>>> neighbors_local_indices;
    std::vector<std::vector<da_vector::da_vector<T>>> neighbors_local_distances;
    try {
        neighbors_local_indices.resize(n_threads);
        neighbors_local_distances.resize(n_threads);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, "Memory allocation failed.");
    }

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

template <typename T> da_status neighbors<T>::serialize(serialization_buffer &buffer) {

    da_status status = da_status_success;
    auto io_dispatch = [&buffer, &status](auto &data) -> void {
        if (status != da_status_success) {
            return;
        }
        status = buffer.dispatch_buffer_io(data);
        return;
    };

    io_dispatch(this->order);
    io_dispatch(this->n_samples);
    io_dispatch(this->n_features);
    io_dispatch(this->istrained_labels);
    io_dispatch(this->istrained_targets);
    io_dispatch(this->is_up_to_date);
    io_dispatch(this->classes_computed);
    io_dispatch(this->classes);
    io_dispatch(this->sort_results);
    io_dispatch(this->model_trained);
    io_dispatch(this->rnn_return_distances);
    io_dispatch(this->n_neighbors);
    io_dispatch(this->algo);
    io_dispatch(this->working_algo);
    io_dispatch(this->metric);
    io_dispatch(this->internal_metric);
    io_dispatch(this->leaf_size);
    io_dispatch(this->get_squares);
    io_dispatch(this->p);
    io_dispatch(this->weights);
    io_dispatch(this->outlier_handling);
    io_dispatch(this->n_classes);
    io_dispatch(this->radius);
    io_dispatch(this->radius_neighbors_count);
    io_dispatch(this->radius_neighbors_indices);
    io_dispatch(this->radius_neighbors_distances);
    io_dispatch(this->istrained_Xtrain);
    io_dispatch(this->most_frequent_label_computed);
    io_dispatch(this->most_frequent_label);
    io_dispatch(this->manual_outlier_label_checked);
    io_dispatch(this->manual_label_index);
    io_dispatch(this->manual_label);
    io_dispatch(this->mean_target_computed);
    io_dispatch(this->mean_target);
    io_dispatch(this->manual_target);

    if (status != da_status_success)
        return status;

    if (buffer.get_mode() != deserialize) {
        // Model always transposes data to use column major
        status = buffer.serialize_user_data(this->X_train, column_major, this->n_samples,
                                            this->n_features, this->ldx_train);
        if (status != da_status_success)
            return status;
        status = buffer.serialize_user_data(this->y_train_class, this->order,
                                            this->n_samples, 1, this->n_samples);
        if (status != da_status_success)
            return status;
        status = buffer.serialize_user_data(this->y_train_reg, this->order,
                                            this->n_samples, 1, this->n_samples);
        if (status != da_status_success)
            return status;

    } else {
        io_dispatch(this->X_int);
        // Set X_train here as it might be needed for tree
        // deserialization below
        this->X_train = this->X_int.data();

        io_dispatch(this->y_train_class_int);
        io_dispatch(this->y_train_reg_int);
        if (status != da_status_success)
            return status;
        // Model always transposes data to use column major
        this->ldx_train = this->n_samples;
    }

    if (this->working_algo == da_neighbors_types::nn_algorithm::kd_tree) {
        if (buffer.get_mode() == deserialize) {
            try {
                this->internal_kd_tree =
                    std::make_unique<ARCH::da_binary_tree::kd_tree<T>>(this->X_train,
                                                                       this->ldx_train);
            } catch (std::bad_alloc const &) {
                return da_error(this->err, da_status_memory_error,
                                "Failing to allocate enough memory."); // LCOV_EXCL_LINE
            }
        }
        status = this->internal_kd_tree->serialize(buffer);
        if (status != da_status_success)
            return status;
    }

    if (this->working_algo == da_neighbors_types::nn_algorithm::ball_tree) {
        if (buffer.get_mode() == deserialize) {
            try {
                this->internal_ball_tree =
                    std::make_unique<ARCH::da_binary_tree::ball_tree<T>>(this->X_train,
                                                                         this->ldx_train);
            } catch (std::bad_alloc const &) {
                return da_error(this->err, da_status_memory_error,
                                "Failing to allocate enough memory."); // LCOV_EXCL_LINE
            }
        }
        status = this->internal_ball_tree->serialize(buffer);
        if (status != da_status_success)
            return status;
    }
    return status;
}

template <typename T> da_status neighbors<T>::save_model(serialization_buffer &buffer) {

    if (!this->istrained_Xtrain) {
        return da_error(this->err, da_status_no_data,
                        "No training data have been set. Please call "
                        "da_nn_set_data_s or da_nn_set_data_d.");
    }

    da_status status = basic_handle<T>::save_model(buffer);
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure serializing model.");

    return status;
}

template <typename T> da_status neighbors<T>::load_model(serialization_buffer &buffer) {
    da_status status = basic_handle<T>::load_model(buffer);
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure deserializing model.");

    if (this->y_train_class_int.size() > 0) {
        this->y_train_class = this->y_train_class_int.data();
    }
    if (this->y_train_reg_int.size() > 0) {
        this->y_train_reg = this->y_train_reg_int.data();
    }

    return status;
}

template class neighbors<double>;
template class neighbors<float>;
} // namespace da_neighbors

} // namespace ARCH
