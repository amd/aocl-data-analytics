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

#include "approximate_neighbors.hpp"
#include "aoclda.h"
#include "aoclda_types.h"
#include "approximate_neighbors_options.hpp"
#include "binary_tree.hpp"
#include "context.hpp"
#include "da_error.hpp"
#include "da_omp.hpp"
#include "da_utils.hpp"
#include "kmeans/kmeans.hpp"
#include "miscellaneous.hpp"
#include "model_persistence.hpp"
#include "nearest_neighbors_utils.hpp"
#include "pairwise_distances.hpp"

#include <algorithm>

namespace ARCH {

namespace da_approx_nn {

using namespace da_model_persistence;

template <typename T>
approximate_neighbors<T>::approximate_neighbors(da_errors::da_error_t &err)
    : basic_handle<T>(err) {
    // Initialize the options registry
    // Any error is stored err->status[.] and this NEEDS to be checked
    // by the caller.
    register_approximate_neighbors_options<T>(this->opts, *this->err);
}

template <typename T> approximate_neighbors<T>::~approximate_neighbors() {}

template <typename T>
da_status approximate_neighbors<T>::get_result(da_result query, da_int *dim, T *result) {
    if (!this->model_trained) {
        return da_warn(this->err, da_status_no_data,
                       "Index has not yet been trained. Please call da_approx_nn_train_d "
                       "or da_approx_nn_train_s before extracting results.");
    }

    da_int rinfo_size = 4;

    switch (query) {
    case da_result::da_rinfo:
        if (*dim < rinfo_size) {
            *dim = rinfo_size;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(rinfo_size) + ".");
        }
        result[0] = static_cast<T>(n_list);
        result[1] = static_cast<T>(n_index);
        result[2] = static_cast<T>(n_features);
        result[3] = static_cast<T>(kmeans_iter);
        break;
    case da_result::da_approx_nn_cluster_centroids:
        if (*dim < n_list * n_features) {
            *dim = n_list * n_features;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(n_list * n_features) + ".");
        }

        if (this->order == column_major) {
            for (da_int i = 0; i < n_list; i++) {
                for (da_int j = 0; j < n_features; j++) {
                    result[i + j * ld_centroids] = centroids[i + j * ld_centroids];
                }
            }
        } else {
            for (da_int j = 0; j < n_features; j++) {
                for (da_int i = 0; i < n_list; i++) {
                    result[j + i * ld_centroids] = centroids[j + i * ld_centroids];
                }
            }
        }
        break;
    default:
        return da_warn(this->err, da_status_unknown_query,
                       "The requested result could not be found.");
    }
    return da_status_success;
}

template <typename T>
da_status approximate_neighbors<T>::get_result(da_result query, da_int *dim,
                                               da_int *result) {
    // check to see if user needs common stuff from the basic handle first
    da_status status = this->get_result_common(query, dim, result);
    if (status != da_status_unknown_query) {
        return status; // either got requested info or error
    }
    if (!this->model_trained) {
        return da_warn(this->err, da_status_no_data,
                       "Index has not yet been trained. Please call da_approx_nn_train_d "
                       "or da_approx_nn_train_s before extracting results.");
    }

    switch (query) {
    case da_result::da_approx_nn_list_sizes:
        if (*dim < n_list) {
            *dim = n_list;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(n_list) + ".");
        }

        for (da_int i = 0; i < this->n_list; i++) {
            result[i] = this->list_sizes[i];
        }
        break;

    default:
        return da_warn(this->err, da_status_unknown_query,
                       "The requested result could not be found.");
    }
    return da_status_success;
}

template <typename T> da_status approximate_neighbors<T>::check_options_update() {
    // To be called before kneighbors computation or add is performed to check for
    // any relevant options updates.
    bool opt_pass = true;
    da_int local_algo, local_metric, local_n_list;
    std::string opt_val;
    // nprobe is free to change between queries
    opt_pass &= this->opts.get("n_probe", n_probe) == da_status_success;
    // n_neighbors is free to change between queries
    opt_pass &= this->opts.get("number of neighbors", n_neighbors) == da_status_success;

    if (!opt_pass)
        return da_error_bypass(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                               "Unexpected error while reading the optional parameters.");

    // Other search related options are not allowed to change
    opt_pass &= this->opts.get("n_list", local_n_list) == da_status_success;
    if (local_n_list != this->n_list) {
        return da_error_bypass(this->err, da_status_option_locked,
                               "n_list cannot be changed after calling train().");
    }
    opt_pass &= this->opts.get("algorithm", opt_val, local_algo) == da_status_success;
    if (local_algo != this->algo) {
        return da_error_bypass(this->err, da_status_option_locked,
                               "algorithm cannot be changed after calling train().");
    }
    opt_pass &= this->opts.get("metric", opt_val, local_metric) == da_status_success;
    if (local_metric != this->metric) {
        return da_error_bypass(this->err, da_status_option_locked,
                               "metric cannot be changed after calling train().");
    }

    return da_status_success;
}

template <typename T>
da_status approximate_neighbors<T>::set_training_data(da_int n_samples, da_int n_features,
                                                      const T *X_train_in,
                                                      da_int ldx_train_in) {
    // Guard against errors due to multiple calls with the same handle
    // Reset any state variables
    this->train_data_is_set = false;
    this->model_trained = false;
    this->data_is_added = false;
    this->n_index = 0;

    bool opt_pass = true;
    std::string opt_val;
    da_int iorder;

    opt_pass &= this->opts.get("storage order", opt_val, iorder) == da_status_success;
    this->order = da_order(iorder);

    if (!opt_pass)
        return da_error_bypass(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                               "Unexpected error while reading parameters.");

    // Check input is okay
    // No additional storage needed as we natively handle row major data
    da_status status =
        this->check_2D_array(this->order, n_samples, n_features, X_train_in, ldx_train_in,
                             "n_samples", "n_features", "X_train", "ldx_train");

    if (status != da_status_success) {
        return status;
    }

    // Set internal pointer to user data
    this->X_train = X_train_in;
    this->ldx_train = ldx_train_in;
    this->n_features = n_features;
    // Number of samples of train data passed by the user
    this->n_samples = n_samples;
    // We may subsample training data - how many samples we actually use
    this->n_samples_train = n_samples;

    train_data_is_set = true;

    return da_status_success;
}

template <typename T> da_status approximate_neighbors<T>::read_training_options() {
    // Read any options relevant to training
    bool opt_pass = true;
    std::string opt_val;
    da_int iorder, imetric;

    // Integer options - don't need n_probe or n_neighbors until search time
    opt_pass &= this->opts.get("n_list", n_list) == da_status_success;
    opt_pass &= this->opts.get("k-means_iter", max_iter) == da_status_success;
    opt_pass &= this->opts.get("seed", seed) == da_status_success;

    // fp options
    opt_pass &= this->opts.get("train fraction", train_fraction) == da_status_success;

    // string options
    opt_pass &= this->opts.get("algorithm", opt_val, algo) == da_status_success;
    this->internal_algo = (this->algo == approx_nn_algorithm::automatic)
                              ? static_cast<da_int>(approx_nn_algorithm::ivfflat)
                              : this->algo;

    opt_pass &= this->opts.get("metric", opt_val, imetric) == da_status_success;
    opt_pass &= this->opts.get("storage order", opt_val, iorder) == da_status_success;

    if (!opt_pass)
        return da_error_bypass(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                               "Unexpected error while reading parameters.");

    if (this->n_list > n_samples) {
        return da_error(
            this->err, da_status_invalid_array_dimension,
            "n_samples = " + std::to_string(n_samples) +
                " must be at least as large as n_list = " + std::to_string(n_list));
    }

    this->order = da_order(iorder);
    this->metric = approx_nn_metric(imetric);
    this->internal_metric = (this->metric == approx_nn_metric::euclidean)
                                ? approx_nn_metric::sqeuclidean
                                : this->metric;

    return da_status_success;
}

template <typename T>
da_status
approximate_neighbors<T>::subsample_training_data(std::vector<T> &X_train_sampled,
                                                  da_int &ldx_train_sampled) {
    ldx_train_sampled =
        (this->order == column_major) ? this->n_samples_train : this->n_features;
    std::vector<da_int> perm, indices;
    // Allocate memory for sampled data
    try {
        perm.resize(this->n_samples_train, 0);
        indices.resize(this->n_samples, 0);
        X_train_sampled.resize(static_cast<size_t>(this->n_samples_train) *
                                   static_cast<size_t>(this->n_features),
                               0.0);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    da_std::iota(indices.begin(), indices.end(), 0);
    // Get indices of rows to get
    da_std::sample(indices.begin(), indices.end(), perm.begin(), this->n_samples_train,
                   this->mt_engine);

    [[maybe_unused]] da_int n_threads = omp_get_max_threads();

    // Grab random subset of train data
    if (this->order == column_major) {
#pragma omp parallel for num_threads(std::min(n_threads, n_features)) default(none)      \
    shared(n_features, n_samples_train, X_train_sampled, ldx_train_sampled, X_train,     \
               ldx_train, perm)
        for (da_int j = 0; j < this->n_features; j++) {
            da_int sampled_col_offset = j * ldx_train_sampled;
            da_int train_col_offset = j * this->ldx_train;
            for (da_int i = 0; i < this->n_samples_train; i++) {
                X_train_sampled[i + sampled_col_offset] =
                    this->X_train[perm[i] + train_col_offset];
            }
        }
    } else if (this->order == row_major) {
        da_int row_idx;
        size_t row_bytes = static_cast<size_t>(this->n_features) * sizeof(T);
#pragma omp parallel for num_threads(std::min(n_threads, n_samples_train)) default(none) \
    shared(n_features, n_samples_train, X_train_sampled, ldx_train_sampled, X_train,     \
               ldx_train, perm, row_bytes) private(row_idx)
        for (da_int i = 0; i < this->n_samples_train; i++) {
            row_idx = perm[i];
            memcpy(X_train_sampled.data() +
                       static_cast<size_t>(i) * static_cast<size_t>(ldx_train_sampled),
                   this->X_train + static_cast<size_t>(row_idx) *
                                       static_cast<size_t>(this->ldx_train),
                   row_bytes);
        }
    }
    return da_status_success;
}

// Kernel to train ivfflat index
template <typename T> da_status approximate_neighbors<T>::train_ivfflat() {
    /*
    Overview:
    1. Potentially subsample training data.
    2. Set up k-means model and perform clustering.
    3. Extract k-means cluster centers to centroids.
    */
    da_status status = da_status_success;

    this->ld_centroids = (this->order == column_major) ? this->n_list : n_features;
    // We can now allocate memory for centroids, list_sizes and global_indices
    // as we have read n_list in read_training_options
    // Note use of assign and not resize as n_list may change between train calls
    try {
        this->list_sizes.assign(n_list, 0);
        this->old_list_sizes.assign(n_list, 0);
        this->centroids.assign(
            static_cast<size_t>(n_list) * static_cast<size_t>(n_features), 0.0);

        // Use resize + clear for nested containers
        this->indexed_vectors.resize(n_list);
        this->list_norms.resize(n_list);
        this->global_indices.resize(n_list);
        for (da_int i = 0; i < n_list; i++) {
            this->indexed_vectors[i].clear();
            this->list_norms[i].clear();
            this->global_indices[i].clear();
        }

    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    const T *train_ptr = this->X_train;
    da_int ld_train = this->ldx_train;

    // Check if we need to subsample or normalize train data
    std::vector<T> X_train_work;
    da_int ldx_train_work;
    bool need_copy = (this->train_fraction < static_cast<T>(1.0)) ||
                     (this->internal_metric == approx_nn_metric::cosine);

    if (need_copy) {
        if (this->train_fraction < static_cast<T>(1.0)) {
            // Calculate n_samples_train
            // Bound below by n_list. We have an earlier check that n_list < n_samples
            this->n_samples_train = std::max(
                static_cast<da_int>(this->n_samples_train * this->train_fraction),
                this->n_list);

            status = this->subsample_training_data(X_train_work, ldx_train_work);
            if (status != da_status_success)
                return status;
        } else {
            // Full copy for normalization
            ldx_train_work =
                (this->order == column_major) ? this->n_samples_train : this->n_features;
            try {
                X_train_work.resize(static_cast<size_t>(this->n_samples_train) *
                                    static_cast<size_t>(this->n_features));
            } catch (std::bad_alloc const &) {
                return da_error(this->err, da_status_memory_error,
                                "Memory allocation failed.");
            }

            if (this->order == column_major) {
                for (da_int j = 0; j < this->n_features; j++) {
                    for (da_int i = 0; i < this->n_samples_train; i++) {
                        X_train_work[i + j * ldx_train_work] =
                            this->X_train[i + j * this->ldx_train];
                    }
                }
            } else {
                size_t row_bytes = static_cast<size_t>(this->n_features) * sizeof(T);
                for (da_int i = 0; i < this->n_samples_train; i++) {
                    memcpy(X_train_work.data() + static_cast<size_t>(i) *
                                                     static_cast<size_t>(ldx_train_work),
                           this->X_train + static_cast<size_t>(i) *
                                               static_cast<size_t>(this->ldx_train),
                           row_bytes);
                }
            }
        }

        // Normalize training data in-place for cosine metric
        if (this->internal_metric == approx_nn_metric::cosine) {
            status = da_utils::normalize_rows_inplace(
                this->order, this->n_samples_train, this->n_features, X_train_work.data(),
                ldx_train_work, static_cast<T *>(nullptr));
            if (status != da_status_success)
                return status;
        }

        train_ptr = X_train_work.data();
        ld_train = ldx_train_work;
    }

    // Create kmeans_model
    // approximate_neighbors is friended in k-means so we can set private member
    // variables
    ARCH::da_kmeans::kmeans kmeans_model = ARCH::da_kmeans::kmeans<T>(*this->err);

    // Set k-means options
    kmeans_model.algorithm = ARCH::da_kmeans::lloyd;
    kmeans_model.init_method = ARCH::da_kmeans::random_samples;
    kmeans_model.n_clusters = this->n_list;
    kmeans_model.n_init = this->n_init;
    kmeans_model.max_iter = this->max_iter;
    kmeans_model.tol = this->kmeans_tol;
    kmeans_model.seed = this->internal_seed;

    // Set the train data for k-means
    kmeans_model.order = this->order;
    kmeans_model.A_usr = train_ptr;
    kmeans_model.lda_usr = ld_train;
    kmeans_model.n_samples = this->n_samples_train;
    kmeans_model.n_features = this->n_features;

    // Set some k-means internal state variables
    kmeans_model.initdone = true;
    kmeans_model.check_options = false;

    // Do we need to do spherical k-means?
    if (this->internal_metric == approx_nn_metric::inner_product ||
        this->internal_metric == approx_nn_metric::cosine) {
        kmeans_model.do_spherical = true;
    }

    // Compute
    status = kmeans_model.compute();
    if ((status != da_status_success) && (status != da_status_maxit))
        return status;

    // Extract centroids from k-means_model
    da_int centroids_size = this->n_list * this->n_features;
    status = kmeans_model.get_result(da_result::da_kmeans_cluster_centres,
                                     &centroids_size, this->centroids.data());
    this->kmeans_iter = kmeans_model.best_n_iter;

    if (status != da_status_success)
        return status;

    if (this->internal_metric == approx_nn_metric::sqeuclidean) {
        try {
            this->centroid_norms.resize(this->n_list);
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }
        da_utils::compute_squared_row_norms(this->order, this->n_list, this->n_features,
                                            this->centroids.data(), this->ld_centroids,
                                            this->centroid_norms.data());
    }

    return da_status_success;
}

template <typename T> da_status approximate_neighbors<T>::train() {
    if (!train_data_is_set) {
        return da_error(
            this->err, da_status_no_data,
            "No data has been passed to the handle. Please call "
            "da_approx_nn_set_training_data_s or da_approx_nn_set_training_data_d.");
    }

    da_status status = this->read_training_options();
    if (status != da_status_success) {
        return status;
    }

    this->internal_seed = this->seed;
    if (internal_seed == -1) {
        std::random_device r;
        this->internal_seed = std::abs(static_cast<da_int>(r()));
    }
    this->mt_engine.seed(this->internal_seed);

    if (this->internal_algo == da_approx_nn_types::approx_nn_algorithm::ivfflat) {
        status = this->train_ivfflat();
    } else {
        return da_error_bypass(this->err, da_status_invalid_input, "Unknown algorithm.");
    }
    if (status != da_status_success)
        return status;

    this->model_trained = true;

    return da_status_success;
}

template <typename T>
da_status approximate_neighbors<T>::add(da_int n_samples_add, da_int n_features,
                                        const T *X_add, da_int ldx_add) {
    // Check we have already trained successfully
    if (!this->model_trained) {
        return da_error(this->err, da_status_no_data,
                        "No index has been trained. Please call "
                        "da_approx_nn_train_s or da_approx_nn_train_d.");
    }

    // Check nothing has changed that isn't allowed to
    da_status status = this->check_options_update();
    if (status != da_status_success)
        return status;

    // Check input is okay
    status = this->check_2D_array(this->order, n_samples_add, n_features, X_add, ldx_add,
                                  "n_samples", "n_features", "X_add", "ldx_add");
    if (status != da_status_success)
        return status;

    if (n_features != this->n_features)
        return da_error(
            this->err, da_status_invalid_input,
            "The function was called with n_features = " + std::to_string(n_features) +
                " but the index has been trained with " +
                std::to_string(this->n_features) + " features.");

    if (this->internal_algo == da_approx_nn_types::approx_nn_algorithm::ivfflat) {
        status = this->add_ivfflat(n_samples_add, n_features, X_add, ldx_add);
    } else {
        return da_error_bypass(this->err, da_status_invalid_input, "Unknown algorithm.");
    }
    if (status != da_status_success)
        return status;

    return da_status_success;
}

// Kernel to add data to a trained ivfflat index
template <typename T>
da_status approximate_neighbors<T>::add_ivfflat(da_int n_samples_add, da_int n_features,
                                                const T *X_add, da_int ldx_add) {
    /*
    Overview:
    1. For cosine metric, normalize X_add upfront
    2. Compute distance from each row of X_add to each centroid
    3. Identify closest centroid for each row.
    4. Iterate over indexed_vectors, adding the appropriate rows of X_add to
    the appropriate list of indexed_vectors.
    */

    // For cosine metric, normalize X_add before computing distances
    std::vector<T> X_add_work;
    const T *X_add_ptr = X_add;
    da_int ldx_add_ptr = ldx_add;

    if (this->internal_metric == approx_nn_metric::cosine) {
        da_int ldx_add_work = (this->order == column_major) ? n_samples_add : n_features;
        try {
            X_add_work.resize(static_cast<size_t>(n_samples_add) *
                              static_cast<size_t>(n_features));
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error,
                            "Memory allocation failed.");
        }
        da_status status = da_utils::normalize_rows(
            this->order, n_samples_add, n_features, X_add, ldx_add, X_add_work.data(),
            ldx_add_work, static_cast<T *>(nullptr));
        if (status != da_status_success)
            return status;
        X_add_ptr = X_add_work.data();
        ldx_add_ptr = ldx_add_work;
    }

    // local_indices - For each centroid this stores indices of rows of X_add to be added
    // nearest_centroid - flat array: nearest centroid index for each point in X_add
    std::vector<da_vector::da_vector<da_int>> local_indices;
    std::vector<da_int> nearest_centroid;

    const da_int n_list = this->n_list;

    // Compute blk_sz so that total distance memory across all threads stays ≤ 64 MB.
    // Each thread allocates an n_list × blk_sz distance buffer independently.
    const da_int add_budget = (64 << 20) / static_cast<da_int>(sizeof(T));
    [[maybe_unused]] da_int n_threads = static_cast<da_int>(omp_get_max_threads());

    const da_int block_sz_ub =
        std::max(static_cast<da_int>(1), static_cast<da_int>(n_samples_add / n_threads));
    da_int blk_sz = std::min(
        block_sz_ub,
        std::max(static_cast<da_int>(1),
                 add_budget / (n_threads * std::max(static_cast<da_int>(1), n_list))));

    // Debug override: allows tests to force specific add block size via da_debug_set
    {
        auto &hidden = context::get_context()->hidden_settings;
        auto it = hidden.find("ivf.add_blk_sz");
        if (it != hidden.end() && !it->second.empty())
            blk_sz = std::min(std::max(static_cast<da_int>(std::stoi(it->second)),
                                       static_cast<da_int>(1)),
                              n_samples_add);
    }

    da_int n_blocks, block_rem;
    da_utils::blocking_scheme(n_samples_add, blk_sz, n_blocks, block_rem);
    n_threads = std::min(n_threads, n_blocks);

    using namespace std::string_literals;
    context_set_hidden_settings("ivf.add_blk_sz_used"s, std::to_string(blk_sz));
    context_set_hidden_settings("ivf.add_n_blocks"s, std::to_string(n_blocks));

    da_int threading_error = 0;
    const da_int ld_distances = n_list;

    try {
        local_indices.resize(n_list);
        nearest_centroid.resize(n_samples_add);

        // If we are adding data to an index which already has data in it, we need
        // to remember old list sizes
        if (data_is_added) {
            this->old_list_sizes = this->list_sizes;
        }
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    da_int avg_list_size = n_samples_add / n_list;
    try {
        for (da_int i = 0; i < n_list; i++) {
            local_indices[i].reserve(avg_list_size);
            this->global_indices[i].reserve(this->global_indices[i].size() +
                                            avg_list_size);
        }
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    // Phase 1: parallel block loop — GEMM + argmin, no sharing between threads.
    // Each thread owns its distance buffer; argmin result stored in nearest_centroid[j]
    // (one slot per point). We formulate the GEMMs so that the centroid distances for
    // each row of X_add are contiguous in the distance buffer, hence the differing order
    // for row-major vs column-major.
#pragma omp parallel default(none) num_threads(n_threads)                                \
    shared(X_add_ptr, n_samples_add, n_features, n_blocks, block_rem, blk_sz,            \
               ldx_add_ptr, ld_distances, n_list, nearest_centroid, threading_error)
    {
        std::vector<T> thread_dists, thread_work1, thread_work2;
        try {
            thread_dists.resize(static_cast<size_t>(n_list) * static_cast<size_t>(blk_sz),
                                0.0);
            thread_work1.resize(n_list, 0.0);
            thread_work2.resize(blk_sz, 0.0);
        } catch (std::bad_alloc const &) {
#pragma omp atomic write
            threading_error = 1;
        }
#pragma omp barrier

        if (!threading_error) {
#pragma omp for schedule(dynamic)
            for (da_int blk = 0; blk < n_blocks; blk++) {
                const da_int blk_start = blk * blk_sz;
                const da_int this_blk_sz =
                    (blk == n_blocks - 1 && block_rem > 0) ? block_rem : blk_sz;

                const T *X_blk = (this->order == column_major)
                                     ? X_add_ptr + blk_start
                                     : X_add_ptr + blk_start * ldx_add_ptr;

                if (this->internal_metric == approx_nn_metric::sqeuclidean) {
                    if (this->order == column_major) {
                        // in column major compute -2 * C * (X_blk)^T + (matrix norms)
                        ARCH::euclidean_gemm_distance(
                            this->order, this->n_list, this_blk_sz, n_features,
                            this->centroids.data(), this->ld_centroids, X_blk,
                            ldx_add_ptr, thread_dists.data(), ld_distances,
                            thread_work1.data(), 2, thread_work2.data(), 2, true, false);
                    } else {
                        // in row major compute -2 * X_blk * C^T + (matrix norms)
                        ARCH::euclidean_gemm_distance(
                            this->order, this_blk_sz, this->n_list, n_features, X_blk,
                            ldx_add_ptr, this->centroids.data(), this->ld_centroids,
                            thread_dists.data(), ld_distances, thread_work2.data(), 2,
                            thread_work1.data(), 2, true, false);
                    }
                } else {
                    // inner product or cosine - gemm with alpha=-1
                    if (this->order == column_major) {
                        // in column major compute C * (X_blk)^T
                        da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                            this->n_list, this_blk_sz, n_features,
                                            static_cast<T>(-1.0), this->centroids.data(),
                                            this->ld_centroids, X_blk, ldx_add_ptr,
                                            static_cast<T>(0.0), thread_dists.data(),
                                            ld_distances);
                    } else {
                        // in row major compute X_blk * C^T
                        da_blas::cblas_gemm(
                            CblasRowMajor, CblasNoTrans, CblasTrans, this_blk_sz,
                            this->n_list, n_features, static_cast<T>(-1.0), X_blk,
                            ldx_add_ptr, this->centroids.data(), this->ld_centroids,
                            static_cast<T>(0.0), thread_dists.data(), ld_distances);
                    }
                }

                // Argmin: write to nearest_centroid[blk_start+i]
                for (da_int i = 0; i < this_blk_sz; i++) {
                    const T *row = &thread_dists[i * n_list];
                    nearest_centroid[blk_start + i] = static_cast<da_int>(
                        std::distance(row, std::min_element(row, row + n_list)));
                }
            }
        }
    } // end parallel region

    if (threading_error)
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed in parallel region.");

    // Phase 2: serial accumulation — O(n_add), no sharing needed
    for (da_int i = 0; i < n_samples_add; i++) {
        const da_int c = nearest_centroid[i];
        local_indices[c].push_back(i);
        this->global_indices[c].push_back(i + n_index);
        this->list_sizes[c]++;
    }

    // Reset n_threads for the next parallel region
    // n_threads is at most n_list
    n_threads = std::min(static_cast<da_int>(omp_get_max_threads()), this->n_list);
    size_t row_bytes = static_cast<size_t>(this->n_features) * sizeof(T);

    bool is_euclidean = (this->internal_metric == approx_nn_metric::sqeuclidean);

    // Resize indexed_vectors (and list_norms for euclidean) to accommodate new data
    // old_list_sizes is initialized to 0 in train_ivfflat(), so the loop below
    // works for both first call (old_size=0) and subsequent calls
    try {
        for (da_int i = 0; i < n_list; i++) {
            this->indexed_vectors[i].resize(static_cast<size_t>(this->list_sizes[i]) *
                                            static_cast<size_t>(n_features));
            if (is_euclidean)
                this->list_norms[i].resize(this->list_sizes[i]);
        }
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    // Read from X_add_ptr and write to indexed_vectors (always stored as row major)
    // Leading dim of each indexed_vector[list_idx] is n_features
    // old_list_sizes[i] is 0 on first call, so we iterate from 0 to list_sizes[i]
#pragma omp parallel for num_threads(n_threads) schedule(dynamic) default(none)          \
    shared(old_list_sizes, list_sizes, indexed_vectors, list_norms, local_indices,       \
               X_add_ptr, n_list, n_features, ldx_add_ptr, row_bytes, is_euclidean)
    for (da_int list_idx = 0; list_idx < n_list; list_idx++) {
        da_int old_size = this->old_list_sizes[list_idx];
        da_int new_size = this->list_sizes[list_idx];
        T *list_ptr = this->indexed_vectors[list_idx].data();
        da_int *indices_to_add = local_indices[list_idx].data();

        if (is_euclidean) {
            // Norm computation done at same time for euclidean metrics
            T *norms_ptr = this->list_norms[list_idx].data();
            if (this->order == column_major) {
                for (da_int i = old_size; i < new_size; i++) {
                    da_int add_row_idx = indices_to_add[i - old_size];
                    da_int row_idx = i * n_features;
                    T norm = 0;
                    for (da_int j = 0; j < n_features; j++) {
                        T val = X_add_ptr[add_row_idx + j * ldx_add_ptr];
                        list_ptr[row_idx + j] = val;
                        norm += val * val;
                    }
                    norms_ptr[i] = norm;
                }
            } else {
                for (da_int i = old_size; i < new_size; i++) {
                    const T *src = X_add_ptr + ldx_add_ptr * indices_to_add[i - old_size];
                    T *dst = list_ptr + i * n_features;
                    T norm = 0;
                    for (da_int j = 0; j < n_features; j++) {
                        dst[j] = src[j];
                        norm += src[j] * src[j];
                    }
                    norms_ptr[i] = norm;
                }
            }
        } else {
            if (this->order == column_major) {
                for (da_int i = old_size; i < new_size; i++) {
                    da_int add_row_idx = indices_to_add[i - old_size];
                    da_int row_idx = i * n_features;
                    for (da_int j = 0; j < n_features; j++) {
                        list_ptr[row_idx + j] = X_add_ptr[add_row_idx + j * ldx_add_ptr];
                    }
                }
            } else {
                for (da_int i = old_size; i < new_size; i++) {
                    da_int row_idx = indices_to_add[i - old_size];
                    memcpy(list_ptr +
                               static_cast<size_t>(i) * static_cast<size_t>(n_features),
                           X_add_ptr + static_cast<size_t>(ldx_add_ptr) *
                                           static_cast<size_t>(row_idx),
                           row_bytes);
                }
            }
        }
    }

    this->n_index += n_samples_add;
    this->data_is_added = true;
    return da_status_success;
}

template <typename T> da_status approximate_neighbors<T>::train_and_add() {
    // Let train and add do the error checking
    // No extra functionality here. It just saves the user passing the same data twice.
    da_status status = train();
    if (status != da_status_success)
        return status;

    status = add(this->n_samples, this->n_features, this->X_train, this->ldx_train);
    if (status != da_status_success)
        return status;

    return da_status_success;
}

template <typename T>
da_status approximate_neighbors<T>::kneighbors(da_int n_queries, da_int n_features,
                                               const T *X_test, da_int ldx_test,
                                               da_int *n_ind, T *n_dist, da_int k_neigh,
                                               bool return_distance) {
    // Make sure the index has been trained
    if (!this->model_trained) {
        return da_error(this->err, da_status_no_data,
                        "No index has been trained. Please call"
                        "da_approx_nn_train_s or da_approx_nn_train_d.");
    }

    // Make sure some data has been added
    if (!this->data_is_added) {
        return da_error(this->err, da_status_no_data,
                        "No data has been added. Please call"
                        "da_approx_nn_add_s or da_approx_nn_add_d");
    }

    // Check nothing has changed that isn't allowed to
    da_status status = this->check_options_update();
    if (status != da_status_success)
        return status;

    // If k_neigh is <= 0, use the default value in n_neighbors.
    if (k_neigh <= 0)
        k_neigh = this->n_neighbors;

    // Number of neighbors must be greater than number of samples added to the index
    if (k_neigh > this->n_index) {
        return da_error_bypass(
            this->err, da_status_invalid_input,
            std::to_string(k_neigh) + " neighbors were requested but only " +
                std::to_string(this->n_index) + " samples have been added to the index");
    }

    if (this->n_probe > this->n_list) {
        return da_error_bypass(
            this->err, da_status_invalid_input,
            "n_probe=" + std::to_string(this->n_probe) +
                " must be no larger than n_list=" + std::to_string(this->n_list));
    }

    // Check pointer for output indices is valid
    if (n_ind == nullptr) {
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "n_ind is not a valid pointer.");
    }
    // If distances are requested, check the pointer for output distances is valid.
    if (return_distance) {
        if (n_dist == nullptr) {
            return da_error_bypass(this->err, da_status_invalid_pointer,
                                   "n_dist is not a valid pointer.");
        }
    }

    // Check data is okay
    status = this->check_2D_array(this->order, n_queries, n_features, X_test, ldx_test,
                                  "n_samples", "n_features", "X_test", "ldx_test");
    if (status != da_status_success)
        return status;

    // Check feature dimension is okay
    if (n_features != this->n_features)
        return da_error(
            this->err, da_status_invalid_input,
            "The function was called with n_features = " + std::to_string(n_features) +
                " but the index has been trained with " +
                std::to_string(this->n_features) + " features.");

    // and compute
    if (this->internal_algo == da_approx_nn_types::approx_nn_algorithm::ivfflat) {
        status =
            this->kneighbors_compute_ivfflat(n_queries, n_features, X_test, ldx_test,
                                             n_ind, n_dist, k_neigh, return_distance);
    } else {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "Unknown algorithm: " + std::to_string(internal_algo) +
                                   ".");
    }
    if (status != da_status_success)
        return status;

    return da_status_success;
}

template <typename T>
void approximate_neighbors<T>::update_heaps_from_list_blk(
    da_int this_list_blk_sz, da_int q_count, const da_int *list_blk_global_idx,
    const T *fine_distances, da_int block_start, const da_int *queries_processed,
    da_binary_tree::MaxHeap<T> *heaps) {
    for (da_int q = 0; q < q_count; q++) {
        auto &heap = heaps[queries_processed[q] - block_start];
        T max_dist = heap.GetMaxDist();
        const T *list_blk_dists = fine_distances + q * this_list_blk_sz;
        for (da_int v = 0; v < this_list_blk_sz; v++) {
            if (list_blk_dists[v] < max_dist) {
                heap.Insert(list_blk_global_idx[v], list_blk_dists[v]);
                max_dist = heap.GetMaxDist();
            }
        }
    }
}

// Basic struture:
//    - Parallel loop over blocks of queries. For each block:
//         - Calculate coarse query-centroid distances.
//         - Loop over lists. For each list:
//             - Gather all queries that probe that list
//             - GEMM based distance calculation
//             - Update per query heaps
//         - Write results from heaps directly to output arrays
template <typename T>
da_status approximate_neighbors<T>::ivfflat_search_query_parallel(
    da_int n_queries, da_int n_features, da_int k_neigh, bool return_distance,
    const T *X_test_ptr, da_int ldx_test, const T *centroids_ptr,
    da_int ld_centroids_local, da_int query_blk_sz, da_int list_blk_sz, da_int n_blocks,
    da_int final_query_blk_sz, [[maybe_unused]] da_int n_threads, da_int *n_ind,
    T *n_dist) {

    da_int n_list = this->n_list;
    da_int n_probe = this->n_probe;
    bool is_euclidean = this->internal_metric == approx_nn_metric::sqeuclidean;
    bool is_cosine = this->internal_metric == approx_nn_metric::cosine;
    da_int threading_error = 0;

#pragma omp parallel default(none) num_threads(n_threads)                                \
    shared(X_test_ptr, ldx_test, n_queries, n_features, query_blk_sz, list_blk_sz,       \
               k_neigh, n_list, n_probe, is_euclidean, is_cosine, final_query_blk_sz,    \
               n_blocks, centroids_ptr, ld_centroids_local, n_dist, n_ind,               \
               return_distance, threading_error)
    {
        // Per-thread work buffers:
        // coarse_distances_buf - store distances from query to centroid
        // fine_distances_buf - store distances from query to list vectors
        // query_buf - for a given list, store all queries that probe that list
        // query_cos_buf - normalized query block for cosine metric
        // qnorms_buf - work array used for query norms in euclidean computations
        // centroid_indices_buf - indices backing the centroid selection max-heap
        // cent_sel_dists_buf - distances backing the centroid selection max-heap
        // queries_per_centroid_buf - for a given list, which queries probe it
        // queries_per_centroid_cnt - for a given list, how many queries probe it
        // local_heap_dists
        // heap_indices_buf - underlying data for heaps
        // topk_indices_buf - work array used when writing back to results
        std::vector<T> coarse_distances_buf, local_heap_dists, fine_distances_buf,
            query_buf, query_cos_buf, qnorms_buf, cent_sel_dists_buf;
        std::vector<da_int> centroid_indices_buf, queries_per_centroid_buf,
            queries_per_centroid_cnt, heap_indices_buf, topk_indices_buf;
        std::vector<da_binary_tree::MaxHeap<T>> heaps_buf;

        try {
            coarse_distances_buf.resize(query_blk_sz * n_list);
            centroid_indices_buf.resize(n_probe);
            cent_sel_dists_buf.resize(n_probe);
            queries_per_centroid_buf.resize(n_list * query_blk_sz);
            queries_per_centroid_cnt.resize(n_list, 0);
            heap_indices_buf.resize(query_blk_sz * k_neigh, -1);
            local_heap_dists.resize(query_blk_sz * k_neigh,
                                    std::numeric_limits<T>::infinity());
            heaps_buf.resize(query_blk_sz);
            fine_distances_buf.resize(query_blk_sz * list_blk_sz, 0.0);
            query_buf.resize(query_blk_sz * n_features, 0.0);
            if (is_cosine)
                query_cos_buf.resize(query_blk_sz * n_features, 0.0);
            if (is_euclidean)
                qnorms_buf.resize(query_blk_sz, 0.0);
            topk_indices_buf.resize(k_neigh, 0);
        } catch (std::bad_alloc const &) {
#pragma omp atomic write
            threading_error = 1;
        }

#pragma omp barrier

        da_int *heap_indices = heap_indices_buf.data();
        T *heap_distances = local_heap_dists.data();
        auto heaps = heaps_buf.data();
        T *fine_distances = fine_distances_buf.data();
        T *queries = query_buf.data();
        T *query_cos = is_cosine ? query_cos_buf.data() : nullptr;
        T *qnorms = is_euclidean ? qnorms_buf.data() : nullptr;
        T *coarse_dist = coarse_distances_buf.data();
        da_int *cent_idx = centroid_indices_buf.data();
        T *cent_sel_dists = cent_sel_dists_buf.data();
        da_int *qpc = queries_per_centroid_buf.data();
        da_int *qpc_count = queries_per_centroid_cnt.data();

        if (!threading_error) {
#pragma omp for schedule(dynamic) nowait
            for (da_int i = 0; i < n_blocks; i++) {
                da_int this_query_blk_sz;
                if ((i == n_blocks - 1) && final_query_blk_sz > 0) {
                    this_query_blk_sz = final_query_blk_sz;
                } else {
                    this_query_blk_sz = query_blk_sz;
                }

                da_int block_start = i * query_blk_sz;
                da_int block_end = std::min((i + 1) * query_blk_sz, n_queries);

                // Reset heaps for this block
                da_std::fill(heap_indices, heap_indices + k_neigh * query_blk_sz, -1);
                da_std::fill(heap_distances, heap_distances + k_neigh * query_blk_sz,
                             std::numeric_limits<T>::infinity());
                for (da_int ii = 0; ii < query_blk_sz; ii++) {
                    heaps[ii] =
                        da_binary_tree::MaxHeap<T>(k_neigh, heap_indices + ii * k_neigh,
                                                   heap_distances + ii * k_neigh);
                }

                const T *query_blk_ptr = X_test_ptr + block_start * ldx_test;
                if (is_cosine) {
                    for (da_int ii = 0; ii < this_query_blk_sz; ii++) {
                        memcpy(query_cos + static_cast<size_t>(ii) *
                                               static_cast<size_t>(n_features),
                               query_blk_ptr + static_cast<size_t>(ii) *
                                                   static_cast<size_t>(ldx_test),
                               static_cast<size_t>(n_features) * sizeof(T));
                    }
                    da_utils::normalize_rows_inplace(row_major, this_query_blk_sz,
                                                     n_features, query_cos, n_features,
                                                     static_cast<T *>(nullptr));
                    query_blk_ptr = query_cos;
                }

                // Coarse GEMM: query-to-centroid distances for this thread's block
                da_std::fill(qpc_count, qpc_count + n_list, 0);

                // Calculate coarse distance for queries
                if (is_euclidean) {
                    ARCH::euclidean_gemm_distance(
                        row_major, this_query_blk_sz, n_list, n_features,
                        X_test_ptr + block_start * ldx_test, ldx_test, centroids_ptr,
                        ld_centroids_local, coarse_dist, n_list, qnorms, 2,
                        this->centroid_norms.data(), 1, true, false);
                } else {
                    da_blas::cblas_gemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans, this_query_blk_sz,
                        n_list, n_features, static_cast<T>(-1.0), query_blk_ptr,
                        is_cosine ? n_features : ldx_test, centroids_ptr,
                        ld_centroids_local, static_cast<T>(0.0), coarse_dist, n_list);
                }

                // For each query, find the n_probe nearest centroids.
                // Use a max-heap linear scan
                for (da_int q = block_start; q < block_end; q++) {
                    const T *query_distances = coarse_dist + (q - block_start) * n_list;
                    da_std::fill(cent_sel_dists, cent_sel_dists + n_probe,
                                 std::numeric_limits<T>::max());
                    da_binary_tree::MaxHeap<T> cent_heap(n_probe, cent_idx,
                                                         cent_sel_dists);
                    T max_cent_dist = cent_heap.GetMaxDist();
                    for (da_int c = 0; c < n_list; c++) {
                        T d = query_distances[c];
                        if (d < max_cent_dist) {
                            cent_heap.Insert(c, d);
                            max_cent_dist = cent_heap.GetMaxDist();
                        }
                    }
                    for (da_int p = 0; p < n_probe; p++) {
                        da_int c = cent_idx[p];
                        qpc[c * query_blk_sz + qpc_count[c]] = q;
                        qpc_count[c]++;
                    }
                }

                // Sequential loop over all lists for this thread's block
                for (da_int j = 0; j < n_list; j++) {
                    da_int list_size = this->list_sizes[j];
                    const T *this_list = this->indexed_vectors[j].data();
                    T *this_list_norms = this->list_norms[j].data();
                    const da_int *this_list_idx = this->global_indices[j].data();
                    const da_int *qpc_j = qpc + j * query_blk_sz;
                    const da_int q_count = qpc_count[j];

                    if (list_size > 0 && q_count > 0) {
                        // Gather queries
                        for (da_int k = 0; k < q_count; k++) {
                            const T *query_ptr =
                                is_cosine
                                    ? query_cos + (qpc_j[k] - block_start) * n_features
                                    : X_test_ptr + qpc_j[k] * ldx_test;
                            memcpy(queries + static_cast<size_t>(k) *
                                                 static_cast<size_t>(n_features),
                                   query_ptr,
                                   static_cast<size_t>(n_features) * sizeof(T));
                        }

                        // For each block in list vectors, calculate distances and update heaps
                        for (da_int t = 0; t < list_size; t += list_blk_sz) {
                            da_int this_list_blk_sz =
                                std::min(list_blk_sz, list_size - t);
                            if (is_euclidean) {
                                ARCH::euclidean_gemm_distance(
                                    row_major, q_count, this_list_blk_sz, n_features,
                                    queries, n_features, this_list + t * n_features,
                                    n_features, fine_distances, this_list_blk_sz, qnorms,
                                    2, this_list_norms + t, 1, true, false);
                            } else {
                                da_blas::cblas_gemm(
                                    CblasRowMajor, CblasNoTrans, CblasTrans, q_count,
                                    this_list_blk_sz, n_features, static_cast<T>(-1.0),
                                    queries, n_features, this_list + t * n_features,
                                    n_features, static_cast<T>(0.0), fine_distances,
                                    this_list_blk_sz);
                            }
                            update_heaps_from_list_blk(this_list_blk_sz, q_count,
                                                       this_list_idx + t, fine_distances,
                                                       block_start, qpc_j, heaps);
                        }
                    }
                }

                // Write results directly — no merge needed, each thread owns its output rows
                for (da_int ii = 0; ii < this_query_blk_sz; ii++) {
                    da_neighbors::sorted_n_dist_n_ind(
                        k_neigh, heap_distances + ii * k_neigh,
                        heap_indices + ii * k_neigh,
                        n_dist + (block_start + ii) * k_neigh,
                        n_ind + (block_start + ii) * k_neigh, topk_indices_buf.data(),
                        return_distance, false);
                }
            }
        } // if (!threading_error)
    }     // pragma omp parallel

    if (threading_error)
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");

    return da_status_success;
}

// Performs common setup then delegates to the appropriate kernel (of which there is only one for now)
template <typename T>
da_status approximate_neighbors<T>::ivf_search(da_int n_queries, da_int n_features,
                                               const T *X_test, da_int ldx_test,
                                               da_int *n_ind, T *n_dist, da_int k_neigh,
                                               bool return_distance) {
    da_int max_list_size =
        *std::max_element(this->list_sizes.begin(), this->list_sizes.end());

    // Select query_blk_sz and list_blk_sz to produce square-ish fine GEMMs.
    // Square optimum (query_blk_sz = list_blk_sz = s) solves
    //   s^2 + 2(s)(n_features) = budget  ->  s = (-n_features + sqrt(n_features^2 + budget))
    // query_ub = ceil(n_queries / max_threads) caps block size so n_blocks >= max_threads.
    // It also subsumes the n_queries upper bound (always <=).
    // If one dimension is clamped, the freed budget is redirected to the other.
    const da_int budget = (8 << 20) / static_cast<da_int>(sizeof(T));
    da_int s = static_cast<da_int>(
        (-n_features + std::sqrt(static_cast<double>(n_features * n_features + budget))));

    da_int max_threads = static_cast<da_int>(omp_get_max_threads());
    da_int query_ub =
        std::max(static_cast<da_int>(1), static_cast<da_int>(n_queries / max_threads));
    da_int query_blk_sz = std::min(s, query_ub);
    da_int list_blk_sz = std::min(s, max_list_size);

    if (query_blk_sz < s) {
        da_int numer = budget - query_blk_sz * n_features;
        da_int denom = query_blk_sz + n_features;
        list_blk_sz =
            std::min(std::max(static_cast<da_int>(1), numer / denom), max_list_size);
    }
    if (list_blk_sz < s) {
        da_int numer = budget - list_blk_sz * n_features;
        da_int denom = list_blk_sz + n_features;
        query_blk_sz =
            std::min(std::max(static_cast<da_int>(1), numer / denom), query_ub);
    }

    // Debug override: allows tests to force specific block sizes via da_debug_set
    {
        auto &hidden = context::get_context()->hidden_settings;
        auto it_q = hidden.find("ivf.query_blk_sz");
        if (it_q != hidden.end() && !it_q->second.empty())
            query_blk_sz = std::min(std::max(static_cast<da_int>(std::stoi(it_q->second)),
                                             static_cast<da_int>(1)),
                                    n_queries);
        auto it_l = hidden.find("ivf.list_blk_sz");
        if (it_l != hidden.end() && !it_l->second.empty())
            list_blk_sz = std::min(std::max(static_cast<da_int>(std::stoi(it_l->second)),
                                            static_cast<da_int>(1)),
                                   max_list_size);
    }

    da_int n_blocks, final_query_blk_sz;
    da_utils::blocking_scheme(n_queries, query_blk_sz, n_blocks, final_query_blk_sz);

    da_int n_threads = std::min(max_threads, n_blocks);

    using namespace std::string_literals;
    context_set_hidden_settings("ivf.query_blk_sz_used"s, std::to_string(query_blk_sz));
    context_set_hidden_settings("ivf.list_blk_sz_used"s, std::to_string(list_blk_sz));
    context_set_hidden_settings("ivf.n_blocks"s, std::to_string(n_blocks));

    // For column-major, transpose queries and centroids to row-major for uniform processing
    std::vector<T> X_test_copy, centroids_copy;
    const T *X_test_ptr = X_test;
    const T *centroids_ptr = centroids.data();
    da_int ld_centroids_local = this->ld_centroids;

    if (this->order == column_major) {
        try {
            X_test_copy.resize(n_queries * n_features);
            centroids_copy.resize(n_list * n_features);
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }
        da_blas::omatcopy('T', n_queries, n_features, static_cast<T>(1.0), X_test,
                          ldx_test, X_test_copy.data(), n_features);
        X_test_ptr = X_test_copy.data();
        ldx_test = n_features;
        da_blas::omatcopy('T', n_list, n_features, static_cast<T>(1.0), centroids.data(),
                          this->ld_centroids, centroids_copy.data(), n_features);
        centroids_ptr = centroids_copy.data();
        ld_centroids_local = n_features;
    }

    da_status status;
    status = ivfflat_search_query_parallel(
        n_queries, n_features, k_neigh, return_distance, X_test_ptr, ldx_test,
        centroids_ptr, ld_centroids_local, query_blk_sz, list_blk_sz, n_blocks,
        final_query_blk_sz, n_threads, n_ind, n_dist);

    if (status != da_status_success)
        return status;

    if (return_distance) {
        switch (this->metric) {
        case approx_nn_metric::euclidean: {
            for (da_int i = 0; i < n_queries; i++) {
                for (da_int j = 0; j < k_neigh; j++) {
                    T val = n_dist[i * k_neigh + j];
                    n_dist[i * k_neigh + j] = (val < 0) ? val : std::sqrt(val);
                }
            }
            break;
        }
        case approx_nn_metric::cosine: {
#pragma omp simd
            for (da_int i = 0; i < k_neigh * n_queries; i++)
                n_dist[i] = static_cast<T>(1.0) + n_dist[i];
            break;
        }
        case approx_nn_metric::inner_product: {
            da_blas::cblas_scal(k_neigh * n_queries, static_cast<T>(-1.0), n_dist, 1);
            break;
        }
        default:
            break;
        }
    }

    return da_status_success;
}

template <typename T>
da_status approximate_neighbors<T>::kneighbors_compute_ivfflat(
    da_int n_queries, da_int n_features, const T *X_test, da_int ldx_test, da_int *n_ind,
    T *n_dist, da_int k_neigh, bool return_distance) {

    da_status status = ivf_search(n_queries, n_features, X_test, ldx_test, n_ind, n_dist,
                                  k_neigh, return_distance);

    if (status != da_status_success)
        return status;

    if (this->order == column_major) {
// If da_int is 64 bit, cast to double
#if defined(AOCLDA_ILP64)
        da_blas::imatcopy('T', k_neigh, n_queries, 1.0, reinterpret_cast<double *>(n_ind),
                          k_neigh, n_queries);
#else // da_int is 32 bit, cast to float
        da_blas::imatcopy('T', k_neigh, n_queries, 1.0, reinterpret_cast<float *>(n_ind),
                          k_neigh, n_queries);
#endif
        // transpose distances
        if (return_distance) {
            da_blas::imatcopy('T', k_neigh, n_queries, 1.0, n_dist, k_neigh, n_queries);
        }
    }

    return da_status_success;
}

template <typename T>
da_status approximate_neighbors<T>::serialize(serialization_buffer &buffer) {
    da_status status = da_status_success;
    auto io_dispatch = [&buffer, &status](auto &data) -> void {
        if (status != da_status_success) {
            return;
        }
        status = buffer.dispatch_buffer_io(data);
        return;
    };

    io_dispatch(this->train_data_is_set);
    io_dispatch(this->model_trained);
    io_dispatch(this->data_is_added);
    io_dispatch(this->n_neighbors);
    io_dispatch(this->algo);
    io_dispatch(this->internal_algo);
    io_dispatch(this->internal_metric);
    io_dispatch(this->metric);
    io_dispatch(this->internal_seed);
    io_dispatch(this->seed);
    io_dispatch(this->n_samples_train);
    io_dispatch(this->order);
    io_dispatch(this->n_samples);
    io_dispatch(this->n_features);
    io_dispatch(this->train_fraction);
    io_dispatch(this->n_list);
    io_dispatch(this->n_probe);
    io_dispatch(this->max_iter);
    io_dispatch(this->kmeans_iter);
    io_dispatch(this->n_init);
    io_dispatch(this->kmeans_tol);
    io_dispatch(this->centroids);
    io_dispatch(this->ld_centroids);
    io_dispatch(this->n_index);
    io_dispatch(this->indexed_vectors);
    io_dispatch(this->global_indices);
    io_dispatch(this->list_sizes);
    io_dispatch(this->old_list_sizes);
    io_dispatch(this->list_norms);
    io_dispatch(this->centroid_norms);

    if (status != da_status_success)
        return status;

    if (buffer.get_mode() == deserialize) {
        status = buffer.deserialize_data(this->X_int);
    } else {
        status = buffer.serialize_user_data(this->X_train, this->order, this->n_samples,
                                            this->n_features, this->ldx_train);
    }

    return status;
}

template <typename T>
da_status approximate_neighbors<T>::save_model(serialization_buffer &buffer) {
    if (!this->train_data_is_set || !this->model_trained) {
        return da_error(
            this->err, da_status_no_data,
            "Index has not yet been trained. Please call da_approx_nn_train_d "
            "or da_approx_nn_train_s before saving the model.");
    }

    da_status status = basic_handle<T>::save_model(buffer);
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure serializing model.");

    return status;
}

template <typename T>
da_status approximate_neighbors<T>::load_model(serialization_buffer &buffer) {

    da_status status = basic_handle<T>::load_model(buffer);
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure deserializing model.");

    this->X_train = this->X_int.data();
    if (this->order == column_major) {
        this->ldx_train = this->n_samples;
    } else {
        this->ldx_train = this->n_features;
    }

    return status;
}

template class approximate_neighbors<double>;
template class approximate_neighbors<float>;

} // namespace da_approx_nn

} // namespace ARCH
