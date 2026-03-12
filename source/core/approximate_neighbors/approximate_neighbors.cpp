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
#include "da_error.hpp"
#include "da_omp.hpp"
#include "da_utils.hpp"
#include "kmeans.hpp"
#include "nearest_neighbors_utils.hpp"
#include "pairwise_distances.hpp"

#include <algorithm>

namespace ARCH {

namespace da_approx_nn {

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
da_status approximate_neighbors<T>::get_result([[maybe_unused]] da_result query,
                                               [[maybe_unused]] da_int *dim,
                                               [[maybe_unused]] T *result) {
    if (!index_is_trained) {
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
        result[0] = (T)n_list;
        result[1] = (T)n_index;
        result[2] = (T)n_features;
        result[3] = (T)kmeans_iter;
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
da_status approximate_neighbors<T>::get_result([[maybe_unused]] da_result query,
                                               [[maybe_unused]] da_int *dim,
                                               [[maybe_unused]] da_int *result) {
    if (!index_is_trained) {
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
    this->index_is_trained = false;
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
                              ? approx_nn_algorithm::ivfflat
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
        X_train_sampled.resize(this->n_samples_train * this->n_features, 0.0);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    da_std::iota(indices.begin(), indices.end(), 0);
    // Get indices of rows to get
    da_std::sample(indices.begin(), indices.end(), perm.begin(), this->n_samples_train,
                   this->mt_engine);

    da_int n_threads = omp_get_max_threads();

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
        da_int row_bytes = this->n_features * sizeof(T);
#pragma omp parallel for num_threads(std::min(n_threads, n_samples_train)) default(none) \
    shared(n_features, n_samples_train, X_train_sampled, ldx_train_sampled, X_train,     \
               ldx_train, perm, row_bytes) private(row_idx)
        for (da_int i = 0; i < this->n_samples_train; i++) {
            row_idx = perm[i];
            memcpy(X_train_sampled.data() + i * ldx_train_sampled,
                   this->X_train + row_idx * this->ldx_train, row_bytes);
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
        this->centroids.assign(n_list * n_features, 0.0);

        // Use resize + clear for nested da_vector containers
        this->indexed_vectors.resize(n_list);
        this->global_indices.resize(n_list);
        for (da_int i = 0; i < n_list; i++) {
            this->indexed_vectors[i].clear();
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
    bool need_copy = (this->train_fraction < (T)1.0) ||
                     (this->internal_metric == approx_nn_metric::cosine);

    if (need_copy) {
        if (this->train_fraction < (T)1.0) {
            // Calculate n_samples_train
            // Bound below by n_list. We have an earlier check that n_list < n_samples
            this->n_samples_train = std::max(
                (da_int)(this->n_samples_train * this->train_fraction), this->n_list);

            status = this->subsample_training_data(X_train_work, ldx_train_work);
            if (status != da_status_success)
                return status;
        } else {
            // Full copy for normalization
            ldx_train_work =
                (this->order == column_major) ? this->n_samples_train : this->n_features;
            try {
                X_train_work.resize(this->n_samples_train * this->n_features);
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
                da_int row_bytes = this->n_features * sizeof(T);
                for (da_int i = 0; i < this->n_samples_train; i++) {
                    memcpy(X_train_work.data() + i * ldx_train_work,
                           this->X_train + i * this->ldx_train, row_bytes);
                }
            }
        }

        // Normalize training data in-place for cosine metric
        if (this->internal_metric == approx_nn_metric::cosine) {
            status = da_utils::normalize_rows_inplace(
                this->order, this->n_samples_train, this->n_features, X_train_work.data(),
                ldx_train_work, (T *)nullptr);
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
    kmeans_model.do_options_check = false;

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
        this->internal_seed = std::abs((da_int)r());
    }
    this->mt_engine.seed(this->internal_seed);

    if (this->internal_algo == da_approx_nn_types::approx_nn_algorithm::ivfflat) {
        status = this->train_ivfflat();
    } else {
        return da_error_bypass(this->err, da_status_invalid_input,
                               "Unknown algorithm: " + std::to_string(internal_algo) +
                                   ".");
    }
    if (status != da_status_success)
        return status;

    index_is_trained = true;

    return da_status_success;
}

template <typename T>
da_status approximate_neighbors<T>::add(da_int n_samples_add, da_int n_features,
                                        const T *X_add, da_int ldx_add) {
    // Check we have already trained successfully
    if (!index_is_trained) {
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
        return da_error_bypass(this->err, da_status_invalid_input,
                               "Unknown algorithm: " + std::to_string(internal_algo) +
                                   ".");
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
            X_add_work.resize(n_samples_add * n_features);
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error,
                            "Memory allocation failed.");
        }
        da_status status = da_utils::normalize_rows(
            this->order, n_samples_add, n_features, X_add, ldx_add, X_add_work.data(),
            ldx_add_work, (T *)nullptr);
        if (status != da_status_success)
            return status;
        X_add_ptr = X_add_work.data();
        ldx_add_ptr = ldx_add_work;
    }

    // distances will store distances from X_add to individual centroids
    // work1 and work2 are for vector norms in euclidean_gemm_distance
    std::vector<T> distances, work1, work2;
    // local_indices - For each centroid this stores indices of rows of X_add
    // that will be added to it
    std::vector<da_vector::da_vector<da_int>> local_indices;

    da_int ld_distances = n_list;

    try {
        distances.resize(n_list * n_samples_add, 0.0);
        work1.resize(n_list, 0.0);
        work2.resize(n_samples_add, 0.0);
        local_indices.resize(n_list);

        // If we are adding data to an index which already has data in it, we need
        // to remember old list sizes
        if (data_is_added) {
            this->old_list_sizes = this->list_sizes;
        }
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    // Store in distances the distance from each row of X_add to the individual centroids
    // We formulate the gemms so that the centroid distances for each row of X_add
    // are contiguous in distances, hence the differing order of gemm for row and column
    // major
    if (this->internal_metric == approx_nn_metric::sqeuclidean) {
        // centroids = C
        if (this->order == column_major) {
            // in column major compute -2 * C * (X_add)^T + (appropriate matrix norms)
            // as handled by euclidean_gemm_distance
            ARCH::euclidean_gemm_distance(this->order, this->n_list, n_samples_add,
                                          n_features, this->centroids.data(),
                                          this->ld_centroids, X_add_ptr, ldx_add_ptr,
                                          distances.data(), ld_distances, work1.data(), 2,
                                          work2.data(), 2, true, false);
        } else {
            // in row major compute -2 * X_add * C^T + (appropriate matrix norms)
            // as handled by euclidean_gemm_distance
            ARCH::euclidean_gemm_distance(
                this->order, n_samples_add, this->n_list, n_features, X_add_ptr,
                ldx_add_ptr, this->centroids.data(), this->ld_centroids, distances.data(),
                ld_distances, work2.data(), 2, work1.data(), 2, true, false);
        }
    } else {
        // inner product or cosine - we do gemm with alpha=-1
        if (this->order == column_major) {
            // in column major compute  C * (X_add)^T
            da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, this->n_list,
                                n_samples_add, n_features, (T)-1.0,
                                this->centroids.data(), this->ld_centroids, X_add_ptr,
                                ldx_add_ptr, (T)0.0, distances.data(), ld_distances);

        } else {
            // in row major compute X_add * C^T
            da_blas::cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_samples_add,
                                this->n_list, n_features, (T)-1.0, X_add_ptr, ldx_add_ptr,
                                this->centroids.data(), this->ld_centroids, (T)0.0,
                                distances.data(), ld_distances);
        }
    }

    // For each sample in distances, get the index of the centroid with the smallest distance
    // Update local_indices, global_indices and list_sizes as appropriate
    // We can traverse distances in exactly the same fashion for row and column major due
    // to how we formulated the gemm above

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

    da_int nearest_centroid_idx;
    const T *sample_start, *sample_end;

    for (da_int j = 0; j < n_samples_add; j++) {
        sample_start = &distances[j * n_list];
        sample_end = sample_start + n_list;
        nearest_centroid_idx =
            std::distance(sample_start, std::min_element(sample_start, sample_end));
        local_indices[nearest_centroid_idx].push_back(j);
        this->global_indices[nearest_centroid_idx].push_back(j + n_index);
        this->list_sizes[nearest_centroid_idx]++;
    }

    // n_threads is at most n_list
    da_int n_threads = std::min((da_int)omp_get_max_threads(), this->n_list);
    da_int sizeof_T = sizeof(T);
    da_int row_bytes = sizeof_T * this->n_features;

    // Resize indexed_vectors to accommodate new data
    // old_list_sizes is initialized to 0 in train_ivfflat(), so the loop below
    // works for both first call (old_size=0) and subsequent calls
    try {
        for (da_int i = 0; i < n_list; i++) {
            this->indexed_vectors[i].resize(this->list_sizes[i] * n_features);
        }
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    // Read from X_add_ptr and write to indexed_vectors (always stored as row major)
    // Leading dim of each indexed_vector[list_idx] is n_features
    // old_list_sizes[i] is 0 on first call, so we iterate from 0 to list_sizes[i]
#pragma omp parallel for num_threads(n_threads) schedule(dynamic) default(none)          \
    shared(old_list_sizes, list_sizes, indexed_vectors, local_indices, X_add_ptr,        \
               n_list, n_features, ldx_add_ptr, row_bytes)
    for (da_int list_idx = 0; list_idx < n_list; list_idx++) {
        da_int old_size = this->old_list_sizes[list_idx];
        da_int new_size = this->list_sizes[list_idx];
        T *list_ptr = this->indexed_vectors[list_idx].data();
        da_int *indices_to_add = local_indices[list_idx].data();

        if (this->order == column_major) {
            // Transpose from column major input to row major storage
            for (da_int i = old_size; i < new_size; i++) {
                da_int add_row_idx = indices_to_add[i - old_size];
                da_int row_idx = i * n_features;
                for (da_int j = 0; j < n_features; j++) {
                    list_ptr[row_idx + j] = X_add_ptr[add_row_idx + j * ldx_add_ptr];
                }
            }
        } else {
            // Row major input - direct copy, append new rows
            da_int row_idx;
            for (da_int i = old_size; i < new_size; i++) {
                row_idx = indices_to_add[i - old_size];
                memcpy(list_ptr + i * n_features, X_add_ptr + ldx_add_ptr * row_idx,
                       row_bytes);
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
    if (!this->index_is_trained) {
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

// Kernel to compute nearest neighbours when metric is euclidean or sqeuclidean
template <typename T>
da_status approximate_neighbors<T>::euclidean_search(da_int n_queries, da_int n_features,
                                                     const T *X_test, da_int ldx_test,
                                                     da_int *n_ind, T *n_dist,
                                                     da_int k_neigh,
                                                     bool return_distance) {
    /*  
        Overview:
        1. Compute distance from each row of X_test to each centroid
        2. Iterate over queries (rows of X_test).
        3. Per query: identify n_probe closest centroids. Iterate over n_probe lists corresponding 
        to each centroid. Compute k_neigh closest indexed_vectors seen for each query.
    
        coarse_distances - query to centroid distances
        fine_distances - query to indexed vector distances
        centroid_top_k_distances - best k distances for a query <-> list search
        centroid_top_k_indices - best k indices for a query <-> list search
        norms_work1, norms_work2 - workspace for gemm computations
        lists_to_probe - which lists to search per query
        heap_distances, heap_indices - arrays for per thread heaps
        list_norms - precomputed norms of all indexed vectors
        norms_prefix_sum - used to determine per list boundaries in list_norms 
        query - in case of column major, we will gather individual queries to be a contiguous vector
    */

    std::vector<T> coarse_distances, list_norms, fine_distances, centroid_top_k_distances,
        norms_work1, norms_work2, heap_distances, query;
    std::vector<da_int> centroid_top_k_indices, lists_to_probe, heap_indices,
        norms_prefix_sum;

    da_int ld_distances = n_list;
    da_int max_list_size = *std::max_element(list_sizes.begin(), list_sizes.end());

    try {
        //shared arrays
        coarse_distances.resize(n_list * n_queries, 0.0);
        norms_prefix_sum.resize(n_list, 0);
        list_norms.resize(n_index, 0.0);

        // thread local work arrays
        // like below and firstprivate, or shared and padded??
        fine_distances.resize(max_list_size, 0.0);
        centroid_top_k_distances.resize(k_neigh, 0.0);
        centroid_top_k_indices.resize(k_neigh, -1);
        lists_to_probe.resize(n_list, 0);
        heap_distances.resize(k_neigh, std::numeric_limits<T>::infinity());
        heap_indices.resize(k_neigh, -1);
        norms_work1.resize(std::max(n_list, max_list_size), 0.0);
        norms_work2.resize(std::max(n_queries, (da_int)1), 0.0);

        if (this->order == column_major) {
            query.resize(n_features);
        }

    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    // Calculate distances from queries to centroids
    // This calculation is done so the coarse distances for each query are always contiguous
    // in memory, hence the differing order for column and row major
    if (this->order == column_major) {
        ARCH::euclidean_gemm_distance(
            this->order, this->n_list, n_queries, n_features, centroids.data(),
            this->ld_centroids, X_test, ldx_test, coarse_distances.data(), ld_distances,
            norms_work1.data(), 2, norms_work2.data(), 2, true, false);
    } else {
        ARCH::euclidean_gemm_distance(
            this->order, n_queries, this->n_list, n_features, X_test, ldx_test,
            centroids.data(), this->ld_centroids, coarse_distances.data(), ld_distances,
            norms_work2.data(), 2, norms_work1.data(), 2, true, false);
    }

    // Precompute list_norms
    // indexed_vectors is always stored in row major format with leading dimension n_features

    da_int list_start = 0;
    for (da_int list_idx = 0; list_idx < this->n_list; list_idx++) {
        const T *this_list = this->indexed_vectors[list_idx].data();
        da_int list_size = this->list_sizes[list_idx];
        if (list_idx > 0) {
            norms_prefix_sum[list_idx] +=
                norms_prefix_sum[list_idx - 1] + list_sizes[list_idx - 1];
        }

        for (da_int i = 0; i < list_size; i++) {
            da_int row_offset = i * n_features;
            for (da_int j = 0; j < n_features; j++) {
                list_norms[list_start + i] +=
                    this_list[j + row_offset] * this_list[j + row_offset];
            }
        }
        list_start += list_size;
    }

    // Loop over queries
    da_int n_threads = std::min((da_int)omp_get_max_threads(), n_queries);
    const T *query_ptr;

#pragma omp parallel for schedule(static) num_threads(n_threads) default(none)           \
    firstprivate(fine_distances, centroid_top_k_distances, centroid_top_k_indices,       \
                     lists_to_probe, heap_distances, heap_indices, query)                \
    shared(n_probe, n_list, max_list_size, n_queries, n_features, X_test, ldx_test,      \
               n_ind, n_dist, k_neigh, return_distance, coarse_distances, list_norms,    \
               list_sizes, indexed_vectors, global_indices,                              \
               norms_prefix_sum) private(query_ptr)
    for (da_int i = 0; i < n_queries; i++) {
        // For column-major let's gather each query to be contiguous at query time
        if (this->order == column_major) {
            for (da_int j = 0; j < n_features; j++) {
                query[j] = X_test[i + j * ldx_test];
            }
            query_ptr = query.data();
        } else {
            query_ptr = &X_test[i * ldx_test];
        }

        // Precompute query norm
        T query_norm = 0;
        for (da_int ii = 0; ii < n_features; ii++) {
            query_norm += query_ptr[ii] * query_ptr[ii];
        }

        // Search coarse_distances for shortest n_probe distances
        // Now the first n_probe elements in list_distances are the indices of the
        // relevant lists to search for this query
        T *list_distances = coarse_distances.data() + i * n_list;
        da_std::iota(lists_to_probe.begin(), lists_to_probe.end(), 0);
        std::partial_sort(lists_to_probe.begin(), lists_to_probe.begin() + n_probe,
                          lists_to_probe.end(), [list_distances](da_int a, da_int b) {
                              return list_distances[a] < list_distances[b];
                          });

        // Fill heap with sentinel indices and distances
        da_std::fill(heap_distances.begin(), heap_distances.end(),
                     std::numeric_limits<T>::infinity());
        da_std::fill(heap_indices.begin(), heap_indices.end(), -1);
        auto heap = da_binary_tree::MaxHeap<T>(k_neigh, heap_indices.data(),
                                               heap_distances.data());

        // Loop over lists to probe, maintaining a per query heap of k-best candidates
        for (da_int j = 0; j < n_probe; j++) {
            da_int list_idx = lists_to_probe[j];
            da_int list_size = this->list_sizes[list_idx];
            da_int norms_idx = norms_prefix_sum[list_idx];
            const T *this_list = this->indexed_vectors[list_idx].data();
            const da_int *this_list_idx = this->global_indices[list_idx].data();

            if (list_size > 0) {
                // Compute the gemv -2 * X * q
                // X = indexed vectors for this list (always row major with ld = n_features)
                // q = query vector
                da_blas::cblas_gemv(CblasRowMajor, CblasNoTrans, list_size, n_features,
                                    -2.0, this_list, n_features, query_ptr, 1, 0.0,
                                    fine_distances.data(), 1);

                // Add list and query norms to each distance
                // query_norm is strictly only needed if return_distance = true
                for (da_int sample = 0; sample < list_size; sample++) {
                    fine_distances[sample] += list_norms[norms_idx + sample] + query_norm;
                }

                // Get the k smallest indices and distances from fine_distances
                std::copy_n(fine_distances.data(), std::min(list_size, k_neigh),
                            centroid_top_k_distances.data());
                da_neighbors::smaller_values_and_indices(
                    list_size, fine_distances.data(), std::min(k_neigh, list_size),
                    centroid_top_k_indices.data(), centroid_top_k_distances.data(), 0,
                    true);

                // Get the top k distances, and see what can be added to the heap
                T dis;
                for (da_int k = 0; k < std::min(list_size, k_neigh); k++) {
                    dis = centroid_top_k_distances[k];
                    if (dis < heap.GetMaxDist()) {
                        heap.Insert(this_list_idx[centroid_top_k_indices[k]], dis);
                    }
                }
            }
        }

        // Write heaps to results, with distances if needed.
        // This works, but given it is a heap there is probably a better sort to use?
        da_neighbors::sorted_n_dist_n_ind(
            k_neigh, heap_distances.data(), heap_indices.data(), n_dist + i * k_neigh,
            n_ind + i * k_neigh, centroid_top_k_indices.data(), return_distance, false);

        // If metric is euclidean, return sqrt of distances
        if ((this->metric == approx_nn_metric::euclidean) && (return_distance)) {
            for (da_int j = 0; j < k_neigh; j++) {
                T val = n_dist[i * k_neigh + j];
                n_dist[i * k_neigh + j] = (val < 0) ? val : std::sqrt(val);
            }
        }
    }
    return da_status_success;
}

// This is a very similar codepath to euclidean_search and can probably be refactored
// Leaving separate for now for clarity while developing
template <typename T>
da_status approximate_neighbors<T>::inner_product_search(
    da_int n_queries, da_int n_features, const T *X_test, da_int ldx_test, da_int *n_ind,
    T *n_dist, da_int k_neigh, bool return_distance) {
    /*  
        Overview:
        1. Compute distance from each row of X_test to each centroid
        2. Iterate over queries (rows of X_test).
        3. Per query: identify n_probe closest centroids. Iterate over n_probe lists corresponding 
        to each centroid. Compute k_neigh closest indexed_vectors seen for each query.
    
        coarse_distances - query to centroid distances
        fine_distances - query to indexed vector distances
        centroid_top_k_distances - best k distances for a query <-> list search
        centroid_top_k_indices - best k local indices for a query <-> list search
        lists_to_probe - which lists to search per query
        heap_distances, heap_indices - arrays for per thread heaps
    */

    std::vector<T> coarse_distances, fine_distances, centroid_top_k_distances,
        heap_distances, query;
    std::vector<da_int> centroid_top_k_indices, lists_to_probe, heap_indices;

    da_int ld_distances = n_list;
    da_int max_list_size =
        *std::max_element(this->list_sizes.begin(), this->list_sizes.end());

    try {
        //shared arrays
        coarse_distances.resize(this->n_list * n_queries, 0.0);
        // thread local work arrays
        fine_distances.resize(max_list_size, 0.0);
        centroid_top_k_distances.resize(k_neigh, 0.0);
        centroid_top_k_indices.resize(k_neigh, -1);
        lists_to_probe.resize(this->n_list, 0);
        heap_distances.resize(k_neigh, std::numeric_limits<T>::infinity());
        heap_indices.resize(k_neigh, -1);
        if (this->order == column_major ||
            this->internal_metric == approx_nn_metric::cosine) {
            query.resize(n_features);
        }
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    // This calculation is done so the coarse distances for each query are always contiguous
    // in memory, hence the differing order for column and row major
    if (this->order == column_major) {
        // In column-major compute -centroids *  X_test^T
        da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, this->n_list,
                            n_queries, n_features, (T)-1.0, centroids.data(),
                            this->ld_centroids, X_test, ldx_test, (T)0.0,
                            coarse_distances.data(), ld_distances);
    } else {
        // In row-major compute -X_test * centroids^T
        da_blas::cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_queries,
                            this->n_list, n_features, (T)-1.0, X_test, ldx_test,
                            centroids.data(), this->ld_centroids, (T)0.0,
                            coarse_distances.data(), ld_distances);
    }

    // Step 2: Loop over queries
    da_int n_threads = std::min((da_int)omp_get_max_threads(), n_queries);
    const T *query_ptr;

#pragma omp parallel for schedule(static) num_threads(n_threads) default(none)           \
    firstprivate(fine_distances, centroid_top_k_distances, centroid_top_k_indices,       \
                     lists_to_probe, heap_distances, heap_indices, query)                \
    shared(n_probe, n_list, max_list_size, n_queries, n_features, X_test, ldx_test,      \
               n_ind, n_dist, k_neigh, return_distance, coarse_distances, list_sizes,    \
               indexed_vectors, global_indices) private(query_ptr)
    for (da_int i = 0; i < n_queries; i++) {
        // For column-major let's gather each query at query time
        // For cosine metric, we also need to normalize the query
        if (this->order == column_major) {
            for (da_int j = 0; j < n_features; j++) {
                query[j] = X_test[i + j * ldx_test];
            }
            if (this->internal_metric == approx_nn_metric::cosine) {
                T norm = da_blas::cblas_nrm2(n_features, query.data(), 1);
                if (norm > 0) {
                    da_blas::cblas_scal(n_features, (T)1.0 / norm, query.data(), 1);
                }
            }
            query_ptr = query.data();
        } else if (this->internal_metric == approx_nn_metric::cosine) {
            // Row-major cosine: copy and normalize in one pass
            const T *src = &X_test[i * ldx_test];
            T norm = da_blas::cblas_nrm2(n_features, src, 1);
            T inv_norm = (norm > 0) ? (T)1.0 / norm : (T)1.0;
            for (da_int j = 0; j < n_features; j++) {
                query[j] = src[j] * inv_norm;
            }
            query_ptr = query.data();
        } else {
            query_ptr = &X_test[i * ldx_test];
        }

        // Search coarse_distances for shortest n_probe distances
        // Now the first n_probe elements in list_distances are the indices of the
        // relevant lists to search for this query
        T *list_distances = coarse_distances.data() + i * n_list;
        da_std::iota(lists_to_probe.begin(), lists_to_probe.end(), 0);
        std::partial_sort(lists_to_probe.begin(), lists_to_probe.begin() + n_probe,
                          lists_to_probe.end(), [list_distances](da_int a, da_int b) {
                              return list_distances[a] < list_distances[b];
                          });

        // Fill heap with sentinel indices and distances
        da_std::fill(heap_distances.begin(), heap_distances.end(),
                     std::numeric_limits<T>::infinity());
        da_std::fill(heap_indices.begin(), heap_indices.end(), -1);
        auto heap = da_binary_tree::MaxHeap<T>(k_neigh, heap_indices.data(),
                                               heap_distances.data());

        // Step 3: Loop over probes, maintaining a per-query heap of k-best candidates
        for (da_int j = 0; j < n_probe; j++) {
            da_int list_idx = lists_to_probe[j];
            da_int list_size = this->list_sizes[list_idx];
            const T *this_list = this->indexed_vectors[list_idx].data();
            const da_int *this_list_idx = this->global_indices[list_idx].data();

            if (list_size > 0) {
                // Compute the gemv -X * q
                // X = indexed vectors for this list (always row major with ld = n_features)
                // q = query vector
                da_blas::cblas_gemv(CblasRowMajor, CblasNoTrans, list_size, n_features,
                                    -1.0, this_list, n_features, query_ptr, 1, 0.0,
                                    fine_distances.data(), 1);

                // Get the k smallest indices and distances from fine_distances
                std::copy_n(fine_distances.data(), std::min(list_size, k_neigh),
                            centroid_top_k_distances.data());
                da_neighbors::smaller_values_and_indices(
                    list_size, fine_distances.data(), std::min(k_neigh, list_size),
                    centroid_top_k_indices.data(), centroid_top_k_distances.data(), 0,
                    true);

                // Get the top k distances, and see what can be added to the heap
                T dis;
                for (da_int k = 0; k < std::min(list_size, k_neigh); k++) {
                    dis = centroid_top_k_distances[k];
                    if (dis < heap.GetMaxDist()) {
                        heap.Insert(this_list_idx[centroid_top_k_indices[k]], dis);
                    }
                }
            }
        }
        // Write heaps to results, with distances if needed.
        // This works, but given it is a heap there is probably a better sort to use
        da_neighbors::sorted_n_dist_n_ind(
            k_neigh, heap_distances.data(), heap_indices.data(), n_dist + i * k_neigh,
            n_ind + i * k_neigh, centroid_top_k_indices.data(), return_distance, false);
    }
    if (return_distance) {
        if (this->internal_metric == approx_nn_metric::cosine) {
            // For cosine: distance = 1 - similarity = 1 - (-negated_value) = 1 + negated_value
#pragma omp simd
            for (da_int i = 0; i < k_neigh * n_queries; i++) {
                n_dist[i] = (T)1.0 + n_dist[i];
            }
        } else {
            // For inner product need to reverse distances by -1
            da_blas::cblas_scal(k_neigh * n_queries, (T)-1.0, n_dist, 1);
        }
    }
    return da_status_success;
}

template <typename T>
da_status approximate_neighbors<T>::kneighbors_compute_ivfflat(
    da_int n_queries, da_int n_features, const T *X_test, da_int ldx_test, da_int *n_ind,
    T *n_dist, da_int k_neigh, bool return_distance) {

    da_status status;
    if (this->internal_metric == approx_nn_metric::sqeuclidean) {
        status = euclidean_search(n_queries, n_features, X_test, ldx_test, n_ind, n_dist,
                                  k_neigh, return_distance);
    } else if (this->internal_metric == approx_nn_metric::inner_product ||
               this->internal_metric == approx_nn_metric::cosine) {
        status = inner_product_search(n_queries, n_features, X_test, ldx_test, n_ind,
                                      n_dist, k_neigh, return_distance);
    } else {
        return da_error(this->err, da_status_internal_error,
                        "Unknown metric: " + std::to_string(this->internal_metric) + ".");
    }

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
} // namespace da_approx_nn

template class approximate_neighbors<double>;
template class approximate_neighbors<float>;

} // namespace da_approx_nn

} // namespace ARCH
