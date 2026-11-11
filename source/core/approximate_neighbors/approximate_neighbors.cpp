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

#include "scalar_quantizer.hpp"
#include <cstring>
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wpass-failed"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wopenmp-simd"
#endif

#include "aoclda.h"
#include "aoclda_types.h"
#include "approximate_neighbors.hpp"
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
        da_utils::copy_array_convert_precision(
            this->order, this->n_list, this->n_features, centroids.data(),
            this->ld_centroids, result, this->ld_centroids);
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
    da_int local_algo, local_metric, local_n_list, local_quantization,
        local_quantization_scheme, local_quantization_residual;
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
    // Only read quantization options if the selected algorithm needs them.
    if (this->algo == approx_nn_algorithm::ivfsq) {
        opt_pass &= this->opts.get("quantization", opt_val, local_quantization) ==
                    da_status_success;
        if (scalar_quantizer::code_kind(local_quantization) != this->sq.get_code_kind()) {
            return da_error_bypass(
                this->err, da_status_option_locked,
                "quantization cannot be changed after calling train().");
        }
        opt_pass &= this->opts.get("quantization_scheme", opt_val,
                                   local_quantization_scheme) == da_status_success;
        if (scalar_quantizer::scheme(local_quantization_scheme) !=
            this->sq.get_scheme()) {
            return da_error_bypass(
                this->err, da_status_option_locked,
                "quantization_scheme cannot be changed after calling train().");
        }
        opt_pass &= this->opts.get("quantization_residual", opt_val,
                                   local_quantization_residual) == da_status_success;
        if ((local_quantization_residual != 0) != this->quantize_residual) {
            return da_error_bypass(
                this->err, da_status_option_locked,
                "quantization_residual cannot be changed after calling train().");
        }
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
    da_int iquantization = scalar_quantizer::none;
    da_int iquantization_scheme = scalar_quantizer::per_dimension;
    da_int iquantization_residual = 0;

    if (this->opts.get("algorithm", opt_val, algo) != da_status_success)
        return da_error_bypass(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                               "Unexpected error while reading algorithm.");

    // Integer options - don't need n_probe or n_neighbors until search time
    opt_pass &= this->opts.get("n_list", n_list) == da_status_success;
    opt_pass &= this->opts.get("k-means_iter", max_iter) == da_status_success;
    opt_pass &= this->opts.get("seed", seed) == da_status_success;

    // fp options
    opt_pass &= this->opts.get("train fraction", train_fraction) == da_status_success;

    // string options
    opt_pass &= this->opts.get("metric", opt_val, imetric) == da_status_success;
    opt_pass &= this->opts.get("storage order", opt_val, iorder) == da_status_success;
    // Only read quantization options if the selected algorithm needs them.
    if (this->algo == approx_nn_algorithm::ivfsq) {
        opt_pass &=
            this->opts.get("quantization", opt_val, iquantization) == da_status_success;
        opt_pass &= this->opts.get("quantization_scheme", opt_val,
                                   iquantization_scheme) == da_status_success;
        opt_pass &= this->opts.get("quantization_residual", opt_val,
                                   iquantization_residual) == da_status_success;
    }

    if (!opt_pass)
        return da_error_bypass(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                               "Unexpected error while reading parameters.");

    if (this->n_list > n_samples) {
        return da_error(
            this->err, da_status_invalid_array_dimension,
            "n_samples = " + std::to_string(n_samples) +
                " must be at least as large as n_list = " + std::to_string(n_list));
    }

    const auto quantization = scalar_quantizer::code_kind(iquantization);
    this->internal_algo = approx_nn_algorithm::ivfflat;

    // If we select bfloat, just force the scheme to be direct_signed
    if (quantization == scalar_quantizer::bfloat16 &&
        scalar_quantizer::scheme(iquantization_scheme) !=
            scalar_quantizer::direct_signed) {
        if (this->opts.set("quantization_scheme", std::string("direct_signed"),
                           da_options::solver) != da_status_success) {
            return da_error_bypass(
                this->err, da_status_internal_error,
                "Unexpected error while updating quantization_scheme.");
        }
        iquantization_scheme = static_cast<int>(scalar_quantizer::direct_signed);
    }

    this->order = da_order(iorder);
    this->metric = approx_nn_metric(imetric);
    this->internal_metric = (this->metric == approx_nn_metric::euclidean)
                                ? approx_nn_metric::sqeuclidean
                                : this->metric;

    this->quantize_residual = (iquantization_residual != 0);
    this->sq = scalar_quantizer::scalar_quantizer<T>(
        scalar_quantizer::scheme(iquantization_scheme), quantization);

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

    // Select and default-construct the typed variant alternative. The list vector is
    // still empty here; its allocation is handled with the other allocations below.
    switch (this->sq.get_code_kind()) {
    case scalar_quantizer::none:
        this->indexed_storage = dense_indexed_lists<T>{};
        break;
    case scalar_quantizer::int8:
        this->indexed_storage = dense_indexed_lists<int8_t>{};
        break;
    case scalar_quantizer::bfloat16:
        this->indexed_storage = dense_indexed_lists<da_bf16::storage_type>{};
        break;
    case scalar_quantizer::int4:
        this->indexed_storage = dense_indexed_lists<uint8_t>{};
        break;
    default:
        return da_error(this->err, da_status_invalid_input,
                        "No indexed storage supports the selected code format.");
    }

    // We can now allocate memory for centroids, list_sizes and global_indices
    // as we have read n_list in read_training_options
    // Note use of assign and not resize as n_list may change between train calls
    try {
        this->list_sizes.assign(n_list, 0);
        this->old_list_sizes.assign(n_list, 0);
        this->centroids.assign(
            static_cast<size_t>(n_list) * static_cast<size_t>(n_features), 0.0);

        // visit passes the active typed storage to the lambda, so common operations
        // do not need another format switch or std::get.
        std::visit([&](auto &storage) { storage.lists.resize(n_list); },
                   this->indexed_storage);

        this->list_norms.resize(n_list);
        this->global_indices.resize(n_list);
        for (da_int i = 0; i < n_list; i++) {
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

            da_utils::copy_array_convert_precision(
                this->order, this->n_samples_train, this->n_features, this->X_train,
                this->ldx_train, X_train_work.data(), ldx_train_work);
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

    // Train the scalar quantizer so each target component is reconstructed as
    // target[d] ~= bias[d] + scale[d] * code[d]. Normally target is x; with residual
    // encoding it is x - centroid[label]. Reuse the k-means labels so the quantizer
    // learns the values it will encode without recomputing full-precision assignments.
    if (this->quantize_db()) {
        const T *sq_train_ptr = train_ptr;
        da_int sq_ld = ld_train;
        std::vector<T> residuals;

        if (this->quantize_residual) {
            std::vector<da_int> train_labels;
            try {
                train_labels.resize(this->n_samples_train);
                residuals.resize(static_cast<size_t>(this->n_samples_train) *
                                 static_cast<size_t>(this->n_features));
            } catch (std::bad_alloc const &) {
                return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                                "Memory allocation failed.");
            }

            da_int labels_size = this->n_samples_train;
            status = kmeans_model.get_result(da_result::da_kmeans_labels, &labels_size,
                                             train_labels.data());
            if (status != da_status_success)
                return status;

            da_int ld_res =
                (this->order == column_major) ? this->n_samples_train : this->n_features;
            if (this->order == column_major) {
                for (da_int j = 0; j < this->n_features; j++) {
                    for (da_int i = 0; i < this->n_samples_train; i++) {
                        da_int c = train_labels[i];
                        residuals[i + static_cast<size_t>(j) * ld_res] =
                            train_ptr[i + static_cast<size_t>(j) * ld_train] -
                            this->centroids[c +
                                            static_cast<size_t>(j) * this->ld_centroids];
                    }
                }
            } else {
                for (da_int i = 0; i < this->n_samples_train; i++) {
                    da_int c = train_labels[i];
                    for (da_int j = 0; j < this->n_features; j++) {
                        residuals[j + static_cast<size_t>(i) * ld_res] =
                            train_ptr[j + static_cast<size_t>(i) * ld_train] -
                            this->centroids[j +
                                            static_cast<size_t>(c) * this->ld_centroids];
                    }
                }
            }
            sq_train_ptr = residuals.data();
            sq_ld = ld_res;
        }

        status = this->sq.train(this->order, sq_train_ptr, this->n_samples_train,
                                this->n_features, sq_ld);
        if (status != da_status_success) {
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }
        for (da_int d = 0; d < this->n_features; d++) {
            if (!std::isfinite(this->sq.scale()[d]) || !std::isfinite(this->sq.bias()[d]))
                return da_error(this->err, da_status_numerical_difficulties,
                                "Scalar quantizer training produced non-finite "
                                "parameters.");
        }
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

    // Retraining replaces the centroids and indexed lists, so previously added
    // samples no longer belong to the model being built.
    this->model_trained = false;
    this->data_is_added = false;
    this->n_index = 0;

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
    4. Store the assigned rows in the active native or quantized list representation.
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

                const T *X_blk =
                    (this->order == column_major)
                        ? X_add_ptr + blk_start
                        : X_add_ptr + static_cast<size_t>(blk_start) * ldx_add_ptr;

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
                    const T *row = &thread_dists[static_cast<size_t>(i) * n_list];
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

    bool is_euclidean = (this->internal_metric == approx_nn_metric::sqeuclidean);
    da_status storage_status;
    switch (this->sq.get_code_kind()) {
    case scalar_quantizer::none: {
        // if indexed_storage contains dense_indexed_lists<T>, get a pointer to it
        // otherwise return nullptr, and so on for other variants
        auto *storage = std::get_if<dense_indexed_lists<T>>(&this->indexed_storage);
        if (!storage)
            return da_error(this->err, da_status_internal_error,
                            "Native indexed storage is not active.");
        storage_status = this->store_unquantized_rows(
            *storage, local_indices, X_add_ptr, ldx_add_ptr, n_threads, is_euclidean);
        break;
    }
    case scalar_quantizer::int8: {
        auto *storage = std::get_if<dense_indexed_lists<int8_t>>(&this->indexed_storage);
        if (!storage)
            return da_error(this->err, da_status_internal_error,
                            "Int8 indexed storage is not active.");
        storage_status = this->store_quantized_rows(*storage, local_indices, X_add_ptr,
                                                    ldx_add_ptr, n_threads, is_euclidean);
        break;
    }
    case scalar_quantizer::bfloat16: {
        auto *storage = std::get_if<dense_indexed_lists<da_bf16::storage_type>>(
            &this->indexed_storage);
        if (!storage)
            return da_error(this->err, da_status_internal_error,
                            "bfloat16 indexed storage is not active.");
        storage_status = this->store_quantized_rows(*storage, local_indices, X_add_ptr,
                                                    ldx_add_ptr, n_threads, is_euclidean);
        break;
    }
    case scalar_quantizer::int4: {
        auto *storage = std::get_if<dense_indexed_lists<uint8_t>>(&this->indexed_storage);
        if (!storage)
            return da_error(this->err, da_status_internal_error,
                            "Int4 indexed storage is not active.");
        storage_status = this->store_quantized_rows(*storage, local_indices, X_add_ptr,
                                                    ldx_add_ptr, n_threads, is_euclidean);
        break;
    }
    default:
        return da_error(this->err, da_status_invalid_input,
                        "No indexed storage supports the selected code format.");
    }
    if (storage_status != da_status_success)
        return storage_status;

    this->n_index += n_samples_add;
    this->data_is_added = true;
    return da_status_success;
}

template <typename T>
da_status approximate_neighbors<T>::store_unquantized_rows(
    dense_indexed_lists<T> &storage,
    const std::vector<da_vector::da_vector<da_int>> &local_indices, const T *X_add,
    da_int ldx_add, [[maybe_unused]] da_int n_threads, bool is_euclidean) {
    try {
        for (da_int list = 0; list < this->n_list; list++) {
            storage.lists[list].resize(static_cast<size_t>(this->list_sizes[list]) *
                                       static_cast<size_t>(this->n_features));
            if (is_euclidean)
                this->list_norms[list].resize(this->list_sizes[list]);
        }
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

#pragma omp parallel for num_threads(n_threads) schedule(dynamic) default(none)          \
    shared(storage, local_indices, X_add, ldx_add, is_euclidean)
    for (da_int list = 0; list < this->n_list; list++) {
        const da_int old_size = this->old_list_sizes[list];
        const da_int new_size = this->list_sizes[list];
        const da_int *indices_to_add = local_indices[list].data();
        const da_int src_stride = (this->order == column_major) ? ldx_add : 1;
        T *destination = storage.lists[list].data();
        T *norms = is_euclidean ? this->list_norms[list].data() : nullptr;

        for (da_int row = old_size; row < new_size; row++) {
            da_int source_row = indices_to_add[row - old_size];
            const T *source = (this->order == column_major)
                                  ? X_add + source_row
                                  : X_add + static_cast<size_t>(source_row) * ldx_add;
            T *output = destination + static_cast<size_t>(row) * this->n_features;
            T norm = T(0);
            for (da_int feature = 0; feature < this->n_features; feature++) {
                T value = source[static_cast<size_t>(feature) * src_stride];
                output[feature] = value;
                norm += value * value;
            }
            if (is_euclidean)
                norms[row] = norm;
        }
    }
    return da_status_success;
}

template <typename T>
template <typename Q>
da_status approximate_neighbors<T>::store_quantized_rows(
    dense_indexed_lists<Q> &storage,
    const std::vector<da_vector::da_vector<da_int>> &local_indices, const T *X_add,
    da_int ldx_add, da_int n_threads, bool is_euclidean) {
    // local_indices contains list assignments already computed in add_ivfflat from
    // full-precision vector-to-centroid scores. The quantizer was trained in
    // train_ivfflat(); here we only encode x, or x - centroid[list] in residual mode.
    const da_int stored_row_width = [](da_int n_features) {
        if constexpr (std::is_same_v<Q, uint8_t>)
            return static_cast<da_int>(
                scalar_quantizer::int4_packing::packed_size(n_features));
        else
            return n_features;
    }(this->n_features);
    std::vector<T> row_workspace;
    try {
        row_workspace.resize(static_cast<size_t>(n_threads) * this->n_features);
        for (da_int list = 0; list < this->n_list; list++) {
            storage.lists[list].resize(static_cast<size_t>(this->list_sizes[list]) *
                                       static_cast<size_t>(stored_row_width));
            if (is_euclidean)
                this->list_norms[list].resize(this->list_sizes[list]);
        }
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

#pragma omp parallel for num_threads(n_threads) schedule(dynamic) default(none)          \
    shared(storage, local_indices, X_add, ldx_add, is_euclidean, row_workspace,          \
               stored_row_width)
    for (da_int list = 0; list < this->n_list; list++) {
        const da_int old_size = this->old_list_sizes[list];
        const da_int new_size = this->list_sizes[list];
        const da_int *indices_to_add = local_indices[list].data();
        const da_int src_stride = (this->order == column_major) ? ldx_add : 1;
        const da_int centroid_stride =
            (this->order == column_major) ? this->ld_centroids : 1;
        const T *centroid =
            (this->order == column_major)
                ? this->centroids.data() + list
                : this->centroids.data() + static_cast<size_t>(list) * this->ld_centroids;
        Q *destination = storage.lists[list].data();
        T *norms = is_euclidean ? this->list_norms[list].data() : nullptr;
        T *encode_row = row_workspace.data() +
                        static_cast<size_t>(omp_get_thread_num()) * this->n_features;

        for (da_int row = old_size; row < new_size; row++) {
            da_int source_row = indices_to_add[row - old_size];
            const T *source = (this->order == column_major)
                                  ? X_add + source_row
                                  : X_add + static_cast<size_t>(source_row) * ldx_add;
            for (da_int feature = 0; feature < this->n_features; feature++) {
                encode_row[feature] = source[static_cast<size_t>(feature) * src_stride];
                if (this->quantize_residual)
                    encode_row[feature] -=
                        centroid[static_cast<size_t>(feature) * centroid_stride];
            }

            Q *codes = destination + static_cast<size_t>(row) * stored_row_width;
            this->sq.quantize(row_major, encode_row, 1, this->n_features,
                              this->n_features, codes, stored_row_width);
            if (is_euclidean)
                norms[row] = this->reconstructed_squared_norm(
                    codes, this->quantize_residual ? centroid : nullptr, centroid_stride);
        }
    }
    return da_status_success;
}

// Return ||x_hat||^2 for the vector represented by the codes, where
// x_hat = bias + scale * code (+ centroid for residual encoding). Fine Euclidean
// search uses ||q - x_hat||^2, so using the original ||x||^2 would be inconsistent
// with its approximate reconstructed dot product.
template <typename T>
template <typename Q>
T approximate_neighbors<T>::reconstructed_squared_norm(const Q *codes, const T *centroid,
                                                       da_int centroid_stride) {
    const auto &scale = this->sq.scale();
    const auto &bias = this->sq.bias();
    T norm = T(0);
    for (da_int feature = 0; feature < n_features; feature++) {
        T code_value;
        if constexpr (std::is_same_v<Q, uint8_t>) {
            code_value =
                static_cast<T>(scalar_quantizer::int4_packing::load(codes, feature));
        } else {
            code_value = static_cast<T>(codes[feature]);
        }
        if constexpr (std::is_same_v<Q, da_bf16::storage_type>) {
            if (this->sq.get_code_kind() == scalar_quantizer::bfloat16)
                code_value = static_cast<T>(da_bf16::to_float(codes[feature]));
        }
        T value = bias[feature] + scale[feature] * code_value;
        if (centroid)
            value += centroid[static_cast<size_t>(feature) * centroid_stride];
        norm += value * value;
    }
    return norm;
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
        const T *list_blk_dists =
            fine_distances + static_cast<size_t>(q) * this_list_blk_sz;
        for (da_int v = 0; v < this_list_blk_sz; v++) {
            if (list_blk_dists[v] < max_dist) {
                heap.Insert(list_blk_global_idx[v], list_blk_dists[v]);
                max_dist = heap.GetMaxDist();
            }
        }
    }
}

// Scan one quantized IVF list. process_block() prepares the backend-specific query
// representation once for the full query block. Here
// Backend::gather_list_queries() packs the prepared rows that probe this list and
// Backend::compute_products() forms their products with the encoded list vectors.
//
// A database vector x is encoded either directly or as a residual from this list's
// centroid:
//
//   encoded_value = x                    without residual encoding
//   encoded_value = x - centroid[j]      with residual encoding
//
// Scalar quantization represents each encoded component approximately as
//
//   encoded_value[d] = scale[d] * code[d] + bias[d].
//
// For a query q, Backend::compute_products() calculates the code-dependent part of the
// dot
// product:
//
//   code_product(q, code) = sum_d (q[d] * scale[d]) * code[d].
//
// process_block() separately calculates the code-independent bias contribution
//
//   bias_dot_product(q) = sum_d q[d] * bias[d].
//
// Their sum approximates dot(q, encoded_value). With residual encoding,
// dot(q, centroid[j]) must also be included. Terms shared by every vector in this list
// are collected once per query in distance_offsets. Each vector's internal distance is
// then:
//
//   squared Euclidean: distance_offset - 2 * code_product + squared_norm(x)
//   inner product/cosine: distance_offset - code_product
//
// Inner-product and cosine search use negative dot product internally, allowing all
// metrics to share the same min-distance heap logic.
template <typename T>
template <typename Backend>
bool approximate_neighbors<T>::process_list_quantized(
    search_workspace<Backend> &ws, const search_params &p,
    const dense_indexed_lists<typename Backend::stored_type> &storage, da_int j,
    da_int block_start) {
    static_assert(Backend::is_quantized);

    da_int list_size = this->list_sizes[j];
    const da_int *this_list_idx = this->global_indices[j].data();
    const da_int *qpc_j = ws.qpc + static_cast<size_t>(j) * p.query_blk_sz;
    const da_int q_count = ws.qpc_count[j];

    if (list_size == 0 || q_count == 0)
        return true;

    const typename Backend::Q *this_list_codes = storage.lists[j].data();
    const T *this_list_norms = p.is_euclidean ? this->list_norms[j].data() : nullptr;
    const T dot_multiplier = p.is_euclidean ? T(2.0) : T(1.0);

    Backend::gather_list_queries(*ws.quantized, qpc_j, block_start, q_count,
                                 p.n_features);

    // Compute the query/list terms that are shared by every vector in this list.
    T *distance_offsets = ws.distance_offsets;
    for (da_int query_idx = 0; query_idx < q_count; query_idx++) {
        da_int block_query_idx = qpc_j[query_idx] - block_start;
        T base_distance;
        if (this->quantize_residual) {
            T coarse_distance =
                ws.coarse_dist[static_cast<size_t>(block_query_idx) * this->n_list + j];
            base_distance = p.is_euclidean ? (coarse_distance - this->centroid_norms[j])
                                           : coarse_distance;
        } else {
            base_distance = p.is_euclidean ? ws.qnorms[block_query_idx] : T(0.0);
        }
        distance_offsets[query_idx] =
            base_distance - dot_multiplier * ws.bias_dot_products[block_query_idx];
    }

    // For each list block, compute fine distances.
    // Perform per-query adjustments, and pass to heap updates
    for (da_int t = 0; t < list_size; t += p.list_blk_sz) {
        da_int this_list_blk_sz = std::min(p.list_blk_sz, list_size - t);
        if (!Backend::compute_products(*ws.quantized,
                                       this_list_codes +
                                           Backend::stored_row_offset(t, p.n_features),
                                       q_count, this_list_blk_sz, p.n_features))
            return false;

        for (da_int query_idx = 0; query_idx < q_count; query_idx++) {
            const typename Backend::T *query_dot_products =
                ws.quantized->dot_products.data() +
                static_cast<size_t>(query_idx) * static_cast<size_t>(this_list_blk_sz);
            T *fine_distances =
                ws.fine_distances +
                static_cast<size_t>(query_idx) * static_cast<size_t>(this_list_blk_sz);
            T distance_offset = distance_offsets[query_idx];
#pragma omp simd
            for (da_int vector_idx = 0; vector_idx < this_list_blk_sz; vector_idx++) {
                T distance =
                    distance_offset -
                    dot_multiplier * static_cast<T>(query_dot_products[vector_idx]);
                if (p.is_euclidean) {
                    distance += this_list_norms[t + vector_idx];
                    distance = std::max(distance, T(0));
                }
                fine_distances[vector_idx] = distance;
            }
        }
        update_heaps_from_list_blk(this_list_blk_sz, q_count, this_list_idx + t,
                                   ws.fine_distances, block_start, qpc_j, ws.heaps);
    }
    return true;
}

// Process one IVF list for the current query block: gather the queries that probe it,
// compute fine query-vector distances via blocked GEMM, and update the per-query heaps.
template <typename T>
template <typename Backend>
bool approximate_neighbors<T>::process_list(
    search_workspace<Backend> &ws, const search_params &p,
    const dense_indexed_lists<typename Backend::stored_type> &storage, da_int j,
    da_int block_start) {
    if constexpr (Backend::is_quantized) {
        return process_list_quantized<Backend>(ws, p, storage, j, block_start);
    } else {

        T *fine_distances = ws.fine_distances;
        T *queries = ws.queries;
        T *query_cos = ws.query_cos;
        T *qnorms = ws.qnorms;
        auto heaps = ws.heaps;

        da_int list_size = this->list_sizes[j];
        const T *this_list = storage.lists[j].data();
        T *this_list_norms = this->list_norms[j].data();
        const da_int *this_list_idx = this->global_indices[j].data();
        const da_int *qpc_j = ws.qpc + static_cast<size_t>(j) * p.query_blk_sz;
        const da_int q_count = ws.qpc_count[j];

        if (list_size == 0 || q_count == 0)
            return true;

        // Gather queries
        for (da_int k = 0; k < q_count; k++) {
            const T *query_ptr =
                p.is_cosine ? query_cos + static_cast<size_t>(qpc_j[k] - block_start) *
                                              p.n_features
                            : p.X_test_ptr + static_cast<size_t>(qpc_j[k]) * p.ldx_test;
            memcpy(queries + static_cast<size_t>(k) * static_cast<size_t>(p.n_features),
                   query_ptr, static_cast<size_t>(p.n_features) * sizeof(T));
        }

        // For each block in list vectors, calculate distances and update heaps
        for (da_int t = 0; t < list_size; t += p.list_blk_sz) {
            da_int this_list_blk_sz = std::min(p.list_blk_sz, list_size - t);
            if (p.is_euclidean) {
                ARCH::euclidean_gemm_distance(
                    row_major, q_count, this_list_blk_sz, p.n_features, queries,
                    p.n_features, this_list + static_cast<size_t>(t) * p.n_features,
                    p.n_features, fine_distances, this_list_blk_sz, qnorms, 2,
                    this_list_norms + t, 1, true, false);
            } else {
                da_blas::cblas_gemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans, q_count, this_list_blk_sz,
                    p.n_features, static_cast<T>(-1.0), queries, p.n_features,
                    this_list + static_cast<size_t>(t) * p.n_features, p.n_features,
                    static_cast<T>(0.0), fine_distances, this_list_blk_sz);
            }
            update_heaps_from_list_blk(this_list_blk_sz, q_count, this_list_idx + t,
                                       fine_distances, block_start, qpc_j, heaps);
        }
        return true;
    }
}

// Process one query block: reset heaps, compute coarse query-centroid distances, select
// the n_probe nearest centroids per query, scan the probed lists, then write results.
template <typename T>
template <typename Backend>
bool approximate_neighbors<T>::process_block(
    search_workspace<Backend> &ws, const search_params &p,
    const dense_indexed_lists<typename Backend::stored_type> &storage, da_int i) {
    da_int n_list = this->n_list;
    da_int n_probe = this->n_probe;

    da_int *heap_indices = ws.heap_indices;
    T *heap_distances = ws.heap_distances;
    auto heaps = ws.heaps;
    T *query_cos = ws.query_cos;
    T *qnorms = ws.qnorms;
    T *coarse_dist = ws.coarse_dist;
    da_int *cent_idx = ws.cent_idx;
    T *cent_sel_dists = ws.cent_sel_dists;
    da_int *qpc = ws.qpc;
    da_int *qpc_count = ws.qpc_count;

    da_int this_query_blk_sz;
    if ((i == p.n_blocks - 1) && p.final_query_blk_sz > 0) {
        this_query_blk_sz = p.final_query_blk_sz;
    } else {
        this_query_blk_sz = p.query_blk_sz;
    }

    da_int block_start = i * p.query_blk_sz;
    da_int block_end = std::min((i + 1) * p.query_blk_sz, p.n_queries);

    // Reset heaps for this block
    da_std::fill(heap_indices,
                 heap_indices + static_cast<size_t>(p.k_neigh) * p.query_blk_sz, -1);
    da_std::fill(heap_distances,
                 heap_distances + static_cast<size_t>(p.k_neigh) * p.query_blk_sz,
                 std::numeric_limits<T>::infinity());
    for (da_int ii = 0; ii < p.query_blk_sz; ii++) {
        heaps[ii] = da_binary_tree::MaxHeap<T>(
            p.k_neigh, heap_indices + static_cast<size_t>(ii) * p.k_neigh,
            heap_distances + static_cast<size_t>(ii) * p.k_neigh);
    }

    const T *query_blk_ptr = p.X_test_ptr + static_cast<size_t>(block_start) * p.ldx_test;
    if (p.is_cosine) {
        for (da_int ii = 0; ii < this_query_blk_sz; ii++) {
            memcpy(
                query_cos + static_cast<size_t>(ii) * static_cast<size_t>(p.n_features),
                query_blk_ptr + static_cast<size_t>(ii) * static_cast<size_t>(p.ldx_test),
                static_cast<size_t>(p.n_features) * sizeof(T));
        }
        da_utils::normalize_rows_inplace(row_major, this_query_blk_sz, p.n_features,
                                         query_cos, p.n_features,
                                         static_cast<T *>(nullptr));
        query_blk_ptr = query_cos;
    }

    // Coarse GEMM: query-to-centroid distances for this thread's block
    da_std::fill(qpc_count, qpc_count + n_list, 0);

    // Calculate coarse distance for queries
    if (p.is_euclidean) {
        ARCH::euclidean_gemm_distance(
            row_major, this_query_blk_sz, n_list, p.n_features,
            p.X_test_ptr + static_cast<size_t>(block_start) * p.ldx_test, p.ldx_test,
            p.centroids_ptr, p.ld_centroids_local, coarse_dist, n_list, qnorms, 2,
            this->centroid_norms.data(), 1, true, false);
    } else {
        da_blas::cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, this_query_blk_sz,
                            n_list, p.n_features, static_cast<T>(-1.0), query_blk_ptr,
                            p.is_cosine ? p.n_features : p.ldx_test, p.centroids_ptr,
                            p.ld_centroids_local, static_cast<T>(0.0), coarse_dist,
                            n_list);
    }

    // For each query, find the n_probe nearest centroids.
    // Use a max-heap linear scan
    for (da_int q = block_start; q < block_end; q++) {
        const T *query_distances = coarse_dist + (q - block_start) * n_list;
        da_std::fill(cent_sel_dists, cent_sel_dists + n_probe,
                     std::numeric_limits<T>::max());
        da_binary_tree::MaxHeap<T> cent_heap(n_probe, cent_idx, cent_sel_dists);
        T max_cent_dist = cent_heap.GetMaxDist();
        for (da_int c = 0; c < n_list; c++) {
            T d = query_distances[c];
            if (d < max_cent_dist) {
                cent_heap.Insert(c, d);
                max_cent_dist = cent_heap.GetMaxDist();
            }
        }
        for (da_int pp = 0; pp < n_probe; pp++) {
            da_int c = cent_idx[pp];
            qpc[static_cast<size_t>(c) * p.query_blk_sz + qpc_count[c]] = q;
            qpc_count[c]++;
        }
    }

    // The database quantizer reconstructs one encoded component as
    //
    //   encoded_value[d] ~= scale[d] * code[d] + bias[d].
    //
    // Therefore dot(query, encoded_value) separates into a code-dependent term
    // dot(query * scale, code) and a shared bias term dot(query, bias). Build both
    // once for the query block; every probed list reuses them.
    if constexpr (Backend::is_quantized) {
        const T *scale = this->sq.scale().data();
        const T *bias = this->sq.bias().data();
        da_int ld_q = p.is_cosine ? p.n_features : p.ldx_test;
        for (da_int ii = 0; ii < this_query_blk_sz; ii++) {
            const T *row =
                query_blk_ptr + static_cast<size_t>(ii) * static_cast<size_t>(ld_q);
            T bias_dot_product = T(0.0);
            for (da_int d = 0; d < p.n_features; d++) {
                Backend::set_query_value(
                    *ws.quantized, static_cast<size_t>(ii) * p.n_features + d,
                    static_cast<typename Backend::T>(row[d] * scale[d]));
                bias_dot_product += row[d] * bias[d];
            }
            if (!std::isfinite(bias_dot_product))
                return false;
            ws.bias_dot_products[ii] = bias_dot_product;
        }
    }

    // Sequential loop over all lists for this thread's block
    for (da_int j = 0; j < n_list; j++) {
        if (!process_list<Backend>(ws, p, storage, j, block_start))
            return false;
    }

    // Write results directly — no merge needed, each thread owns its output rows
    for (da_int ii = 0; ii < this_query_blk_sz; ii++) {
        da_neighbors::sorted_n_dist_n_ind(
            p.k_neigh, heap_distances + static_cast<size_t>(ii) * p.k_neigh,
            heap_indices + static_cast<size_t>(ii) * p.k_neigh,
            p.n_dist + static_cast<size_t>(block_start + ii) * p.k_neigh,
            p.n_ind + static_cast<size_t>(block_start + ii) * p.k_neigh, ws.topk_indices,
            p.return_distance, false);
    }
    return true;
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
template <typename Backend>
da_status approximate_neighbors<T>::ivfflat_search_query_parallel_impl(
    const dense_indexed_lists<typename Backend::stored_type> &storage, da_int n_queries,
    da_int n_features, da_int k_neigh, bool return_distance, const T *X_test_ptr,
    da_int ldx_test, const T *centroids_ptr, da_int ld_centroids_local,
    da_int query_blk_sz, da_int list_blk_sz, da_int n_blocks, da_int final_query_blk_sz,
    [[maybe_unused]] da_int n_threads, da_int *n_ind, T *n_dist) {

    da_int n_list = this->n_list;
    da_int n_probe = this->n_probe;
    bool is_euclidean = this->internal_metric == approx_nn_metric::sqeuclidean;
    bool is_cosine = this->internal_metric == approx_nn_metric::cosine;
    da_int threading_error = 0;
    da_int numerical_error = 0;
    bool use_optimized_kernel = false;
    if constexpr (Backend::is_quantized) {
        const auto *ctx = context::get_context();
        use_optimized_kernel = da_quantized_backend::use_optimized_backend(
            Backend::code_kind, ctx->has_avx512_bf16);
    }

    const search_params p{n_queries,    n_features,  k_neigh,       return_distance,
                          X_test_ptr,   ldx_test,    centroids_ptr, ld_centroids_local,
                          query_blk_sz, list_blk_sz, n_blocks,      final_query_blk_sz,
                          is_euclidean, is_cosine,   n_dist,        n_ind};

#pragma omp parallel default(none) num_threads(n_threads)                                \
    shared(p, threading_error, numerical_error, n_blocks, query_blk_sz, n_list, n_probe, \
               k_neigh, list_blk_sz, n_features, is_cosine, is_euclidean, storage,       \
               use_optimized_kernel)
    {
#ifndef _WIN32
        // DLP thread-local settings are not inherited by OpenMP workers. Keep DLP
        // serial while AOCL-DA parallelizes query blocks across these workers.
        if constexpr (Backend::is_quantized &&
                      ((Backend::code_kind == scalar_quantizer::code_kind::int4) ||
                       (Backend::code_kind == scalar_quantizer::code_kind::bfloat16)))
            dlp_thread_set_num_threads_local(1);
#endif

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
        // Quantized product scratch is backend-owned; metric correction remains
        // in input precision.
        typename Backend::workspace_type quantized_workspace_buf;
        std::vector<T> bias_dot_products_buf, distance_offsets_buf;

        try {
            coarse_distances_buf.resize(static_cast<size_t>(query_blk_sz) * n_list);
            centroid_indices_buf.resize(n_probe);
            cent_sel_dists_buf.resize(n_probe);
            queries_per_centroid_buf.resize(static_cast<size_t>(n_list) * query_blk_sz);
            queries_per_centroid_cnt.resize(n_list, 0);
            heap_indices_buf.resize(static_cast<size_t>(query_blk_sz) * k_neigh, -1);
            local_heap_dists.resize(static_cast<size_t>(query_blk_sz) * k_neigh,
                                    std::numeric_limits<T>::infinity());
            heaps_buf.resize(query_blk_sz);
            fine_distances_buf.resize(static_cast<size_t>(query_blk_sz) * list_blk_sz,
                                      0.0);
            if (is_cosine)
                query_cos_buf.resize(static_cast<size_t>(query_blk_sz) * n_features, 0.0);
            if (is_euclidean)
                qnorms_buf.resize(query_blk_sz, 0.0);
            topk_indices_buf.resize(k_neigh, 0);
            if constexpr (Backend::is_quantized) {
                // Backend workspace allocation must remain inside this bad_alloc
                // boundary; list processing assumes its scratch is already reserved.
                quantized_workspace_buf.use_optimized_kernel = use_optimized_kernel;
                Backend::resize_workspace(quantized_workspace_buf, query_blk_sz,
                                          list_blk_sz, n_features);
                bias_dot_products_buf.resize(query_blk_sz, 0.0);
                distance_offsets_buf.resize(query_blk_sz, 0.0);
            } else {
                query_buf.resize(static_cast<size_t>(query_blk_sz) * n_features, 0.0);
            }
        } catch (std::bad_alloc const &) {
#pragma omp atomic write
            threading_error = 1;
        }

#pragma omp barrier

        typename Backend::workspace_type *quantized_workspace_ptr = nullptr;
        T *bias_dot_products_ptr = nullptr;
        T *distance_offsets_ptr = nullptr;
        if constexpr (Backend::is_quantized) {
            quantized_workspace_ptr = &quantized_workspace_buf;
            bias_dot_products_ptr = bias_dot_products_buf.data();
            distance_offsets_ptr = distance_offsets_buf.data();
        }

        search_workspace<Backend> ws{heap_indices_buf.data(),
                                     local_heap_dists.data(),
                                     heaps_buf.data(),
                                     fine_distances_buf.data(),
                                     query_buf.data(),
                                     is_cosine ? query_cos_buf.data() : nullptr,
                                     is_euclidean ? qnorms_buf.data() : nullptr,
                                     coarse_distances_buf.data(),
                                     centroid_indices_buf.data(),
                                     cent_sel_dists_buf.data(),
                                     queries_per_centroid_buf.data(),
                                     queries_per_centroid_cnt.data(),
                                     topk_indices_buf.data(),
                                     quantized_workspace_ptr,
                                     bias_dot_products_ptr,
                                     distance_offsets_ptr};

        if (!threading_error) {
#pragma omp for schedule(dynamic) nowait
            for (da_int i = 0; i < n_blocks; i++) {
                if (!process_block<Backend>(ws, p, storage, i)) {
#pragma omp atomic write
                    numerical_error = 1;
                }
            }
        } // if (!threading_error)
#ifndef _WIN32
        // Remove this worker's override and return to the inherited DLP policy.
        if constexpr (Backend::is_quantized &&
                      ((Backend::code_kind == scalar_quantizer::code_kind::int4) ||
                       (Backend::code_kind == scalar_quantizer::code_kind::bfloat16)))
            dlp_thread_set_num_threads_local(-1);
#endif
    } // pragma omp parallel

    if (threading_error)
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    if (numerical_error)
        return da_error(this->err, da_status_numerical_difficulties,
                        "Quantized query values cannot be represented by the selected "
                        "GEMM backend.");

    return da_status_success;
}

template <typename T>
da_status approximate_neighbors<T>::ivfflat_search_query_parallel(
    da_int n_queries, da_int n_features, da_int k_neigh, bool return_distance,
    const T *X_test_ptr, da_int ldx_test, const T *centroids_ptr,
    da_int ld_centroids_local, da_int query_blk_sz, da_int list_blk_sz, da_int n_blocks,
    da_int final_query_blk_sz, da_int n_threads, da_int *n_ind, T *n_dist) {
    using namespace da_quantized_backend;
    switch (this->sq.get_code_kind()) {
    case scalar_quantizer::code_kind::none: {
        const auto *storage = std::get_if<dense_indexed_lists<T>>(&this->indexed_storage);
        if (!storage)
            return da_error(this->err, da_status_internal_error,
                            "Native indexed storage is not active.");
        return ivfflat_search_query_parallel_impl<unquantized_backend<T>>(
            *storage, n_queries, n_features, k_neigh, return_distance, X_test_ptr,
            ldx_test, centroids_ptr, ld_centroids_local, query_blk_sz, list_blk_sz,
            n_blocks, final_query_blk_sz, n_threads, n_ind, n_dist);
    }
    case scalar_quantizer::code_kind::int8: {
        const auto *storage =
            std::get_if<dense_indexed_lists<int8_t>>(&this->indexed_storage);
        if (!storage)
            return da_error(this->err, da_status_internal_error,
                            "Int8 indexed storage is not active.");
        return ivfflat_search_query_parallel_impl<backend<int8_t>>(
            *storage, n_queries, n_features, k_neigh, return_distance, X_test_ptr,
            ldx_test, centroids_ptr, ld_centroids_local, query_blk_sz, list_blk_sz,
            n_blocks, final_query_blk_sz, n_threads, n_ind, n_dist);
    }
    case scalar_quantizer::code_kind::bfloat16: {
        const auto *storage = std::get_if<dense_indexed_lists<da_bf16::storage_type>>(
            &this->indexed_storage);
        if (!storage)
            return da_error(this->err, da_status_internal_error,
                            "bfloat16 indexed storage is not active.");
        return ivfflat_search_query_parallel_impl<backend<da_bf16::storage_type>>(
            *storage, n_queries, n_features, k_neigh, return_distance, X_test_ptr,
            ldx_test, centroids_ptr, ld_centroids_local, query_blk_sz, list_blk_sz,
            n_blocks, final_query_blk_sz, n_threads, n_ind, n_dist);
    }
    case scalar_quantizer::code_kind::int4: {
        const auto *storage =
            std::get_if<dense_indexed_lists<uint8_t>>(&this->indexed_storage);
        if (!storage)
            return da_error(this->err, da_status_internal_error,
                            "Int4 indexed storage is not active.");
        return ivfflat_search_query_parallel_impl<backend<uint8_t>>(
            *storage, n_queries, n_features, k_neigh, return_distance, X_test_ptr,
            ldx_test, centroids_ptr, ld_centroids_local, query_blk_sz, list_blk_sz,
            n_blocks, final_query_blk_sz, n_threads, n_ind, n_dist);
    }
    default:
        return da_error(this->err, da_status_invalid_input,
                        "No quantized GEMM backend supports the selected code format.");
    }
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
            X_test_copy.resize(static_cast<size_t>(n_queries) * n_features);
            centroids_copy.resize(static_cast<size_t>(n_list) * n_features);
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
                    T val = n_dist[static_cast<size_t>(i) * k_neigh + j];
                    n_dist[static_cast<size_t>(i) * k_neigh + j] =
                        (val < 0) ? val : std::sqrt(val);
                }
            }
            break;
        }
        case approx_nn_metric::cosine: {
#pragma omp simd
            for (da_int query = 0; query < n_queries; query++) {
                for (da_int neighbor = 0; neighbor < k_neigh; neighbor++) {
                    n_dist[static_cast<size_t>(query) * k_neigh + neighbor] =
                        static_cast<T>(1.0) +
                        n_dist[static_cast<size_t>(query) * k_neigh + neighbor];
                }
            }
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
    io_dispatch(this->list_sizes);
    io_dispatch(this->old_list_sizes);
    io_dispatch(this->quantize_residual);

    if (status != da_status_success)
        return status;

    status = this->sq.serialize(buffer);
    if (status != da_status_success)
        return status;

    const da_status invalid_storage = buffer.get_mode() == deserialize
                                          ? da_status_invalid_file_data
                                          : da_status_internal_error;
    switch (this->sq.get_code_kind()) {
    case scalar_quantizer::none: {
        if (buffer.get_mode() == deserialize)
            this->indexed_storage = dense_indexed_lists<T>{};
        auto *storage = std::get_if<dense_indexed_lists<T>>(&this->indexed_storage);
        if (!storage)
            return invalid_storage;
        io_dispatch(storage->lists);
        break;
    }
    case scalar_quantizer::int8: {
        if (buffer.get_mode() == deserialize)
            this->indexed_storage = dense_indexed_lists<int8_t>{};
        auto *storage = std::get_if<dense_indexed_lists<int8_t>>(&this->indexed_storage);
        if (!storage)
            return invalid_storage;
        io_dispatch(storage->lists);
        break;
    }
    case scalar_quantizer::bfloat16: {
        if (buffer.get_mode() == deserialize)
            this->indexed_storage = dense_indexed_lists<da_bf16::storage_type>{};
        auto *storage = std::get_if<dense_indexed_lists<da_bf16::storage_type>>(
            &this->indexed_storage);
        if (!storage)
            return invalid_storage;
        io_dispatch(storage->lists);
        break;
    }
    case scalar_quantizer::int4: {
        if (buffer.get_mode() == deserialize)
            this->indexed_storage = dense_indexed_lists<uint8_t>{};
        auto *storage = std::get_if<dense_indexed_lists<uint8_t>>(&this->indexed_storage);
        if (!storage)
            return invalid_storage;
        io_dispatch(storage->lists);
        break;
    }
    default:
        return invalid_storage;
    }
    if (status != da_status_success)
        return status;

    io_dispatch(this->global_indices);
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

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
