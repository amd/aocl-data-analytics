/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "kmeans.hpp"
#include "kmeans_elkan.hpp"
#include "kmeans_hartigan_wong.hpp"
#include "kmeans_lloyd.hpp"
#include "kmeans_macqueen.hpp"
#include "kmeans_options.hpp"
#include "kmeans_types.hpp"
#include "macros.h"
#include "miscellaneous.hpp"
#include "model_persistence.hpp"
#include "pairwise_distances.hpp"
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>

namespace ARCH {

namespace da_kmeans {

using namespace da_model_persistence;
using namespace da_kmeans_types;
using namespace std::literals::string_literals;

template <typename T> inline T safe_inv_sqrt(T x) {
    return (x > (T)0.0) ? (T)1.0 / std::sqrt(x) : (T)0.0;
}

template <typename T> inline T clamp_cosine(T x) {
    return std::max((T)-1.0, std::min((T)1.0, x));
}

// Convert dot products to cosine distances (column-major layout)
// inv_norms[i] is the inverse norm for sample i (or 1.0 if data is pre-normalized)
template <typename T>
inline void dot_to_cosine_distance_colmaj(da_int m_samples, da_int n_clusters,
                                          T *X_transform, da_int ldx_transform,
                                          const T *inv_norms) {
    for (da_int j = 0; j < n_clusters; j++) {
        da_int idx = j * ldx_transform;
        for (da_int i = 0; i < m_samples; i++) {
            T &d = X_transform[i + idx];
            d = (T)1.0 - clamp_cosine(d * inv_norms[i]);
        }
    }
}

// Convert dot products to cosine distances (row-major layout)
template <typename T>
inline void dot_to_cosine_distance_rowmaj(da_int m_samples, da_int n_clusters,
                                          T *X_transform, da_int ldx_transform,
                                          const T *inv_norms) {
    for (da_int i = 0; i < m_samples; i++) {
        da_int idx = i * ldx_transform;
        T inv_norm = inv_norms[i];
        for (da_int j = 0; j < n_clusters; j++) {
            T &d = X_transform[idx + j];
            d = (T)1.0 - clamp_cosine(d * inv_norm);
        }
    }
}

template <typename T> kmeans<T>::~kmeans() {
    // Destructor needs to handle arrays that were allocated due to row major storage of input data
    if (C_temp)
        delete[] (C_temp);
    if (A_temp)
        delete[] (A_temp);
}

template <typename T>
kmeans<T>::kmeans(da_errors::da_error_t &err) : basic_handle<T>(err) {
    // Initialize the options registry
    // Any error is stored err->status[.] and this needs to be checked
    // by the caller.
    register_kmeans_options<T>(this->opts, *this->err);
};

template <typename T>
kmeans<T>::kmeans(da_errors::da_error_t &err, da_order A_order, da_order order,
                  da_int algorithm, da_int init_method, da_int seed, T tol,
                  da_int max_iter, da_int n_samples, da_int n_features, da_int n_clusters,
                  da_int n_init, const T *A, da_int lda, const T *A_usr, da_int lda_usr,
                  const T *C, da_int ldc, bool initdone, bool centres_supplied,
                  bool use_mixed_precision, da_int empty_cluster_handling,
                  da_int afk_mcmc_samples, bool do_spherical, bool normalize_data)
    : basic_handle<T>(err), n_samples(n_samples), n_features(n_features),
      initdone(initdone), centres_supplied(centres_supplied), algorithm(algorithm),
      init_method(init_method), empty_cluster_handling(empty_cluster_handling),
      n_clusters(n_clusters), n_init(n_init), max_iter(max_iter), A_order(A_order),
      afk_mcmc_samples(afk_mcmc_samples), tol(tol),
      use_mixed_precision(use_mixed_precision), seed(seed), do_spherical(do_spherical),
      normalize_data(normalize_data), A_usr(A_usr), A(A), C(C), lda(lda),
      lda_usr(lda_usr), ldc(ldc) {
    this->order = order;
    // We have already set all options through the constructor, so skip checking them again in compute()
    this->check_options = false;
};

template <typename T>
da_status kmeans<T>::get_result(da_result query, da_int *dim, T *result) {
    // Don't return anything if k-means has not been computed
    if (!this->model_trained) {
        return da_warn(this->err, da_status_no_data,
                       "k-means clustering has not yet been computed. Please call "
                       "da_kmeans_compute_s "
                       "or da_kmeans_compute_d before extracting results.");
    }

    da_int rinfo_size = 6;

    switch (query) {
    case da_result::da_rinfo:
        if (*dim < rinfo_size) {
            *dim = rinfo_size;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(rinfo_size) + ".");
        }
        result[0] = (T)n_samples;
        result[1] = (T)n_features;
        result[2] = (T)n_clusters;
        result[3] = (T)best_n_iter;
        result[4] = best_inertia;
        result[5] = (T)best_lp_n_iter;
        break;
    case da_result::da_kmeans_cluster_centres:
        if (*dim < n_clusters * n_features) {
            *dim = n_clusters * n_features;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(n_clusters * n_features) + ".");
        }
        this->copy_2D_results_array(n_clusters, n_features,
                                    (*best_cluster_centres).data(), n_clusters, result);
        break;
    default:
        return da_warn(this->err, da_status_unknown_query,
                       "The requested result could not be found.");
    }
    return da_status_success;
};

template <typename T>
da_status kmeans<T>::get_result(da_result query, da_int *dim, da_int *result) {
    // check to see if user needs common stuff from the basic handle first
    da_status status = this->get_result_common(query, dim, result);
    if (status != da_status_unknown_query) {
        return status; // either got requested info or error
    }
    // Don't return anything if k-means clustering has not been computed
    if (!this->model_trained) {
        return da_warn(this->err, da_status_no_data,
                       "k-means clustering has not yet been computed. Please call "
                       "da_kmeans_compute_s "
                       "or da_kmeans_compute_d before extracting results.");
    }

    switch (query) {
    case da_result::da_kmeans_labels:
        if (*dim < n_samples) {
            *dim = n_samples;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(n_samples) + ".");
        }
        for (da_int i = 0; i < n_samples; i++)
            result[i] = (*best_labels)[i];
        break;
    default:
        return da_warn(this->err, da_status_unknown_query,
                       "The requested result could not be found.");
    }

    return da_status_success;
};

template <typename T> void kmeans<T>::refresh() {

    // Reset internal class variables to their defaults
    best_n_iter = 0;
    current_n_iter = 0;
    warn_maxit_reached = false;
    converged = 0;
    normc = 0.0;
    max_block_size = 0;
    n_blocks = 0;
    block_rem = 0;
    ldworkcs1 = 0;
    best_inertia = 0.0;
    current_inertia = 0.0;
    padding = 0;
    lp_n_iter = 0;
    best_lp_n_iter = 0;
    empty_cluster_found = false;
}

/* Store details about user's data matrix in preparation for k-means computation */
template <typename T>
da_status kmeans<T>::set_data(da_int n_samples, da_int n_features, const T *A_in,
                              da_int lda_in) {

    // Guard against errors due to multiple calls using the same class instantiation
    this->refresh();

    // Read in data storage option
    std::string opt_order;
    da_int iorder;
    da_status status = this->opts.get("storage order", opt_order, iorder);
    this->order = da_order(iorder);

    // Check for illegal arguments
    status = this->check_2D_array(this->order, n_samples, n_features, A_in, lda_in,
                                  "n_samples", "n_features", "A", "lda", 1, 1);
    if (status != da_status_success)
        return status;

    // Store dimensions of A and pointer to user's data
    this->lda_usr = lda_in;
    this->A_usr = A_in;
    this->n_samples = n_samples;
    this->n_features = n_features;

    // Record that initialization is complete but computation has not yet been performed
    initdone = true;
    this->model_trained = false;

    // Now that we have a data matrix we can re-register the n_clusters option with new constraints
    da_int temp_clusters;
    this->opts.get("n_clusters", temp_clusters);

    reregister_kmeans_option<T>(this->opts, n_samples);

    this->opts.set("n_clusters", std::min(temp_clusters, n_samples));

    if (temp_clusters > n_samples)
        return da_warn(this->err, da_status_incompatible_options,
                       "The requested number of clusters has been decreased from " +
                           std::to_string(temp_clusters) + " to " +
                           std::to_string(n_samples) +
                           " due to the size of the data array.");

    return da_status_success;
}

template <typename T>
da_status kmeans<T>::set_init_centres(const T *C_in, da_int ldc_in) {

    if (initdone == false)
        return da_error(this->err, da_status_no_data,
                        "No data has been passed to the handle. Please call "
                        "da_kmeans_set_data_s or da_kmeans_set_data_d.");

    // Guard against errors due to multiple calls using the same class instantiation
    if (C_temp) {
        delete[] (C_temp);
        C_temp = nullptr;
    }

    // Check options weren't changed to incompatible values after data was set
    std::string opt_order;
    da_int iorder;
    this->opts.get("storage order", opt_order, iorder);
    if (this->order != da_order(iorder)) {
        return da_error(this->err, da_status_incompatible_options,
                        "The storage order option was changed after data was set.");
    }

    // Check for illegal arguments
    this->opts.get("n_clusters", n_clusters);
    da_status status =
        this->check_2D_array(this->order, n_clusters, n_features, C_in, ldc_in,
                             "n_clusters", "n_features", "C", "ldc", 1, 1);
    if (status != da_status_success)
        return status;

    // We'll always store C in column-major format internally
    if (this->order == row_major) {
        // Transpose C to column-major
        try {
            C_temp = new T[(size_t)n_clusters * (size_t)n_features];
            ARCH::da_utils::copy_transpose_2D_array_row_to_column_major<T>(
                n_clusters, n_features, C_in, ldc_in, C_temp, n_clusters);
            C = C_temp;
            ldc = n_clusters;
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }
    } else {
        C = C_in;
        ldc = ldc_in;
    }

    // Record that centres have been set
    centres_supplied = true;

    return da_status_success;
}

/* Compute the k-means clusters */
template <typename T> da_status kmeans<T>::compute() {

    da_status status = da_status_success;
    if (initdone == false)
        return da_error(this->err, da_status_no_data,
                        "No data has been passed to the handle. Please call "
                        "da_kmeans_set_data_s or da_kmeans_set_data_d.");

    if (check_options) {

        // Check options weren't changed to incompatible values after data was set
        std::string opt_order;
        da_int iorder;
        this->opts.get("storage order", opt_order, iorder);
        if (this->order != da_order(iorder)) {
            return da_error(this->err, da_status_incompatible_options,
                            "The storage order option was changed after data was set.");
        }

        // Read in other options and store in class
        this->opts.get("n_clusters", n_clusters);

        std::string opt_method;
        this->opts.get("initialization method", opt_method, init_method);

        this->opts.get("n_init", n_init);

        this->opts.get("max_iter", max_iter);

        this->opts.get("convergence tolerance", tol);

        this->opts.get("seed", seed);

        this->opts.get("afk-mc2 samples", afk_mcmc_samples);

        std::string opt_alg;
        this->opts.get("algorithm", opt_alg, this->algorithm);

        std::string opt_mp;
        da_int int_mp;
        this->opts.get("mixed precision", opt_mp, int_mp);
        this->use_mixed_precision = (int_mp == 1);

        std::string opt_ec;
        this->opts.get("empty clusters", opt_ec, this->empty_cluster_handling);

        std::string opt_dist;
        da_int int_dist;
        this->opts.get("distance", opt_dist, int_dist);
        this->do_spherical = (int_dist == 1);

        std::string opt_norm;
        da_int int_norm;
        this->opts.get("normalize data", opt_norm, int_norm);
        this->normalize_data = (int_norm == 1);

        // Remove the constraint on n_clusters, in case the user re-uses the handle with different data
        da_int n_clusters_temp = n_clusters;
        reregister_kmeans_option<T>(this->opts, std::numeric_limits<da_int>::max());
        this->opts.set("n_clusters", n_clusters_temp);
    }

    // Check for conflicting options
    if (n_init > 1 && init_method == supplied) {
        std::string buff = "n_init was set to " + std::to_string(n_init) +
                           " but the initialization method was set to 'supplied'. The "
                           "k-means algorithm will only be run once.";
        n_init = 1;
        da_warn(this->err, da_status_incompatible_options, buff);
    }

    if (algorithm == hartigan_wong && (n_clusters == 1 || n_clusters >= n_samples)) {
        return da_error(this->err, da_status_incompatible_options,
                        "The Hartigan-Wong algorithm requires 1 < k < n_samples.");
    }

    // Spherical k-means is not compatible with Hartigan-Wong
    if (do_spherical && algorithm == hartigan_wong) {
        return da_error(this->err, da_status_incompatible_options,
                        "Cosine distance is not compatible with the Hartigan-Wong "
                        "algorithm. Please use Lloyd, Elkan, or MacQueen.");
    }

    // Hartigan-Wong does not support empty cluster recovery, so force error mode
    if (algorithm == hartigan_wong && empty_cluster_handling != error) {
        std::string buff = "The selected empty cluster handling mode is not supported "
                           "for the Hartigan-Wong "
                           "algorithm and will be overridden to 'error'.";
        da_warn(this->err, da_status_incompatible_options, buff);
        empty_cluster_handling = error;
    }

    // This can only be triggered if the user re-uses the handle, otherwise the option handling should catch it
    if (n_clusters > n_samples) {
        return da_error(this->err, da_status_incompatible_options,
                        "n_clusters = " + std::to_string(n_clusters) +
                            ", and n_samples = " + std::to_string(n_samples) +
                            ". Constraint: n_clusters <= n_samples.");
    }

    if (init_method == supplied && centres_supplied == false) {
        return da_error(this->err, da_status_no_data,
                        "The initialization method was set to 'supplied' but no initial "
                        "centres have been provided.");
    }

    if (A_temp) {
        delete[] (A_temp);
        A_temp = nullptr;
    }

    // Different algorithms need different storage of the user's data for optimal performance
    try {
        switch (this->algorithm) {
        case elkan: {
            // Store A as row-major
            this->A_order = row_major;
            if (this->order == column_major) {
                // Transpose A to row-major
                A_temp = new T[(size_t)n_samples * (size_t)n_features];
                ARCH::da_utils::copy_transpose_2D_array_column_to_row_major<T>(
                    n_samples, n_features, A_usr, lda_usr, A_temp, n_features);
                this->A = A_temp;
                this->lda = n_features;
            } else {
                this->A = A_usr;
                this->lda = lda_usr;
            }
            break;
        }
        case hartigan_wong: {
            // Store A as column-major
            this->A_order = column_major;
            if (this->order == row_major) {
                // Transpose A to column-major
                A_temp = new T[(size_t)n_samples * (size_t)n_features];
                ARCH::da_utils::copy_transpose_2D_array_row_to_column_major<T>(
                    n_samples, n_features, A_usr, lda_usr, A_temp, n_samples);
                this->A = A_temp;
                this->lda = n_samples;
            } else {
                this->A = A_usr;
                this->lda = lda_usr;
            }
            break;
        }
        case lloyd: {
            // For small numbers of clusters, Lloyd works best with A as it was provided, otherwise column-major is better
            if (n_clusters < KMEANS_LLOYD_BLOCK_SIZE<T>) {
                this->A_order = this->order;
                this->A = A_usr;
                this->lda = lda_usr;
            } else {
                this->A_order = column_major;
                if (this->order == row_major) {
                    // Transpose A to column-major
                    A_temp = new T[(size_t)n_samples * (size_t)n_features];
                    ARCH::da_utils::copy_transpose_2D_array_row_to_column_major<T>(
                        n_samples, n_features, A_usr, lda_usr, A_temp, n_samples);
                    this->A = A_temp;
                    this->lda = n_samples;
                } else {
                    this->A = A_usr;
                    this->lda = lda_usr;
                }
            }
            break;
        }
        default: {
            // MacQueen works best with A left as it was provided
            this->A_order = this->order;
            this->A = A_usr;
            this->lda = lda_usr;
            break;
        }
        }
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    // Set up some more algorithm-specific parameters
    switch (algorithm) {
    case lloyd:
        max_block_size = KMEANS_LLOYD_BLOCK_SIZE<T>;
        // Assign lloyd_kernel to the correct AVX kernel and get the required padding for use in memory allocation
        assign_lloyd_kernel(lloyd_kernel, this->padding, n_clusters);
        single_iteration = std::bind(&kmeans<T>::lloyd_iteration, this,
                                     std::placeholders::_1, std::placeholders::_2);
        // Lloyd requires no further initialization so set initialize_algorithm to nullptr
        initialize_algorithm = nullptr;
        break;
    case elkan:
        max_block_size = KMEANS_ELKAN_BLOCK_SIZE<T>;
        // Assign elkan_kernel to the correct AVX kernel and get the required padding for use in memory allocation
        assign_elkan_kernels(elkan_update_kernel, elkan_reduce_kernel, this->padding,
                             n_clusters, n_features);
        single_iteration = std::bind(&kmeans<T>::elkan_iteration, this,
                                     std::placeholders::_1, std::placeholders::_2);
        initialize_algorithm = std::bind(&kmeans<T>::init_elkan, this);
        break;
    case macqueen:
        max_block_size = KMEANS_MACQUEEN_BLOCK_SIZE<T>;
        single_iteration = std::bind(&kmeans<T>::macqueen_iteration, this,
                                     std::placeholders::_1, std::placeholders::_2);
        initialize_algorithm = std::bind(&kmeans<T>::init_macqueen, this);
        break;
    default:
        max_block_size = n_samples;
        break;
    }

    max_block_size = std::min(max_block_size, n_samples);
    ldworkcs1 = n_clusters + padding;

    da_int n_threads = omp_get_max_threads();

    // Initialize some arrays
    try {
        current_cluster_centres->resize((size_t)n_clusters * (size_t)n_features, 0.0);
        previous_cluster_centres->resize((size_t)n_clusters * (size_t)n_features, 0.0);
        thd_cluster_centres.resize(n_threads);
        thd_work1.resize(n_threads);
        thd_work2.resize(n_threads);
        thd_work3.resize(n_threads);
        thd_work4.resize(n_threads);
        thd_work_int.resize(n_threads);
        if (use_mixed_precision) {
            // We will need to store a lower precision copy of A and C
            A_lp.resize((size_t)n_samples * (size_t)n_features);
            C_lp.resize((size_t)n_clusters * (size_t)n_features);
        }
        // Allocate per-thread storage with padding to avoid false sharing
        da_int pad_T = 128 / sizeof(T);
        da_int pad_int = 128 / sizeof(da_int);
        for (da_int t = 0; t < n_threads; t++) {
            thd_cluster_centres[t].resize((size_t)n_clusters * (size_t)n_features + pad_T,
                                          0.0);
            thd_work1[t].resize(n_clusters + pad_T, 0.0);
            thd_work2[t].resize(n_clusters + pad_T, 0.0);
            thd_work3[t].resize(n_clusters + pad_T, 0.0);
            thd_work4[t].resize(n_clusters + pad_T, 0.0);
            thd_work_int[t].resize(n_clusters + pad_int, 0);
        }

        cluster_count.resize(n_clusters, 0);
        work_int1.resize(n_clusters, 0);
        work_int2.resize(n_samples, 0);
        // Extra bit on workc1 just to enable some padding to be done for vectorization
        workc1.resize(n_clusters + padding, 0.0);
        current_labels->resize(n_samples, 0);
        previous_labels->resize(n_samples, 0);
        if (n_init > 1) {
            best_cluster_centres->resize((size_t)n_clusters * (size_t)n_features, 0.0);
            best_labels->resize(n_samples, 0);
        }
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    // Ensure the extra padding in workc1 (for vectorization) won't interfere with any computation
    da_std::fill(workc1.end() - padding, workc1.end(),
                 std::numeric_limits<T>::infinity());

    // Based on what algorithms we are using, allocate the remaining memory
    try {

        switch (algorithm) {
        case elkan:
            workcc1.resize((size_t)n_clusters * (size_t)n_clusters, 0.0);
            workcs1.resize((size_t)n_samples * (size_t)(n_clusters + padding), 0.0);
            works1.resize(n_samples, 0.0);
            break;
        case macqueen:
            workcs1.resize((size_t)max_block_size * (size_t)n_clusters, 0.0);
            workc2.resize(n_clusters, 0.0);
            works1.resize(n_samples, 0.0);
            break;
        case lloyd:
            workcs1.resize((size_t)max_block_size * (size_t)(n_clusters + padding) *
                               (size_t)n_threads,
                           0.0);
            works1.resize(n_samples, 0.0);
            break;
        case hartigan_wong:
            works1.resize(n_samples, 0.0);
            workc2.resize(n_clusters, 0.0);
            workc3.resize(n_clusters, 0.0);
            work_int3.resize(n_clusters, 0);
            work_int4.resize(n_clusters, 0);
            break;
        }

        if (init_method == kmeanspp) {
            works1.resize(n_samples, 0.0);
            works2.resize(n_samples, 0.0);
            works3.resize(n_samples, 0.0);
            works4.resize(n_samples, 0.0);
            works5.resize(n_samples, 0.0);
        }

        if (init_method == afk_mcmc) {
            works1.resize(n_samples, 0.0);
            works2.resize(n_samples, 0.0);
            works3.resize(n_samples, 0.0);
        }

    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    if (centres_supplied && init_method == supplied) {
        // Copy the initial centres matrix into internal matrix buffer
        for (da_int j = 0; j < n_features; j++) {
            for (da_int i = 0; i < n_clusters; i++) {
                (*current_cluster_centres)[i + j * n_clusters] = C[i + ldc * j];
            }
        }
    }

    // If needed, initialize random number generation
    kmeans<T>::initialize_rng();

    // Set the initial best_inertia over all the runs to something large
    best_inertia = std::numeric_limits<T>::infinity();

    // Precompute data point norms for spherical k-means
    if (do_spherical && normalize_data) {
        try {
            data_norms.resize(n_samples, (T)0.0);
            data_inv_norms.resize(n_samples, (T)0.0);
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }
        if (A_order == column_major) {
            da_utils::compute_squared_row_norms(column_major, n_samples, n_features, A,
                                                lda, data_norms.data());
            for (da_int i = 0; i < n_samples; i++) {
                data_norms[i] = std::sqrt(data_norms[i]);
            }
        } else {
            for (da_int i = 0; i < n_samples; i++) {
                data_norms[i] = da_blas::cblas_nrm2(n_features, &A[i * lda], 1);
            }
        }
        for (da_int i = 0; i < n_samples; i++) {
            data_inv_norms[i] = safe_inv(data_norms[i]);
        }
    }

    if (use_mixed_precision) {
        // Store a lower precision copy of A
        da_utils::copy_array_convert_precision(
            A_order, n_samples, n_features, A, lda, A_lp.data(),
            (A_order == column_major) ? n_samples : n_features);
    }

    // Run k-means algorithm n_init times and select the run with the lowest inertia
    bool valid_run_found = false;
    for (da_int run = 0; run < n_init; run++) {

        // Initialize the centres if needed
        kmeans<T>::initialize_centres();

        // Iteratively refine the clusters using lower precision if needed
        if (this->use_mixed_precision) {
            status = kmeans<T>::lower_precision_init();
            if (status != da_status_success)
                return status;
        }

        // Perform k-means using current_inertia, current_cluster_centres and current_labels
        kmeans<T>::perform_kmeans();

        // If an empty cluster was found, skip this run
        if (empty_cluster_found) {
            empty_cluster_found = false;
            continue;
        }

        valid_run_found = true;

        // Check if it's the best run yet. Also accept the first valid run
        // unconditionally so best_cluster_centres is populated even when
        // current_inertia is non-finite (e.g. NaN/Inf input poisons the
        // comparison and would otherwise leave best_cluster_centres empty).
        if (current_inertia < best_inertia || best_cluster_centres->empty()) {
            best_inertia = current_inertia;
            best_n_iter = current_n_iter;
            best_lp_n_iter = lp_n_iter;
            // If this run hit the maximum number of iterations, a warning is required
            warn_maxit_reached = (converged == 0) ? true : false;
            std::swap(best_cluster_centres, current_cluster_centres);
            std::swap(best_labels, current_labels);
        }
    }

    // If no valid run was found, all runs encountered empty clusters
    if (!valid_run_found) {
        return da_error(
            this->err, da_status_empty_clusters,
            "All " + std::to_string(n_init) +
                " run(s) of the k-means algorithm resulted in at least one empty "
                "cluster during the k-means iterations. A better set of initial "
                "cluster centers is needed.");
    }

    // Compute the squared norms of the cluster centres in preparation for the predict phase of the algorithm; store in workc1
    // For spherical k-means, centres are unit-normalized so norms are all 1 — leave workc1 as zeros for predict
    for (da_int i = 0; i < n_clusters; i++) {
        workc1[i] = (T)0.0;
    }

    if (!do_spherical) {
        da_utils::compute_squared_row_norms(column_major, n_clusters, n_features,
                                            (*best_cluster_centres).data(), n_clusters,
                                            workc1.data());
    }

    this->model_trained = true;

    if (warn_maxit_reached)
        return da_warn(this->err, da_status_maxit,
                       "The maximum number of iterations was reached.");

    return status;
}

template <typename T>
da_status kmeans<T>::transform(da_int m_samples, da_int m_features, const T *X,
                               da_int ldx, T *X_transform, da_int ldx_transform) {
    if (!this->model_trained) {
        return da_warn(
            this->err, da_status_no_data,
            "The k-means has not been computed. Please call da_kmeans_compute_s or "
            "da_kmeans_compute_d.");
    }

    if (m_features != n_features)
        return da_error(
            this->err, da_status_invalid_input,
            "The function was called with m_features = " + std::to_string(m_features) +
                " but the k-means has been computed with " + std::to_string(n_features) +
                " features.");
    // Check the arguments
    da_status status = this->check_2D_array(this->order, m_samples, m_features, X, ldx,
                                            "m_samples", "m_features", "X", "ldx", 1, 1);
    if (status != da_status_success)
        return status;
    status = this->check_2D_array(this->order, m_samples, n_clusters, X_transform,
                                  ldx_transform, "m_samples", "n_clusters", "X_transform",
                                  "ldx_transform", 1, 1);
    if (status != da_status_success)
        return status;

    std::vector<T> x_work;

    try {
        x_work.resize(m_samples);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    if (do_spherical) {
        // Cosine distance: D_{ij} = 1 - (X_i · C_j) / ||X_i||  (centres are unit-norm)
        // First compute dot products X * C^T into X_transform
        if (this->order == column_major) {
            // X col-major (m_samples x n_features, ldx), C col-major (n_clusters x n_features, n_clusters)
            // X_transform col-major (m_samples x n_clusters, ldx_transform) = X * C^T
            da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, m_samples,
                                n_clusters, n_features, (T)1.0, X, ldx,
                                (*best_cluster_centres).data(), n_clusters, (T)0.0,
                                X_transform, ldx_transform);
        } else {
            // X is row-major (m_samples x n_features, ldx)
            // C is col-major (n_clusters x n_features, ld=n_clusters)
            // Want X_transform row-major (m_samples x n_clusters, ldx_transform)
            // X_transform_{ij} = sum_k X_{ik} C_{jk}
            // Row-major GEMM: X_transform = X * C^T
            da_blas::cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m_samples,
                                n_clusters, n_features, (T)1.0, X, ldx,
                                (*best_cluster_centres).data(), n_clusters, (T)0.0,
                                X_transform, ldx_transform);
        }

        // Compute inverse norms for normalization, or set to 1.0 if data is pre-normalized
        if (normalize_data) {
            da_utils::compute_squared_row_norms(this->order, m_samples, n_features, X,
                                                ldx, x_work.data());
            for (da_int i = 0; i < m_samples; i++) {
                x_work[i] = safe_inv_sqrt(x_work[i]);
            }
        } else {
            da_std::fill(x_work.begin(), x_work.begin() + m_samples, (T)1.0);
        }

        // Convert dot products to cosine distances
        if (this->order == column_major) {
            dot_to_cosine_distance_colmaj(m_samples, n_clusters, X_transform,
                                          ldx_transform, x_work.data());
        } else {
            dot_to_cosine_distance_rowmaj(m_samples, n_clusters, X_transform,
                                          ldx_transform, x_work.data());
        }
        return da_status_success;
    }

    if (this->order == column_major) {
        // Compute m_samples x n_clusters matrix of distances to cluster centres
        ARCH::euclidean_gemm_distance(column_major, m_samples, n_clusters, n_features, X,
                                      ldx, (*best_cluster_centres).data(), n_clusters,
                                      X_transform, ldx_transform, x_work.data(), 2,
                                      workc1.data(), 1, false, false);
    } else {
        // For row-major, we will transpose the cluster centres to row-major format
        std::vector<T> C_row_major;
        try {
            C_row_major.resize((size_t)n_clusters * (size_t)n_features);
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }
        da_utils::copy_transpose_2D_array_column_to_row_major(
            n_clusters, n_features, (*best_cluster_centres).data(), n_clusters,
            C_row_major.data(), n_features);
        // Compute m_samples x n_clusters matrix of distances to cluster centres
        ARCH::euclidean_gemm_distance(row_major, m_samples, n_clusters, n_features, X,
                                      ldx, C_row_major.data(), n_features, X_transform,
                                      ldx_transform, x_work.data(), 2, workc1.data(), 1,
                                      false, false);
    }

    return da_status_success;
}

template <typename T>
da_status kmeans<T>::predict(da_int k_samples, da_int k_features, const T *Y, da_int ldy,
                             da_int *Y_labels) {

    if (!this->model_trained) {
        return da_warn(
            this->err, da_status_no_data,
            "The k-means has not been computed. Please call da_kmeans_compute_s or "
            "da_kmeans_compute_d.");
    }

    if (k_features != n_features)
        return da_error(
            this->err, da_status_invalid_input,
            "The function was called with k_features = " + std::to_string(k_features) +
                " but the k-means has been computed with " + std::to_string(n_features) +
                " features.");

    // Check the arguments
    da_status status = this->check_2D_array(this->order, k_samples, k_features, Y, ldy,
                                            "k_samples", "k_features", "Y", "ldy", 1, 1);
    if (status != da_status_success)
        return status;

    // Check for illegal output arguments
    if (Y_labels == nullptr)
        return da_error(this->err, da_status_invalid_pointer,
                        "The array Y_labels is null.");

    // Compute nearest cluster centre for each sample in Y; essentially a single blocked step of the Lloyd iteration.
    std::vector<T> y_work;

    max_block_size = std::min(KMEANS_LLOYD_BLOCK_SIZE<T>, k_samples);

    da_utils::blocking_scheme(k_samples, max_block_size, n_blocks, block_rem);

    da_int n_threads = da_utils::get_n_threads_loop(n_blocks);

    da_int ldy_work;
    // Assign predict_kernel to the correct lloyd kernel and get the required padding
    da_int padding = 0;
    assign_lloyd_kernel(predict_kernel, padding, n_clusters);

    try {
        y_work.resize((size_t)max_block_size * (size_t)(n_clusters + padding) *
                      (size_t)n_threads);
        // Add padding to workc1 if needed but don't overwrite existing values
        workc1.resize(n_clusters + padding);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    da_std::fill(workc1.end() - padding, workc1.end(),
                 std::numeric_limits<T>::infinity());

    ldy_work = n_clusters + padding;

    da_int *dummy_int = nullptr;
    da_int block_index;
    da_int block_size = max_block_size;

    // For Y row-major we treat it as column-major storage of Y^T in GEMM calls
    auto Y_blas_trans = (this->order == column_major) ? CblasTrans : CblasNoTrans;

#pragma omp parallel firstprivate(block_size) private(block_index)                       \
    shared(n_blocks, block_rem, k_samples, max_block_size, best_cluster_centres, workc1, \
               dummy_int, Y_labels, y_work, padding, ldy_work, ldy, Y,                   \
               Y_blas_trans) default(none) num_threads(n_threads)
    {
        da_int y_work_index =
            ((da_int)omp_get_thread_num()) * max_block_size * (n_clusters + padding);
#pragma omp for schedule(dynamic)
        for (da_int i = 0; i < n_blocks; i++) {
            if (i == n_blocks - 1 && block_rem > 0) {
                block_index = k_samples - block_rem;
                block_size = block_rem;
            } else {
                block_index = i * max_block_size;
            }
            da_int Y_index =
                (this->order == column_major) ? block_index : block_index * ldy;
            // Compute the matrix D where D_{ij} = ||C_j||^2 - 2 Y C^T
            // Don't form it explicitly though: just form -2YC^T and add the ||C_j||^2 as and when we need them
            // Array access patterns mean for this loop it is quicker to form -2CY^T
            // If Y is row-major we use a trick which treats it as column-major storage of Y^T
            // For spherical k-means, use -CY^T (centres are unit-normalized, workc1 is zeros)
            T gemm_scalar_predict = do_spherical ? (T)-1.0 : (T)-2.0;
            da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, Y_blas_trans, n_clusters,
                                block_size, n_features, gemm_scalar_predict,
                                (*best_cluster_centres).data(), n_clusters, &Y[Y_index],
                                ldy, 0.0, &y_work[y_work_index], ldy_work);

            // Loop through the samples and find the closest cluster centre and its label
            predict_kernel(false, block_size, workc1.data(), dummy_int,
                           &Y_labels[block_index], &y_work[y_work_index], ldy_work,
                           n_clusters);
        }
    }

    return da_status_success;
}

/* Perform a single run of k-means */
template <typename T> void kmeans<T>::perform_kmeans() {

    // Special case for Hartigan-Wong algorithm which has a different structure
    if (algorithm == hartigan_wong) {
        perform_hartigan_wong();
        return;
    }
    da_utils::blocking_scheme(n_samples, max_block_size, n_blocks, block_rem);

    da_int n_threads = da_utils::get_n_threads_loop(n_blocks);
    if (initialize_algorithm)
        initialize_algorithm();

    // Ensure previous runs convergence test doesn't interfere with this run
    converged = 0;

    for (current_n_iter = 0; current_n_iter < max_iter; current_n_iter++) {
        // Start with the 'old' centres stored in previous_cluster_centres
        std::swap(previous_cluster_centres, current_cluster_centres);
        std::swap(previous_labels, current_labels);

        single_iteration(true, n_threads);

        // Handle empty clusters if needed
        if (empty_cluster_handling != ignore) {
            bool clusters_split = false;
            da_status ec_status = handle_empty_clusters(clusters_split);
            if (ec_status != da_status_success)
                return;
            if (clusters_split)
                continue; // Skip convergence test — centres were modified
        }

        // Check for convergence
        converged = convergence_test();
        if (converged > 0) {
            break;
        }
    }
    if (converged == 1 || current_n_iter == max_iter) {
        // Tolerance-based convergence OR max_iter exit: rerun the labelling
        // step against the latest centres without recomputing them, so the
        // returned labels are always consistent with cluster_centres and
        // with predict() on the training data.
        std::swap(previous_labels, current_labels);
        std::swap(previous_cluster_centres, current_cluster_centres);
        // Perform one more iteration to update labels, but without updating the cluster centres
        single_iteration(false, n_threads);
        std::swap(previous_cluster_centres, current_cluster_centres);
    }

    // For Elkan, we need to transpose the centres back to column-major order
    if (this->algorithm == elkan) {
        da_utils::copy_transpose_2D_array_row_to_column_major(
            n_clusters, n_features, (*current_cluster_centres).data(), n_features,
            (*previous_cluster_centres).data(), n_clusters);

        std::swap(current_cluster_centres, previous_cluster_centres);
    }

    // Finished this run, so compute current_inertia
    compute_current_inertia();
}

/* Compute current_inertia based on the current_cluster_centres */
template <typename T> void kmeans<T>::compute_current_inertia() {
    current_inertia = 0;
    T tmp;

    if (do_spherical) {
        // For spherical k-means, inertia = sum of (1 - cosine_similarity)
        // Centres are unit-normalized, so cos_sim = (x_i · c_label) / ||x_i||
        T cos_sim = (T)0.0;
        if (this->A_order == column_major) {
            for (da_int i = 0; i < n_samples; i++) {
                da_int label = (*current_labels)[i];
                T dot = (T)0.0;
                for (da_int j = 0; j < n_features; j++) {
                    dot += A[i + j * lda] *
                           (*current_cluster_centres)[label + j * n_clusters];
                }
                if (normalize_data)
                    cos_sim = dot * data_inv_norms[i];
                else
                    cos_sim = dot;

                current_inertia += (T)1.0 - cos_sim;
            }
        } else {
            for (da_int i = 0; i < n_samples; i++) {
                da_int label = (*current_labels)[i];
                T dot = (T)0.0;
                for (da_int j = 0; j < n_features; j++) {
                    dot += A[i * lda + j] *
                           (*current_cluster_centres)[label + j * n_clusters];
                }
                if (normalize_data)
                    cos_sim = dot * data_inv_norms[i];
                else
                    cos_sim = dot;

                current_inertia += (T)1.0 - cos_sim;
            }
        }
        return;
    }

    if (this->A_order == column_major) {
        for (da_int j = 0; j < n_features; j++) {
            da_int idx = j * lda;
            da_int cidx = j * n_clusters;
            for (da_int i = 0; i < n_samples; i++) {
                da_int label = (*current_labels)[i];
                tmp = A[i + idx] - (*current_cluster_centres)[label + cidx];
                current_inertia += tmp * tmp;
            }
        }
    } else {
        // A is stored in row-major order
        for (da_int i = 0; i < n_samples; i++) {
            da_int label = (*current_labels)[i];
            da_int idx = i * lda;
            for (da_int j = 0; j < n_features; j++) {
                tmp = A[idx + j] - (*current_cluster_centres)[label + j * n_clusters];
                current_inertia += tmp * tmp;
            }
        }
    }
}

/* Check for empty clusters after an iteration and handle according to empty_cluster_handling.
 * Sets clusters_split = true if any empty clusters were resolved by splitting.
 * Returns da_status_empty_clusters if empty_cluster_handling == error and empty clusters exist. */
template <typename T> da_status kmeans<T>::handle_empty_clusters(bool &clusters_split) {
    clusters_split = false;

    // Recount cluster sizes from labels (cluster_count may have been modified by
    // scale_current_cluster_centres setting empty counts to 1 to avoid div-by-zero)
    da_std::fill(work_int1.begin(), work_int1.end(), 0);
    for (da_int i = 0; i < n_samples; i++)
        work_int1[(*current_labels)[i]] += 1;

    // Count empty clusters
    da_int n_empty = 0;
    for (da_int i = 0; i < n_clusters; i++) {
        if (work_int1[i] == 0)
            n_empty++;
    }
    if (n_empty == 0)
        return da_status_success;

    // Empty clusters exist
    if (empty_cluster_handling == error) {
        empty_cluster_found = true;
        // Don't write to error buffer here, as if n_init>1 other runs may work fine
        return da_status_empty_clusters;
    }

    // Split mode: reassign furthest points from non-singleton clusters to empty clusters

    // Precompute strides to avoid branching inside inner loops
    // Data matrix A: row stride = A_rstride, column stride = A_cstride
    da_int A_rstride = (A_order == column_major) ? 1 : lda;
    da_int A_cstride = (A_order == column_major) ? lda : 1;
    // Centre matrix: row stride = C_rstride, column stride = C_cstride
    da_int C_rstride, C_cstride;
    if (algorithm == elkan) {
        C_rstride = n_features; // row-major centres
        C_cstride = 1;
    } else {
        C_rstride = 1; // column-major centres
        C_cstride = n_clusters;
    }

    // Compute distance from each sample to its assigned centre; reuse works1 (size n_samples)
    // For spherical: use cosine distance (1 - cos_sim); for Euclidean: use squared Euclidean distance
    for (da_int i = 0; i < n_samples; i++) {
        da_int label = (*current_labels)[i];
        if (do_spherical) {
            T dot = (T)0.0, cos_sim = (T)0.0;
            da_int idx = i * A_rstride;
            da_int cidx = label * C_rstride;
            for (da_int j = 0; j < n_features; j++) {
                dot += A[idx + j * A_cstride] *
                       (*current_cluster_centres)[cidx + j * C_cstride];
            }
            if (normalize_data)
                cos_sim = dot * data_inv_norms[i];
            else
                cos_sim = dot;

            works1[i] = (T)1.0 - cos_sim;
        } else {
            T dist = (T)0.0;
            da_int idx = i * A_rstride;
            da_int cidx = label * C_rstride;
            for (da_int j = 0; j < n_features; j++) {
                T diff = A[idx + j * A_cstride] -
                         (*current_cluster_centres)[cidx + j * C_cstride];
                dist += diff * diff;
            }
            works1[i] = dist;
        }
    }

    // For each empty cluster, find the furthest point from a non-singleton cluster
    for (da_int c = 0; c < n_clusters; c++) {
        if (work_int1[c] != 0)
            continue;

        // Find sample with maximum distance whose cluster has >1 member
        da_int best_idx = -1;
        T best_dist = (T)-1.0;
        for (da_int i = 0; i < n_samples; i++) {
            if (work_int1[(*current_labels)[i]] > 1 && works1[i] > best_dist) {
                best_dist = works1[i];
                best_idx = i;
            }
        }

        // If no suitable donor found (all non-empty clusters are singletons), stop
        if (best_idx < 0)
            break;

        da_int old_label = (*current_labels)[best_idx];
        da_int n_old = work_int1[old_label];

        // Precompute scale factor for normalized data
        T inv_norm_best =
            (do_spherical && normalize_data) ? data_inv_norms[best_idx] : (T)1.0;

        // Update old cluster centre: remove point contribution
        // new_centre = (centre * n_old - point) / (n_old - 1)
        for (da_int j = 0; j < n_features; j++) {
            T a_val = A[best_idx * A_rstride + j * A_cstride] * inv_norm_best;
            T &centre_val =
                (*current_cluster_centres)[old_label * C_rstride + j * C_cstride];
            centre_val = (centre_val * n_old - a_val) / (n_old - 1);
        }

        // Set new cluster centre = the relocated point
        for (da_int j = 0; j < n_features; j++) {
            (*current_cluster_centres)[c * C_rstride + j * C_cstride] =
                A[best_idx * A_rstride + j * A_cstride] * inv_norm_best;
        }

        // For spherical k-means, normalize both the modified donor centre and the new centre
        if (do_spherical) {
            // Normalize donor centre
            T norm_sq = (T)0.0;
            for (da_int j = 0; j < n_features; j++) {
                T val = (*current_cluster_centres)[old_label * C_rstride + j * C_cstride];
                norm_sq += val * val;
            }
            if (norm_sq > (T)0.0) {
                T inv_norm = (T)1.0 / std::sqrt(norm_sq);
                for (da_int j = 0; j < n_features; j++) {
                    (*current_cluster_centres)[old_label * C_rstride + j * C_cstride] *=
                        inv_norm;
                }
            }
            // Normalize new centre
            norm_sq = (T)0.0;
            for (da_int j = 0; j < n_features; j++) {
                T val = (*current_cluster_centres)[c * C_rstride + j * C_cstride];
                norm_sq += val * val;
            }
            if (norm_sq > (T)0.0) {
                T inv_norm = (T)1.0 / std::sqrt(norm_sq);
                for (da_int j = 0; j < n_features; j++) {
                    (*current_cluster_centres)[c * C_rstride + j * C_cstride] *= inv_norm;
                }
            }
        }

        // Update bookkeeping
        work_int1[old_label] -= 1;
        work_int1[c] = 1;
        (*current_labels)[best_idx] = c;
        works1[best_idx] = (T)0.0;

        // Recompute distances for samples still assigned to the modified donor cluster
        // (only needed when filling multiple empty clusters, to keep subsequent selections accurate)
        if (n_empty > 1) {
            for (da_int i = 0; i < n_samples; i++) {
                if ((*current_labels)[i] == old_label) {
                    if (do_spherical) {
                        T dot = (T)0.0, cos_sim = (T)0.0;
                        da_int idx = i * A_rstride;
                        da_int cidx = old_label * C_rstride;
                        for (da_int j = 0; j < n_features; j++) {
                            dot += A[idx + j * A_cstride] *
                                   (*current_cluster_centres)[cidx + j * C_cstride];
                        }
                        if (normalize_data)
                            cos_sim = dot * data_inv_norms[i];
                        else
                            cos_sim = dot;
                        works1[i] = (T)1.0 - cos_sim;
                    } else {
                        T dist = (T)0.0;
                        da_int idx = i * A_rstride;
                        da_int cidx = old_label * C_rstride;
                        for (da_int j = 0; j < n_features; j++) {
                            T diff = A[idx + j * A_cstride] -
                                     (*current_cluster_centres)[cidx + j * C_cstride];
                            dist += diff * diff;
                        }
                        works1[i] = dist;
                    }
                }
            }
        }
    }

    // Copy counts back to cluster_count
    for (da_int i = 0; i < n_clusters; i++)
        cluster_count[i] = work_int1[i];

    // Algorithm-specific cleanup
    if (algorithm == elkan) {
        // Invalidate bounds to force full distance recomputation next iteration
        da_std::fill(works1.begin(), works1.end(), std::numeric_limits<T>::max());
        for (da_int i = 0; i < n_samples * ldworkcs1; i++)
            workcs1[i] = (T)0.0;
    } else if (algorithm == macqueen) {
        // Recompute squared centre norms (workc1) used in GEMM distance
        da_utils::compute_squared_row_norms(column_major, n_clusters, n_features,
                                            (*current_cluster_centres).data(), n_clusters,
                                            workc1.data());
    }

    clusters_split = true;
    return da_status_success;
}

/* Compute the difference between the current and previous centres and store in previous_cluster_centres */
template <typename T> void kmeans<T>::compute_centre_shift() {

    // Before overwriting previous_cluster_centres, compute and store its norm, for use in convergence test

    normc = (T)0.0;

    for (da_int i = 0; i < n_clusters * n_features; i++) {
        normc += (*previous_cluster_centres)[i] * (*previous_cluster_centres)[i];
        (*previous_cluster_centres)[i] -= (*current_cluster_centres)[i];
    }

    normc = std::sqrt(normc);
}

/* Check if the k-means iteration has converged */
/* 0 means no convergence, 1 is tol-based convergence, 2 is strict convergence (labels didn't change) */
template <typename T> da_int kmeans<T>::convergence_test() {

    da_int convergence_test = 0;

    // Check if labels have changed, but only after we've done at least one complete iteration
    if (current_n_iter > 1) {
        convergence_test = 2;
        for (da_int i = 0; i < n_samples; i++) {
            if ((*current_labels)[i] != (*previous_labels)[i]) {
                convergence_test = 0;
                break;
            }
        }
    }

    if (convergence_test > 0)
        return convergence_test;

    // Recall that that the end of each iteration previous_cluster_centres contains the shift made in that particular iteration
    // dlange is expecting column major here, but it actually doesn't matter since we're just computing the Frobenius norm
    char norm = 'F';
    if (da::lange(&norm, &n_clusters, &n_features, (*previous_cluster_centres).data(),
                  &n_clusters, nullptr) < tol * normc)
        convergence_test = 1;

    return convergence_test;
}

/* Initialize the centres, if needed, for the start of k-means computation*/
template <typename T> void kmeans<T>::initialize_centres() {
    da_std::fill(previous_cluster_centres->begin(), previous_cluster_centres->end(), 0.0);
    switch (init_method) {
    case random_samples: {
        // Select randomly (without replacement) from the data points
        da_std::iota(work_int2.begin(), work_int2.end(), 0);
        da_std::sample(work_int2.begin(), work_int2.end(), std::begin(work_int1),
                       n_clusters, mt_gen);
        if (this->A_order == column_major) {
            for (da_int j = 0; j < n_clusters; j++) {
                for (da_int i = 0; i < n_features; i++) {
                    (*current_cluster_centres)[i * n_clusters + j] =
                        A[i * lda + work_int1[j]];
                }
            }
        } else {
            // A is row-major
            for (da_int j = 0; j < n_clusters; j++) {
                for (da_int i = 0; i < n_features; i++) {
                    (*current_cluster_centres)[i * n_clusters + j] =
                        A[i + lda * work_int1[j]];
                }
            }
        }
        break;
    }
    case random_partitions: { // Zero out relevant arrays
        for (da_int i = 0; i < n_clusters; i++) {
            work_int1[i] = 0;
        }
        for (da_int j = 0; j < n_clusters * n_features; j++)
            (*current_cluster_centres)[j] = (T)0.0;

        // Assign each sample point to a random cluster
        std::uniform_int_distribution<> dis_int(0, n_clusters - 1);
        for (da_int i = 0; i < n_samples; i++) {
            da_int workcc1_index = dis_int(mt_gen);
            (*current_labels)[i] = workcc1_index;
            work_int1[workcc1_index] += 1;
            // Add this sample to the relevant cluster mean
            if (this->A_order == column_major) {
                for (da_int j = 0; j < n_features; j++) {
                    (*current_cluster_centres)[workcc1_index + j * n_clusters] +=
                        A[i + j * lda];
                }
            } else {
                // A is row-major but cluster centres are column-major
                for (da_int j = 0; j < n_features; j++) {
                    (*current_cluster_centres)[workcc1_index + j * n_clusters] +=
                        A[i * lda + j];
                }
            }
        }

// Scale to get proper column means (cluster_count contains the number of data points in each cluster)
#pragma omp simd collapse(2)
        for (da_int j = 0; j < n_features; j++) {
            for (da_int i = 0; i < n_clusters; i++) {
                (*current_cluster_centres)[i + j * n_clusters] /= work_int1[i];
            }
        }

        break;
    }
    case kmeanspp: {
        kmeans_plusplus();
        break;
    }
    case afk_mcmc: {
        afk_mcmc_init();
        break;
    }
    default:
        // No need to do anything as initial centres were provided and have been stored in current_cluster_centres already
        break;
    }

    // If doing spherical k-means, we need to normalize initialized centres
    // Centres are always in column-major order at this point (Elkan transposes separately)
    // workc1 has been initialized to zeros earlier
    if (do_spherical) {
        da_utils::normalize_rows_inplace(column_major, n_clusters, n_features,
                                         (*current_cluster_centres).data(), n_clusters,
                                         workc1.data());
    }
}

/* Compute distance from all samples to a single data point (used in k-means++ and AFK-MC²)
 * For Euclidean: squared Euclidean distance; For spherical: cosine distance (1 - cos_sim)
 * centre_idx is the index of the centre point in A; works1 must contain squared norms */
template <typename T>
void kmeans<T>::compute_distances_to_point(da_int centre_idx, T *dist_out) {
    da_int A_rstride = (this->A_order == column_major) ? 1 : lda;
    da_int A_cstride = (this->A_order == column_major) ? lda : 1;
    if (do_spherical) {
        for (da_int i = 0; i < n_samples; i++) {
            T dot = (T)0.0, cos_sim = (T)0.0;
            for (da_int j = 0; j < n_features; j++) {
                dot += A[i * A_rstride + j * A_cstride] *
                       A[centre_idx * A_rstride + j * A_cstride];
            }
            if (normalize_data) {
                cos_sim = dot * data_inv_norms[i] * data_inv_norms[centre_idx];
            } else {
                cos_sim = dot;
            }
            dist_out[i] = std::max((T)0.0, (T)1.0 - cos_sim);
        }
    } else {
        T dummy = (T)0.0;
        if (this->A_order == column_major) {
            ARCH::euclidean_gemm_distance(column_major, n_samples, 1, n_features, A, lda,
                                          &A[centre_idx], lda, dist_out, n_samples,
                                          works1.data(), 1, &dummy, 2, true, false);
        } else {
            ARCH::euclidean_gemm_distance(row_major, n_samples, 1, n_features, A, lda,
                                          &A[centre_idx * lda], lda, dist_out, 1,
                                          works1.data(), 1, &dummy, 2, true, false);
        }
    }
}

/* Initialize centres using k-means++ */
template <typename T> void kmeans<T>::kmeans_plusplus() {

    if (do_spherical && normalize_data) {
        // Reuse precomputed data_norms (||x_i||) to populate works1 with squared norms
        for (da_int i = 0; i < n_samples; i++)
            works1[i] = data_norms[i] * data_norms[i];
    } else if (!do_spherical) {
        // Compute squared norms of the data points and store in works1
        da_utils::compute_squared_row_norms(this->A_order, n_samples, n_features, A, lda,
                                            works1.data());
    }

    da_int n_trials = 2 + (da_int)std::log(n_clusters);

    // Pick first centre randomly from the sample data points and store which one it was in work_int1
    std::uniform_int_distribution<> dis_int(0, n_samples - 1);
    da_int random_int = dis_int(mt_gen);
    work_int1[0] = random_int;
    if (this->A_order == column_major) {
        for (da_int i = 0; i < n_features; i++) {
            (*current_cluster_centres)[i * n_clusters] = A[i * lda + work_int1[0]];
        }
    } else {
        // A is row-major
        for (da_int i = 0; i < n_features; i++) {
            (*current_cluster_centres)[i * n_clusters] = A[i + lda * work_int1[0]];
        }
    }

    // In works3 form the distance of each point in A to the first chosen centre
    compute_distances_to_point(random_int, works3.data());

    // Numerical errors could cause one of the distances to be slightly
    // negative, and non-finite input data (NaN/Inf) could make a distance
    // NaN or Inf. Both lead to undefined behaviour or an assertion failure
    // in std::discrete_distribution (the weight sum is no longer > 0), so
    // clamp negative and non-finite distances to zero before using works3
    // as a weight vector.
    works3[random_int] = (T)0.0;
    for (da_int i = 0; i < n_samples; i++) {
        if (!std::isfinite(works3[i]) || works3[i] < (T)0.0)
            works3[i] = (T)0.0;
    }

    // Need to catch an edge case where all points are the same
    bool coincident_points = true;

    for (da_int i = 0; i < n_samples; i++) {
        if (works3[i] > (T)0.0) {
            coincident_points = false;
            break;
        }
    }

    if (coincident_points) {
        // Doesn't matter which ones we choose, this is just to prevent exceptions later, so just use the first ones
        if (this->A_order == column_major) {
            for (da_int j = 0; j < n_features; j++) {
                for (da_int k = 0; k < n_clusters; k++) {
                    (*current_cluster_centres)[j * n_clusters + k] = A[j * lda + k];
                }
            }
        } else {
            // A is row-major
            for (da_int k = 0; k < n_clusters; k++) {
                for (da_int j = 0; j < n_features; j++) {
                    (*current_cluster_centres)[j * n_clusters + k] = A[k * lda + j];
                }
            }
        }
    } else {

        for (da_int k = 1; k < n_clusters; k++) {

            // Choose n_trials new sample points as the next centre, randomly, weighted by works3, the min distance
            // Don't need to worry about replacement because probability of zero of picking previously chosen point

            da_int best_candidate = 0;
            T best_candidate_cost = std::numeric_limits<T>::infinity();

            std::discrete_distribution<> weighted_dis(works3.begin(), works3.end());
            for (da_int trials = 0; trials < n_trials; trials++) {
                // Our candidate points are stored in work_int2
                work_int2[trials] = weighted_dis(mt_gen);
            }

            for (da_int trials = 0; trials < n_trials; trials++) {

                // It's worth checking in case we've selected a candidate point twice, in which case ignore it
                bool repeat_sample = false;
                for (da_int j = 0; j < trials; j++) {
                    if (work_int2[j] == work_int2[trials]) {
                        repeat_sample = true;
                        break;
                    }
                }
                if (repeat_sample)
                    continue;

                // Calculate cost function for this candidate point
                T current_cost = (T)0.0;
                da_int current_candidate = work_int2[trials];

                // Compute the distance from each point to the candidate centre and store in works4
                compute_distances_to_point(current_candidate, works4.data());
                // Get minimum squared distance of each sample point to potential centre
                current_cost = 0;
                for (da_int j = 0; j < n_samples; j++) {
                    works5[j] = std::max((T)0.0, std::min(works3[j], works4[j]));
                    current_cost += works5[j];
                }

                if (current_cost < best_candidate_cost) {
                    best_candidate_cost = current_cost;
                    best_candidate = work_int2[trials];
                    std::swap(works2, works5);
                }
            }

            // Place the best candidate as the next cluster centre
            if (this->A_order == column_major) {
                for (da_int i = 0; i < n_features; i++) {
                    (*current_cluster_centres)[i * n_clusters + k] =
                        A[i * lda + best_candidate];
                }
            } else {
                // A is row-major
                for (da_int i = 0; i < n_features; i++) {
                    (*current_cluster_centres)[i * n_clusters + k] =
                        A[i + lda * best_candidate];
                }
            }
            work_int1[k] = best_candidate;
            std::swap(works3, works2);
            // Guard against negative probabilities again
            works3[best_candidate] = (T)0.0;
        }
    }
    // Now we have n_clusters entries in current_cluster_centres
}

/* Initialize centres using AFK-MC² (Assumption-Free K-MC²) from Bachem et al. (2016) */
template <typename T> void kmeans<T>::afk_mcmc_init() {

    if (do_spherical && normalize_data) {
        // Reuse precomputed data_norms (||x_i||) to populate works1 with squared norms
        for (da_int i = 0; i < n_samples; i++)
            works1[i] = data_norms[i] * data_norms[i];
    } else if (!do_spherical) {
        // Precompute squared norms of the data points and store in works1
        da_utils::compute_squared_row_norms(this->A_order, n_samples, n_features, A, lda,
                                            works1.data());
    }

    // Step 1: Sample first centre c1 uniformly from A
    std::uniform_int_distribution<> dis_int(0, n_samples - 1);
    da_int c1_idx = dis_int(mt_gen);
    if (this->A_order == column_major) {
        for (da_int j = 0; j < n_features; j++) {
            (*current_cluster_centres)[j * n_clusters] = A[j * lda + c1_idx];
        }
    } else {
        for (da_int j = 0; j < n_features; j++) {
            (*current_cluster_centres)[j * n_clusters] = A[c1_idx * lda + j];
        }
    }

    // For spherical with unnormalized data: normalize the first centre using precomputed norm
    if (do_spherical && normalize_data) {
        for (da_int j = 0; j < n_features; j++) {
            (*current_cluster_centres)[j * n_clusters] *= data_inv_norms[c1_idx];
        }
    }

    // Step 2-3: Compute proposal distribution q(x) = 0.5 * d(x,c1)^2 / sum_x' d(x',c1)^2 + 1/(2n)
    // For spherical: use cosine distance instead of squared Euclidean
    // Compute distances to c1 and store in works3
    compute_distances_to_point(c1_idx, works3.data());

    // Clamp negative distances and compute sum of squared distances
    T sum_dist = (T)0.0;
    for (da_int i = 0; i < n_samples; i++) {
        works3[i] = std::max((T)0.0, works3[i]);
        sum_dist += works3[i];
    }

    // Build q(x) in works2: q(x) = 0.5 * d(x,c1)^2 / sum_dist + 1/(2n)
    T inv_2n = (T)1.0 / (2 * n_samples);
    if (sum_dist > (T)0.0) {
        for (da_int i = 0; i < n_samples; i++) {
            works2[i] = (T)0.5 * works3[i] / sum_dist + inv_2n;
        }
    } else {
        // All points coincident: uniform proposal
        for (da_int i = 0; i < n_samples; i++) {
            works2[i] = (T)1.0 / n_samples;
        }
    }

    // Precompute data strides for dist_to_nearest_centre lambda
    da_int A_rstride = (this->A_order == column_major) ? 1 : lda;
    da_int A_cstride = (this->A_order == column_major) ? lda : 1;

    // Helper lambda: compute distance from sample point idx to nearest centre in C_{i-1}
    // (first n_centres entries stored column-major in current_cluster_centres)
    // Returns squared Euclidean distance (or cosine distance for spherical)
    auto dist_to_nearest_centre = [&](da_int idx, da_int n_centres) -> T {
        T min_dist = std::numeric_limits<T>::infinity();
        if (do_spherical) {
            for (da_int c = 0; c < n_centres; c++) {
                T dot = (T)0.0, cos_sim = (T)0.0;
                for (da_int j = 0; j < n_features; j++) {
                    dot += A[idx * A_rstride + j * A_cstride] *
                           (*current_cluster_centres)[c + j * n_clusters];
                }
                // Centres are unit-normalized for spherical
                if (normalize_data) {
                    cos_sim = dot * data_inv_norms[idx];
                } else {
                    cos_sim = dot;
                }
                T dist = std::max((T)0.0, (T)1.0 - cos_sim);
                if (dist < min_dist)
                    min_dist = dist;
            }
        } else {
            for (da_int c = 0; c < n_centres; c++) {
                T dist = (T)0.0;
                for (da_int j = 0; j < n_features; j++) {
                    T diff = A[idx * A_rstride + j * A_cstride] -
                             (*current_cluster_centres)[c + j * n_clusters];
                    dist += diff * diff;
                }
                if (dist < min_dist)
                    min_dist = dist;
            }
        }
        return min_dist;
    };

    // Steps 4-12: Main MCMC loop to select centres 2..k
    std::discrete_distribution<> q_dist(works2.begin(), works2.end());
    std::uniform_real_distribution<T> unif((T)0.0, (T)1.0);

    for (da_int i = 1; i < n_clusters; i++) {

        // Step 6: Sample initial candidate x from q
        da_int x_idx = q_dist(mt_gen);
        T dx = dist_to_nearest_centre(x_idx, i);

        // Steps 8-11: Markov chain of length afk_mcmc_samples
        for (da_int step = 0; step < afk_mcmc_samples; step++) {
            // Step 9: Sample y from q
            da_int y_idx = q_dist(mt_gen);
            // Step 10: Compute d(y, C_{i-1})^2
            T dy = dist_to_nearest_centre(y_idx, i);

            // Step 11: Accept y with probability min(1, dy*q(x) / (dx*q(y)))
            // Guard against division by zero: if dx*q(y) == 0, accept
            T dx_qy = dx * works2[y_idx];
            if (dx_qy == (T)0.0 || (dy * works2[x_idx]) / dx_qy > unif(mt_gen)) {
                x_idx = y_idx;
                dx = dy;
            }
        }

        // Step 12: Add x as the i-th centre
        if (this->A_order == column_major) {
            for (da_int j = 0; j < n_features; j++) {
                (*current_cluster_centres)[j * n_clusters + i] = A[j * lda + x_idx];
            }
        } else {
            for (da_int j = 0; j < n_features; j++) {
                (*current_cluster_centres)[j * n_clusters + i] = A[x_idx * lda + j];
            }
        }

        // For spherical with unnormalized data: normalize using precomputed norm
        if (do_spherical && normalize_data) {
            for (da_int j = 0; j < n_features; j++) {
                (*current_cluster_centres)[i + j * n_clusters] *= data_inv_norms[x_idx];
            }
        }
    }
    // Now we have n_clusters entries in current_cluster_centres
}

/* Initialize the random number generator, if needed */
template <typename T> void kmeans<T>::initialize_rng() {
    if (init_method != supplied) {
        if (seed == -1) {
            std::random_device r;
            seed = std::abs((da_int)r());
        }
        mt_gen.seed(seed);
    }
}

/* Iterative refinement */
template <> da_status kmeans<double>::lower_precision_init() {

    da_status status;

    this->opts.get("low precision convergence tolerance", lp_tol);

    this->opts.get("low precision max_iter", lp_max_iter);

    if (lp_tol <= tol) {
        return da_error(
            this->err, da_status_incompatible_options,
            "Low precision convergence tolerance must be greater than "
            "convergence tolerance. Current values: low precision convergence "
            "tolerance = " +
                std::to_string(lp_tol) +
                ", convergence tolerance = " + std::to_string(tol) + ".");
    }

    // Store lower precision version of the initial cluster centres
    da_utils::copy_array_convert_precision(column_major, n_clusters, n_features,
                                           (*current_cluster_centres).data(), n_clusters,
                                           C_lp.data(), n_clusters);

    // Create a single-precision kmeans object and populate it with relevant data from this double-precision object
    da_int lda_lp = (this->A_order == column_major) ? this->n_samples : this->n_features;
    kmeans<float> km_float(
        *this->err, this->A_order, this->order, this->algorithm, supplied, this->seed,
        (float)this->lp_tol, this->lp_max_iter, this->n_samples, this->n_features,
        this->n_clusters, 1, A_lp.data(), lda_lp, A_lp.data(), lda_lp, C_lp.data(),
        this->n_clusters, true, true, false, this->empty_cluster_handling,
        this->afk_mcmc_samples, this->do_spherical, this->normalize_data);

    // Now compute the k-means in lower precision
    status = km_float.compute();
    if (status != da_status_success && status != da_status_maxit)
        return status;
    lp_n_iter = km_float.best_n_iter;

    // Copy the centres back to this double-precision object
    da_utils::copy_array_convert_precision(
        column_major, n_clusters, n_features, (*km_float.best_cluster_centres).data(),
        n_clusters, (*current_cluster_centres).data(), n_clusters);

    return da_status_success;
}
template <> da_status kmeans<float>::lower_precision_init() {
    return da_status_invalid_option;
}

template <typename T> da_status kmeans<T>::serialize(serialization_buffer &buffer) {

    da_status status = da_status_success;
    auto io_dispatch = [&buffer, &status](auto &data) -> void {
        if (status != da_status_success) {
            return;
        }
        status = buffer.dispatch_buffer_io(data);
        return;
    };

    io_dispatch(this->model_trained);
    io_dispatch(this->order);
    io_dispatch(this->algorithm);
    io_dispatch(this->n_samples);
    io_dispatch(this->n_features);
    io_dispatch(this->best_n_iter);
    io_dispatch(this->best_inertia);
    io_dispatch(this->seed);
    io_dispatch(this->n_clusters);
    io_dispatch(this->tol);
    io_dispatch(this->max_iter);
    io_dispatch(this->n_init);
    io_dispatch(*this->best_labels);
    io_dispatch(*this->best_cluster_centres);
    io_dispatch(this->workc1);
    io_dispatch(this->do_spherical);
    io_dispatch(this->normalize_data);

    return status;
}

template <typename T> da_status kmeans<T>::save_model(serialization_buffer &buffer) {

    if (!this->model_trained) {
        return da_error(this->err, da_status_no_data,
                        "k-means clustering has not yet been computed. Please call "
                        "da_kmeans_compute_s "
                        "or da_kmeans_compute_d before saving the model.");
    }

    da_status status = basic_handle<T>::save_model(buffer);
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure serializing model.");

    return status;
}

template <typename T> da_status kmeans<T>::load_model(serialization_buffer &buffer) {
    da_status status = basic_handle<T>::load_model(buffer);
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure deserializing model.");

    return status;
}

template class kmeans<double>;
template class kmeans<float>;

} // namespace da_kmeans

} // namespace ARCH
