/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
#include "aoclda.h"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "da_omp.hpp"
#include "hartigan_wong.hpp"
#include "kmeans_loop_unrolls.hpp"
#include "kmeans_options.hpp"
#include "kmeans_types.hpp"
#include "lapack_templates.hpp"
#include "macros.h"
#include "pairwise_distances.hpp"
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>

#ifndef NOMINMAX
#define NOMINMAX
#endif

namespace ARCH {

namespace da_kmeans {

using namespace da_kmeans_types;

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
da_status kmeans<T>::get_result(da_result query, da_int *dim, T *result) {
    // Don't return anything if k-means has not been computed
    if (!iscomputed) {
        return da_warn(this->err, da_status_no_data,
                       "k-means clustering has not yet been computed. Please call "
                       "da_kmeans_compute_s "
                       "or da_kmeans_compute_d before extracting results.");
    }

    da_int rinfo_size = 5;

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
    // Don't return anything if k-means has not been computed
    if (!iscomputed) {
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

/* Store details about user's data matrix in preparation for k-means computation */
template <typename T>
da_status kmeans<T>::set_data(da_int n_samples, da_int n_features, const T *A_in,
                              da_int lda_in) {

    // Guard against errors due to multiple calls using the same class instantiation
    if (A_temp) {
        delete[] (A_temp);
        A_temp = nullptr;
    }

    da_status status =
        this->store_2D_array(n_samples, n_features, A_in, lda_in, &A_temp, &A, lda,
                             "n_samples", "n_features", "A", "lda");
    if (status != da_status_success)
        return status;

    // Store dimensions of A
    this->n_samples = n_samples;
    this->n_features = n_features;

    // Record that initialization is complete but computation has not yet been performed
    initdone = true;
    iscomputed = false;

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

    // Check for illegal arguments
    this->opts.get("n_clusters", n_clusters);

    da_status status =
        this->store_2D_array(n_clusters, n_features, C_in, ldc_in, &C_temp, &C, ldc,
                             "n_clusters", "n_features", "C", "ldc");
    if (status != da_status_success)
        return status;

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

    // Read in options and store in class
    this->opts.get("n_clusters", n_clusters);

    std::string opt_method;
    this->opts.get("initialization method", opt_method, init_method);

    std::string opt_alg;
    this->opts.get("algorithm", opt_alg, algorithm);

    this->opts.get("n_init", n_init);

    this->opts.get("max_iter", max_iter);

    this->opts.get("convergence tolerance", tol);

    this->opts.get("seed", seed);

    // Remove the constraint on n_clusters, in case the user re-uses the handle with different data
    reregister_kmeans_option<T>(this->opts, std::numeric_limits<da_int>::max());
    this->opts.set("n_clusters", n_samples);

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

    switch (algorithm) {
    case lloyd:
        max_block_size = KMEANS_LLOYD_BLOCK_SIZE;
        initialize_algorithm = &kmeans<T>::init_lloyd;
        break;
    case elkan:
        max_block_size = KMEANS_ELKAN_BLOCK_SIZE;
        initialize_algorithm = &kmeans<T>::init_elkan;
        break;
    case macqueen:
        max_block_size = KMEANS_MACQUEEN_BLOCK_SIZE;
        initialize_algorithm = &kmeans<T>::init_macqueen;
        break;
    default:
        max_block_size = n_samples;
        break;
    }
    max_block_size = std::min(max_block_size, n_samples);

    da_int n_threads = omp_get_max_threads();

    // Initialize some arrays
    try {
        current_cluster_centres->resize(n_clusters * n_features, 0.0);
        previous_cluster_centres->resize(n_clusters * n_features, 0.0);
        thread_cluster_centres.resize(n_clusters * n_features * n_threads, 0.0);
        cluster_count.resize(n_clusters, 0);
        work_int1.resize(n_clusters * n_threads, 0);
        work_int2.resize(n_samples, 0);
        // Extra bit on workc1 just to enable some padding to be done if loop unrolling occurs
        workc1.resize(n_clusters + 8, 0.0);
        current_labels->resize(n_samples, 0);
        previous_labels->resize(n_samples, 0);
        if (n_init > 1) {
            best_cluster_centres->resize(n_clusters * n_features);
            best_labels->resize(n_samples);
        }
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    // Ensure the extra padding in workc1 (for vectorization) won't interfere with any computation
    std::fill(workc1.end() - 8, workc1.end(), std::numeric_limits<T>::infinity());

    // Based on what algorithms we are using, allocate the remaining memory
    try {

        switch (algorithm) {
        case elkan:
            workcc1.resize(n_clusters * n_clusters, 0.0);
            workcs1.resize(n_samples * (n_clusters + 8), 0.0);
            works1.resize(n_samples, 0.0);
            break;
        case macqueen:
            workcs1.resize(max_block_size * n_clusters, 0.0);
            workc2.resize(n_clusters, 0.0);
            break;
        case lloyd:
            workcs1.resize(max_block_size * (n_clusters + 8) * n_threads, 0.0);
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

    // Run k-means algorithm n_init times and select the run with the lowest inertia
    for (da_int run = 0; run < n_init; run++) {

        // Initialize the centres if needed
        kmeans<T>::initialize_centres();

        // Perform k-means using current_inertia, current_cluster_centres and current_labels
        kmeans<T>::perform_kmeans();

        // Check if it's the best run yet
        if (current_inertia < best_inertia) {
            best_inertia = current_inertia;
            best_n_iter = current_n_iter;
            // If this run hit the maximum number of iterations, a warning is required
            warn_maxit_reached = (converged == 0) ? true : false;
            std::swap(best_cluster_centres, current_cluster_centres);
            std::swap(best_labels, current_labels);
        }
    }

    // Compute the squared norms of the cluster centres in preparation for the predict phase of the algorithm; store in workc1
    for (da_int i = 0; i < n_clusters; i++) {
        workc1[i] = (T)0.0;
    }

    T tmp;
    for (da_int j = 0; j < n_features; j++) {
        for (da_int i = 0; i < n_clusters; i++) {
            tmp = (*best_cluster_centres)[i + j * n_clusters];
            workc1[i] += tmp * tmp;
        }
    }

    iscomputed = true;

    if (warn_maxit_reached)
        return da_warn(this->err, da_status_maxit,
                       "The maximum number of iterations was reached.");

    return status;
}

template <typename T>
da_status kmeans<T>::transform(da_int m_samples, da_int m_features, const T *X,
                               da_int ldx, T *X_transform, da_int ldx_transform) {

    if (!iscomputed) {
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

    const T *X_temp;
    T *utility_ptr1;
    T *utility_ptr2;
    da_int ldx_temp;
    T *X_transform_temp;
    da_int ldx_transform_temp;

    da_status status =
        this->store_2D_array(m_samples, m_features, X, ldx, &utility_ptr1, &X_temp,
                             ldx_temp, "m_samples", "m_features", "X", "ldx");
    if (status != da_status_success)
        return status;

    status = this->store_2D_array(
        m_samples, n_clusters, X_transform, ldx_transform, &utility_ptr2,
        const_cast<const T **>(&X_transform_temp), ldx_transform_temp, "m_samples",
        "n_clusters", "X_transform", "ldx_transform", 1);
    if (status != da_status_success)
        return status;

    std::vector<T> x_work;

    try {
        x_work.resize(m_samples);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    // Compute m_samples x n_clusters matrix of distances to cluster centres
    ARCH::euclidean_distance(column_major, m_samples, n_clusters, n_features, X_temp,
                             ldx_temp, (*best_cluster_centres).data(), n_clusters,
                             X_transform_temp, ldx_transform_temp, x_work.data(), 2,
                             workc1.data(), 1, false, false);

    if (this->order == row_major) {

        da_utils::copy_transpose_2D_array_column_to_row_major(
            m_samples, n_clusters, X_transform_temp, ldx_transform_temp, X_transform,
            ldx_transform);

        delete[] (utility_ptr1);
        delete[] (utility_ptr2);
    }

    return da_status_success;
}

template <typename T>
da_status kmeans<T>::predict(da_int k_samples, da_int k_features, const T *Y, da_int ldy,
                             da_int *Y_labels) {

    if (!iscomputed) {
        return da_warn(
            this->err, da_status_no_data,
            "The k-means has not been computed. Please call da_kmeans_compute_s or "
            "da_kmeans_compute_d.");
    }

    const T *Y_temp;
    T *utility_ptr;
    da_int ldy_temp;

    da_status status =
        this->store_2D_array(k_samples, k_features, Y, ldy, &utility_ptr, &Y_temp,
                             ldy_temp, "k_samples", "k_features", "Y", "ldy");
    if (status != da_status_success)
        return status;

    // Check for illegal output arguments
    if (Y_labels == nullptr)
        return da_error(this->err, da_status_invalid_pointer,
                        "The array Y_labels is null.");

    // Compute nearest cluster centre for each sample in Y; essentially a single blocked step of the Lloyd iteration.
    std::vector<T> y_work;

    max_block_size = std::min(KMEANS_LLOYD_BLOCK_SIZE, k_samples);

    da_utils::blocking_scheme(k_samples, max_block_size, n_blocks, block_rem);

    da_int n_threads = da_utils::get_n_threads_loop(n_blocks);

    da_int ldy_work;

    try {
        y_work.resize(max_block_size * (n_clusters + 8) * n_threads);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    if (n_clusters < 4) {
        ldy_work = n_clusters + 8;
        predict_block = &kmeans<T>::lloyd_iteration_block_no_unroll;
    } else if (n_clusters < 6) {
        ldy_work = max_block_size;
        predict_block = &kmeans<T>::lloyd_iteration_block_unroll_4_T;
    } else if (n_clusters < 16) {
        predict_block = &kmeans<T>::lloyd_iteration_block_unroll_4;
        ldy_work = n_clusters + 8;
    } else {
        predict_block = &kmeans<T>::lloyd_iteration_block_unroll_8;
        ldy_work = n_clusters + 8;
    }

    T *dummy = nullptr;
    da_int *dummy_int = nullptr;
    da_int block_index;
    da_int block_size = max_block_size;

#pragma omp parallel firstprivate(block_size) private(block_index)                       \
    shared(n_blocks, block_rem, k_samples, max_block_size, Y_temp, ldy_temp,             \
               best_cluster_centres, dummy, workc1, dummy_int, Y_labels, y_work,         \
               ldy_work) default(none) num_threads(n_threads)
    {
        da_int y_work_index =
            ((da_int)omp_get_thread_num()) * max_block_size * (n_clusters + 8);
#pragma omp for schedule(dynamic)
        for (da_int i = 0; i < n_blocks; i++) {
            if (i == n_blocks - 1 && block_rem > 0) {
                block_index = k_samples - block_rem;
                block_size = block_rem;
            } else {
                block_index = i * max_block_size;
            }
            (this->*predict_block)(false, block_size, &Y_temp[block_index], ldy_temp,
                                   (*best_cluster_centres).data(), dummy, workc1.data(),
                                   dummy_int, &Y_labels[block_index],
                                   &y_work[y_work_index], ldy_work);
        }
    }

    if (this->order == row_major)
        delete[] (utility_ptr);

    return da_status_success;
}

/* Initialization function for Elkan's algorithm */
template <typename T> void kmeans<T>::init_elkan() {
    ldworkcs1 = n_clusters + 8;
    if (n_clusters < 4) {
        elkan_iteration_update_block = &kmeans<T>::elkan_iteration_update_block_no_unroll;
    } else if (n_clusters < 16) {
        elkan_iteration_update_block = &kmeans<T>::elkan_iteration_update_block_unroll_4;
    } else {
        elkan_iteration_update_block = &kmeans<T>::elkan_iteration_update_block_unroll_8;
    }
    init_elkan_bounds();
    single_iteration = &kmeans<T>::elkan_iteration;
}

/* Initialize the upper and lower bounds for Elkan's method; stored in works1 and workcs1 */
template <typename T> void kmeans<T>::init_elkan_bounds() {

    compute_centre_half_distances();
    da_int label;
    da_int tmp_int;
    T smallest_dist, dist, tmp;

// For every sample, set upper bound (works1) to be distance to closest centre and update label
// Lower bound (workcs1) will contain distance from each sample to each cluster centre, if computed
#pragma omp parallel for schedule(dynamic, KMEANS_ELKAN_BLOCK_SIZE) private(             \
        label, tmp_int, smallest_dist, dist, tmp)                                        \
    shared(A, lda, current_cluster_centres, n_clusters, workcc1, workcs1, ldworkcs1,     \
               works1, current_labels) default(none)
    for (da_int i = 0; i < n_samples; i++) {

        da_int index = i * ldworkcs1;
        label = 0;
        smallest_dist = (T)0.0;

#pragma omp simd reduction(+ : smallest_dist)
        for (da_int k = 0; k < n_features; k++) {
            tmp = A[i + k * lda] - (*current_cluster_centres)[k * n_clusters];
            smallest_dist += tmp * tmp;
        }

        smallest_dist = std::sqrt(smallest_dist);
        workcs1[index] = smallest_dist;

        for (da_int j = 1; j < n_clusters; j++) {
            // Compute distance between the ith sample and the jth centre only if needed
            workcs1[index + j] = (T)0.0;
            tmp_int = label * n_clusters + j;
            if (smallest_dist > workcc1[tmp_int]) {

                dist = (T)0.0;
#pragma omp simd reduction(+ : dist)
                for (da_int k = 0; k < n_features; k++) {
                    tmp = A[i + k * lda] - (*current_cluster_centres)[j + k * n_clusters];
                    dist += tmp * tmp;
                }
                dist = std::sqrt(dist);
                workcs1[index + j] = dist;

                if (dist < smallest_dist) {
                    label = j;
                    smallest_dist = dist;
                }
            }
        }

        (*current_labels)[i] = label;
        works1[i] = smallest_dist;
    }
}

/* Perform a single iteration of Elkan's method */
template <typename T>
void kmeans<T>::elkan_iteration(bool update_centres, da_int n_threads) {

    if (update_centres) {
        for (da_int j = 0; j < n_clusters; j++)
            cluster_count[j] = 0;

        for (da_int j = 0; j < n_clusters * n_features; j++)
            (*current_cluster_centres)[j] = (T)0.0;

        if (n_threads > 1) {
            for (da_int j = 0; j < n_clusters * n_threads; j++)
                work_int1[j] = 0;

            for (da_int j = 0; j < n_clusters * n_features * n_threads; j++)
                thread_cluster_centres[j] = (T)0.0;
        }
    }

    // At this point workc1 contains distance of each cluster centre to the next nearest
    // The latest labels and centres are in 'previous' so we can update them to current

    da_int block_size = max_block_size;
    da_int block_index;
    if (n_threads > 1) {

        omp_lock_t cluster_count_lock, cluster_centres_lock;
        omp_init_lock(&cluster_count_lock);
        omp_init_lock(&cluster_centres_lock);

#pragma omp parallel shared(                                                             \
        thread_cluster_centres, work_int1, n_blocks, block_rem, update_centres, A, lda,  \
            previous_cluster_centres, current_cluster_centres, cluster_count, workc1,    \
            workcc1, ldworkcs1, max_block_size, current_labels, previous_labels, works1, \
            workcs1, cluster_count_lock, cluster_centres_lock)                           \
    firstprivate(block_size) private(block_index) default(none) num_threads(n_threads)
        {
            da_int work_int1_index = (da_int)omp_get_thread_num() * n_clusters;
            da_int thread_cluster_centres_index =
                (da_int)omp_get_thread_num() * n_clusters * n_features;
#pragma omp for schedule(dynamic) nowait
            for (da_int i = 0; i < n_blocks; i++) {
                if (i == n_blocks - 1 && block_rem > 0) {
                    block_index = n_samples - block_rem;
                    block_size = block_rem;
                } else {
                    block_index = i * max_block_size;
                }
                elkan_iteration_assign_block(
                    update_centres, block_size, &A[block_index], lda,
                    (*previous_cluster_centres).data(),
                    &thread_cluster_centres[thread_cluster_centres_index],
                    &works1[block_index], &workcs1[block_index * ldworkcs1], ldworkcs1,
                    &(*previous_labels)[block_index], &(*current_labels)[block_index],
                    workcc1.data(), workc1.data(), &work_int1[work_int1_index]);
            }
            // Now aggregate work_int1 into cluster_count and thread_cluster_centres into current_cluster_centres
            // The while loop is used because we don't mind what order each thread executes the two critical regions
            bool reduced_cluster_count = false, reduced_cluster_centres = false;
            while (!reduced_cluster_count || !reduced_cluster_centres) {
                if (!reduced_cluster_count) {

                    omp_set_lock(&cluster_count_lock);

                    for (da_int i = 0; i < n_clusters; i++) {
                        cluster_count[i] += work_int1[work_int1_index + i];
                    }
                    omp_unset_lock(&cluster_count_lock);
                    reduced_cluster_count = true;
                }
                if (!reduced_cluster_centres) {
                    omp_set_lock(&cluster_centres_lock);
                    for (da_int i = 0; i < n_clusters * n_features; i++) {
                        (*current_cluster_centres)[i] +=
                            thread_cluster_centres[thread_cluster_centres_index + i];
                    }
                    omp_unset_lock(&cluster_centres_lock);
                    reduced_cluster_centres = true;
                }
            }
        } // end parallel region
        omp_destroy_lock(&cluster_count_lock);
        omp_destroy_lock(&cluster_centres_lock);
    } else {

        for (da_int i = 0; i < n_blocks; i++) {
            if (i == n_blocks - 1 && block_rem > 0) {
                block_index = n_samples - block_rem;
                block_size = block_rem;
            } else {
                block_index = i * max_block_size;
            }
            elkan_iteration_assign_block(
                update_centres, block_size, &A[block_index], lda,
                (*previous_cluster_centres).data(), (*current_cluster_centres).data(),
                &works1[block_index], &workcs1[block_index * ldworkcs1], ldworkcs1,
                &(*previous_labels)[block_index], &(*current_labels)[block_index],
                workcc1.data(), workc1.data(), cluster_count.data());
        }
    }

    if (update_centres) {
        T tmp;

        scale_current_cluster_centres();

        // Update upper and lower bounds and compute shift in centres
        compute_centre_shift();
        for (da_int i = 0; i < n_clusters; i++) {
            T tmp2 = 0.0;
#pragma omp simd reduction(+ : tmp2)
            for (da_int j = 0; j < n_features; j++) {
                tmp = (*previous_cluster_centres)[i + j * n_clusters];
                tmp2 += tmp * tmp;
            }
            workc1[i] = std::sqrt(tmp2);
        }

        if (n_threads > 1) {
            block_size = max_block_size;
#pragma omp parallel for default(none) schedule(dynamic)                                 \
    shared(n_blocks, n_samples, workcs1, ldworkcs1, works1, workc1, current_labels)      \
    firstprivate(block_size) private(block_index)
            for (da_int i = 0; i < n_blocks; i++) {
                if (i == n_blocks - 1 && block_rem > 0) {
                    block_index = n_samples - block_rem;
                    block_size = block_rem;
                } else {
                    block_index = i * max_block_size;
                }
                (this->*elkan_iteration_update_block)(
                    block_size, &workcs1[block_index * ldworkcs1], ldworkcs1,
                    &works1[block_index], workc1.data(), &(*current_labels)[block_index]);
            }
        } else {

            (this->*elkan_iteration_update_block)(n_samples, workcs1.data(), ldworkcs1,
                                                  works1.data(), workc1.data(),
                                                  (*current_labels).data());
        }
    }

    compute_centre_half_distances();
}

/* Within Elkan iteration, assign a block of the labels*/
template <typename T>
void kmeans<T>::elkan_iteration_assign_block(
    bool update_centres, da_int block_size, const T *data, da_int lddata,
    T *old_cluster_centres, T *new_cluster_centres, T *u_bounds, T *l_bounds,
    da_int ldl_bounds, da_int *old_labels, da_int *new_labels, T *centre_half_distances,
    T *next_centre_distances, da_int *cluster_counts) {

    da_int l_bounds_index = 0;

    for (da_int i = 0; i < block_size; i++) {

        // New labels remain the same until we change them
        da_int label = old_labels[i];
        T u_bound = u_bounds[i];

        // This will be true if the upper and lower bounds are equal
        bool tight_bounds = false;

        // Only proceed if distance to closest centre exceeds 0.5* distance to next centre
        if (u_bound > next_centre_distances[label]) {

            for (da_int j = 0; j < n_clusters; j++) {
                // Check if this centre is a good candidate for relabelling the sample
                da_int centre_half_distances_index = label * n_clusters + j;
                T l_bound = l_bounds[l_bounds_index + j];
                T centre_half_distance =
                    centre_half_distances[centre_half_distances_index];

                if (j != label && u_bound > l_bound && u_bound > centre_half_distance) {

                    if (tight_bounds == false) {
                        // Get distance from sample point to currently assigned centre
                        u_bound = (T)0.0;
#pragma omp simd reduction(+ : u_bound)
                        for (da_int k = 0; k < n_features; k++) {
                            T tmp = data[i + k * lddata] -
                                    old_cluster_centres[label + k * n_clusters];
                            u_bound += tmp * tmp;
                        }
                        u_bound = std::sqrt(u_bound);
                        l_bounds[l_bounds_index + label] = u_bound;
                        tight_bounds = true;
                    }

                    // If condition still holds then compute distance to candidate centre and check
                    if (u_bound > l_bound || u_bound > centre_half_distance) {
                        T dist = (T)0.0;
#pragma omp simd reduction(+ : dist)
                        for (da_int k = 0; k < n_features; k++) {
                            T tmp = data[i + k * lddata] -
                                    old_cluster_centres[j + k * n_clusters];
                            dist += tmp * tmp;
                        }

                        dist = std::sqrt(dist);
                        l_bounds[l_bounds_index + j] = dist;
                        if (dist < u_bound) {
                            u_bound = dist;
                            label = j;
                        }
                    }
                }
            }
        }

        u_bounds[i] = u_bound;
        new_labels[i] = label;

        if (update_centres) {
            cluster_counts[label] += 1;
            // Add this sample to the cluster mean
            for (da_int j = 0; j < n_features; j++) {
                new_cluster_centres[label + j * n_clusters] += data[i + j * lddata];
            }
        }
        l_bounds_index += ldl_bounds;
    }
}

/* In the Elkan algorithm, compute the half distances between centres in current_cluster_centres and
   the distance to next closest centre. This matrix is symmetric so only the upper triangle is computed
   and stored. */
template <typename T> void kmeans<T>::compute_centre_half_distances() {
    T *dummy = nullptr;

    ARCH::euclidean_distance(column_major, n_clusters, n_clusters, n_features,
                             (*current_cluster_centres).data(), n_clusters, dummy, 0,
                             workcc1.data(), n_clusters, workc1.data(), 2, dummy, 0,
                             false, true);
    // For each centre, compute the half distance to next closest centre and store in workc1
    std::fill(workc1.begin(), workc1.begin() + n_clusters,
              std::numeric_limits<T>::infinity());

    for (da_int j = 0; j < n_clusters; j++) {
        for (da_int i = 0; i < j; i++) {
            T tmp = (T)0.5 * workcc1[i + j * n_clusters];
            // Update so we have centre half distances since euclidean_distance gave us whole distances
            workcc1[i + j * n_clusters] = tmp;
            if (tmp < workc1[i])
                workc1[i] = tmp;
            if (tmp < workc1[j])
                workc1[j] = tmp;
        }
    }
}

template <typename T> void kmeans<T>::init_lloyd() {
    single_iteration = &kmeans<T>::lloyd_iteration;
    ldworkcs1 = n_clusters + 8;
    if (n_clusters < 4) {
        lloyd_iteration_block = &kmeans<T>::lloyd_iteration_block_no_unroll;
    } else if (n_clusters < 6) {
        ldworkcs1 = max_block_size;
        lloyd_iteration_block = &kmeans<T>::lloyd_iteration_block_unroll_4_T;
    } else if (n_clusters < 16) {
        lloyd_iteration_block = &kmeans<T>::lloyd_iteration_block_unroll_4;
    } else {
        lloyd_iteration_block = &kmeans<T>::lloyd_iteration_block_unroll_8;
    }
}

/* Perform a single iteration of Lloyd's method */
template <typename T>
void kmeans<T>::lloyd_iteration(bool update_centres, da_int n_threads) {

    if (update_centres) {
        for (da_int j = 0; j < n_clusters; j++)
            cluster_count[j] = 0;

        for (da_int j = 0; j < n_clusters * n_features; j++)
            (*current_cluster_centres)[j] = (T)0.0;

        if (n_threads > 1) {
            for (da_int j = 0; j < n_clusters * n_threads; j++)
                work_int1[j] = 0;
            for (da_int j = 0; j < n_clusters * n_features * n_threads; j++)
                thread_cluster_centres[j] = (T)0.0;
        }
    }

    // Compute the squared norms of the previous cluster centres to avoid recomputing them repeatedly in the blocked section
    for (da_int i = 0; i < n_clusters; i++) {
        workc1[i] = (T)0.0;
    }

    T tmp;
    for (da_int j = 0; j < n_features; j++) {
        for (da_int i = 0; i < n_clusters; i++) {
            tmp = (*previous_cluster_centres)[i + j * n_clusters];
            workc1[i] += tmp * tmp;
        }
    }

    // Distance matrix part of the computation needs to be done in blocks since it is memory intensive
    da_int block_index;
    da_int block_size = max_block_size;

    if (n_threads > 1) {

        omp_lock_t cluster_count_lock, cluster_centres_lock;
        omp_init_lock(&cluster_count_lock);
        omp_init_lock(&cluster_centres_lock);

#pragma omp parallel shared(n_blocks, block_rem, update_centres, A, lda,                 \
                                previous_cluster_centres, current_cluster_centres,       \
                                cluster_count, current_labels, workc1, workcs1,          \
                                ldworkcs1, max_block_size, thread_cluster_centres,       \
                                work_int1, cluster_centres_lock, cluster_count_lock)     \
    firstprivate(block_size) private(block_index) default(none) num_threads(n_threads)
        {
            da_int work_int1_index = (da_int)omp_get_thread_num() * n_clusters;
            da_int thread_cluster_centres_index =
                (da_int)omp_get_thread_num() * n_clusters * n_features;
            da_int workcs1_index =
                ((da_int)omp_get_thread_num()) * max_block_size * (n_clusters + 8);
#pragma omp for nowait schedule(dynamic)
            for (da_int i = 0; i < n_blocks; i++) {
                if (i == n_blocks - 1 && block_rem > 0) {
                    block_index = n_samples - block_rem;
                    block_size = block_rem;
                } else {
                    block_index = i * max_block_size;
                }
                (this->*lloyd_iteration_block)(
                    update_centres, block_size, &A[block_index], lda,
                    (*previous_cluster_centres).data(),
                    &thread_cluster_centres[thread_cluster_centres_index], workc1.data(),
                    &work_int1[work_int1_index], &(*current_labels)[block_index],
                    &workcs1[workcs1_index], ldworkcs1);
            }
            // Now aggregate work_int1 into cluster_count and thread_cluster_centres into current_cluster_centres
            // The while loop is used because we don't mind what order each thread executes the two critical regions
            bool reduced_cluster_count = false, reduced_cluster_centres = false;
            while (!reduced_cluster_count || !reduced_cluster_centres) {
                if (!reduced_cluster_count) {

                    omp_set_lock(&cluster_count_lock);
                    for (da_int i = 0; i < n_clusters; i++) {
                        cluster_count[i] += work_int1[work_int1_index + i];
                    }
                    omp_unset_lock(&cluster_count_lock);
                    reduced_cluster_count = true;
                }
                if (!reduced_cluster_centres) {
                    omp_set_lock(&cluster_centres_lock);
                    for (da_int i = 0; i < n_clusters * n_features; i++) {
                        (*current_cluster_centres)[i] +=
                            thread_cluster_centres[thread_cluster_centres_index + i];
                    }
                    omp_unset_lock(&cluster_centres_lock);
                    reduced_cluster_centres = true;
                }
            }
        } // end of parallel region
        omp_destroy_lock(&cluster_count_lock);
        omp_destroy_lock(&cluster_centres_lock);
    } else {

        for (da_int i = 0; i < n_blocks; i++) {
            if (i == n_blocks - 1 && block_rem > 0) {
                block_index = n_samples - block_rem;
                block_size = block_rem;
            } else {
                block_index = i * max_block_size;
            }
            (this->*lloyd_iteration_block)(
                update_centres, block_size, &A[block_index], lda,
                (*previous_cluster_centres).data(), (*current_cluster_centres).data(),
                workc1.data(), cluster_count.data(), &(*current_labels)[block_index],
                workcs1.data(), ldworkcs1);
        }
    }

    if (update_centres) {
        scale_current_cluster_centres();

        // Compute change in centres in this iteration
        compute_centre_shift();
    }
}

/* Scaling phase for the current cluster centres; part of both the Elkan and Lloyd algorithms */
template <typename T> void kmeans<T>::scale_current_cluster_centres() {
    // Guard against empty clusters - avoid division by zero below
    for (da_int i = 0; i < n_clusters; i++) {
        if (cluster_count[i] == 0)
            cluster_count[i] = 1;
    }

// Scale to get proper column means (cluster_count contains the number of data points in each cluster)
#pragma omp simd collapse(2)
    for (da_int j = 0; j < n_features; j++) {
        for (da_int i = 0; i < n_clusters; i++) {
            (*current_cluster_centres)[i + j * n_clusters] /= cluster_count[i];
        }
    }
}

/* Initialization for MacQueen's method */
template <typename T> void kmeans<T>::init_macqueen() {
    ldworkcs1 = n_clusters;

    single_iteration = &kmeans<T>::macqueen_iteration;

    for (da_int j = 0; j < n_clusters; j++) {
        cluster_count[j] = 0; // Initialize to zero for use later
    }

    for (da_int i = 0; i < n_clusters * n_features; i++)
        (*previous_cluster_centres)[i] = (*current_cluster_centres)[i];

    // Compute the squared norms of the initial cluster centres to avoid recomputing them repeatedly in the blocked section; store in workc1
    for (da_int i = 0; i < n_clusters; i++) {
        workc1[i] = (T)0.0;
    }

    T tmp;
    for (da_int j = 0; j < n_features; j++) {
        for (da_int i = 0; i < n_clusters; i++) {
            tmp = (*current_cluster_centres)[i + j * n_clusters];
            (*previous_cluster_centres)[i + j * n_clusters] = tmp;
            (*current_cluster_centres)[i + j * n_clusters] = 0.0;
            workc1[i] += tmp * tmp;
        }
    }

    // Distance matrix computation needs to be done in blocks due to memory use
    for (da_int i = 0; i < n_blocks; i++) {
        if (i == n_blocks - 1 && block_rem > 0) {
            init_macqueen_block(block_rem, n_samples - block_rem);
        } else {
            init_macqueen_block(max_block_size, i * max_block_size);
        }
    }

    // Finish updating cluster centres - being careful to guard against zero division in empty clusters
    for (da_int j = 0; j < n_features; j++) {
        for (da_int i = 0; i < n_clusters; i++) {
            if (cluster_count[i] > 0)
                (*current_cluster_centres)[i + j * n_clusters] /= cluster_count[i];
        }
    }

    // Re-zero previous clusters, which were used temporarily here
    for (da_int i = 0; i < n_clusters * n_features; i++)
        (*previous_cluster_centres)[i] = 0;
}

/* Chunked part of MacQueen's method initialization */
template <typename T>
void kmeans<T>::init_macqueen_block(da_int block_size, da_int block_index) {

    // Compute the matrix D where D_{ij} = ||C_j||^2 - 2 A C^T
    // Don't form it explicitly though: just form -2AC^T and add the ||C_j||^2 as and when we need them
    // Array access patterns mean for this loop it is quicker to form -2CA^T

    T tmp_dist;

    da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, n_clusters, block_size,
                        n_features, -2.0, (*previous_cluster_centres).data(), n_clusters,
                        &A[block_index], lda, 0.0, workcs1.data(), ldworkcs1);

    for (da_int i = block_index; i < block_index + block_size; i++) {
        T smallest_dist = workcs1[i - block_index] + workc1[0];
        da_int index = (i - block_index) * ldworkcs1;
        da_int label = 0;
        for (da_int j = 1; j < n_clusters; j++) {
            tmp_dist = workcs1[index + j] + workc1[j];
            if (tmp_dist < smallest_dist) {
                label = j;
                smallest_dist = tmp_dist;
            }
        }
        (*current_labels)[i] = label;
        // Also want to be counting number of points in each initial cluster
        cluster_count[label] += 1;

        // Update clusters now that we have assigned points to them
        for (da_int j = 0; j < n_features; j++) {
            (*current_cluster_centres)[label + j * n_clusters] += A[i + j * lda];
        }
    }
}

/* Perform single iteration of MacQueen's method */
template <typename T>
void kmeans<T>::macqueen_iteration(bool update_centres,
                                   [[maybe_unused]] da_int n_threads) {

    // Copy data from previous iteration since it's updated in place; no way round this since we need previous iteration for convergence test
    for (da_int i = 0; i < n_clusters * n_features; i++)
        (*current_cluster_centres)[i] = (*previous_cluster_centres)[i];

    for (da_int i = 0; i < n_samples; i++)
        (*current_labels)[i] = (*previous_labels)[i];

    for (da_int i = 0; i < n_samples; i++) {
        // For sample point i, compute the cluster centre distances in workc2

        T *dummy = nullptr;
        T tmp;
        ARCH::euclidean_distance(column_major, 1, n_clusters, n_features, &A[i], lda,
                                 (*current_cluster_centres).data(), n_clusters,
                                 workc2.data(), 1, dummy, 0, workc1.data(), 1, true,
                                 false);

        T smallest_dist = workc2[0];
        da_int closest_centre = 0;
        for (da_int j = 1; j < n_clusters; j++) {
            if (workc2[j] < smallest_dist) {
                smallest_dist = workc2[j];
                closest_centre = j;
            }
        }

        if ((*current_labels)[i] != closest_centre) {
            da_int old_centre = (*current_labels)[i];
            (*current_labels)[i] = closest_centre;

            if (update_centres) {
                // Now need to update the two affected centres: closest_centre and old_centre
                cluster_count[closest_centre] += 1;
                cluster_count[old_centre] -= 1;
                workc1[old_centre] = (T)0.0;
                workc1[closest_centre] = (T)0.0;

                // Clear closest_centre and old_centre cluster centres ahead of recomputation
                for (da_int j = 0; j < n_features; j++) {
                    (*current_cluster_centres)[old_centre + j * n_clusters] = (T)0.0;
                    (*current_cluster_centres)[closest_centre + j * n_clusters] = (T)0.0;
                }

                for (da_int k = 0; k < n_samples; k++) {
                    if ((*current_labels)[k] == closest_centre) {
                        for (da_int j = 0; j < n_features; j++) {
                            (*current_cluster_centres)[closest_centre + j * n_clusters] +=
                                A[k + j * lda];
                        }
                    } else if ((*current_labels)[k] == old_centre) {
                        for (da_int j = 0; j < n_features; j++) {
                            (*current_cluster_centres)[old_centre + j * n_clusters] +=
                                A[k + j * lda];
                        }
                    }
                }

                // Scale to get proper mean and update the squared centre norms
                for (da_int j = 0; j < n_features; j++) {
                    if (cluster_count[old_centre] > 0) {
                        (*current_cluster_centres)[old_centre + j * n_clusters] /=
                            cluster_count[old_centre];
                        tmp = (*current_cluster_centres)[old_centre + j * n_clusters];
                        workc1[old_centre] += tmp * tmp;
                    }
                    if (cluster_count[closest_centre] > 0) {
                        (*current_cluster_centres)[closest_centre + j * n_clusters] /=
                            cluster_count[closest_centre];
                        tmp = (*current_cluster_centres)[closest_centre + j * n_clusters];
                        workc1[closest_centre] += tmp * tmp;
                    }
                }
            }
        }
    }

    if (update_centres) {
        // Compute change in centres in this iteration
        compute_centre_shift();
    }
}

template <typename T> void kmeans<T>::perform_hartigan_wong() {
    // Based on MIT licensed open-source implementation
    da_int ifault;

    kmns(A, n_samples, n_features, lda, &(*current_cluster_centres)[0], n_clusters,
         &(*current_labels)[0], work_int1.data(), max_iter, workc1.data(), &ifault,
         &current_n_iter, work_int2.data(), workc2.data(), workc3.data(),
         &(*previous_labels)[0], works1.data(), work_int3.data(), work_int4.data());
    // Record if it converged or ran into maximum number of iterations
    converged = (ifault == 2) ? 0 : 1;
    current_inertia = (T)0.0;
    // Hartigan-Wong implementation counts from 1 rather than 0, so correct this
    for (da_int i = 0; i < n_samples; i++)
        (*current_labels)[i] -= 1;
    for (da_int i = 0; i < n_clusters; i++)
        current_inertia += workc1[i];
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

    (this->*initialize_algorithm)();

    for (current_n_iter = 0; current_n_iter < max_iter; current_n_iter++) {
        // Start with the 'old' centres stored in previous_cluster_centres
        std::swap(previous_cluster_centres, current_cluster_centres);
        std::swap(previous_labels, current_labels);

        (this->*single_iteration)(true, n_threads);

        // Check for convergence
        converged = convergence_test();
        if (converged > 0) {
            break;
        }
    }

    if (converged == 1) {
        // Tolerance-based convergence: means we should rerun labelling step without recomputing centres
        std::swap(previous_labels, current_labels);
        std::swap(previous_cluster_centres, current_cluster_centres);
        // Perform one more iteration to update labels, but without updating the cluster centres
        (this->*single_iteration)(false, n_threads);
        std::swap(previous_cluster_centres, current_cluster_centres);
    }

    // Finished this run, so compute current_inertia
    compute_current_inertia();
}

/* Compute current_inertia based on the current_cluster_centres */
template <typename T> void kmeans<T>::compute_current_inertia() {
    current_inertia = 0;
    T tmp;
    for (da_int j = 0; j < n_features; j++) {
        for (da_int i = 0; i < n_samples; i++) {
            da_int label = (*current_labels)[i];
            tmp = A[i + j * lda] - (*current_cluster_centres)[label + j * n_clusters];
            current_inertia += tmp * tmp;
        }
    }
}

/* Compute the difference between the current and previous centres and store in previous_cluster_centres */
template <typename T> void kmeans<T>::compute_centre_shift() {

    // Before overwriting previous_cluster_centres, compute and store its norm, for use in convergence test
    char norm = 'F';
    normc = da::lange(&norm, &n_clusters, &n_features, (*previous_cluster_centres).data(),
                      &n_clusters, nullptr);

    for (da_int i = 0; i < n_clusters * n_features; i++) {
        (*previous_cluster_centres)[i] -= (*current_cluster_centres)[i];
    }
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

    char norm = 'F';
    if (da::lange(&norm, &n_clusters, &n_features, (*previous_cluster_centres).data(),
                  &n_clusters, nullptr) < tol * normc)
        convergence_test = 1;

    return convergence_test;
}

/* Initialize the centres, if needed, for the start of k-means computation*/
template <typename T> void kmeans<T>::initialize_centres() {
    std::fill(previous_cluster_centres->begin(), previous_cluster_centres->end(), 0.0);
    switch (init_method) {
    case random_samples: {
        // Select randomly (without replacement) from the data points
        std::iota(work_int2.begin(), work_int2.end(), 0);
        std::sample(work_int2.begin(), work_int2.end(), std::begin(work_int1), n_clusters,
                    mt_gen);
        for (da_int j = 0; j < n_clusters; j++) {
            for (da_int i = 0; i < n_features; i++) {
                (*current_cluster_centres)[i * n_clusters + j] =
                    A[i * lda + work_int1[j]];
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
            for (da_int j = 0; j < n_features; j++) {
                (*current_cluster_centres)[workcc1_index + j * n_clusters] +=
                    A[i + j * lda];
            }
        }

        scale_current_cluster_centres();

        break;
    }
    case kmeanspp: {
        kmeans_plusplus();
        break;
    }
    default:
        // No need to do anything as initial centres were provided and have been stored in current_cluster_centres already
        break;
    }
}

/* Initialize centres using k-means++ */
template <typename T> void kmeans<T>::kmeans_plusplus() {

    // Compute squared norms of the data points and store in works1
    for (da_int j = 0; j < n_features; j++) {
        for (da_int i = 0; i < n_samples; i++) {
            works1[i] += A[j * lda + i] * A[j * lda + i];
        }
    }

    da_int n_trials = 2 + (da_int)std::log(n_clusters);

    // Pick first centre randomly from the sample data points and store which one it was in work_int1
    std::uniform_int_distribution<> dis_int(0, n_samples - 1);
    da_int random_int = dis_int(mt_gen);
    work_int1[0] = random_int;
    for (da_int i = 0; i < n_features; i++) {
        (*current_cluster_centres)[i * n_clusters] = A[i * lda + work_int1[0]];
    }

    T dummy = (T)0.0;
    ARCH::euclidean_distance(column_major, n_samples, 1, n_features, A, lda,
                             (*current_cluster_centres).data(), n_clusters, works3.data(),
                             n_samples, works1.data(), 1, &dummy, 2, true, false);

    // Numerical errors could cause one of the distances to be slightly negative, leading to undefined behaviour in std::discrete_distribution
    works3[random_int] = (T)0.0;

    // Need to catch an edge case where all points are the same
    bool coincident_points = true;

    for (int i = 0; i < n_samples; i++) {
        if (works3[i] > (T)0.0) {
            coincident_points = false;
            break;
        }
    }

    if (coincident_points) {
        // Doesn't matter which ones we choose, this is just to prevent exceptions later, so just use the first ones
        for (da_int j = 0; j < n_features; j++) {
            for (da_int k = 0; k < n_clusters; k++) {
                (*current_cluster_centres)[j * n_clusters + k] = A[j * lda + k];
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
                    break;

                // Calculate cost function for this candidate point
                T current_cost = (T)0.0;
                da_int current_candidate = work_int2[trials];

                // Compute the distance from each point to the candidate centre and store in works4
                ARCH::euclidean_distance(column_major, n_samples, 1, n_features, A, lda,
                                         &A[current_candidate], lda, works4.data(),
                                         n_samples, works1.data(), 1,
                                         &works1[current_candidate], 1, true, false);
                // Get minimum squared distance of each sample point to potential centre
                current_cost = 0;
                for (da_int j = 0; j < n_samples; j++) {
                    works5[j] = std::min(works3[j], works4[j]);
                    current_cost += works5[j];
                }

                if (current_cost < best_candidate_cost) {
                    best_candidate_cost = current_cost;
                    best_candidate = work_int2[trials];
                    for (da_int j = 0; j < n_samples; j++) {
                        works2[j] = works5[j];
                    }
                }
            }

            // Place the best candidate as the next cluster centre
            for (da_int i = 0; i < n_features; i++) {
                (*current_cluster_centres)[i * n_clusters + k] =
                    A[i * lda + best_candidate];
            }
            work_int1[k] = best_candidate;
            for (da_int j = 0; j < n_samples; j++) {
                works3[j] = works2[j];
            }
            // Guard against negative probabilities again
            works3[best_candidate] = (T)0.0;
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

template class kmeans<double>;
template class kmeans<float>;

} // namespace da_kmeans

} // namespace ARCH