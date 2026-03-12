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

#include "dbscan.hpp"
#include "aoclda.h"
#include "context.hpp"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "da_omp.hpp"
#include "da_std.hpp"
#include "da_vector.hpp"
#include "dbscan_options.hpp"
#include "lapack_templates.hpp"
#include "macros.h"
#include "miscellaneous.hpp"
#include "pairwise_distances.hpp"
#include "radius_neighbors.hpp"
#include <algorithm>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

#define NOISE -1
#define UNVISITED -2

namespace ARCH {

namespace da_dbscan {

using namespace da_neighbors_types;
using namespace std::literals::string_literals;

template <typename T> dbscan<T>::~dbscan() {
    // Destructor needs to handle arrays that were allocated due to row major storage of input data
    if (A_temp)
        delete[] (A_temp);
}

template <typename T>
dbscan<T>::dbscan(da_errors::da_error_t &err) : basic_handle<T>(err) {
    // Initialize the options registry
    // Any error is stored err->status[.] and this needs to be checked
    // by the caller.
    register_dbscan_options<T>(this->opts, *this->err);
};

template <typename T>
da_status dbscan<T>::get_result(da_result query, da_int *dim, T *result) {
    // Don't return anything if DBSCAN has not been computed
    if (!iscomputed) {
        return da_warn(this->err, da_status_no_data,
                       "DBSCAN clustering has not yet been computed. Please call "
                       "da_dbscan_compute_s "
                       "or da_dbscan_compute_d before extracting results.");
    }

    da_int rinfo_size = 9;

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
        result[2] = (T)lda_in;
        result[3] = eps;
        result[4] = (T)min_samples;
        result[5] = (T)leaf_size;
        result[6] = p;
        result[7] = (T)n_core_samples;
        result[8] = (T)n_clusters;
        break;
    default:
        return da_warn(this->err, da_status_unknown_query,
                       "The requested result could not be found.");
    }
    return da_status_success;
};

template <typename T>
da_status dbscan<T>::get_result(da_result query, da_int *dim, da_int *result) {
    // Don't return anything if DBSCAN has not been computed
    if (!iscomputed) {
        return da_warn(this->err, da_status_no_data,
                       "DBSCAN clustering has not yet been computed. Please call "
                       "da_dbscan_compute_s "
                       "or da_dbscan_compute_d before extracting results.");
    }

    switch (query) {
    case da_result::da_dbscan_labels:
        if (*dim < n_samples) {
            *dim = n_samples;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(n_samples) + ".");
        }
        for (da_int i = 0; i < n_samples; i++)
            result[i] = labels[i];
        break;
    case da_result::da_dbscan_core_sample_indices: {
        if (*dim < n_core_samples) {
            *dim = n_core_samples;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(n_core_samples) + ".");
        }
        da_int idx = 0;
        for (da_int i = 0; i < n_samples; i++) {
            if (is_core_sample[i]) {
                result[idx++] = i;
            }
        }
        break;
    }
    case da_result::da_dbscan_n_clusters:
        *result = n_clusters;
        break;
    case da_result::da_dbscan_n_core_samples:
        *result = n_core_samples;
        break;
    default:
        return da_warn(this->err, da_status_unknown_query,
                       "The requested result could not be found.");
    }

    return da_status_success;
};

template <typename T> void dbscan<T>::refresh() {
    if (A_temp) {
        delete[] (A_temp);
        A_temp = nullptr;
    }
    iscomputed = false;
    neighbors.clear();
    labels.clear();
    is_core_sample.clear();
    n_core_samples = 0;
    n_clusters = 0;
}

/* Store details about user's data matrix in preparation for DBSCAN computation */
template <typename T>
da_status dbscan<T>::set_data(da_int n_samples, da_int n_features, const T *A_in,
                              da_int lda_in) {

    // Guard against errors due to multiple calls using the same class instantiation
    refresh();

    da_status status =
        this->store_2D_array(n_samples, n_features, A_in, lda_in, &A_temp, &A, lda,
                             "n_samples", "n_features", "A", "lda");
    if (status != da_status_success)
        return status; // LCOV_EXCL_LINE

    // Store dimensions of A
    this->n_samples = n_samples;
    this->n_features = n_features;
    this->lda_in = lda_in;

    // Record that initialization is complete but computation has not yet been performed
    initdone = true;
    iscomputed = false;

    return da_status_success;
}

/* Compute the DBSCAN clusters */
template <typename T> da_status dbscan<T>::compute() {
    da_status status = da_status_success;
    if (initdone == false)
        return da_error(this->err, da_status_no_data,
                        "No data has been passed to the handle. Please call "
                        "da_dbscan_set_data_s or da_dbscan_set_data_d.");

    // Read in options and store in class

    this->opts.get("eps", eps);

    this->opts.get("min samples", min_samples);

    this->opts.get("leaf size", leaf_size);

    this->opts.get("power", p);

    std::string opt_tmp;
    this->opts.get("algorithm", opt_tmp, algorithm);

    this->opts.get("metric", opt_tmp, metric);

    // Allocate memory
    try {
        labels.resize(n_samples);
        neighbors.resize(n_samples);
        is_core_sample.resize(n_samples);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    da_std::fill(is_core_sample.begin(), is_core_sample.end(), false);
    metric_internal = da_metric(metric);

    // Check for incompatible options
    if (algorithm == kd_tree || algorithm == ball_tree) {
        if (metric == da_cosine || metric == da_sqeuclidean_gemm) {
            return da_error(this->err, da_status_incompatible_options,
                            "Tree algorithms are not compatible with the cosine or "
                            "squared Euclidean distances.");
        } else if (metric == da_minkowski && p < (T)1.0) {
            return da_error(this->err, da_status_incompatible_options,
                            "Tree algorithms are not compatible with the Minkowski "
                            "metric when p < 1.");
        }
    }

    alg_internal = da_neighbors_types::nn_algorithm(algorithm);
    if (alg_internal == automatic) {
        // If the user has not specified an algorithm, we will use the k-d tree if the data is small
        // in dimension and the Minkowski options allow it. Otherwise we will use brute force.
        if (n_features <= 15 && !(metric_internal == da_minkowski && p < (T)1.0)) {
            alg_internal = kd_tree;
        } else {
            alg_internal = brute;
        }
    }

    // Form in neighbors the list of indices within the epsilon neighborhood of each sample point
    if (alg_internal == brute) {

        status = da_radius_neighbors::radius_neighbors_brute(
            n_samples, n_features, A, lda, eps, metric_internal, p, neighbors, this->err);

    } else if (alg_internal == kd_tree) {
        status = da_radius_neighbors::radius_neighbors_kd_tree(
            n_samples, n_features, A, lda, eps, metric_internal, p, leaf_size, neighbors,
            this->err);
    } else if (alg_internal == ball_tree) {
        status = da_radius_neighbors::radius_neighbors_ball_tree(
            n_samples, n_features, A, lda, eps, metric_internal, p, leaf_size, neighbors,
            this->err);
    }

    if (status != da_status_success)
        return da_error(this->err, status, // LCOV_EXCL_LINE
                        "Failed to compute radius neighbors prior to clustering.");
    status = dbscan_clusters();
    if (status != da_status_success)
        return da_error(this->err, status, // LCOV_EXCL_LINE
                        "Failed to compute DBSCAN clustering.");

    iscomputed = true;
    neighbors.clear(); // Free up memory since we no longer need this

    return status;
}

/* Compute the DBSCAN clusters using parallel method */
template <typename T> da_status dbscan<T>::dbscan_clusters_parallel() {

    da_status status = da_status_success;
    std::vector<da_int> remap;
    da_int next_id = 0;
    // Add telemetry to the context class
    context_set_hidden_settings("dbscan.setup"s, "clustering=parallel"s);

    try {
        duplicate_labels.resize(n_samples);
        remap.resize(n_samples, NOISE);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    da_std::fill(labels.begin(), labels.end(), NOISE);
    da_std::fill(duplicate_labels.begin(), duplicate_labels.end(), NOISE);

#pragma omp parallel default(none)                                                       \
    shared(labels, neighbors, n_clusters, n_core_samples, min_samples_m1, n_samples,     \
               status, duplicate_labels, is_core_sample, next_id, remap)
    {

// Parallel loop to compute DBSCAN clusters
#pragma omp for schedule(guided, 64)
        for (da_int i = 0; i < n_samples; i++) {
            if (is_core_sample[i]) {
                // This is a core point
                da_int tmp_label_i;
#pragma omp atomic read
                tmp_label_i = labels[i];
                if (i < tmp_label_i || tmp_label_i == NOISE) {
#pragma omp atomic write
                    // Assign it its own index as the cluster label - we will combine clusters later
                    labels[i] = i;
                }

                // Loop through each point in the epsilon neighborhood of point i
                auto &this_neighbor = neighbors[i];
                for (da_int j = 0; j < (da_int)this_neighbor.size(); j++) {
                    da_int sample_point_j = this_neighbor[j];
                    da_int tmp_label_j;
#pragma omp atomic read
                    tmp_label_j = labels[sample_point_j];
                    // Record that i and j are in the same cluster
                    if (i < sample_point_j)
                        duplicate_labels[sample_point_j] = i;
                    else
                        duplicate_labels[i] = sample_point_j;

                    if (i < tmp_label_j || tmp_label_j == NOISE) {
#pragma omp atomic write
                        labels[sample_point_j] = i;
                    }
                }
            }
        }

#pragma omp for schedule(guided, 64)
        for (da_int i = 0; i < n_samples; i++) {
            da_int current_label = labels[i];
            if (current_label == NOISE) {
                continue;
            }

            // Follow the chain of labels to find the lowest one
            while (duplicate_labels[current_label] != NOISE) {
                current_label = duplicate_labels[current_label];
            }

            labels[i] = current_label;
        }

        // Relabel clusters to have consecutive numbering starting from 0
#pragma omp single
        {
            for (da_int i = 0; i < n_samples; i++) {
                da_int lab = labels[i];
                if (lab == NOISE)
                    continue;
                if (remap[lab] == NOISE) {
                    remap[lab] = next_id;
                    next_id++;
                }
            }
            n_clusters = next_id;
        }

#pragma omp barrier
#pragma omp for schedule(static)
        for (da_int i = 0; i < n_samples; ++i) {
            da_int lab = labels[i];
            if (lab == NOISE)
                continue;
            labels[i] = remap[lab];
        }
    } // End of parallel region

    return status;
}

/* Compute the DBSCAN clusters using serial method */
template <typename T> da_status dbscan<T>::dbscan_clusters_serial() {

    // Add telemetry to the context class
    context_set_hidden_settings("dbscan.setup"s, "clustering=serial"s);
    da_status status = da_status_success;
    da_std::fill(labels.begin(), labels.end(), UNVISITED);

    // Serial loop for computing DBSCAN clusters
    try {
        for (da_int i = 0; i < n_samples; i++) {

            // If we've already looked at this point we can go to the next loop iteration
            if (labels[i] != UNVISITED)
                continue;

            // Find the neighbors of the current sample
            if (is_core_sample[i] == false) {
                //Epsilon neighborhood is too small to form a cluster; label as noise
                labels[i] = NOISE;
            } else {
                // Form a new cluster and label this as a core sample
                labels[i] = n_clusters;

                da_vector::da_vector<da_int> search_indices;
                // The epsilon neighbors of this point form the start of our search vector
                search_indices.append(neighbors[i]);

                for (da_int j = 0; j < (da_int)search_indices.size(); j++) {
                    da_int neigh = search_indices[j];

                    if (labels[neigh] == NOISE) {
                        // If the point was previously labeled as noise, it is now part of a cluster
                        labels[neigh] = n_clusters;
                        continue;
                    }

                    if (labels[neigh] != UNVISITED)
                        // If the point has already been visited we don't need to look at it
                        continue;

                    labels[neigh] = n_clusters;

                    if (is_core_sample[neigh]) {
                        // This point is also a core sample point so mark it as such and add its neighbors to the search vector
                        search_indices.append(neighbors[neigh]);
                    }
                }

                n_clusters++;
            }
        }

    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    return status;
}

/* Compute the DBSCAN clusters */
template <typename T> da_status dbscan<T>::dbscan_clusters() {
    da_status status = da_status_success;

    n_clusters = 0;
    // Work with min_samples - 1 since we are not counting points as being in their own neighbourhood
    min_samples_m1 = min_samples - 1;

    // Find core samples
    n_core_samples = 0;
    for (da_int i = 0; i < n_samples; i++) {
        if ((da_int)neighbors[i].size() >= min_samples_m1) {
            is_core_sample[i] = true;
            n_core_samples++;
        }
    }

    // Check to see if there is an override in the context class
    const char cluster_methods[]{"dbscan.cluster_methods"};
    if (context::get_context()->hidden_settings.find(cluster_methods) !=
        context::get_context()->hidden_settings.end()) {
        std::string cluster_method =
            context::get_context()->hidden_settings[cluster_methods];
        if (cluster_method == "parallel"s) {
            status = dbscan_clusters_parallel();
        } else {
            status = dbscan_clusters_serial();
        }
    } else {
        // If no override is found, we will use the parallel method if we have more than 32 threads available
        // Otherwise we will use the serial method
        if (omp_get_max_threads() > 1) {
            status = dbscan_clusters_parallel();
        } else {
            status = dbscan_clusters_serial();
        }
    }

    return status;
}

template class dbscan<double>;
template class dbscan<float>;

} // namespace da_dbscan

} // namespace ARCH