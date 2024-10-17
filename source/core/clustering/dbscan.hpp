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

#ifndef DBSCAN_HPP
#define DBSCAN_HPP

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "aoclda.h"
#include "basic_handle.hpp"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "da_omp.hpp"
#include "da_utils.hpp"
#include "da_vector.hpp"
#include "dbscan_options.hpp"
#include "dbscan_types.hpp"
#include "euclidean_distance.hpp"
#include "lapack_templates.hpp"
#include "radius_neighbors.hpp"
#include <algorithm>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace da_dbscan {

/* Utility function to add a rule in an unordered map. Recursively searches for rules with the same
   key and updates them accordingly. */
inline void add_to_label_map(std::unordered_map<da_int, da_int> &map, da_int key,
                             da_int value) {
    // Try to insert the key_value pair into the map
    if (value >= key)
        return;

    auto [it, inserted] = map.emplace(key, value);
    if (!inserted) {
        // A map with that key already exists, so update the value with minimum
        std::pair<da_int, da_int> tmp_values = std::minmax(it->second, value);
        it->second = tmp_values.first;
        // Recursively update the rest of the map: add a map to correct the larger value
        add_to_label_map(map, tmp_values.second, tmp_values.first);
    }
    return;
}

/* Utility function to merge unordered maps, keeping minimum values of duplicate keys.
   Used in parallel DBSCAN loop. */
inline void merge_unordered_maps(std::unordered_map<da_int, da_int> &map1,
                                 std::unordered_map<da_int, da_int> &map2) {
    //std::unordered_map<da_int, da_int> result = map1;
    for (const auto &[key, value] : map2) {
        //auto [it, inserted] = map1.emplace(key, value);
        //if (!inserted) {
        // key already exists so update value with minimum
        //  it->second = std::min(it->second, value);
        //}
        add_to_label_map(map1, key, value);
    }
    return;
}

/* DBSCAN class */
template <typename T> class da_dbscan : public basic_handle<T> {
  public:
    ~da_dbscan() {
        // Destructor needs to handle arrays that were allocated due to row major storage of input data
        if (A_temp)
            delete[] (A_temp);
    }

  private:
    da_int n_samples = 0;
    da_int n_features = 0;

    // Set true when initialization is complete
    bool initdone = false;

    // Set true when dbscan clustering is computed successfully
    bool iscomputed = false;

    // User's data
    const T *A = nullptr;
    da_int lda = 0;
    da_int lda_in = 0;

    // Utility pointer to column major allocated copy of user's data
    T *A_temp = nullptr;

    // Options
    T eps = 0.5;
    da_int min_samples = 5;
    da_int leaf_size = 30;
    T p = 2.0;

    da_int algorithm = brute;
    da_int metric = euclidean;

    // Scalar outputs
    da_int n_core_samples = 0;
    da_int n_clusters = 0;

    // Arrays containing output data
    da_vector::da_vector<da_int>
        core_sample_indices; // Use da_vector since we will be dynamically expanding this array
    std::vector<da_int> labels;

    // Internal arrays
    std::vector<da_vector::da_vector<da_int>>
        neighbors; // Use da_vector since we will be dynamically expanding this array

    da_status dbscan_clusters();

  public:
    da_dbscan(da_errors::da_error_t &err) : basic_handle<T>(err) {
        // Initialize the options registry
        // Any error is stored err->status[.] and this needs to be checked
        // by the caller.
        register_dbscan_options<T>(this->opts, *this->err);
    };

    da_status get_result(da_result query, da_int *dim, T *result) {
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

    da_status get_result(da_result query, da_int *dim, da_int *result) {
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
        case da_result::da_dbscan_core_sample_indices:
            if (*dim < n_core_samples) {
                *dim = n_core_samples;
                return da_warn(this->err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(n_core_samples) + ".");
            }
            for (da_int i = 0; i < n_core_samples; i++)
                result[i] = core_sample_indices[i];
            break;
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

    void refresh() {
        if (A_temp) {
            delete[] (A_temp);
            A_temp = nullptr;
        }
        iscomputed = false;
    }

    /* Store details about user's data matrix in preparation for DBSCAN computation */
    da_status set_data(da_int n_samples, da_int n_features, const T *A_in,
                       da_int lda_in) {

        // Guard against errors due to multiple calls using the same class instantiation
        refresh();

        da_status status =
            this->store_2D_array(n_samples, n_features, A_in, lda_in, &A_temp, &A, lda,
                                 "n_samples", "n_features", "A", "lda");
        if (status != da_status_success)
            return status;

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
    da_status compute() {

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

        // Currently we only support the brute-force method
        if (algorithm != brute && algorithm != automatic && algorithm != brute_serial) {
            return da_error(this->err, da_status_invalid_option,
                            "The only supported algorithm is 'brute'.");
        }

        // Currently only support Euclidean distance
        this->opts.get("metric", opt_tmp, metric);

        if (metric != euclidean) {
            return da_error(this->err, da_status_invalid_option,
                            "The only supported metric is 'euclidean'.");
        }

        // Allocate memory
        try {
            labels.resize(
                n_samples,
                NOISE); // Initialize to NOISE to indicate that the point has not been assigned to a cluster
            neighbors.resize(n_samples);
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }

        // Form in neighbors the list of indices within the epsilon neighborhood of each sample point
        status = da_radius_neighbors::radius_neighbors(n_samples, n_features, A, lda, eps,
                                                       neighbors, this->err);
        if (status != da_status_success)
            return da_error(this->err, status, // LCOV_EXCL_LINE
                            "Failed to compute radius neighbors prior to clustering.");

        status = dbscan_clusters();
        if (status != da_status_success)
            return da_error(this->err, status, // LCOV_EXCL_LINE
                            "Failed to compute DBSCAN clustering.");

        n_core_samples = core_sample_indices.size();
        iscomputed = true;

        return status;
    }
};

/* Compute the DBSCAN clusters */
template <typename T> da_status da_dbscan<T>::dbscan_clusters() {
    da_status status = da_status_success;

    if (algorithm == brute_serial) {
        std::fill(labels.begin(), labels.end(), UNVISITED);

        // Serial loop for computing DBSCAN clusters
        for (da_int i = 0; i < n_samples; i++) {
            if (status == da_status_memory_error)
                continue;
            try {
                // If we've already looked at this point we can go to the next loop iteration
                if (labels[i] != UNVISITED)
                    continue;

                // Find the neighbors of the current sample
                if ((da_int)neighbors[i].size() < min_samples) {
                    //Epsilon neighborhood is too small to form a cluster; label as noise
                    labels[i] = NOISE;
                } else {
                    // Form a new cluster and label this as a core sample
                    labels[i] = n_clusters;
                    core_sample_indices.push_back(i);

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

                        if ((da_int)neighbors[neigh].size() >= min_samples) {
                            // This point is also a core sample point so mark it as such and add its neighbors to the search vector
                            search_indices.append(neighbors[neigh]);
                            core_sample_indices.push_back(neigh);
                        }
                    }

                    n_clusters++;
                }
            } catch (std::bad_alloc const &) {
                status = da_status_memory_error;
                continue;
            }
        }

    } else {

        std::unordered_map<da_int, da_int> label_map;

#pragma omp declare reduction(                                                           \
        merge_unordered_maps_red : std::unordered_map<                                   \
                da_int, da_int> : merge_unordered_maps(omp_out, omp_in))                 \
    initializer(omp_priv = omp_orig)

#pragma omp parallel default(none)                                                       \
    shared(labels, neighbors, n_clusters, n_core_samples, core_sample_indices,           \
               min_samples, n_samples, status, label_map)
        {
            bool local_failure = false;

            da_vector::da_vector<da_int> local_core_sample_indices;

            try {
// Parallel loop to compute DBSCAN clusters
#pragma omp for schedule(dynamic, 32) reduction(merge_unordered_maps_red : label_map)    \
    nowait
                for (da_int i = 0; i < n_samples; i++) {
                    if ((da_int)neighbors[i].size() >= min_samples) {
                        // This is a core point
                        da_int tmp_label_i;
#pragma omp atomic read
                        tmp_label_i = labels[i];
                        if (i < tmp_label_i || tmp_label_i == NOISE) {
#pragma omp atomic write
                            // Assign it it's own index as the cluster label - we will combine clusters later
                            labels[i] = i;
                        }
                        // Record that it's a core sample point
                        local_core_sample_indices.push_back(i);
                        // Loop through each point in the epsilon neighborhood of point i
                        for (da_int j = 0; j < (da_int)neighbors[i].size(); j++) {
                            da_int sample_point_j = neighbors[i][j];
                            da_int tmp_label_j;
#pragma omp atomic read
                            tmp_label_j = labels[sample_point_j];
                            // Record that i and j are in the same cluster
                            std::pair<da_int, da_int> key_value_tmp =
                                std::minmax(i, sample_point_j);
                            // Add this pair to the label map to deal with duplicate cluster labels
                            add_to_label_map(label_map, key_value_tmp.second,
                                             key_value_tmp.first);

                            if (i < tmp_label_j || tmp_label_j == NOISE) {
#pragma omp atomic write
                                labels[sample_point_j] = i;
                            }
                        }
                    }
                }

#pragma omp critical
                { core_sample_indices.append(local_core_sample_indices); }
            } catch (std::bad_alloc const &) {
                local_failure = true; // LCOV_EXCL_LINE
            }
#pragma omp barrier
            if (!local_failure) {
                try {
#pragma omp for schedule(dynamic, 32)
                    for (da_int i = 0; i < n_samples; i++) {
                        da_int current_label = labels[i];
                        if (current_label == NOISE) {
                            continue;
                        }

                        auto it = label_map.find(current_label);
                        while (it != label_map.end()) {
                            current_label = it->second;
                            it = label_map.find(current_label);
                        }
                        labels[i] = current_label;
                    }
                } catch (std::bad_alloc const &) {
                    local_failure = true;
                }
            }
            if (local_failure)
                status = da_status_memory_error;
        }

        if (status == da_status_memory_error)
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
    }

    // Record how many distinct clusters and how many core samples we have and form a new map for relabeling
    n_core_samples = core_sample_indices.size();
    std::sort(&core_sample_indices[0], &core_sample_indices[0] + n_core_samples);
    std::set<da_int> unique_labels;

    try {
        for (auto &label : labels) {
            if (label != NOISE)
                unique_labels.insert(label);
        }
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    n_clusters = unique_labels.size();

    std::unordered_map<da_int, da_int> relabel_map;

    // This map is because labels may not be named 0, 1, 2, etc
    da_int label_count = 0;
    for (const auto &label : unique_labels) {
        relabel_map[label] = label_count++;
    }

#pragma omp parallel for default(none) schedule(dynamic, 32)                             \
    shared(relabel_map, labels, n_samples)
    for (da_int i = 0; i < n_samples; i++) {
        auto it = relabel_map.find(labels[i]);
        if (it != relabel_map.end()) {
            labels[i] = it->second;
        }
    }

    if (status == da_status_memory_error) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    return da_status_success;
}

} // namespace da_dbscan

#endif // DBSCAN_HPP
