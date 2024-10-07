/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef KNN_HPP
#define KNN_HPP

#include "aoclda.h"
#include "basic_handle.hpp"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "euclidean_distance.hpp"
#include "knn_options.hpp"
#include <numeric>

namespace da_knn {

/* k-nearest neighbors class */
template <typename T> class da_knn : public basic_handle<T> {
  private:
    // Set true when initialization is complete by set_params() function
    bool is_up_to_date = false;
    // Set true if training data has been provided via set_training_data()
    bool istrained = false;
    // Set true if the available classes have been computed via a call to available_classes()
    bool classes_computed = false;

    // Pointer to error trace
    da_errors::da_error_t *err = nullptr;

    // Number of neighbors to be considered
    da_int n_neighbors = 5;
    // Algorithm to be used for the knn computation
    da_int algo = da_brute_force;
    // Metric to be used for the distance computation
    da_int metric = da_euclidean;
    // Internal metric to be used for the distance computation.
    // We want to avoid squaring the distance unless it's necessary.
    da_int internal_metric = da_sqeuclidean;
    // Weight function used to compute the k-nearest neighbors
    da_int weights = da_knn_uniform;
    // User's data
    da_int n_samples = 0, n_features = 0, ldx_train = 0;
    const T *X_train = nullptr /*n_samples-by-n_features*/;
    const da_int *y_train = nullptr /*n_samples*/;

  public:
    std::vector<da_int> classes;
    da_int n_classes = -1;
    da_options::OptionRegistry opts;

    da_knn(da_errors::da_error_t &err, da_status &status) {
        // Assumes that err is valid
        this->err = &err;
        // Initialize the options registry
        status = register_knn_options<T>(opts);
    };

    /* get_result (required to be defined by basic_handle) */
    da_status get_result(da_result query, da_int *dim, T *result);
    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] da_int *result);
    // Set input parameters
    da_status set_params();
    // Set the training data
    da_status set_training_data(da_int n_samples, da_int n_features, const T *X_train,
                                da_int ldx_train, const da_int *y_train);
    // Compute the k-nearest neighbors and optionally the corresponding distances
    da_status kneighbors(da_int n_queries, da_int n_features, const T *X_test,
                         da_int ldx_test, da_int *n_ind, T *n_dist, da_int k = 0,
                         bool return_distance = 0);
    // Compute probability estimates for provided test data
    da_status predict_proba(da_int n_queries, da_int n_features, const T *X_test,
                            da_int ldx_test, T *proba);
    // Predict the labels for provided test data
    da_status predict(da_int n_queries, da_int n_features, const T *X_test,
                      da_int ldx_test, da_int *y_test);
    // Internal function used to compute the std::vector that holds the available classes
    da_status available_classes();

    // Implementing refresh
    void refresh() { is_up_to_date = false; }
};

template <typename T>
da_status da_knn<T>::get_result(da_result query, da_int *dim, T *result) {
    return da_warn_bypass(err, da_status_unknown_query,
                          "There are no floating-point results available for this API.");
}

template <typename T>
da_status da_knn<T>::get_result([[maybe_unused]] da_result query,
                                [[maybe_unused]] da_int *dim,
                                [[maybe_unused]] da_int *result) {
    da_status status = da_status_success;
    if (!istrained)
        return da_warn_bypass(this->err, da_status_unknown_query,
                              "Handle does not contain data relevant to this query. "
                              "Model needs to be trained.");
    if (!is_up_to_date)
        status = da_knn<T>::set_params();
    // Pointers were already tested in the generic get_result
    da_int knn_info_size = 6;
    switch (query) {
    case da_result::da_knn_model_params:
        if (*dim < knn_info_size) {
            *dim = knn_info_size;
            return da_warn_bypass(err, da_status_invalid_array_dimension,
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
        return da_warn_bypass(err, da_status_unknown_query,
                              "The requested result could not be found.");
    }
    return status;
}

template <typename T> da_status da_knn<T>::set_params() {
    // Extract options
    std::string opt_val;
    bool opt_pass = true;
    opt_pass &= opts.get("number of neighbors", n_neighbors) == da_status_success;
    opt_pass &= opts.get("algorithm", opt_val, algo) == da_status_success;
    opt_pass &= opts.get("metric", opt_val, metric) == da_status_success;
    opt_pass &= opts.get("weights", opt_val, weights) == da_status_success;
    if (!opt_pass)
        return da_error_bypass(err, da_status_internal_error, // LCOV_EXCL_LINE
                               "Unexpected error while reading the optional parameters.");
    if (metric == da_euclidean || metric == da_sqeuclidean)
        internal_metric = da_sqeuclidean;
    this->is_up_to_date = true;
    return da_status_success;
}

template <typename T>
da_status da_knn<T>::set_training_data(da_int n_samples, da_int n_features,
                                       const T *X_train, da_int ldx_train,
                                       const da_int *y_train) {
    if (X_train == nullptr || y_train == nullptr)
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "Either X_train, or y_train are not valid pointers.");
    if (n_samples <= 0 || n_features <= 0) {
        return da_error_bypass(
            this->err, da_status_invalid_array_dimension,
            "n_samples = " + std::to_string(n_samples) +
                ", n_features = " + std::to_string(n_features) +
                ", the values of n_samples and n_features need to be greater than 0");
    }
    if (ldx_train < n_samples) {
        return da_error_bypass(
            this->err, da_status_invalid_leading_dimension,
            "n_samples = " + std::to_string(n_samples) +
                ", ldx_train = " + std::to_string(ldx_train) +
                ", the value of ldx_train needs to be at least as big as the value "
                "of n_samples");
    }

    // Set internal pointers to user data
    this->X_train = X_train;
    this->y_train = y_train;
    this->n_samples = n_samples;
    this->n_features = n_features;
    this->ldx_train = ldx_train;
    this->istrained = true;
    return da_status_success;
}

/**
 * Returns the indices of the k-nearest neighbors for each point in a test data set and, optionally, the
 * corresponding distances to each neighbor.
 *
 * - If X_test is a nullptr, then throw an error
 * and compute the k-nearest neighbors of the training data matrix provided via set_training_data(),
 * not considering itself as a neighbor.
 * - If X_test is not nullptr, then X_test is the test data matrix of size m-by-n, and for each of its points
 * kneighbors() computes its neighbors in the training data matrix.
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
da_status da_knn<T>::kneighbors(da_int n_queries, da_int n_features, const T *X_test,
                                da_int ldx_test, da_int *n_ind, T *n_dist, da_int n_neigh,
                                bool return_distance) {
    da_status status = da_status_success;
    // Return if there are no training data
    if (!istrained)
        return da_error_bypass(this->err, da_status_no_data,
                               "No data has been passed to the handle. Please call "
                               "da_knn_set_data_s or da_knn_set_data_d.");
    if (!is_up_to_date)
        status = da_knn<T>::set_params();

    if (X_test == nullptr)
        return da_error_bypass(this->err, da_status_invalid_input,
                               "X_test is not a valid pointer.");

    // Add error checking for input parameters
    if (n_ind == nullptr) {
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "n_ind is not a valid pointer.");
    }
    // This check is added based on the functionality that will be added in the future.
    // X can be nullptr. Only check parameters related to X, if X is not nullptr.
    if (X_test != nullptr) {
        if (n_queries < 1 || n_features < 1) {
            return da_error_bypass(this->err, da_status_invalid_array_dimension,
                                   "n_queries and n_features must be greater than 0.");
        }
        if (ldx_test < n_queries) {
            return da_error_bypass(
                this->err, da_status_invalid_leading_dimension,
                "n_queries = " + std::to_string(n_queries) +
                    ", ldx_test = " + std::to_string(ldx_test) +
                    ", the value of ldx_test needs to be at least as big as the value "
                    "of n_queries");
        }
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

    // Create memory to store the distance matrix
    try {
        std::vector<T> D(this->n_samples * n_queries);

        // Call function that computes the squared euclidean distance
        status = da_metrics::pairwise_distances::euclidean(
            this->n_samples, n_queries, n_features, this->X_train, this->ldx_train,
            X_test, ldx_test, D.data(), this->n_samples, true);
        if (status != da_status_success)
            return da_error_bypass(
                this->err, status,
                "Failed to compute k-nearest neighbors due to an internal "
                "error in the distance computation.");

        std::vector<da_int> indices(this->n_samples * n_queries);

        // Initialize each column of "indices" with values from 0 to n_samples-1.
        // Those are the initial indices before sorting.
        for (da_int i = 0; i < n_queries; i++) {
            std::iota(indices.begin() + i * this->n_samples,
                      indices.begin() + (i + 1) * this->n_samples, 0);
        }

        // The first n_queries elements hold the indices for the sorted distances of the first test data and so on.
        for (da_int k = 0; k < n_queries; k++)
            std::stable_sort(
                indices.begin() + k * this->n_samples,
                indices.begin() + (k + 1) * this->n_samples, [&](da_int i, da_int j) {
                    return D[i + k * this->n_samples] < D[j + k * this->n_samples];
                });

        // n_ind holds the info for the nearest neighbors, by using only the first
        // n_neigh elements for each test data point.
        for (da_int i = 0; i < n_queries; i++) {
            for (da_int j = 0; j < n_neigh; j++) {
                // Copy the data
                n_ind[i + j * n_queries] = indices[j + i * this->n_samples];
            }
        }

        // Storing the distance of the neighbors
        if (return_distance) {
            // If metric is da_euclidean, we need to compute the square root of the elements before returning.
            if ((this->internal_metric == da_sqeuclidean) &&
                (this->metric != da_sqeuclidean)) {
                for (da_int i = 0; i < n_queries; i++) {
                    for (da_int j = 0; j < n_neigh; j++) {
                        n_dist[i + j * n_queries] =
                            std::sqrt(D[n_ind[i + j * n_queries] + i * this->n_samples]);
                    }
                }
            } else {
                for (da_int i = 0; i < n_queries; i++) {
                    for (da_int j = 0; j < n_neigh; j++) {
                        n_dist[i + j * n_queries] =
                            D[n_ind[i + j * n_queries] + i * this->n_samples];
                    }
                }
            }
        }
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    return da_status_success;
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

template <typename T> da_status da_knn<T>::available_classes() {
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
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
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
da_status da_knn<T>::predict_proba(da_int n_queries, da_int n_features, const T *X_test,
                                   da_int ldx_test, T *proba) {
    da_status status = da_status_success;
    // Return if there are no training data
    if (!istrained)
        return da_error_bypass(this->err, da_status_no_data,
                               "No data has been passed to the handle. Please call "
                               "da_knn_set_data_s or da_knn_set_data_d.");

    if (!is_up_to_date)
        status = da_knn<T>::set_params();

    if (!this->classes_computed) {
        // From the input data y_train, find the available classes.
        status = da_knn<T>::available_classes();
    }

    if (status != da_status_success)
        return da_error_bypass(this->err, status,
                               "Failed to compute probabilities due to an internal error "
                               "of the available classes computation.");

    if (proba == nullptr)
        return da_error_bypass(this->err, da_status_invalid_pointer,
                               "proba is not a valid pointer.");

    // This check is added based on the functionality that will be added in the future.
    // X can be nullptr. Only check parameters related to X, if X is not nullptr.
    if (X_test != nullptr) {
        if (n_queries < 1 || n_features < 1) {
            return da_error_bypass(this->err, da_status_invalid_array_dimension,
                                   "n_queries and n_features must be greater than 0.");
        }
        if (ldx_test < n_queries) {
            return da_error_bypass(
                this->err, da_status_invalid_leading_dimension,
                "n_queries = " + std::to_string(n_queries) +
                    ", ldx_test = " + std::to_string(ldx_test) +
                    ", the value of ldx_test needs to be at least as big as the value "
                    "of n_queries");
        }
        // Data matrix X must have the same number of columns as X_train.
        if (n_features != this->n_features) {
            return da_error_bypass(this->err, da_status_invalid_array_dimension,
                                   "n_features = " + std::to_string(n_features) +
                                       " doesn't match the expected value " +
                                       std::to_string(this->n_features) + ".");
        }
    }
    // Allocate memory to set neighbors' indices and corresponding distances.
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
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    return status;
}

/*
 * Predict the labels y_test for the provided test data.
 * Compute the probabilities for the labels and return the corresponding label according to that.
 */
template <typename T>
da_status da_knn<T>::predict(da_int n_queries, da_int n_features, const T *X_test,
                             da_int ldx_test, da_int *y_test) {

    da_status status = da_status_success;
    if (!is_up_to_date)
        status = da_knn<T>::set_params();

    // Return if there are no training data
    if (!istrained)
        return da_error_bypass(this->err, da_status_no_data,
                               "No data has been passed to the handle. Please call "
                               "da_knn_set_data_s or da_knn_set_data_d.");

    if (!this->classes_computed) {
        // From the input data y_train, find the available classes.
        status = da_knn<T>::available_classes();
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

        status = da_knn<T>::predict_proba(n_queries, n_features, X_test, ldx_test,
                                          proba.data());
        if (status != da_status_success)
            return da_error_bypass(
                this->err, status,
                "Failed to compute predicted labels due to an internal "
                "error of predicting the probabilities.");

        // For each row in n_queries, check which label appears the most times.
        // In case of a tie, return the first label.
        da_int max_index;
        for (da_int i = 0; i < n_queries; i++) {
            max_index =
                da_blas::cblas_iamax(this->n_classes, proba.data() + i, n_queries);
            y_test[i] = this->classes[max_index];
        }
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    return status;
}

} // namespace da_knn

#endif
