/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
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

#ifndef KMEANS_HPP
#define KMEANS_HPP

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "aoclda.h"
#include "aoclda_kmeans.h"
#include "basic_handle.hpp"
#include "callbacks.hpp"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "kmeans_options.hpp"
#include "kmeans_types.hpp"
#include "lapack_templates.hpp"
#include "options.hpp"
#include <iostream>
#include <string.h>

namespace da_kmeans {

/* k-means class */
template <typename T> class da_kmeans : public basic_handle<T> {
  private:
    // n x p (samples x features)
    da_int n_samples = 0;
    da_int n_features = 0;

    // Set true when initialization is complete
    bool initdone = false;

    // Set true when k-means is computed successfully
    bool iscomputed = false;

    // Underlying algorithm
    da_int algorithm = lloyd;

    // Initialization method
    da_int init_method = random_centres;

    // Number of clusters requested
    da_int n_clusters = 1;

    // Number of runs to perform
    da_int n_init = 1;

    // Max iterations
    da_int max_iter = 1;

    // Actual number of iterations performed
    da_int n_iter = 0;

    // Convergence tolerance
    T tol = 1.0;

    // Random seed
    da_int seed = 0;

    // Pointer to error trace
    da_errors::da_error_t *err = nullptr;

    // Arrays used by the SVD, and to store results
    std::vector<da_int> labels;     // labels
    std::vector<T> cluster_centres; // cluster centres
    T inertia = 0.0;                // Inertia
    std::vector<T> A_copy, C_copy;

  public:
    da_options::OptionRegistry opts;

    da_kmeans(da_errors::da_error_t &err) {
        this->err = &err;
        register_kmeans_options<T>(opts);
    };

    da_status set_data(da_int n_samples, da_int n_features, const T *A, da_int lda);

    da_status set_init_centres(const T *C, da_int ldc);

    da_status compute();

    da_status transform(da_int m_samples, da_int m_features, const T *X, da_int ldx,
                        T *X_transform, da_int ldx_transform);

    da_status predict(da_int k_samples, da_int k_features, const T *Y, da_int ldy,
                      da_int *Y_predict);

    da_status get_result(da_result query, da_int *dim, T *result) {
        // Don't return anything if k-means has not been computed
        if (!iscomputed) {
            return da_warn(err, da_status_no_data,
                           "k-means clustering has not yet been computed. Please call "
                           "da_kmeans_compute_s "
                           "or da_kmeans_compute_d before extracting results.");
        }

        da_int rinfo_size = 5;

        if (result == nullptr || *dim <= 0) {
            return da_warn(err, da_status_invalid_array_dimension,
                           "The results array has not been allocated, or an unsuitable "
                           "dimension has been provided.");
        }

        switch (query) {
        case da_result::da_rinfo:
            if (*dim < rinfo_size) {
                *dim = rinfo_size;
                return da_warn(err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(rinfo_size) + ".");
            }
            result[0] = (T)n_samples;
            result[1] = (T)n_features;
            result[2] = (T)n_clusters;
            result[3] = (T)n_iter;
            result[4] = inertia;
            break;
        case da_result::da_kmeans_cluster_centres:
            if (*dim < n_clusters * n_features) {
                *dim = n_clusters * n_features;
                return da_warn(err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(n_clusters * n_features) + ".");
            }
            for (da_int i = 0; i < n_clusters * n_features; i++)
                result[i] = cluster_centres[i];
            break;
        default:
            return da_warn(err, da_status_unknown_query,
                           "The requested result could not be found.");
        }
        return da_status_success;
    };

    da_status get_result(da_result query, da_int *dim, da_int *result) {
        // Don't return anything if k-means has not been computed
        if (!iscomputed) {
            return da_warn(err, da_status_no_data,
                           "k-means clustering has not yet been computed. Please call "
                           "da_kmeans_compute_s "
                           "or da_kmeans_compute_d before extracting results.");
        }

        if (result == nullptr || *dim <= 0) {
            return da_warn(err, da_status_invalid_array_dimension,
                           "The results array has not been allocated, or an unsuitable "
                           "dimension has been provided.");
        }
        switch (query) {
        case da_result::da_kmeans_labels:
            if (*dim < n_samples) {
                *dim = n_samples;
                return da_warn(err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(n_samples) + ".");
            }
            for (da_int i = 0; i < n_samples; i++)
                result[i] = labels[i];
            break;
        default:
            return da_warn(err, da_status_unknown_query,
                           "The requested result could not be found.");
        }

        return da_status_success;
    };
};

/* Store the user's data matrix in preparation for PCA computation */
template <typename T>
da_status da_kmeans<T>::set_data(da_int n_samples, da_int n_features, const T *A,
                                 da_int lda) {
    // Check for illegal arguments and function calls
    if (n_samples < 1)
        return da_error(err, da_status_invalid_input,
                        "The function was called with n_samples = " +
                            std::to_string(n_samples) + ". Constraint: n_samples >= 1.");
    if (n_features < 1)
        return da_error(
            err, da_status_invalid_input,
            "The function was called with n_features = " + std::to_string(n_features) +
                ". Constraint: n_features >= 1.");
    if (lda < n_samples)
        return da_error(
            err, da_status_invalid_input,
            "The function was called with n_samples = " + std::to_string(n_samples) +
                " and lda = " + std::to_string(lda) + ". Constraint: lda >= n_samples.");
    if (A == nullptr)
        return da_error(err, da_status_invalid_pointer, "The array A is null.");

    // Store dimensions of A
    this->n_samples = n_samples;
    this->n_features = n_features;
    try {
        A_copy.resize(n_samples * n_features);
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error,
                        "Memory allocation failed."); // LCOV_EXCL_LINE
    }
    // Copy the input matrix into internal matrix buffer
    for (da_int j = 0; j < n_features; j++) {
        for (da_int i = 0; i < n_samples; i++) {
            A_copy[i + j * n_samples] = A[i + lda * j];
        }
    }

    // Record that initialization is complete but computation has not yet been performed
    initdone = true;
    iscomputed = false;

    // Now that we have a data matrix we can re-register the n_clusters option with new constraints
    da_int temp_clusters;
    opts.get("n_clusters", temp_clusters);

    reregister_kmeans_option<T>(opts, n_samples);

    opts.set("n_clusters", std::min(temp_clusters, n_samples));

    if (temp_clusters > n_samples)
        return da_warn(
            err, da_status_incompatible_options,
            "The requested number of principal components has been decreased from " +
                std::to_string(temp_clusters) + " to " + std::to_string(n_samples) +
                " due to the size of the data array.");

    return da_status_success;
}

template <typename T> da_status da_kmeans<T>::set_init_centres(const T *C, da_int ldc) {

    if (initdone == false)
        return da_error(err, da_status_no_data,
                        "No data has been passed to the handle. Please call "
                        "da_kmeans_set_data_s or da_kmeans_set_data_d.");

    /* Check for illegal arguments */

    if (ldc < n_clusters)
        return da_error(
            err, da_status_invalid_input,
            "The function was called ldc = " + std::to_string(ldc) +
                " and n_clustersis currently set to = " + std::to_string(n_clusters) +
                ". Constraint: ldc >= n_clusters.");

    try {
        C_copy.resize(n_clusters * n_features);
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error,
                        "Memory allocation failed."); // LCOV_EXCL_LINE
    }
    // Copy the input matrix into internal matrix buffer
    for (da_int j = 0; j < n_features; j++) {
        for (da_int i = 0; i < n_clusters; i++) {
            C_copy[i + j * n_clusters] = C[i + ldc * j];
        }
    }

    return da_status_success;
}

/* Compute the k-means clusters */
template <typename T> da_status da_kmeans<T>::compute() {

    if (initdone == false)
        return da_error(err, da_status_no_data,
                        "No data has been passed to the handle. Please call "
                        "da_kmeans_set_data_s or da_kmeans_set_data_d.");

    // Read in options and store in class together with associated variables
    this->opts.get("n_clusters", n_clusters);

    std::string opt_method;

    this->opts.get("initialization method", opt_method, init_method);

    std::string opt_alg;
    this->opts.get("algorithm", opt_alg, algorithm);

    // Initialize some workspace arrays
    try {
        cluster_centres.resize(n_clusters * n_features);
        labels.resize(n_samples);
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error,
                        "Memory allocation failed."); // LCOV_EXCL_LINE
    }

    iscomputed = true;

    return da_status_success;
}

template <typename T>
da_status da_kmeans<T>::transform(da_int m_samples, da_int m_features, const T *X,
                                  da_int ldx, T *X_transform, da_int ldx_transform) {

    if (!iscomputed) {
        return da_warn(
            err, da_status_no_data,
            "The k-means has not been computed. Please call da_kmeans_compute_s or "
            "da_kmeans_compute_d.");
    }

    /* Check for illegal arguments */
    if (m_samples < 1)
        return da_error(err, da_status_invalid_input,
                        "The function was called with m_samples = " +
                            std::to_string(m_samples) + ". Constraint: m_samples >= 1.");
    if (m_features != n_features)
        return da_error(
            err, da_status_invalid_input,
            "The function was called with m_features = " + std::to_string(m_features) +
                " but the k-means has been computed with " + std::to_string(n_features) +
                " features.");
    if (ldx < m_samples)
        return da_error(
            err, da_status_invalid_input,
            "The function was called with m_samples = " + std::to_string(m_samples) +
                " and ldx = " + std::to_string(ldx) + ". Constraint: ldx >= m_samples.");

    if (ldx_transform < m_samples)
        return da_error(
            err, da_status_invalid_input,
            "The function was called with m_samples = " + std::to_string(m_samples) +
                " and ldx_transform = " + std::to_string(ldx_transform) +
                ". Constraint: ldx_transform >= m_samples.");

    if (X == nullptr)
        return da_error(err, da_status_invalid_pointer, "The array X is null.");

    if (X_transform == nullptr)
        return da_error(err, da_status_invalid_pointer, "The array X_transform is null.");

    return da_status_success;
}

template <typename T>
da_status da_kmeans<T>::predict(da_int k_samples, da_int k_features, const T *Y,
                                da_int ldy, da_int *Y_labels) {

    if (!iscomputed) {
        return da_warn(
            err, da_status_no_data,
            "The k-means has not been computed. Please call da_kmeans_compute_s or "
            "da_kmeans_compute_d.");
    }

    /* Check for illegal arguments */
    if (k_samples < 1)
        return da_error(err, da_status_invalid_input,
                        "The function was called with k_samples = " +
                            std::to_string(k_samples) + ". Constraint: k_samples >= 1.");
    if (k_features != n_features)
        return da_error(
            err, da_status_invalid_input,
            "The function was called with k_features = " + std::to_string(k_features) +
                " but the k-means has been computed with " + std::to_string(n_features) +
                " features.");
    if (ldy < k_samples)
        return da_error(
            err, da_status_invalid_input,
            "The function was called with k_samples = " + std::to_string(k_samples) +
                " and ldy = " + std::to_string(ldy) + ". Constraint: ldy >= k_samples.");

    if (Y == nullptr)
        return da_error(err, da_status_invalid_pointer, "The array Y is null.");

    if (Y_labels == nullptr)
        return da_error(err, da_status_invalid_pointer, "The array Y_labels is null.");

    return da_status_success;
}

} // namespace da_kmeans

#endif //KMEANS_HPP
