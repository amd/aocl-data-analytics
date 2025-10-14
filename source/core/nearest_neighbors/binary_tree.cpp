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

#include "binary_tree.hpp"
#include "aoclda.h"
#include "da_error.hpp"
#include "da_omp.hpp"
#include "da_std.hpp"
#include "da_vector.hpp"
#include "pairwise_distances.hpp"
#include <algorithm>
#include <memory>
#include <vector>

#define BT_MAX_BLOCK_SIZE da_int(256)

namespace ARCH {

namespace da_binary_tree {

//Constructor for a node of the binary tree
template <typename T>
node<T>::node(da_int depth, da_int *indices, da_int n_indices)
    : depth(depth), indices(indices), n_indices(n_indices){};

//Constructors for a node of the ball tree
template <typename T>
ball_node<T>::ball_node(da_int depth, da_int *indices, da_int n_indices,
                        std::vector<T> centroid, T radius)
    : node<T>(depth, indices, n_indices), centroid(centroid), radius(radius) {}

template <typename T>
ball_node<T>::ball_node(da_int depth, da_int *indices, da_int n_indices)
    : node<T>(depth, indices, n_indices), centroid(std::vector<T>()), radius((T)0.0) {}

// Constructors for a node of the k-d tree
template <typename T>
kd_node<T>::kd_node(da_int dim, da_int depth, da_int *indices, da_int n_indices,
                    std::vector<T> min_bounds, std::vector<T> max_bounds)
    : node<T>(depth, indices, n_indices), dim(dim), min_bounds(min_bounds),
      max_bounds(max_bounds) {}

template <typename T>
kd_node<T>::kd_node(da_int dim, da_int depth, da_int *indices, da_int n_indices)
    : node<T>(depth, indices, n_indices), dim(dim), min_bounds(std::vector<T>()),
      max_bounds(std::vector<T>()) {}

// Lightweight partial MaxHeap implementation to keep track of k-NN k-d tree searches
template <typename T>
MaxHeap<T>::MaxHeap(da_int capacity, da_int *indices, T *distances)
    : indices(indices), distances(distances), capacity(capacity) {
    size = 0;
}

template <typename T> T MaxHeap<T>::GetMaxDist() {
    // Return the maximum distance in the heap, or the maximum possible value if the heap is not full
    return (size < capacity) ? std::numeric_limits<T>::max() : distances[0];
}

// Insert a new point to the heap if the distance is smaller than the max, maintaining the max-heap property
template <typename T> void MaxHeap<T>::Insert(da_int index, T distance) {
    if (size < capacity) {
        indices[size] = index;
        distances[size] = distance;
        heapify_up(size);
        size++;
    } else if (distance < distances[0]) {
        indices[0] = index;
        distances[0] = distance;
        heapify_down(0);
    }
}

template <typename T> da_int MaxHeap<T>::GetSize() {
    // Return the current size of the heap
    return size;
}

template <typename T> void MaxHeap<T>::heapify_up(da_int index) {
    // Move the element at index up the heap until the max-heap property is restored
    while (index > 0) {
        da_int parent = (index - 1) / 2;
        if (distances[index] > distances[parent]) {
            // If the current element is greater than its parent, swap them
            std::swap(indices[index], indices[parent]);
            std::swap(distances[index], distances[parent]);
            index = parent;
        } else {
            break; // The max-heap property is restored
        }
    }
}

template <typename T> void MaxHeap<T>::heapify_down(da_int index) {
    // Move the element at index down the heap until the max-heap property is restored
    while (true) {
        da_int left = 2 * index + 1;
        da_int right = 2 * index + 2;
        da_int largest = index;

        if (left < size && distances[left] > distances[largest]) {
            largest = left;
        }
        if (right < size && distances[right] > distances[largest]) {
            largest = right;
        }
        if (largest != index) {
            std::swap(distances[index], distances[largest]);
            std::swap(indices[index], indices[largest]);
            index = largest;
        } else {
            break;
        }
    }
}

// Compute the distance between the point at index_A in A and the point X. For Euclidean distance
// the squared distance is returned, otherwise the distance is returned.
template <typename Derived, typename NodeType>
da_status binary_tree<Derived, NodeType>::compute_distance(T &dist, da_int index_A, T *X,
                                                           T X_norm) {
    if (this->metric == da_euclidean_gemm) {
        // Special case for Euclidean distance using precomputed norms
        dist = 0.0;
        // Typically expect this to be a small number of features so use a simple loop rather than BLAS call
        for (da_int i = 0; i < this->n_features; i++) {
            dist += X[i] * this->A[index_A + i * this->lda];
        }
        dist = X_norm + this->A_norms[index_A] - 2 * dist;
    } else {
        // Compute the distance matrix using the specified metric
        da_status status = ARCH::da_metrics::pairwise_distances::pairwise_distance_kernel(
            da_order::column_major, 1, 1, this->n_features, X, 1, &this->A[index_A],
            this->lda, &dist, 1, this->p, this->metric_internal);
        if (status != da_status_success) {
            return status; // LCOV_EXCL_LINE
        }
    }
    return da_status_success;
}

template <typename Derived, typename NodeType>
void binary_tree<Derived, NodeType>::store_data(da_int n_samples_in, da_int n_features_in,
                                                const T *A_in, da_int lda_in,
                                                da_int leaf_size_in, da_metric metric_in,
                                                T p_in) {
    this->metric =
        (metric_in == da_minkowski && p_in == T(2.0)) ? da_euclidean : metric_in;
    this->metric_internal = this->metric;
    if (this->metric == da_euclidean) {
        this->metric_internal = da_sqeuclidean;
    }
    // We use p and p_inv when checking bounding boxes
    this->p = p_in;
    this->p_inv = (T)1.0 / this->p;
    if (this->metric == da_manhattan) {
        this->p = (T)1.0;
        this->p_inv = (T)1.0;
    }

    this->n_samples = n_samples_in;
    this->n_features = n_features_in;
    this->lda = lda_in;
    this->A = A_in;
    this->leaf_size = leaf_size_in;

    if (this->metric == da_euclidean_gemm) {

        this->A_norms.resize(this->n_samples);
        // Guard against multiple calls
        da_std::fill(this->A_norms.begin(), this->A_norms.end(), (T)0.0);

        // Precompute the row norms of A to speed up Euclidean distance computation
        da_int n_blocks = 0, block_rem = 0;
        da_utils::blocking_scheme(n_samples, BT_MAX_BLOCK_SIZE, n_blocks, block_rem);
        [[maybe_unused]] da_int n_threads = da_utils::get_n_threads_loop(n_blocks);
        da_int block_index;
        da_int block_size = BT_MAX_BLOCK_SIZE;

// Careful use of default shared needed because we can't use this-> in OpenMP directives
#pragma omp parallel default(shared) firstprivate(block_size) private(block_index)       \
    num_threads(n_threads)
        {
#pragma omp for schedule(static)
            for (da_int k = 0; k < n_blocks; k++) {
                if (k == n_blocks - 1 && block_rem > 0) {
                    block_index = n_samples - block_rem;
                    block_size = block_rem;
                } else {
                    block_index = k * BT_MAX_BLOCK_SIZE;
                }
                for (da_int j = 0; j < this->n_features; j++) {
                    for (da_int i = 0; i < block_size; i++) {
                        da_int A_index = i + block_index;
                        this->A_norms[A_index] += this->A[A_index + j * this->lda] *
                                                  this->A[A_index + j * this->lda];
                    }
                }
            }
        }
    }
}
// Utility function called prior to forming radius/k neighbors. Checks whether X_in is null (in which
// case we should be using the original data matrix, this->A) and updates m_samples, m_features, ldx
// and X accordingly.
template <typename Derived, typename NodeType>
da_status binary_tree<Derived, NodeType>::preprocess_data(
    const T *X_in, da_int m_samples_in, da_int m_features_in, da_int ldx_in, const T **X,
    bool &X_is_A, da_int &m_samples, da_int &ldx, da_errors::da_error_t *err) {
    if (m_features_in != this->n_features) {
        return da_error(                  // LCOV_EXCL_LINE
            err, da_status_invalid_input, // LCOV_EXCL_LINE
            "Number of features in X does not match the number of features in the "
            "tree.");
    }

    // If X_in is null then we use the original dataset which was used to construct the tree
    if (X_in == nullptr) {
        m_samples = this->n_samples;
        ldx = this->lda;
        *X = this->A;
        X_is_A = true;
    } else {
        m_samples = m_samples_in;
        ldx = ldx_in;
        *X = X_in;
        X_is_A = false;
    }

    return da_status_success;
}

// Find the k nearest neighbors of a point using the k-d tree
template <typename Derived, typename NodeType>
da_status binary_tree<Derived, NodeType>::k_neighbors(da_int m_samples_in,
                                                      da_int m_features_in, const T *X_in,
                                                      da_int ldx_in, da_int k,
                                                      da_int *k_ind, T *k_dist,
                                                      da_errors::da_error_t *err) {
    // We assume here that the k-d tree is already built and that m_samples, m_features, ldx and k
    // are valid and da_metric is valid (i.e. not cosine distance)
    da_status status = da_status_success;

    std::vector<T> X_row;
    const T *X = nullptr;
    bool X_is_A = false;
    da_int ldx = 0;
    da_int m_samples = 0;

    // Call utility function to preprocess the data: check if X_in is null (in which case we should
    // be using the original data matrix, this->A) and update m_samples, m_features, ldx, X, accordingly
    status = this->preprocess_data(X_in, m_samples_in, m_features_in, ldx_in, &X, X_is_A,
                                   m_samples, ldx, err);
    if (status != da_status_success) {
        return status; // LCOV_EXCL_LINE
    }

    try {
        X_row.resize(this->n_features * omp_get_max_threads());
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    // Loop over the samples in X and find the neighbors - careful use of default shared needed because we can't use this-> in OpenMP directives
#pragma omp parallel default(shared)
    {
#pragma omp for schedule(dynamic, 128)
        for (da_int i = 0; i < m_samples; i++) {
            da_int X_row_index = this->n_features * omp_get_thread_num();

            T X_norm = 0.0;
            if (this->metric == da_euclidean_gemm) {
                X_norm = this->A_norms[i];
                if (!(X_is_A)) {
                    X_norm = 0.0;
                    for (da_int j = 0; j < this->n_features; j++) {
                        X_norm += X[i + j * ldx] * X[i + j * ldx];
                    }
                }
            }

            // For better data access later, copy this row of X into a contiguous piece of a vector
            for (da_int j = 0; j < this->n_features; j++) {
                X_row[j + X_row_index] = X[i + j * ldx];
            }

            auto heap = MaxHeap<T>(k, &k_ind[i * k], &k_dist[i * k]);

            da_status tmp_status = static_cast<Derived *>(this)->k_neighbors_recursive(
                this->root, &X_row[X_row_index], k, X_is_A, i, X_norm, heap);
            if (tmp_status != da_status_success) {
// If there was an error, set the status and break out of the loop
#pragma omp atomic write
                status = tmp_status;
            }
        }
    }
    if (status != da_status_success) {
        return da_error(err, status, // LCOV_EXCL_LINE
                        "Failed to compute radius neighbors.");
    }

    return da_status_success;
}

template <typename Derived, typename NodeType>
da_status binary_tree<Derived, NodeType>::radius_neighbors(
    da_int m_samples_in, da_int m_features_in, const T *X_in, da_int ldx_in, T eps,
    std::vector<da_vector::da_vector<da_int>> &neighbors, da_errors::da_error_t *err) {
    // We assume here that the tree is already built and that m_samples, m_features and ldx are
    // valid and da_metric is valid (not cosine distance)
    da_status status = da_status_success;

    std::vector<T> X_row;
    const T *X = nullptr;
    bool X_is_A = false;
    da_int ldx = 0;
    da_int m_samples = 0;

    // For da_euclidean it is more efficient to use the squared distance for some of the computation
    T eps_internal =
        ((this->metric == da_euclidean) || (this->metric == da_euclidean_gemm))
            ? eps * eps
            : eps;

    // Call utility function to preprocess the data: check is X_in is null (in which case we should
    // be using the original data matrix, this->A) and update m_samples, m_features, ldx, X accordingly
    status = this->preprocess_data(X_in, m_samples_in, m_features_in, ldx_in, &X, X_is_A,
                                   m_samples, ldx, err);
    if (status != da_status_success) {
        return status; // LCOV_EXCL_LINE
    }

    try {
        X_row.resize(this->n_features * omp_get_max_threads());
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

// Loop over the samples in X and find the radius neighbors - careful use of default shared needed because we can't use this-> in OpenMP directives
#pragma omp parallel default(shared)
    {
#pragma omp for schedule(dynamic, 128)
        for (da_int i = 0; i < m_samples; i++) {
            da_int X_row_index = this->n_features * omp_get_thread_num();

            T X_norm = 0.0;
            if (this->metric == da_euclidean_gemm) {
                X_norm = this->A_norms[i];
                if (!(X_is_A)) {
                    X_norm = 0.0;
                    for (da_int j = 0; j < this->n_features; j++) {
                        X_norm += X[i + j * ldx] * X[i + j * ldx];
                    }
                }
            }

            // For better data access later, copy this row of X into a contiguous piece of a vector
            for (da_int j = 0; j < this->n_features; j++) {
                X_row[j + X_row_index] = X[i + j * ldx];
            }

            // Find the epsilon radius neighbors of the ith point in X by recursively searching the tree
            da_status tmp_status =
                static_cast<Derived *>(this)->radius_neighbors_recursive(
                    this->root, &X_row[X_row_index], eps, eps_internal, neighbors[i],
                    X_is_A, i, X_norm);
            if (tmp_status != da_status_success) {
// If there was an error, set the status and break out of the loop
#pragma omp atomic write
                status = tmp_status;
            }
        }
    }

    if (status != da_status_success) {
        return da_error(err, status, // LCOV_EXCL_LINE
                        "Failed to compute radius neighbors.");
    }
    return da_status_success;
}

template <typename Derived, typename NodeType>
const std::vector<da_int> &binary_tree<Derived, NodeType>::get_indices() {
    return this->indices;
}

template struct node<double>;
template struct node<float>;

template class MaxHeap<double>;
template class MaxHeap<float>;

template struct kd_node<double>;
template struct kd_node<float>;

template struct ball_node<double>;
template struct ball_node<float>;

template class binary_tree<kd_tree<double>, kd_node<double>>;
template class binary_tree<kd_tree<float>, kd_node<float>>;

template class binary_tree<ball_tree<double>, ball_node<double>>;
template class binary_tree<ball_tree<float>, ball_node<float>>;

} // namespace da_binary_tree
} // namespace ARCH