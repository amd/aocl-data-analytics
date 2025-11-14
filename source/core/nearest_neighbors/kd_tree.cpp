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

#include "basic_statistics.hpp"
#include "binary_tree.hpp"
#include "da_omp.hpp"
#include "da_std.hpp"
#include <algorithm>
#include <limits>
#include <stack>

#define KD_TREE_MIN_TASK_SIZE da_int(2048)
#define KD_TREE_BLOCK_SIZE da_int(256)

namespace ARCH {

namespace da_binary_tree {

// Compute the variance of each column in the dataset: blocked and parallelized for tall thin datasets
template <typename T>
da_status parallel_variance(da_int n_samples, da_int n_features, const T *A, da_int lda,
                            std::vector<T> &means, std::vector<T> &variances) {
    // Check for valid input
    da_int n_blocks = std::min((da_int)omp_get_max_threads(),
                               std::max(n_samples / KD_TREE_BLOCK_SIZE, (da_int)1));
    da_int block_size = n_samples / n_blocks;

    da_int block_rem = block_size + n_samples % block_size;

    bool internal_error = false;

    da_std::fill(means.begin(), means.end(), 0.0);
    da_std::fill(variances.begin(), variances.end(), 0.0);
    std::vector<T> thd_means;
    std::vector<T> thd_variances;
    try {
        thd_means.resize(n_features * n_blocks);
        thd_variances.resize(n_features * n_blocks);
    } catch (std::bad_alloc const &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    }

    da_int thd_block_size = block_size;
    da_int block_index;

#pragma omp parallel num_threads(n_blocks)                                               \
    firstprivate(thd_block_size) private(block_index) default(none)                      \
    shared(n_blocks, block_rem, n_samples, A, lda, thd_means, thd_variances, n_features, \
               internal_error)
    {

        // Ensure each thread gets a single contiguous block of the dataset so we can combine the means/variances afterwards
#pragma omp for schedule(static, 1)
        for (da_int i = 0; i < n_blocks; i++) {
            block_index = i * thd_block_size;
            if (i == n_blocks - 1)
                thd_block_size = block_rem;

            da_status thd_status = da_basic_statistics::variance(
                column_major, da_axis_col, thd_block_size, n_features, &A[block_index],
                lda, (da_int)(-1), thd_means.data() + i * n_features,
                thd_variances.data() + i * n_features);
            if (thd_status != da_status_success) {
#pragma omp atomic write
                internal_error = true;
            }
        }
    }

    if (internal_error) {
        return da_status_internal_error; // LCOV_EXCL_LINE
    }

    // Combine the means and variances computed by each thread
    for (da_int j = 0; j < n_features; j++) {
        for (da_int i = 0; i < n_blocks - 1; i++) {
            means[j] += thd_means[i * n_features + j] * block_size;
        }
        means[j] += thd_means[(n_blocks - 1) * n_features + j] * block_rem;
        means[j] /= n_samples;

        for (da_int i = 0; i < n_blocks - 1; i++) {
            variances[j] += (thd_variances[i * n_features + j] +
                             (means[j] - thd_means[i * n_features + j]) *
                                 (means[j] - thd_means[i * n_features + j])) *
                            block_size;
        }
        variances[j] += (thd_variances[(n_blocks - 1) * n_features + j] +
                         (means[j] - thd_means[(n_blocks - 1) * n_features + j]) *
                             (means[j] - thd_means[(n_blocks - 1) * n_features + j])) *
                        block_rem;
        variances[j] /= n_samples;
    }

    return da_status_success;
}

template <typename T>
kd_tree<T>::kd_tree(da_int n_samples_in, da_int n_features_in, const T *A_in,
                    da_int lda_in, da_int leaf_size_in, da_metric metric_in, T p_in) {
    // Initialize the k-d tree
    this->store_data(n_samples_in, n_features_in, A_in, lda_in, leaf_size_in, metric_in,
                     p_in);
    // Allocate memory for the indices (which will initially be filled with 0, 1, 2, ..., n_samples - 1)
    // and the initial bounding box
    this->indices = std::vector<da_int>(this->n_samples);
    std::vector<T> min_bounds(this->n_features, std::numeric_limits<T>::max());
    std::vector<T> max_bounds(this->n_features, std::numeric_limits<T>::lowest());
    std::vector<T> means(this->n_features);
    std::vector<T> variances(this->n_features);
    // If memory allocation failed an exception will be thrown, so the constructor must be wrapped in a try...catch
    da_std::iota(this->indices.begin(), this->indices.end(), 0);

    // Compute the bounding box for the dataset; parallelism optimized for tall, skinny dataset
    da_int n_blocks, block_rem;
    da_int max_block_size = std::min(KD_TREE_BLOCK_SIZE, this->n_samples);
    da_utils::blocking_scheme(this->n_samples, max_block_size, n_blocks, block_rem);

    da_int block_index;
    da_int block_size = max_block_size;

    bool memory_alloc_failed = false;

    // Careful use of default shared needed because we can't use this-> in OpenMP directives
#pragma omp parallel firstprivate(block_size) private(block_index) default(shared)
    {
        // Each thread needs its own copy of the min and max bounds; but these are likely to be small
        std::vector<T> min_bounds_private;
        std::vector<T> max_bounds_private;
        try {
            min_bounds_private.resize(this->n_features, std::numeric_limits<T>::max());
            max_bounds_private.resize(this->n_features, std::numeric_limits<T>::lowest());
        } catch (std::bad_alloc const &) {
// If memory allocation failed, set a flag so we can throw an exception outside the parallel region
#pragma omp atomic write
            memory_alloc_failed = true;
        }

        if (!memory_alloc_failed) {
#pragma omp for schedule(static)
            for (da_int i = 0; i < n_blocks; i++) {
                if (i == n_blocks - 1 && block_rem > 0) {
                    block_index = this->n_samples - block_rem;
                    block_size = block_rem;
                } else {
                    block_index = i * max_block_size;
                }
                da_int A_offset = 0;
                for (da_int j = 0; j < this->n_features; j++) {
                    A_offset = j * this->lda;
                    for (da_int k = 0; k < block_size; k++) {
                        min_bounds_private[j] = std::min(
                            min_bounds_private[j], this->A[k + block_index + A_offset]);
                        max_bounds_private[j] = std::max(
                            max_bounds_private[j], this->A[k + block_index + A_offset]);
                    }
                }
            }
#pragma omp critical(min_bounds)
            {
                for (da_int j = 0; j < this->n_features; j++) {
                    min_bounds[j] = std::min(min_bounds[j], min_bounds_private[j]);
                }
            }
#pragma omp critical(max_bounds)
            {
                for (da_int j = 0; j < this->n_features; j++) {
                    max_bounds[j] = std::max(max_bounds[j], max_bounds_private[j]);
                }
            }
        }

    } // End of parallel region

    if (memory_alloc_failed) {
        throw std::bad_alloc(); // LCOV_EXCL_LINE
    }

    // Compute the column variances for the dataset; parallelism optimized for tall, skinny dataset
    if (parallel_variance(this->n_samples, this->n_features, this->A, this->lda, means,
                          variances) != da_status_success) {
        throw std::runtime_error( // LCOV_EXCL_LINE
            "Failed to compute column variances for the dataset");
    }

    da_int root_dim = da_blas::cblas_iamax(this->n_features, variances.data(), (da_int)1);

// Build the k-d tree - careful use of default shared needed because we can't use this-> in OpenMP directives
#pragma omp parallel default(shared)
    {
#pragma omp single
        {
            this->root = build_tree(0, this->indices.data(), this->n_samples, &min_bounds,
                                    &max_bounds, root_dim);
        }
    }
}

// Use Welford's online single pass algorithm to compute the variance of the entries in A given by indices in dimension dim
template <typename T>
T kd_tree<T>::single_pass_variance(da_int *indices, da_int n_indices, da_int dim) {
    T current_mean = 0.0;
    T current_variance = 0.0;

    for (da_int i = 0; i < n_indices; i++) {
        T value = this->A[indices[i] + dim * this->lda];
        T delta = value - current_mean;
        current_mean += delta / (i + 1);
        current_variance += delta * (value - current_mean);
    }

    return current_variance / n_indices;
}

// Recursive function to build the k-d tree
// The k-d tree is built in a top-down manner, starting from the root node and recursively splitting
template <typename T>
std::shared_ptr<kd_node<T>>
kd_tree<T>::build_tree(da_int depth, da_int *indices, da_int n_indices,
                       std::vector<T> *min_bounds, std::vector<T> *max_bounds,
                       da_int root_dim) {

    // If there are no indices, return nullptr
    if (n_indices == 0) {
        return nullptr; // LCOV_EXCL_LINE
    }

    da_int dim = 0;

    // Create a new node for this part of the tree, with sensible defaults. Only the root node will have
    // min_bounds and max_bounds supplied; for all other nodes these will be computed later
    auto this_node = (min_bounds == nullptr)
                         ? std::make_shared<kd_node<T>>(dim, depth, indices, n_indices)
                         : std::make_shared<kd_node<T>>(dim, depth, indices, n_indices,
                                                        *min_bounds, *max_bounds);

    // If needed compute the bounding box for the node
    if (min_bounds == nullptr) {
        this_node->min_bounds.resize(this->n_features, std::numeric_limits<T>::max());
        this_node->max_bounds.resize(this->n_features, std::numeric_limits<T>::lowest());

        for (da_int j = 0; j < this->n_features; j++) {
            da_int A_offset_tmp = j * this->lda;
            for (da_int i = 0; i < n_indices; i++) {
                this_node->min_bounds[j] = std::min(this_node->min_bounds[j],
                                                    this->A[indices[i] + A_offset_tmp]);
                this_node->max_bounds[j] = std::max(this_node->max_bounds[j],
                                                    this->A[indices[i] + A_offset_tmp]);
            }
        }
    }

    // If the number of indices is such that further splitting would reduce it to below leaf size,
    // or result in an empty child node, then set the node to be a leaf node then return
    if (n_indices < 2 * this->leaf_size || n_indices == 2) {
        this_node->is_leaf = true;
        return this_node;
    }

    // Find the dimension to split on
    if (depth == 0) {
        dim = root_dim;
    } else {
        da_int best_dim = 0;
        T current_variance = 0.0, best_variance = 0.0;
        for (da_int i = 0; i < this->n_features; i++) {
            current_variance = single_pass_variance(indices, n_indices, i);
            if (current_variance > best_variance) {
                best_variance = current_variance;
                best_dim = i;
            }
        }
        dim = best_dim;
    }

    da_int A_offset = dim * this->lda;
    this_node->dim = dim;

    // Find the median point in the current dimension, accounting for zero-based indexing
    da_int mid = (n_indices - 1) / 2;

    std::nth_element(indices, indices + mid, indices + n_indices,
                     [this, &A_offset](da_int x, da_int y) {
                         return this->A[x + A_offset] < this->A[y + A_offset];
                     });

    // Assign the median point as the node's splitting point
    this_node->point = indices[mid];

    // Recursively build the left and right child nodes, but only spawn tasks if the workload is large enough
    if (mid > KD_TREE_MIN_TASK_SIZE) {
        // Some older compilers don't like the use of "omp task if" so use an explicit if statement
#pragma omp task firstprivate(depth, mid, indices, this_node)
        { this_node->left_child = build_tree(depth + 1, indices, mid); }
#pragma omp task firstprivate(depth, mid, indices, n_indices, this_node)
        {
            this_node->right_child =
                build_tree(depth + 1, indices + mid + 1, n_indices - mid - 1);
        }
    } else {
        this_node->left_child = build_tree(depth + 1, indices, mid);
        this_node->right_child =
            build_tree(depth + 1, indices + mid + 1, n_indices - mid - 1);
    }

    return this_node;
}

// Recursive function to find the radius neighbors of a point (determined by index_X) in X
template <typename T>
da_status kd_tree<T>::radius_neighbors_recursive(std::shared_ptr<kd_node<T>> current_node,
                                                 T *X, T eps, T eps_internal,
                                                 da_vector::da_vector<da_int> &neighbors,
                                                 bool X_is_A, da_int index_X, T X_norm) {

    da_status status = da_status_success;

    // Check the bounding box for quick pruning of the search space
    da_neighbors_types::nn_check_region proximity = check_bounding_box(
        X, eps_internal, current_node->min_bounds, current_node->max_bounds);
    if (proximity == da_neighbors_types::pt_outside_eps) {
        // The point is too far from the bounding box for this node, we can return and ignore all sub-nodes
        return da_status_success;
    }

    if (proximity == da_neighbors_types::region_within_eps) {
        // The entire bounding box is inside the search radius, so we can add all points in the node
        for (da_int i = 0; i < current_node->n_indices; i++) {
            da_int index_A = current_node->indices[i];
            if (X_is_A && index_A == index_X) {
                // If we are using the original dataset, skip the point itself so we don't add it to its own neighbors
                continue;
            }
            neighbors.push_back(index_A);
        }
        return da_status_success;
    }

    T dist;

    if (current_node->is_leaf) {
        // Check all the points in the node

        for (da_int i = 0; i < current_node->n_indices; i++) {

            da_int index_A = current_node->indices[i];

            if (X_is_A && index_A == index_X) {
                // If we are using the original dataset, skip the point itself so we don't add it to its own neighbors
                continue;
            }

            status = this->compute_distance(dist, index_A, X, X_norm);
            if (status != da_status_success) {
                return status; // LCOV_EXCL_LINE
            }

            if (dist <= eps_internal) {
                // If the distance is less than or equal to eps_internal, add the point to the neighbors list
                neighbors.push_back(index_A);
            }
        }

    } else {
        // This is not a leaf node, so only has a single point, which we need to check
        da_int index_A = current_node->point;

        if (!(X_is_A && index_A == index_X)) {
            // If we are using the original dataset, make sure we don't add the point to its own neighbors

            status = this->compute_distance(dist, index_A, X, X_norm);
            if (status != da_status_success) {
                return status; // LCOV_EXCL_LINE
            }

            if (dist <= eps_internal) {
                // If the distance is less than or equal to eps_internal, add the point to the neighbors list
                neighbors.push_back(index_A);
            }
        }

        // Check the splitting dimension
        da_int dim = current_node->dim;
        T diff = X[dim] - this->A[index_A + this->lda * dim];

        if (diff <= eps) {
            // Check the left child
            radius_neighbors_recursive(current_node->left_child, X, eps, eps_internal,
                                       neighbors, X_is_A, index_X, X_norm);
        }
        if (diff >= -eps) {
            // Check the right child
            radius_neighbors_recursive(current_node->right_child, X, eps, eps_internal,
                                       neighbors, X_is_A, index_X, X_norm);
        }
    }
    return da_status_success;
}

// Recursive function to find the k nearest neighbors of a point (determined by index_X) in X
template <typename T>
da_status kd_tree<T>::k_neighbors_recursive(std::shared_ptr<kd_node<T>> current_node,
                                            T *X, da_int k, bool X_is_A, da_int index_X,
                                            T X_norm, MaxHeap<T> &heap) {

    da_status status = da_status_success;

    // If the heap is full we need to check the bounding box, otherwise we can skip this check
    da_neighbors_types::nn_check_region proximity =
        (heap.GetSize() < k)
            ? da_neighbors_types::pt_within_eps
            : check_bounding_box(X, heap.GetMaxDist(), current_node->min_bounds,
                                 current_node->max_bounds);

    // If the point is too far from the bounding box for this node, we can return and ignore all sub-nodes
    if (proximity == da_neighbors_types::pt_outside_eps) {
        return da_status_success;
    }

    if (current_node->is_leaf) {
        // Check all the points in the node
        for (da_int i = 0; i < current_node->n_indices; i++) {

            da_int index_A = current_node->indices[i];

            if (X_is_A && index_A == index_X) {
                // If we are using the original dataset, skip the point itself so we don't add it to its own neighbors
                continue;
            }

            T dist;
            status = this->compute_distance(dist, index_A, X, X_norm);
            if (status != da_status_success) {
                return status; // LCOV_EXCL_LINE
            }

            // Add the point to the heap (the heap itself will handle the max distance)
            heap.Insert(index_A, dist);
        }

    } else {
        // This is not a leaf node, so only has a single point, which we should check
        da_int index_A = current_node->point;

        if (!(X_is_A && index_A == index_X)) {
            T dist;
            status = this->compute_distance(dist, index_A, X, X_norm);
            if (status != da_status_success) {
                return status; // LCOV_EXCL_LINE
            }

            heap.Insert(index_A, dist);
        }

        // Check the splitting dimension
        da_int dim = current_node->dim;
        T diff = X[dim] - this->A[index_A + this->lda * dim];

        // diff_tmp accounts for the square of the distances used in da_euclidean
        T diff_tmp = (this->metric == da_euclidean) || (this->metric == da_euclidean_gemm)
                         ? diff * std::abs(diff)
                         : diff;

        // Whether we check the left or right child first depends on the sign of diff - this has a significant performance impact

        if (diff < (T)0.0) {
            // Check the left child first
            k_neighbors_recursive(current_node->left_child, X, k, X_is_A, index_X, X_norm,
                                  heap);

            if (diff_tmp >= -heap.GetMaxDist()) {
                // Check the right child
                k_neighbors_recursive(current_node->right_child, X, k, X_is_A, index_X,
                                      X_norm, heap);
            }
        } else {
            // Check the right child first
            k_neighbors_recursive(current_node->right_child, X, k, X_is_A, index_X,
                                  X_norm, heap);

            if (diff_tmp <= heap.GetMaxDist()) {
                // Check the left child
                k_neighbors_recursive(current_node->left_child, X, k, X_is_A, index_X,
                                      X_norm, heap);
            }
        }
    }

    return da_status_success;
}

/* Check if a point X might be within distance eps of a box defined by the min_bounds and max_bounds coordinates
*  Return: 0 if X is further than eps from the box
*          1 if X is within eps of the box
*          2 if the entirety of the box is within eps of X
*/
template <typename T>
da_neighbors_types::nn_check_region
kd_tree<T>::check_bounding_box(T *X, T eps, std::vector<T> &min_bounds,
                               std::vector<T> &max_bounds) {

    // Note that if the user specified metric = da_euclidean, eps will have been squared to enable us to avoid taking square roots

    // min_dist will be the minimum distance from X to the bounding box - zero if X is inside
    T min_dist = 0.0;

    // max_dist will be the maximum distance from X to any corner of the bounding box
    T max_dist = 0.0;

    T tmp_min_dist, tmp_max_dist;

    // Special case for (squared)-Euclidean distance as it's a bit faster
    if ((this->metric == da_euclidean) || (this->metric == da_euclidean_gemm)) {
        for (da_int i = 0; i < this->n_features; i++) {

            if (X[i] < min_bounds[i]) {
                tmp_min_dist = min_bounds[i] - X[i];
                tmp_max_dist = max_bounds[i] - X[i];
            } else if (X[i] > max_bounds[i]) {
                tmp_min_dist = X[i] - max_bounds[i];
                tmp_max_dist = X[i] - min_bounds[i];
            } else {
                tmp_max_dist = std::max(max_bounds[i] - X[i], X[i] - min_bounds[i]);
                tmp_min_dist = (T)0.0;
            }
            tmp_min_dist *= tmp_min_dist;
            tmp_max_dist *= tmp_max_dist;
            if (tmp_min_dist > eps) {
                // Quick return here as we know X is further than eps from the bounding box
                return da_neighbors_types::pt_outside_eps;
            }
            min_dist += tmp_min_dist;
            max_dist += tmp_max_dist;
        }
    } else {

        for (da_int i = 0; i < this->n_features; i++) {

            if (X[i] < min_bounds[i]) {
                tmp_min_dist = min_bounds[i] - X[i];
                tmp_max_dist = max_bounds[i] - X[i];
            } else if (X[i] > max_bounds[i]) {
                tmp_min_dist = X[i] - max_bounds[i];
                tmp_max_dist = X[i] - min_bounds[i];
            } else {
                tmp_max_dist = std::max(max_bounds[i] - X[i], X[i] - min_bounds[i]);
                tmp_min_dist = (T)0.0;
            }
            if (tmp_min_dist > eps) {
                // Quick return here as we know X is further than eps from the bounding box
                return da_neighbors_types::pt_outside_eps;
            }
            if (this->metric == da_manhattan) {
                min_dist += tmp_min_dist;
                max_dist += tmp_max_dist;
            } else {
                min_dist += std::pow(tmp_min_dist, this->p);
                max_dist += std::pow(tmp_max_dist, this->p);
            }
        }
        if (this->metric != da_manhattan) {
            min_dist = std::pow(min_dist, this->p_inv);
            max_dist = std::pow(max_dist, this->p_inv);
        }
    }

    if (max_dist <= eps) {
        // If the maximum distance is less than eps, then the entire bounding box is within eps of X
        return da_neighbors_types::region_within_eps;
    }
    // If the minimum distance is less than eps, then X is within eps of the bounding box
    if (min_dist <= eps) {
        return da_neighbors_types::pt_within_eps;
    }
    // Otherwise, the point is outside the bounding box
    return da_neighbors_types::pt_outside_eps;
}

// Explicit instantiation of the k-d tree class for double and float types

template class kd_tree<double>;
template class kd_tree<float>;

} // namespace da_binary_tree
} // namespace ARCH
