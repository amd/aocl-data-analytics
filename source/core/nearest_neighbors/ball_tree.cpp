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
#include "pairwise_distances.hpp"
#include <algorithm>
#include <limits>
#include <new>
#include <stack>

#define BALL_TREE_MIN_TASK_SIZE da_int(2048)
#define BALL_TREE_BLOCK_SIZE da_int(256)

namespace ARCH {

namespace da_binary_tree {

// Compute centroid and radius for the whole dataset: blocked and parallelized for tall thin datasets
template <typename T>
da_status parallel_centroid_radius(da_int n_samples, da_int n_features, const T *A,
                                   da_int lda, da_metric metric, T p,
                                   std::vector<T> &centroid, T &radius) {
    // Check for valid input
    da_int n_blocks = std::min((da_int)omp_get_max_threads(),
                               std::max(n_samples / BALL_TREE_BLOCK_SIZE, (da_int)1));
    da_int block_size = n_samples / n_blocks;

    da_int block_rem = block_size + n_samples % block_size;

    bool internal_error = false;

    da_std::fill(centroid.begin(), centroid.end(), 0.0);
    std::vector<T> thd_means, thd_radius;
    try {
        thd_means.resize(n_features * n_blocks);
        thd_radius.resize(n_blocks);
    } catch (std::bad_alloc const &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    }

    da_int thd_block_size = block_size;
    da_int block_index;

#pragma omp parallel num_threads(n_blocks)                                               \
    firstprivate(thd_block_size) private(block_index) default(none)                      \
    shared(n_blocks, block_rem, n_samples, A, lda, thd_means, n_features, thd_radius,    \
               internal_error, centroid, metric, p, block_size)
    {
        da_status thd_status = da_status_success;
        // Ensure each thread gets a single contiguous block of the dataset so we can combine the means afterwards
#pragma omp for schedule(static, 1)
        for (da_int i = 0; i < n_blocks; i++) {
            block_index = i * thd_block_size;
            if (i == n_blocks - 1)
                thd_block_size = block_rem;

            thd_status = da_basic_statistics::mean(
                column_major, da_axis_col, thd_block_size, n_features, &A[block_index],
                lda, thd_means.data() + i * n_features);
            if (thd_status != da_status_success) {
#pragma omp atomic write
                internal_error = true;
            } else {
                // Scale the mean by the block size
                for (da_int j = 0; j < n_features; j++) {
                    thd_means[i * n_features + j] *= thd_block_size;
                }
            }
        }

#pragma omp single
        {
            // Combine the means computed by each thread
            for (da_int j = 0; j < n_features; j++) {
                for (da_int i = 0; i < n_blocks; i++) {
                    centroid[j] += thd_means[i * n_features + j];
                }
                centroid[j] /= n_samples;
            }
        }

        // Compute the radius of the ball
        thd_block_size = block_size;
#pragma omp for schedule(static, 1)
        for (da_int i = 0; i < n_blocks; i++) {
            block_index = i * thd_block_size;
            if (i == n_blocks - 1)
                thd_block_size = block_rem;

            // Each thread is only doing one block so only one allocation per thread here, which is fine
            std::vector<T> distances;
            try {
                distances.resize(thd_block_size);
            } catch (std::bad_alloc const &) {
#pragma omp atomic write
                internal_error = true;
            }
            if (!internal_error) {
                thd_status =
                    ARCH::da_metrics::pairwise_distances::pairwise_distance_kernel(
                        da_order::column_major, thd_block_size, 1, n_features,
                        &A[block_index], lda, centroid.data(), 1, distances.data(),
                        thd_block_size, p, metric);
                if (thd_status != da_status_success) {
#pragma omp atomic write
                    internal_error = true;
                } else {
                    // Compute the radius as the maximum distance from the centroid
                    da_int max_index_thd =
                        da_blas::cblas_iamax(thd_block_size, distances.data(), (da_int)1);
                    thd_radius[i] = distances[max_index_thd];
                }
            }
        }
    }

    if (internal_error) {
        return da_status_internal_error; // LCOV_EXCL_LINE
    }

    // Combine the radii computed by each thread
    da_int max_index = da_blas::cblas_iamax(n_blocks, thd_radius.data(), (da_int)1);
    radius = thd_radius[max_index];
    return da_status_success;
}

template <typename T>
ball_tree<T>::ball_tree(da_int n_samples_in, da_int n_features_in, const T *A_in,
                        da_int lda_in, da_int leaf_size_in, da_metric metric_in, T p_in) {
    // Initialize the ball tree
    this->store_data(n_samples_in, n_features_in, A_in, lda_in, leaf_size_in, metric_in,
                     p_in);

    // Allocate memory for the indices (which will initially be filled with 0, 1, 2, ..., n_samples - 1)
    this->indices = std::vector<da_int>(this->n_samples);

    da_int n_threads = omp_get_max_threads();
    // If memory allocation fails an exception will be thrown, so the constructor must be wrapped in a try...catch
    this->A_row1.resize(this->n_features * n_threads);
    this->A_row2.resize(this->n_features * n_threads);

    da_std::iota(this->indices.begin(), this->indices.end(), 0);

    // Compute the centroid and radius of the ball enclosing the whole dataset
    std::vector<T> centroid(this->n_features);
    T radius = 0.0;

    if (parallel_centroid_radius(this->n_samples, this->n_features, this->A, this->lda,
                                 this->metric, this->p, centroid,
                                 radius) != da_status_success) {
        throw std::runtime_error( // LCOV_EXCL_LINE
            "Failed to compute column means for the dataset");
    }

// Build the ball tree - careful use of default shared needed because we can't use this-> in OpenMP directives
#pragma omp parallel default(shared)
    {
#pragma omp single
        {
            this->root =
                build_tree(0, this->indices.data(), this->n_samples, &centroid, radius);
        }
    }
}

template <typename T>
void ball_tree<T>::node_centroid_radius(da_int *indices, da_int n_indices,
                                        std::vector<T> &centroid, T &radius) {

    for (da_int i = 0; i < n_indices; i++) {
        da_int index = indices[i];
        for (da_int j = 0; j < this->n_features; j++) {
            centroid[j] += this->A[index + j * this->lda];
        }
    }
    T X_norm = 0.0;
    for (da_int j = 0; j < this->n_features; j++) {
        centroid[j] /= n_indices;
        if (this->metric == da_euclidean_gemm)
            X_norm += centroid[j] * centroid[j];
    }
    radius = 0.0;
    for (da_int i = 0; i < n_indices; i++) {
        T tmp_dist = 0.0;
        da_status status =
            this->compute_distance(tmp_dist, indices[i], centroid.data(), X_norm);
        // This error should not be possible, but will be caught by the constructor
        if (status != da_status_success) {
            throw std::bad_alloc(); // LCOV_EXCL_LINE
        }

        radius = std::max(radius, tmp_dist);
    }

    if (this->metric == da_euclidean || this->metric == da_euclidean_gemm) {
        radius = std::sqrt(radius);
    }
}

// Find the furthest point in the dataset to the point at the given index
template <typename T>
void ball_tree<T>::furthest_point(da_int *indices, da_int n_indices, da_int index,
                                  da_int &furthest_index) {

    T max_distance = 0.0;

    da_int thread_id = omp_get_thread_num();

    // Copy the point at index to the workspace for better cache use
    da_int A_index = indices[index];
    for (da_int i = 0; i < this->n_features; i++) {
        this->A_row1[thread_id * this->n_features + i] = this->A[A_index + i * this->lda];
    }

    furthest_index = index; // Default to the point itself if no other point is found
    T dist = 0.0;

    T norm = (this->metric == da_euclidean_gemm) ? this->A_norms[indices[index]] : 0.0;

    for (da_int i = 0; i < n_indices; i++) {
        if (i != index) {
            da_status status = this->compute_distance(
                dist, indices[i], &this->A_row1[thread_id * this->n_features], norm);
            if (status != da_status_success) {
                throw std::bad_alloc(); // LCOV_EXCL_LINE
            }
            if (dist > max_distance) {
                max_distance = dist;
                furthest_index = i;
            }
        }
    }
}

// Return true if the point at index i is closer to centroid c1 than to centroid c2, where i
// is an index in the dataset, A
template <typename T>
bool ball_tree<T>::choose_centroid(T *c1, T c1_norm, da_int c1_index, T *c2, T c2_norm,
                                   da_int c2_index, da_int i) {

    if (c1_index == i) {
        // If the point at index i is c1, then it is closer to c1 than to c2
        return true;
    }
    if (c2_index == i) {
        // If the point at index i is c2, then it is closer to c2 than to c1
        return false;
    }

    T dist_c1 = 0.0;
    T dist_c2 = 0.0;

    // Compute the distance from the point at index i to the centroids A and B
    da_status status = this->compute_distance(dist_c1, i, c1, c1_norm);
    if (status != da_status_success) {
        throw std::bad_alloc(); // LCOV_EXCL_LINE
    }
    status = this->compute_distance(dist_c2, i, c2, c2_norm);
    if (status != da_status_success) {
        throw std::bad_alloc(); // LCOV_EXCL_LINE
    }

    return dist_c1 < dist_c2;
}

// Recursive function to build the ball tree
// The ball tree is built in a top-down manner, starting from the root node and recursively splitting
template <typename T>
std::shared_ptr<ball_node<T>>
ball_tree<T>::build_tree(da_int depth, da_int *indices, da_int n_indices,
                         std::vector<T> *centroid, T radius) {

    // If there are no indices, return nullptr
    if (n_indices == 0) {
        return nullptr; // LCOV_EXCL_LINE
    }

    // Create a new node for this part of the tree, with sensible defaults. Only the root node will have
    // centroid and radius supplied; for all other nodes these will be computed later
    auto this_node = (centroid == nullptr)
                         ? std::make_shared<ball_node<T>>(depth, indices, n_indices)
                         : std::make_shared<ball_node<T>>(depth, indices, n_indices,
                                                          *centroid, radius);

    // If needed compute the centroid and radius for the node
    if (centroid == nullptr) {
        this_node->centroid.resize(this->n_features, 0.0);
        this_node->radius = 0.0;

        node_centroid_radius(indices, n_indices, this_node->centroid, this_node->radius);
    }

    // If the number of indices is less than or equal to the leaf size, then set the node to be a leaf node then return
    if (n_indices <= this->leaf_size) {
        this_node->is_leaf = true;
        return this_node;
    }

    // Find the most distant point, c1 from the first point in the list of indices
    da_int c1 = 0;
    this->furthest_point(indices, n_indices, 0, c1);

    // Find the most distant point, c2, from c1
    da_int c2 = 0;
    this->furthest_point(indices, n_indices, c1, c2);

    // c1 and c2 represent where in the indices array the centroids are.
    // It will also be useful to store where they are within A
    da_int c1_index = indices[c1];
    da_int c2_index = indices[c2];

    // c1 and c2 form the initial centroids of the left and right child nodes; copy into workspace arrays
    // for better cache use
    da_int thread_id = omp_get_thread_num();
    da_int ws_index = thread_id * this->n_features;
    for (da_int i = 0; i < this->n_features; i++) {
        this->A_row1[ws_index + i] = this->A[c1_index + i * this->lda];
        this->A_row2[ws_index + i] = this->A[c2_index + i * this->lda];
    }

    // Assign each point to one of the two child nodes based on which centroid it is closer to
    da_int mid = 0;

    T norm_c1 = (this->metric == da_euclidean_gemm) ? this->A_norms[c1_index] : 0.0;
    T norm_c2 = (this->metric == da_euclidean_gemm) ? this->A_norms[c2_index] : 0.0;

    for (da_int i = 0; i < n_indices; i++) {
        // If the point is closer to c1 than to c2, it goes in the left child node
        if (this->choose_centroid(&this->A_row1[ws_index], norm_c1, c1_index,
                                  &this->A_row2[ws_index], norm_c2, c2_index,
                                  indices[i])) {
            std::swap(indices[i], indices[mid]);
            mid++;
        }
    }

    // mid now contains the number of points that are closer to c1 than to c2, so the left child node

    // If one of the child nodes is smaller than the leaf size, we cannot proceed with the splitting
    // Set the node to be a leaf node then return
    if (mid < this->leaf_size || n_indices - mid < this->leaf_size) {
        this_node->is_leaf = true;
        return this_node;
    }

    // Recursively build the left and right child nodes, but only spawn tasks if the workload is large enough
    if (mid > BALL_TREE_MIN_TASK_SIZE) {
        // Some older compilers don't like the use of "omp task if" so use an explicit if statement
#pragma omp task firstprivate(depth, mid, indices, this_node)
        { this_node->left_child = build_tree(depth + 1, indices, mid); }
#pragma omp task firstprivate(depth, mid, indices, n_indices, this_node)
        {
            this_node->right_child =
                build_tree(depth + 1, indices + mid, n_indices - mid);
        }
    } else {
        this_node->left_child = build_tree(depth + 1, indices, mid);
        this_node->right_child = build_tree(depth + 1, indices + mid, n_indices - mid);
    }

    return this_node;
}

/* Check if a point X might be within distance eps of a ball defined by a centroid and radius.
*  Return: pt_outside_eps if X is further than eps from the edge of the ball
*          pt_within_eps if X is within eps of the edge of the ball
*          region_within_eps if the entirety of the ball is within eps of X
*/
template <typename T>
da_neighbors_types::nn_check_region
ball_tree<T>::check_ball(T *X, T eps, std::vector<T> &centroid, T radius, T &dist) {

    dist = 0.0;

    // Special case for Euclidean distance as it's a bit faster
    if (this->metric == da_euclidean || this->metric == da_euclidean_gemm) {
        for (da_int i = 0; i < this->n_features; i++) {
            T tmp = X[i] - centroid[i];
            tmp *= tmp;
            dist += tmp;
        }
        dist = std::sqrt(dist);
    } else {
        for (da_int i = 0; i < this->n_features; i++) {
            T tmp = std::abs(X[i] - centroid[i]);

            if (this->metric == da_manhattan) {
                dist += tmp;
            } else {
                dist += std::pow(tmp, this->p);
            }
        }
        if (this->metric != da_manhattan) {
            dist = std::pow(dist, this->p_inv);
        }
    }

    if (dist + radius <= eps) {
        // The entire ball is within eps of X
        return da_neighbors_types::region_within_eps;
    }
    // If the minimum distance is less than eps, then X is within eps of the edge of the ball
    if (dist - radius <= eps) {
        return da_neighbors_types::pt_within_eps;
    }
    // Otherwise, the point is outside the ball by at least eps
    return da_neighbors_types::pt_outside_eps;
}

// Recursive function to find the radius neighbors of a point (determined by index_X) in X
template <typename T>
da_status ball_tree<T>::radius_neighbors_recursive(
    std::shared_ptr<ball_node<T>> current_node, T *X, T eps, T eps_internal,
    da_vector::da_vector<da_int> &neighbors, bool X_is_A, da_int index_X, T X_norm) {

    da_status status = da_status_success;

    // Check the ball for quick pruning of the search space
    T dist;
    da_neighbors_types::nn_check_region proximity =
        check_ball(X, eps, current_node->centroid, current_node->radius, dist);
    if (proximity == da_neighbors_types::pt_outside_eps) {
        // The point is too far from the bounding ball for this node, we can return and ignore all sub-nodes
        return da_status_success;
    }

    if (proximity == da_neighbors_types::region_within_eps) {
        // The entire ball is inside the search radius, so we can add all points in the node
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
        // This is not a leaf node, so check the sub-nodes

        // Check the left child
        radius_neighbors_recursive(current_node->left_child, X, eps, eps_internal,
                                   neighbors, X_is_A, index_X, X_norm);

        // Check the right child
        radius_neighbors_recursive(current_node->right_child, X, eps, eps_internal,
                                   neighbors, X_is_A, index_X, X_norm);
    }
    return da_status_success;
}

// Recursive function to find the k nearest neighbors of a point (determined by index_X) in X
template <typename T>
da_status ball_tree<T>::k_neighbors_recursive(std::shared_ptr<ball_node<T>> current_node,
                                              T *X, da_int k, bool X_is_A, da_int index_X,
                                              T X_norm, MaxHeap<T> &heap) {

    da_status status = da_status_success;
    T dist;
    // If the heap is full we need to check the ball, otherwise we can skip this check
    T heap_max_dist = 0.0;

    da_int ball;
    if (heap.GetSize() < k)
        ball = 1;
    else {
        heap_max_dist =
            (this->metric == da_euclidean) || (this->metric == da_euclidean_gemm)
                ? std::sqrt(heap.GetMaxDist())
                : heap.GetMaxDist();
        ball = check_ball(X, heap_max_dist, current_node->centroid, current_node->radius,
                          dist);
    }
    // If the point is too far from the bounding box for this node, we can return and ignore all sub-nodes
    if (ball == 0) {
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
        // This is not a leaf node, so check the sub-nodes
        T dist_left = 0.0, dist_right = 0.0;
        da_int ball_left = 0, ball_right = 0;

        heap_max_dist =
            (this->metric == da_euclidean) || (this->metric == da_euclidean_gemm)
                ? std::sqrt(heap.GetMaxDist())
                : heap.GetMaxDist();
        ball_left = check_ball(X, heap_max_dist, current_node->left_child->centroid,
                               current_node->left_child->radius, dist_left);
        ball_right = check_ball(X, heap_max_dist, current_node->right_child->centroid,
                                current_node->right_child->radius, dist_right);

        // Whether we check the left or right child first depends on which is closest
        if (dist_left < dist_right) {
            // Check the left child first
            if (ball_left != 0)
                k_neighbors_recursive(current_node->left_child, X, k, X_is_A, index_X,
                                      X_norm, heap);

            // Check the right child
            heap_max_dist =
                (this->metric == da_euclidean) || (this->metric == da_euclidean_gemm)
                    ? std::sqrt(heap.GetMaxDist())
                    : heap.GetMaxDist();
            if (ball_right != 0 &&
                dist_right - current_node->right_child->radius <= heap_max_dist)
                k_neighbors_recursive(current_node->right_child, X, k, X_is_A, index_X,
                                      X_norm, heap);

        } else {
            // Check the right child first
            if (ball_right != 0)
                k_neighbors_recursive(current_node->right_child, X, k, X_is_A, index_X,
                                      X_norm, heap);

            // Check the left child
            heap_max_dist =
                (this->metric == da_euclidean) || (this->metric == da_euclidean_gemm)
                    ? std::sqrt(heap.GetMaxDist())
                    : heap.GetMaxDist();
            if (ball_left != 0 &&
                dist_left - current_node->left_child->radius <= heap_max_dist)
                k_neighbors_recursive(current_node->left_child, X, k, X_is_A, index_X,
                                      X_norm, heap);
        }
    }
    return da_status_success;
}

// Explicit instantiation of the ball tree class for double and float types
template class ball_tree<double>;
template class ball_tree<float>;

} // namespace da_binary_tree
} // namespace ARCH
