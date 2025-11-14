/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include "radius_neighbors.hpp"
#include "aoclda.h"
#include "binary_tree.hpp"
#include "da_error.hpp"
#include "da_omp.hpp"
#include "da_vector.hpp"
#include "macros.h"
#include "pairwise_distances.hpp"
#include <vector>

#define RADIUS_NEIGHBORS_BLOCK_SIZE da_int(512)

namespace ARCH {

namespace da_radius_neighbors {

using neighbors_t = std::vector<da_vector::da_vector<da_int>>;

/*
Compute the radius neighbors: for each sample point, the indices of the samples within a given
radius are returned. The brute-force method is used.
*/
template <typename T>
da_status radius_neighbors_brute(da_int n_samples, da_int n_features, const T *A,
                                 da_int lda, T eps, da_metric metric, T p,
                                 std::vector<da_vector::da_vector<da_int>> &neighbors,
                                 da_errors::da_error_t *err) {

    // 2D blocking scheme and threading scheme
    da_int max_block_size = std::min(RADIUS_NEIGHBORS_BLOCK_SIZE, n_samples);
    da_int max_block_size_sq = max_block_size * max_block_size;

    da_int block_rem, n_blocks;
    ARCH::da_utils::blocking_scheme(n_samples, max_block_size, n_blocks, block_rem);

    da_int n_threads = ARCH::da_utils::get_n_threads_loop(n_blocks * n_blocks);

    std::vector<T> A_norms;
    std::vector<std::vector<T>> D;

    // For da_euclidean it is more efficient to use the squared distance
    T eps_internal = (metric == da_euclidean_gemm) ? eps * eps : eps;

    da_metric metric_internal =
        (metric == da_euclidean_gemm || (metric == da_minkowski && p == T(2.0)))
            ? da_sqeuclidean_gemm
            : metric;

    try {
        D.resize(n_threads);
        if (metric_internal == da_sqeuclidean_gemm)
            A_norms.resize(n_samples);
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    da_int ldd = max_block_size;

    if (metric_internal == da_sqeuclidean_gemm) {
        // Precompute the row norms of A to speed up Euclidean distance computation
        for (da_int j = 0; j < n_features; j++) {
            for (da_int i = 0; i < n_samples; i++) {
                A_norms[i] += A[i + j * lda] * A[i + j * lda];
            }
        }
    }

    da_int threading_error = 0;

    // Local storage for neighbors to help avoid thread contention
    std::vector<neighbors_t> neighbors_local(n_threads);

#pragma omp parallel num_threads(n_threads) default(none)                                \
    shared(threading_error, neighbors, max_block_size, max_block_size_sq, n_samples, D,  \
               A, A_norms, block_rem, ldd, lda, eps_internal, n_blocks, n_features,      \
               neighbors_local, metric_internal, p)
    {

        // Thread 0 can write to neighbors; all other threads need to use neighbors_local
        da_int this_thread = omp_get_thread_num();

        try {
            if (this_thread > 0) {
                neighbors_local[this_thread].resize(n_samples);
            }
            D[this_thread].resize(max_block_size_sq);
        } catch (std::bad_alloc const &) {
#pragma omp atomic write
            threading_error = 1;
        }

#pragma omp single
        {
            for (da_int block_j = 0; block_j < n_blocks; block_j++) {
                for (da_int block_i = 0; block_i <= block_j; block_i++) {
#pragma omp task firstprivate(block_i, block_j)
                    {
                        da_int task_error;
#pragma omp atomic read
                        task_error = threading_error;
                        if (task_error == 0) {
                            // Need a separate thread_num variable here since we don't know which thread will execute this task
                            da_int task_thread = omp_get_thread_num();
                            //da_int D_index = task_thread * max_block_size_sq;
                            da_int A_index_block_i = block_i * max_block_size;
                            da_int A_index_block_j = block_j * max_block_size;
                            da_int block_size_dim1 = max_block_size;
                            if (block_i == n_blocks - 1 && block_rem > 0)
                                block_size_dim1 = block_rem;
                            da_int block_size_dim2 = max_block_size;
                            if (block_j == n_blocks - 1 && block_rem > 0)
                                block_size_dim2 = block_rem;
                            bool diagonal_block = (block_i == block_j) ? true : false;

                            // Compute the distance matrix
                            if (metric_internal == da_sqeuclidean_gemm) {
                                ARCH::euclidean_gemm_distance(
                                    da_order::column_major, block_size_dim1,
                                    block_size_dim2, n_features, &A[A_index_block_i], lda,
                                    &A[A_index_block_j], lda, D[task_thread].data(), ldd,
                                    &A_norms[A_index_block_i], 1,
                                    &A_norms[A_index_block_j], 1, true, diagonal_block);
                            } else {
                                // Compute the distance matrix using the specified metric
                                const T *A_j =
                                    diagonal_block ? nullptr : &A[A_index_block_j];
                                da_status thd_status = ARCH::da_metrics::
                                    pairwise_distances::pairwise_distance_kernel(
                                        da_order::column_major, block_size_dim1,
                                        block_size_dim2, n_features, &A[A_index_block_i],
                                        lda, A_j, lda, D[task_thread].data(), ldd, p,
                                        metric_internal);
                                if (thd_status != da_status_success) {
#pragma omp atomic write
                                    threading_error = 1;
                                }
                            }
                            // Iterate through the distance matrix and store the indices of the samples within the radius
                            for (da_int jj = 0; jj < block_size_dim2; jj++) {
                                da_int ii_max =
                                    (diagonal_block) ? jj : block_size_dim1 - 1;
                                da_int j = A_index_block_j + jj;
                                da_int D_index_offset = ldd * jj;
                                for (da_int ii = 0; ii <= ii_max; ii++) {
                                    // i and j correspond to the actual sample point indices we are considering
                                    da_int i = A_index_block_i + ii;
                                    if (D[task_thread][D_index_offset + ii] <=
                                            eps_internal &&
                                        i != j) {
                                        try {
                                            if (task_thread == 0) {
                                                neighbors[i].push_back(j);
                                                neighbors[j].push_back(i);
                                            } else {
                                                neighbors_local[task_thread][i].push_back(
                                                    j);
                                                neighbors_local[task_thread][j].push_back(
                                                    i);
                                            }
                                        } catch (std::bad_alloc const &) {
#pragma omp atomic write
                                            threading_error = 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

// Merge the local neighbors into the global neighbors
#pragma omp critical
        {
            if (threading_error == 0 && this_thread != 0) {
                for (da_int i = 0; i < n_samples; i++) {
                    neighbors[i].append(neighbors_local[this_thread][i]);
                }
            }
        }
        neighbors_local[this_thread] = neighbors_t{};
    } // End of parallel region
    if (threading_error != 0)
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");

    return da_status_success;
}

/*
Compute the radius neighbors: for each sample point, the indices of the samples within a given
radius are returned. The k-d tree method is used.
*/
template <typename T>
da_status radius_neighbors_kd_tree(da_int n_samples, da_int n_features, const T *A,
                                   da_int lda, T eps, da_metric metric, T p,
                                   da_int leaf_size,
                                   std::vector<da_vector::da_vector<da_int>> &neighbors,
                                   da_errors::da_error_t *err) {
    try {
        // Form a k-d tree from the dataset
        auto tree = ARCH::da_binary_tree::kd_tree<T>(n_samples, n_features, A, lda,
                                                     leaf_size, metric, p);
        return tree.radius_neighbors(n_samples, n_features, nullptr, 0, eps, neighbors,
                                     err);

    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
}

/*
Compute the radius neighbors: for each sample point, the indices of the samples within a given
radius are returned. The ball tree method is used.
*/
template <typename T>
da_status radius_neighbors_ball_tree(da_int n_samples, da_int n_features, const T *A,
                                     da_int lda, T eps, da_metric metric, T p,
                                     da_int leaf_size,
                                     std::vector<da_vector::da_vector<da_int>> &neighbors,
                                     da_errors::da_error_t *err) {
    try {
        // Form a ball tree from the dataset
        auto tree = ARCH::da_binary_tree::ball_tree<T>(n_samples, n_features, A, lda,
                                                       leaf_size, metric, p);
        return tree.radius_neighbors(n_samples, n_features, nullptr, 0, eps, neighbors,
                                     err);

    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
}

template da_status
radius_neighbors_brute<double>(da_int n_samples, da_int n_features, const double *A,
                               da_int lda, double eps, da_metric metric, double p,
                               std::vector<da_vector::da_vector<da_int>> &neighbors,
                               da_errors::da_error_t *err);
template da_status
radius_neighbors_brute<float>(da_int n_samples, da_int n_features, const float *A,
                              da_int lda, float eps, da_metric metric, float p,
                              std::vector<da_vector::da_vector<da_int>> &neighbors,
                              da_errors::da_error_t *err);

template da_status radius_neighbors_kd_tree<double>(
    da_int n_samples, da_int n_features, const double *A, da_int lda, double eps,
    da_metric metric, double p, da_int leaf_size,
    std::vector<da_vector::da_vector<da_int>> &neighbors, da_errors::da_error_t *err);
template da_status radius_neighbors_kd_tree<float>(
    da_int n_samples, da_int n_features, const float *A, da_int lda, float eps,
    da_metric metric, float p, da_int leaf_size,
    std::vector<da_vector::da_vector<da_int>> &neighbors, da_errors::da_error_t *err);

template da_status radius_neighbors_ball_tree<double>(
    da_int n_samples, da_int n_features, const double *A, da_int lda, double eps,
    da_metric metric, double p, da_int leaf_size,
    std::vector<da_vector::da_vector<da_int>> &neighbors, da_errors::da_error_t *err);
template da_status radius_neighbors_ball_tree<float>(
    da_int n_samples, da_int n_features, const float *A, da_int lda, float eps,
    da_metric metric, float p, da_int leaf_size,
    std::vector<da_vector::da_vector<da_int>> &neighbors, da_errors::da_error_t *err);

} // namespace da_radius_neighbors

} // namespace ARCH
