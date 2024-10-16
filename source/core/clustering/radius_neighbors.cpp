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
#include "da_error.hpp"
#include "da_omp.hpp"
#include "da_vector.hpp"
#include "macros.h"
#include "pairwise_distances.hpp"
#include <vector>

#define RADIUS_NEIGHBORS_BLOCK_SIZE da_int(128)

namespace ARCH {

namespace da_radius_neighbors {

using neighbors_t = std::vector<da_vector::da_vector<da_int>>;

/*
Compute the radius neighbors: for each sample point, the indices of the samples within a given
radius are returned. The brute-force method is used.
*/
template <typename T>
da_status radius_neighbors(da_int n_samples, da_int n_features, const T *A, da_int lda,
                           T eps, std::vector<da_vector::da_vector<da_int>> &neighbors,
                           da_errors::da_error_t *err) {

    // 2D blocking scheme and threading scheme
    da_int max_block_size = std::min(RADIUS_NEIGHBORS_BLOCK_SIZE, n_samples);

    da_int block_rem, n_blocks;
    ARCH::da_utils::blocking_scheme(n_samples, max_block_size, n_blocks, block_rem);

    da_int n_threads = ARCH::da_utils::get_n_threads_loop(n_blocks * n_blocks);

    std::vector<T> D, A_norms;
    T eps_squared = eps * eps;

    try {
        D.resize(max_block_size * max_block_size * n_threads);
        A_norms.resize(n_samples);
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    da_int ldd = max_block_size;

    // Precompute the row norms of A to speed up Euclidean distance computation
    for (da_int j = 0; j < n_features; j++) {
        for (da_int i = 0; i < n_samples; i++) {
            A_norms[i] += A[i + j * lda] * A[i + j * lda];
        }
    }

    da_int threading_error = 0;

    // Local storage for neighbors to help avoid thread contention
    std::vector<neighbors_t> neighbors_local(n_threads);

#pragma omp parallel num_threads(n_threads) default(none)                                \
    shared(threading_error, neighbors, max_block_size, n_samples, D, A, A_norms,         \
               block_rem, ldd, lda, eps_squared, n_blocks, n_features, neighbors_local)
    {

        // Thread 0 can write to neighbors; all other threads need to use neighbors_local
        da_int this_thread = omp_get_thread_num();
        if (this_thread > 0) {
            try {
                neighbors_local[this_thread].resize(n_samples);
            } catch (std::bad_alloc const &) {
#pragma omp atomic write
                threading_error = 1;
            }
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
                            da_int D_index =
                                task_thread * max_block_size * max_block_size;
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
                            ARCH::euclidean_distance(
                                da_order::column_major, block_size_dim1, block_size_dim2,
                                n_features, &A[A_index_block_i], lda, &A[A_index_block_j],
                                lda, &D[D_index], ldd, &A_norms[A_index_block_i], 1,
                                &A_norms[A_index_block_j], 1, true, diagonal_block);

                            // Iterate through the distance matrix and store the indices of the samples within the radius
                            for (da_int jj = 0; jj < block_size_dim2; jj++) {
                                da_int ii_max =
                                    (diagonal_block) ? jj : block_size_dim1 - 1;
                                for (da_int ii = 0; ii <= ii_max; ii++) {
                                    // i and j correspond to the actual sample point indices we are considering
                                    da_int i = A_index_block_i + ii;
                                    da_int j = A_index_block_j + jj;
                                    if (D[D_index + ii + ldd * jj] <= eps_squared &&
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
    } // End of parallel region
    if (threading_error != 0)
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");

    return da_status_success;
}

template da_status
radius_neighbors<double>(da_int n_samples, da_int n_features, const double *A, da_int lda,
                         double eps, std::vector<da_vector::da_vector<da_int>> &neighbors,
                         da_errors::da_error_t *err);
template da_status
radius_neighbors<float>(da_int n_samples, da_int n_features, const float *A, da_int lda,
                        float eps, std::vector<da_vector::da_vector<da_int>> &neighbors,
                        da_errors::da_error_t *err);

} // namespace da_radius_neighbors

} // namespace ARCH
