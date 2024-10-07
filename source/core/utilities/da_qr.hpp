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

#ifndef DA_QR_HPP
#define DA_QR_HPP

#include "aoclda.h"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "da_omp.hpp"
#include "lapack_templates.hpp"
#include <vector>

/* This function implements a blocked, parallelized QR algorithm for tall skinny matrices, similar to

   “Communication-Optimal Parallel and Sequential QR and LU Factorizations,”
   J. Demmel, L. Grigori, M. Hoemmen, J. Langou, SIAM J. Sci. Comput, vol. 34, no. 1, 2012

   Here we use geqrf as the 'local' QR routine and we only implement one level of recursion rather than a full tree

   If the internal heuristic suggests that blocked QR is not warranted for this matrix
   size, then geqrf is called instead: effectively blocked QR with one block only.

   For the blocked QR factorization, the m x n matrix A, is split vertically up into n_blocks blocks of
   size block_size x n, except for the final block which may be larger (the values of n_blocks, block_size
   and final_block_size are computed and output by da_qr):

        /A1\
   A = | A2 |
       | .. |
        \Ak/

    For each block, i, geqrf is called to compute Qi * Ri, where each Qi is of size block_size x n, and each
    Ri is of size n x n. The Ai are overwritten as per the LAPACK routine, and the vector tau is used
    to store the elementary reflectors.

         /Q1 R1\    [Q1 Q2 ... Qk] * /R1\
    A = | Q2 R2 | =                 | R2 |
        |  ...  |                   | .. |
         \Qk Rk/                     \Rk/

    The Ri are then stacked into a single matrix, Rt, of size (n * n_blocks) x n, which is passed to geqrf:

          /R1\
    Rt = | R2 | = Q_R * R
         | .. |
          \Rk/

    where Q is of size (n * n_blocks) x n and R is of size n x n.

    The overall QR factorization of A is given by A = [Q1 Q2 ... Qk] * Q_R * R.

    Other than the initial matrix, A, da_qr performs all required memory allocation. The matrix R is
    returned in the std::vector R, the elementary reflectors and details of Q are stored in the lower
    triangle of R and in tau_R. The elementary reflectors and details of the Qi are stored in the
    lower triangles of the chunks of the overwritten A and in the corresponding entries of tau, which
    will be of size n_blocks * min(m, n).

    If, on output, n_blocks = 1, then the routine determined that block QR was not warranted and only A and
    tau are used, as per geqrf, with R and tau_R untouched. Either way, the final n x n triangular matrix
    is stored in R.

    The argument store_factors determines whether tau and tau_R_blocked are allocated and used to store
    details of the Q factors for reconstruction by da_qr_apply. If store_factors is false, then the
    variables are ignored and an alternative, faster, QR factorization method, dgeqrt3, is used.

 */

#define MAX_NUM_BLOCKS da_int(256)
#define MIN_BLOCK_SIZE da_int(1024)
template <typename T>
da_status da_qr(da_int m, da_int n, std::vector<T> &A, da_int lda, std::vector<T> &tau,
                std::vector<T> &R_blocked, std::vector<T> &tau_R_blocked,
                std::vector<T> &R, da_int &n_blocks, da_int &block_size,
                da_int &final_block_size, bool store_factors,
                da_errors::da_error_t *err) {

    // Find out how many threads we have available
    da_int n_threads = (da_int)omp_get_max_threads();

    if (omp_get_max_active_levels() == omp_get_level())
        n_threads = 1;

    /* Heuristic based on flop counts to determine the level of blocking. We need the following:
       1. m > n else it's never cheaper to do blocked QR
       2. block_size > n for same reason, which is implied by condition 4
       3. n_blocks < m / n x [ (n_threads-1) / (3 x n_threads - 1) ]
       4. n_blocks < m / n (which is implied by condition 3)
       5. n_blocks < 256 or some other suitable value e.g. number of cores on a node
       6. block_size > 2048 to prevent excessively small geqrf calls
       7. block_size to be rounded up to the nearest multiple of 256 for better cache use
       8. block_size < m
       9. the remainder, final_block_size, or size of the last block, must be no smaller than n to avoid a short wide QR
       10. n_blocks should exceed the number of threads available so we can exploit parallelism
     */
    da_int max_blocks =
        std::max(std::min(MAX_NUM_BLOCKS,
                          (da_int)(3 * (n_threads - 2) * m / ((3 * n_threads - 1) * n))),
                 (da_int)1);

    max_blocks = std::min(max_blocks, n_threads * 2);

    if (max_blocks == 1) {
        n_blocks = 1;
        final_block_size = m;
        block_size = m;
    } else {

        block_size = std::min(MIN_BLOCK_SIZE, m);

        if (m / block_size > max_blocks) {
            block_size = m / max_blocks;
            // Round up to nearest multiple of 256 as long as we don't exceed m
            block_size = std::min(((block_size + 255) >> 8) << 8, m);
        }

        n_blocks = m / block_size;
        final_block_size = m % block_size;
        // Count the remainder in the number of blocks but ensure it's larger than n, else concatenate with previous block
        if (final_block_size >= n) {
            n_blocks += 1;
        } else {
            final_block_size += block_size;
            if (n_blocks == 1)
                // Special case of 1 block with a small remainder
                block_size = final_block_size;
        }
    }

    n_threads = std::min(n_threads, n_blocks);

    da_int max_block_size = std::max(block_size, final_block_size);
    da_int mr = n_blocks * n;
    da_int tau_size;
    da_int lwork = -1, lwork_R = -1, info = 0;
    T dummy[1], dummy_R[1];
    std::vector<T> work, work_R;
    std::vector<T> Tr;

    tau_size = (n_blocks - 1) * std::min(block_size, n) + std::min(final_block_size, n);

    // Memory allocation for triangular factors
    try {
        R.resize(n * n, 0.0);
        if (n_blocks > 1) {
            R_blocked.resize(mr * n, 0.0);
        }
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    if (store_factors) {
        // Workspace queries to allocate memory for geqrf

        try {
            tau.resize(tau_size);
            if (n_blocks > 1) {
                tau_R_blocked.resize(n, 0.0);
            }
        } catch (std::bad_alloc const &) {
            return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }

        // Workspace queries to geqrf to allocate remaining memory
        da::geqrf(&max_block_size, &n, A.data(), &lda, tau.data(), dummy, &lwork, &info);
        if (info != 0) {
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "An internal error occurred. Please check "
                            "the input data for undefined values.");
        }
        lwork = (da_int)dummy[0];
        try {
            work.resize(n_threads * lwork);
        } catch (std::bad_alloc const &) {
            return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }

        if (n_blocks > 1) {
            da::geqrf(&mr, &n, R_blocked.data(), &mr, tau_R_blocked.data(), dummy_R,
                      &lwork_R, &info);
            if (info != 0) {
                return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                                "An internal error occurred. Please check "
                                "the input data for undefined values.");
            }
            try {
                lwork_R = (da_int)dummy_R[0];
                work_R.resize(lwork_R);
            } catch (std::bad_alloc const &) {
                return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                                "Memory allocation failed.");
            }
        }

    } else {
        // Allocate memory for dgeqrt3
        try {
            Tr.resize(n_threads * n * n);
        } catch (std::bad_alloc const &) {
            return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }
    }

    da_int info_local = 0;
    da_int this_block_size = block_size;
#pragma omp parallel for if (n_blocks > 1) num_threads(n_threads) default(none)          \
    firstprivate(info_local, this_block_size)                                            \
    shared(n_blocks, final_block_size, n, A, lda, tau, work, lwork, R_blocked, info, mr, \
               block_size, Tr, store_factors)
    for (da_int i = 0; i < n_blocks; i++) {
        if (info != 0)
            continue;
        if (i == n_blocks - 1)
            this_block_size = final_block_size;
        da_int this_thread = (da_int)omp_get_thread_num();
        if (store_factors) {
            da::geqrf(&this_block_size, &n, &A[i * block_size], &lda, &tau[i * n],
                      &work[this_thread * lwork], &lwork, &info_local);
        } else {
            da::geqrt3(&this_block_size, &n, &A[i * block_size], &lda,
                       &Tr[this_thread * n * n], &n, &info_local);
        }
        if (info_local != 0) {
#pragma omp atomic
            info += info_local;
        }
        if (n_blocks > 1) {
            // Copy the relevant n x n upper triangle into R_blocked
            for (da_int j = 0; j < n; j++) {
                for (da_int k = 0; k <= j; k++) {
                    R_blocked[i * n + mr * j + k] = A[i * block_size + lda * j + k];
                }
            }
        }
    }

    if (info != 0) {
        return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                        "An internal error occurred. Please check "
                        "the input data for undefined values.");
    }

    if (n_blocks > 1) {
        if (store_factors) {
            da::geqrf(&mr, &n, R_blocked.data(), &mr, tau_R_blocked.data(), work_R.data(),
                      &lwork_R, &info);
        } else {
            da::geqrt3(&mr, &n, R_blocked.data(), &mr, Tr.data(), &n, &info_local);
        }
        if (info != 0) {
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "An internal error occurred. Please check "
                            "the input data for undefined values.");
        }
        for (da_int j = 0; j < n; j++) {
            for (da_int i = 0; i <= j; i++) {
                R[i + j * n] = R_blocked[i + j * mr];
            }
        }
    } else {
        // Standard QR was performed so relevant matrix is in upper triangle of A
        for (da_int j = 0; j < n; j++) {
            for (da_int i = 0; i <= j; i++) {
                R[i + j * n] = A[i + j * lda];
            }
        }
    }

    return da_status_success;
}

/* This function complements the blocked, parallelized QR algorithm for tall, skinny matrices
   implemented above. It applies the orthogonal output, Q, to a matrix C in a similar manner to ormqr in LAPACK.

   The n, A, lda, tau, R_blocked, tau_R_blocked, n_blocks, block_size and final_block_size used by da_qr
   should be passed, unchanged, to this routine. They will be overwritten by the routine.

   Recall that, for the blocked QR factorization, the m x n matrix A, is split vertically up into n_blocks blocks of
   size block_size x n, except for the final block which may be larger:

        /A1\     /Q1 R1\    [Q1 Q2 ... Qk] * /R1\      [Q1 Q2 ... Qk] * Q_R * R
   A = | A2 | = | Q2 R2 | =                 | R2 | =
       | .. |   |  ...  |                   | .. |
        \Ak/     \Qk Rk/                     \Rk/

    where Q is of size (n * n_blocks) x n and the Q_i are of size block_size x n.

    Given an n x r matrix, C, this function computes Q, then the (n * n_blocks) x r matrix Q_R*C. It then splits Q_R*C
    into vertical blocks and for each block multiplies by Q_i.

    Note that C is overwritten with [Q1 Q2 ... Qk] * Q_R * C, which is m x r, thus it is important that the supplied C
    is large enough to contain this larger matrix. No argument checking takes place in this function.

 */

template <typename T>
da_status da_qr_apply(da_int n, std::vector<T> &A, da_int lda, std::vector<T> &tau,
                      std::vector<T> &R_blocked, std::vector<T> &tau_R_blocked,
                      da_int n_blocks, da_int block_size, da_int final_block_size,
                      da_int r, std::vector<T> &C, da_int ldc,
                      da_errors::da_error_t *err) {

    // Find out how many threads we have available
    da_int n_threads = (da_int)omp_get_max_threads();

    if (omp_get_max_active_levels() == omp_get_level())
        n_threads = 1;

    n_threads = std::min(n_threads, n_blocks);

    // Workspace queries
    da_int lwork = -1, lwork_R = -1, info = 0;
    T dummy[1], dummy_R[1];
    std::vector<T> work, work_R, Q_RxC;
    da_int mr = n_blocks * n;
    da_int k = std::min(final_block_size, n);
    da_int max_block_size = std::max(final_block_size, block_size);

    da::orgqr(&max_block_size, &k, &k, A.data(), &lda, tau.data(), dummy, &lwork, &info);
    if (info != 0) {
        return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                        "An internal error occurred in orgqr. Please check "
                        "the input data for undefined values.");
    }
    lwork = (da_int)dummy[0];
    try {
        work.resize(n_threads * lwork);
        Q_RxC.resize(mr * r, 0.0);
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    if (n_blocks > 1) {
        // For the QR decomposition of the mr x n matrix R_blocked
        da::orgqr(&mr, &n, &n, R_blocked.data(), &mr, tau_R_blocked.data(), dummy_R,
                  &lwork_R, &info);
        if (info != 0) {
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "An internal error occurred. Please check "
                            "the input data for undefined values.");
        }
        lwork_R = (da_int)dummy_R[0];
        try {
            work_R.resize(lwork_R);
        } catch (std::bad_alloc const &) {
            return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }
    }

    // Now apply the Q factors to C, starting with the Q_R if it was computed
    if (n_blocks > 1) {
        da::orgqr(&mr, &n, &n, R_blocked.data(), &mr, tau_R_blocked.data(), work_R.data(),
                  &lwork_R, &info);
        if (info != 0) {
            return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                            "An internal error occurred. Please check "
                            "the input data for undefined values.");
        }

        da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, mr, r, n, 1.0,
                            R_blocked.data(), mr, C.data(), ldc, 0.0, Q_RxC.data(), mr);
    } else {
        for (da_int j = 0; j < r; j++) {
            for (da_int i = 0; i < n; i++) {
                Q_RxC[j * n + i] = C[j * ldc + i];
            }
        }
    }

    da_int info_local = 0;
    da_int this_block_size = block_size;

#pragma omp parallel for num_threads(n_threads) default(none)                            \
    firstprivate(info_local, this_block_size)                                            \
    shared(n_blocks, final_block_size, block_size, n, r, A, lda, tau, work, lwork,       \
               Q_RxC, C, ldc, k, mr, info)
    for (da_int i = 0; i < n_blocks; i++) {
        if (info != 0)
            continue;
        if (i == n_blocks - 1)
            this_block_size = final_block_size;
        da_int this_thread = (da_int)omp_get_thread_num();
        da::orgqr(&this_block_size, &k, &k, &A[i * block_size], &lda, &tau[i * n],
                  &work[this_thread * lwork], &lwork, &info_local);
        if (info_local != 0) {
#pragma omp atomic
            info += info_local;
        } else {
            da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                this_block_size, r, k, 1.0, &A[i * block_size], lda,
                                &Q_RxC[i * n], mr, 0.0, &C[i * block_size], ldc);
        }
    }

    if (info != 0) {
        return da_error(err, da_status_internal_error, // LCOV_EXCL_LINE
                        "An internal error occurred. Please check "
                        "the input data for undefined values.");
    }

    return da_status_success;
}

#endif
