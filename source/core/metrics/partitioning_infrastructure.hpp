/*
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclda_types.h"
#include "da_omp.hpp"
#include "da_utils.hpp"
#include "macros.h"
#include "metrics_kernels.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <memory>
#include <vector>

#define a_matrix(i, j) A[(j) * lda + (i)] // map a_matrix( i,j ) to array A
#define b_matrix(i, j) B[(j) * ldb + (i)] // map b_matrix( i,j ) to array B
#define c_matrix(i, j) C[(j) * ldc + (i)] // map c_matrix( i,j ) to array C

template <typename T> struct BlockSizes {
#if defined(__AVX512F__)
    static constexpr da_int MR = std::is_same<T, float>::value ? 16 : 8;
    static constexpr da_int NR = std::is_same<T, float>::value ? 8 : 8;
    static constexpr da_int MC = std::is_same<T, float>::value ? 256 : 128;
    static constexpr da_int NC = std::is_same<T, float>::value ? 1024 : 1024;
    static constexpr da_int KC = std::is_same<T, float>::value ? 256 : 512;
#elif defined(__AVX2__)
    static constexpr da_int MR = std::is_same<T, float>::value ? 8 : 4;
    static constexpr da_int NR = std::is_same<T, float>::value ? 8 : 4;
    static constexpr da_int MC = std::is_same<T, float>::value ? 512 : 256;
    static constexpr da_int NC = std::is_same<T, float>::value ? 1024 : 1024;
    static constexpr da_int KC = std::is_same<T, float>::value ? 64 : 32;
#endif
};

#ifdef _WIN32
#define aligned_malloc(ptr, size, alignment)                                             \
    (ptr) = static_cast<T *>(_aligned_malloc(size, alignment))
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(ptr, size, alignment)                                             \
    do {                                                                                 \
        if (posix_memalign(reinterpret_cast<void **>(&(ptr)), (alignment), (size)))      \
            (ptr) = nullptr;                                                             \
    } while (0)
#define aligned_free(ptr) free(ptr)
#endif

namespace ARCH {
namespace da_metrics {
namespace pairwise_distances {

// Pack a micro-panel into buffer pointed to by Xtilde.
template <typename T, da_int BLOCK>
inline void PackMicroPanel_BLOCKxKC(da_int m, da_int k, const T *X, da_int ldx,
                                    T *Xtilde) {
    if (m == BLOCK) {
        // Full micro-panel: straight copy, column by column.
        for (da_int p = 0; p < k; p++) {
            std::memcpy(Xtilde, X + p * ldx, BLOCK * sizeof(T));
            Xtilde += BLOCK;
        }
    } else {
        // Partial micro-panel: copy valid rows, zero-pad the rest.
        da_int pad = BLOCK - m;
        for (da_int p = 0; p < k; p++) {
            std::memcpy(Xtilde, X + p * ldx, m * sizeof(T));
            std::memset(Xtilde + m, 0, pad * sizeof(T));
            Xtilde += BLOCK;
        }
    }
}

// Pack an m x k block of X into a BLOCK x KC buffer.
// The block is packed into Xtilde a micro-panel at a time.
// If necessary, the last micro-panel is padded with rows of zeroes.
template <typename T, da_int BLOCK>
void PackBlock_BLOCKxKC(da_int m, da_int k, const T *X, da_int ldx, T *Xtilde) {
    for (da_int i = 0; i < m; i += BLOCK) {
        da_int ib = (std::min)(BLOCK, m - i);
        PackMicroPanel_BLOCKxKC<T, BLOCK>(ib, k, &X[i], ldx, Xtilde);
        Xtilde += BLOCK * k;
    }
}

template <typename T, da_int MR, da_int NR, da_int MC, da_int NC, da_int KC,
          typename Kernel_Operator, bool is_first_slice>
inline void LoopOne_packed(da_int m, da_int n, da_int k, const T *Atilde,
                           const T *MicroPanelB, T *C, da_int ldc,
                           Kernel_Operator kernelOp) {
    for (da_int i = 0; i < m; i += MR) {
        da_int ib = (std::min)(MR, m - i);
        // Prefetch C output addresses for the next micro-panel
        // Use __MM_HINT_T0 since it will be written during kernelOp()
        // This way we prefetch the next block into L1
        if (i + MR < m) {
            for (da_int j = 0; j < n; j++)
                _mm_prefetch((const char *)(&c_matrix(i + MR, j)), _MM_HINT_T0);
        }
        kernelOp.template operator()<is_first_slice>(ib, n, k, &Atilde[i * k],
                                                     MicroPanelB, &c_matrix(i, 0), ldc);
    }
}

template <typename T, da_int MR, da_int NR, da_int MC, da_int NC, da_int KC,
          typename Kernel_Operator, bool is_first_slice>
inline void LoopTwo_packed(da_int m, da_int n, da_int k, const T *Atilde, const T *Btilde,
                           T *C, da_int ldc, Kernel_Operator kernelOp) {
    for (da_int j = 0; j < n; j += NR) {
        da_int jb = (std::min)(NR, n - j);
        // Prefetch next Btilde micro-panel into L2 (__MM_HINT_T1)
        // Since the micro-panel is of NRxKC size and is packed contiguously, adjacent elements
        // share cache lines so to prefetch the entire micro-panel we need 64/sizeof(T)
        // elements (64 bytes-one cache line)
        if (j + NR < n) {
            const T *next_Bpanel = &Btilde[(j + NR) * k];
            for (da_int p = 0; p < NR * k; p += 64 / sizeof(T))
                _mm_prefetch((const char *)(next_Bpanel + p), _MM_HINT_T1);
        }
        LoopOne_packed<T, MR, NR, MC, NC, KC, Kernel_Operator, is_first_slice>(
            m, jb, k, Atilde, &Btilde[j * k], &c_matrix(0, j), ldc, kernelOp);
    }
}

template <typename T, da_int MR, da_int NR, da_int MC, da_int NC, da_int KC,
          typename Kernel_Operator, bool is_first_slice>
inline void LoopThree_packed(da_int m, da_int n, da_int k, const T *A, da_int lda,
                             const T *Btilde, T *C, da_int ldc, Kernel_Operator kernelOp,
                             T *Atilde_all, [[maybe_unused]] da_int n_threads) {
    constexpr size_t atilde_size = MC * KC;

    // Parallelize the MC loop: each thread gets its own Atilde slice to pack into.
    // Btilde is read-only (packed once in LoopFour) so sharing is safe.
    // Each thread writes to its own disjoint MC-row block of C, so no write conflicts.

#pragma omp parallel for schedule(static) num_threads(n_threads) default(none)           \
    shared(m, n, k, A, lda, Btilde, C, ldc, kernelOp, Atilde_all, atilde_size)
    for (da_int i = 0; i < m; i += MC) {
        da_int ib = (std::min)(MC, m - i); // Last loop may not involve a full block
        // Atilde is the thread-specific slice of packed A
        T *Atilde = Atilde_all + omp_get_thread_num() * atilde_size;
        PackBlock_BLOCKxKC<T, MR>(ib, k, &a_matrix(i, 0), lda, Atilde);
        LoopTwo_packed<T, MR, NR, MC, NC, KC, Kernel_Operator, is_first_slice>(
            ib, n, k, Atilde, Btilde, &c_matrix(i, 0), ldc, kernelOp);
    }
}

template <typename T, da_int MR, da_int NR, da_int MC, da_int NC, da_int KC,
          typename Kernel_Operator>
inline void LoopFour_packed(da_int m, da_int n, da_int k, const T *A, da_int lda,
                            const T *B, da_int ldb, T *C, da_int ldc,
                            Kernel_Operator kernelOp, T *Btilde, T *Atilde_all,
                            da_int n_threads) {
    // First KC slice: zero-initialize C (is_first_slice = true)
    da_int pb = (std::min)(KC, k);
    PackBlock_BLOCKxKC<T, NR>(n, pb, &b_matrix(0, 0), ldb, Btilde);
    LoopThree_packed<T, MR, NR, MC, NC, KC, Kernel_Operator, true>(
        m, n, pb, &a_matrix(0, 0), lda, Btilde, C, ldc, kernelOp, Atilde_all, n_threads);
    // Remaining KC slices: accumulate into C (is_first_slice = false)
    for (da_int p = KC; p < k; p += KC) {
        pb = (std::min)(KC, k - p);
        PackBlock_BLOCKxKC<T, NR>(n, pb, &b_matrix(0, p), ldb, Btilde);
        LoopThree_packed<T, MR, NR, MC, NC, KC, Kernel_Operator, false>(
            m, n, pb, &a_matrix(0, p), lda, Btilde, C, ldc, kernelOp, Atilde_all,
            n_threads);
    }
}

template <typename T, da_int MR, da_int NR, da_int MC, da_int NC, da_int KC,
          typename Kernel_Operator, typename Post_Operator>
inline da_status LoopFive_packed(da_int m, da_int n, da_int k, const T *A, da_int lda,
                                 const T *B, da_int ldb, T *C, da_int ldc,
                                 Kernel_Operator kernelOp, Post_Operator postOp) {
    static_assert(MC % MR == 0, "MC must be a multiple of MR");
    static_assert(NC % NR == 0, "NC must be a multiple of NR");

    // See comments in the other LoopFive_packed overload for buffer design.
    constexpr size_t btilde_size = NC * KC;
    T *Btilde;
    aligned_malloc(Btilde, btilde_size * sizeof(T), 64);
    if (!Btilde)
        return da_status_memory_error;

    // compute how many MC-row blocks fit in
    da_int mc_block_size =
        std::min(MC, m); // actual block size (last block may be smaller)
    da_int n_mc_blocks = 0, mc_block_rem = 0;
    da_utils::blocking_scheme(m, mc_block_size, n_mc_blocks, mc_block_rem);
    da_int n_threads = da_utils::get_n_threads_loop(n_mc_blocks);
    // Atilde (A packing buffer): one copy per thread. Each thread in LoopThree packs
    // its own MC×KC block into its private Atilde slice, so n_threads copies are needed.
    // get_n_threads_loop returns 1 when called from inside an existing OMP region
    // (e.g. KNN predict, radius_neighbors) and omp_get_max_threads when called at
    // top level (e.g. public pairwise_distances API), so the allocation adapts
    // automatically.
    constexpr size_t atilde_size = MC * KC;
    T *Atilde_all;
    aligned_malloc(Atilde_all, n_threads * atilde_size * sizeof(T), 64);
    if (!Atilde_all) {
        aligned_free(Btilde);
        return da_status_memory_error;
    }

    for (da_int j = 0; j < n; j += NC) {
        da_int jb = (std::min)(NC, n - j); // Last loop may not involve a full block
        LoopFour_packed<T, MR, NR, MC, NC, KC, Kernel_Operator>(
            m, jb, k, A, lda, &b_matrix(j, 0), ldb, &c_matrix(0, j), ldc, kernelOp,
            Btilde, Atilde_all, n_threads);
        postOp(m, jb, &c_matrix(0, j), ldc);
    }

    aligned_free(Atilde_all);
    aligned_free(Btilde);
    return da_status_success;
}

template <typename T, da_int MR, da_int NR, da_int MC, da_int NC, da_int KC,
          typename Kernel_Operator>
inline da_status LoopFive_packed(da_int m, da_int n, da_int k, const T *A, da_int lda,
                                 const T *B, da_int ldb, T *C, da_int ldc,
                                 Kernel_Operator kernelOp) {
    static_assert(MC % MR == 0, "MC must be a multiple of MR");
    static_assert(NC % NR == 0, "NC must be a multiple of NR");

    // Btilde (B packing buffer): single copy, packed once per KC slice in LoopFour,
    // then shared read-only across all threads in LoopThree.
    constexpr size_t btilde_size = NC * KC;
    T *Btilde;
    aligned_malloc(Btilde, btilde_size * sizeof(T), 64);
    if (!Btilde)
        return da_status_memory_error;

    // compute how many MC-row blocks fit in
    da_int mc_block_size =
        std::min(MC, m); // actual block size (last block may be smaller)
    da_int n_mc_blocks = 0, mc_block_rem = 0;
    da_utils::blocking_scheme(m, mc_block_size, n_mc_blocks, mc_block_rem);
    da_int n_threads = da_utils::get_n_threads_loop(n_mc_blocks);
    // Atilde (A packing buffer): one copy per thread. Each thread in LoopThree packs
    // its own MC×KC block into its private Atilde slice, so n_threads copies are needed.
    // get_n_threads_loop returns 1 when called from inside an existing OMP region
    // (e.g. KNN predict, radius_neighbors) and omp_get_max_threads when called at
    // top level (e.g. public pairwise_distances API), so the allocation adapts
    // automatically.
    constexpr size_t atilde_size = MC * KC;
    T *Atilde_all;
    aligned_malloc(Atilde_all, n_threads * atilde_size * sizeof(T), 64);
    if (!Atilde_all) {
        aligned_free(Btilde);
        return da_status_memory_error;
    }

    for (da_int j = 0; j < n; j += NC) {
        da_int jb = (std::min)(NC, n - j); // Last loop may not involve a full block
        LoopFour_packed<T, MR, NR, MC, NC, KC, Kernel_Operator>(
            m, jb, k, A, lda, &b_matrix(j, 0), ldb, &c_matrix(0, j), ldc, kernelOp,
            Btilde, Atilde_all, n_threads);
    }

    aligned_free(Atilde_all);
    aligned_free(Btilde);
    return da_status_success;
}

// Shared dispatch for packed distance computation (without post-op).
// Handles Y==nullptr (X_is_Y), row-major transpose, and delegates to LoopFive_packed.
template <typename T, typename Kernel_Operator>
inline da_status dispatch_packed_distance(da_order order, da_int m, da_int n, da_int k,
                                          const T *X, da_int ldx, const T *Y, da_int ldy,
                                          T *D, da_int ldd, Kernel_Operator kernelOp) {
    constexpr da_int MR = BlockSizes<T>::MR;
    constexpr da_int NR = BlockSizes<T>::NR;
    constexpr da_int MC = BlockSizes<T>::MC;
    constexpr da_int NC = BlockSizes<T>::NC;
    constexpr da_int KC = BlockSizes<T>::KC;

    const T *Y_new = Y;
    if (!Y) {
        n = m;
        ldy = ldx;
        Y_new = X;
    }

    da_status status;
    if (order == row_major) {
        std::vector<T> X_col, Y_col, D_col;
        try {
            X_col.resize(m * k);
            Y_col.resize(n * k);
            D_col.resize(m * n);
        } catch (std::bad_alloc const &) {
            return da_status_memory_error;
        }
        da_utils::copy_transpose_2D_array_row_to_column_major(m, k, X, ldx, X_col.data(),
                                                              m);
        da_utils::copy_transpose_2D_array_row_to_column_major(n, k, Y_new, ldy,
                                                              Y_col.data(), n);
        status = LoopFive_packed<T, MR, NR, MC, NC, KC>(
            m, n, k, X_col.data(), m, Y_col.data(), n, D_col.data(), m, kernelOp);
        if (status != da_status_success)
            return status;
        da_utils::copy_transpose_2D_array_column_to_row_major(m, n, D_col.data(), m, D,
                                                              ldd);
    } else {
        status = LoopFive_packed<T, MR, NR, MC, NC, KC>(m, n, k, X, ldx, Y_new, ldy, D,
                                                        ldd, kernelOp);
        if (status != da_status_success)
            return status;
    }

    return da_status_success;
}

// Shared dispatch for packed distance computation (with post-op).
// Handles Y==nullptr (X_is_Y), row-major transpose, and delegates to LoopFive_packed.
template <typename T, typename Kernel_Operator, typename Post_Operator>
inline da_status dispatch_packed_distance(da_order order, da_int m, da_int n, da_int k,
                                          const T *X, da_int ldx, const T *Y, da_int ldy,
                                          T *D, da_int ldd, Kernel_Operator kernelOp,
                                          Post_Operator postOp) {
    constexpr da_int MR = BlockSizes<T>::MR;
    constexpr da_int NR = BlockSizes<T>::NR;
    constexpr da_int MC = BlockSizes<T>::MC;
    constexpr da_int NC = BlockSizes<T>::NC;
    constexpr da_int KC = BlockSizes<T>::KC;

    const T *Y_new = Y;
    if (!Y) {
        n = m;
        ldy = ldx;
        Y_new = X;
    }

    da_status status;
    if (order == row_major) {
        std::vector<T> X_col, Y_col, D_col;
        try {
            X_col.resize(m * k);
            Y_col.resize(n * k);
            D_col.resize(m * n);
        } catch (std::bad_alloc const &) {
            return da_status_memory_error;
        }
        da_utils::copy_transpose_2D_array_row_to_column_major(m, k, X, ldx, X_col.data(),
                                                              m);
        da_utils::copy_transpose_2D_array_row_to_column_major(n, k, Y_new, ldy,
                                                              Y_col.data(), n);
        status = LoopFive_packed<T, MR, NR, MC, NC, KC>(
            m, n, k, X_col.data(), m, Y_col.data(), n, D_col.data(), m, kernelOp, postOp);
        if (status != da_status_success)
            return status;
        da_utils::copy_transpose_2D_array_column_to_row_major(m, n, D_col.data(), m, D,
                                                              ldd);
    } else {
        status = LoopFive_packed<T, MR, NR, MC, NC, KC>(m, n, k, X, ldx, Y_new, ldy, D,
                                                        ldd, kernelOp, postOp);
        if (status != da_status_success)
            return status;
    }

    return da_status_success;
}

} // namespace pairwise_distances
} // namespace da_metrics
} // namespace ARCH

#undef a_matrix
#undef b_matrix
#undef c_matrix