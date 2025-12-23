/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclda.h"
#include "da_cblas.hh"
#include "kernels.hpp"
#include "macros.h"
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <memory>

#define a_matrix(i, j) A[(j) * lda + (i)]         // map a_matrix( i,j ) to array A
#define b_matrix(i, j) B[(j) * ldb + (i)]         // map b_matrix( i,j ) to array B
#define c_matrix(i, j) C[(j) * ldc + (i)]         // map c_matrix( i,j ) to array C
#define ctemp_matrix(i, j) C_temp[(j) * MR + (i)] // map ctemp_matrix( i,j ) to array C

template <typename T> struct BlockSizes {
#if defined(__AVX512F__)
    static constexpr da_int MR = std::is_same<T, float>::value ? 16 : 8;
    static constexpr da_int NR = std::is_same<T, float>::value ? 8 : 8;
    static constexpr da_int MC = std::is_same<T, float>::value ? 512 : 256;
    static constexpr da_int NC = std::is_same<T, float>::value ? 4096 : 2048;
    static constexpr da_int KC = std::is_same<T, float>::value ? 64 : 32;
#elif defined(__AVX2__)
    static constexpr da_int MR = std::is_same<T, float>::value ? 8 : 4;
    static constexpr da_int NR = std::is_same<T, float>::value ? 8 : 4;
    static constexpr da_int MC = std::is_same<T, float>::value ? 512 : 256;
    static constexpr da_int NC = std::is_same<T, float>::value ? 4096 : 2048;
    static constexpr da_int KC = std::is_same<T, float>::value ? 64 : 32;
#endif
};

#ifdef _WIN32
#define aligned_malloc(ptr, size, alignment)                                             \
    (ptr) = static_cast<T *>(_aligned_malloc(size, alignment))
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(ptr, size, alignment)                                             \
    posix_memalign(reinterpret_cast<void **>(&(ptr)), (alignment), (size))
#define aligned_free(ptr) free(ptr)
#endif

namespace ARCH {
namespace da_metrics {
namespace pairwise_distances {

// Pack a micro-panel into buffer pointed to by Xtilde.
// This is an unoptimized implementation for general BLOCK and KC.
template <typename T, da_int BLOCK>
inline void PackMicroPanel_BLOCKxKC(da_int m, da_int k, const T *X, da_int ldx,
                                    [[maybe_unused]] T *Xtilde) {
    // March through A in column-major order, packing into Atilde as we go.
    if (m == BLOCK) {
        // Full row size micro-panel.
        for (da_int p = 0; p < k; p++)
            for (da_int i = 0; i < BLOCK; i++)
                *Xtilde++ = X[i + p * ldx];
    } else {
        /* Not a full row size micro-panel.  We pad with zeroes.  To be  added */
        /* Full row size micro-panel.*/
        for (da_int p = 0; p < k; p++)
            for (da_int i = 0; i < BLOCK; i++) {
                if (i < m)
                    *Xtilde++ = X[i + p * ldx];
                else
                    *Xtilde++ = T(0);
            }
    }
}

// Pack a m x k block of X into a BLOCK x KC buffer.
// The block is packed into Xtilde a micro-panel at a time.
// If necessary, the last micro-panel is padded with rows of zeroes.
template <typename T, da_int BLOCK>
void PackBlock_BLOCKxKC(da_int m, da_int k, const T *X, da_int ldx, T *Xtilde) {
    for (da_int i = 0; i < m; i += BLOCK) {
        da_int ib = (std::min)(BLOCK, m - i);
        PackMicroPanel_BLOCKxKC<T, BLOCK>(ib, k, &X[i], ldx, Xtilde);
        Xtilde += ib * k;
    }
}

template <typename T, da_int MR, da_int NR, da_int MC, da_int NC, da_int KC,
          typename Kernel_Operator>
inline void LoopFive_packed(da_int m, da_int n, da_int k, const T *A, da_int lda,
                            const T *B, da_int ldb, T *C, da_int ldc,
                            Kernel_Operator kernelOp);
template <typename T, da_int MR, da_int NR, da_int MC, da_int NC, da_int KC,
          typename Kernel_Operator, typename Post_Operator>
inline void LoopFive_packed(da_int m, da_int n, da_int k, const T *A, da_int lda,
                            const T *B, da_int ldb, T *C, da_int ldc,
                            Kernel_Operator kernelOp, Post_Operator postOp);
template <typename T, da_int MR, da_int NR, da_int MC, da_int NC, da_int KC,
          typename Kernel_Operator>
inline void LoopFour_packed(da_int m, da_int n, da_int k, const T *A, da_int lda,
                            const T *B, da_int ldb, T *C, da_int ldc,
                            Kernel_Operator kernelOp);
template <typename T, da_int MR, da_int NR, da_int MC, da_int NC, da_int KC,
          typename Kernel_Operator>
inline void LoopThree_packed(da_int m, da_int n, da_int k, const T *A, da_int lda,
                             const T *Btilde, T *C, da_int ldc, T *C_temp,
                             Kernel_Operator kernelOp);
template <typename T, da_int MR, da_int NR, da_int MC, da_int NC, da_int KC,
          typename Kernel_Operator>
inline void LoopTwo_packed(da_int m, da_int n, da_int k, const T *Atilde, const T *Btilde,
                           T *C, da_int ldc, T *C_temp, Kernel_Operator kernelOp);
template <typename T, da_int MR, da_int NR, da_int MC, da_int NC, da_int KC,
          typename Kernel_Operator>
inline void LoopOne_packed(da_int m, da_int n, da_int k, const T *Atilde,
                           const T *MicroPanelB, T *C, da_int ldc, T *C_temp,
                           Kernel_Operator kernelOp);

template <typename T, da_int MR, da_int NR, da_int MC, da_int NC, da_int KC,
          typename Kernel_Operator>
inline void LoopFive_packed(da_int m, da_int n, da_int k, const T *A, da_int lda,
                            const T *B, da_int ldb, T *C, da_int ldc,
                            Kernel_Operator kernelOp) {
    static_assert(MC % MR == 0, "MC must be a multiple of MR");
    static_assert(NC % NR == 0, "NC must be a multiple of NR");
    // Zero out C
    for (da_int j = 0; j < n; j++)
        da_std::fill(C + j * ldc, C + j * ldc + m, 0.0);

    for (da_int j = 0; j < n; j += NC) {
        da_int jb = (std::min)(NC, n - j); // Last loop may not involve a full block
        LoopFour_packed<T, MR, NR, MC, NC, KC, Kernel_Operator>(
            m, jb, k, A, lda, &b_matrix(j, 0), ldb, &c_matrix(0, j), ldc, kernelOp);
    }
}

template <typename T, da_int MR, da_int NR, da_int MC, da_int NC, da_int KC,
          typename Kernel_Operator, typename Post_Operator>
inline void LoopFive_packed(da_int m, da_int n, da_int k, const T *A, da_int lda,
                            const T *B, da_int ldb, T *C, da_int ldc,
                            Kernel_Operator kernelOp, Post_Operator postOp) {
    static_assert(MC % MR == 0, "MC must be a multiple of MR");
    static_assert(NC % NR == 0, "NC must be a multiple of NR");
    // Zero out C
    for (da_int j = 0; j < n; j++)
        da_std::fill(C + j * ldc, C + j * ldc + m, 0.0);

    for (da_int j = 0; j < n; j += NC) {
        da_int jb = (std::min)(NC, n - j); // Last loop may not involve a full block
        LoopFour_packed<T, MR, NR, MC, NC, KC, Kernel_Operator>(
            m, jb, k, A, lda, &b_matrix(j, 0), ldb, &c_matrix(0, j), ldc, kernelOp);
        postOp(m, jb, &c_matrix(0, j), ldc);
    }
}

template <typename T, da_int MR, da_int NR, da_int MC, da_int NC, da_int KC,
          typename Kernel_Operator>
inline void LoopFour_packed(da_int m, da_int n, da_int k, const T *A, da_int lda,
                            const T *B, da_int ldb, T *C, da_int ldc,
                            Kernel_Operator kernelOp) {
    T C_temp[MR * NR] __attribute__((aligned(64)));

    // Use heap allocation for large arrays with proper alignment
    constexpr size_t btilde_size = NC * KC;
    T *Btilde;

#if defined(_WIN32)
    // Use heap allocation only for Windows builds to avoid stack overflow
    if constexpr (btilde_size > 4096) { // Threshold for stack vs heap
        aligned_malloc(Btilde, btilde_size * sizeof(T), 64);
    } else {
        T Btilde_stack[btilde_size] __attribute__((aligned(64)));
        Btilde = Btilde_stack;
    }
#else
    // Use stack allocation for all other platforms/configurations
    T Btilde_stack[btilde_size] __attribute__((aligned(64)));
    Btilde = Btilde_stack;
#endif

    for (da_int p = 0; p < k; p += KC) {
        da_int pb = (std::min)(KC, k - p); // Last loop may not involve a full block
        PackBlock_BLOCKxKC<T, NR>(n, pb, &b_matrix(0, p), ldb, Btilde);
        LoopThree_packed<T, MR, NR, MC, NC, KC, Kernel_Operator>(
            m, n, pb, &a_matrix(0, p), lda, Btilde, C, ldc, C_temp, kernelOp);
    }

#if defined(_WIN32)
    // Clean up heap allocation only for Windows builds
    if constexpr (btilde_size > 4096) {
        aligned_free(Btilde);
    }
#endif
}

template <typename T, da_int MR, da_int NR, da_int MC, da_int NC, da_int KC,
          typename Kernel_Operator>
inline void LoopThree_packed(da_int m, da_int n, da_int k, const T *A, da_int lda,
                             const T *Btilde, T *C, da_int ldc, T *C_temp,
                             Kernel_Operator kernelOp) {
    // Use heap allocation for large arrays
    constexpr size_t atilde_size = MC * KC;
    T *Atilde;

#if defined(_WIN32)
    // Use heap allocation only for Windows builds to avoid stack overflow
    if constexpr (atilde_size > 4096) { // Threshold for stack vs heap
        aligned_malloc(Atilde, atilde_size * sizeof(T), 64);
    } else {
        T Atilde_stack[atilde_size] __attribute__((aligned(64)));
        Atilde = Atilde_stack;
    }
#else
    // Use stack allocation for all other platforms/configurations
    T Atilde_stack[atilde_size] __attribute__((aligned(64)));
    Atilde = Atilde_stack;
#endif

    for (da_int i = 0; i < m; i += MC) {
        da_int ib = (std::min)(MC, m - i); // Last loop may not involve a full block
        PackBlock_BLOCKxKC<T, MR>(ib, k, &a_matrix(i, 0), lda, Atilde);
        LoopTwo_packed<T, MR, NR, MC, NC, KC, Kernel_Operator>(
            ib, n, k, Atilde, Btilde, &c_matrix(i, 0), ldc, C_temp, kernelOp);
    }

#if defined(_WIN32)
    // Clean up heap allocation only for Windows builds
    if constexpr (atilde_size > 4096) {
        aligned_free(Atilde);
    }
#endif
}

template <typename T, da_int MR, da_int NR, da_int MC, da_int NC, da_int KC,
          typename Kernel_Operator>
inline void LoopTwo_packed(da_int m, da_int n, da_int k, const T *Atilde, const T *Btilde,
                           T *C, da_int ldc, T *C_temp, Kernel_Operator kernelOp) {
    for (da_int j = 0; j < n; j += NR) {
        da_int jb = (std::min)(NR, n - j);
        LoopOne_packed<T, MR, NR, MC, NC, KC, Kernel_Operator>(
            m, jb, k, Atilde, &Btilde[j * k], &c_matrix(0, j), ldc, C_temp, kernelOp);
    }
}

template <typename T, da_int MR, da_int NR, da_int MC, da_int NC, da_int KC,
          typename Kernel_Operator>
inline void LoopOne_packed(da_int m, da_int n, da_int k, const T *Atilde,
                           const T *MicroPanelB, T *C, da_int ldc, T *C_temp,
                           Kernel_Operator kernelOp) {
    for (da_int i = 0; i < m; i += MR) {
        da_int ib = (std::min)(MR, m - i);
        kernelOp(ib, n, k, &Atilde[i * k], MicroPanelB, &c_matrix(i, 0), ldc, C_temp);
    }
}
} // namespace pairwise_distances
} // namespace da_metrics
} // namespace ARCH