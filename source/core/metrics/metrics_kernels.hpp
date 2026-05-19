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
#ifndef METRICS_KERNELS_HPP
#define METRICS_KERNELS_HPP

#include "aoclda.h"
#include "kt.hpp"
#include "macros.h"
#include <cmath>
#include <immintrin.h>
#include <type_traits>

#define a_matrix(i, j) A[(j) * lda + (i)]         // map a_matrix( i,j ) to array A
#define b_matrix(i, j) B[(j) * ldb + (i)]         // map b_matrix( i,j ) to array B
#define c_matrix(i, j) C[(j) * ldc + (i)]         // map c_matrix( i,j ) to array C
#define ctemp_matrix(i, j) C_temp[(j) * MR + (i)] // map ctemp_matrix( i,j ) to array C

namespace ARCH {

using namespace kernel_templates;

// Helper for absolute value operations - templated on bit size
template <bsz BSZ, typename T>
inline __attribute__((__always_inline__)) avxvector_t<BSZ, T>
simd_abs(avxvector_t<BSZ, T> x) {
#ifdef __AVX512F__
    if constexpr (BSZ == bsz::b512) {
        if constexpr (std::is_same_v<T, float>) {
            return _mm512_abs_ps(x);
        } else {
            return _mm512_abs_pd(x);
        }
    } else
#endif
        if constexpr (BSZ == bsz::b256) {
        if constexpr (std::is_same_v<T, float>) {
            const __m256 sign_mask = kt_set1_p<BSZ, T>(-0.0f);
            return _mm256_andnot_ps(sign_mask, x);
        } else {
            const __m256d sign_mask = kt_set1_p<BSZ, T>(-0.0);
            return _mm256_andnot_pd(sign_mask, x);
        }
    } else {
        static_assert(BSZ == bsz::b256 || BSZ == bsz::b512, "Unsupported bit size");
    }
}

// Template for K-way unrolling with compile-time recursion
// BSZ is the bit size (bsz::b256 or bsz::b512)
// T is the data type (float or double)
// K is the number of columns to unroll
// CurrentK is the current column being processed in the unrolling
template <bsz BSZ, typename T, da_int K, da_int CurrentK = 0> struct k_unroll_packed {
    // Set beta values for K rows
    static inline __attribute__((__always_inline__)) void
    set_betas(avxvector_t<BSZ, T> *betas, const T *Btilde, da_int idx, da_int NR) {
        betas[CurrentK] = kt_set1_p<BSZ, T>(Btilde[idx + CurrentK * NR]);
        k_unroll_packed<BSZ, T, K, CurrentK + 1>::set_betas(betas, Btilde, idx, NR);
    }

    // Compute differences for K rows
    static inline __attribute__((__always_inline__)) void
    compute_diffs(avxvector_t<BSZ, T> *temps, const avxvector_t<BSZ, T> *alphas,
                  const avxvector_t<BSZ, T> *betas) {
        temps[CurrentK] = kt_sub_p<BSZ, T>(alphas[CurrentK], betas[CurrentK]);
        k_unroll_packed<BSZ, T, K, CurrentK + 1>::compute_diffs(temps, alphas, betas);
    }

    // Update gamma with FMA for K rows (squared Euclidean)
    static inline __attribute__((__always_inline__)) void
    sqeuclidean_update_gamma(avxvector_t<BSZ, T> &gamma,
                             const avxvector_t<BSZ, T> *temps) {
        gamma = kt_fmadd_p<BSZ, T>(temps[CurrentK], temps[CurrentK], gamma);
        k_unroll_packed<BSZ, T, K, CurrentK + 1>::sqeuclidean_update_gamma(gamma, temps);
    }

    // Update gamma for Manhattan distance
    static inline __attribute__((__always_inline__)) void
    manhattan_update_gamma(avxvector_t<BSZ, T> &gamma, const avxvector_t<BSZ, T> *temps) {
        gamma = kt_add_p<BSZ, T>(gamma, simd_abs<BSZ, T>(temps[CurrentK]));
        k_unroll_packed<BSZ, T, K, CurrentK + 1>::manhattan_update_gamma(gamma, temps);
    }

    // Update gamma for Cosine distance
    static inline __attribute__((__always_inline__)) void
    cosine_update_gamma(avxvector_t<BSZ, T> &gamma, const avxvector_t<BSZ, T> *alphas,
                        const avxvector_t<BSZ, T> *betas) {
        gamma = kt_fmadd_p<BSZ, T>(alphas[CurrentK], betas[CurrentK], gamma);
        k_unroll_packed<BSZ, T, K, CurrentK + 1>::cosine_update_gamma(gamma, alphas,
                                                                      betas);
    }
};

// Base case for k_unroll_packed recursion
template <bsz BSZ, typename T, da_int K> struct k_unroll_packed<BSZ, T, K, K> {
    static inline __attribute__((__always_inline__)) void
    set_betas([[maybe_unused]] avxvector_t<BSZ, T> *betas,
              [[maybe_unused]] const T *Btilde, [[maybe_unused]] da_int idx,
              [[maybe_unused]] da_int NR) {}

    static inline __attribute__((__always_inline__)) void
    compute_diffs([[maybe_unused]] avxvector_t<BSZ, T> *temps,
                  [[maybe_unused]] const avxvector_t<BSZ, T> *alphas,
                  [[maybe_unused]] const avxvector_t<BSZ, T> *betas) {}

    static inline __attribute__((__always_inline__)) void
    sqeuclidean_update_gamma([[maybe_unused]] avxvector_t<BSZ, T> &gamma,
                             [[maybe_unused]] const avxvector_t<BSZ, T> *temps) {}

    static inline __attribute__((__always_inline__)) void
    manhattan_update_gamma([[maybe_unused]] avxvector_t<BSZ, T> &gamma,
                           [[maybe_unused]] const avxvector_t<BSZ, T> *temps) {}

    static inline __attribute__((__always_inline__)) void
    cosine_update_gamma([[maybe_unused]] avxvector_t<BSZ, T> &gamma,
                        [[maybe_unused]] const avxvector_t<BSZ, T> *alphas,
                        [[maybe_unused]] const avxvector_t<BSZ, T> *betas) {}
};

// Combined template for both row and column unrolling
// BSZ is the bit size (bsz::b256 or bsz::b512)
// T is the data type (float or double)
// I is the current column index in the unrolling process
// N is the total number of columns to process
// K is the number of rows to unroll
template <bsz BSZ, typename T, da_int I, da_int N, da_int K>
struct template_unroll_k_packed {
    // Load gamma vectors - column unrolled
    static inline __attribute__((__always_inline__)) void
    load_gammas(avxvector_t<BSZ, T> *gammas, T *C, da_int ldc) {
        gammas[I] = kt_loadu_p<BSZ, T>(&c_matrix(0, I));
        template_unroll_k_packed<BSZ, T, I + 1, N, K>::load_gammas(gammas, C, ldc);
    }

    // Process columns with K rows of A - combined row and column unrolling
    static inline __attribute__((__always_inline__)) void
    sqeuclidean_process_k_cols(avxvector_t<BSZ, T> *gammas, avxvector_t<BSZ, T> *alphas,
                               avxvector_t<BSZ, T> *betas, avxvector_t<BSZ, T> *temps,
                               const T *Btilde, da_int NR) {
        // Set beta values for all K rows using row-wise template unrolling
        k_unroll_packed<BSZ, T, K>::set_betas(betas, Btilde, I, NR);

        // Compute differences for all K rows
        k_unroll_packed<BSZ, T, K>::compute_diffs(temps, alphas, betas);

        // Update gamma with FMA for all K rows
        k_unroll_packed<BSZ, T, K>::sqeuclidean_update_gamma(gammas[I], temps);

        // Continue to next column
        template_unroll_k_packed<BSZ, T, I + 1, N, K>::sqeuclidean_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);
    }

    // Process columns for Manhattan distance
    static inline __attribute__((__always_inline__)) void
    manhattan_process_k_cols(avxvector_t<BSZ, T> *gammas, avxvector_t<BSZ, T> *alphas,
                             avxvector_t<BSZ, T> *betas, avxvector_t<BSZ, T> *temps,
                             const T *Btilde, da_int NR) {
        // Set beta values for all K rows using row-wise template unrolling
        k_unroll_packed<BSZ, T, K>::set_betas(betas, Btilde, I, NR);

        // Compute differences for all K rows
        k_unroll_packed<BSZ, T, K>::compute_diffs(temps, alphas, betas);

        // Update gamma with absolute differences for all K rows
        k_unroll_packed<BSZ, T, K>::manhattan_update_gamma(gammas[I], temps);

        // Continue to next column
        template_unroll_k_packed<BSZ, T, I + 1, N, K>::manhattan_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);
    }

    // Process columns for Minkowski distance
    static inline __attribute__((__always_inline__)) void
    minkowski_process_k_cols_direct(const avxvector_t<BSZ, T> *alphas, const T *Btilde,
                                    T *C, da_int ldc, da_int NR, T p) {
        // Process each column directly
        constexpr da_int elem_count = tsz_v<BSZ, T>;
        for (da_int col = 0; col < N; col++) {
            // Temporary storage for calculations
            alignas(64) T values[K][elem_count];
            avxvector_t<BSZ, T> beta[K];
            avxvector_t<BSZ, T> temp[K];

            // Set beta values for all K rows
            for (da_int k = 0; k < K; k++) {
                beta[k] = kt_set1_p<BSZ, T>(Btilde[col + k * NR]);
            }

            // Calculate differences and absolute values for all K rows
            for (da_int k = 0; k < K; k++) {
                temp[k] = simd_abs<BSZ, T>(kt_sub_p<BSZ, T>(alphas[k], beta[k]));
                kt_storeu_p<BSZ, T>(values[k], temp[k]);
            }

            // Apply power and accumulate directly into C matrix
            for (da_int j = 0; j < elem_count; j++) {
                for (da_int k = 0; k < K; k++) {
                    c_matrix(j, col) += std::pow(values[k][j], p);
                }
            }
        }
    }

    // Process columns for Cosine distance
    static inline __attribute__((__always_inline__)) void
    cosine_process_k_cols(avxvector_t<BSZ, T> *gammas, avxvector_t<BSZ, T> *alphas,
                          avxvector_t<BSZ, T> *betas, const T *Btilde, da_int NR) {
        // Set beta values for all K rows using row-wise template unrolling
        k_unroll_packed<BSZ, T, K>::set_betas(betas, Btilde, I, NR);

        // Update gamma with dot products of alphas and betas for all K rows (cosine similarity)
        k_unroll_packed<BSZ, T, K>::cosine_update_gamma(gammas[I], alphas, betas);

        // Continue to next column
        template_unroll_k_packed<BSZ, T, I + 1, N, K>::cosine_process_k_cols(
            gammas, alphas, betas, Btilde, NR);
    }

    // Store gamma vectors - column unrolled
    static inline __attribute__((__always_inline__)) void
    store_gammas(avxvector_t<BSZ, T> *gammas, T *C, da_int ldc) {
        kt_storeu_p<BSZ, T>(&c_matrix(0, I), gammas[I]);
        template_unroll_k_packed<BSZ, T, I + 1, N, K>::store_gammas(gammas, C, ldc);
    }
};

// Base case for template_unroll_k_packed column recursion
template <bsz BSZ, typename T, da_int N, da_int K>
struct template_unroll_k_packed<BSZ, T, N, N, K> {
    static inline __attribute__((__always_inline__)) void
    load_gammas([[maybe_unused]] avxvector_t<BSZ, T> *gammas, [[maybe_unused]] T *C,
                [[maybe_unused]] da_int ldc) {}

    static inline __attribute__((__always_inline__)) void
    sqeuclidean_process_k_cols([[maybe_unused]] avxvector_t<BSZ, T> *gammas,
                               [[maybe_unused]] avxvector_t<BSZ, T> *alphas,
                               [[maybe_unused]] avxvector_t<BSZ, T> *betas,
                               [[maybe_unused]] avxvector_t<BSZ, T> *temps,
                               [[maybe_unused]] const T *Btilde,
                               [[maybe_unused]] da_int NR) {}

    static inline __attribute__((__always_inline__)) void
    manhattan_process_k_cols([[maybe_unused]] avxvector_t<BSZ, T> *gammas,
                             [[maybe_unused]] avxvector_t<BSZ, T> *alphas,
                             [[maybe_unused]] avxvector_t<BSZ, T> *betas,
                             [[maybe_unused]] avxvector_t<BSZ, T> *temps,
                             [[maybe_unused]] const T *Btilde,
                             [[maybe_unused]] da_int NR) {}

    static inline __attribute__((__always_inline__)) void
    minkowski_process_k_cols_direct([[maybe_unused]] const avxvector_t<BSZ, T> *alphas,
                                    [[maybe_unused]] const T *Btilde,
                                    [[maybe_unused]] T *C, [[maybe_unused]] da_int ldc,
                                    [[maybe_unused]] da_int NR, [[maybe_unused]] T p) {}

    static inline __attribute__((__always_inline__)) void
    cosine_process_k_cols([[maybe_unused]] avxvector_t<BSZ, T> *gammas,
                          [[maybe_unused]] avxvector_t<BSZ, T> *alphas,
                          [[maybe_unused]] avxvector_t<BSZ, T> *betas,
                          [[maybe_unused]] const T *Btilde, [[maybe_unused]] da_int NR) {}

    static inline __attribute__((__always_inline__)) void
    store_gammas([[maybe_unused]] avxvector_t<BSZ, T> *gammas, [[maybe_unused]] T *C,
                 [[maybe_unused]] da_int ldc) {}
};

// Generic kernel implementations templated on bit size
template <bsz BSZ, typename T, da_int MR, da_int NR>
inline __attribute__((__always_inline__)) void
sqeuclidean_kernel_packed_impl(da_int k, const T *Atilde, const T *Btilde, T *C,
                               da_int ldc) {
    // Declare vector registers as arrays
    avxvector_t<BSZ, T> gammas[NR];

    // Maximum number of rows we'll process at once
    constexpr da_int MAX_K = 4; // Up to 4-way unrolling

    // Arrays for vector operations, sized for maximum K
    avxvector_t<BSZ, T> alphas[MAX_K];
    avxvector_t<BSZ, T> betas[MAX_K];
    avxvector_t<BSZ, T> temps[MAX_K];

    // Load C into gamma vectors
    template_unroll_k_packed<BSZ, T, 0, NR, 1>::load_gammas(gammas, C, ldc);

    da_int i = 0;
    // 4-way unrolled loop
    for (; (i + 4) <= k; i += 4) {
        // Load 4 rows of A
        alphas[0] = kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kt_load_p<BSZ, T>(Atilde + MR);
        alphas[2] = kt_load_p<BSZ, T>(Atilde + 2 * MR);
        alphas[3] = kt_load_p<BSZ, T>(Atilde + 3 * MR);
        _mm_prefetch((const char *)(Atilde + 4 * MR), _MM_HINT_T0);

        // Process with 4-way row unrolling
        template_unroll_k_packed<BSZ, T, 0, NR, 4>::sqeuclidean_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += 4 * MR;
        Btilde += 4 * NR;
    }

    // 3-way unrolled loop
    for (; (i + 3) <= k; i += 3) {
        // Load 3 rows of A
        alphas[0] = kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kt_load_p<BSZ, T>(Atilde + MR);
        alphas[2] = kt_load_p<BSZ, T>(Atilde + 2 * MR);
        _mm_prefetch((const char *)(Atilde + 3 * MR), _MM_HINT_T0);

        // Process with 3-way row unrolling
        template_unroll_k_packed<BSZ, T, 0, NR, 3>::sqeuclidean_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += 3 * MR;
        Btilde += 3 * NR;
    }

    // 2-way unrolled loop
    for (; (i + 2) <= k; i += 2) {
        // Load 2 rows of A
        alphas[0] = kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kt_load_p<BSZ, T>(Atilde + MR);
        _mm_prefetch((const char *)(Atilde + 2 * MR), _MM_HINT_T0);

        // Process with 2-way row unrolling
        template_unroll_k_packed<BSZ, T, 0, NR, 2>::sqeuclidean_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += 2 * MR;
        Btilde += 2 * NR;
    }

    // Handle remaining single rows
    for (; i < k; i++) {
        alphas[0] = kt_load_p<BSZ, T>(Atilde);

        // Process with 1-way row unrolling
        template_unroll_k_packed<BSZ, T, 0, NR, 1>::sqeuclidean_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += MR;
        Btilde += NR;
    }

    // Store results
    template_unroll_k_packed<BSZ, T, 0, NR, 1>::store_gammas(gammas, C, ldc);
}

template <bsz BSZ, typename T, da_int MR, da_int NR>
inline __attribute__((__always_inline__)) void
manhattan_kernel_packed_impl(da_int k, const T *Atilde, const T *Btilde, T *C,
                             da_int ldc) {
    // Declare vector registers as arrays
    avxvector_t<BSZ, T> gammas[NR];

    // Maximum number of rows we'll process at once
    constexpr da_int MAX_K = 4; // Up to 4-way unrolling

    // Arrays for vector operations, sized for maximum K
    avxvector_t<BSZ, T> alphas[MAX_K];
    avxvector_t<BSZ, T> betas[MAX_K];
    avxvector_t<BSZ, T> temps[MAX_K];

    // Load C into gamma vectors
    template_unroll_k_packed<BSZ, T, 0, NR, 1>::load_gammas(gammas, C, ldc);

    da_int i = 0;

    // 4-way unrolled loop
    for (; (i + 4) <= k; i += 4) {
        // Load 4 rows of A
        alphas[0] = kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kt_load_p<BSZ, T>(Atilde + MR);
        alphas[2] = kt_load_p<BSZ, T>(Atilde + 2 * MR);
        alphas[3] = kt_load_p<BSZ, T>(Atilde + 3 * MR);
        _mm_prefetch((const char *)(Atilde + 4 * MR), _MM_HINT_T0);

        // Process with 4-way row unrolling for Manhattan distance
        template_unroll_k_packed<BSZ, T, 0, NR, 4>::manhattan_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += 4 * MR;
        Btilde += 4 * NR;
    }

    // 3-way unrolled loop
    for (; (i + 3) <= k; i += 3) {
        // Load 3 rows of A
        alphas[0] = kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kt_load_p<BSZ, T>(Atilde + MR);
        alphas[2] = kt_load_p<BSZ, T>(Atilde + 2 * MR);
        _mm_prefetch((const char *)(Atilde + 3 * MR), _MM_HINT_T0);

        // Process with 3-way row unrolling for Manhattan distance
        template_unroll_k_packed<BSZ, T, 0, NR, 3>::manhattan_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += 3 * MR;
        Btilde += 3 * NR;
    }

    // 2-way unrolled loop
    for (; (i + 2) <= k; i += 2) {
        // Load 2 rows of A
        alphas[0] = kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kt_load_p<BSZ, T>(Atilde + MR);
        _mm_prefetch((const char *)(Atilde + 2 * MR), _MM_HINT_T0);

        // Process with 2-way row unrolling for Manhattan distance
        template_unroll_k_packed<BSZ, T, 0, NR, 2>::manhattan_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += 2 * MR;
        Btilde += 2 * NR;
    }

    // Handle remaining single rows
    for (; i < k; i++) {
        alphas[0] = kt_load_p<BSZ, T>(Atilde);

        // Process with 1-way row unrolling for Manhattan distance
        template_unroll_k_packed<BSZ, T, 0, NR, 1>::manhattan_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += MR;
        Btilde += NR;
    }

    // Store results
    template_unroll_k_packed<BSZ, T, 0, NR, 1>::store_gammas(gammas, C, ldc);
}

template <bsz BSZ, typename T, da_int MR, da_int NR>
inline __attribute__((__always_inline__)) void
cosine_kernel_packed_impl(da_int k, const T *Atilde, const T *Btilde, T *C, da_int ldc) {
    // Declare vector registers as arrays
    avxvector_t<BSZ, T> gammas[NR];

    // Maximum number of rows we'll process at once
    constexpr da_int MAX_K = 4; // Up to 4-way unrolling

    // Arrays for vector operations, sized for maximum K
    avxvector_t<BSZ, T> alphas[MAX_K];
    avxvector_t<BSZ, T> betas[MAX_K];

    // Load C into gamma vectors
    template_unroll_k_packed<BSZ, T, 0, NR, 1>::load_gammas(gammas, C, ldc);

    da_int i = 0;

    // 4-way unrolled loop
    for (; (i + 4) <= k; i += 4) {
        // Load 4 rows of A
        alphas[0] = kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kt_load_p<BSZ, T>(Atilde + MR);
        alphas[2] = kt_load_p<BSZ, T>(Atilde + 2 * MR);
        alphas[3] = kt_load_p<BSZ, T>(Atilde + 3 * MR);
        _mm_prefetch((const char *)(Atilde + 4 * MR), _MM_HINT_T0);

        // Process with 4-way row unrolling
        template_unroll_k_packed<BSZ, T, 0, NR, 4>::cosine_process_k_cols(
            gammas, alphas, betas, Btilde, NR);

        Atilde += 4 * MR;
        Btilde += 4 * NR;
    }

    // 3-way unrolled loop
    for (; (i + 3) <= k; i += 3) {
        // Load 3 rows of A
        alphas[0] = kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kt_load_p<BSZ, T>(Atilde + MR);
        alphas[2] = kt_load_p<BSZ, T>(Atilde + 2 * MR);
        _mm_prefetch((const char *)(Atilde + 3 * MR), _MM_HINT_T0);

        // Process with 3-way row unrolling
        template_unroll_k_packed<BSZ, T, 0, NR, 3>::cosine_process_k_cols(
            gammas, alphas, betas, Btilde, NR);

        Atilde += 3 * MR;
        Btilde += 3 * NR;
    }

    // 2-way unrolled loop
    for (; (i + 2) <= k; i += 2) {
        // Load 2 rows of A
        alphas[0] = kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kt_load_p<BSZ, T>(Atilde + MR);
        _mm_prefetch((const char *)(Atilde + 2 * MR), _MM_HINT_T0);

        // Process with 2-way row unrolling
        template_unroll_k_packed<BSZ, T, 0, NR, 2>::cosine_process_k_cols(
            gammas, alphas, betas, Btilde, NR);

        Atilde += 2 * MR;
        Btilde += 2 * NR;
    }

    // Handle remaining single rows
    for (; i < k; i++) {
        alphas[0] = kt_load_p<BSZ, T>(Atilde);

        // Process with 1-way row unrolling
        template_unroll_k_packed<BSZ, T, 0, NR, 1>::cosine_process_k_cols(
            gammas, alphas, betas, Btilde, NR);

        Atilde += MR;
        Btilde += NR;
    }

    // Store results
    template_unroll_k_packed<BSZ, T, 0, NR, 1>::store_gammas(gammas, C, ldc);
}

template <bsz BSZ, typename T, da_int MR, da_int NR>
inline __attribute__((__always_inline__)) void
minkowski_kernel_packed_impl(da_int k, const T *Atilde, const T *Btilde, T *C, da_int ldc,
                             T exponent) {
    // Maximum number of rows we'll process at once
    constexpr da_int MAX_K = 4; // Up to 4-way unrolling

    // Arrays for vector operations, sized for maximum K
    avxvector_t<BSZ, T> alphas[MAX_K];

    da_int i = 0;

    // 4-way unrolled loop
    for (; (i + 4) <= k; i += 4) {
        // Load 4 rows of A
        alphas[0] = kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kt_load_p<BSZ, T>(Atilde + MR);
        alphas[2] = kt_load_p<BSZ, T>(Atilde + 2 * MR);
        alphas[3] = kt_load_p<BSZ, T>(Atilde + 3 * MR);
        _mm_prefetch((const char *)(Atilde + 4 * MR), _MM_HINT_T0);

        // Process with 4-way row unrolling for Minkowski distance
        template_unroll_k_packed<BSZ, T, 0, NR, 4>::minkowski_process_k_cols_direct(
            alphas, Btilde, C, ldc, NR, exponent);

        Atilde += 4 * MR;
        Btilde += 4 * NR;
    }

    // 3-way unrolled loop
    for (; (i + 3) <= k; i += 3) {
        // Load 3 rows of A
        alphas[0] = kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kt_load_p<BSZ, T>(Atilde + MR);
        alphas[2] = kt_load_p<BSZ, T>(Atilde + 2 * MR);
        _mm_prefetch((const char *)(Atilde + 3 * MR), _MM_HINT_T0);

        // Process with 3-way row unrolling for Minkowski distance
        template_unroll_k_packed<BSZ, T, 0, NR, 3>::minkowski_process_k_cols_direct(
            alphas, Btilde, C, ldc, NR, exponent);

        Atilde += 3 * MR;
        Btilde += 3 * NR;
    }

    // 2-way unrolled loop
    for (; (i + 2) <= k; i += 2) {
        // Load 2 rows of A
        alphas[0] = kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kt_load_p<BSZ, T>(Atilde + MR);
        _mm_prefetch((const char *)(Atilde + 2 * MR), _MM_HINT_T0);

        // Process with 2-way row unrolling for Minkowski distance
        template_unroll_k_packed<BSZ, T, 0, NR, 2>::minkowski_process_k_cols_direct(
            alphas, Btilde, C, ldc, NR, exponent);

        Atilde += 2 * MR;
        Btilde += 2 * NR;
    }

    // Handle remaining single rows
    for (; i < k; i++) {
        alphas[0] = kt_load_p<BSZ, T>(Atilde);

        // Process with 1-way row unrolling for Minkowski distance
        template_unroll_k_packed<BSZ, T, 0, NR, 1>::minkowski_process_k_cols_direct(
            alphas, Btilde, C, ldc, NR, exponent);

        Atilde += MR;
        Btilde += NR;
    }
}

#ifdef __AVX2__
namespace avx2 {

template <typename T, da_int MR, da_int NR>
void sqeuclidean_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C,
                               da_int ldc);

template <typename T, da_int MR, da_int NR>
void manhattan_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C,
                             da_int ldc);
template <typename T, da_int MR, da_int NR>
void minkowski_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C, da_int ldc,
                             T p);
template <typename T, da_int MR, da_int NR>
void cosine_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C, da_int ldc);
} // namespace avx2
#endif

#ifdef __AVX512F__
namespace avx512 {
template <typename T, da_int MR, da_int NR>
void sqeuclidean_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C,
                               da_int ldc);
template <typename T, da_int MR, da_int NR>
void manhattan_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C,
                             da_int ldc);
template <typename T, da_int MR, da_int NR>
void minkowski_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C, da_int ldc,
                             T p);
template <typename T, da_int MR, da_int NR>
void cosine_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C, da_int ldc);
} //namespace avx512
#endif

} // namespace ARCH

#endif // METRICS_KERNELS_HPP
