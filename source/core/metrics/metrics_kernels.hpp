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

#include "aoclda_types.h"
#include "kt.hpp"
#include "macros.h"
#include <cmath>
#include <immintrin.h>
#include <type_traits>

#define a_matrix(i, j) A[(j) * lda + (i)] // map a_matrix( i,j ) to array A
#define b_matrix(i, j) B[(j) * ldb + (i)] // map b_matrix( i,j ) to array B
#define c_matrix(i, j) C[(j) * ldc + (i)] // map c_matrix( i,j ) to array C

namespace ARCH {

using namespace kernel_templates;

// Helper for absolute value operations - templated on bit size
template <bsz BSZ, typename T>
inline __attribute__((__always_inline__)) avxvector_t<BSZ, T>
simd_abs(avxvector_t<BSZ, T> x) {
#ifdef __AVX512F__
    if constexpr (BSZ == kernel_templates::bsz::b512) {
        if constexpr (std::is_same_v<T, float>) {
            return _mm512_abs_ps(x);
        } else {
            return _mm512_abs_pd(x);
        }
    } else
#endif
        if constexpr (BSZ == kernel_templates::bsz::b256) {
        if constexpr (std::is_same_v<T, float>) {
            const __m256 sign_mask = kernel_templates::kt_set1_p<BSZ, T>(-0.0f);
            return _mm256_andnot_ps(sign_mask, x);
        } else {
            const __m256d sign_mask = kernel_templates::kt_set1_p<BSZ, T>(-0.0);
            return _mm256_andnot_pd(sign_mask, x);
        }
    } else {
        static_assert(BSZ == kernel_templates::bsz::b256 ||
                          BSZ == kernel_templates::bsz::b512,
                      "Unsupported bit size");
    }
}

// -----------------------------------------------------------------------
// Mask type for masked SIMD operations
// -----------------------------------------------------------------------
template <bsz BSZ, typename T> struct simd_mask_type_s;

#ifdef __AVX512F__
template <> struct simd_mask_type_s<kernel_templates::bsz::b512, float> {
    using type = __mmask16;
};
template <> struct simd_mask_type_s<kernel_templates::bsz::b512, double> {
    using type = __mmask8;
};
#endif

template <> struct simd_mask_type_s<kernel_templates::bsz::b256, float> {
    using type = __m256i;
};
template <> struct simd_mask_type_s<kernel_templates::bsz::b256, double> {
    using type = __m256i;
};

template <bsz BSZ, typename T>
using simd_mask_t = typename simd_mask_type_s<BSZ, T>::type;

// Create a mask for m valid elements out of the full SIMD width
template <bsz BSZ, typename T>
inline __attribute__((__always_inline__)) simd_mask_t<BSZ, T> simd_create_mask(da_int m) {
#ifdef __AVX512F__
    if constexpr (BSZ == kernel_templates::bsz::b512) {
        if constexpr (std::is_same_v<T, float>) {
            return __mmask16((1u << m) - 1);
        } else {
            return __mmask8((1u << m) - 1);
        }
    } else
#endif
        if constexpr (BSZ == kernel_templates::bsz::b256) {
        if constexpr (std::is_same_v<T, float>) {
            alignas(32) static const int32_t lut[16] = {-1, -1, -1, -1, -1, -1, -1, -1,
                                                        0,  0,  0,  0,  0,  0,  0,  0};
            return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lut + 8 - m));
        } else {
            alignas(32) static const int64_t lut[8] = {-1, -1, -1, -1, 0, 0, 0, 0};
            return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lut + 4 - m));
        }
    } else {
        static_assert(BSZ == kernel_templates::bsz::b256 ||
                          BSZ == kernel_templates::bsz::b512,
                      "Unsupported bit size");
    }
}

// Masked load: load m valid elements, zero out the rest
template <bsz BSZ, typename T>
inline __attribute__((__always_inline__)) avxvector_t<BSZ, T>
simd_maskz_loadu(const T *addr, simd_mask_t<BSZ, T> mask) {
#ifdef __AVX512F__
    if constexpr (BSZ == kernel_templates::bsz::b512) {
        if constexpr (std::is_same_v<T, float>) {
            return _mm512_maskz_loadu_ps(mask, addr);
        } else {
            return _mm512_maskz_loadu_pd(mask, addr);
        }
    } else
#endif
        if constexpr (BSZ == kernel_templates::bsz::b256) {
        if constexpr (std::is_same_v<T, float>) {
            return _mm256_maskload_ps(addr, mask);
        } else {
            return _mm256_maskload_pd(addr, mask);
        }
    } else {
        static_assert(BSZ == kernel_templates::bsz::b256 ||
                          BSZ == kernel_templates::bsz::b512,
                      "Unsupported bit size");
    }
}

// Masked store: store only the valid elements indicated by the mask
template <bsz BSZ, typename T>
inline __attribute__((__always_inline__)) void
simd_mask_storeu(T *addr, simd_mask_t<BSZ, T> mask, avxvector_t<BSZ, T> val) {
#ifdef __AVX512F__
    if constexpr (BSZ == kernel_templates::bsz::b512) {
        if constexpr (std::is_same_v<T, float>) {
            _mm512_mask_storeu_ps(addr, mask, val);
        } else {
            _mm512_mask_storeu_pd(addr, mask, val);
        }
    } else
#endif
        if constexpr (BSZ == kernel_templates::bsz::b256) {
        if constexpr (std::is_same_v<T, float>) {
            _mm256_maskstore_ps(addr, mask, val);
        } else {
            _mm256_maskstore_pd(addr, mask, val);
        }
    } else {
        static_assert(BSZ == kernel_templates::bsz::b256 ||
                          BSZ == kernel_templates::bsz::b512,
                      "Unsupported bit size");
    }
}

// Template for K-way unrolling with compile-time recursion
// BSZ is the bit size (kernel_templates::bsz::b256 or kernel_templates::bsz::b512)
// T is the data type (float or double)
// K is the number of columns to unroll
// CurrentK is the current column being processed in the unrolling
template <bsz BSZ, typename T, da_int K, da_int CurrentK = 0> struct k_unroll_packed {
    // Set beta values for K rows
    static inline __attribute__((__always_inline__)) void
    set_betas(avxvector_t<BSZ, T> *betas, const T *Btilde, da_int idx, da_int NR) {
        betas[CurrentK] =
            kernel_templates::kt_set1_p<BSZ, T>(Btilde[idx + CurrentK * NR]);
        k_unroll_packed<BSZ, T, K, CurrentK + 1>::set_betas(betas, Btilde, idx, NR);
    }

    // Compute differences for K rows
    static inline __attribute__((__always_inline__)) void
    compute_diffs(avxvector_t<BSZ, T> *temps, const avxvector_t<BSZ, T> *alphas,
                  const avxvector_t<BSZ, T> *betas) {
        temps[CurrentK] =
            kernel_templates::kt_sub_p<BSZ, T>(alphas[CurrentK], betas[CurrentK]);
        k_unroll_packed<BSZ, T, K, CurrentK + 1>::compute_diffs(temps, alphas, betas);
    }

    // Update gamma with FMA for K rows (squared Euclidean)
    static inline __attribute__((__always_inline__)) void
    sqeuclidean_update_gamma(avxvector_t<BSZ, T> &gamma,
                             const avxvector_t<BSZ, T> *temps) {
        gamma =
            kernel_templates::kt_fmadd_p<BSZ, T>(temps[CurrentK], temps[CurrentK], gamma);
        k_unroll_packed<BSZ, T, K, CurrentK + 1>::sqeuclidean_update_gamma(gamma, temps);
    }

    // Update gamma for Manhattan distance
    static inline __attribute__((__always_inline__)) void
    manhattan_update_gamma(avxvector_t<BSZ, T> &gamma, const avxvector_t<BSZ, T> *temps) {
        gamma =
            kernel_templates::kt_add_p<BSZ, T>(gamma, simd_abs<BSZ, T>(temps[CurrentK]));
        k_unroll_packed<BSZ, T, K, CurrentK + 1>::manhattan_update_gamma(gamma, temps);
    }

    // Update gamma for Cosine distance
    static inline __attribute__((__always_inline__)) void
    cosine_update_gamma(avxvector_t<BSZ, T> &gamma, const avxvector_t<BSZ, T> *alphas,
                        const avxvector_t<BSZ, T> *betas) {
        gamma = kernel_templates::kt_fmadd_p<BSZ, T>(alphas[CurrentK], betas[CurrentK],
                                                     gamma);
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
// BSZ is the bit size (kernel_templates::bsz::b256 or kernel_templates::bsz::b512)
// T is the data type (float or double)
// I is the current column index in the unrolling process
// N is the total number of columns to process
// K is the number of rows to unroll
template <bsz BSZ, typename T, da_int I, da_int N, da_int K>
struct template_unroll_k_packed {
    // Load gamma vectors - column unrolled
    static inline __attribute__((__always_inline__)) void
    load_gammas(avxvector_t<BSZ, T> *gammas, T *C, da_int ldc) {
        gammas[I] = kernel_templates::kt_loadu_p<BSZ, T>(&c_matrix(0, I));
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
                beta[k] = kernel_templates::kt_set1_p<BSZ, T>(Btilde[col + k * NR]);
            }

            // Calculate differences and absolute values for all K rows
            for (da_int k = 0; k < K; k++) {
                temp[k] = simd_abs<BSZ, T>(
                    kernel_templates::kt_sub_p<BSZ, T>(alphas[k], beta[k]));
                kernel_templates::kt_storeu_p<BSZ, T>(values[k], temp[k]);
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

    // Zero gamma vectors - column unrolled
    static inline __attribute__((__always_inline__)) void
    zero_gammas(avxvector_t<BSZ, T> *gammas) {
        gammas[I] = kernel_templates::kt_setzero_p<BSZ, T>();
        template_unroll_k_packed<BSZ, T, I + 1, N, K>::zero_gammas(gammas);
    }

    // Store gamma vectors - column unrolled
    static inline __attribute__((__always_inline__)) void
    store_gammas(avxvector_t<BSZ, T> *gammas, T *C, da_int ldc) {
        kernel_templates::kt_storeu_p<BSZ, T>(&c_matrix(0, I), gammas[I]);
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
    zero_gammas([[maybe_unused]] avxvector_t<BSZ, T> *gammas) {}

    static inline __attribute__((__always_inline__)) void
    store_gammas([[maybe_unused]] avxvector_t<BSZ, T> *gammas, [[maybe_unused]] T *C,
                 [[maybe_unused]] da_int ldc) {}
};

// Generic kernel implementations templated on bit size
// Uses dual accumulators to break FMA dependency chains:
// gammas[] accumulates k-rows 0,1 of each 4-unrolled step
// gammas2[] accumulates k-rows 2,3 of each 4-unrolled step
// Final result = gammas[] + gammas2[], halving the critical path
template <bsz BSZ, typename T, da_int MR, da_int NR, bool is_first_slice>
inline __attribute__((__always_inline__)) void
sqeuclidean_kernel_packed_impl(da_int k, const T *Atilde, const T *Btilde, T *C,
                               da_int ldc) {
    // Primary and secondary accumulator registers
    avxvector_t<BSZ, T> gammas[NR];
    avxvector_t<BSZ, T> gammas2[NR];

    // Arrays for vector operations
    avxvector_t<BSZ, T> alphas[2];
    avxvector_t<BSZ, T> betas[2];
    avxvector_t<BSZ, T> temps[2];

    // Load C into gamma vectors or zero-initialize on first KC slice
    if constexpr (is_first_slice)
        template_unroll_k_packed<BSZ, T, 0, NR, 1>::zero_gammas(gammas);
    else
        template_unroll_k_packed<BSZ, T, 0, NR, 1>::load_gammas(gammas, C, ldc);

    // Always zero the secondary accumulators
    template_unroll_k_packed<BSZ, T, 0, NR, 1>::zero_gammas(gammas2);

    da_int i = 0;

    // Main loop: process 4 k-rows per iteration using dual accumulators
    // k-rows 0,1 → gammas, k-rows 2,3 → gammas2
    for (; (i + 4) <= k; i += 4) {
        // Process k-rows 0,1 into gammas
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);

        template_unroll_k_packed<BSZ, T, 0, NR, 2>::sqeuclidean_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        // Process k-rows 2,3 into gammas2
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 2 * MR);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 3 * MR);
        _mm_prefetch((const char *)(Atilde + 8 * MR), _MM_HINT_T0);
        _mm_prefetch((const char *)(Btilde + 4 * NR), _MM_HINT_T0);
        _mm_prefetch((const char *)(Btilde + 4 * NR) + 64, _MM_HINT_T0);

        template_unroll_k_packed<BSZ, T, 0, NR, 2>::sqeuclidean_process_k_cols(
            gammas2, alphas, betas, temps, Btilde + 2 * NR, NR);

        Atilde += 4 * MR;
        Btilde += 4 * NR;
    }

    // Merge secondary accumulators into primary
    for (da_int j = 0; j < NR; j++)
        gammas[j] = kernel_templates::kt_add_p<BSZ, T>(gammas[j], gammas2[j]);

    // Handle remaining k-rows (0-3 left)
    // 2-way
    for (; (i + 2) <= k; i += 2) {
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);

        template_unroll_k_packed<BSZ, T, 0, NR, 2>::sqeuclidean_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += 2 * MR;
        Btilde += 2 * NR;
    }

    // 1-way
    for (; i < k; i++) {
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);

        template_unroll_k_packed<BSZ, T, 0, NR, 1>::sqeuclidean_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += MR;
        Btilde += NR;
    }

    // Store results
    template_unroll_k_packed<BSZ, T, 0, NR, 1>::store_gammas(gammas, C, ldc);
}

// Masked variant for partial blocks (m < MR and/or n < NR).
// Uses masked loads/stores to avoid the C_temp copy overhead.
// Atilde and Btilde are already zero-padded by the packing routines.
template <bsz BSZ, typename T, da_int MR, da_int NR, bool is_first_slice>
inline __attribute__((__always_inline__)) void
sqeuclidean_kernel_packed_masked_impl(da_int m, da_int n, da_int k, const T *Atilde,
                                      const T *Btilde, T *C, da_int ldc) {
    avxvector_t<BSZ, T> gammas[NR];
    avxvector_t<BSZ, T> gammas2[NR];

    avxvector_t<BSZ, T> alphas[2];
    avxvector_t<BSZ, T> betas[2];
    avxvector_t<BSZ, T> temps[2];

    // Create mask for the m valid row elements
    auto mask = simd_create_mask<BSZ, T>(m);

    // Load C into gamma vectors (masked) or zero-initialize on first KC slice
    if constexpr (is_first_slice) {
        for (da_int j = 0; j < NR; j++)
            gammas[j] = kernel_templates::kt_setzero_p<BSZ, T>();
    } else {
        for (da_int j = 0; j < n; j++)
            gammas[j] = simd_maskz_loadu<BSZ, T>(&c_matrix(0, j), mask);
        for (da_int j = n; j < NR; j++)
            gammas[j] = kernel_templates::kt_setzero_p<BSZ, T>();
    }

    // Zero secondary accumulators
    for (da_int j = 0; j < NR; j++)
        gammas2[j] = kernel_templates::kt_setzero_p<BSZ, T>();

    // Same dual-accumulator pattern as the full kernel
    da_int i = 0;
    // Main loop: k-rows 0,1 → gammas, k-rows 2,3 → gammas2
    for (; (i + 4) <= k; i += 4) {
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);

        template_unroll_k_packed<BSZ, T, 0, NR, 2>::sqeuclidean_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 2 * MR);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 3 * MR);
        _mm_prefetch((const char *)(Atilde + 8 * MR), _MM_HINT_T0);

        template_unroll_k_packed<BSZ, T, 0, NR, 2>::sqeuclidean_process_k_cols(
            gammas2, alphas, betas, temps, Btilde + 2 * NR, NR);

        Atilde += 4 * MR;
        Btilde += 4 * NR;
    }

    // Merge secondary accumulators
    for (da_int j = 0; j < NR; j++)
        gammas[j] = kernel_templates::kt_add_p<BSZ, T>(gammas[j], gammas2[j]);

    // 2-way remainder
    for (; (i + 2) <= k; i += 2) {
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);

        template_unroll_k_packed<BSZ, T, 0, NR, 2>::sqeuclidean_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += 2 * MR;
        Btilde += 2 * NR;
    }

    // 1-way remainder
    for (; i < k; i++) {
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);

        template_unroll_k_packed<BSZ, T, 0, NR, 1>::sqeuclidean_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += MR;
        Btilde += NR;
    }

    // Store only the n valid columns, with mask for m valid rows
    for (da_int j = 0; j < n; j++)
        simd_mask_storeu<BSZ, T>(&c_matrix(0, j), mask, gammas[j]);
}

template <bsz BSZ, typename T, da_int MR, da_int NR, bool is_first_slice>
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

    // Load C into gamma vectors or zero-initialize on first KC slice
    if constexpr (is_first_slice)
        template_unroll_k_packed<BSZ, T, 0, NR, 1>::zero_gammas(gammas);
    else
        template_unroll_k_packed<BSZ, T, 0, NR, 1>::load_gammas(gammas, C, ldc);

    da_int i = 0;

    // 4-way unrolled loop
    for (; (i + 4) <= k; i += 4) {
        // Load 4 rows of A
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);
        alphas[2] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 2 * MR);
        alphas[3] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 3 * MR);
        _mm_prefetch((const char *)(Atilde + 8 * MR), _MM_HINT_T0);

        // Process with 4-way row unrolling for Manhattan distance
        template_unroll_k_packed<BSZ, T, 0, NR, 4>::manhattan_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += 4 * MR;
        Btilde += 4 * NR;
    }

    // 3-way unrolled loop
    for (; (i + 3) <= k; i += 3) {
        // Load 3 rows of A
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);
        alphas[2] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 2 * MR);

        // Process with 3-way row unrolling for Manhattan distance
        template_unroll_k_packed<BSZ, T, 0, NR, 3>::manhattan_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += 3 * MR;
        Btilde += 3 * NR;
    }

    // 2-way unrolled loop
    for (; (i + 2) <= k; i += 2) {
        // Load 2 rows of A
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);

        // Process with 2-way row unrolling for Manhattan distance
        template_unroll_k_packed<BSZ, T, 0, NR, 2>::manhattan_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += 2 * MR;
        Btilde += 2 * NR;
    }

    // Handle remaining single rows
    for (; i < k; i++) {
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);

        // Process with 1-way row unrolling for Manhattan distance
        template_unroll_k_packed<BSZ, T, 0, NR, 1>::manhattan_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += MR;
        Btilde += NR;
    }

    // Store results
    template_unroll_k_packed<BSZ, T, 0, NR, 1>::store_gammas(gammas, C, ldc);
}

template <bsz BSZ, typename T, da_int MR, da_int NR, bool is_first_slice>
inline __attribute__((__always_inline__)) void
cosine_kernel_packed_impl(da_int k, const T *Atilde, const T *Btilde, T *C, da_int ldc) {
    // Declare vector registers as arrays
    avxvector_t<BSZ, T> gammas[NR];

    // Maximum number of rows we'll process at once
    constexpr da_int MAX_K = 4; // Up to 4-way unrolling

    // Arrays for vector operations, sized for maximum K
    avxvector_t<BSZ, T> alphas[MAX_K];
    avxvector_t<BSZ, T> betas[MAX_K];

    // Load C into gamma vectors or zero-initialize on first KC slice
    if constexpr (is_first_slice)
        template_unroll_k_packed<BSZ, T, 0, NR, 1>::zero_gammas(gammas);
    else
        template_unroll_k_packed<BSZ, T, 0, NR, 1>::load_gammas(gammas, C, ldc);

    da_int i = 0;

    // 4-way unrolled loop
    for (; (i + 4) <= k; i += 4) {
        // Load 4 rows of A
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);
        alphas[2] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 2 * MR);
        alphas[3] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 3 * MR);
        _mm_prefetch((const char *)(Atilde + 8 * MR), _MM_HINT_T0);

        // Process with 4-way row unrolling
        template_unroll_k_packed<BSZ, T, 0, NR, 4>::cosine_process_k_cols(
            gammas, alphas, betas, Btilde, NR);

        Atilde += 4 * MR;
        Btilde += 4 * NR;
    }

    // 3-way unrolled loop
    for (; (i + 3) <= k; i += 3) {
        // Load 3 rows of A
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);
        alphas[2] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 2 * MR);

        // Process with 3-way row unrolling
        template_unroll_k_packed<BSZ, T, 0, NR, 3>::cosine_process_k_cols(
            gammas, alphas, betas, Btilde, NR);

        Atilde += 3 * MR;
        Btilde += 3 * NR;
    }

    // 2-way unrolled loop
    for (; (i + 2) <= k; i += 2) {
        // Load 2 rows of A
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);

        // Process with 2-way row unrolling
        template_unroll_k_packed<BSZ, T, 0, NR, 2>::cosine_process_k_cols(
            gammas, alphas, betas, Btilde, NR);

        Atilde += 2 * MR;
        Btilde += 2 * NR;
    }

    // Handle remaining single rows
    for (; i < k; i++) {
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);

        // Process with 1-way row unrolling
        template_unroll_k_packed<BSZ, T, 0, NR, 1>::cosine_process_k_cols(
            gammas, alphas, betas, Btilde, NR);

        Atilde += MR;
        Btilde += NR;
    }

    // Store results
    template_unroll_k_packed<BSZ, T, 0, NR, 1>::store_gammas(gammas, C, ldc);
}

template <bsz BSZ, typename T, da_int MR, da_int NR, bool is_first_slice>
inline __attribute__((__always_inline__)) void
minkowski_kernel_packed_impl(da_int k, const T *Atilde, const T *Btilde, T *C, da_int ldc,
                             T exponent) {
    // Zero the output tile on first KC slice since minkowski accumulates directly into C
    if constexpr (is_first_slice) {
        for (da_int j = 0; j < NR; j++)
            for (da_int i = 0; i < MR; i++)
                c_matrix(i, j) = T(0);
    }

    // Maximum number of rows we'll process at once
    constexpr da_int MAX_K = 4; // Up to 4-way unrolling

    // Arrays for vector operations, sized for maximum K
    avxvector_t<BSZ, T> alphas[MAX_K];

    da_int i = 0;

    // 4-way unrolled loop
    for (; (i + 4) <= k; i += 4) {
        // Load 4 rows of A
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);
        alphas[2] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 2 * MR);
        alphas[3] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 3 * MR);
        _mm_prefetch((const char *)(Atilde + 8 * MR), _MM_HINT_T0);

        // Process with 4-way row unrolling for Minkowski distance
        template_unroll_k_packed<BSZ, T, 0, NR, 4>::minkowski_process_k_cols_direct(
            alphas, Btilde, C, ldc, NR, exponent);

        Atilde += 4 * MR;
        Btilde += 4 * NR;
    }

    // 3-way unrolled loop
    for (; (i + 3) <= k; i += 3) {
        // Load 3 rows of A
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);
        alphas[2] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 2 * MR);

        // Process with 3-way row unrolling for Minkowski distance
        template_unroll_k_packed<BSZ, T, 0, NR, 3>::minkowski_process_k_cols_direct(
            alphas, Btilde, C, ldc, NR, exponent);

        Atilde += 3 * MR;
        Btilde += 3 * NR;
    }

    // 2-way unrolled loop
    for (; (i + 2) <= k; i += 2) {
        // Load 2 rows of A
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);

        // Process with 2-way row unrolling for Minkowski distance
        template_unroll_k_packed<BSZ, T, 0, NR, 2>::minkowski_process_k_cols_direct(
            alphas, Btilde, C, ldc, NR, exponent);

        Atilde += 2 * MR;
        Btilde += 2 * NR;
    }

    // Handle remaining single rows
    for (; i < k; i++) {
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);

        // Process with 1-way row unrolling for Minkowski distance
        template_unroll_k_packed<BSZ, T, 0, NR, 1>::minkowski_process_k_cols_direct(
            alphas, Btilde, C, ldc, NR, exponent);

        Atilde += MR;
        Btilde += NR;
    }
}

// Masked variant for partial blocks (m < MR and/or n < NR) - Manhattan distance.
// Uses masked loads/stores to avoid the C_temp copy overhead.
// Atilde and Btilde are already zero-padded by the packing routines.
template <bsz BSZ, typename T, da_int MR, da_int NR, bool is_first_slice>
inline __attribute__((__always_inline__)) void
manhattan_kernel_packed_masked_impl(da_int m, da_int n, da_int k, const T *Atilde,
                                    const T *Btilde, T *C, da_int ldc) {
    avxvector_t<BSZ, T> gammas[NR];

    constexpr da_int MAX_K = 4;
    avxvector_t<BSZ, T> alphas[MAX_K];
    avxvector_t<BSZ, T> betas[MAX_K];
    avxvector_t<BSZ, T> temps[MAX_K];

    auto mask = simd_create_mask<BSZ, T>(m);

    if constexpr (is_first_slice) {
        for (da_int j = 0; j < NR; j++)
            gammas[j] = kernel_templates::kt_setzero_p<BSZ, T>();
    } else {
        for (da_int j = 0; j < n; j++)
            gammas[j] = simd_maskz_loadu<BSZ, T>(&c_matrix(0, j), mask);
        for (da_int j = n; j < NR; j++)
            gammas[j] = kernel_templates::kt_setzero_p<BSZ, T>();
    }

    da_int i = 0;
    for (; (i + 4) <= k; i += 4) {
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);
        alphas[2] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 2 * MR);
        alphas[3] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 3 * MR);
        _mm_prefetch((const char *)(Atilde + 8 * MR), _MM_HINT_T0);

        template_unroll_k_packed<BSZ, T, 0, NR, 4>::manhattan_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += 4 * MR;
        Btilde += 4 * NR;
    }

    for (; (i + 3) <= k; i += 3) {
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);
        alphas[2] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 2 * MR);

        template_unroll_k_packed<BSZ, T, 0, NR, 3>::manhattan_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += 3 * MR;
        Btilde += 3 * NR;
    }

    for (; (i + 2) <= k; i += 2) {
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);

        template_unroll_k_packed<BSZ, T, 0, NR, 2>::manhattan_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += 2 * MR;
        Btilde += 2 * NR;
    }

    for (; i < k; i++) {
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);

        template_unroll_k_packed<BSZ, T, 0, NR, 1>::manhattan_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += MR;
        Btilde += NR;
    }

    for (da_int j = 0; j < n; j++)
        simd_mask_storeu<BSZ, T>(&c_matrix(0, j), mask, gammas[j]);
}

// Masked variant for partial blocks (m < MR and/or n < NR) - Cosine distance.
// Uses masked loads/stores to avoid the C_temp copy overhead.
// Atilde and Btilde are already zero-padded by the packing routines.
template <bsz BSZ, typename T, da_int MR, da_int NR, bool is_first_slice>
inline __attribute__((__always_inline__)) void
cosine_kernel_packed_masked_impl(da_int m, da_int n, da_int k, const T *Atilde,
                                 const T *Btilde, T *C, da_int ldc) {
    avxvector_t<BSZ, T> gammas[NR];

    constexpr da_int MAX_K = 4;
    avxvector_t<BSZ, T> alphas[MAX_K];
    avxvector_t<BSZ, T> betas[MAX_K];

    auto mask = simd_create_mask<BSZ, T>(m);

    if constexpr (is_first_slice) {
        for (da_int j = 0; j < NR; j++)
            gammas[j] = kernel_templates::kt_setzero_p<BSZ, T>();
    } else {
        for (da_int j = 0; j < n; j++)
            gammas[j] = simd_maskz_loadu<BSZ, T>(&c_matrix(0, j), mask);
        for (da_int j = n; j < NR; j++)
            gammas[j] = kernel_templates::kt_setzero_p<BSZ, T>();
    }

    da_int i = 0;
    for (; (i + 4) <= k; i += 4) {
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);
        alphas[2] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 2 * MR);
        alphas[3] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 3 * MR);
        _mm_prefetch((const char *)(Atilde + 8 * MR), _MM_HINT_T0);

        template_unroll_k_packed<BSZ, T, 0, NR, 4>::cosine_process_k_cols(
            gammas, alphas, betas, Btilde, NR);

        Atilde += 4 * MR;
        Btilde += 4 * NR;
    }

    for (; (i + 3) <= k; i += 3) {
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);
        alphas[2] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 2 * MR);

        template_unroll_k_packed<BSZ, T, 0, NR, 3>::cosine_process_k_cols(
            gammas, alphas, betas, Btilde, NR);

        Atilde += 3 * MR;
        Btilde += 3 * NR;
    }

    for (; (i + 2) <= k; i += 2) {
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);

        template_unroll_k_packed<BSZ, T, 0, NR, 2>::cosine_process_k_cols(
            gammas, alphas, betas, Btilde, NR);

        Atilde += 2 * MR;
        Btilde += 2 * NR;
    }

    for (; i < k; i++) {
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);

        template_unroll_k_packed<BSZ, T, 0, NR, 1>::cosine_process_k_cols(
            gammas, alphas, betas, Btilde, NR);

        Atilde += MR;
        Btilde += NR;
    }

    for (da_int j = 0; j < n; j++)
        simd_mask_storeu<BSZ, T>(&c_matrix(0, j), mask, gammas[j]);
}

// Masked variant for partial blocks (m < MR and/or n < NR) - Minkowski distance.
// Uses masked loads/stores to avoid the C_temp copy overhead.
// Atilde and Btilde are already zero-padded by the packing routines.
// Note: Minkowski accumulates directly into C via scalar ops, so we
// zero only the valid (m x n) portion and let the scalar loop handle bounds.
template <bsz BSZ, typename T, da_int MR, da_int NR, bool is_first_slice>
inline __attribute__((__always_inline__)) void
minkowski_kernel_packed_masked_impl(da_int m, da_int n, da_int k, const T *Atilde,
                                    const T *Btilde, T *C, da_int ldc, T exponent) {
    // Zero only the valid m x n portion on first KC slice
    if constexpr (is_first_slice) {
        for (da_int j = 0; j < n; j++)
            for (da_int i = 0; i < m; i++)
                c_matrix(i, j) = T(0);
    }

    constexpr da_int MAX_K = 4;
    avxvector_t<BSZ, T> alphas[MAX_K];

    da_int i = 0;

    // Minkowski uses minkowski_process_k_cols_direct which writes into C directly
    // with scalar loops over MR x NR. We need to use the full kernel since Atilde/Btilde
    // are zero-padded, but accumulate only into the valid (m x n) region.
    // We use a local C_local buffer of size MR x NR to capture the full kernel output,
    // then copy the valid portion back.
    T C_local[MR * NR] = {};

    // If not first slice, seed C_local with the valid portion of C
    if constexpr (!is_first_slice) {
        for (da_int j = 0; j < n; j++)
            for (da_int ii = 0; ii < m; ii++)
                C_local[ii + j * MR] = c_matrix(ii, j);
    }

    for (; (i + 4) <= k; i += 4) {
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);
        alphas[2] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 2 * MR);
        alphas[3] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 3 * MR);
        _mm_prefetch((const char *)(Atilde + 8 * MR), _MM_HINT_T0);

        template_unroll_k_packed<BSZ, T, 0, NR, 4>::minkowski_process_k_cols_direct(
            alphas, Btilde, C_local, MR, NR, exponent);

        Atilde += 4 * MR;
        Btilde += 4 * NR;
    }

    for (; (i + 3) <= k; i += 3) {
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);
        alphas[2] = kernel_templates::kt_load_p<BSZ, T>(Atilde + 2 * MR);

        template_unroll_k_packed<BSZ, T, 0, NR, 3>::minkowski_process_k_cols_direct(
            alphas, Btilde, C_local, MR, NR, exponent);

        Atilde += 3 * MR;
        Btilde += 3 * NR;
    }

    for (; (i + 2) <= k; i += 2) {
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);
        alphas[1] = kernel_templates::kt_load_p<BSZ, T>(Atilde + MR);

        template_unroll_k_packed<BSZ, T, 0, NR, 2>::minkowski_process_k_cols_direct(
            alphas, Btilde, C_local, MR, NR, exponent);

        Atilde += 2 * MR;
        Btilde += 2 * NR;
    }

    for (; i < k; i++) {
        alphas[0] = kernel_templates::kt_load_p<BSZ, T>(Atilde);

        template_unroll_k_packed<BSZ, T, 0, NR, 1>::minkowski_process_k_cols_direct(
            alphas, Btilde, C_local, MR, NR, exponent);

        Atilde += MR;
        Btilde += NR;
    }

    // Copy valid portion back to C
    for (da_int j = 0; j < n; j++)
        for (da_int ii = 0; ii < m; ii++)
            c_matrix(ii, j) = C_local[ii + j * MR];
}

#ifdef __AVX2__
namespace avx2 {

template <typename T, da_int MR, da_int NR, bool is_first_slice>
void sqeuclidean_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C,
                               da_int ldc);

template <typename T, da_int MR, da_int NR, bool is_first_slice>
void manhattan_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C,
                             da_int ldc);
template <typename T, da_int MR, da_int NR, bool is_first_slice>
void minkowski_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C, da_int ldc,
                             T p);
template <typename T, da_int MR, da_int NR, bool is_first_slice>
void cosine_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C, da_int ldc);
template <typename T, da_int MR, da_int NR, bool is_first_slice>
void sqeuclidean_kernel_packed_masked(da_int m, da_int n, da_int k, const T *Atilde,
                                      const T *Btilde, T *C, da_int ldc);
template <typename T, da_int MR, da_int NR, bool is_first_slice>
void manhattan_kernel_packed_masked(da_int m, da_int n, da_int k, const T *Atilde,
                                    const T *Btilde, T *C, da_int ldc);
template <typename T, da_int MR, da_int NR, bool is_first_slice>
void cosine_kernel_packed_masked(da_int m, da_int n, da_int k, const T *Atilde,
                                 const T *Btilde, T *C, da_int ldc);
template <typename T, da_int MR, da_int NR, bool is_first_slice>
void minkowski_kernel_packed_masked(da_int m, da_int n, da_int k, const T *Atilde,
                                    const T *Btilde, T *C, da_int ldc, T p);
} // namespace avx2
#endif

#ifdef __AVX512F__
namespace avx512 {
template <typename T, da_int MR, da_int NR, bool is_first_slice>
void sqeuclidean_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C,
                               da_int ldc);
template <typename T, da_int MR, da_int NR, bool is_first_slice>
void manhattan_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C,
                             da_int ldc);
template <typename T, da_int MR, da_int NR, bool is_first_slice>
void minkowski_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C, da_int ldc,
                             T p);
template <typename T, da_int MR, da_int NR, bool is_first_slice>
void cosine_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C, da_int ldc);
template <typename T, da_int MR, da_int NR, bool is_first_slice>
void sqeuclidean_kernel_packed_masked(da_int m, da_int n, da_int k, const T *Atilde,
                                      const T *Btilde, T *C, da_int ldc);
template <typename T, da_int MR, da_int NR, bool is_first_slice>
void manhattan_kernel_packed_masked(da_int m, da_int n, da_int k, const T *Atilde,
                                    const T *Btilde, T *C, da_int ldc);
template <typename T, da_int MR, da_int NR, bool is_first_slice>
void cosine_kernel_packed_masked(da_int m, da_int n, da_int k, const T *Atilde,
                                 const T *Btilde, T *C, da_int ldc);
template <typename T, da_int MR, da_int NR, bool is_first_slice>
void minkowski_kernel_packed_masked(da_int m, da_int n, da_int k, const T *Atilde,
                                    const T *Btilde, T *C, da_int ldc, T p);
} //namespace avx512
#endif

} // namespace ARCH

#undef a_matrix
#undef b_matrix
#undef c_matrix

#endif // METRICS_KERNELS_HPP
