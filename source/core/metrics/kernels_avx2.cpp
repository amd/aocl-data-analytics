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

#include "aoclda_types.h"
#include "da_kernel_utils.hpp"
#include "kernels.hpp"
#include <cmath>
#include <immintrin.h>
#include <type_traits>

// Add after the existing includes
#ifdef __AVX2__

namespace ARCH {
namespace avx2 {

// Type traits for SIMD vector types
template <typename T> struct simd_vector_type {};

template <> struct simd_vector_type<float> {
    using type = v8sf_t;
    using vector_type = __m256;
};

template <> struct simd_vector_type<double> {
    using type = v4df_t;
    using vector_type = __m256d;
};

// Generic intrinsics dispatch based on type
template <typename T>
inline typename simd_vector_type<T>::vector_type _mm256_loadu_px(const void *mem_addr) {
    if constexpr (std::is_same_v<T, float>)
        return _mm256_loadu_ps(static_cast<const float *>(mem_addr));
    else
        return _mm256_loadu_pd(static_cast<const double *>(mem_addr));
}

template <typename T>
inline typename simd_vector_type<T>::vector_type _mm256_load_px(const void *mem_addr) {
    if constexpr (std::is_same_v<T, float>)
        return _mm256_load_ps(static_cast<const float *>(mem_addr));
    else
        return _mm256_load_pd(static_cast<const double *>(mem_addr));
}

template <typename T>
inline typename simd_vector_type<T>::vector_type _mm256_set1_px(T a) {
    if constexpr (std::is_same_v<T, float>)
        return _mm256_set1_ps(a);
    else
        return _mm256_set1_pd(a);
}

template <typename T>
inline typename simd_vector_type<T>::vector_type
_mm256_sub_px(typename simd_vector_type<T>::vector_type a,
              typename simd_vector_type<T>::vector_type b) {
    if constexpr (std::is_same_v<T, float>)
        return _mm256_sub_ps(a, b);
    else
        return _mm256_sub_pd(a, b);
}

template <typename T>
inline typename simd_vector_type<T>::vector_type
_mm256_fmadd_px(typename simd_vector_type<T>::vector_type a,
                typename simd_vector_type<T>::vector_type b,
                typename simd_vector_type<T>::vector_type c) {
    if constexpr (std::is_same_v<T, float>)
        return _mm256_fmadd_ps(a, b, c);
    else
        return _mm256_fmadd_pd(a, b, c);
}

template <typename T>
inline void _mm256_storeu_px(void *mem_addr,
                             typename simd_vector_type<T>::vector_type a) {
    if constexpr (std::is_same_v<T, float>)
        _mm256_storeu_ps(static_cast<float *>(mem_addr), a);
    else
        _mm256_storeu_pd(static_cast<double *>(mem_addr), a);
}

template <typename T>
inline void _mm256_store_px(void *mem_addr, typename simd_vector_type<T>::vector_type a) {
    if constexpr (std::is_same_v<T, float>)
        _mm256_store_ps(static_cast<float *>(mem_addr), a);
    else
        _mm256_store_pd(static_cast<double *>(mem_addr), a);
}

template <typename T>
inline typename simd_vector_type<T>::vector_type _mm256_setzero_px() {
    if constexpr (std::is_same_v<T, float>)
        return _mm256_setzero_ps();
    else
        return _mm256_setzero_pd();
}

// Helper for absolute value operations with AVX2
template <typename T>
inline typename simd_vector_type<T>::vector_type
_mm256_abs_px(typename simd_vector_type<T>::vector_type x) {
    if constexpr (std::is_same_v<T, float>) {
        // For single precision - clear sign bit
        // Create a mask with sign bits cleared (all 1s except sign bit)
        const __m256 sign_mask = _mm256_set1_ps(-0.0f);
        // Clear the sign bit using andnot (equivalent to x & ~sign_mask)
        return _mm256_andnot_ps(sign_mask, x);
    } else {
        // For double precision
        // Create a mask with sign bits cleared (all 1s except sign bit)
        const __m256d sign_mask = _mm256_set1_pd(-0.0);
        // Clear the sign bit using andnot (equivalent to x & ~sign_mask)
        return _mm256_andnot_pd(sign_mask, x);
    }
}

// Add function for adding vectors
template <typename T>
inline typename simd_vector_type<T>::vector_type
_mm256_add_px(typename simd_vector_type<T>::vector_type a,
              typename simd_vector_type<T>::vector_type b) {
    if constexpr (std::is_same_v<T, float>)
        return _mm256_add_ps(a, b);
    else
        return _mm256_add_pd(a, b);
}

// Template for K-way unrolling with compile-time recursion
template <typename T, da_int K, da_int CurrentK = 0> struct k_unroll_packed {
    // Set beta values for K rows
    static inline void set_betas(typename simd_vector_type<T>::type *betas,
                                 const T *Btilde, da_int idx, da_int NR) {
        betas[CurrentK].v = _mm256_set1_px<T>(Btilde[idx + CurrentK * NR]);
        k_unroll_packed<T, K, CurrentK + 1>::set_betas(betas, Btilde, idx, NR);
    }

    // Compute differences for K rows
    static inline void compute_diffs(typename simd_vector_type<T>::type *temps,
                                     const typename simd_vector_type<T>::type *alphas,
                                     const typename simd_vector_type<T>::type *betas) {
        temps[CurrentK].v = _mm256_sub_px<T>(alphas[CurrentK].v, betas[CurrentK].v);
        k_unroll_packed<T, K, CurrentK + 1>::compute_diffs(temps, alphas, betas);
    }

    // Update gamma with FMA for K rows
    static inline void
    sqeuclidean_update_gamma(typename simd_vector_type<T>::type &gamma,
                             const typename simd_vector_type<T>::type *temps) {
        gamma.v = _mm256_fmadd_px<T>(temps[CurrentK].v, temps[CurrentK].v, gamma.v);
        k_unroll_packed<T, K, CurrentK + 1>::sqeuclidean_update_gamma(gamma, temps);
    }

    // Update gamma with FMA for K rows
    static inline void
    manhattan_update_gamma(typename simd_vector_type<T>::type &gamma,
                           const typename simd_vector_type<T>::type *temps) {
        gamma.v = _mm256_add_px<T>(gamma.v, _mm256_abs_px<T>(temps[CurrentK].v));
        k_unroll_packed<T, K, CurrentK + 1>::manhattan_update_gamma(gamma, temps);
    }

    // Update gamma with FMA for K rows
    static inline void
    cosine_update_gamma(typename simd_vector_type<T>::type &gamma,
                        const typename simd_vector_type<T>::type *alphas,
                        const typename simd_vector_type<T>::type *betas) {
        // Calculate absolute value of the difference and add to accumulator
        gamma.v = _mm256_fmadd_px<T>(alphas[CurrentK].v, betas[CurrentK].v, gamma.v);
        // Continue to next row
        k_unroll_packed<T, K, CurrentK + 1>::cosine_update_gamma(gamma, alphas, betas);
    }
};

// Base case for k_unroll_packed recursion
template <typename T, da_int K> struct k_unroll_packed<T, K, K> {
    static inline void set_betas([[maybe_unused]]
                                 typename simd_vector_type<T>::type *betas,
                                 [[maybe_unused]] const T *Btilde,
                                 [[maybe_unused]] da_int idx,
                                 [[maybe_unused]] da_int NR) {}

    static inline void
    compute_diffs([[maybe_unused]] typename simd_vector_type<T>::type *temps,
                  [[maybe_unused]] const typename simd_vector_type<T>::type *alphas,
                  [[maybe_unused]] const typename simd_vector_type<T>::type *betas) {}

    static inline void sqeuclidean_update_gamma(
        [[maybe_unused]] typename simd_vector_type<T>::type &gamma,
        [[maybe_unused]] const typename simd_vector_type<T>::type *temps) {}

    static inline void manhattan_update_gamma(
        [[maybe_unused]] typename simd_vector_type<T>::type &gamma,
        [[maybe_unused]] const typename simd_vector_type<T>::type *temps) {}

    static inline void cosine_update_gamma(
        [[maybe_unused]] typename simd_vector_type<T>::type &gamma,
        [[maybe_unused]] const typename simd_vector_type<T>::type *alphas,
        [[maybe_unused]] const typename simd_vector_type<T>::type *betas) {}
};

// Combined template for both row and column unrolling
template <typename T, da_int I, da_int N, da_int K> struct template_unroll_k_packed {
    // Load gamma vectors - column unrolled
    static inline void load_gammas(typename simd_vector_type<T>::type *gammas, T *C,
                                   da_int ldc) {
        gammas[I].v = _mm256_loadu_px<T>(&c_matrix(0, I));
        template_unroll_k_packed<T, I + 1, N, K>::load_gammas(gammas, C, ldc);
    }

    // Process columns with K rows of A - combined row and column unrolling
    static inline __attribute__((__always_inline__)) void
    sqeuclidean_process_k_cols(typename simd_vector_type<T>::type *gammas,
                               typename simd_vector_type<T>::type *alphas,
                               typename simd_vector_type<T>::type *betas,
                               typename simd_vector_type<T>::type *temps, const T *Btilde,
                               da_int NR) {

        // Set beta values for all K rows using row-wise template unrolling
        k_unroll_packed<T, K>::set_betas(betas, Btilde, I, NR);

        // Compute differences for all K rows
        k_unroll_packed<T, K>::compute_diffs(temps, alphas, betas);

        // Update gamma with FMA for all K rows
        k_unroll_packed<T, K>::sqeuclidean_update_gamma(gammas[I], temps);

        // Continue to next column
        template_unroll_k_packed<T, I + 1, N, K>::sqeuclidean_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);
    }

    // Process columns for Manhattan distance
    static inline __attribute__((__always_inline__)) void
    manhattan_process_k_cols(typename simd_vector_type<T>::type *gammas,
                             typename simd_vector_type<T>::type *alphas,
                             typename simd_vector_type<T>::type *betas,
                             typename simd_vector_type<T>::type *temps, const T *Btilde,
                             da_int NR) {

        // Set beta values for all K rows using row-wise template unrolling
        k_unroll_packed<T, K>::set_betas(betas, Btilde, I, NR);

        // Compute differences for all K rows
        k_unroll_packed<T, K>::compute_diffs(temps, alphas, betas);

        // Update gamma with absolute differences for all K rows
        k_unroll_packed<T, K>::manhattan_update_gamma(gammas[I], temps);

        // Continue to next column
        template_unroll_k_packed<T, I + 1, N, K>::manhattan_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);
    }
    // In the template_unroll_k_packed struct
    static inline __attribute__((__always_inline__)) void
    minkowski_process_k_cols_direct(const typename simd_vector_type<T>::type *alphas,
                                    const T *Btilde, T *C, da_int ldc, da_int NR,
                                    float p) {
        // Process each column directly
        for (da_int col = 0; col < N; col++) {
            // Temporary storage for calculations
            alignas(64) T values[K][std::is_same_v<T, float> ? 8 : 4];
            typename simd_vector_type<T>::type beta[K];
            typename simd_vector_type<T>::type temp[K];

            // Set beta values for all K rows
            for (da_int k = 0; k < K; k++) {
                beta[k].v = _mm256_set1_px<T>(Btilde[col + k * NR]);
            }

            // Calculate differences and absolute values for all K rows
            for (da_int k = 0; k < K; k++) {
                temp[k].v = _mm256_abs_px<T>(_mm256_sub_px<T>(alphas[k].v, beta[k].v));
                _mm256_store_px<T>(values[k], temp[k].v);
            }

            // Apply power and accumulate directly into C matrix
            const da_int elem_count = std::is_same_v<T, float> ? 8 : 4;
            for (da_int j = 0; j < elem_count; j++) {
                for (da_int k = 0; k < K; k++) {
                    c_matrix(j, col) += std::pow(values[k][j], p);
                }
            }
        }
    }

    // Process columns for Cosine distance
    static inline void cosine_process_k_cols(typename simd_vector_type<T>::type *gammas,
                                             typename simd_vector_type<T>::type *alphas,
                                             typename simd_vector_type<T>::type *betas,
                                             const T *Btilde, da_int NR) {
        // Set beta values for all K rows using row-wise template unrolling
        k_unroll_packed<T, K>::set_betas(betas, Btilde, I, NR);

        // Update gamma with absolute differences for all K rows
        k_unroll_packed<T, K>::cosine_update_gamma(gammas[I], alphas, betas);

        // Continue to next column
        template_unroll_k_packed<T, I + 1, N, K>::cosine_process_k_cols(
            gammas, alphas, betas, Btilde, NR);
    }

    // Store gamma vectors - column unrolled
    static inline void store_gammas(typename simd_vector_type<T>::type *gammas, T *C,
                                    da_int ldc) {
        _mm256_storeu_px<T>(&c_matrix(0, I), gammas[I].v);
        template_unroll_k_packed<T, I + 1, N, K>::store_gammas(gammas, C, ldc);
    }
};

// Base case for template_unroll_k_packed column recursion
template <typename T, da_int N, da_int K> struct template_unroll_k_packed<T, N, N, K> {
    static inline void load_gammas([[maybe_unused]]
                                   typename simd_vector_type<T>::type *gammas,
                                   [[maybe_unused]] T *C, [[maybe_unused]] da_int ldc) {}

    static inline void sqeuclidean_process_k_cols(
        [[maybe_unused]] typename simd_vector_type<T>::type *gammas,
        [[maybe_unused]] typename simd_vector_type<T>::type *alphas,
        [[maybe_unused]] typename simd_vector_type<T>::type *betas,
        [[maybe_unused]] typename simd_vector_type<T>::type *temps,
        [[maybe_unused]] const T *Btilde, [[maybe_unused]] da_int NR) {}

    static inline void
    manhattan_process_k_cols([[maybe_unused]] typename simd_vector_type<T>::type *gammas,
                             [[maybe_unused]] typename simd_vector_type<T>::type *alphas,
                             [[maybe_unused]] typename simd_vector_type<T>::type *betas,
                             [[maybe_unused]] typename simd_vector_type<T>::type *temps,
                             [[maybe_unused]] const T *Btilde,
                             [[maybe_unused]] da_int NR) {}
    static inline void minkowski_process_k_cols_direct(
        [[maybe_unused]] const typename simd_vector_type<T>::type *alphas,
        [[maybe_unused]] const T *Btilde, [[maybe_unused]] T *C,
        [[maybe_unused]] da_int ldc, [[maybe_unused]] da_int NR,
        [[maybe_unused]] float p) {}
    static inline void
    cosine_process_k_cols([[maybe_unused]] typename simd_vector_type<T>::type *gammas,
                          [[maybe_unused]] typename simd_vector_type<T>::type *alphas,
                          [[maybe_unused]] typename simd_vector_type<T>::type *betas,
                          [[maybe_unused]] const T *Btilde, [[maybe_unused]] da_int NR) {}

    static inline void store_gammas([[maybe_unused]]
                                    typename simd_vector_type<T>::type *gammas,
                                    [[maybe_unused]] T *C, [[maybe_unused]] da_int ldc) {}
};

// Generic implementation using templates
template <typename T, da_int MR, da_int NR>
void sqeuclidean_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C,
                               da_int ldc) {
    // Declare vector registers as arrays
    typename simd_vector_type<T>::type gammas[NR];

    // Maximum number of rows we'll process at once
    constexpr da_int MAX_K = 4; // Up to 4-way unrolling
    // Arrays for vector operations
    typename simd_vector_type<T>::type
        alphas[MAX_K]; // Support up to 2-way unrolling for AVX2
    typename simd_vector_type<T>::type betas[MAX_K];
    typename simd_vector_type<T>::type temps[MAX_K];

    // Load C into gamma vectors
    template_unroll_k_packed<T, 0, NR, 1>::load_gammas(gammas, C, ldc);

    da_int i = 0;
    // 4-way unrolled loop
    for (; (i + 4) <= k; i += 4) {
        // Load 4 rows of A
        alphas[0].v = _mm256_load_px<T>(Atilde);
        alphas[1].v = _mm256_load_px<T>(Atilde + MR);
        alphas[2].v = _mm256_load_px<T>(Atilde + 2 * MR);
        alphas[3].v = _mm256_load_px<T>(Atilde + 3 * MR);
        _mm_prefetch((const char *)(Atilde + 4 * MR), _MM_HINT_T0);

        // Process with 4-way row unrolling
        template_unroll_k_packed<T, 0, NR, 4>::sqeuclidean_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += 4 * MR;
        Btilde += 4 * NR;
    }

    // 3-way unrolled loop
    for (; (i + 3) <= k; i += 4) {
        // Load 4 rows of A
        alphas[0].v = _mm256_load_px<T>(Atilde);
        alphas[1].v = _mm256_load_px<T>(Atilde + MR);
        alphas[2].v = _mm256_load_px<T>(Atilde + 2 * MR);
        _mm_prefetch((const char *)(Atilde + 3 * MR), _MM_HINT_T0);

        // Process with 4-way row unrolling
        template_unroll_k_packed<T, 0, NR, 3>::sqeuclidean_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += 3 * MR;
        Btilde += 3 * NR;
    }
    // 2-way unrolled loop (typical for AVX2)
    for (; (i + 2) <= k; i += 2) {
        // Load 2 rows of A
        alphas[0].v = _mm256_load_px<T>(Atilde);
        alphas[1].v = _mm256_load_px<T>(Atilde + MR);
        _mm_prefetch((const char *)(Atilde + 2 * MR), _MM_HINT_T0);

        // Process with 2-way row unrolling
        template_unroll_k_packed<T, 0, NR, 2>::sqeuclidean_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += 2 * MR;
        Btilde += 2 * NR;
    }

    // Handle remaining single rows
    for (; i < k; i++) {
        alphas[0].v = _mm256_load_px<T>(Atilde);

        // Process with 1-way row unrolling
        template_unroll_k_packed<T, 0, NR, 1>::sqeuclidean_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += MR;
        Btilde += NR;
    }

    // Store results
    template_unroll_k_packed<T, 0, NR, 1>::store_gammas(gammas, C, ldc);
}

template <typename T, da_int MR, da_int NR>
void manhattan_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C,
                             da_int ldc) {
    // Declare vector registers as arrays
    typename simd_vector_type<T>::type gammas[NR];

    // Maximum number of rows we'll process at once
    constexpr da_int MAX_K = 4; // Up to 4-way unrolling

    // Arrays for vector operations, sized for maximum K
    typename simd_vector_type<T>::type alphas[MAX_K];
    typename simd_vector_type<T>::type betas[MAX_K];
    typename simd_vector_type<T>::type temps[MAX_K];

    // Load C into gamma vectors
    template_unroll_k_packed<T, 0, NR, 1>::load_gammas(gammas, C, ldc);

    da_int i = 0;

    // 4-way unrolled loop
    for (; (i + 4) <= k; i += 4) {
        // Load 4 rows of A
        alphas[0].v = _mm256_load_px<T>(Atilde);
        alphas[1].v = _mm256_load_px<T>(Atilde + MR);
        alphas[2].v = _mm256_load_px<T>(Atilde + 2 * MR);
        alphas[3].v = _mm256_load_px<T>(Atilde + 3 * MR);
        _mm_prefetch((const char *)(Atilde + 4 * MR), _MM_HINT_T0);

        // Process with 4-way row unrolling for Manhattan distance
        template_unroll_k_packed<T, 0, NR, 4>::manhattan_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += 4 * MR;
        Btilde += 4 * NR;
    }

    // 3-way unrolled loop
    for (; (i + 3) <= k; i += 4) {
        // Load 4 rows of A
        alphas[0].v = _mm256_load_px<T>(Atilde);
        alphas[1].v = _mm256_load_px<T>(Atilde + MR);
        alphas[2].v = _mm256_load_px<T>(Atilde + 2 * MR);
        _mm_prefetch((const char *)(Atilde + 3 * MR), _MM_HINT_T0);

        // Process with 4-way row unrolling for Manhattan distance
        template_unroll_k_packed<T, 0, NR, 3>::manhattan_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += 3 * MR;
        Btilde += 3 * NR;
    }
    // 2-way unrolled loop
    for (; (i + 2) <= k; i += 2) {
        // Load 2 rows of A
        alphas[0].v = _mm256_load_px<T>(Atilde);
        alphas[1].v = _mm256_load_px<T>(Atilde + MR);
        _mm_prefetch((const char *)(Atilde + 2 * MR), _MM_HINT_T0);

        // Process with 2-way row unrolling for Manhattan distance
        template_unroll_k_packed<T, 0, NR, 2>::manhattan_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += 2 * MR;
        Btilde += 2 * NR;
    }
    // Handle remaining single rows
    for (; i < k; i++) {
        alphas[0].v = _mm256_load_px<T>(Atilde);

        // Process with 1-way row unrolling for Manhattan distance
        template_unroll_k_packed<T, 0, NR, 1>::manhattan_process_k_cols(
            gammas, alphas, betas, temps, Btilde, NR);

        Atilde += MR;
        Btilde += NR;
    }

    // Store results
    template_unroll_k_packed<T, 0, NR, 1>::store_gammas(gammas, C, ldc);
}

template <typename T, da_int MR, da_int NR>
void cosine_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C, da_int ldc) {
    // Declare vector registers as arrays
    typename simd_vector_type<T>::type gammas[NR];

    // Maximum number of rows we'll process at once
    constexpr da_int MAX_K = 4; // Up to 4-way unrolling

    // Arrays for vector operations, sized for maximum K
    typename simd_vector_type<T>::type alphas[MAX_K];
    typename simd_vector_type<T>::type betas[MAX_K];

    // Load C into gamma vectors
    template_unroll_k_packed<T, 0, NR, 1>::load_gammas(gammas, C, ldc);

    da_int i = 0;

    // 4-way unrolled loop
    for (; (i + 4) <= k; i += 4) {
        // Load 4 rows of A
        alphas[0].v = _mm256_load_px<T>(Atilde);
        alphas[1].v = _mm256_load_px<T>(Atilde + MR);
        alphas[2].v = _mm256_load_px<T>(Atilde + 2 * MR);
        alphas[3].v = _mm256_load_px<T>(Atilde + 3 * MR);
        _mm_prefetch((const char *)(Atilde + 4 * MR), _MM_HINT_T0);

        // Process with 4-way row unrolling for Manhattan distance
        template_unroll_k_packed<T, 0, NR, 4>::cosine_process_k_cols(gammas, alphas,
                                                                     betas, Btilde, NR);

        Atilde += 4 * MR;
        Btilde += 4 * NR;
    }

    // 3-way unrolled loop
    for (; (i + 3) <= k; i += 4) {
        // Load 4 rows of A
        alphas[0].v = _mm256_load_px<T>(Atilde);
        alphas[1].v = _mm256_load_px<T>(Atilde + MR);
        alphas[2].v = _mm256_load_px<T>(Atilde + 2 * MR);
        _mm_prefetch((const char *)(Atilde + 3 * MR), _MM_HINT_T0);

        // Process with 4-way row unrolling for Manhattan distance
        template_unroll_k_packed<T, 0, NR, 3>::cosine_process_k_cols(gammas, alphas,
                                                                     betas, Btilde, NR);

        Atilde += 3 * MR;
        Btilde += 3 * NR;
    }
    // 2-way unrolled loop
    for (; (i + 2) <= k; i += 2) {
        // Load 2 rows of A
        alphas[0].v = _mm256_load_px<T>(Atilde);
        alphas[1].v = _mm256_load_px<T>(Atilde + MR);
        _mm_prefetch((const char *)(Atilde + 2 * MR), _MM_HINT_T0);

        // Process with 2-way row unrolling for Manhattan distance
        template_unroll_k_packed<T, 0, NR, 2>::cosine_process_k_cols(gammas, alphas,
                                                                     betas, Btilde, NR);

        Atilde += 2 * MR;
        Btilde += 2 * NR;
    }
    // Handle remaining single rows
    for (; i < k; i++) {
        alphas[0].v = _mm256_load_px<T>(Atilde);

        // Process with 1-way row unrolling for Manhattan distance
        template_unroll_k_packed<T, 0, NR, 1>::cosine_process_k_cols(gammas, alphas,
                                                                     betas, Btilde, NR);

        Atilde += MR;
        Btilde += NR;
    }

    // Store results
    template_unroll_k_packed<T, 0, NR, 1>::store_gammas(gammas, C, ldc);
}

template <typename T, da_int MR, da_int NR>
void minkowski_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C, da_int ldc,
                             T exponent) {

    // Maximum number of rows we'll process at once
    constexpr da_int MAX_K = 4; // Up to 4-way unrolling

    // Arrays for vector operations, sized for maximum K
    typename simd_vector_type<T>::type alphas[MAX_K];

    da_int i = 0;

    // 4-way unrolled loop
    for (; (i + 4) <= k; i += 4) {
        // Load 4 rows of A
        alphas[0].v = _mm256_load_px<T>(Atilde);
        alphas[1].v = _mm256_load_px<T>(Atilde + MR);
        alphas[2].v = _mm256_load_px<T>(Atilde + 2 * MR);
        alphas[3].v = _mm256_load_px<T>(Atilde + 3 * MR);
        _mm_prefetch((const char *)(Atilde + 4 * MR), _MM_HINT_T0);

        // Process with 4-way row unrolling for Manhattan distance
        template_unroll_k_packed<T, 0, NR, 4>::minkowski_process_k_cols_direct(
            alphas, Btilde, C, ldc, NR, exponent);

        Atilde += 4 * MR;
        Btilde += 4 * NR;
    }

    // 3-way unrolled loop
    for (; (i + 3) <= k; i += 4) {
        // Load 4 rows of A
        alphas[0].v = _mm256_load_px<T>(Atilde);
        alphas[1].v = _mm256_load_px<T>(Atilde + MR);
        alphas[2].v = _mm256_load_px<T>(Atilde + 2 * MR);
        _mm_prefetch((const char *)(Atilde + 3 * MR), _MM_HINT_T0);

        // Process with 4-way row unrolling for Manhattan distance
        template_unroll_k_packed<T, 0, NR, 3>::minkowski_process_k_cols_direct(
            alphas, Btilde, C, ldc, NR, exponent);

        Atilde += 3 * MR;
        Btilde += 3 * NR;
    }
    // 2-way unrolled loop
    for (; (i + 2) <= k; i += 2) {
        // Load 2 rows of A
        alphas[0].v = _mm256_load_px<T>(Atilde);
        alphas[1].v = _mm256_load_px<T>(Atilde + MR);
        _mm_prefetch((const char *)(Atilde + 2 * MR), _MM_HINT_T0);

        // Process with 2-way row unrolling for Manhattan distance
        template_unroll_k_packed<T, 0, NR, 2>::minkowski_process_k_cols_direct(
            alphas, Btilde, C, ldc, NR, exponent);

        Atilde += 2 * MR;
        Btilde += 2 * NR;
    }
    // Handle remaining single rows
    for (; i < k; i++) {
        alphas[0].v = _mm256_loadu_px<T>(Atilde);

        // Process with 1-way row unrolling for Manhattan distance
        template_unroll_k_packed<T, 0, NR, 1>::minkowski_process_k_cols_direct(
            alphas, Btilde, C, ldc, NR, exponent);

        Atilde += MR;
        Btilde += NR;
    }
}

// Use our generic implementation in the actual kernel functions
template void sqeuclidean_kernel_packed<float, 8, 8>(da_int k, const float *Atilde,
                                                     const float *Btilde, float *C,
                                                     da_int ldc);
template void sqeuclidean_kernel_packed<double, 4, 4>(da_int k, const double *Atilde,
                                                      const double *Btilde, double *C,
                                                      da_int ldc);
template void manhattan_kernel_packed<float, 8, 8>(da_int k, const float *Atilde,
                                                   const float *Btilde, float *C,
                                                   da_int ldc);
template void manhattan_kernel_packed<double, 4, 4>(da_int k, const double *Atilde,
                                                    const double *Btilde, double *C,
                                                    da_int ldc);
template void minkowski_kernel_packed<float, 8, 8>(da_int k, const float *Atilde,
                                                   const float *Btilde, float *C,
                                                   da_int ldc, float exponent);
template void minkowski_kernel_packed<double, 4, 4>(da_int k, const double *Atilde,
                                                    const double *Btilde, double *C,
                                                    da_int ldc, double exponent);
template void cosine_kernel_packed<float, 8, 8>(da_int k, const float *Atilde,
                                                const float *Btilde, float *C,
                                                da_int ldc);
template void cosine_kernel_packed<double, 4, 4>(da_int k, const double *Atilde,
                                                 const double *Btilde, double *C,
                                                 da_int ldc);

} // namespace avx2
} // namespace ARCH

#endif