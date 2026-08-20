/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef DA_SIMD_MATH_HPP
#define DA_SIMD_MATH_HPP

#include "aoclda_types.h"
#include "kt.hpp"
#include "macros.h"
#include <cmath>
#include <immintrin.h>
#include <type_traits>

namespace ARCH {

namespace da_simd_math {

#if defined(__AVX512F__) || defined(__AVX2__)
// NaN-preserving clamp: sets negative values to zero, preserves NaN.
// Uses compare(< 0)/blend rather than max(vec, 0) which converts NaN to 0.
template <kernel_templates::bsz BSZ, typename T>
inline __attribute__((__always_inline__)) kernel_templates::avxvector_t<BSZ, T>
simd_clamp_neg_to_zero(kernel_templates::avxvector_t<BSZ, T> v) {
    auto zero = kernel_templates::kt_setzero_p<BSZ, T>();
#ifdef __AVX512F__
    if constexpr (BSZ == kernel_templates::bsz::b512) {
        if constexpr (std::is_same_v<T, float>) {
            auto mask = _mm512_cmp_ps_mask(v, zero, _CMP_LT_OQ);
            return _mm512_mask_blend_ps(mask, v, zero);
        } else {
            auto mask = _mm512_cmp_pd_mask(v, zero, _CMP_LT_OQ);
            return _mm512_mask_blend_pd(mask, v, zero);
        }
    } else
#endif
        if constexpr (BSZ == kernel_templates::bsz::b256) {
        if constexpr (std::is_same_v<T, float>) {
            auto mask = _mm256_cmp_ps(v, zero, _CMP_LT_OQ);
            return _mm256_blendv_ps(v, zero, mask);
        } else {
            auto mask = _mm256_cmp_pd(v, zero, _CMP_LT_OQ);
            return _mm256_blendv_pd(v, zero, mask);
        }
    } else {
        static_assert(BSZ == kernel_templates::bsz::b256 ||
                          BSZ == kernel_templates::bsz::b512,
                      "Unsupported bit size");
    }
}

// Vectorized square root (no kt_sqrt_p in the external KT library).
template <kernel_templates::bsz BSZ, typename T>
inline __attribute__((__always_inline__)) kernel_templates::avxvector_t<BSZ, T>
simd_sqrt(kernel_templates::avxvector_t<BSZ, T> v) {
#ifdef __AVX512F__
    if constexpr (BSZ == kernel_templates::bsz::b512) {
        if constexpr (std::is_same_v<T, float>)
            return _mm512_sqrt_ps(v);
        else
            return _mm512_sqrt_pd(v);
    } else
#endif
        if constexpr (BSZ == kernel_templates::bsz::b256) {
        if constexpr (std::is_same_v<T, float>)
            return _mm256_sqrt_ps(v);
        else
            return _mm256_sqrt_pd(v);
    } else {
        static_assert(BSZ == kernel_templates::bsz::b256 ||
                          BSZ == kernel_templates::bsz::b512,
                      "Unsupported bit size");
    }
}
#endif

// Vectorized square root on a contiguous array of length len.
// Assumes all elements are non-negative. NaN values are preserved (sqrt(NaN) == NaN).
template <typename T> inline void sqrt_vec(T *x, da_int len) {
#if defined(__AVX512F__)
    constexpr kernel_templates::bsz BSZ = kernel_templates::bsz::b512;
#elif defined(__AVX2__)
    constexpr kernel_templates::bsz BSZ = kernel_templates::bsz::b256;
#endif
#if defined(__AVX512F__) || defined(__AVX2__)
    constexpr da_int reg_cap = kernel_templates::tsz_v<BSZ, T>;
    da_int i = 0;
    for (; i + reg_cap <= len; i += reg_cap) {
        kernel_templates::avxvector_t<BSZ, T> vec =
            kernel_templates::kt_loadu_p<BSZ, T>(x + i);
        vec = simd_sqrt<BSZ, T>(vec);
        kernel_templates::kt_storeu_p<BSZ, T>(x + i, vec);
    }
    for (; i < len; ++i)
        x[i] = std::sqrt(x[i]);
#else
    for (da_int i = 0; i < len; ++i)
        x[i] = std::sqrt(x[i]);
#endif
}

// Vectorized square root on a contiguous array of length len.
// Negative values are clamped to zero before taking the square root.
// NaN values are preserved: the clamp uses (x < 0) which is false for NaN,
// so NaN passes through to sqrt(NaN) == NaN.
template <typename T> inline void sqrt_clamp_vec(T *x, da_int len) {
#if defined(__AVX512F__)
    constexpr kernel_templates::bsz BSZ = kernel_templates::bsz::b512;
#elif defined(__AVX2__)
    constexpr kernel_templates::bsz BSZ = kernel_templates::bsz::b256;
#endif
#if defined(__AVX512F__) || defined(__AVX2__)
    constexpr da_int reg_cap = kernel_templates::tsz_v<BSZ, T>;
    da_int i = 0;
    // Single call to simd_sqrt instead of loop unrolling is sufficient here
    // since there are no dependencies the compiler can unroll successfully.
    for (; i + reg_cap <= len; i += reg_cap) {
        kernel_templates::avxvector_t<BSZ, T> vec =
            kernel_templates::kt_loadu_p<BSZ, T>(x + i);
        vec = simd_sqrt<BSZ, T>(simd_clamp_neg_to_zero<BSZ, T>(vec));
        kernel_templates::kt_storeu_p<BSZ, T>(x + i, vec);
    }
    for (; i < len; ++i) {
        if (x[i] < static_cast<T>(0))
            x[i] = static_cast<T>(0);
        x[i] = std::sqrt(x[i]);
    }
#else
    for (da_int i = 0; i < len; ++i) {
        if (x[i] < static_cast<T>(0))
            x[i] = static_cast<T>(0);
        x[i] = std::sqrt(x[i]);
    }
#endif
}

// Vectorized square root on a column-major m x n sub-matrix with leading dimension ldc.
// Assumes all elements are non-negative.
// When columns are contiguous (m == ldc), processes the whole block in one call.
template <typename T> inline void sqrt_matrix(da_int m, da_int n, T *C, da_int ldc) {
    if (m == ldc) {
        sqrt_vec(C, m * n);
    } else {
        for (da_int j = 0; j < n; j++) {
            sqrt_vec(C + j * ldc, m);
        }
    }
}

// Vectorized square root on a column-major m x n sub-matrix with leading dimension ldc.
// Negative values are clamped to zero before taking the square root. NaN values are preserved.
// When columns are contiguous (m == ldc), processes the whole block in one call.
template <typename T>
inline void sqrt_clamp_matrix(da_int m, da_int n, T *C, da_int ldc) {
    if (m == ldc) {
        sqrt_clamp_vec(C, m * n);
    } else {
        for (da_int j = 0; j < n; j++) {
            sqrt_clamp_vec(C + j * ldc, m);
        }
    }
}

// Clamp negative values to zero in a contiguous array of length len.
// NaN values are preserved: the clamp uses (x < 0) which is false for NaN,
// so NaN lanes are left unchanged.
template <typename T> inline void clamp_nonneg_vec(T *x, da_int len) {
#if defined(__AVX512F__)
    constexpr kernel_templates::bsz BSZ = kernel_templates::bsz::b512;
#elif defined(__AVX2__)
    constexpr kernel_templates::bsz BSZ = kernel_templates::bsz::b256;
#endif
#if defined(__AVX512F__) || defined(__AVX2__)
    constexpr da_int reg_cap = kernel_templates::tsz_v<BSZ, T>;
    da_int i = 0;
    for (; i + reg_cap <= len; i += reg_cap) {
        kernel_templates::avxvector_t<BSZ, T> vec =
            kernel_templates::kt_loadu_p<BSZ, T>(x + i);
        vec = simd_clamp_neg_to_zero<BSZ, T>(vec);
        kernel_templates::kt_storeu_p<BSZ, T>(x + i, vec);
    }
    for (; i < len; ++i) {
        if (x[i] < static_cast<T>(0))
            x[i] = static_cast<T>(0);
    }
#else
    for (da_int i = 0; i < len; ++i) {
        if (x[i] < static_cast<T>(0))
            x[i] = static_cast<T>(0);
    }
#endif
}

// Clamp negative values to zero in a column-major m x n sub-matrix with leading dimension ldc.
// When columns are contiguous (m == ldc), processes the whole block in one call.
template <typename T>
inline void clamp_nonneg_matrix(da_int m, da_int n, T *C, da_int ldc) {
    if (m == ldc) {
        clamp_nonneg_vec(C, m * n);
    } else {
        for (da_int j = 0; j < n; j++) {
            clamp_nonneg_vec(C + j * ldc, m);
        }
    }
}
} // namespace da_simd_math
} // namespace ARCH

#endif // DA_SIMD_MATH_HPP
