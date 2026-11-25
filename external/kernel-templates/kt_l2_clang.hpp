/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef DA_KT_HPP
#error "Never use ``kt_l2_clang.hpp'' directly; include ``kt.hpp'' instead."
#endif

#ifndef _KT_L2_CLANG_
#define _KT_L2_CLANG_

// When using AOCC, we use the AMD libm vectorized functions
// Include amdlibm_vec.h and enable 'AMD_LIBM_VEC_EXPERIMENTAL'
// macro to use AMD libm vectorized functions
#define AMD_LIBM_VEC_EXPERIMENTAL
#include "amdlibm_vec.h"
#include "kt_exp.hpp"
#include <immintrin.h>

/*
 * NOTE: This file needs Clang compiler and AOCL libm support.
 */

// Computes the exponential of the given AVX vector using AOCC-specific intrinsics.
template <kernel_templates::bsz SZ, typename SUF>
KT_FORCE_INLINE kernel_templates::avxvector_t<SZ, SUF>
kernel_templates::kt_exp_p(const kernel_templates::avxvector_t<SZ, SUF> a) noexcept {

    using namespace kernel_templates;

    if constexpr (SZ == bsz::b128) {
        if constexpr (std::is_same_v<SUF, double>) {
            return amd_vrd2_exp(a);
        } else if constexpr (std::is_same_v<SUF, float>) {
            return amd_vrs4_expf(a);
        }
#ifdef __AVX512FP16__
        else if constexpr (std::is_same_v<SUF, _Float16>) {
            avxvector_t<bsz::b256, float> a32 = _mm256_cvtph_ps((__m128i)a);
            avxvector_t<bsz::b256, float> r32 = amd_vrs8_expf(a32);
            return (__m128h)_mm256_cvtps_ph(r32, _MM_FROUND_CUR_DIRECTION);
        }
#endif
        else {
            static_assert(false, "Unsupported type for kt_exp_p");
        }
    } else if constexpr (SZ == bsz::b256) {
        if constexpr (std::is_same_v<SUF, double>) {
            return amd_vrd4_exp(a);
        } else if constexpr (std::is_same_v<SUF, float>) {
            return amd_vrs8_expf(a);
        }
#ifdef __AVX512FP16__
        else if constexpr (std::is_same_v<SUF, _Float16>) {
            avxvector_t<bsz::b512, float> a32 = _mm512_cvtph_ps((__m256i)a);
            avxvector_t<bsz::b512, float> r32 = amd_vrs16_expf(a32);
            return (__m256h)_mm512_cvtps_ph(r32,
                                            _MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC);
        }
#endif
        else {
            static_assert(false, "Unsupported type for kt_exp_p");
        }
    }
#if defined(__AVX512F__)
    else if constexpr (SZ == bsz::b512) {
        if constexpr (std::is_same_v<SUF, double>) {
            return amd_vrd8_exp(a);
        } else if constexpr (std::is_same_v<SUF, float>) {
            return amd_vrs16_expf(a);
        }
#ifdef __AVX512FP16__
        else if constexpr (std::is_same_v<SUF, _Float16>) {
            __m256i lo = _mm512_extracti64x4_epi64(a, 0);
            __m256i hi = _mm512_extracti64x4_epi64(a, 1);
            avxvector_t<SZ, float> lo32 = _mm512_cvtph_ps(lo);
            avxvector_t<SZ, float> hi32 = _mm512_cvtph_ps(hi);
            avxvector_t<SZ, float> r32lo = amd_vrs16_expf(lo32);
            avxvector_t<SZ, float> r32hi = amd_vrs16_expf(hi32);
            avxvector_t<bsz::b256, _Float16> rlo = (__m256h)_mm512_cvtps_ph(
                r32lo, _MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC);
            avxvector_t<bsz::b256, _Float16> rhi = (__m256h)_mm512_cvtps_ph(
                r32hi, _MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC);
            avxvector_t<SZ, _Float16> result = _mm512_castph256_ph512(rlo);
            return (__m512h)_mm512_inserti64x4(result, rhi, 1);
        }
#endif
        else {
            static_assert(false, "Unsupported type for kt_exp_p");
        }
    }
#endif // __AVX512F__
    else {
        static_assert(false, "Unsupported vector size for kt_exp_p");
    }
}

#endif // _KT_L2_CLANG_