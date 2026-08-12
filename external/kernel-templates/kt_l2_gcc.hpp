/* ************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************
 */

#ifndef DA_KT_HPP
#error "Never use ``kt_l2_gcc.hpp'' directly; include ``kt.hpp'' instead."
#endif

#ifndef _KT_L2_GCC_
#define _KT_L2_GCC_

#include "kt_exp.hpp"
#include <immintrin.h>

/*
 * NOTE: This file needs gcc compiler.
 */

// Forward declarations for GCC vectorized functions
//----------------------------------------------------------------
extern "C" {
__m128d _ZGVbN2v_exp(__m128d x); // AVX exp for 2 doubles
__m128 _ZGVbN4v_expf(__m128 x);  // AVX exp for 4 floats
__m256d _ZGVdN4v_exp(__m256d x); // AVX2 exp for 4 doubles
__m256 _ZGVdN8v_expf(__m256 x);  // AVX2 exp for 8 floats
__m512d _ZGVeN8v_exp(__m512d x); // AVX512 exp for 8 doubles
__m512 _ZGVeN16v_expf(__m512 x); // AVX512 exp for 16 floats
} // extern "C"
//----------------------------------------------------------------

// Computes the exponential of the given AVX vector using GCC-specific intrinsics.
template <kernel_templates::bsz SZ, typename SUF>
KT_FORCE_INLINE kernel_templates::avxvector_t<SZ, SUF>
kernel_templates::kt_exp_p(const kernel_templates::avxvector_t<SZ, SUF> a) noexcept {

    using namespace kernel_templates;

    if constexpr (SZ == bsz::b128) {
        if constexpr (std::is_same_v<SUF, double>) {
            return _ZGVbN2v_exp(a);
        } else if constexpr (std::is_same_v<SUF, float>) {
            return _ZGVbN4v_expf(a);
        }
#ifdef __AVX512FP16__
        else if constexpr (std::is_same_v<SUF, _Float16>) {
            avxvector_t<bsz::b256, float> a32 = _mm256_cvtph_ps((__m128i)a);
            avxvector_t<bsz::b256, float> r32 = _ZGVdN8v_expf(a32);
            return (__m128h)_mm256_cvtps_ph(r32, _MM_FROUND_CUR_DIRECTION);
        }
#endif
        else {
            static_assert(kt_always_false_v<SUF>, "Unsupported type for kt_exp_p");
        }
    } else if constexpr (SZ == bsz::b256) {
        if constexpr (std::is_same_v<SUF, double>) {
            return _ZGVdN4v_exp(a);
        } else if constexpr (std::is_same_v<SUF, float>) {
            return _ZGVdN8v_expf(a);
        }
#ifdef __AVX512FP16__
        else if constexpr (std::is_same_v<SUF, _Float16>) {
            avxvector_t<bsz::b512, float> a32 = _mm512_cvtph_ps((__m256i)a);
            avxvector_t<bsz::b512, float> r32 = _ZGVeN16v_expf(a32);
            return (__m256h)_mm512_cvtps_ph(r32,
                                            _MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC);
        }
#endif
        else {
            static_assert(kt_always_false_v<SUF>, "Unsupported type for kt_exp_p");
        }
    }
#if defined(__AVX512F__)
    else if constexpr (SZ == bsz::b512) {
        if constexpr (std::is_same_v<SUF, double>) {
            return _ZGVeN8v_exp(a);
        } else if constexpr (std::is_same_v<SUF, float>) {
            return _ZGVeN16v_expf(a);
        }
#ifdef __AVX512FP16__
        else if constexpr (std::is_same_v<SUF, _Float16>) {
            __m256i lo = _mm512_extracti64x4_epi64(_mm512_castph_si512(a), 0);
            __m256i hi = _mm512_extracti64x4_epi64(_mm512_castph_si512(a), 1);
            avxvector_t<SZ, float> lo32 = _mm512_cvtph_ps(lo);
            avxvector_t<SZ, float> hi32 = _mm512_cvtph_ps(hi);
            avxvector_t<SZ, float> r32lo = _ZGVeN16v_expf(lo32);
            avxvector_t<SZ, float> r32hi = _ZGVeN16v_expf(hi32);
            avxvector_t<bsz::b256, _Float16> rlo = (__m256h)_mm512_cvtps_ph(
                r32lo, _MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC);
            avxvector_t<bsz::b256, _Float16> rhi = (__m256h)_mm512_cvtps_ph(
                r32hi, _MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC);
            avxvector_t<SZ, _Float16> result = _mm512_castph256_ph512(rlo);
            return _mm512_castsi512_ph(_mm512_inserti64x4(_mm512_castph_si512(result),
                                                          _mm256_castph_si256(rhi), 1));
        }
#endif
        else {
            static_assert(kt_always_false_v<SUF>, "Unsupported type for kt_exp_p");
        }
    }
#endif // __AVX512F__
    else {
        static_assert(kt_always_false_v<SUF>, "Unsupported vector size for kt_exp_p");
    }
}

#endif // _KT_L2_GCC_
