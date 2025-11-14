/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
KT_FORCE_INLINE kernel_templates::avxvector_t<SZ, SUF> kernel_templates::kt_exp_p(const kernel_templates::avxvector_t<SZ, SUF> a) noexcept {

    using namespace kernel_templates;

    if constexpr (SZ == bsz::b128) {
        if constexpr (std::is_same_v<SUF, double>) {
            return _ZGVbN2v_exp(a);
        } else if constexpr (std::is_same_v<SUF, float>) {
            return _ZGVbN4v_expf(a);
        }
    } else if constexpr (SZ == bsz::b256) {
        if constexpr (std::is_same_v<SUF, double>) {
            return _ZGVdN4v_exp(a);
        } else if constexpr (std::is_same_v<SUF, float>) {
            return _ZGVdN8v_expf(a);
        }

    }
#if defined(__AVX512F__)
    else if constexpr (SZ == bsz::b512) {
        if constexpr (std::is_same_v<SUF, double>) {
            return _ZGVeN8v_exp(a);
        } else if constexpr (std::is_same_v<SUF, float>) {
            return _ZGVeN16v_expf(a);
        }
    }
#endif // __AVX512F__
}

#endif // _KT_L2_GCC_