/* ************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#error "Never use ``kt_l2_fallback.hpp'' directly; include ``kt.hpp'' instead."
#endif

#pragma message("Warning: Using fallback scalar implementation for kt_exp_p.")

#ifndef _KT_L2_FALLBACK_
#define _KT_L2_FALLBACK_

#include "kt_exp.hpp"
#include <immintrin.h>

// Computes the exponential of the given AVX vector using fallback intrinsics.
template <kernel_templates::bsz SZ, typename SUF>
KT_FORCE_INLINE kernel_templates::avxvector_t<SZ, SUF>
kernel_templates::kt_exp_p(const kernel_templates::avxvector_t<SZ, SUF> a) noexcept {

    using namespace kernel_templates;
    SUF *v = new SUF[tsz_v<SZ, SUF>];
    kt_storeu_p<SZ>(&v[0], a);
    for (size_t i = 0; i < tsz_v<SZ, SUF>; ++i) {
#ifdef __AVX512FP16__
        if constexpr (std::is_same_v<SUF, _Float16>)
            v[i] = static_cast<_Float16>(std::exp(static_cast<float>(v[i])));
        else
#endif
            v[i] = std::exp(v[i]);
    }
    kernel_templates::avxvector_t<SZ, SUF> result = kt_loadu_p<SZ>(&v[0]);
    delete[] v;
    return result;
}

#endif // _KT_L2_FALLBACK_
