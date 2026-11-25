/* ************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
