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

/*
 * NOTE: This file needs Clang compiler and AOCL libm support.
 */

// Computes the exponential of the given AVX vector using AOCC-specific intrinsics.
template <kernel_templates::bsz SZ, typename SUF>
KT_FORCE_INLINE kernel_templates::avxvector_t<SZ, SUF> kernel_templates::kt_exp_p(const kernel_templates::avxvector_t<SZ, SUF> a) noexcept {

    using namespace kernel_templates;

    if constexpr (SZ == bsz::b128) {
        if constexpr (std::is_same_v<SUF, double>) {
            return amd_vrd2_exp(a);
        } else if constexpr (std::is_same_v<SUF, float>) {
            return amd_vrs4_expf(a);
        }
    } else if constexpr (SZ == bsz::b256) {
        if constexpr (std::is_same_v<SUF, double>) {
            return amd_vrd4_exp(a);
        } else if constexpr (std::is_same_v<SUF, float>) {
            return amd_vrs8_expf(a);
        }
    }
#if defined(__AVX512F__)
    else if constexpr (SZ == bsz::b512) {
        if constexpr (std::is_same_v<SUF, double>) {
            return amd_vrd8_exp(a);
        } else if constexpr (std::is_same_v<SUF, float>) {
            return amd_vrs16_expf(a);
        }
    }
#endif // __AVX512F__
}


#endif // _KT_L2_CLANG_