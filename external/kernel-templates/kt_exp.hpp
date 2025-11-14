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

#include "aoclda.h"

// Set the kt_int
using kt_int_t = da_int;
#include "kernel-templates/kernel_templates.hpp"
// L2 micro kernels for different compilers
#if defined(__aocc__) && __has_include("amdlibm_vec.h")
    #include "kt_l2_clang.hpp"
#elif defined(__GNUC__)
    // When using GCC, we use the GCC vectorized functions
    #include "kt_l2_gcc.hpp"
#endif
// clang-format on

namespace kernel_templates
{

// -----------------------------------------------------------------------
// Computes the exponential of the given AVX vector
// using compiler specific intrinsic functions.
template <bsz SZ, typename SUF>
KT_FORCE_INLINE avxvector_t<SZ, SUF> kt_exp_p(const avxvector_t<SZ, SUF> a) noexcept;

}