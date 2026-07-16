/*
 * Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
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
#define DA_KT_HPP

// This header includes the kernel templates and additionally L2 micro-kernel
// that are specific to AOCL-DA project.
// ------------------------------------------------------------------------------

#include "aoclda.h"
using kt_int_t = da_int;
#include "kernel-templates/kernel_templates.hpp"
// L2 micro kernels for different compilers
#if defined(__aocc__) && __has_include("amdlibm_vec.h")
#include "kt_l2_clang.hpp"
#elif defined(__GNUC__)
// When using GCC, we use the GCC vectorized functions
#include "kt_l2_gcc.hpp"
#endif

#include "kt_exp.hpp"
// ------------------------------------------------------------------------------

// Helper functions and macros that assist in instantiation of KT-based functions
// ------------------------------------------------------------------------------

// Instantiates a kernel template function defined by "FUNC"
#define DA_KT_INSTANTIATE(FUNC, BSZ)                                                     \
    FUNC(BSZ, float);                                                                    \
    FUNC(BSZ, double);

// Instantiates a kernel template function defined by "FUNC"
#define DA_KT_INSTANTIATE_EXT(FUNC, BSZ, EXT)                                            \
    FUNC(BSZ, float, EXT);                                                               \
    FUNC(BSZ, double, EXT);

// Instantiates a kernel template function defined by "FUNC" for _Float16
#ifdef __AVX512FP16__
#define DA_KT_INSTANTIATE_FP16(FUNC, BSZ) FUNC(BSZ, _Float16);
#define DA_KT_INSTANTIATE_FP16_EXT(FUNC, BSZ, EXT) FUNC(BSZ, _Float16, EXT);
#else
#define DA_KT_INSTANTIATE_FP16(FUNC, BSZ)
#define DA_KT_INSTANTIATE_FP16_EXT(FUNC, BSZ, EXT)
#endif

#endif // DA_KT_HPP