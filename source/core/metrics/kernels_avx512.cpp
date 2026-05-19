/*
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "kt.hpp"
#include "metrics_kernels.hpp"
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <type_traits>

#ifdef __AVX512F__

namespace ARCH {

namespace avx512 {

using namespace kernel_templates;

//***************************************************************
//                            PACKED
//***************************************************************

template <typename T, da_int MR, da_int NR>
void sqeuclidean_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C,
                               da_int ldc) {
    sqeuclidean_kernel_packed_impl<bsz::b512, T, MR, NR>(k, Atilde, Btilde, C, ldc);
}

template <typename T, da_int MR, da_int NR>
void manhattan_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C,
                             da_int ldc) {
    manhattan_kernel_packed_impl<bsz::b512, T, MR, NR>(k, Atilde, Btilde, C, ldc);
}

template <typename T, da_int MR, da_int NR>
void minkowski_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C, da_int ldc,
                             T exponent) {
    minkowski_kernel_packed_impl<bsz::b512, T, MR, NR>(k, Atilde, Btilde, C, ldc,
                                                       exponent);
}

template <typename T, da_int MR, da_int NR>
void cosine_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C, da_int ldc) {
    cosine_kernel_packed_impl<bsz::b512, T, MR, NR>(k, Atilde, Btilde, C, ldc);
}

template void sqeuclidean_kernel_packed<float, 16, 16>(da_int k, const float *Atilde,
                                                       const float *Btilde, float *C,
                                                       da_int ldc);

template void sqeuclidean_kernel_packed<double, 8, 8>(da_int k, const double *Atilde,
                                                      const double *Btilde, double *C,
                                                      da_int ldc);

template void sqeuclidean_kernel_packed<float, 16, 8>(da_int k, const float *Atilde,
                                                      const float *Btilde, float *C,
                                                      da_int ldc);

template void sqeuclidean_kernel_packed<float, 16, 4>(da_int k, const float *Atilde,
                                                      const float *Btilde, float *C,
                                                      da_int ldc);

template void sqeuclidean_kernel_packed<double, 8, 4>(da_int k, const double *Atilde,
                                                      const double *Btilde, double *C,
                                                      da_int ldc);

template void manhattan_kernel_packed<float, 16, 16>(da_int k, const float *Atilde,
                                                     const float *Btilde, float *C,
                                                     da_int ldc);

template void manhattan_kernel_packed<double, 8, 8>(da_int k, const double *Atilde,
                                                    const double *Btilde, double *C,
                                                    da_int ldc);

template void manhattan_kernel_packed<float, 16, 8>(da_int k, const float *Atilde,
                                                    const float *Btilde, float *C,
                                                    da_int ldc);

template void manhattan_kernel_packed<float, 16, 4>(da_int k, const float *Atilde,
                                                    const float *Btilde, float *C,
                                                    da_int ldc);

template void manhattan_kernel_packed<double, 8, 4>(da_int k, const double *Atilde,
                                                    const double *Btilde, double *C,
                                                    da_int ldc);

template void minkowski_kernel_packed<float, 16, 16>(da_int k, const float *Atilde,
                                                     const float *Btilde, float *C,
                                                     da_int ldc, float exponent);

template void minkowski_kernel_packed<double, 8, 8>(da_int k, const double *Atilde,
                                                    const double *Btilde, double *C,
                                                    da_int ldc, double exponent);
template void minkowski_kernel_packed<float, 16, 8>(da_int k, const float *Atilde,
                                                    const float *Btilde, float *C,
                                                    da_int ldc, float exponent);
template void minkowski_kernel_packed<float, 16, 4>(da_int k, const float *Atilde,
                                                    const float *Btilde, float *C,
                                                    da_int ldc, float exponent);

template void minkowski_kernel_packed<double, 8, 4>(da_int k, const double *Atilde,
                                                    const double *Btilde, double *C,
                                                    da_int ldc, double exponent);

template void cosine_kernel_packed<float, 16, 16>(da_int k, const float *Atilde,
                                                  const float *Btilde, float *C,
                                                  da_int ldc);

template void cosine_kernel_packed<double, 8, 8>(da_int k, const double *Atilde,
                                                 const double *Btilde, double *C,
                                                 da_int ldc);

template void cosine_kernel_packed<float, 16, 8>(da_int k, const float *Atilde,
                                                 const float *Btilde, float *C,
                                                 da_int ldc);
template void cosine_kernel_packed<float, 16, 4>(da_int k, const float *Atilde,
                                                 const float *Btilde, float *C,
                                                 da_int ldc);

template void cosine_kernel_packed<double, 8, 4>(da_int k, const double *Atilde,
                                                 const double *Btilde, double *C,
                                                 da_int ldc);
} // namespace avx512
} // namespace ARCH

#endif