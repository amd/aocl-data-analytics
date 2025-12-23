/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclda.h"
#include "macros.h"

#define a_matrix(i, j) A[(j) * lda + (i)]         // map a_matrix( i,j ) to array A
#define b_matrix(i, j) B[(j) * ldb + (i)]         // map b_matrix( i,j ) to array B
#define c_matrix(i, j) C[(j) * ldc + (i)]         // map c_matrix( i,j ) to array C
#define ctemp_matrix(i, j) C_temp[(j) * MR + (i)] // map ctemp_matrix( i,j ) to array C

namespace ARCH {

#ifdef __AVX2__
namespace avx2 {

template <typename T, da_int MR, da_int NR>
void sqeuclidean_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C,
                               da_int ldc);

template <typename T, da_int MR, da_int NR>
void manhattan_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C,
                             da_int ldc);
template <typename T, da_int MR, da_int NR>
void minkowski_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C, da_int ldc,
                             T p);
template <typename T, da_int MR, da_int NR>
void cosine_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C, da_int ldc);
} // namespace avx2
#endif

#ifdef __AVX512F__
namespace avx512 {
template <typename T, da_int MR, da_int NR>
void sqeuclidean_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C,
                               da_int ldc);
template <typename T, da_int MR, da_int NR>
void manhattan_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C,
                             da_int ldc);
template <typename T, da_int MR, da_int NR>
void minkowski_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C, da_int ldc,
                             T p);
template <typename T, da_int MR, da_int NR>
void cosine_kernel_packed(da_int k, const T *Atilde, const T *Btilde, T *C, da_int ldc);
} //namespace avx512
#endif

} // namespace ARCH