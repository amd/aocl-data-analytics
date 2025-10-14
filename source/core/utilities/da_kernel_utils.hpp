/*
 * Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef __KERNEL_UTILS__
#define __KERNEL_UTILS__

#include <immintrin.h>
#include <stdint.h>

enum vectorization_type { scalar = 0, avx = 2, avx2 = 5, avx512 = 11 };

/*****************************
  * Internal types
  *****************************/
typedef union {
    float f;
    int32_t i;
    uint32_t u;
} flt32_t;

typedef union {
    double d;
    int64_t i;
    uint64_t u;
} flt64_t;

/*****************************
  * Internal vector types
  *****************************/

#ifdef __AVX__
/*
  * (u)int32 - 4 elements - 128 bits
  */
typedef union {
    int32_t i[4] __attribute__((aligned(16)));
    uint32_t u[4] __attribute__((aligned(16)));
    __m128i v;
} v4i32_t;

/*
  * (u)int64 - 2 elements - 128 bits
  */
typedef union {
    int64_t i[2] __attribute__((aligned(16)));
    uint64_t u[2] __attribute__((aligned(16)));
    __m128i v;
} v2i64_t;

/*
  * float32 - 4 elements - 128 bits
  */
typedef union {
    uint32_t u[4] __attribute__((aligned(16)));
    int32_t i[4] __attribute__((aligned(16)));
    float f[4] __attribute__((aligned(16)));
    __m128 v;
} v4sf_t;

/*
  * float64 - 2 element - 128 bits
  */
typedef union {
    uint64_t u[2] __attribute__((aligned(16)));
    int64_t i[2] __attribute__((aligned(16)));
    double d[2] __attribute__((aligned(16)));
    __m128d v;
} v2df_t;

#endif

#ifdef __AVX2__
/*
  * (u)int32 - 8 elements - 256 bits
  */
typedef union {
    int32_t i[8] __attribute__((aligned(32)));
    uint32_t u[8] __attribute__((aligned(32)));
    __m256i v;
} v8i32_t;

/*
  * (u)int64 - 4 elements - 256 bits
  */
typedef union {
    int64_t i[4] __attribute__((aligned(32)));
    uint64_t u[4] __attribute__((aligned(32)));
    __m256i v;
} v4i64_t;

/*
  * float32 - 8 elements - 256 bits
  */
typedef union {
    uint32_t u[8] __attribute__((aligned(32)));
    int32_t i[8] __attribute__((aligned(32)));
    float f[8] __attribute__((aligned(32)));
    __m256 v;
} v8sf_t;

/*
  * float64 - 4 elements - 256 bits
  */
typedef union {
    uint64_t u[4] __attribute__((aligned(32)));
    int64_t i[4] __attribute__((aligned(32)));
    double d[4] __attribute__((aligned(32)));
    __m256d v;
} v4df_t;

#endif

#ifdef __AVX512F__

/*
  * (u)int32 - 16 elements - 512 bits
  */
typedef union {
    int32_t i[16] __attribute__((aligned(64)));
    uint32_t u[16] __attribute__((aligned(64)));
    __m512i v;
} v16i32_t;

/*
  * (u)int64 - 8 elements - 512 bits
  */
typedef union {
    int64_t i[8] __attribute__((aligned(64)));
    uint64_t u[8] __attribute__((aligned(64)));
    __m512i v;
} v8i64_t;

/*
  * float64 - 16 elements - 512 bits
  */
typedef union {
    uint32_t u[16] __attribute__((aligned(64)));
    int32_t i[16] __attribute__((aligned(64)));
    float f[16] __attribute__((aligned(64)));
    __m512 v;
} v16sf_t;

/*
  * float64 - 8 elements - 512 bits
  */
typedef union {
    uint64_t u[8] __attribute__((aligned(64)));
    int64_t i[8] __attribute__((aligned(64)));
    double d[8] __attribute__((aligned(64)));
    __m512d v;
} v8df_t;

#endif

#endif