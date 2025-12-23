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
#ifndef __INTRINSIC_TYPES_H__
#define __INTRINSIC_TYPES_H__
#include <float.h>
#include <immintrin.h>
#include <stdint.h>

typedef float f32_t;
typedef double f64_t;

/*****************************
 * Internal types
 *****************************/
typedef union {
    f32_t f;
    int32_t i;
    uint32_t u;
} flt32_t;
typedef union {
    f64_t d;
    int64_t i;
    uint64_t u;
} flt64_t;
/*****************************
 * Internal vector types
 *****************************/
/*
 * (u)int32 - 4 elements - 128 bit
 */
typedef union {
    int32_t i[4];
    uint32_t u[4];
    __m128i m128i;
} int128_t;
/*
 * float32 - 4 elements - 128 bit
 */
typedef union {
    uint32_t u[4];
    int32_t i[4];
    float f[4];
    __m128 m128;
} flt128_t;
/*
 * (u)int32 - 8 elements - 256 bit
 */
typedef union {
    int32_t i[8];
    uint32_t u[8];
    __m256i m256i;
} int256_t;
/*
 * (u)int64 - 2 elements - 128 bit
 */
typedef union {
    int64_t i[2];
    uint64_t u[2];
    __m128i m128i;
} int64x2_t;
/*
 * (u)int64 - 4 elements - 256 bit
 */
typedef union {
    int64_t i[4];
    uint64_t u[4];
    __m256i m256i;
} int64x4_t;
/*
 * float32 - 8 element - 256 bits
 */
typedef union {
    uint32_t u[8];
    int32_t i[8];
    float f[8];
    __m256 m256;
} flt256f_t;
/*
 * float64 - 2 element - 128 bits
 */
typedef union {
    uint64_t u[2];
    int64_t i[2];
    double d[2];
    __m128d m128;
} flt128d_t;
/*
 * float64 - 4 element - 256 bits
 */
typedef union {
    uint64_t u[4];
    int64_t i[4];
    double d[4];
    __m256d m256d;
} flt256d_t;

#ifdef __AVX512F__
/*
 * float64 - 8 element - 512 bits
 */
typedef union {
    uint64_t u[8];
    int64_t i[8];
    double d[8];
    __m512d m512d;
} flt512d_t;

/*
 * (u)int64 - 8 elements - 512 bit
 */
typedef union {
    int64_t i[8];
    uint64_t u[8];
    __m512i m512i;
} int64x8_t;
/*
 * (u)int32 - 16 elements - 512 bit
 */
typedef union {
    int32_t i[16];
    uint32_t u[16];
    __m512i m512i;
} int512_t;

/*
 * float32 - 16 element - 512 bits
 */
typedef union {
    uint32_t u[16];
    int32_t i[16];
    float f[16];
    __m512 m512;
} flt512f_t;
#endif

#endif