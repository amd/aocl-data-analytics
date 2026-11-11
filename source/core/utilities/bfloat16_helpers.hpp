/******************************************************************************
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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#ifndef BFLOAT16_HELPERS_HPP
#define BFLOAT16_HELPERS_HPP

#include "aoclda_types.h"
#include <cstdint>
#include <cstring>
#include <vector>

#ifndef _WIN32
#include "aocl_dlp.h"
#endif

extern "C" {
#ifndef BLIS_ENABLE_CBLAS
#define BLIS_ENABLE_CBLAS
#endif
#include "cblas.h"
}

namespace da_bf16 {

// AOCL-DLP defines bfloat16 as int16_t. Using the same underlying type lets
// stored buffers pass directly to DLP without casts or a second representation.
using storage_type = int16_t;

inline uint16_t f32_to_bf16_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));

    // Catch NaNs:
    // bits & 0x7fffffffu clears the sign bit
    // + inf has exponent all 1s, significand 0
    // nan has exponent all 1s, significand non zero
    if ((bits & 0x7fffffffu) > 0x7f800000u)
        // Set most significant significand bit to 1 to get quiet NaN
        return static_cast<uint16_t>((bits >> 16) | 0x0040u);

    // retained_lsb = 1 if (bits >> 16) is odd, 0 otherwise
    const uint32_t retained_lsb = (bits >> 16) & 1u;
    // 0x7fffu is one less than 0x8000 = 1000 0000 0000 0000
    // so creates a carry to the retained bits only when discarded portion
    // is greater than halfway. retained_lsb breaks the tie if it is exactly halfway
    return static_cast<uint16_t>((bits + 0x7fffu + retained_lsb) >> 16);
}

inline float bf16_bits_to_f32(uint16_t bits) {
    const uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &f32_bits, sizeof(value));
    return value;
}

inline storage_type from_float(float value) {
    const uint16_t bits = f32_to_bf16_bits(value);
    storage_type result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline uint16_t to_bits(storage_type value) {
    uint16_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline float to_float(storage_type value) { return bf16_bits_to_f32(to_bits(value)); }

} // namespace da_bf16

namespace da_blas {

// bfloat16 (A) x bfloat16 (B) -> f32 (C). AOCL-DA's bfloat16 representation matches
// the storage expected by AOCL-DLP, so typed buffers pass directly to this boundary.
inline void cblas_gemm(CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                       da_int m, da_int n, da_int k, float alpha,
                       const da_bf16::storage_type *A, da_int lda,
                       const da_bf16::storage_type *B, da_int ldb, float beta, float *C,
                       da_int ldc) {
#ifndef _WIN32
    const char order = (layout == CblasRowMajor) ? 'r' : 'c';
    const char transa = (transA == CblasNoTrans) ? 'n' : 't';
    const char transb = (transB == CblasNoTrans) ? 'n' : 't';
    aocl_gemm_bf16bf16f32of32(order, transa, transb, m, n, k, alpha, A, lda, 'n', B, ldb,
                              'n', beta, C, ldc, nullptr);
#else
    // Portable fallback: widen both operands to float and call sgemm.
    // LCOV_EXCL_START
    if (layout != CblasRowMajor || transA != CblasNoTrans || transB != CblasTrans)
        return;
    std::vector<float> Af(static_cast<size_t>(m) * k);
    std::vector<float> Bf(static_cast<size_t>(n) * k);
    for (da_int row = 0; row < m; row++)
        for (da_int feature = 0; feature < k; feature++)
            Af[static_cast<size_t>(row) * k + feature] =
                da_bf16::to_float(A[static_cast<size_t>(row) * lda + feature]);
    for (da_int row = 0; row < n; row++)
        for (da_int feature = 0; feature < k; feature++)
            Bf[static_cast<size_t>(row) * k + feature] =
                da_bf16::to_float(B[static_cast<size_t>(row) * ldb + feature]);
    cblas_sgemm(layout, transA, transB, m, n, k, alpha, Af.data(), k, Bf.data(), k, beta,
                C, ldc);
    // LCOV_EXCL_STOP
#endif
}

} // namespace da_blas

#endif // BFLOAT16_HELPERS_HPP