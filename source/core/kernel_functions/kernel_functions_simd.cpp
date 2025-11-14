/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * ************************************************************************ */

#include "aoclda_types.h"
#include "da_kernel_utils.hpp"
#include "immintrin.h"
#include "kernel_functions.hpp"
#include "kt.hpp"
#include "macros.h"
#include <cmath>

namespace ARCH {

namespace da_kernel_functions {

using namespace da_kernel_functions_types;
using namespace kernel_templates;

/* These functions contain performance-critical loops which must vectorize for performance. */

/* first_dim represents the dimension we iterate over first, for example in column-major it is number of rows.
second_dim represents the dimension we iterate over second, for example in column-major it is number of columns.
This is to prevent creating switch-case for row/column major data. */

template <typename T>
void exp_kernel_scalar(da_int first_dim, da_int second_dim, T *data, da_int ldd,
                       T multiplier) {
    for (da_int i = 0; i < second_dim; i++) {
        T *data_ptr = &data[i * ldd];
        for (da_int j = 0; j < first_dim; j++) {
            data_ptr[j] = exp(multiplier * data_ptr[j]);
        }
    }
}

template <typename T>
void pow_kernel_scalar(da_int first_dim, da_int second_dim, T *data, da_int ldd, T coef0,
                       da_int degree) {
    for (da_int i = 0; i < second_dim; i++) {
        T *data_ptr = &data[i * ldd];
        for (da_int j = 0; j < first_dim; j++) {
            data_ptr[j] = pow(data_ptr[j] + coef0, degree);
        }
    }
}

template <typename T>
void tanh_kernel_scalar(da_int first_dim, da_int second_dim, T *data, da_int ldd,
                        T coef0) {
    for (da_int i = 0; i < second_dim; i++) {
        T *data_ptr = &data[i * ldd];
        for (da_int j = 0; j < first_dim; j++) {
            data_ptr[j] = tanh(data_ptr[j] + coef0);
        }
    }
}

template <bsz SZ, typename SUF>
inline __attribute__((__always_inline__)) void
exp_kt(da_int first_dim, da_int second_dim, SUF *data, da_int ldd, SUF multiplier) {
    const da_int simd_length{tsz_v<SZ, SUF>};
    da_int remainder = first_dim % simd_length;
    da_int size = first_dim - remainder;
    for (da_int i = 0; i < second_dim; i++) {
        da_int offset = i * ldd;
        for (da_int j = 0; j < size; j += simd_length) {
            avxvector_t<SZ, SUF> v_data = kt_loadu_p<SZ>(&data[j + offset]);
            v_data = kt_mul_p<SZ, SUF>(v_data, kt_set1_p<SZ>(multiplier));
            v_data = kt_exp_p<SZ, SUF>(v_data);
            kt_storeu_p<SZ>(&data[j + offset], v_data);
        }
        // Handle the remaining elements
        for (da_int j = 0; j < remainder; j++) {
            data[j + offset + size] = exp(multiplier * data[j + offset + size]);
        }
    }
}

#ifndef USE_SCALAR_MATH // Compiler macro defined in the CMake
// Single set of instantiations for the detected compiler
template void exp_kt<bsz::b128, float>(da_int first_dim, da_int second_dim, float *data,
                                       da_int ldd, float multiplier);
template void exp_kt<bsz::b128, double>(da_int first_dim, da_int second_dim, double *data,
                                        da_int ldd, double multiplier);
template void exp_kt<bsz::b256, float>(da_int first_dim, da_int second_dim, float *data,
                                       da_int ldd, float multiplier);
template void exp_kt<bsz::b256, double>(da_int first_dim, da_int second_dim, double *data,
                                        da_int ldd, double multiplier);
#ifdef __AVX512F__
template void exp_kt<bsz::b512, float>(da_int first_dim, da_int second_dim, float *data,
                                       da_int ldd, float multiplier);
template void exp_kt<bsz::b512, double>(da_int first_dim, da_int second_dim, double *data,
                                        da_int ldd, double multiplier);
#endif // __AVX512F__
#endif // USE_SCALAR_MATH

// Simplified exp_kernel specializations using constexpr
template <>
void exp_kernel<float, scalar>(da_int first_dim, da_int second_dim, float *data,
                               da_int ldd, float multiplier) {
    exp_kernel_scalar(first_dim, second_dim, data, ldd, multiplier);
}

template <>
void exp_kernel<double, scalar>(da_int first_dim, da_int second_dim, double *data,
                                da_int ldd, double multiplier) {
    exp_kernel_scalar(first_dim, second_dim, data, ldd, multiplier);
}

template <>
void exp_kernel<float, avx>(da_int first_dim, da_int second_dim, float *data, da_int ldd,
                            float multiplier) {
#ifndef USE_SCALAR_MATH
    exp_kt<bsz::b128, float>(first_dim, second_dim, data, ldd, multiplier);
#else
    exp_kernel_scalar(first_dim, second_dim, data, ldd, multiplier);
#endif
}

template <>
void exp_kernel<double, avx>(da_int first_dim, da_int second_dim, double *data,
                             da_int ldd, double multiplier) {
#ifndef USE_SCALAR_MATH
    exp_kt<bsz::b128, double>(first_dim, second_dim, data, ldd, multiplier);
#else
    exp_kernel_scalar(first_dim, second_dim, data, ldd, multiplier);
#endif
}

template <>
void exp_kernel<float, avx2>(da_int first_dim, da_int second_dim, float *data, da_int ldd,
                             float multiplier) {
#ifndef USE_SCALAR_MATH
    exp_kt<bsz::b256, float>(first_dim, second_dim, data, ldd, multiplier);
#else
    exp_kernel_scalar(first_dim, second_dim, data, ldd, multiplier);
#endif
}

template <>
void exp_kernel<double, avx2>(da_int first_dim, da_int second_dim, double *data,
                              da_int ldd, double multiplier) {
#ifndef USE_SCALAR_MATH
    exp_kt<bsz::b256, double>(first_dim, second_dim, data, ldd, multiplier);
#else
    exp_kernel_scalar(first_dim, second_dim, data, ldd, multiplier);
#endif
}

#ifdef __AVX512F__
template <>
void exp_kernel<float, avx512>(da_int first_dim, da_int second_dim, float *data,
                               da_int ldd, float multiplier) {
#ifndef USE_SCALAR_MATH
    exp_kt<bsz::b512, float>(first_dim, second_dim, data, ldd, multiplier);
#else
    exp_kernel_scalar(first_dim, second_dim, data, ldd, multiplier);
#endif
}

template <>
void exp_kernel<double, avx512>(da_int first_dim, da_int second_dim, double *data,
                                da_int ldd, double multiplier) {
#ifndef USE_SCALAR_MATH
    exp_kt<bsz::b512, double>(first_dim, second_dim, data, ldd, multiplier);
#else
    exp_kernel_scalar(first_dim, second_dim, data, ldd, multiplier);
#endif
}
#endif

// Temporary solution - we only have scalar implementations (still get vectorization for AOCC)
template <>
void pow_kernel<float, scalar>(da_int first_dim, da_int second_dim, float *data,
                               da_int ldd, float coef0, da_int degree) {
    pow_kernel_scalar(first_dim, second_dim, data, ldd, coef0, degree);
}
template <>
void pow_kernel<double, scalar>(da_int first_dim, da_int second_dim, double *data,
                                da_int ldd, double coef0, da_int degree) {
    pow_kernel_scalar(first_dim, second_dim, data, ldd, coef0, degree);
}
template <>
void pow_kernel<float, avx>(da_int first_dim, da_int second_dim, float *data, da_int ldd,
                            float coef0, da_int degree) {
    pow_kernel_scalar(first_dim, second_dim, data, ldd, coef0, degree);
}
template <>
void pow_kernel<double, avx>(da_int first_dim, da_int second_dim, double *data,
                             da_int ldd, double coef0, da_int degree) {
    pow_kernel_scalar(first_dim, second_dim, data, ldd, coef0, degree);
}
template <>
void pow_kernel<float, avx2>(da_int first_dim, da_int second_dim, float *data, da_int ldd,
                             float coef0, da_int degree) {
    pow_kernel_scalar(first_dim, second_dim, data, ldd, coef0, degree);
}
template <>
void pow_kernel<double, avx2>(da_int first_dim, da_int second_dim, double *data,
                              da_int ldd, double coef0, da_int degree) {
    pow_kernel_scalar(first_dim, second_dim, data, ldd, coef0, degree);
}
template <>
void pow_kernel<float, avx512>(da_int first_dim, da_int second_dim, float *data,
                               da_int ldd, float coef0, da_int degree) {
    pow_kernel_scalar(first_dim, second_dim, data, ldd, coef0, degree);
}
template <>
void pow_kernel<double, avx512>(da_int first_dim, da_int second_dim, double *data,
                                da_int ldd, double coef0, da_int degree) {
    pow_kernel_scalar(first_dim, second_dim, data, ldd, coef0, degree);
}

template <>
void tanh_kernel<float, scalar>(da_int first_dim, da_int second_dim, float *data,
                                da_int ldd, float coef0) {
    tanh_kernel_scalar(first_dim, second_dim, data, ldd, coef0);
}
template <>
void tanh_kernel<double, scalar>(da_int first_dim, da_int second_dim, double *data,
                                 da_int ldd, double coef0) {
    tanh_kernel_scalar(first_dim, second_dim, data, ldd, coef0);
}
template <>
void tanh_kernel<float, avx>(da_int first_dim, da_int second_dim, float *data, da_int ldd,
                             float coef0) {
    tanh_kernel_scalar(first_dim, second_dim, data, ldd, coef0);
}
template <>
void tanh_kernel<double, avx>(da_int first_dim, da_int second_dim, double *data,
                              da_int ldd, double coef0) {
    tanh_kernel_scalar(first_dim, second_dim, data, ldd, coef0);
}
template <>
void tanh_kernel<float, avx2>(da_int first_dim, da_int second_dim, float *data,
                              da_int ldd, float coef0) {
    tanh_kernel_scalar(first_dim, second_dim, data, ldd, coef0);
}
template <>
void tanh_kernel<double, avx2>(da_int first_dim, da_int second_dim, double *data,
                               da_int ldd, double coef0) {
    tanh_kernel_scalar(first_dim, second_dim, data, ldd, coef0);
}
template <>
void tanh_kernel<float, avx512>(da_int first_dim, da_int second_dim, float *data,
                                da_int ldd, float coef0) {
    tanh_kernel_scalar(first_dim, second_dim, data, ldd, coef0);
}
template <>
void tanh_kernel<double, avx512>(da_int first_dim, da_int second_dim, double *data,
                                 da_int ldd, double coef0) {
    tanh_kernel_scalar(first_dim, second_dim, data, ldd, coef0);
}

} // namespace da_kernel_functions

} // namespace ARCH