/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
#include "fp16_helpers.hpp"
#include "immintrin.h"
#include "kernel_functions.hpp"
#include "kt.hpp"
#include "macros.h"
#include <cmath>
#include <type_traits>

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
                       T multiplier, const T *first_dim_norms,
                       const T *second_dim_norms) {
    for (da_int i = 0; i < second_dim; i++) {
        T *data_ptr = &data[i * ldd];
        T second_dim_norm = second_dim_norms[i];
        if constexpr (std::is_same_v<T, _Float16>) {
            // Promote to float for the transcendental and cast back; libm has no
            // native _Float16 overload of exp().
            for (da_int j = 0; j < first_dim; j++) {
                float x = static_cast<float>(multiplier) *
                          (static_cast<float>(data_ptr[j]) +
                           static_cast<float>(first_dim_norms[j]) +
                           static_cast<float>(second_dim_norm));
                data_ptr[j] = static_cast<T>(exp(x));
            }
        } else {
            for (da_int j = 0; j < first_dim; j++) {
                data_ptr[j] = exp(multiplier *
                                  (data_ptr[j] + first_dim_norms[j] + second_dim_norm));
            }
        }
    }
}

template <typename T>
void pow_kernel_scalar(da_int first_dim, da_int second_dim, T *data, da_int ldd, T coef0,
                       da_int degree) {
    // Integer power by repeated multiplication. Matches the SIMD path
    // bit-for-bit, and avoids std::pow which is slower than a handful of
    // multiplications for small degree.
    // Precondition: degree >= 1. All callers in the library (public
    // polynomial_kernel API and SVM polynomial option) validate this.
    for (da_int i = 0; i < second_dim; i++) {
        T *data_ptr = &data[i * ldd];
        for (da_int j = 0; j < first_dim; j++) {
            T base = data_ptr[j] + coef0;
            T result = base;
            for (da_int k = 1; k < degree; k++)
                result *= base;
            data_ptr[j] = result;
        }
    }
}

template <typename T>
void tanh_kernel_scalar(da_int first_dim, da_int second_dim, T *data, da_int ldd,
                        T coef0) {
    for (da_int i = 0; i < second_dim; i++) {
        T *data_ptr = &data[i * ldd];
        if constexpr (std::is_same_v<T, _Float16>) {
            // Promote to float for the transcendental and cast back; libm has no
            // native _Float16 overload of tanh().
            for (da_int j = 0; j < first_dim; j++) {
                float x = static_cast<float>(data_ptr[j]) + static_cast<float>(coef0);
                data_ptr[j] = static_cast<T>(tanh(x));
            }
        } else {
            for (da_int j = 0; j < first_dim; j++) {
                data_ptr[j] = tanh(data_ptr[j] + coef0);
            }
        }
    }
}

template <bsz SZ, typename SUF>
inline __attribute__((__always_inline__)) void
exp_kt(da_int first_dim, da_int second_dim, SUF *data, da_int ldd, SUF multiplier,
       const SUF *first_dim_norms, const SUF *second_dim_norms) {
    const da_int simd_length{tsz_v<SZ, SUF>};
    da_int remainder = first_dim % simd_length;
    da_int size = first_dim - remainder;
    da_int offset = 0;
    avxvector_t<SZ, SUF> v_multiplier = kt_set1_p<SZ>(multiplier);
    for (da_int i = 0; i < second_dim; i++) {
        avxvector_t<SZ, SUF> v_second_dim_norm = kt_set1_p<SZ>(second_dim_norms[i]);
        for (da_int j = 0; j < size; j += simd_length) {
            avxvector_t<SZ, SUF> v_data = kt_loadu_p<SZ>(&data[j + offset]);
            avxvector_t<SZ, SUF> v_first_dim_norms = kt_loadu_p<SZ>(&first_dim_norms[j]);
            // data + first_dim_norms + second_dim_norms
            v_data = kt_add_p<SZ, SUF>(v_data, v_first_dim_norms);
            v_data = kt_add_p<SZ, SUF>(v_data, v_second_dim_norm);
            // multiply by multiplier
            v_data = kt_mul_p<SZ, SUF>(v_data, v_multiplier);
            // apply exp
            v_data = kt_exp_p<SZ, SUF>(v_data);
            kt_storeu_p<SZ>(&data[j + offset], v_data);
        }
        // Handle the remaining elements
        SUF second_dim_norm = second_dim_norms[i];
        da_int idx = offset + size;
        for (da_int j = 0; j < remainder; j++) {
            if constexpr (std::is_same_v<SUF, _Float16>) {
                float x = static_cast<float>(multiplier) *
                          (static_cast<float>(data[idx]) +
                           static_cast<float>(first_dim_norms[j + size]) +
                           static_cast<float>(second_dim_norm));
                data[idx] = static_cast<SUF>(exp(x));
            } else {
                data[idx] = exp(multiplier * (data[idx] + first_dim_norms[j + size] +
                                              second_dim_norm));
            }
            idx++;
        }
        offset += ldd;
    }
}

template <bsz SZ, typename SUF>
inline __attribute__((__always_inline__)) void pow_kt(da_int first_dim, da_int second_dim,
                                                      SUF *data, da_int ldd, SUF coef0,
                                                      da_int degree) {
    // Precondition: degree >= 1. All callers in the library (public
    // polynomial_kernel API and SVM polynomial option) validate this.
    const da_int simd_length{tsz_v<SZ, SUF>};
    da_int remainder = first_dim % simd_length;
    da_int size = first_dim - remainder;
    da_int offset = 0;
    avxvector_t<SZ, SUF> v_coef0 = kt_set1_p<SZ>(coef0);
    for (da_int i = 0; i < second_dim; i++) {
        for (da_int j = 0; j < size; j += simd_length) {
            avxvector_t<SZ, SUF> v_data = kt_loadu_p<SZ>(&data[j + offset]);
            // add coef0
            v_data = kt_add_p<SZ, SUF>(v_data, v_coef0);
            // Compute integer power using repeated multiplication;
            // initialise the accumulator with the base to save one multiply.
            avxvector_t<SZ, SUF> v_result = v_data;
            for (da_int k = 1; k < degree; k++)
                v_result = kt_mul_p<SZ, SUF>(v_result, v_data);
            kt_storeu_p<SZ>(&data[j + offset], v_result);
        }
        // Handle the remaining elements (matches the SIMD body above:
        // integer power by repeated multiplication, no libm call).
        da_int idx = offset + size;
        for (da_int j = 0; j < remainder; j++) {
            SUF base = data[idx] + coef0;
            SUF result = base;
            for (da_int k = 1; k < degree; k++)
                result *= base;
            data[idx] = result;
            idx++;
        }
        offset += ldd;
    }
}

template <bsz SZ, typename SUF>
inline __attribute__((__always_inline__)) void
tanh_kt(da_int first_dim, da_int second_dim, SUF *data, da_int ldd, SUF coef0) {
    const da_int simd_length{tsz_v<SZ, SUF>};
    da_int remainder = first_dim % simd_length;
    da_int size = first_dim - remainder;
    da_int offset = 0;
    avxvector_t<SZ, SUF> v_coef0 = kt_set1_p<SZ>(coef0);

    avxvector_t<SZ, SUF> v_two = kt_set1_p<SZ>(static_cast<SUF>(2.0));
    avxvector_t<SZ, SUF> v_one = kt_set1_p<SZ>(static_cast<SUF>(1.0));
    avxvector_t<SZ, SUF> v_neg_two = kt_set1_p<SZ>(static_cast<SUF>(-2.0));

    for (da_int i = 0; i < second_dim; i++) {
        for (da_int j = 0; j < size; j += simd_length) {
            avxvector_t<SZ, SUF> v_data = kt_loadu_p<SZ>(&data[j + offset]);
            // add coef0
            v_data = kt_add_p<SZ, SUF>(v_data, v_coef0);

            // Numerically stable approach: tanh(x) = 2/(1 + e^(-2x)) - 1
            // large positive x, e^(-2x) approaches 0
            // large negative x, approaches infinity in denominator, but we avoid problems since 2/inf=0
            avxvector_t<SZ, SUF> v_neg_2x = kt_mul_p<SZ, SUF>(v_data, v_neg_two);
            avxvector_t<SZ, SUF> v_exp_neg_2x = kt_exp_p<SZ, SUF>(v_neg_2x);
            avxvector_t<SZ, SUF> v_denominator = kt_add_p<SZ, SUF>(v_one, v_exp_neg_2x);
            avxvector_t<SZ, SUF> v_result = kt_div_p<SZ, SUF>(v_two, v_denominator);
            v_result = kt_sub_p<SZ, SUF>(v_result, v_one);
            kt_storeu_p<SZ>(&data[j + offset], v_result);
        }
        // Handle the remaining elements
        da_int idx = offset + size;
        for (da_int j = 0; j < remainder; j++) {
            if constexpr (std::is_same_v<SUF, _Float16>) {
                float x = static_cast<float>(data[idx]) + static_cast<float>(coef0);
                data[idx] = static_cast<SUF>(tanh(x));
            } else {
                data[idx] = tanh((data[idx]) + coef0);
            }
            idx++;
        }
        offset += ldd;
    }
}

// _Float16 instantiations of exp_kt, pow_kt and tanh_kt (below, guarded by
// __AVX512FP16__) go through the primary templates: the kt FP16 extension
// layer (kt_fp16.hpp) provides _Float16 specializations of every kt_*
// primitive used here, including kt_exp_p<bsz::*, _Float16> which promotes
// to FP32, calls the existing kt_exp_p<bsz::*, float> kernel, and rounds
// back to FP16.

#ifndef USE_SCALAR_MATH // Compiler macro defined in the CMake
// Single set of instantiations for the detected compiler
template void exp_kt<bsz::b128, float>(da_int first_dim, da_int second_dim, float *data,
                                       da_int ldd, float multiplier,
                                       const float *first_dim_norms,
                                       const float *second_dim_norms);
template void exp_kt<bsz::b128, double>(da_int first_dim, da_int second_dim, double *data,
                                        da_int ldd, double multiplier,
                                        const double *first_dim_norms,
                                        const double *second_dim_norms);
template void exp_kt<bsz::b256, float>(da_int first_dim, da_int second_dim, float *data,
                                       da_int ldd, float multiplier,
                                       const float *first_dim_norms,
                                       const float *second_dim_norms);
template void exp_kt<bsz::b256, double>(da_int first_dim, da_int second_dim, double *data,
                                        da_int ldd, double multiplier,
                                        const double *first_dim_norms,
                                        const double *second_dim_norms);
template void pow_kt<bsz::b128, float>(da_int first_dim, da_int second_dim, float *data,
                                       da_int ldd, float coef0, da_int degree);
template void pow_kt<bsz::b128, double>(da_int first_dim, da_int second_dim, double *data,
                                        da_int ldd, double coef0, da_int degree);
template void pow_kt<bsz::b256, float>(da_int first_dim, da_int second_dim, float *data,
                                       da_int ldd, float coef0, da_int degree);
template void pow_kt<bsz::b256, double>(da_int first_dim, da_int second_dim, double *data,
                                        da_int ldd, double coef0, da_int degree);
template void tanh_kt<bsz::b128, float>(da_int first_dim, da_int second_dim, float *data,
                                        da_int ldd, float coef0);
template void tanh_kt<bsz::b128, double>(da_int first_dim, da_int second_dim,
                                         double *data, da_int ldd, double coef0);
template void tanh_kt<bsz::b256, float>(da_int first_dim, da_int second_dim, float *data,
                                        da_int ldd, float coef0);
template void tanh_kt<bsz::b256, double>(da_int first_dim, da_int second_dim,
                                         double *data, da_int ldd, double coef0);
#ifdef __AVX512F__
template void exp_kt<bsz::b512, float>(da_int first_dim, da_int second_dim, float *data,
                                       da_int ldd, float multiplier,
                                       const float *first_dim_norms,
                                       const float *second_dim_norms);
template void exp_kt<bsz::b512, double>(da_int first_dim, da_int second_dim, double *data,
                                        da_int ldd, double multiplier,
                                        const double *first_dim_norms,
                                        const double *second_dim_norms);
template void pow_kt<bsz::b512, float>(da_int first_dim, da_int second_dim, float *data,
                                       da_int ldd, float coef0, da_int degree);
template void pow_kt<bsz::b512, double>(da_int first_dim, da_int second_dim, double *data,
                                        da_int ldd, double coef0, da_int degree);
template void tanh_kt<bsz::b512, float>(da_int first_dim, da_int second_dim, float *data,
                                        da_int ldd, float coef0);
template void tanh_kt<bsz::b512, double>(da_int first_dim, da_int second_dim,
                                         double *data, da_int ldd, double coef0);
#endif // __AVX512F__
#ifdef __AVX512FP16__
template void exp_kt<bsz::b128, _Float16>(da_int first_dim, da_int second_dim,
                                          _Float16 *data, da_int ldd, _Float16 multiplier,
                                          const _Float16 *first_dim_norms,
                                          const _Float16 *second_dim_norms);
template void exp_kt<bsz::b256, _Float16>(da_int first_dim, da_int second_dim,
                                          _Float16 *data, da_int ldd, _Float16 multiplier,
                                          const _Float16 *first_dim_norms,
                                          const _Float16 *second_dim_norms);
template void exp_kt<bsz::b512, _Float16>(da_int first_dim, da_int second_dim,
                                          _Float16 *data, da_int ldd, _Float16 multiplier,
                                          const _Float16 *first_dim_norms,
                                          const _Float16 *second_dim_norms);
template void pow_kt<bsz::b128, _Float16>(da_int first_dim, da_int second_dim,
                                          _Float16 *data, da_int ldd, _Float16 coef0,
                                          da_int degree);
template void pow_kt<bsz::b256, _Float16>(da_int first_dim, da_int second_dim,
                                          _Float16 *data, da_int ldd, _Float16 coef0,
                                          da_int degree);
template void pow_kt<bsz::b512, _Float16>(da_int first_dim, da_int second_dim,
                                          _Float16 *data, da_int ldd, _Float16 coef0,
                                          da_int degree);
template void tanh_kt<bsz::b128, _Float16>(da_int first_dim, da_int second_dim,
                                           _Float16 *data, da_int ldd, _Float16 coef0);
template void tanh_kt<bsz::b256, _Float16>(da_int first_dim, da_int second_dim,
                                           _Float16 *data, da_int ldd, _Float16 coef0);
template void tanh_kt<bsz::b512, _Float16>(da_int first_dim, da_int second_dim,
                                           _Float16 *data, da_int ldd, _Float16 coef0);
#endif // __AVX512FP16__
#endif // USE_SCALAR_MATH

// Explicit instantiations for the scalar kernel functions so their symbols
// are available to other translation units (e.g. kf_exp_implementations table).
template void exp_kernel_scalar<float>(da_int, da_int, float *, da_int, float,
                                       const float *, const float *);
template void exp_kernel_scalar<double>(da_int, da_int, double *, da_int, double,
                                        const double *, const double *);
template void pow_kernel_scalar<float>(da_int, da_int, float *, da_int, float, da_int);
template void pow_kernel_scalar<double>(da_int, da_int, double *, da_int, double, da_int);
template void tanh_kernel_scalar<float>(da_int, da_int, float *, da_int, float);
template void tanh_kernel_scalar<double>(da_int, da_int, double *, da_int, double);
#ifdef __AVX512FP16__
template void exp_kernel_scalar<_Float16>(da_int, da_int, _Float16 *, da_int, _Float16,
                                          const _Float16 *, const _Float16 *);
template void pow_kernel_scalar<_Float16>(da_int, da_int, _Float16 *, da_int, _Float16,
                                          da_int);
template void tanh_kernel_scalar<_Float16>(da_int, da_int, _Float16 *, da_int, _Float16);
#endif

} // namespace da_kernel_functions

} // namespace ARCH
