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
#include "macros.h"
#include "svm.hpp"
#include <array>
namespace ARCH {

namespace da_svm {

/* These functions contain performance-critical loops which must vectorize for performance. */

/*-------------------------------------------------------------
  ---------------------------  WSSI --------------------------- 
  ------------------------------------------------------------- */

template <typename T>
void wssi_kernel_scalar(da_int *I_up, T *gradient, da_int &min_grad_idx,
                        T &min_grad_value, da_int ws_size) {
    for (da_int iter = 0; iter < ws_size; iter++) {
        if (I_up[iter] && gradient[iter] < min_grad_value) {
            min_grad_value = gradient[iter];
            min_grad_idx = iter;
        }
    }
}

template <>
void wssi_kernel<float, vectorization_type::scalar>(da_int *I_up, float *gradient,
                                                    da_int &min_grad_idx,
                                                    float &min_grad_value,
                                                    da_int ws_size) {
    wssi_kernel_scalar(I_up, gradient, min_grad_idx, min_grad_value, ws_size);
}

template <>
void wssi_kernel<double, vectorization_type::scalar>(da_int *I_up, double *gradient,
                                                     da_int &min_grad_idx,
                                                     double &min_grad_value,
                                                     da_int ws_size) {
    wssi_kernel_scalar(I_up, gradient, min_grad_idx, min_grad_value, ws_size);
}

// LCOV_EXCL_START
template <>
void wssi_kernel<double, vectorization_type::avx>(da_int *I_up, double *gradient,
                                                  da_int &min_grad_idx,
                                                  double &min_grad_value,
                                                  da_int ws_size) {
    v2df_t v_smallest_gradients;
    v2i64_t v_smallest_idx;
    // Initialise vectors with the initial min values
    v_smallest_gradients.v = _mm_set1_pd(min_grad_value);
    v_smallest_idx.v = _mm_set1_epi64x(min_grad_idx);
    // I_up and gradient arrays are expected to be padded at this point
    for (da_int iter = 0; iter < ws_size; iter += 2) {
        // Load the I_up condition
        __m128i i_up_mask = _mm_set_epi64x(I_up[iter + 1], I_up[iter]);
        // Load the gradients
        __m128d gradient_v = _mm_loadu_pd(&gradient[iter]);
        // Compare gradients with the smallest gradients and store result in cmp_mask
        __m128d cmp_mask = _mm_cmp_pd(gradient_v, v_smallest_gradients.v, _CMP_LT_OQ);
        // Combine I_up condition with the comparison mask
        __m128d combined_mask = _mm_and_pd(cmp_mask, _mm_castsi128_pd(i_up_mask));
        // If any of the gradients are smaller, update the smallest gradients and indices
        if (_mm_movemask_pd(combined_mask)) {
            v_smallest_gradients.v =
                _mm_blendv_pd(v_smallest_gradients.v, gradient_v, combined_mask);
            v_smallest_idx.v =
                _mm_blendv_epi8(v_smallest_idx.v, _mm_set_epi64x(iter + 1, iter),
                                _mm_castpd_si128(combined_mask));
        }
    }
    // Finish early if no valid indices found
    std::array<da_int, 2> smallest_idx({static_cast<da_int>(v_smallest_idx.i[0]),
                                        static_cast<da_int>(v_smallest_idx.i[1])});
    if (*std::max_element(smallest_idx.begin(), smallest_idx.end()) == -1)
        return;
    // Reduction while ensuring consistency with scalar version
    // (If both values are equal, choose the one with the smaller index)
    min_grad_value = v_smallest_gradients.d[0];
    min_grad_idx = v_smallest_idx.i[0];
    if (v_smallest_gradients.d[0] == v_smallest_gradients.d[1]) {
        if (v_smallest_idx.i[1] < v_smallest_idx.i[0]) {
            min_grad_value = v_smallest_gradients.d[1];
            min_grad_idx = v_smallest_idx.i[1];
        }
    } else if (v_smallest_gradients.d[1] < v_smallest_gradients.d[0]) {
        min_grad_value = v_smallest_gradients.d[1];
        min_grad_idx = v_smallest_idx.i[1];
    }
}

template <>
void wssi_kernel<float, vectorization_type::avx>(da_int *I_up, float *gradient,
                                                 da_int &min_grad_idx,
                                                 float &min_grad_value, da_int ws_size) {
    v4sf_t v_smallest_gradients;
#if defined(AOCLDA_ILP64)
    // Because we are dealing with 64 bit integers, we need to use two integer vectors for the labels
    v2i64_t v_smallest_idx1, v_smallest_idx2;
    v_smallest_idx1.v = _mm_set1_epi64x(min_grad_idx);
    v_smallest_idx2.v = _mm_set1_epi64x(min_grad_idx);
#else
    v4i32_t v_smallest_idx;
    v_smallest_idx.v = _mm_set1_epi32(min_grad_idx);
#endif
    v_smallest_gradients.v = _mm_set1_ps(min_grad_value);
    // I_up and gradient arrays are expected to be padded at this point
    for (da_int iter = 0; iter < ws_size; iter += 4) {
        __m128 gradient_v = _mm_loadu_ps(&gradient[iter]);
        __m128 cmp_mask = _mm_cmp_ps(gradient_v, v_smallest_gradients.v, _CMP_LT_OQ);

        // Combine I_up condition with the cmp_mask
        __m128i i_up_mask =
            _mm_set_epi32(I_up[iter + 3], I_up[iter + 2], I_up[iter + 1], I_up[iter]);
        __m128 combined_mask = _mm_and_ps(cmp_mask, _mm_castsi128_ps(i_up_mask));

        // If any of the gradients are smaller, update the smallest gradients and indices
        if (_mm_movemask_ps(combined_mask)) {
            v_smallest_gradients.v =
                _mm_blendv_ps(v_smallest_gradients.v, gradient_v, combined_mask);
#if defined(AOCLDA_ILP64)
            __m128 lower_mask = _mm_permute_ps(combined_mask, _MM_SHUFFLE(1, 1, 0, 0));
            __m128 upper_mask = _mm_permute_ps(combined_mask, _MM_SHUFFLE(3, 3, 2, 2));
            v_smallest_idx1.v =
                _mm_blendv_epi8(v_smallest_idx1.v, _mm_set_epi64x(iter + 1, iter),
                                _mm_castps_si128(lower_mask));
            v_smallest_idx2.v =
                _mm_blendv_epi8(v_smallest_idx2.v, _mm_set_epi64x(iter + 3, iter + 2),
                                _mm_castps_si128(upper_mask));
#else
            v_smallest_idx.v = _mm_blendv_epi8(
                v_smallest_idx.v, _mm_set_epi32(iter + 3, iter + 2, iter + 1, iter),
                _mm_castps_si128(combined_mask));
#endif
        }
    }

#if defined(AOCLDA_ILP64)
    std::array<da_int, 4> smallest_idx({v_smallest_idx1.i[0], v_smallest_idx1.i[1],
                                        v_smallest_idx2.i[0], v_smallest_idx2.i[1]});
#else
    std::array<da_int, 4> smallest_idx({v_smallest_idx.i[0], v_smallest_idx.i[1],
                                        v_smallest_idx.i[2], v_smallest_idx.i[3]});
#endif
    if (*std::max_element(smallest_idx.begin(), smallest_idx.end()) == -1)
        return;
    // Reduction while ensuring consistency with scalar version
    // If both values are equal, choose the one with the smaller index
    min_grad_value = v_smallest_gradients.f[0];
    min_grad_idx = smallest_idx[0];
    for (da_int i = 1; i <= 3; i++) {
        if (v_smallest_gradients.f[i] == min_grad_value) {
            if (smallest_idx[i] < min_grad_idx) {
                min_grad_value = v_smallest_gradients.f[i];
                min_grad_idx = smallest_idx[i];
            }
        } else if (v_smallest_gradients.f[i] < min_grad_value) {
            min_grad_value = v_smallest_gradients.f[i];
            min_grad_idx = smallest_idx[i];
        }
    }
}

template <>
void wssi_kernel<double, vectorization_type::avx2>(da_int *I_up, double *gradient,
                                                   da_int &min_grad_idx,
                                                   double &min_grad_value,
                                                   da_int ws_size) {
    v4df_t v_smallest_gradients;
    v4i64_t v_smallest_idx;
    // Initialise vectors with the initial min values
    v_smallest_gradients.v = _mm256_set1_pd(min_grad_value);
    v_smallest_idx.v = _mm256_set1_epi64x(min_grad_idx);
    // I_up and gradient arrays are expected to be padded at this point
    for (da_int iter = 0; iter < ws_size; iter += 4) {
        // Load the I_up condition
        __m256i i_up_mask =
            _mm256_set_epi64x(I_up[iter + 3], I_up[iter + 2], I_up[iter + 1], I_up[iter]);
        // Load the gradients
        __m256d gradient_v = _mm256_loadu_pd(&gradient[iter]);
        // Compare gradients with the smallest gradients and store result in cmp_mask
        __m256d cmp_mask = _mm256_cmp_pd(gradient_v, v_smallest_gradients.v, _CMP_LT_OQ);
        // Combine I_up condition with the comparison mask
        __m256d combined_mask = _mm256_and_pd(cmp_mask, _mm256_castsi256_pd(i_up_mask));
        // If any of the gradients are smaller, update the smallest gradients and indices
        if (_mm256_movemask_pd(combined_mask)) {
            v_smallest_gradients.v =
                _mm256_blendv_pd(v_smallest_gradients.v, gradient_v, combined_mask);
            v_smallest_idx.v = _mm256_blendv_epi8(
                v_smallest_idx.v, _mm256_set_epi64x(iter + 3, iter + 2, iter + 1, iter),
                _mm256_castpd_si256(combined_mask));
        }
    }
    // Finish early if no valid indices found
    std::array<da_int, 4> smallest_idx({static_cast<da_int>(v_smallest_idx.i[0]),
                                        static_cast<da_int>(v_smallest_idx.i[1]),
                                        static_cast<da_int>(v_smallest_idx.i[2]),
                                        static_cast<da_int>(v_smallest_idx.i[3])});
    if (*std::max_element(smallest_idx.begin(), smallest_idx.end()) == -1)
        return;

    // Reduction while ensuring consistency with scalar version
    // If both values are equal, choose the one with the smaller index
    min_grad_value = v_smallest_gradients.d[0];
    min_grad_idx = v_smallest_idx.i[0];
    for (da_int i = 1; i <= 3; i++) {
        if (v_smallest_gradients.d[i] == v_smallest_gradients.d[0]) {
            if (v_smallest_idx.i[i] < v_smallest_idx.i[0]) {
                min_grad_value = v_smallest_gradients.d[i];
                v_smallest_idx.i[0] = v_smallest_idx.i[i];
                min_grad_idx = v_smallest_idx.i[i];
            }
        } else if (v_smallest_gradients.d[i] < v_smallest_gradients.d[0]) {
            v_smallest_gradients.d[0] = v_smallest_gradients.d[i];
            min_grad_value = v_smallest_gradients.d[i];
            v_smallest_idx.i[0] = v_smallest_idx.i[i];
            min_grad_idx = v_smallest_idx.i[i];
        }
    }
}

template <>
void wssi_kernel<float, vectorization_type::avx2>(da_int *I_up, float *gradient,
                                                  da_int &min_grad_idx,
                                                  float &min_grad_value, da_int ws_size) {
    v8sf_t v_smallest_gradients;
#if defined(AOCLDA_ILP64)
    // Because we are dealing with 64 bit integers, we need to use two integer vectors for the labels
    v4i64_t v_smallest_idx1, v_smallest_idx2;
    v_smallest_idx1.v = _mm256_set1_epi64x(min_grad_idx);
    v_smallest_idx2.v = _mm256_set1_epi64x(min_grad_idx);
#else
    v8i32_t v_smallest_idx;
    v_smallest_idx.v = _mm256_set1_epi32(min_grad_idx);
#endif
    v_smallest_gradients.v = _mm256_set1_ps(min_grad_value);
    // I_up and gradient arrays are expected to be padded at this point
    for (da_int iter = 0; iter < ws_size; iter += 8) {
        __m256 gradient_v = _mm256_loadu_ps(&gradient[iter]);
        __m256 cmp_mask = _mm256_cmp_ps(gradient_v, v_smallest_gradients.v, _CMP_LT_OQ);

        // Combine I_up condition with the comparison mask
        __m256i i_up_mask = _mm256_set_epi32(
            I_up[iter + 7], I_up[iter + 6], I_up[iter + 5], I_up[iter + 4],
            I_up[iter + 3], I_up[iter + 2], I_up[iter + 1], I_up[iter]);
        __m256 combined_mask = _mm256_and_ps(cmp_mask, _mm256_castsi256_ps(i_up_mask));
        if (_mm256_movemask_ps(combined_mask)) {
            v_smallest_gradients.v =
                _mm256_blendv_ps(v_smallest_gradients.v, gradient_v, combined_mask);
#if defined(AOCLDA_ILP64)
            __m256i control_mask_lower = _mm256_set_epi32(5, 5, 4, 4, 1, 1, 0, 0);
            __m256i control_mask_upper = _mm256_set_epi32(7, 7, 6, 6, 3, 3, 2, 2);
            __m256 lower_mask =
                _mm256_permutevar8x32_ps(combined_mask, control_mask_lower);
            __m256 upper_mask =
                _mm256_permutevar8x32_ps(combined_mask, control_mask_upper);
            v_smallest_idx1.v = _mm256_blendv_epi8(
                v_smallest_idx1.v, _mm256_set_epi64x(iter + 5, iter + 4, iter + 1, iter),
                _mm256_castps_si256(lower_mask));
            v_smallest_idx2.v = _mm256_blendv_epi8(
                v_smallest_idx2.v,
                _mm256_set_epi64x(iter + 7, iter + 6, iter + 3, iter + 2),
                _mm256_castps_si256(upper_mask));
#else
            v_smallest_idx.v = _mm256_blendv_epi8(
                v_smallest_idx.v,
                _mm256_set_epi32(iter + 7, iter + 6, iter + 5, iter + 4, iter + 3,
                                 iter + 2, iter + 1, iter),
                _mm256_castps_si256(combined_mask));
#endif
        }
    }

#if defined(AOCLDA_ILP64)
    std::array<da_int, 8> smallest_idx({v_smallest_idx1.i[0], v_smallest_idx1.i[1],
                                        v_smallest_idx2.i[0], v_smallest_idx2.i[1],
                                        v_smallest_idx1.i[2], v_smallest_idx1.i[3],
                                        v_smallest_idx2.i[2], v_smallest_idx2.i[3]});
#else
    std::array<da_int, 8> smallest_idx({v_smallest_idx.i[0], v_smallest_idx.i[1],
                                        v_smallest_idx.i[2], v_smallest_idx.i[3],
                                        v_smallest_idx.i[4], v_smallest_idx.i[5],
                                        v_smallest_idx.i[6], v_smallest_idx.i[7]});
#endif
    if (*std::max_element(smallest_idx.begin(), smallest_idx.end()) == -1)
        return;
    // Reduction while ensuring consistency with scalar version
    // If both values are equal, choose the one with the smaller index
    min_grad_value = v_smallest_gradients.f[0];
    min_grad_idx = smallest_idx[0];
    for (da_int i = 1; i <= 7; i++) {
        if (v_smallest_gradients.f[i] == min_grad_value) {
            if (smallest_idx[i] < min_grad_idx) {
                min_grad_value = v_smallest_gradients.f[i];
                min_grad_idx = smallest_idx[i];
            }
        } else if (v_smallest_gradients.f[i] < min_grad_value) {
            min_grad_value = v_smallest_gradients.f[i];
            min_grad_idx = smallest_idx[i];
        }
    }
}

#ifdef __AVX512F__
template <>
void wssi_kernel<double, vectorization_type::avx512>(da_int *I_up, double *gradient,
                                                     da_int &min_grad_idx,
                                                     double &min_grad_value,
                                                     da_int ws_size) {
    v8df_t v_smallest_gradients;
    v8i64_t v_smallest_idx;
    v_smallest_gradients.v = _mm512_set1_pd(min_grad_value);
    v_smallest_idx.v = _mm512_set1_epi64(min_grad_idx);
    for (da_int iter = 0; iter < ws_size; iter += 8) {
// Create __mmask8 type from vector of bools I_up
#ifdef AOCLDA_ILP64
        __m512i i_up =
            _mm512_loadu_si512(reinterpret_cast<const __m512i_u *>(&I_up[iter]));
        __mmask8 i_up_mask =
            _mm512_cmp_epi64_mask(i_up, _mm512_setzero_si512(), _MM_CMPINT_NE);
#else
        __m256i i_up =
            _mm256_loadu_si256(reinterpret_cast<const __m256i_u *>(&I_up[iter]));
        __mmask8 i_up_mask =
            _mm256_cmp_epi32_mask(i_up, _mm256_setzero_si256(), _MM_CMPINT_NE);
#endif
        // Load gradient and do comparison
        __m512d gradient_v = _mm512_loadu_pd(&gradient[iter]);
        __mmask8 cmp_mask =
            _mm512_cmp_pd_mask(gradient_v, v_smallest_gradients.v, _CMP_LT_OQ);
        // Combine I_up condition with the comparison mask
        __mmask8 combined_mask = i_up_mask & cmp_mask;

        if (combined_mask) {
            v_smallest_gradients.v =
                _mm512_mask_blend_pd(combined_mask, v_smallest_gradients.v, gradient_v);
            v_smallest_idx.v = _mm512_mask_blend_epi64(
                combined_mask, v_smallest_idx.v,
                _mm512_set_epi64(iter + 7, iter + 6, iter + 5, iter + 4, iter + 3,
                                 iter + 2, iter + 1, iter));
        }
    }

    // Finish early if no valid indices found
    if (_mm512_reduce_max_epi64(v_smallest_idx.v) == -1)
        return;

    // Reduction while ensuring consistency with scalar version
    // If both values are equal, choose the one with the smaller index
    min_grad_value = v_smallest_gradients.d[0];
    min_grad_idx = v_smallest_idx.i[0];
    for (da_int i = 1; i <= 7; i++) {
        if (v_smallest_gradients.d[i] == v_smallest_gradients.d[0]) {
            if (v_smallest_idx.i[i] < v_smallest_idx.i[0]) {
                min_grad_value = v_smallest_gradients.d[i];
                v_smallest_idx.i[0] = v_smallest_idx.i[i];
                min_grad_idx = v_smallest_idx.i[i];
            }
        } else if (v_smallest_gradients.d[i] < v_smallest_gradients.d[0]) {
            v_smallest_gradients.d[0] = v_smallest_gradients.d[i];
            min_grad_value = v_smallest_gradients.d[i];
            v_smallest_idx.i[0] = v_smallest_idx.i[i];
            min_grad_idx = v_smallest_idx.i[i];
        }
    }
}

template <>
void wssi_kernel<float, vectorization_type::avx512>(da_int *I_up, float *gradient,
                                                    da_int &min_grad_idx,
                                                    float &min_grad_value,
                                                    da_int ws_size) {
    v16sf_t v_smallest_gradients;
#if defined(AOCLDA_ILP64)
    // Because we are dealing with 64 bit integers, we need to use two integer vectors for the labels
    v8i64_t v_smallest_idx1, v_smallest_idx2;
    v_smallest_idx1.v = _mm512_set1_epi64(min_grad_idx);
    v_smallest_idx2.v = _mm512_set1_epi64(min_grad_idx);
#else
    v16i32_t v_smallest_idx;
    v_smallest_idx.v = _mm512_set1_epi32(min_grad_idx);
#endif
    v_smallest_gradients.v = _mm512_set1_ps(min_grad_value);
    // I_up and gradient arrays are expected to be padded at this point
    for (da_int iter = 0; iter < ws_size; iter += 16) {
        __m512 gradient_v = _mm512_loadu_ps(&gradient[iter]);
        __mmask16 cmp_mask =
            _mm512_cmp_ps_mask(gradient_v, v_smallest_gradients.v, _CMP_LT_OQ);

        // Combine I_up condition with the comparison mask
        // Create __mmask16 type from vector of bools I_up
#if defined(AOCLDA_ILP64)
        __m512i i_up_lower = _mm512_loadu_si512(&I_up[iter]);
        __m512i i_up_upper = _mm512_loadu_si512(&I_up[iter + 8]);
        __mmask8 i_up_mask_lower =
            _mm512_cmp_epi64_mask(i_up_lower, _mm512_setzero_si512(), _MM_CMPINT_NE);
        __mmask8 i_up_mask_upper =
            _mm512_cmp_epi64_mask(i_up_upper, _mm512_setzero_si512(), _MM_CMPINT_NE);
        __mmask16 i_up_mask = i_up_mask_lower | (i_up_mask_upper << 8);
#else
        __m512i i_up = _mm512_loadu_si512(&I_up[iter]);
        __mmask16 i_up_mask =
            _mm512_cmp_epi32_mask(i_up, _mm512_setzero_si512(), _MM_CMPINT_NE);
#endif
        // Combine I_up condition with the comparison mask
        __mmask16 combined_mask = i_up_mask & cmp_mask;

        // If any of the gradients are smaller, update the smallest gradients and indices
        if (combined_mask) {
            v_smallest_gradients.v =
                _mm512_mask_blend_ps(combined_mask, v_smallest_gradients.v, gradient_v);
#if defined(AOCLDA_ILP64)
            v_smallest_idx1.v = _mm512_mask_blend_epi64(
                combined_mask & 0xFF, v_smallest_idx1.v,
                _mm512_set_epi64(iter + 7, iter + 6, iter + 5, iter + 4, iter + 3,
                                 iter + 2, iter + 1, iter));
            v_smallest_idx2.v = _mm512_mask_blend_epi64(
                combined_mask >> 8, v_smallest_idx2.v,
                _mm512_set_epi64(iter + 15, iter + 14, iter + 13, iter + 12, iter + 11,
                                 iter + 10, iter + 9, iter + 8));
#else
            v_smallest_idx.v = _mm512_mask_blend_epi32(
                combined_mask, v_smallest_idx.v,
                _mm512_set_epi32(iter + 15, iter + 14, iter + 13, iter + 12, iter + 11,
                                 iter + 10, iter + 9, iter + 8, iter + 7, iter + 6,
                                 iter + 5, iter + 4, iter + 3, iter + 2, iter + 1, iter));
#endif
        }
    }

#if defined(AOCLDA_ILP64)
    if (_mm512_reduce_max_epi64(v_smallest_idx1.v) == -1 &&
        _mm512_reduce_max_epi64(v_smallest_idx2.v) == -1)
        return;
    std::array<da_int, 16> smallest_idx(
        {v_smallest_idx1.i[0], v_smallest_idx1.i[1], v_smallest_idx1.i[2],
         v_smallest_idx1.i[3], v_smallest_idx1.i[4], v_smallest_idx1.i[5],
         v_smallest_idx1.i[6], v_smallest_idx1.i[7], v_smallest_idx2.i[0],
         v_smallest_idx2.i[1], v_smallest_idx2.i[2], v_smallest_idx2.i[3],
         v_smallest_idx2.i[4], v_smallest_idx2.i[5], v_smallest_idx2.i[6],
         v_smallest_idx2.i[7]});
#else
    if (_mm512_reduce_max_epi32(v_smallest_idx.v) == -1)
        return;
    std::array<da_int, 16> smallest_idx(
        {v_smallest_idx.i[0], v_smallest_idx.i[1], v_smallest_idx.i[2],
         v_smallest_idx.i[3], v_smallest_idx.i[4], v_smallest_idx.i[5],
         v_smallest_idx.i[6], v_smallest_idx.i[7], v_smallest_idx.i[8],
         v_smallest_idx.i[9], v_smallest_idx.i[10], v_smallest_idx.i[11],
         v_smallest_idx.i[12], v_smallest_idx.i[13], v_smallest_idx.i[14],
         v_smallest_idx.i[15]});
#endif
    // Reduction while ensuring consistency with scalar version
    // If both values are equal, choose the one with the smaller index
    min_grad_value = v_smallest_gradients.f[0];
    min_grad_idx = smallest_idx[0];
    for (da_int i = 1; i <= 15; i++) {
        if (v_smallest_gradients.f[i] == min_grad_value) {
            if (smallest_idx[i] < min_grad_idx) {
                min_grad_value = v_smallest_gradients.f[i];
                min_grad_idx = smallest_idx[i];
            }
        } else if (v_smallest_gradients.f[i] < min_grad_value) {
            min_grad_value = v_smallest_gradients.f[i];
            min_grad_idx = smallest_idx[i];
        }
    }
}
#endif

/*-------------------------------------------------------------
  ---------------------------  WSSJ --------------------------- 
  ------------------------------------------------------------- */

template <typename T>
void wssj_kernel_scalar(da_int *I_low, T *gradient, T *K_ith_row, T *K_diagonal, T &K_ii,
                        da_int &max_grad_idx, T &max_grad_value, T &min_grad, T &max_fun,
                        T &delta, T &tau, da_int ws_size) {
    for (da_int iter = 0; iter < ws_size; iter++) {
        if (!I_low[iter])
            continue; // Skip non-I_low elements early

        T current_gradient = gradient[iter];
        max_grad_value = std::max(max_grad_value, current_gradient);

        // Calculate gradient difference
        T b = current_gradient - min_grad;
        if (b < 0)
            continue; // Skip if b is negative

        T a = K_ii + K_diagonal[iter] - 2 * K_ith_row[iter];

        // Ensure a is positive
        if (a <= 0)
            a = tau;

        T ratio = b / a;
        T function_val = ratio * b;

        if (function_val > max_fun) {
            max_fun = function_val;
            max_grad_idx = iter;
            delta = ratio;
        }
    }
}

template <>
void wssj_kernel<float, vectorization_type::scalar>(
    da_int *I_low, float *gradient, float *K_ith_row, float *K_diagonal, float &K_ii,
    da_int &max_grad_idx, float &max_grad_value, float &min_grad, float &max_fun,
    float &delta, float &tau, da_int ws_size) {
    wssj_kernel_scalar(I_low, gradient, K_ith_row, K_diagonal, K_ii, max_grad_idx,
                       max_grad_value, min_grad, max_fun, delta, tau, ws_size);
}

template <>
void wssj_kernel<double, vectorization_type::scalar>(
    da_int *I_low, double *gradient, double *K_ith_row, double *K_diagonal, double &K_ii,
    da_int &max_grad_idx, double &max_grad_value, double &min_grad, double &max_fun,
    double &delta, double &tau, da_int ws_size) {
    wssj_kernel_scalar(I_low, gradient, K_ith_row, K_diagonal, K_ii, max_grad_idx,
                       max_grad_value, min_grad, max_fun, delta, tau, ws_size);
}

template <>
void wssj_kernel<double, vectorization_type::avx>(
    da_int *I_low, double *gradient, double *K_ith_row, double *K_diagonal, double &K_ii,
    da_int &max_grad_idx, double &max_grad_value, double &min_grad, double &max_fun,
    double &delta, double &tau, da_int ws_size) {
    v2df_t v_largest_gradients, v_largest_fun, v_delta;
    v2i64_t v_largest_idx;
    // Initialise vectors with the initial values
    v_largest_gradients.v = _mm_set1_pd(max_grad_value);
    v_largest_fun.v = _mm_set1_pd(max_fun);
    v_largest_idx.v = _mm_set1_epi64x(max_grad_idx);
    v_delta.v = _mm_set1_pd(delta);
    // I_low and gradient arrays are expected to be padded at this point
    for (da_int iter = 0; iter < ws_size; iter += 2) {
        // Load the I_low condition
        __m128i i_low_mask = _mm_set_epi64x(I_low[iter + 1], I_low[iter]);
        // Load the gradients
        __m128d gradient_v = _mm_loadu_pd(&gradient[iter]);
        // Compare gradients with the largest gradients and store result in cmp_mask
        __m128d cmp_mask = _mm_cmp_pd(gradient_v, v_largest_gradients.v, _CMP_GT_OQ);
        // Combine I_low condition with the comparison mask
        __m128d max_grad_mask = _mm_and_pd(cmp_mask, _mm_castsi128_pd(i_low_mask));
        // Update largest gradients
        v_largest_gradients.v =
            _mm_blendv_pd(v_largest_gradients.v, gradient_v, max_grad_mask);
        // Calculate the gradient difference
        __m128d b = _mm_sub_pd(gradient_v, _mm_set1_pd(min_grad));
        // Create a mask for positive b values
        __m128d positive_b_mask = _mm_cmp_pd(b, _mm_setzero_pd(), _CMP_GT_OQ);
        __m128d combined_ilow_posb =
            _mm_and_pd(_mm_castsi128_pd(i_low_mask), positive_b_mask);
        // Calculate a
        __m128d a = _mm_add_pd(_mm_set1_pd(K_ii), _mm_loadu_pd(&K_diagonal[iter]));
        a = _mm_sub_pd(a, _mm_mul_pd(_mm_set1_pd(2.0), _mm_loadu_pd(&K_ith_row[iter])));
        // When a is negative or zero, set it to tau
        __m128d negative_a_mask = _mm_cmp_pd(a, _mm_setzero_pd(), _CMP_LE_OQ);
        a = _mm_blendv_pd(a, _mm_set1_pd(tau), negative_a_mask);
        // Calculate ratio
        __m128d ratio = _mm_div_pd(b, a);
        // Calculate function value
        __m128d function_val = _mm_mul_pd(ratio, b);
        // Create a mask for function value larger than max_fun
        __m128d max_fun_mask = _mm_cmp_pd(function_val, v_largest_fun.v, _CMP_GT_OQ);
        // Combine the masks
        __m128d combined_all = _mm_and_pd(max_fun_mask, combined_ilow_posb);
        // If any of the gradients are larger, update the largest gradients and indices
        v_largest_fun.v = _mm_blendv_pd(v_largest_fun.v, function_val, combined_all);
        v_largest_idx.v = _mm_blendv_epi8(v_largest_idx.v, _mm_set_epi64x(iter + 1, iter),
                                          _mm_castpd_si128(combined_all));
        v_delta.v = _mm_blendv_pd(v_delta.v, ratio, combined_all);
    }
    // Pick max_grad_value
    max_grad_value = v_largest_gradients.d[0];
    if (v_largest_gradients.d[1] > max_grad_value) {
        max_grad_value = v_largest_gradients.d[1];
    }
    // Finish early if no valid indices found
    std::array<da_int, 2> largest_idx = {static_cast<da_int>(v_largest_idx.i[0]),
                                         static_cast<da_int>(v_largest_idx.i[1])};
    if (*std::max_element(largest_idx.begin(), largest_idx.end()) == -1)
        return;
    // Reduction while ensuring consistency with scalar version
    // If both values are equal, choose the one with the smaller index
    max_grad_idx = v_largest_idx.i[0];
    max_fun = v_largest_fun.d[0];
    delta = v_delta.d[0];
    if (v_largest_fun.d[1] == max_fun) {
        if (v_largest_idx.i[1] < max_grad_idx) {
            max_grad_idx = v_largest_idx.i[1];
            max_fun = v_largest_fun.d[1];
            delta = v_delta.d[1];
        }
    } else if (v_largest_fun.d[1] > max_fun) {
        max_grad_idx = v_largest_idx.i[1];
        max_fun = v_largest_fun.d[1];
        delta = v_delta.d[1];
    }
}

template <>
void wssj_kernel<float, vectorization_type::avx>(da_int *I_low, float *gradient,
                                                 float *K_ith_row, float *K_diagonal,
                                                 float &K_ii, da_int &max_grad_idx,
                                                 float &max_grad_value, float &min_grad,
                                                 float &max_fun, float &delta, float &tau,
                                                 da_int ws_size) {
    v4sf_t v_largest_gradients, v_largest_fun, v_delta;
#if defined(AOCLDA_ILP64)
    // Because we are dealing with 64 bit integers, we need to use two integer vectors for the labels
    v2i64_t v_largest_idx1, v_largest_idx2;
    v_largest_idx1.v = _mm_set1_epi64x(max_grad_idx);
    v_largest_idx2.v = _mm_set1_epi64x(max_grad_idx);
#else
    v4i32_t v_largest_idx;
    v_largest_idx.v = _mm_set1_epi32(max_grad_idx);
#endif
    // Initialise vectors with the initial values
    v_largest_gradients.v = _mm_set1_ps(max_grad_value);
    v_largest_fun.v = _mm_set1_ps(max_fun);
    v_delta.v = _mm_set1_ps(delta);
    // I_low and gradient arrays are expected to be padded at this point
    for (da_int iter = 0; iter < ws_size; iter += 4) {
        // Load the gradients
        __m128 gradient_v = _mm_loadu_ps(&gradient[iter]);
        // Compare gradients with the largest gradients and store result in cmp_mask
        __m128 cmp_mask = _mm_cmp_ps(gradient_v, v_largest_gradients.v, _CMP_GT_OQ);
        // Combine I_low condition with the cmp_mask into max_grad_mask
        __m128i i_low_mask =
            _mm_set_epi32(I_low[iter + 3], I_low[iter + 2], I_low[iter + 1], I_low[iter]);
        __m128 max_grad_mask = _mm_and_ps(cmp_mask, _mm_castsi128_ps(i_low_mask));
        // Update largest gradients
        v_largest_gradients.v =
            _mm_blendv_ps(v_largest_gradients.v, gradient_v, max_grad_mask);
        // Calculate the gradient difference
        __m128 b = _mm_sub_ps(gradient_v, _mm_set1_ps(min_grad));
        // Create a mask for positive b values
        __m128 positive_b_mask = _mm_cmp_ps(b, _mm_setzero_ps(), _CMP_GT_OQ);
        // Combine I_low condition with the positive b mask
        __m128 combined_ilow_posb =
            _mm_and_ps(_mm_castsi128_ps(i_low_mask), positive_b_mask);
        // Calculate a
        __m128 a = _mm_add_ps(_mm_set1_ps(K_ii), _mm_loadu_ps(&K_diagonal[iter]));
        a = _mm_sub_ps(a, _mm_mul_ps(_mm_set1_ps(2.0f), _mm_loadu_ps(&K_ith_row[iter])));
        // When a is negative or zero, set it to tau
        __m128 negative_a_mask = _mm_cmp_ps(a, _mm_setzero_ps(), _CMP_LE_OQ);
        a = _mm_blendv_ps(a, _mm_set1_ps(tau), negative_a_mask);
        // Calculate ratio
        __m128 ratio = _mm_div_ps(b, a);
        // Calculate function value
        __m128 function_val = _mm_mul_ps(ratio, b);
        // Create a mask for function value larger than max_fun
        __m128 max_fun_mask = _mm_cmp_ps(function_val, v_largest_fun.v, _CMP_GT_OQ);
        // Combine the masks
        __m128 combined_all = _mm_and_ps(max_fun_mask, combined_ilow_posb);
        // If any of the gradients are larger, update the largest gradients and indices
        v_largest_fun.v = _mm_blendv_ps(v_largest_fun.v, function_val, combined_all);
        v_delta.v = _mm_blendv_ps(v_delta.v, ratio, combined_all);
#if defined(AOCLDA_ILP64)
        __m128 lower_mask = _mm_permute_ps(combined_all, _MM_SHUFFLE(1, 1, 0, 0));
        __m128 upper_mask = _mm_permute_ps(combined_all, _MM_SHUFFLE(3, 3, 2, 2));
        v_largest_idx1.v =
            _mm_blendv_epi8(v_largest_idx1.v, _mm_set_epi64x(iter + 1, iter),
                            _mm_castps_si128(lower_mask));
        v_largest_idx2.v =
            _mm_blendv_epi8(v_largest_idx2.v, _mm_set_epi64x(iter + 3, iter + 2),
                            _mm_castps_si128(upper_mask));
#else
        v_largest_idx.v = _mm_blendv_epi8(
            v_largest_idx.v, _mm_set_epi32(iter + 3, iter + 2, iter + 1, iter),
            _mm_castps_si128(combined_all));
#endif
    }
    // Pick max_grad_value
    max_grad_value = v_largest_gradients.f[0];
    for (da_int i = 1; i <= 3; i++) {
        if (v_largest_gradients.f[i] > max_grad_value) {
            max_grad_value = v_largest_gradients.f[i];
        }
    }
#if defined(AOCLDA_ILP64)
    std::array<da_int, 4> largest_idx({v_largest_idx1.i[0], v_largest_idx1.i[1],
                                       v_largest_idx2.i[0], v_largest_idx2.i[1]});
#else
    std::array<da_int, 4> largest_idx(
        {v_largest_idx.i[0], v_largest_idx.i[1], v_largest_idx.i[2], v_largest_idx.i[3]});
#endif
    if (*std::max_element(largest_idx.begin(), largest_idx.end()) == -1)
        return;
    max_grad_idx = largest_idx[0];
    max_fun = v_largest_fun.f[0];
    delta = v_delta.f[0];
    for (da_int i = 1; i <= 3; i++) {
        if (v_largest_fun.f[i] == max_fun) {
            if (largest_idx[i] < max_grad_idx) {
                max_grad_idx = largest_idx[i];
                max_fun = v_largest_fun.f[i];
                delta = v_delta.f[i];
            }
        } else if (v_largest_fun.f[i] > max_fun) {
            max_grad_idx = largest_idx[i];
            max_fun = v_largest_fun.f[i];
            delta = v_delta.f[i];
        }
    }
}

template <>
void wssj_kernel<double, vectorization_type::avx2>(
    da_int *I_low, double *gradient, double *K_ith_row, double *K_diagonal, double &K_ii,
    da_int &max_grad_idx, double &max_grad_value, double &min_grad, double &max_fun,
    double &delta, double &tau, da_int ws_size) {
    v4df_t v_largest_gradients, v_largest_fun, v_delta;
    v4i64_t v_largest_idx;
    // Initialise vectors with the initial values
    v_largest_gradients.v = _mm256_set1_pd(max_grad_value);
    v_largest_fun.v = _mm256_set1_pd(max_fun);
    v_largest_idx.v = _mm256_set1_epi64x(max_grad_idx);
    v_delta.v = _mm256_set1_pd(delta);
    // I_low and gradient arrays are expected to be padded at this point
    for (da_int iter = 0; iter < ws_size; iter += 4) {
        // Load the I_low condition
        __m256i i_low_mask = _mm256_set_epi64x(I_low[iter + 3], I_low[iter + 2],
                                               I_low[iter + 1], I_low[iter]);
        // Load the gradients
        __m256d gradient_v = _mm256_loadu_pd(&gradient[iter]);
        // Compare gradients with the largest gradients and store result in cmp_mask
        __m256d cmp_mask = _mm256_cmp_pd(gradient_v, v_largest_gradients.v, _CMP_GT_OQ);
        // Combine I_low condition with the comparison mask
        __m256d max_grad_mask = _mm256_and_pd(cmp_mask, _mm256_castsi256_pd(i_low_mask));
        // Update largest gradients
        v_largest_gradients.v =
            _mm256_blendv_pd(v_largest_gradients.v, gradient_v, max_grad_mask);
        // Calculate the gradient difference
        __m256d b = _mm256_sub_pd(gradient_v, _mm256_set1_pd(min_grad));
        // Create a mask for positive b values
        __m256d positive_b_mask = _mm256_cmp_pd(b, _mm256_setzero_pd(), _CMP_GT_OQ);
        __m256d combined_ilow_posb =
            _mm256_and_pd(_mm256_castsi256_pd(i_low_mask), positive_b_mask);
        // Calculate a
        __m256d a =
            _mm256_add_pd(_mm256_set1_pd(K_ii), _mm256_loadu_pd(&K_diagonal[iter]));
        a = _mm256_sub_pd(
            a, _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_loadu_pd(&K_ith_row[iter])));
        // When a is negative or zero, set it to tau
        __m256d negative_a_mask = _mm256_cmp_pd(a, _mm256_setzero_pd(), _CMP_LE_OQ);
        a = _mm256_blendv_pd(a, _mm256_set1_pd(tau), negative_a_mask);
        // Calculate ratio
        __m256d ratio = _mm256_div_pd(b, a);
        // Calculate function value
        __m256d function_val = _mm256_mul_pd(ratio, b);
        // Create a mask for function value larger than max_fun
        __m256d max_fun_mask = _mm256_cmp_pd(function_val, v_largest_fun.v, _CMP_GT_OQ);
        // Combine the masks
        __m256d combined_all = _mm256_and_pd(max_fun_mask, combined_ilow_posb);
        // If any of the gradients are larger, update the largest gradients and indices
        v_largest_fun.v = _mm256_blendv_pd(v_largest_fun.v, function_val, combined_all);
        v_largest_idx.v = _mm256_blendv_epi8(
            v_largest_idx.v, _mm256_set_epi64x(iter + 3, iter + 2, iter + 1, iter),
            _mm256_castpd_si256(combined_all));
        v_delta.v = _mm256_blendv_pd(v_delta.v, ratio, combined_all);
    }
    // Pick max_grad_value
    max_grad_value = v_largest_gradients.d[0];
    for (da_int i = 1; i <= 3; i++) {
        if (v_largest_gradients.d[i] > max_grad_value) {
            max_grad_value = v_largest_gradients.d[i];
        }
    }
    // Finish early if no valid indices found
    std::array<da_int, 4> largest_idx({static_cast<da_int>(v_largest_idx.i[0]),
                                       static_cast<da_int>(v_largest_idx.i[1]),
                                       static_cast<da_int>(v_largest_idx.i[2]),
                                       static_cast<da_int>(v_largest_idx.i[3])});
    if (*std::max_element(largest_idx.begin(), largest_idx.end()) == -1)
        return;
    // Reduction while ensuring consistency with scalar version
    // If both values are equal, choose the one with the smaller index
    max_grad_idx = v_largest_idx.i[0];
    max_fun = v_largest_fun.d[0];
    delta = v_delta.d[0];
    for (da_int i = 1; i <= 3; i++) {
        if (v_largest_fun.d[i] == max_fun) {
            if (v_largest_idx.i[i] < max_grad_idx) {
                max_grad_idx = v_largest_idx.i[i];
                max_fun = v_largest_fun.d[i];
                delta = v_delta.d[i];
            }
        } else if (v_largest_fun.d[i] > max_fun) {
            max_grad_idx = v_largest_idx.i[i];
            max_fun = v_largest_fun.d[i];
            delta = v_delta.d[i];
        }
    }
}

template <>
void wssj_kernel<float, vectorization_type::avx2>(da_int *I_low, float *gradient,
                                                  float *K_ith_row, float *K_diagonal,
                                                  float &K_ii, da_int &max_grad_idx,
                                                  float &max_grad_value, float &min_grad,
                                                  float &max_fun, float &delta,
                                                  float &tau, da_int ws_size) {
    v8sf_t v_largest_gradients, v_largest_fun, v_delta;
#if defined(AOCLDA_ILP64)
    // Because we are dealing with 64 bit integers, we need to use two integer vectors
    v4i64_t v_largest_idx1, v_largest_idx2;
    v_largest_idx1.v = _mm256_set1_epi64x(max_grad_idx);
    v_largest_idx2.v = _mm256_set1_epi64x(max_grad_idx);
#else
    v8i32_t v_largest_idx;
    v_largest_idx.v = _mm256_set1_epi32(max_grad_idx);
#endif
    // Initialise vectors with the initial values
    v_largest_gradients.v = _mm256_set1_ps(max_grad_value);
    v_largest_fun.v = _mm256_set1_ps(max_fun);
    v_delta.v = _mm256_set1_ps(delta);
    // I_low and gradient arrays are expected to be padded at this point
    for (da_int iter = 0; iter < ws_size; iter += 8) {
        // Load the gradients
        __m256 gradient_v = _mm256_loadu_ps(&gradient[iter]);
        // Compare gradients with the largest gradients and store result in cmp_mask
        __m256 cmp_mask = _mm256_cmp_ps(gradient_v, v_largest_gradients.v, _CMP_GT_OQ);
        // Combine I_low condition with the cmp_mask into max_grad_mask
        __m256i i_low_mask = _mm256_set_epi32(
            I_low[iter + 7], I_low[iter + 6], I_low[iter + 5], I_low[iter + 4],
            I_low[iter + 3], I_low[iter + 2], I_low[iter + 1], I_low[iter]);
        __m256 max_grad_mask = _mm256_and_ps(cmp_mask, _mm256_castsi256_ps(i_low_mask));
        // Update largest gradients
        v_largest_gradients.v =
            _mm256_blendv_ps(v_largest_gradients.v, gradient_v, max_grad_mask);
        // Calculate the gradient difference
        __m256 b = _mm256_sub_ps(gradient_v, _mm256_set1_ps(min_grad));
        // Create a mask for positive b values
        __m256 positive_b_mask = _mm256_cmp_ps(b, _mm256_setzero_ps(), _CMP_GT_OQ);
        // Combine I_low condition with the positive b mask
        __m256 combined_ilow_posb =
            _mm256_and_ps(_mm256_castsi256_ps(i_low_mask), positive_b_mask);
        // Calculate a
        __m256 a =
            _mm256_add_ps(_mm256_set1_ps(K_ii), _mm256_loadu_ps(&K_diagonal[iter]));
        a = _mm256_sub_ps(
            a, _mm256_mul_ps(_mm256_set1_ps(2.0f), _mm256_loadu_ps(&K_ith_row[iter])));
        // When a is negative or zero, set it to tau
        __m256 negative_a_mask = _mm256_cmp_ps(a, _mm256_setzero_ps(), _CMP_LE_OQ);
        a = _mm256_blendv_ps(a, _mm256_set1_ps(tau), negative_a_mask);
        // Calculate ratio
        __m256 ratio = _mm256_div_ps(b, a);
        // Calculate function value
        __m256 function_val = _mm256_mul_ps(ratio, b);
        // Create a mask for function value larger than max_fun
        __m256 max_fun_mask = _mm256_cmp_ps(function_val, v_largest_fun.v, _CMP_GT_OQ);
        // Combine the masks
        __m256 combined_all = _mm256_and_ps(max_fun_mask, combined_ilow_posb);
        // If any of the gradients are larger, update the largest gradients and indices
        v_largest_fun.v = _mm256_blendv_ps(v_largest_fun.v, function_val, combined_all);
        v_delta.v = _mm256_blendv_ps(v_delta.v, ratio, combined_all);
#if defined(AOCLDA_ILP64)
        __m256i control_mask_lower = _mm256_set_epi32(5, 5, 4, 4, 1, 1, 0, 0);
        __m256i control_mask_upper = _mm256_set_epi32(7, 7, 6, 6, 3, 3, 2, 2);
        __m256 lower_mask = _mm256_permutevar8x32_ps(combined_all, control_mask_lower);
        __m256 upper_mask = _mm256_permutevar8x32_ps(combined_all, control_mask_upper);
        v_largest_idx1.v = _mm256_blendv_epi8(
            v_largest_idx1.v, _mm256_set_epi64x(iter + 5, iter + 4, iter + 1, iter),
            _mm256_castps_si256(lower_mask));
        v_largest_idx2.v = _mm256_blendv_epi8(
            v_largest_idx2.v, _mm256_set_epi64x(iter + 7, iter + 6, iter + 3, iter + 2),
            _mm256_castps_si256(upper_mask));
#else
        v_largest_idx.v =
            _mm256_blendv_epi8(v_largest_idx.v,
                               _mm256_set_epi32(iter + 7, iter + 6, iter + 5, iter + 4,
                                                iter + 3, iter + 2, iter + 1, iter),
                               _mm256_castps_si256(combined_all));
#endif
    }
    // Pick max_grad_value
    max_grad_value = v_largest_gradients.f[0];
    for (da_int i = 1; i <= 7; i++)
        if (v_largest_gradients.f[i] > max_grad_value)
            max_grad_value = v_largest_gradients.f[i];
#if defined(AOCLDA_ILP64)
    std::array<da_int, 8> largest_idx({v_largest_idx1.i[0], v_largest_idx1.i[1],
                                       v_largest_idx2.i[0], v_largest_idx2.i[1],
                                       v_largest_idx1.i[2], v_largest_idx1.i[3],
                                       v_largest_idx2.i[2], v_largest_idx2.i[3]});
#else
    std::array<da_int, 8> largest_idx(
        {v_largest_idx.i[0], v_largest_idx.i[1], v_largest_idx.i[2], v_largest_idx.i[3],
         v_largest_idx.i[4], v_largest_idx.i[5], v_largest_idx.i[6], v_largest_idx.i[7]});
#endif
    if (*std::max_element(largest_idx.begin(), largest_idx.end()) == -1)
        return;
    max_grad_idx = largest_idx[0];
    max_fun = v_largest_fun.f[0];
    delta = v_delta.f[0];
    for (da_int i = 1; i <= 7; i++) {
        if (v_largest_fun.f[i] == max_fun) {
            if (largest_idx[i] < max_grad_idx) {
                max_grad_idx = largest_idx[i];
                max_fun = v_largest_fun.f[i];
                delta = v_delta.f[i];
            }
        } else if (v_largest_fun.f[i] > max_fun) {
            max_grad_idx = largest_idx[i];
            max_fun = v_largest_fun.f[i];
            delta = v_delta.f[i];
        }
    }
}

#ifdef __AVX512F__

template <>
void wssj_kernel<double, vectorization_type::avx512>(
    da_int *I_low, double *gradient, double *K_ith_row, double *K_diagonal, double &K_ii,
    da_int &max_grad_idx, double &max_grad_value, double &min_grad, double &max_fun,
    double &delta, double &tau, da_int ws_size) {
    v8df_t v_largest_gradients, v_largest_fun, v_delta;
    v8i64_t v_largest_idx;
    // Initialise vectors with the initial values
    v_largest_gradients.v = _mm512_set1_pd(max_grad_value);
    v_largest_fun.v = _mm512_set1_pd(max_fun);
    v_largest_idx.v = _mm512_set1_epi64(max_grad_idx);
    v_delta.v = _mm512_set1_pd(delta);
    // I_low and gradient arrays are expected to be padded at this point
    for (da_int iter = 0; iter < ws_size; iter += 8) {
// Create __mmask8 type from vector of chars I_low
#if defined(AOCLDA_ILP64)
        __m512i i_low =
            _mm512_loadu_si512(reinterpret_cast<const __m512i_u *>(&I_low[iter]));
        __mmask8 i_low_mask =
            _mm512_cmp_epi64_mask(i_low, _mm512_setzero_si512(), _MM_CMPINT_NE);
#else
        __m256i i_low =
            _mm256_loadu_si256(reinterpret_cast<const __m256i_u *>(&I_low[iter]));
        __mmask8 i_low_mask =
            _mm256_cmp_epi32_mask(i_low, _mm256_setzero_si256(), _MM_CMPINT_NE);
#endif
        // Load the gradients
        __m512d gradient_v = _mm512_loadu_pd(&gradient[iter]);
        // Compare gradients with the largest gradients and store result in cmp_mask
        __mmask8 cmp_mask =
            _mm512_cmp_pd_mask(gradient_v, v_largest_gradients.v, _CMP_GT_OQ);
        // Combine I_low condition with the comparison mask
        __mmask8 max_grad_mask = i_low_mask & cmp_mask;
        // Update largest gradients
        v_largest_gradients.v =
            _mm512_mask_blend_pd(max_grad_mask, v_largest_gradients.v, gradient_v);
        // Calculate the gradient difference
        __m512d b = _mm512_sub_pd(gradient_v, _mm512_set1_pd(min_grad));
        // Create a mask for positive b values
        __mmask8 positive_b_mask = _mm512_cmp_pd_mask(b, _mm512_setzero_pd(), _CMP_GT_OQ);
        // Combine I_low condition with the positive b mask
        __mmask8 combined_ilow_posb = i_low_mask & positive_b_mask;
        // Calculate a
        __m512d a =
            _mm512_add_pd(_mm512_set1_pd(K_ii), _mm512_loadu_pd(&K_diagonal[iter]));
        a = _mm512_sub_pd(
            a, _mm512_mul_pd(_mm512_set1_pd(2.0), _mm512_loadu_pd(&K_ith_row[iter])));
        // When a is negative or zero, set it to tau
        __mmask8 negative_a_mask = _mm512_cmp_pd_mask(a, _mm512_setzero_pd(), _CMP_LE_OQ);
        a = _mm512_mask_blend_pd(negative_a_mask, a, _mm512_set1_pd(tau));
        // Calculate ratio
        __m512d ratio = _mm512_div_pd(b, a);
        // Calculate function value
        __m512d function_val = _mm512_mul_pd(ratio, b);
        // Create a mask for function value larger than max_fun
        __mmask8 max_fun_mask =
            _mm512_cmp_pd_mask(function_val, v_largest_fun.v, _CMP_GT_OQ);
        // Combine the masks
        __mmask8 combined_all = max_fun_mask & combined_ilow_posb;
        // If any of the gradients are larger, update the largest gradients and indices
        v_largest_fun.v =
            _mm512_mask_blend_pd(combined_all, v_largest_fun.v, function_val);
        v_largest_idx.v = _mm512_mask_blend_epi64(
            combined_all, v_largest_idx.v,
            _mm512_set_epi64(iter + 7, iter + 6, iter + 5, iter + 4, iter + 3, iter + 2,
                             iter + 1, iter));
        v_delta.v = _mm512_mask_blend_pd(combined_all, v_delta.v, ratio);
    }
    // Pick max_grad_value
    max_grad_value = v_largest_gradients.d[0];
    for (da_int i = 1; i <= 7; i++) {
        if (v_largest_gradients.d[i] > max_grad_value) {
            max_grad_value = v_largest_gradients.d[i];
        }
    }
    // Finish early if no valid indices found
    if (_mm512_reduce_max_epi64(v_largest_idx.v) == -1)
        return;
    // Reduction while ensuring consistency with scalar version
    // If both values are equal, choose the one with the smaller index
    max_grad_idx = v_largest_idx.i[0];
    max_fun = v_largest_fun.d[0];
    delta = v_delta.d[0];
    for (da_int i = 1; i <= 7; i++) {
        if (v_largest_fun.d[i] == max_fun) {
            if (v_largest_idx.i[i] < max_grad_idx) {
                max_grad_idx = v_largest_idx.i[i];
                max_fun = v_largest_fun.d[i];
                delta = v_delta.d[i];
            }
        } else if (v_largest_fun.d[i] > max_fun) {
            max_grad_idx = v_largest_idx.i[i];
            max_fun = v_largest_fun.d[i];
            delta = v_delta.d[i];
        }
    }
}

template <>
void wssj_kernel<float, vectorization_type::avx512>(
    da_int *I_low, float *gradient, float *K_ith_row, float *K_diagonal, float &K_ii,
    da_int &max_grad_idx, float &max_grad_value, float &min_grad, float &max_fun,
    float &delta, float &tau, da_int ws_size) {
    v16sf_t v_largest_gradients, v_largest_fun, v_delta;
#if defined(AOCLDA_ILP64)
    // Because we are dealing with 64 bit integers, we need to use two integer vectors
    v8i64_t v_largest_idx1, v_largest_idx2;
    v_largest_idx1.v = _mm512_set1_epi64(max_grad_idx);
    v_largest_idx2.v = _mm512_set1_epi64(max_grad_idx);
#else
    v16i32_t v_largest_idx;
    v_largest_idx.v = _mm512_set1_epi32(max_grad_idx);
#endif
    // Initialise vectors with the initial values
    v_largest_gradients.v = _mm512_set1_ps(max_grad_value);
    v_largest_fun.v = _mm512_set1_ps(max_fun);
    v_delta.v = _mm512_set1_ps(delta);
    // I_low and gradient arrays are expected to be padded at this point
    for (da_int iter = 0; iter < ws_size; iter += 16) {
        // Load the gradients
        __m512 gradient_v = _mm512_loadu_ps(&gradient[iter]);
        // Compare gradients with the largest gradients and store result in cmp_mask
        __mmask16 cmp_mask =
            _mm512_cmp_ps_mask(gradient_v, v_largest_gradients.v, _CMP_GT_OQ);
// Combine I_low condition with the comparison mask
// Create __mmask16 type from vector of chars I_low
#if defined(AOCLDA_ILP64)
        __m512i i_low_lower = _mm512_loadu_si512(&I_low[iter]);
        __m512i i_low_upper = _mm512_loadu_si512(&I_low[iter + 8]);
        __mmask8 i_low_mask_lower =
            _mm512_cmp_epi64_mask(i_low_lower, _mm512_setzero_si512(), _MM_CMPINT_NE);
        __mmask8 i_low_mask_upper =
            _mm512_cmp_epi64_mask(i_low_upper, _mm512_setzero_si512(), _MM_CMPINT_NE);
        __mmask16 i_low_mask = i_low_mask_lower | (i_low_mask_upper << 8);
#else
        __m512i i_low = _mm512_loadu_si512(&I_low[iter]);
        __mmask16 i_low_mask =
            _mm512_cmp_epi32_mask(i_low, _mm512_setzero_si512(), _MM_CMPINT_NE);
#endif
        __mmask16 max_grad_mask = i_low_mask & cmp_mask;
        // Update largest gradients
        v_largest_gradients.v =
            _mm512_mask_blend_ps(max_grad_mask, v_largest_gradients.v, gradient_v);
        // Calculate the gradient difference
        __m512 b = _mm512_sub_ps(gradient_v, _mm512_set1_ps(min_grad));
        // Create a mask for positive b values
        __mmask16 positive_b_mask =
            _mm512_cmp_ps_mask(b, _mm512_setzero_ps(), _CMP_GT_OQ);
        // Combine I_low condition with the positive b mask
        __mmask16 combined_ilow_posb = i_low_mask & positive_b_mask;
        // Calculate a
        __m512 a =
            _mm512_add_ps(_mm512_set1_ps(K_ii), _mm512_loadu_ps(&K_diagonal[iter]));
        a = _mm512_sub_ps(
            a, _mm512_mul_ps(_mm512_set1_ps(2.0f), _mm512_loadu_ps(&K_ith_row[iter])));
        // When a is negative or zero, set it to tau
        __mmask16 negative_a_mask =
            _mm512_cmp_ps_mask(a, _mm512_setzero_ps(), _CMP_LE_OQ);
        a = _mm512_mask_blend_ps(negative_a_mask, a, _mm512_set1_ps(tau));
        // Calculate ratio
        __m512 ratio = _mm512_div_ps(b, a);
        // Calculate function value
        __m512 function_val = _mm512_mul_ps(ratio, b);
        // Create a mask for function value larger than max_fun
        __mmask16 max_fun_mask =
            _mm512_cmp_ps_mask(function_val, v_largest_fun.v, _CMP_GT_OQ);
        // Combine the masks
        __mmask16 combined_all = max_fun_mask & combined_ilow_posb;
        // If any of the gradients are larger, update the largest gradients and indices
        v_largest_fun.v =
            _mm512_mask_blend_ps(combined_all, v_largest_fun.v, function_val);
        v_delta.v = _mm512_mask_blend_ps(combined_all, v_delta.v, ratio);
#if defined(AOCLDA_ILP64)
        v_largest_idx1.v = _mm512_mask_blend_epi64(
            combined_all & 0xFF, v_largest_idx1.v,
            _mm512_set_epi64(iter + 7, iter + 6, iter + 5, iter + 4, iter + 3, iter + 2,
                             iter + 1, iter));
        v_largest_idx2.v = _mm512_mask_blend_epi64(
            combined_all >> 8, v_largest_idx2.v,
            _mm512_set_epi64(iter + 15, iter + 14, iter + 13, iter + 12, iter + 11,
                             iter + 10, iter + 9, iter + 8));
#else
        v_largest_idx.v = _mm512_mask_blend_epi32(
            combined_all, v_largest_idx.v,
            _mm512_set_epi32(iter + 15, iter + 14, iter + 13, iter + 12, iter + 11,
                             iter + 10, iter + 9, iter + 8, iter + 7, iter + 6, iter + 5,
                             iter + 4, iter + 3, iter + 2, iter + 1, iter));
#endif
    }
    // Pick max_grad_value
    max_grad_value = v_largest_gradients.f[0];
    for (da_int i = 1; i <= 15; i++)
        if (v_largest_gradients.f[i] > max_grad_value)
            max_grad_value = v_largest_gradients.f[i];
#if defined(AOCLDA_ILP64)
    if (_mm512_reduce_max_epi64(v_largest_idx1.v) == -1 &&
        _mm512_reduce_max_epi64(v_largest_idx2.v) == -1)
        return;
    std::array<da_int, 16> largest_idx(
        {v_largest_idx1.i[0], v_largest_idx1.i[1], v_largest_idx1.i[2],
         v_largest_idx1.i[3], v_largest_idx1.i[4], v_largest_idx1.i[5],
         v_largest_idx1.i[6], v_largest_idx1.i[7], v_largest_idx2.i[0],
         v_largest_idx2.i[1], v_largest_idx2.i[2], v_largest_idx2.i[3],
         v_largest_idx2.i[4], v_largest_idx2.i[5], v_largest_idx2.i[6],
         v_largest_idx2.i[7]});
#else
    if (_mm512_reduce_max_epi32(v_largest_idx.v) == -1)
        return;
    std::array<da_int, 16> largest_idx(
        {v_largest_idx.i[0], v_largest_idx.i[1], v_largest_idx.i[2], v_largest_idx.i[3],
         v_largest_idx.i[4], v_largest_idx.i[5], v_largest_idx.i[6], v_largest_idx.i[7],
         v_largest_idx.i[8], v_largest_idx.i[9], v_largest_idx.i[10], v_largest_idx.i[11],
         v_largest_idx.i[12], v_largest_idx.i[13], v_largest_idx.i[14],
         v_largest_idx.i[15]});
#endif

    max_grad_idx = largest_idx[0];
    max_fun = v_largest_fun.f[0];
    delta = v_delta.f[0];
    for (da_int i = 1; i <= 15; i++) {
        if (v_largest_fun.f[i] == max_fun) {
            if (largest_idx[i] < max_grad_idx) {
                max_grad_idx = largest_idx[i];
                max_fun = v_largest_fun.f[i];
                delta = v_delta.f[i];
            }
        } else if (v_largest_fun.f[i] > max_fun) {
            max_grad_idx = largest_idx[i];
            max_fun = v_largest_fun.f[i];
            delta = v_delta.f[i];
        }
    }
}
#endif // __AVX512F__
} // namespace da_svm

} // namespace ARCH