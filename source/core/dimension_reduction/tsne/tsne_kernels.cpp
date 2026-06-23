/* ************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

#include "tsne_kernels.hpp"
#include "aoclda_types.h"
#include "da_kernel_utils.hpp"
#include "kt.hpp"
#include "macros.h"
#include "tsne.hpp"
#include <immintrin.h>

namespace ARCH {

namespace da_tsne {

using namespace kernel_templates;

/*******************************************************************************
 * 1. SCALAR IMPLEMENTATION
 ******************************************************************************/

template <typename T, int8_t D>
void attractive_forces_scalar_impl(const T *emb_i, const da_int *col_idx, const T *P_vals,
                                   const T *embedding, T exaggeration, da_int start,
                                   da_int end, T *grad_i) {
    for (da_int d = 0; d < D; ++d)
        grad_i[d] = T(0);
    for (da_int idx = start; idx < end; ++idx) {
        da_int j = col_idx[idx];
        T pij = P_vals[idx] * exaggeration;
        const T *emb_j = embedding + j * D;
        T dist2 = T(0);
        T diff[D];
        for (da_int d = 0; d < D; ++d) {
            diff[d] = emb_i[d] - emb_j[d];
            dist2 += diff[d] * diff[d];
        }
        T q = T(1) / (T(1) + dist2);
        for (da_int d = 0; d < D; ++d)
            grad_i[d] += pij * q * diff[d];
    }
}

/*******************************************************************************
 * 2. KT SINGLE-NEIGHBOR SIMD
 *
 * One neighbor per SIMD iteration.  Used for:
 *   d=2 double avx:  (2/2 doubles)
 *   d=2 float  avx:  (2/4 floats)
 *   d=3 double avx2: (3/4 doubles)
 *   d=3 float  avx:  (3/4 floats)
 ******************************************************************************/

template <bsz SZ, typename T, int8_t D>
inline __attribute__((__always_inline__)) void
attractive_forces_kt(const T *emb_i, const da_int *col_idx, const T *P_vals,
                     const T *embedding, T exaggeration, da_int start, da_int end,
                     T *grad_i) {
    constexpr da_int vec_len = static_cast<da_int>(tsz_v<SZ, T>);
    auto v_grad = kt_setzero_p<SZ, T>();
    avxvector_t<SZ, T> v_emb_i;
    if constexpr (D == vec_len)
        v_emb_i = kt_loadu_p<SZ, T>(emb_i);
    else
        v_emb_i = kt_maskz_set_p<SZ, T, kt_avxext::AVX2, D>(emb_i, da_int(0));
    auto v_one = kt_set1_p<SZ>(T(1));

    for (da_int idx = start; idx < end; ++idx) {
        da_int j = col_idx[idx];
        const T *emb_j = embedding + j * D;
        avxvector_t<SZ, T> v_emb_j;
        if constexpr (D == vec_len)
            v_emb_j = kt_loadu_p<SZ, T>(emb_j);
        else
            v_emb_j = kt_maskz_set_p<SZ, T, kt_avxext::AVX2, D>(emb_j, da_int(0));

        auto v_diff = kt_sub_p<SZ, T>(v_emb_i, v_emb_j);
        T dist2 = kt_hsum_p<SZ, T>(kt_pow2_p<SZ, T>(v_diff));
        auto v_pij_q = kt_mul_p<SZ, T>(
            kt_set1_p<SZ>(P_vals[idx] * exaggeration),
            kt_div_p<SZ, T>(v_one, kt_add_p<SZ, T>(v_one, kt_set1_p<SZ>(dist2))));
        v_grad = kt_fmadd_p<SZ, T>(v_pij_q, v_diff, v_grad);
    }

    T result[vec_len];
    kt_storeu_p<SZ>(result, v_grad);
    for (da_int d = 0; d < D; ++d)
        grad_i[d] = result[d];
}

/*******************************************************************************
 * 3. MULTI-NEIGHBOR d=2 SIMD
 *
 * Packs vec_len/2 neighbors per SIMD iteration for the d=2 case.
 * Used for:
 *   d=2 float  avx2:   bsz::b256 (4 neighbors/iter)
 *   d=2 double avx2:   bsz::b256 (2 neighbors/iter)
 *   d=2 float  avx512: bsz::b512 (8 neighbors/iter)
 *   d=2 double avx512: bsz::b512 (4 neighbors/iter)
 *
 * Two small helpers abstract the ISA-specific operations:
 *   swap_adjacent_p  — swaps adjacent (x,y) pairs for dist2 reduction
 *   hreduce_d2       — tree-reduces the wide accumulator to 2 values
 ******************************************************************************/

template <typename T>
inline __attribute__((__always_inline__)) avxvector_t<bsz::b256, T>
swap_adjacent_p(avxvector_t<bsz::b256, T> v) {
    if constexpr (std::is_same_v<T, float>)
        return _mm256_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    else
        return _mm256_shuffle_pd(v, v, 0x5);
}

template <typename T>
inline __attribute__((__always_inline__)) void hreduce_d2(avxvector_t<bsz::b256, T> v,
                                                          T *out) {
    if constexpr (std::is_same_v<T, double>) {
        __m128d v_lo = _mm256_castpd256_pd128(v);
        __m128d v_hi = _mm256_extractf128_pd(v, 1);
        _mm_storeu_pd(out, _mm_add_pd(v_lo, v_hi));
    } else {
        __m128 v_lo = _mm256_castps256_ps128(v);
        __m128 v_hi = _mm256_extractf128_ps(v, 1);
        __m128 v_sum = _mm_add_ps(v_lo, v_hi);
        __m128 v_final = _mm_add_ps(v_sum, _mm_movehl_ps(v_sum, v_sum));
        float result[4];
        _mm_storeu_ps(result, v_final);
        out[0] = result[0];
        out[1] = result[1];
    }
}

#ifdef __AVX512F__

template <typename T>
inline __attribute__((__always_inline__)) avxvector_t<bsz::b512, T>
swap_adjacent_p(avxvector_t<bsz::b512, T> v) {
    if constexpr (std::is_same_v<T, float>)
        return _mm512_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    else
        return _mm512_shuffle_pd(v, v, 0x55);
}

template <typename T>
inline __attribute__((__always_inline__)) void hreduce_d2(avxvector_t<bsz::b512, T> v,
                                                          T *out) {
    if constexpr (std::is_same_v<T, double>)
        hreduce_d2(_mm256_add_pd(_mm512_castpd512_pd256(v), _mm512_extractf64x4_pd(v, 1)),
                   out);
    else
        hreduce_d2(_mm256_add_ps(_mm512_castps512_ps256(v), _mm512_extractf32x8_ps(v, 1)),
                   out);
}

#endif // __AVX512F__

template <bsz SZ, typename T>
inline __attribute__((__always_inline__)) void
attractive_forces_multi_d2(const T *emb_i, const da_int *col_idx, const T *P_vals,
                           const T *embedding, T exaggeration, da_int start, da_int end,
                           T *grad_i) {
    constexpr da_int vec_len = static_cast<da_int>(tsz_v<SZ, T>);
    constexpr da_int N = vec_len / 2;

    auto v_grad = kt_setzero_p<SZ, T>();
    auto v_one = kt_set1_p<SZ>(T(1));
    T emb_i_x = emb_i[0];
    T emb_i_y = emb_i[1];

    alignas(64) T emb_i_buf[vec_len];
    for (da_int k = 0; k < N; ++k) {
        emb_i_buf[2 * k] = emb_i_x;
        emb_i_buf[2 * k + 1] = emb_i_y;
    }
    auto v_emb_i = kt_load_p<SZ, T>(emb_i_buf);

    da_int idx = start;
    for (; idx + N - 1 < end; idx += N) {
        alignas(64) T emb_buf[vec_len];
        alignas(64) T pij_buf[vec_len];
        for (da_int k = 0; k < N; ++k) {
            da_int j = col_idx[idx + k];
            emb_buf[2 * k] = embedding[j * 2];
            emb_buf[2 * k + 1] = embedding[j * 2 + 1];
            T pij = P_vals[idx + k] * exaggeration;
            pij_buf[2 * k] = pij;
            pij_buf[2 * k + 1] = pij;
        }
        auto v_emb_j = kt_load_p<SZ, T>(emb_buf);
        auto v_diff = kt_sub_p<SZ, T>(v_emb_i, v_emb_j);
        auto v_diff2 = kt_pow2_p<SZ, T>(v_diff);
        auto v_dist2 = kt_add_p<SZ, T>(v_diff2, swap_adjacent_p<T>(v_diff2));
        auto v_q = kt_div_p<SZ, T>(v_one, kt_add_p<SZ, T>(v_one, v_dist2));
        auto v_pij = kt_load_p<SZ, T>(pij_buf);
        v_grad = kt_fmadd_p<SZ, T>(kt_mul_p<SZ, T>(v_pij, v_q), v_diff, v_grad);
    }

    hreduce_d2<T>(v_grad, grad_i);

    T tail[2];
    attractive_forces_scalar_impl<T, 2>(emb_i, col_idx, P_vals, embedding, exaggeration,
                                        idx, end, tail);
    grad_i[0] += tail[0];
    grad_i[1] += tail[1];
}

// Explicit instantiations

template void attractive_forces_scalar_impl<float, 1>(const float *, const da_int *,
                                                      const float *, const float *, float,
                                                      da_int, da_int, float *);
template void attractive_forces_scalar_impl<double, 1>(const double *, const da_int *,
                                                       const double *, const double *,
                                                       double, da_int, da_int, double *);
template void attractive_forces_scalar_impl<float, 2>(const float *, const da_int *,
                                                      const float *, const float *, float,
                                                      da_int, da_int, float *);
template void attractive_forces_scalar_impl<double, 2>(const double *, const da_int *,
                                                       const double *, const double *,
                                                       double, da_int, da_int, double *);
template void attractive_forces_scalar_impl<float, 3>(const float *, const da_int *,
                                                      const float *, const float *, float,
                                                      da_int, da_int, float *);
template void attractive_forces_scalar_impl<double, 3>(const double *, const da_int *,
                                                       const double *, const double *,
                                                       double, da_int, da_int, double *);

#define ATTRACTIVE_FORCES_KT_INSTANTIATE(SZ, SUF, D)                                     \
    template void attractive_forces_kt<SZ, SUF, D>(const SUF *, const da_int *,          \
                                                   const SUF *, const SUF *, SUF,        \
                                                   da_int, da_int, SUF *);

DA_KT_INSTANTIATE_EXT(ATTRACTIVE_FORCES_KT_INSTANTIATE, bsz::b128, 1)
DA_KT_INSTANTIATE_EXT(ATTRACTIVE_FORCES_KT_INSTANTIATE, bsz::b128, 2)
ATTRACTIVE_FORCES_KT_INSTANTIATE(bsz::b128, float, 3)
// Do not instantiate attractive_forces<avx, double, d=3>
DA_KT_INSTANTIATE_EXT(ATTRACTIVE_FORCES_KT_INSTANTIATE, bsz::b256, 1)
DA_KT_INSTANTIATE_EXT(ATTRACTIVE_FORCES_KT_INSTANTIATE, bsz::b256, 2)
DA_KT_INSTANTIATE_EXT(ATTRACTIVE_FORCES_KT_INSTANTIATE, bsz::b256, 3)
#ifdef __AVX512F__
DA_KT_INSTANTIATE_EXT(ATTRACTIVE_FORCES_KT_INSTANTIATE, bsz::b512, 2)
DA_KT_INSTANTIATE_EXT(ATTRACTIVE_FORCES_KT_INSTANTIATE, bsz::b512, 3)
#endif

#define ATTRACTIVE_FORCES_MULTI_D2_INSTANTIATE(SZ, T)                                    \
    template void attractive_forces_multi_d2<SZ, T>(                                     \
        const T *emb_i, const da_int *col_idx, const T *P_vals, const T *embedding,      \
        T exaggeration, da_int start, da_int end, T *grad_i);

DA_KT_INSTANTIATE(ATTRACTIVE_FORCES_MULTI_D2_INSTANTIATE, bsz::b256)
#ifdef __AVX512F__
DA_KT_INSTANTIATE(ATTRACTIVE_FORCES_MULTI_D2_INSTANTIATE, bsz::b512)
#endif

#undef ATTRACTIVE_FORCES_KT_INSTANTIATE
#undef ATTRACTIVE_FORCES_MULTI_D2_INSTANTIATE

} // namespace da_tsne

} // namespace ARCH
