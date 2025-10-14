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
#include "kmeans.hpp"
#include "kt.hpp"
#include "macros.h"
#include <cassert>
namespace ARCH {

namespace da_kmeans {

using namespace kernel_templates;

/* These functions contain performance-critical loops which must vectorize for performance. */

// KT variants of elkan_reduce_kernel
template <bsz SZ, typename T>
inline __attribute__((__always_inline__)) T
elkan_reduction_kt(da_int m, const T *x, da_int incx, T *y,
                   da_int incy) { // why isnt' y also const * T ?? FIXME also add noexcept
    using SUF = T;
    avxvector_t<SZ, SUF> vsum{kt_setzero_p<SZ, SUF>()};

    const da_int simd_length{tsz_v<SZ, SUF>}; // type pack size given vector length
    const da_int simd_loop_size{m - m % simd_length};
    const da_int prefetch_condition{simd_loop_size - simd_length};

    da_int indx[simd_length]; // vector containing all the increments
    da_int indy[simd_length]; // vector containing all the increments
    for (da_int i = 0; i < simd_length; ++i) {
        indx[i] = incx * i;
        indy[i] = incy * i;
    }

    const da_int simd_incx{incx * simd_length};
    const da_int simd_incy{incy * simd_length};

    for (da_int k = 0; k < simd_loop_size; k += simd_length) {

        da_int kincx = k * incx;
        da_int kincy = k * incy;
        const T *x_ptr = &x[kincx];
        const T *y_ptr = &y[kincy];

        if (k < prefetch_condition) {
            da_int x_ptr_index = simd_incx;
            da_int y_ptr_index = simd_incy;
            // Prefetch the elements for the next iteration to help with cache misses
            for (da_int j = 0; j < simd_length; j++) {
                _mm_prefetch((const char *)&x_ptr[x_ptr_index], _MM_HINT_T0);
                _mm_prefetch((const char *)&y_ptr[y_ptr_index], _MM_HINT_T0);
                x_ptr_index += incx;
                y_ptr_index += incy;
            }
        }
        // load elements with incx stride
        avxvector_t<SZ, SUF> vx = kt_set_p<SZ, SUF>(x_ptr, indx);
        // load elements with incy stride
        avxvector_t<SZ, SUF> vy = kt_set_p<SZ, SUF>(y_ptr, indy);
        avxvector_t<SZ, SUF> diff{kt_sub_p<SZ, SUF>(vx, vy)};
        vsum = kt_fmadd_p<SZ, SUF>(diff, diff, vsum);
    }

    // hsum of vsum
    T sum{kt_hsum_p<SZ, SUF>(vsum)};
    // Handle the remainder
    for (da_int k = simd_loop_size; k < m; k++) {
        T tmp = x[k * incx] - y[k * incy];
        sum += tmp * tmp;
    }

    return sum;
}

#define ELKAN_REDUCTION_KT_INSTANTIATE(SZ, SUF)                                          \
    template SUF elkan_reduction_kt<SZ, SUF>(da_int m, const SUF *x, da_int incx,        \
                                             SUF *y, da_int incy);

DA_KT_INSTANTIATE(ELKAN_REDUCTION_KT_INSTANTIATE, bsz::b128)
DA_KT_INSTANTIATE(ELKAN_REDUCTION_KT_INSTANTIATE, bsz::b256)
#ifdef __AVX512F__
DA_KT_INSTANTIATE(ELKAN_REDUCTION_KT_INSTANTIATE, bsz::b512)
#endif

/* Reduction part of the elkan iteration, on a pair of scattered vectors */
template <typename T>
T elkan_reduction_kernel_scalar(da_int m, const T *x, da_int incx, T *y, da_int incy) {
    T sum = (T)0.0;
#pragma omp simd reduction(+ : sum)
    for (da_int k = 0; k < m; k++) {
        T tmp = x[k * incx] - y[k * incy];
        sum += tmp * tmp;
    }
    return sum;
}

template <>
float elkan_reduction_kernel<float, vectorization_type::scalar>(da_int m, const float *x,
                                                                da_int incx, float *y,
                                                                da_int incy) {
    return elkan_reduction_kernel_scalar(m, x, incx, y, incy);
}

template <>
double elkan_reduction_kernel<double, vectorization_type::scalar>(da_int m,
                                                                  const double *x,
                                                                  da_int incx, double *y,
                                                                  da_int incy) {
    return elkan_reduction_kernel_scalar(m, x, incx, y, incy);
}

template <>
float elkan_reduction_kernel<float, vectorization_type::avx>(da_int m, const float *x,
                                                             da_int incx, float *y,
                                                             da_int incy) {

    return elkan_reduction_kt<bsz::b128, float>(m, x, incx, y, incy);
}

template <>
double elkan_reduction_kernel<double, vectorization_type::avx>(da_int m, const double *x,
                                                               da_int incx, double *y,
                                                               da_int incy) {
    return elkan_reduction_kt<bsz::b128, double>(m, x, incx, y, incy);
}

template <>
float elkan_reduction_kernel<float, vectorization_type::avx2>(da_int m, const float *x,
                                                              da_int incx, float *y,
                                                              da_int incy) {
    return elkan_reduction_kt<bsz::b256, float>(m, x, incx, y, incy);
}

template <>
double elkan_reduction_kernel<double, vectorization_type::avx2>(da_int m, const double *x,
                                                                da_int incx, double *y,
                                                                da_int incy) {

    return elkan_reduction_kt<bsz::b256, double>(m, x, incx, y, incy);
}

#if defined(__AVX512F__)
template <>
double elkan_reduction_kernel<double, vectorization_type::avx512>(da_int m,
                                                                  const double *x,
                                                                  da_int incx, double *y,
                                                                  da_int incy) {
    return elkan_reduction_kt<bsz::b256, double>(m, x, incx, y, incy);
}

template <>
float elkan_reduction_kernel<float, vectorization_type::avx512>(da_int m, const float *x,
                                                                da_int incx, float *y,
                                                                da_int incy) {
    return elkan_reduction_kt<bsz::b256, float>(m, x, incx, y, incy);
}
#endif

/* Within Elkan iteration update a block of the lower and upper bound matrices*/
template <class T>
void elkan_iteration_kernel_scalar(da_int block_size, T *l_bound, da_int ldl_bound,
                                   T *u_bound, T *centre_shift, da_int *labels,
                                   da_int n_clusters) {

    da_int index = 0;
    for (da_int i = 0; i < block_size; i++) {
        u_bound[i] += centre_shift[labels[i]];
#pragma omp simd
        for (da_int j = 0; j < n_clusters; j++) {
            l_bound[index + j] -= centre_shift[j];
            if (l_bound[index + j] < 0) {
                l_bound[index + j] = (T)0.0;
            }
        }
        index += ldl_bound;
    }
}

template <>
void elkan_iteration_kernel<double, vectorization_type::scalar>(
    da_int block_size, double *l_bound, da_int ldl_bound, double *u_bound,
    double *centre_shift, da_int *labels, da_int n_clusters) {

    elkan_iteration_kernel_scalar(block_size, l_bound, ldl_bound, u_bound, centre_shift,
                                  labels, n_clusters);
}

template <>
void elkan_iteration_kernel<float, vectorization_type::scalar>(
    da_int block_size, float *l_bound, da_int ldl_bound, float *u_bound,
    float *centre_shift, da_int *labels, da_int n_clusters) {

    elkan_iteration_kernel_scalar(block_size, l_bound, ldl_bound, u_bound, centre_shift,
                                  labels, n_clusters);
}

// KT variants of elkan_reduction_kernel
template <bsz SZ, typename SUF>
inline __attribute__((__always_inline__)) void
elkan_iteration_kt(da_int block_size, SUF *l_bound, da_int ldl_bound, SUF *u_bound,
                   SUF *centre_shift, da_int *labels, da_int n_clusters) {
    const avxvector_t<SZ, SUF> v0{kt_setzero_p<SZ, SUF>()};
    const da_int simd_length{tsz_v<SZ, SUF>};

    for (da_int i = 0; i < block_size; i++) {
        da_int col_index = i * ldl_bound;

        for (da_int j = 0; j < n_clusters; j += simd_length) {
            da_int index = col_index + j;
            avxvector_t<SZ, SUF> v_lb = kt_loadu_p<SZ>(&l_bound[index]);
            avxvector_t<SZ, SUF> vc_shift = kt_loadu_p<SZ>(&centre_shift[j]);
            v_lb = kt_sub_p<SZ, SUF>(v_lb, vc_shift);
            v_lb = kt_max_p<SZ, SUF>(v_lb, v0);
            kt_storeu_p<SZ>(&l_bound[index], v_lb);
        }
    }

    const da_int simd_loop_size = block_size - block_size % simd_length;

    for (da_int i = 0; i < simd_loop_size; i += simd_length) {
        avxvector_t<SZ, SUF> vc_shift = kt_set_p<SZ>(centre_shift, &labels[i]);
        avxvector_t<SZ, SUF> v_ub = kt_loadu_p<SZ>(&u_bound[i]);
        v_ub = kt_add_p<SZ, SUF>(v_ub, vc_shift);
        kt_storeu_p<SZ>(&u_bound[i], v_ub);
    }

    // Handle the remainder
    for (da_int i = simd_loop_size; i < block_size; i++) {
        u_bound[i] += centre_shift[labels[i]];
    }
}
// instantiate
template void elkan_iteration_kt<bsz::b128, double>(da_int block_size, double *l_bound,
                                                    da_int ldl_bound, double *u_bound,
                                                    double *centre_shift, da_int *labels,
                                                    da_int n_clusters);
template void elkan_iteration_kt<bsz::b128, float>(da_int block_size, float *l_bound,
                                                   da_int ldl_bound, float *u_bound,
                                                   float *centre_shift, da_int *labels,
                                                   da_int n_clusters);
template void elkan_iteration_kt<bsz::b256, double>(da_int block_size, double *l_bound,
                                                    da_int ldl_bound, double *u_bound,
                                                    double *centre_shift, da_int *labels,
                                                    da_int n_clusters);
template void elkan_iteration_kt<bsz::b256, float>(da_int block_size, float *l_bound,
                                                   da_int ldl_bound, float *u_bound,
                                                   float *centre_shift, da_int *labels,
                                                   da_int n_clusters);
#ifdef __AVX512F__
template void elkan_iteration_kt<bsz::b512, double>(da_int block_size, double *l_bound,
                                                    da_int ldl_bound, double *u_bound,
                                                    double *centre_shift, da_int *labels,
                                                    da_int n_clusters);
template void elkan_iteration_kt<bsz::b512, float>(da_int block_size, float *l_bound,
                                                   da_int ldl_bound, float *u_bound,
                                                   float *centre_shift, da_int *labels,
                                                   da_int n_clusters);
#endif

template <>
void elkan_iteration_kernel<float, vectorization_type::avx>(
    da_int block_size, float *l_bound, da_int ldl_bound, float *u_bound,
    float *centre_shift, da_int *labels, da_int n_clusters) {
    elkan_iteration_kt<bsz::b128, float>(block_size, l_bound, ldl_bound, u_bound,
                                         centre_shift, labels, n_clusters);
}

template <>
void elkan_iteration_kernel<double, vectorization_type::avx>(
    da_int block_size, double *l_bound, da_int ldl_bound, double *u_bound,
    double *centre_shift, da_int *labels, da_int n_clusters) {
    elkan_iteration_kt<bsz::b128, double>(block_size, l_bound, ldl_bound, u_bound,
                                          centre_shift, labels, n_clusters);
}

template <>
void elkan_iteration_kernel<double, vectorization_type::avx2>(
    da_int block_size, double *l_bound, da_int ldl_bound, double *u_bound,
    double *centre_shift, da_int *labels, da_int n_clusters) {
    elkan_iteration_kt<bsz::b256, double>(block_size, l_bound, ldl_bound, u_bound,
                                          centre_shift, labels, n_clusters);
}

template <>
void elkan_iteration_kernel<float, vectorization_type::avx2>(
    da_int block_size, float *l_bound, da_int ldl_bound, float *u_bound,
    float *centre_shift, da_int *labels, da_int n_clusters) {
    elkan_iteration_kt<bsz::b256, float>(block_size, l_bound, ldl_bound, u_bound,
                                         centre_shift, labels, n_clusters);
}

#ifdef __AVX512F__
template <>
void elkan_iteration_kernel<float, vectorization_type::avx512>(
    da_int block_size, float *l_bound, da_int ldl_bound, float *u_bound,
    float *centre_shift, da_int *labels, da_int n_clusters) {
    elkan_iteration_kt<bsz::b512, float>(block_size, l_bound, ldl_bound, u_bound,
                                         centre_shift, labels, n_clusters);
}

template <>
void elkan_iteration_kernel<double, vectorization_type::avx512>(
    da_int block_size, double *l_bound, da_int ldl_bound, double *u_bound,
    double *centre_shift, da_int *labels, da_int n_clusters) {

    elkan_iteration_kt<bsz::b512, double>(block_size, l_bound, ldl_bound, u_bound,
                                          centre_shift, labels, n_clusters);
}
#endif

template <class T>
void lloyd_iteration_kernel_scalar(bool update_centres, da_int block_size,
                                   T *centre_norms, da_int *cluster_count, da_int *labels,
                                   T *work, da_int ldwork, da_int n_clusters) {

    T tmp2 = centre_norms[0];

    // Go through each sample in work and find argmin

    for (da_int i = 0; i < block_size; i++) {
        da_int ind = i * ldwork;
        T smallest_dist = work[ind] + tmp2;
        da_int label = 0;
        for (da_int j = 1; j < n_clusters; j++) {
            da_int index = ind + j;
            T tmp = work[index] + centre_norms[j];
            if (tmp < smallest_dist) {
                label = j;
                smallest_dist = tmp;
            }
        }
        labels[i] = label;
        if (update_centres)
            cluster_count[label] += 1;
    }
}

template <>
void lloyd_iteration_kernel<double, vectorization_type::scalar>(
    bool update_centres, da_int block_size, double *centre_norms, da_int *cluster_count,
    da_int *labels, double *work, da_int ldwork, da_int n_clusters) {

    lloyd_iteration_kernel_scalar(update_centres, block_size, centre_norms, cluster_count,
                                  labels, work, ldwork, n_clusters);
}

template <>
void lloyd_iteration_kernel<float, vectorization_type::scalar>(
    bool update_centres, da_int block_size, float *centre_norms, da_int *cluster_count,
    da_int *labels, float *work, da_int ldwork, da_int n_clusters) {

    lloyd_iteration_kernel_scalar(update_centres, block_size, centre_norms, cluster_count,
                                  labels, work, ldwork, n_clusters);
}

template <>
void lloyd_iteration_kernel<double, vectorization_type::avx>(
    bool update_centres, da_int block_size, double *centre_norms, da_int *cluster_count,
    da_int *labels, double *work, da_int ldwork, da_int n_clusters) {

    // Declare as unions so we can access individual elements later
    v2df_t v_smallest_dists;
    v2i64_t v_labels;

    __m128d v_centre_norms = _mm_loadu_pd(centre_norms);

    for (da_int i = 0; i < block_size; i++) {
        da_int ind_outer = i * ldwork;
        v_smallest_dists.v = _mm_add_pd(_mm_loadu_pd(work + ind_outer), v_centre_norms);

        v_labels.v = _mm_set_epi64x(1, 0);

        // No need to worry about n_clusters not being a multpile of 2 as we have already padded the relevant arrays
        for (da_int j = 2; j < n_clusters; j += 2) {
            da_int ind_inner = ind_outer + j;
            __m128d v_tmp = _mm_add_pd(_mm_loadu_pd(work + ind_inner),
                                       _mm_loadu_pd(centre_norms + j));
            __m128d v_mask = _mm_cmp_pd(v_tmp, v_smallest_dists.v, _CMP_LT_OQ);
            __m128i v_mask_int = _mm_castpd_si128(v_mask);
            // Use 64 bit integers in v_labels and v_indices
            __m128i v_indices = _mm_set_epi64x(j + 1, j);
            v_labels.v = _mm_blendv_epi8(v_labels.v, v_indices, v_mask_int);
            v_smallest_dists.v = _mm_min_pd(v_smallest_dists.v, v_tmp);
        }

        labels[i] = (da_int)v_labels.i[0];

        if (v_smallest_dists.d[1] < v_smallest_dists.d[0]) {
            labels[i] = (da_int)v_labels.i[1];
        }

        if (update_centres)
            cluster_count[labels[i]] += 1;
    }
}

template <>
void lloyd_iteration_kernel<float, vectorization_type::avx>(
    bool update_centres, da_int block_size, float *centre_norms, da_int *cluster_count,
    da_int *labels, float *work, da_int ldwork, da_int n_clusters) {

    v4sf_t v_smallest_dists;
#if defined(AOCLDA_ILP64)
    v2i64_t v_labels1, v_labels2;
#else
    v4i32_t v_labels;
#endif

    __m128 v_centre_norms = _mm_loadu_ps(centre_norms);

    for (da_int i = 0; i < block_size; i++) {
        da_int ind_outer = i * ldwork;
        v_smallest_dists.v = _mm_add_ps(_mm_loadu_ps(work + ind_outer), v_centre_norms);
#if defined(AOCLDA_ILP64)
        // Because we are dealing with 64 bit integers, we need to use two integer vectors for the labels
        v_labels1.v = _mm_set_epi64x(1, 0);
        v_labels2.v = _mm_set_epi64x(3, 2);
#else
        v_labels.v = _mm_set_epi32(3, 2, 1, 0);
#endif
        // No need to worry about n_clusters not being a multpile of 4 as we have already padded the relevant arrays
        for (da_int j = 4; j < n_clusters; j += 4) {
            da_int ind_inner = ind_outer + j;
            __m128 v_tmp = _mm_add_ps(_mm_loadu_ps(work + ind_inner),
                                      _mm_loadu_ps(centre_norms + j));
            __m128 v_mask = _mm_cmp_ps(v_tmp, v_smallest_dists.v, _CMP_LT_OQ);

#if defined(AOCLDA_ILP64)
            // v_mask will currently only work for 32 bit integers, so we need to create two
            // 64 bit integer masks from it

            //  Extract the lower bits of v_mask and create a new mask which duplicates them
            __m128 v_mask_perm_lower = _mm_permute_ps(v_mask, 0b01010000);
            __m128i v_mask_lower = _mm_castps_si128(v_mask_perm_lower);

            //  Extract the upper bits of v_mask and create a new mask which duplicates them
            __m128 v_mask_perm_upper = _mm_permute_ps(v_mask, 0b11111010);
            __m128i v_mask_upper = _mm_castps_si128(v_mask_perm_upper);

            __m128i v_indices1 = _mm_set_epi64x(j + 1, j);
            __m128i v_indices2 = _mm_set_epi64x(j + 3, j + 2);

            // Use our new masks to blend the indices with v_labels1 and 2, all of which are 64 bits
            v_labels1.v = _mm_blendv_epi8(v_labels1.v, v_indices1, v_mask_lower);
            v_labels2.v = _mm_blendv_epi8(v_labels2.v, v_indices2, v_mask_upper);
#else
            __m128i v_mask_int = _mm_castps_si128(v_mask);
            __m128i v_indices = _mm_set_epi32(j + 3, j + 2, j + 1, j);
            v_labels.v = _mm_blendv_epi8(v_labels.v, v_indices, v_mask_int);
#endif
            v_smallest_dists.v = _mm_min_ps(v_smallest_dists.v, v_tmp);
        }

        // Extract the label corresponding to the smallest distance computed (little to be gained from using permute operations here)
#if defined(AOCLDA_ILP64)
        da_int label = v_labels1.i[0];

        if (v_smallest_dists.f[1] < v_smallest_dists.f[0]) {
            v_smallest_dists.f[0] = v_smallest_dists.f[1];
            label = (da_int)v_labels1.i[1];
        }

        if (v_smallest_dists.f[2] < v_smallest_dists.f[0]) {
            v_smallest_dists.f[0] = v_smallest_dists.f[2];
            label = (da_int)v_labels2.i[0];
        }

        if (v_smallest_dists.f[3] < v_smallest_dists.f[0]) {
            label = (da_int)v_labels2.i[1];
        }

#else

        da_int label = (da_int)v_labels.i[0];
        for (da_int j = 1; j < 3; j++) {
            if (v_smallest_dists.f[j] < v_smallest_dists.f[0]) {
                v_smallest_dists.f[0] = v_smallest_dists.f[j];
                label = (da_int)v_labels.i[j];
            }
        }
        if (v_smallest_dists.f[3] < v_smallest_dists.f[0]) {
            label = (da_int)v_labels.i[3];
        }

#endif
        labels[i] = label;
        if (update_centres)
            cluster_count[label] += 1;
    }
}

template <>
void lloyd_iteration_kernel<double, vectorization_type::avx2>(
    bool update_centres, da_int block_size, double *centre_norms, da_int *cluster_count,
    da_int *labels, double *work, da_int ldwork, da_int n_clusters) {

    v4df_t v_smallest_dists;
    v4i64_t v_labels;

    __m256d v_centre_norms = _mm256_loadu_pd(centre_norms);

    for (da_int i = 0; i < block_size; i++) {
        da_int ind_outer = i * ldwork;
        v_smallest_dists.v =
            _mm256_add_pd(_mm256_loadu_pd(work + ind_outer), v_centre_norms);

        v_labels.v = _mm256_set_epi64x(3, 2, 1, 0);

        // No need to worry about n_clusters not being a multpile of 4 as we have already padded the relevant arrays
        for (da_int j = 4; j < n_clusters; j += 4) {
            da_int ind_inner = ind_outer + j;
            __m256d v_tmp = _mm256_add_pd(_mm256_loadu_pd(work + ind_inner),
                                          _mm256_loadu_pd(centre_norms + j));
            __m256d v_mask = _mm256_cmp_pd(v_tmp, v_smallest_dists.v, _CMP_LT_OQ);
            __m256i v_mask_int = _mm256_castpd_si256(v_mask);
            // Note, we are working with 64 bit integers in v_labels and v_indices
            __m256i v_indices = _mm256_set_epi64x(j + 3, j + 2, j + 1, j);
            v_labels.v = _mm256_blendv_epi8(v_labels.v, v_indices, v_mask_int);
            v_smallest_dists.v = _mm256_min_pd(v_smallest_dists.v, v_tmp);
        }

        // Extract the label corresponding to the smallest distance computed (little to be gained from using permute operations here)
        da_int label = (da_int)v_labels.i[0];
        for (da_int j = 1; j < 3; j++) {
            if (v_smallest_dists.d[j] < v_smallest_dists.d[0]) {
                v_smallest_dists.d[0] = v_smallest_dists.d[j];
                label = (da_int)v_labels.i[j];
            }
        }
        if (v_smallest_dists.d[3] < v_smallest_dists.d[0]) {
            label = (da_int)v_labels.i[3];
        }

        labels[i] = label;
        if (update_centres)
            cluster_count[label] += 1;
    }
}

template <>
void lloyd_iteration_kernel<float, vectorization_type::avx2>(
    bool update_centres, da_int block_size, float *centre_norms, da_int *cluster_count,
    da_int *labels, float *work, da_int ldwork, da_int n_clusters) {

    v8sf_t v_smallest_dists;
#if defined(AOCLDA_ILP64)
    v4i64_t v_labels1, v_labels2;
#else
    v8i32_t v_labels;
#endif

    __m256 v_centre_norms = _mm256_loadu_ps(centre_norms);

    for (da_int i = 0; i < block_size; i++) {
        da_int ind_outer = i * ldwork;
        v_smallest_dists.v =
            _mm256_add_ps(_mm256_loadu_ps(work + ind_outer), v_centre_norms);
#if defined(AOCLDA_ILP64)
        // Because we are dealing with 64 bit integers, we need to use two integer vectors for the labels
        v_labels1.v = _mm256_set_epi64x(3, 2, 1, 0);
        v_labels2.v = _mm256_set_epi64x(7, 6, 5, 4);
#else
        v_labels.v = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
#endif
        // No need to worry about n_clusters not being a multpile of 8 as we have already padded the relevant arrays
        for (da_int j = 8; j < n_clusters; j += 8) {
            da_int ind_inner = ind_outer + j;
            __m256 v_tmp = _mm256_add_ps(_mm256_loadu_ps(work + ind_inner),
                                         _mm256_loadu_ps(centre_norms + j));
            __m256 v_mask = _mm256_cmp_ps(v_tmp, v_smallest_dists.v, _CMP_LT_OQ);
            __m256i v_mask_int = _mm256_castps_si256(v_mask);

#if defined(AOCLDA_ILP64)
            // v_mask_int will currently only work for 32 bit integers, so we need to create two
            // 64 bit integer masks from it

            //  Extract the lower bits of v_mask_int and create a new mask which duplicates them
            __m256i control_mask_lower = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
            __m256i v_mask_lower =
                _mm256_permutevar8x32_epi32(v_mask_int, control_mask_lower);

            //  Extract the upper bits of v_mask_int and create a new mask which duplicates them
            __m256i control_mask_upper = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);
            __m256i v_mask_upper =
                _mm256_permutevar8x32_epi32(v_mask_int, control_mask_upper);

            // Use our new masks to blend the indices with v_labels1 and 2, all of which are 64 bits
            __m256i v_indices1 = _mm256_set_epi64x(j + 3, j + 2, j + 1, j);
            __m256i v_indices2 = _mm256_set_epi64x(j + 7, j + 6, j + 5, j + 4);
            v_labels1.v = _mm256_blendv_epi8(v_labels1.v, v_indices1, v_mask_lower);
            v_labels2.v = _mm256_blendv_epi8(v_labels2.v, v_indices2, v_mask_upper);
#else

            __m256i v_indices =
                _mm256_set_epi32(j + 7, j + 6, j + 5, j + 4, j + 3, j + 2, j + 1, j);
            v_labels.v = _mm256_blendv_epi8(v_labels.v, v_indices, v_mask_int);
#endif
            v_smallest_dists.v = _mm256_min_ps(v_smallest_dists.v, v_tmp);
        }

        // Extract the label corresponding to the smallest distance computed (little to be gained from using permute operations here)
#if defined(AOCLDA_ILP64)
        da_int label = v_labels1.i[0];
        for (da_int j = 1; j < 4; j++) {
            if (v_smallest_dists.f[j] < v_smallest_dists.f[0]) {
                v_smallest_dists.f[0] = v_smallest_dists.f[j];
                label = (da_int)v_labels1.i[j];
            }
        }
        for (da_int j = 4; j < 7; j++) {
            if (v_smallest_dists.f[j] < v_smallest_dists.f[0]) {
                v_smallest_dists.f[0] = v_smallest_dists.f[j];
                label = (da_int)v_labels2.i[j - 4];
            }
        }
        if (v_smallest_dists.f[7] < v_smallest_dists.f[0]) {
            label = (da_int)v_labels2.i[3];
        }

#else
        da_int label = (da_int)v_labels.i[0];
        for (da_int j = 1; j < 7; j++) {
            if (v_smallest_dists.f[j] < v_smallest_dists.f[0]) {
                v_smallest_dists.f[0] = v_smallest_dists.f[j];
                label = (da_int)v_labels.i[j];
            }
        }
        if (v_smallest_dists.f[7] < v_smallest_dists.f[0]) {
            label = (da_int)v_labels.i[7];
        }
#endif
        labels[i] = label;
        if (update_centres)
            cluster_count[label] += 1;
    }
}

// LCOV_EXCL_START

#ifdef __AVX512F__
template <>
void lloyd_iteration_kernel<float, vectorization_type::avx512>(
    bool update_centres, da_int block_size, float *centre_norms, da_int *cluster_count,
    da_int *labels, float *work, da_int ldwork, da_int n_clusters) {

    v16sf_t v_smallest_dists;
#if defined(AOCLDA_ILP64)
    v8i64_t v_labels1, v_labels2;
#else
    v16i32_t v_labels;
#endif

    __m512 v_centre_norms = _mm512_loadu_ps(centre_norms);

    for (da_int i = 0; i < block_size; i++) {
        da_int ind_outer = i * ldwork;
        v_smallest_dists.v =
            _mm512_add_ps(_mm512_loadu_ps(work + ind_outer), v_centre_norms);
#if defined(AOCLDA_ILP64)
        // Because we are dealing with 64 bit integers, we need to use two integer vectors for the labels
        v_labels1.v = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        v_labels2.v = _mm512_set_epi64(15, 14, 13, 12, 11, 10, 9, 8);
#else
        v_labels.v =
            _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
#endif
        // No need to worry about n_clusters not being a multpile of 16 as we have already padded the relevant arrays
        for (da_int j = 16; j < n_clusters; j += 16) {
            da_int ind_inner = ind_outer + j;
            __m512 v_tmp = _mm512_add_ps(_mm512_loadu_ps(work + ind_inner),
                                         _mm512_loadu_ps(centre_norms + j));
            __mmask16 v_mask = _mm512_cmp_ps_mask(v_tmp, v_smallest_dists.v, _CMP_LT_OQ);

#if defined(AOCLDA_ILP64)
            // v_mask will currently only work for 32 bit integers, so we need to create two
            // masks from it so we can work with 64 bit integers

            // Split v_mask into two _mmask8 variables
            __mmask8 v_mask_lower = v_mask & 0xFF;        // Keep the lower 8 bits
            __mmask8 v_mask_upper = (v_mask >> 8) & 0xFF; // Get the upper 8 bits

            __m512i v_indices1 =
                _mm512_set_epi64(j + 7, j + 6, j + 5, j + 4, j + 3, j + 2, j + 1, j);
            __m512i v_indices2 = _mm512_set_epi64(j + 15, j + 14, j + 13, j + 12, j + 11,
                                                  j + 10, j + 9, j + 8);

            // Use our new masks to blend the indices with v_labels1 and 2, all of which are 64 bits
            v_labels1.v = _mm512_mask_blend_epi64(v_mask_lower, v_labels1.v, v_indices1);
            v_labels2.v = _mm512_mask_blend_epi64(v_mask_upper, v_labels2.v, v_indices2);
#else

            __m512i v_indices = _mm512_set_epi32(j + 15, j + 14, j + 13, j + 12, j + 11,
                                                 j + 10, j + 9, j + 8, j + 7, j + 6,
                                                 j + 5, j + 4, j + 3, j + 2, j + 1, j);
            v_labels.v = _mm512_mask_blend_epi32(v_mask, v_labels.v, v_indices);
#endif
            v_smallest_dists.v = _mm512_min_ps(v_smallest_dists.v, v_tmp);
        }

        // Extract the label corresponding to the smallest distance computed (little to be gained from using permute operations here)
#if defined(AOCLDA_ILP64)
        da_int label = v_labels1.i[0];
        for (da_int j = 1; j < 8; j++) {
            if (v_smallest_dists.f[j] < v_smallest_dists.f[0]) {
                v_smallest_dists.f[0] = v_smallest_dists.f[j];
                label = (da_int)v_labels1.i[j];
            }
        }
        for (da_int j = 8; j < 15; j++) {
            if (v_smallest_dists.f[j] < v_smallest_dists.f[0]) {
                v_smallest_dists.f[0] = v_smallest_dists.f[j];
                label = (da_int)v_labels2.i[j - 8];
            }
        }
        if (v_smallest_dists.f[15] < v_smallest_dists.f[0]) {
            label = (da_int)v_labels2.i[7];
        }

#else

        da_int label = (da_int)v_labels.i[0];
        for (da_int j = 1; j < 15; j++) {
            if (v_smallest_dists.f[j] < v_smallest_dists.f[0]) {
                v_smallest_dists.f[0] = v_smallest_dists.f[j];
                label = (da_int)v_labels.i[j];
            }
        }
        if (v_smallest_dists.f[15] < v_smallest_dists.f[0]) {
            label = (da_int)v_labels.i[15];
        }

#endif
        labels[i] = label;
        if (update_centres)
            cluster_count[label] += 1;
    }
}

template <>
void lloyd_iteration_kernel<double, vectorization_type::avx512>(
    bool update_centres, da_int block_size, double *centre_norms, da_int *cluster_count,
    da_int *labels, double *work, da_int ldwork, da_int n_clusters) {

    v8df_t v_smallest_dists;
    v8i64_t v_labels;

    __m512d v_centre_norms = _mm512_loadu_pd(centre_norms);

    for (da_int i = 0; i < block_size; i++) {
        da_int ind_outer = i * ldwork;
        v_smallest_dists.v =
            _mm512_add_pd(_mm512_loadu_pd(work + ind_outer), v_centre_norms);

        v_labels.v = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);

        // No need to worry about n_clusters not being a multpile of 8 as we have already padded the relevant arrays
        for (da_int j = 8; j < n_clusters; j += 8) {
            da_int ind_inner = ind_outer + j;
            __m512d v_tmp = _mm512_add_pd(_mm512_loadu_pd(work + ind_inner),
                                          _mm512_loadu_pd(centre_norms + j));
            __mmask8 v_mask = _mm512_cmp_pd_mask(v_tmp, v_smallest_dists.v, _CMP_LT_OQ);
            // Note, we are working with 64 bit integers in v_labels and v_indices
            __m512i v_indices =
                _mm512_set_epi64(j + 7, j + 6, j + 5, j + 4, j + 3, j + 2, j + 1, j);
            v_labels.v = _mm512_mask_blend_epi64(v_mask, v_labels.v, v_indices);
            v_smallest_dists.v = _mm512_min_pd(v_smallest_dists.v, v_tmp);
        }

        da_int label = (da_int)v_labels.i[0];
        for (da_int j = 1; j < 7; j++) {
            if (v_smallest_dists.d[j] < v_smallest_dists.d[0]) {
                v_smallest_dists.d[0] = v_smallest_dists.d[j];
                label = (da_int)v_labels.i[j];
            }
        }
        if (v_smallest_dists.d[7] < v_smallest_dists.d[0]) {
            label = (da_int)v_labels.i[7];
        }
        labels[i] = label;
        if (update_centres)
            cluster_count[label] += 1;
    }
}
#endif

// LCOV_EXCL_STOP

} // namespace da_kmeans

} // namespace ARCH
