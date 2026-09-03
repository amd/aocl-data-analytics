/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef NN_UTILS_HPP
#define NN_UTILS_HPP

#include "aoclda.h"
#include "da_std.hpp"
#include "kt.hpp"
#include <algorithm>
#include <immintrin.h>

namespace ARCH {
namespace da_neighbors {

// Given a vector D of length n and an integer k, this function returns in the first k positions
// of a vector k_dist, the k smaller values of D (unordered) and in the first k positions of a vector
// k_ind, the corresponding indices of the original vector D, where initial indices are
// init_index, init_index+1, ...
template <typename T>
inline __attribute__((__always_inline__)) void
smaller_values_and_indices(da_int n, T *D, da_int k, da_int *k_ind, T *k_dist,
                           da_int init_index, bool init = true) {
    // Initialize the first k values of k_ind with init_index, init_index+1, ..., init_index+k-1
    if (init)
        da_std::iota(k_ind, k_ind + k, init_index);
    // Find the index of the maximum element and the corresponding maximum value.
    da_int max_index = std::max_element(k_dist, k_dist + k) - k_dist;
    T max_val = k_dist[max_index];

    for (da_int i = k; i < n; i++) {
        // Check if an element of D is smaller than the maximum value. If it is,
        // we need to replace it's index in k_ind and replace the corresponding D[i] in k_dist.
        if (D[i] <= max_val) {
            // We know D[i] is smaller than Dmax. So we update k_ind[max_index] and D[max_index]
            // so that they hold the new value.
            k_ind[max_index] = i;
            k_dist[max_index] = D[i];
            // Now we need to find the new maximum so that we compare against that in the next iteration.
            max_index = std::max_element(k_dist, k_dist + k) - k_dist;
            max_val = k_dist[max_index];
        }
    }
}

// Given a vector k_ind of length n, and a vector that holds the corresponding indices, return into two arrays n_dist and n_ind
// the sorted distances of D and the sorted indices, respectively.
template <typename T>
inline void sorted_n_dist_n_ind(da_int n, T *k_dist, da_int *k_ind, T *n_dist,
                                da_int *n_ind, da_int *perm_vector, bool return_distance,
                                bool get_squares, bool ascending = true) {
    // We sort with respect to partial distances and then we use the sorted array to reorder the array of indices.
    da_std::iota(perm_vector, perm_vector + n, 0);

    if (ascending) {
        // sort smallest-first
        std::stable_sort(perm_vector, perm_vector + n,
                         [&](da_int i, da_int j) { return k_dist[i] < k_dist[j]; });
    } else {
        // sort largest-first
        std::stable_sort(perm_vector, perm_vector + n,
                         [&](da_int i, da_int j) { return k_dist[i] > k_dist[j]; });
    }

    for (da_int i = 0; i < n; i++)
        n_ind[i] = k_ind[perm_vector[i]];

    if (return_distance) {
        if (get_squares) {
            for (da_int i = 0; i < n; i++)
                n_dist[i] = std::sqrt(k_dist[perm_vector[i]]);
        } else {
            for (da_int i = 0; i < n; i++)
                n_dist[i] = k_dist[perm_vector[i]];
        }
    }
}

// Vectorized selection utilities (k-smallest / k-largest)

// Returns the index of the maximum value in x[0..n-1].
template <typename T> inline da_int inline_iamax(da_int n, const T *x) {
    da_int idx = 0;
    T maxval = x[0];
    for (da_int i = 1; i < n; i++) {
        if (x[i] > maxval) {
            maxval = x[i];
            idx = i;
        }
    }
    return idx;
}

// Returns the index of the minimum value in x[0..n-1].
template <typename T> inline da_int inline_iamin(da_int n, const T *x) {
    da_int idx = 0;
    T minval = x[0];
    for (da_int i = 1; i < n; i++) {
        if (x[i] < minval) {
            minval = x[i];
            idx = i;
        }
    }
    return idx;
}

// Returns mask where lane i is set if a[i] <= b[i].
template <kernel_templates::bsz BSZ, typename T>
inline auto compare_less_equal_mask(kernel_templates::avxvector_t<BSZ, T> a,
                                    kernel_templates::avxvector_t<BSZ, T> b) {
#ifdef __AVX512F__
    if constexpr (BSZ == kernel_templates::bsz::b512) {
        if constexpr (std::is_same_v<T, float>) {
            return _mm512_cmp_ps_mask(a, b, _CMP_LE_OS);
        } else if constexpr (std::is_same_v<T, double>) {
            return _mm512_cmp_pd_mask(a, b, _CMP_LE_OS);
        }
    } else
#endif
        if constexpr (BSZ == kernel_templates::bsz::b256) {
        if constexpr (std::is_same_v<T, float>) {
            auto cmp = _mm256_cmp_ps(a, b, _CMP_LE_OS);
            return _mm256_movemask_ps(cmp);
        } else if constexpr (std::is_same_v<T, double>) {
            auto cmp = _mm256_cmp_pd(a, b, _CMP_LE_OS);
            return _mm256_movemask_pd(cmp);
        }
    } else {
        static_assert(BSZ == kernel_templates::bsz::b256 ||
                          BSZ == kernel_templates::bsz::b512,
                      "Unsupported bit size");
    }
}

// Returns mask where lane i is set if a[i] >= b[i].
template <kernel_templates::bsz BSZ, typename T>
inline auto compare_greater_equal_mask(kernel_templates::avxvector_t<BSZ, T> a,
                                       kernel_templates::avxvector_t<BSZ, T> b) {
#ifdef __AVX512F__
    if constexpr (BSZ == kernel_templates::bsz::b512) {
        if constexpr (std::is_same_v<T, float>) {
            return _mm512_cmp_ps_mask(a, b, _CMP_GE_OS);
        } else if constexpr (std::is_same_v<T, double>) {
            return _mm512_cmp_pd_mask(a, b, _CMP_GE_OS);
        }
    } else
#endif
        if constexpr (BSZ == kernel_templates::bsz::b256) {
        if constexpr (std::is_same_v<T, float>) {
            auto cmp = _mm256_cmp_ps(a, b, _CMP_GE_OS);
            return _mm256_movemask_ps(cmp);
        } else if constexpr (std::is_same_v<T, double>) {
            auto cmp = _mm256_cmp_pd(a, b, _CMP_GE_OS);
            return _mm256_movemask_pd(cmp);
        }
    } else {
        static_assert(BSZ == kernel_templates::bsz::b256 ||
                          BSZ == kernel_templates::bsz::b512,
                      "Unsupported bit size");
    }
}

// Scans D[0..n-1] against existing top-k in k_ind/k_dist, keeping the k
// smallest values. Uses global_offset for stored indices.
// k_ind and k_dist must already be fully populated with k candidates.
template <kernel_templates::bsz BSZ, typename T>
inline __attribute__((always_inline)) void smaller_values_and_indices_vectorized_kernel(
    da_int n, const T *D, da_int k, da_int *k_ind, T *k_dist, da_int global_offset) {
    constexpr da_int VSIZE = da_int(kernel_templates::tsz_v<BSZ, T>);
    da_int max_index = inline_iamax(k, k_dist);
    T max_val = k_dist[max_index];
    da_int i = 0;
    auto k_dist_max = kernel_templates::kt_set1_p<BSZ, T>(max_val);
    for (; i + VSIZE <= n; i += VSIZE) {
        auto k_dist_vec = kernel_templates::kt_loadu_p<BSZ, T>(D + i);
        auto mask = compare_less_equal_mask<BSZ, T>(k_dist_vec, k_dist_max);
        if (mask == 0)
            continue;
        while (mask) {
            da_int lane = __builtin_ctz(mask);
            T dist_candidate = D[i + lane];
            if (dist_candidate <= max_val) {
                k_dist[max_index] = dist_candidate;
                k_ind[max_index] = i + lane + global_offset;
                max_index = inline_iamax(k, k_dist);
                max_val = k_dist[max_index];
            }
            mask = mask & (mask - 1);
        }
        k_dist_max = kernel_templates::kt_set1_p<BSZ, T>(max_val);
    }
    for (; i < n; i++) {
        if (D[i] <= max_val) {
            k_ind[max_index] = i + global_offset;
            k_dist[max_index] = D[i];
            max_index = inline_iamax(k, k_dist);
            max_val = k_dist[max_index];
        }
    }
}

// Dispatcher: selects AVX512 or AVX2 kernel for k-smallest selection.
template <typename T>
void smaller_values_and_indices_vectorized(da_int n, const T *D, da_int k, da_int *k_ind,
                                           T *k_dist, da_int global_offset) {
#ifdef __AVX512F__
    smaller_values_and_indices_vectorized_kernel<kernel_templates::bsz::b512, T>(
        n, D, k, k_ind, k_dist, global_offset);
#elif defined(__AVX2__)
    smaller_values_and_indices_vectorized_kernel<kernel_templates::bsz::b256, T>(
        n, D, k, k_ind, k_dist, global_offset);
#else
    static_assert(false,
                  "smaller_values_and_indices_vectorized requires AVX2 or AVX512F");
#endif
}

// Mirror of smaller_values_and_indices_vectorized_kernel for inner product MIPS:
// keeps the k largest values seen so far using a min-tracked buffer.
// Replaces the current minimum when a larger value is found.
template <kernel_templates::bsz BSZ, typename T>
inline __attribute__((always_inline)) void
larger_values_and_indices_vectorized_kernel(da_int n, const T *D, da_int k, da_int *k_ind,
                                            T *k_dist, da_int global_offset) {
    constexpr da_int VSIZE = da_int(kernel_templates::tsz_v<BSZ, T>);
    da_int min_index = inline_iamin(k, k_dist);
    T min_val = k_dist[min_index];
    da_int i = 0;
    auto k_dist_min = kernel_templates::kt_set1_p<BSZ, T>(min_val);
    for (; i + VSIZE <= n; i += VSIZE) {
        auto k_dist_vec = kernel_templates::kt_loadu_p<BSZ, T>(D + i);
        auto mask = compare_greater_equal_mask<BSZ, T>(k_dist_vec, k_dist_min);
        if (mask == 0)
            continue;
        while (mask) {
            da_int lane = __builtin_ctz(mask);
            T dist_candidate = D[i + lane];
            if (dist_candidate >= min_val) {
                k_dist[min_index] = dist_candidate;
                k_ind[min_index] = i + lane + global_offset;
                min_index = inline_iamin(k, k_dist);
                min_val = k_dist[min_index];
            }
            mask = mask & (mask - 1);
        }
        k_dist_min = kernel_templates::kt_set1_p<BSZ, T>(min_val);
    }
    for (; i < n; i++) {
        if (D[i] >= min_val) {
            k_ind[min_index] = i + global_offset;
            k_dist[min_index] = D[i];
            min_index = inline_iamin(k, k_dist);
            min_val = k_dist[min_index];
        }
    }
}

// Dispatcher: selects AVX512 or AVX2 kernel for k-largest selection (MIPS).
template <typename T>
void larger_values_and_indices_vectorized(da_int n, const T *D, da_int k, da_int *k_ind,
                                          T *k_dist, da_int global_offset) {
#ifdef __AVX512F__
    larger_values_and_indices_vectorized_kernel<kernel_templates::bsz::b512, T>(
        n, D, k, k_ind, k_dist, global_offset);
#elif defined(__AVX2__)
    larger_values_and_indices_vectorized_kernel<kernel_templates::bsz::b256, T>(
        n, D, k, k_ind, k_dist, global_offset);
#else
    static_assert(false, "larger_values_and_indices_vectorized requires AVX2 or AVX512F");
#endif
}

} // namespace da_neighbors
} // namespace ARCH

#endif // NN_UTILS_HPP