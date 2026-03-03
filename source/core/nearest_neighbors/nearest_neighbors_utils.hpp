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
#include <algorithm>

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
                                bool get_squares) {
    // We sort with respect to partial distances and then we use the sorted array to reorder the array of indices.
    da_std::iota(perm_vector, perm_vector + n, 0);

    std::stable_sort(perm_vector, perm_vector + n,
                     [&](da_int i, da_int j) { return k_dist[i] < k_dist[j]; });

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

} // namespace da_neighbors
} // namespace ARCH

#endif // NN_UTILS_HPP