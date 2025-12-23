/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "decision_tree_misc.hpp"
#include "aoclda.h"
#include "da_std.hpp"
#include "macros.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace ARCH {

namespace da_decision_forest {
da_status compress_count_occurences(std::vector<da_int> &indices,
                                    std::vector<da_int> &count) {
    std::sort(indices.begin(), indices.end());
    memset(count.data(), 0, count.size() * sizeof(da_int));

    size_t head = 0;
    da_int unique_elem_idx = 0;
    while (head < indices.size()) {
        da_int val = indices[head];
        da_int n_elem = 1;
        while (++head < indices.size() && indices[head] == val) {
            n_elem++;
        }
        indices[unique_elem_idx] = val;
        count[val] = n_elem;
        unique_elem_idx++;
    }

    try {
        indices.resize(unique_elem_idx);
    } catch (std::bad_alloc &) {       // LCOV_EXCL_LINE
        return da_status_memory_error; // LCOV_EXCL_LINE
    }
    return da_status_success;
}

/* Choose pivot using median-of-three
 * Note: pivot values need to be sorted in the main algorithm to use this strategy
 */
template <typename T>
T pivot_median3(std::vector<T> &vals, da_int start_idx, da_int n_elem) {
    T v1 = vals[start_idx];
    T v2 = vals[start_idx + n_elem / 2];
    T v3 = vals[start_idx + n_elem - 1];
    if (v1 < v2) {
        return v2 < v3 ? v2 : (v1 < v3 ? v3 : v1);
    } else {
        return v1 < v3 ? v1 : (v2 < v3 ? v3 : v2);
    }
}

/* Swap values in 2 vectors simultaneously */
template <typename T>
void swap(std::vector<da_int> &a, std::vector<T> &b, da_int i, da_int j) {
    std::swap(a[i], a[j]);
    std::swap(b[i], b[j]);
}

template <typename T>
void multi_range_heapify(std::vector<da_int> &indices, std::vector<T> &values,
                         da_int start_idx, da_int n_elem, da_int root_idx) {

    da_int root = root_idx;
    da_int idx_max;
    while (true) {
        idx_max = root + start_idx;
        da_int left_child = root * 2 + 1 + start_idx;
        da_int right_child = left_child + 1;
        if (left_child - start_idx <= n_elem - 1 && values[idx_max] < values[left_child])
            idx_max = left_child;
        if (right_child - start_idx <= n_elem - 1 &&
            values[idx_max] < values[right_child])
            idx_max = right_child;
        if (idx_max != root + start_idx) {
            swap(indices, values, root + start_idx, idx_max);
            root = idx_max - start_idx;
        } else
            break;
    }
}

template <typename T>
void multi_range_heap_sort(std::vector<da_int> &indices, std::vector<T> &values,
                           da_int start_idx, da_int n_elem) {

    // First non-leaf index
    da_int non_leaf_idx = n_elem / 2 - 1;
    for (da_int i = non_leaf_idx; i >= 0; i--) {
        multi_range_heapify(indices, values, start_idx, n_elem, i);
    }

    da_int n_unsorted = n_elem - 1;
    while (n_unsorted > 0) {
        // place the maximum value at the end of the heap and re-heapify
        swap(indices, values, start_idx, start_idx + n_unsorted);
        multi_range_heapify(indices, values, start_idx, n_unsorted, 0);
        n_unsorted--;
    }
}

template <typename T>
void multi_range_intro_sort(std::vector<da_int> &indices, std::vector<T> &values,
                            da_int start_idx, da_int n_elem, da_int max_depth) {
    if (n_elem <= 1)
        return;

    if (max_depth <= 0) {
        // Use heapsort if quicksort is in its quadratic stage
        multi_range_heap_sort(indices, values, start_idx, n_elem);
        return;
    }
    da_int maxd = max_depth;
    da_int n = n_elem;

    while (n > 1) {
        maxd--;

        // Choose a pivot
        T pivot = pivot_median3(values, start_idx, n);

        // Partition the range
        da_int left = start_idx;
        da_int right = start_idx + n;
        da_int pivot_idx = left;
        while (left < right) {
            if (values[left] < pivot) {
                swap(indices, values, left, pivot_idx);
                left++;
                pivot_idx++;
            } else if (values[left] > pivot) {
                right--;
                swap(indices, values, right, left);
            } else {
                left++;
            }
        }

        // Sort the left partition recursively, leaving the pivot as sorted
        // indices [start_idx, start_idx + pivot_idx]
        multi_range_intro_sort(indices, values, start_idx, pivot_idx - start_idx, maxd);

        // set up start_idx and n to sort the right partition
        n -= right - start_idx;
        start_idx += right - start_idx;
    }
}

template <class T>
void bucket_sort_samples(std::vector<da_int> &indices, std::vector<T> &values,
                         da_int n_cat, da_int start_idx, da_int n_elem, da_int ldbuck,
                         std::vector<da_int> &buckets, std::vector<da_int> &bucket_idx) {

    memset(bucket_idx.data(), (da_int)0, (size_t)n_cat * sizeof(da_int));
    for (da_int i = start_idx; i < start_idx + n_elem; i++) {
        da_int sidx = indices[i];
        da_int xval = std::round(values[i]);
        buckets[ldbuck * xval + bucket_idx[xval]] = sidx;
        bucket_idx[xval]++;
    }
    da_int s_idx = start_idx;
    for (da_int bidx = 0; bidx < n_cat; bidx++) {
        memcpy(&indices[s_idx], &buckets[bidx * ldbuck],
               (size_t)bucket_idx[bidx] * sizeof(da_int));
        da_std::fill(values.begin() + s_idx, values.begin() + s_idx + bucket_idx[bidx],
                     (T)bidx);
        s_idx += bucket_idx[bidx];
    }
}

template void multi_range_intro_sort<float>(std::vector<da_int> &indices,
                                            std::vector<float> &values, da_int start_idx,
                                            da_int n_elem, da_int max_depth);
template void multi_range_intro_sort<double>(std::vector<da_int> &indices,
                                             std::vector<double> &values,
                                             da_int start_idx, da_int n_elem,
                                             da_int max_depth);
template void multi_range_intro_sort<da_int>(std::vector<da_int> &indices,
                                             std::vector<da_int> &values,
                                             da_int start_idx, da_int n_elem,
                                             da_int max_depth);

template void swap<float>(std::vector<da_int> &a, std::vector<float> &b, da_int i,
                          da_int j);
template void swap<double>(std::vector<da_int> &a, std::vector<double> &b, da_int i,
                           da_int j);
template void multi_range_heap_sort<float>(std::vector<da_int> &indices,
                                           std::vector<float> &values, da_int start_idx,
                                           da_int n_elem);
template void multi_range_heap_sort<double>(std::vector<da_int> &indices,
                                            std::vector<double> &values, da_int start_idx,
                                            da_int n_elem);
template void bucket_sort_samples<float>(std::vector<da_int> &indices,
                                         std::vector<float> &values, da_int n_cat,
                                         da_int start_idx, da_int n_elem, da_int ldbuck,
                                         std::vector<da_int> &buckets,
                                         std::vector<da_int> &bucket_idx);
template void bucket_sort_samples<double>(std::vector<da_int> &indices,
                                          std::vector<double> &values, da_int n_cat,
                                          da_int start_idx, da_int n_elem, da_int ldbuck,
                                          std::vector<da_int> &buckets,
                                          std::vector<da_int> &bucket_idx);

} // namespace da_decision_forest
} // namespace ARCH