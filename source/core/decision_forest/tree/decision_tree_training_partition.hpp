/*
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TREE_TRAINING_PARTITION_HPP
#define TREE_TRAINING_PARTITION_HPP

#include "aoclda.h"
#include "common/idx_sorting.hpp"
#include "decision_tree.hpp"
#include "macros.h"

namespace ARCH {
namespace da_decision_forest {

/****************************************************************************************
                                Partition functions
 ***************************************************************************************/
/* These functions partition the samples_idx array based on the node's feature and threshold
 * (depending on the strategy adopted for splitting)
 * All partition functions are expected to return the index of the split in samples_idx
 * (e.g., all values in samples_idx below the split index will be set in the left node)
 */
template <typename T>
da_int decision_tree<T>::partition_samples_raw_continuous(const node<T> &nd) {
    /* raw data: look in main feature matrix directly
     * continuous feature: all samples below the node threshold are first
     */
    da_int head_idx = nd.start_idx, tail_idx = nd.end_idx;
    da_int start_col = ldx * nd.feature;
    while (head_idx < tail_idx) {
        da_int h_sidx = samples_idx[head_idx];
        da_int t_sidx = samples_idx[tail_idx];
        T head_val = X[start_col + h_sidx];
        T tail_val = X[start_col + t_sidx];
        if (head_val <= nd.x_threshold)
            head_idx += 1;
        else if (tail_val > nd.x_threshold)
            tail_idx -= 1;
        else {
            std::swap(samples_idx[head_idx], samples_idx[tail_idx]);
        }
    }
    return head_idx - 1;
}

template <typename T>
da_int decision_tree<T>::partition_samples_raw_categorical(const node<T> &nd) {
    /* raw data: look in main feature matrix directly
     * categorical feature: all samples corresponding to the node category are first
     */
    da_int head_idx = nd.start_idx;
    da_int tail_idx = nd.end_idx;
    da_int col_idx = ldx * nd.feature;
    while (head_idx < tail_idx) {
        da_int head = samples_idx[head_idx];
        da_int tail = samples_idx[tail_idx];
        if (std::round(X[col_idx + head]) == nd.category)
            head_idx++;
        else if (std::round(X[col_idx + tail]) != nd.category)
            tail_idx--;
        else
            std::swap(samples_idx[head_idx], samples_idx[tail_idx]);
    }
    return head_idx - 1;
}

template <typename T>
da_int decision_tree<T>::partition_samples_hist_ordered(const node<T> &nd) {
    /* hist: use the binned data X_binned
     * ordered feature: all samples below the node threshold are first
     */
    da_int head_idx = nd.start_idx;
    da_int tail_idx = nd.end_idx;
    uint16_t cat_thresh = nd.category;
    da_int col_idx = ldx * nd.feature;
    while (head_idx < tail_idx) {
        da_int head = samples_idx[head_idx];
        da_int tail = samples_idx[tail_idx];
        uint16_t cat_head = X_binned->binned_data[col_idx + head];
        uint16_t cat_tail = X_binned->binned_data[col_idx + tail];
        if (cat_head <= cat_thresh)
            head_idx++;
        else if (cat_tail > cat_thresh)
            tail_idx--;
        else
            std::swap(samples_idx[head_idx], samples_idx[tail_idx]);
    }
    return head_idx - 1;
}

template <typename T>
da_int decision_tree<T>::partition_samples_hist_onevall(const node<T> &nd) {
    /* hist: use the binned data X_binned
     * categorical feature: all samples corresponding to the node category are first
     */
    da_int head_idx = nd.start_idx;
    da_int tail_idx = nd.end_idx;
    uint16_t cat_thresh = nd.category;
    da_int col_idx = ldx * nd.feature;
    while (head_idx < tail_idx) {
        da_int head = samples_idx[head_idx];
        da_int tail = samples_idx[tail_idx];
        uint16_t cat_head = X_binned->binned_data[col_idx + head];
        uint16_t cat_tail = X_binned->binned_data[col_idx + tail];
        if (cat_head == cat_thresh)
            head_idx++;
        else if (cat_tail != cat_thresh)
            tail_idx--;
        else
            std::swap(samples_idx[head_idx], samples_idx[tail_idx]);
    }
    return head_idx - 1;
}

} // namespace da_decision_forest
} // namespace ARCH

#endif