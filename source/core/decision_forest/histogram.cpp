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

#include "histogram.hpp"
#include "aoclda.h"
#include "da_omp.hpp"
#include "da_utils.hpp"
#include <algorithm>
#include <boost/sort/spreadsort/float_sort.hpp>
#include <chrono>
#include <cmath>

namespace ARCH {

namespace da_decision_forest {
/* Computes the mid-points thresholds values for at most max_bin bins
 * Check for constant values and potentially create fewer bin thresholds is data is categorical
 * The number of bins created is returned in nbins, should be :
 * - max_bin if data is continuous
 * - Number of categories if data is categorical
 */
template <typename T>
void bins<T>::compute_thresholds(std::vector<T> &values, da_int vals_idx, da_int feat_idx,
                                 T tol) {
    boost::sort::spreadsort::float_sort(values.begin() + vals_idx,
                                        values.begin() + vals_idx + n_samples);
    da_int n_elem = n_samples;
    T q_step = 1.0 / (T)max_bin;
    T last_q = 0.;
    da_int bin_idx = 0;
    T q = q_step;
    da_int thresh_start_idx = feat_idx * (max_bin - 1);
    while (q <= (T)1.0 - (T)0.001 * q_step) {
        da_int threshold_idx = (da_int)std::round(q * n_elem) - 1;
        da_int r_idx = threshold_idx;
        // Explore the right side of the threshold for flat regions
        while (r_idx < n_elem - 1 && std::abs(values[vals_idx + threshold_idx] -
                                              values[vals_idx + r_idx + 1]) < tol) {
            ++r_idx;
        }
        if (r_idx >= n_elem - 1) {
            break;
        }
        if (r_idx != threshold_idx) {
            last_q = (T)(r_idx) / (T)(n_elem);
            while (q <= last_q + q_step) {
                q += q_step;
            }
        } else {
            q += q_step;
        }
        thresholds[thresh_start_idx + bin_idx] = values[vals_idx + r_idx + 1];
        ++bin_idx;
    }
    nbins[feat_idx] = bin_idx + 1;
}

template <typename T>
void bins<T>::compute_thresholds_continuous(std::vector<T> &values, da_int vals_idx,
                                            da_int feat_idx, T tol) {

    da_int idx_step = (da_int)n_samples / max_bin;
    da_int thresh_idx = feat_idx * (max_bin - 1);
    auto start_it = values.begin() + vals_idx;
    auto end_it = start_it + n_samples;
    for (da_int bin_idx = 0; bin_idx < max_bin - 1; bin_idx++) {
        std::nth_element(start_it, start_it + idx_step, end_it);
        // Threshold is the first element of the next bin.
        // We can't average with the last element of the previous bin because elements are
        // only partially sorted
        thresholds[thresh_idx + bin_idx] = values[vals_idx + (bin_idx + 1) * idx_step];
        start_it += idx_step;
    }
    nbins[feat_idx] = max_bin;

    // Compress the thresholds if equal values are found
    da_int idx = 0, end_idx = max_bin - 1;
    T low_thres = values[vals_idx];
    while (idx < end_idx) {
        da_int first_idx = idx;
        while (idx < end_idx &&
               std::abs(low_thres - thresholds[thresh_idx + idx]) < tol) {
            idx++;
        }
        da_int n_shift = idx - first_idx;
        if (n_shift > 0) {
            for (da_int i = idx; i < end_idx; i++) {
                thresholds[thresh_idx + i - n_shift] = thresholds[thresh_idx + i];
            }
            nbins[feat_idx] -= n_shift;
            idx = first_idx + 1;
            end_idx -= n_shift;
        } else
            ++idx;
        low_thres = thresholds[thresh_idx + idx - 1];
    }
}

template <typename T>
da_status bins<T>::compute_histograms(const T *X, da_int n_samples, da_int n_features,
                                      da_int ldx) {
    std::vector<T> values;
    da_int n_threads = da_utils::get_n_threads_loop(n_features);
    try {
        thresholds.resize(n_features * (max_bin - 1));
        nbins.resize(n_features);
        values.resize(n_samples * n_threads);
        binned_data.resize(n_features * n_samples);
    } catch (std::bad_alloc &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    }

#pragma omp parallel for shared(values, thresholds, X, ldx, n_samples,                   \
                                    n_features) default(none)
    // Compute the thresholds for each feature
    for (da_int j = 0; j < n_features; j++) {
        da_int vals_idx = (da_int)omp_get_thread_num() * n_samples;
        memcpy(values.data() + vals_idx, X + j * ldx, n_samples * sizeof(T));
        compute_thresholds(values, vals_idx, j);
    }

    // Assign the values in X to their corresponding bins by binary search
#pragma omp parallel for shared(X, max_bin, ldx, n_features, n_samples) default(none)
    for (da_int j = 0; j < n_features; j++) {
        da_int X_col_idx = j * ldx;
        for (da_int i = 0; i < n_samples; i++) {
            da_int start = 0, end = nbins[j] - 1;
            while (start < end) {
                da_int mid = (start + end) / 2;
                if (X[X_col_idx + i] < thresholds[j * (max_bin - 1) + mid]) {
                    end = mid;
                } else {
                    start = mid + 1;
                }
            }
            binned_data[j * n_samples + i] = start;
        }
    }

    return da_status_success;
}

template class bins<double>;
template class bins<float>;

} // namespace da_decision_forest
} // namespace ARCH