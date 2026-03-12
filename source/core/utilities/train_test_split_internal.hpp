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

#ifndef TTS_INTERNAL_HPP
#define TTS_INTERNAL_HPP

#include "aoclda.h"
#include "da_omp.hpp"
#include <algorithm>

namespace ARCH {
namespace da_utils {

// Kernel for splitting column-major data
template <typename T>
void split_column_data(da_int size, const T *X, T *split, const da_int *indices) {

    if (indices == nullptr) {
#pragma omp simd
        for (da_int i = 0; i < size; ++i) {
            split[i] = X[i];
        }
    } else {
#pragma omp simd
        for (da_int i = 0; i < size; ++i) {
            split[i] = X[indices[i]];
        }
    }
}

// Kernel for splitting row-major data when a single thread is available
template <typename T>
void split_row_data_serial(da_int n, da_int train_size, da_int test_size, const T *X,
                           da_int ldx, T *X_train, da_int ldx_train, T *X_test,
                           da_int ldx_test, const da_int *shuffle_array) {
    da_int ldx_remainder = 0;
    da_int m_X_addon_remainder = 0;
    da_int small_split = std::min(train_size, test_size);
    da_int big_split = std::max(train_size, test_size);
    T *X_remainder = nullptr;

    if (train_size > test_size) {
        X_remainder = X_train;
        ldx_remainder = ldx_train;
    } else {
        X_remainder = X_test;
        ldx_remainder = ldx_test;
        m_X_addon_remainder = train_size;
    }

    for (da_int i = 0; i < small_split; ++i) {
        da_int i_ldx_train = i * ldx_train;
        da_int i_ldx_test = i * ldx_test;
        da_int i_ldx = shuffle_array == nullptr ? i * ldx : shuffle_array[i] * ldx;
        da_int i_ldx_test_train = shuffle_array == nullptr
                                      ? (i + train_size) * ldx
                                      : shuffle_array[i + train_size] * ldx;

#pragma omp simd
        for (da_int j = 0; j < n; ++j) {
            X_train[i_ldx_train + j] = X[i_ldx + j];
            X_test[i_ldx_test + j] = X[i_ldx_test_train + j];
        }
    }

    for (da_int i = small_split; i < big_split; ++i) {
        da_int i_ldx = shuffle_array == nullptr
                           ? (i + m_X_addon_remainder) * ldx
                           : shuffle_array[i + m_X_addon_remainder] * ldx;
        da_int i_ldx_remainder = i * ldx_remainder;

#pragma omp simd
        for (da_int j = 0; j < n; ++j) {
            X_remainder[i_ldx_remainder + j] = X[i_ldx + j];
        }
    }
}

// Kernel for splitting row-major data when many threads are available
template <typename T>
void split_row_data_parallel(da_int n, da_int split_size, const T *X, da_int ldx,
                             T *X_split, da_int ldx_split, const da_int *shuffle_array) {

#pragma omp parallel for schedule(static) default(none)                                  \
    shared(n, split_size, X, ldx, X_split, ldx_split, shuffle_array)                     \
    num_threads(std::min((da_int)omp_get_max_threads(), split_size))

    for (da_int i = 0; i < split_size; ++i) {
        const T *X_ptr =
            shuffle_array == nullptr ? X + (i * ldx) : X + (shuffle_array[i] * ldx);
        T *X_split_ptr = X_split + (ldx_split * i);

#pragma omp simd
        for (da_int i = 0; i < n; ++i) {
            X_split_ptr[i] = X_ptr[i];
        }
    }
}
} // namespace da_utils
} // namespace ARCH

#endif // TTS_INTERNAL_HPP