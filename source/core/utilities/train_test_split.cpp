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

#include "train_test_split.hpp"
#include "da_omp.hpp"
#include <algorithm>

namespace ARCH {
namespace da_utils {

template <typename T>
da_status validate_parameters_train_test_split(da_order order, da_int m, da_int n,
                                               const T *X, da_int ldx, da_int train_size,
                                               da_int test_size, T *X_train,
                                               da_int ldx_train, T *X_test,
                                               da_int ldx_test) {

    if (X == nullptr || X_train == nullptr || X_test == nullptr) {
        return da_status_invalid_pointer;
    }

    if (m < 2 || n < 1) {
        return da_status_invalid_array_dimension;
    }

    if (order == row_major) {
        if (ldx < n || ldx_train < n || ldx_test < n) {
            return da_status_invalid_leading_dimension;
        }
    } else if (order == column_major) {
        if (ldx < m || ldx_train < train_size || ldx_test < test_size) {
            return da_status_invalid_leading_dimension;
        }
    }

    if (train_size < 1 || test_size < 1) {
        return da_status_invalid_input;
    }
    if ((train_size + test_size) > m) {
        return da_status_invalid_input;
    }

    return da_status_success;
}

template <typename T>
da_status train_test_split(da_order order, da_int m, da_int n, const T *X, da_int ldx,
                           da_int train_size, da_int test_size,
                           const da_int *shuffle_array, T *X_train, da_int ldx_train,
                           T *X_test, da_int ldx_test) {
    da_status status = validate_parameters_train_test_split(
        order, m, n, X, ldx, train_size, test_size, X_train, ldx_train, X_test, ldx_test);
    if (status != da_status_success) {
        return status;
    }

    if (order == row_major) {
        da_int num_threads = (da_int)omp_get_max_threads();

        if (num_threads > 1) {
            split_row_data_parallel(n, train_size, X, ldx, X_train, ldx_train,
                                    shuffle_array);

            const da_int *test_indices =
                shuffle_array != nullptr ? shuffle_array + train_size : nullptr;
            const T *X_ptr = test_indices == nullptr ? X + (train_size * ldx) : X;

            split_row_data_parallel(n, test_size, X_ptr, ldx, X_test, ldx_test,
                                    test_indices);
        } else {
            split_row_data_serial(n, train_size, test_size, X, ldx, X_train, ldx_train,
                                  X_test, ldx_test, shuffle_array);
        }

    } else if (order == column_major) {

        const da_int *test_indices =
            shuffle_array != nullptr ? shuffle_array + train_size : nullptr;

        const T *X_ptr = shuffle_array == nullptr ? X + train_size : X;

#pragma omp parallel for schedule(static) default(none)                                  \
    shared(order, n, train_size, test_size, X, X_ptr, ldx, X_train, ldx_train, X_test,   \
               ldx_test, shuffle_array, test_indices)                                    \
    num_threads(std::min((da_int)omp_get_max_threads(), n))
        for (da_int i = 0; i < n; ++i) {
            split_column_data(train_size, X + (ldx * i), X_train + (ldx_train * i),
                              shuffle_array);
            split_column_data(test_size, X_ptr + (ldx * i), X_test + (ldx_test * i),
                              test_indices);
        }
    }

    return da_status_success;
}

template da_status validate_parameters_train_test_split<da_int>(
    da_order order, da_int m, da_int n, const da_int *X, da_int ldx, da_int train_size,
    da_int test_size, da_int *X_train, da_int ldx_train, da_int *X_test, da_int ldx_test);
template da_status validate_parameters_train_test_split<float>(
    da_order order, da_int m, da_int n, const float *X, da_int ldx, da_int train_size,
    da_int test_size, float *X_train, da_int ldx_train, float *X_test, da_int ldx_test);
template da_status validate_parameters_train_test_split<double>(
    da_order order, da_int m, da_int n, const double *X, da_int ldx, da_int train_size,
    da_int test_size, double *X_train, da_int ldx_train, double *X_test, da_int ldx_test);

template da_status train_test_split<da_int>(da_order order, da_int m, da_int n,
                                            const da_int *X, da_int ldx,
                                            da_int train_size, da_int test_size,
                                            const da_int *shuffle_array, da_int *X_train,
                                            da_int ldx_train, da_int *X_test,
                                            da_int ldx_test);
template da_status train_test_split<float>(da_order order, da_int m, da_int n,
                                           const float *X, da_int ldx, da_int train_size,
                                           da_int test_size, const da_int *shuffle_array,
                                           float *X_train, da_int ldx_train,
                                           float *X_test, da_int ldx_test);
template da_status train_test_split<double>(da_order order, da_int m, da_int n,
                                            const double *X, da_int ldx,
                                            da_int train_size, da_int test_size,
                                            const da_int *shuffle_array, double *X_train,
                                            da_int ldx_train, double *X_test,
                                            da_int ldx_test);

} // namespace da_utils
} // namespace ARCH