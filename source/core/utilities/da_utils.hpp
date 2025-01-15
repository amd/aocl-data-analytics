/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef DA_UTILITIES_HPP
#define DA_UTILITIES_HPP

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "aoclda.h"
#include "da_cblas.hh"
#include <cmath>
#include <math.h>
#include <type_traits>

namespace da_utils {

/* Convert number into char array, appropriately depending on its type */
template <typename T, size_t U>
constexpr da_status convert_num_to_char(T num, char character[U]) {
    static_assert(std::is_arithmetic_v<T>,
                  "Error in convert_num_to_char function. T must be numerical "
                  "value");
    if constexpr (std::is_same_v<T, float>)
        sprintf(character, "%9.2e", num);
    else if constexpr (std::is_same_v<T, double>)
        sprintf(character, "%9.2e", num);
    else if constexpr (std::is_same_v<T, int>)
        sprintf(character, "%d", num);
    else if constexpr (std::is_same_v<T, long int>)
        sprintf(character, "%ld", num);
    else if constexpr (std::is_same_v<T, long long int>)
        sprintf(character, "%lld", num);
    return da_status_success;
}

template <typename T>
void copy_transpose_2D_array_row_to_column_major(da_int n_rows, da_int n_cols, const T *A,
                                                 da_int lda, T *B, da_int ldb) {
    for (da_int i = 0; i < n_rows; i++) {
        for (da_int j = 0; j < n_cols; j++) {
            B[j * ldb + i] = A[i * lda + j];
        }
    }
}

template <typename T>
void copy_transpose_2D_array_column_to_row_major(da_int n_rows, da_int n_cols, const T *A,
                                                 da_int lda, T *B, da_int ldb) {
    for (da_int j = 0; j < n_cols; j++) {
        for (da_int i = 0; i < n_rows; i++) {
            B[j + i * ldb] = A[i + j * lda];
        }
    }
}

template <typename T>
da_status check_data(da_order order, da_int n_rows, da_int n_cols, const T *X,
                     da_int ldx) {
    if (n_rows < 1 || n_cols < 1)
        return da_status_invalid_array_dimension;

    if (X == nullptr)
        return da_status_invalid_pointer;

    if (order == row_major) {
        if (ldx < n_cols)
            return da_status_invalid_leading_dimension;
        for (da_int i = 0; i < n_rows; i++) {
            for (da_int j = 0; j < n_cols; j++) {
                // x==x+1 check needed to get round a pybind1 + clang 18 Windows release build bug
                if (std::isnan(X[i * ldx + j]) ||
                    X[i * ldx + j] == X[i * ldx + j] + (T)1) {
                    return da_status_invalid_input;
                }
            }
        }
    } else {
        if (ldx < n_rows)
            return da_status_invalid_leading_dimension;
        for (da_int j = 0; j < n_cols; j++) {
            for (da_int i = 0; i < n_rows; i++) {
                if (std::isnan(X[i + j * ldx]) ||
                    X[i + j * ldx] == X[i + j * ldx] + (T)1) {
                    return da_status_invalid_input;
                }
            }
        }
    }
    return da_status_success;
}

template <typename T>
da_status switch_order_copy(da_order order, da_int n_rows, da_int n_cols, const T *X,
                            da_int ldx, T *Y, da_int ldy) {
    if (n_rows < 1 || n_cols < 1)
        return da_status_invalid_array_dimension;
    if (X == nullptr || Y == nullptr)
        return da_status_invalid_pointer;

    if (order == row_major) {
        if (ldy < n_rows || ldx < n_cols)
            return da_status_invalid_leading_dimension;
        copy_transpose_2D_array_row_to_column_major(n_rows, n_cols, X, ldx, Y, ldy);
    } else {
        if (ldx < n_rows || ldy < n_cols)
            return da_status_invalid_leading_dimension;
        copy_transpose_2D_array_column_to_row_major(n_rows, n_cols, X, ldx, Y, ldy);
    }

    return da_status_success;
}

template <typename T>
da_status switch_order_in_place(da_order order_X_in, da_int n_rows, da_int n_cols, T *X,
                                da_int ldx_in, da_int ldx_out) {
    if (n_rows < 1 || n_cols < 1)
        return da_status_invalid_array_dimension;
    if (X == nullptr)
        return da_status_invalid_pointer;

    if (order_X_in == row_major) {
        if (ldx_out < n_rows || ldx_in < n_cols)
            return da_status_invalid_leading_dimension;
        da_blas::imatcopy('T', n_cols, n_rows, (T)1.0, X, ldx_in, ldx_out);
    } else {
        if (ldx_in < n_rows || ldx_out < n_cols)
            return da_status_invalid_leading_dimension;
        da_blas::imatcopy('T', n_rows, n_cols, (T)1.0, X, ldx_in, ldx_out);
    }

    return da_status_success;
}

void blocking_scheme(da_int n_samples, da_int block_size, da_int &n_blocks,
                     da_int &block_rem);

da_int get_n_threads_loop(da_int loop_size);

} // namespace da_utils

#endif
