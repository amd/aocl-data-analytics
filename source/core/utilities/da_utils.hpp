/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "aoclda.h"
#include "da_cblas.hh"
#include "macros.h"
#include <cmath>
#include <math.h>
#include <type_traits>

namespace ARCH {

namespace da_arch {

const char *get_namespace(void);

}

namespace da_utils {

template <typename T>
void copy_transpose_2D_array_row_to_column_major(da_int n_rows, da_int n_cols, const T *A,
                                                 da_int lda, T *B, da_int ldb);
template <typename T>
void copy_transpose_2D_array_column_to_row_major(da_int n_rows, da_int n_cols, const T *A,
                                                 da_int lda, T *B, da_int ldb);

template <typename T>
da_status check_data(da_order order, da_int n_rows, da_int n_cols, const T *X,
                     da_int ldx);

template <typename T>
da_status switch_order_copy(da_order order, da_int n_rows, da_int n_cols, const T *X,
                            da_int ldx, T *Y, da_int ldy);

template <typename T>
da_status switch_order_in_place(da_order order_X_in, da_int n_rows, da_int n_cols, T *X,
                                da_int ldx_in, da_int ldx_out);

void blocking_scheme(da_int n_samples, da_int block_size, da_int &n_blocks,
                     da_int &block_rem);

da_int get_n_threads_loop(da_int loop_size);

} // namespace da_utils

} // namespace ARCH
