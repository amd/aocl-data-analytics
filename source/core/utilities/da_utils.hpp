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

#include "aoclda.h"
#include "boost/random/mersenne_twister.hpp"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "macros.h"
#include <cmath>
#include <math.h>
#include <type_traits>

namespace ARCH {

namespace da_arch {

const char *get_namespace(void);

}

namespace da_utils {

template <typename T> T hidden_settings_query(const std::string &key, T default_value);

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

void tall_skinny_blocking_scheme(da_int n_samples, da_int min_block_size,
                                 da_int max_blocks, da_int min_final_block_size,
                                 da_int &n_blocks, da_int &block_size,
                                 da_int &final_block_size);

da_int get_n_threads_loop(da_int loop_size);

template <typename T>
da_status check_categorical_data(da_int n_data, const T *data, da_int &n_categories,
                                 da_int max_categories, T tol);

template <typename U>
da_status check_1D_array(bool check_data, da_errors::da_error_t *err, da_int n,
                         const U *data, const std::string &n_name,
                         const std::string &data_name, da_int n_min);

template <typename T>
da_status stratified_shuffle(da_int m, boost::random::mt19937 &rand_engine,
                             da_int train_size, da_int test_size, const T *classes,
                             da_int *shuffled_indices);

template <typename T>
da_status convert_fp_classes(da_int m, da_int precision, const T *classes,
                             std::vector<int64_t> &int_classes);

template <typename T>
da_status get_shuffled_indices(da_int m, da_int seed, da_int train_size, da_int test_size,
                               da_int fp_precision, const T *classes,
                               da_int *shuffle_array);

template <typename T>
da_status validate_parameters_train_test_split(da_order order, da_int m, da_int n,
                                               const T *X, da_int ldx, da_int train_size,
                                               da_int test_size, T *X_train,
                                               da_int ldx_train, T *X_test,
                                               da_int ldx_test);

template <typename T>
da_status train_test_split(da_order order, da_int m, da_int n, const T *X, da_int ldx,
                           da_int train_size, da_int test_size,
                           const da_int *shuffle_array, T *X_train, da_int ldx_train,
                           T *X_test, da_int ldx_test);

template <typename T>
da_status check_categorical_data(da_int n_data, const T *data, da_int &n_categories,
                                 da_int max_categories, T tol);

CBLAS_ORDER da_order_to_cblas_order(da_order order);

da_order cblas_order_to_da_order(CBLAS_ORDER order);

CBLAS_UPLO da_uplo_to_cblas_uplo(da_uplo uplo);

da_uplo cblas_uplo_to_da_uplo(CBLAS_UPLO uplo);

CBLAS_TRANSPOSE da_transpose_to_cblas_transpose(da_transpose transpose);

da_transpose cblas_transpose_to_da_transpose(CBLAS_TRANSPOSE transpose);

} // namespace da_utils

} // namespace ARCH
