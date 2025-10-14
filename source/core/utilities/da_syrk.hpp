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

#include "aoclda.h"
#include "context.hpp"
#include "da_cblas.hh"
#include "da_omp.hpp"
#include "da_utils.hpp"
#include "macros.h"
#include "miscellaneous.hpp"
#include <algorithm>
#include <string>
#include <vector>

/*  This function implements a blocked, parallelized syrk algorithm for tall
    skinny matrices.

    If internal heuristics suggest that blocked syrk is not warranted for the
    given matrix size, then syrk is simply called instead.

    The following requirements must be met to perform blocking:
        1. k > n else the 1D blocking scheme used here is not appropriate
        2. block_size > 2 * n. Experiments suggest blocks must be at least this
            tall to justify this approach.
        3. block_size > 2048 to prevent excessively small syrk calls.
        4. block_size to be rounded up to the nearest multiple of 256 for
            better cache use.
        5. block_size < k.
        6. If we can't create at least (n_threads - 1) blocks of sufficiently
            large size from the previous requirements, just call regular syrk.
        7. n_blocks < 512 (or some other large number).
        8. The size of the last block, must be no smaller than n to avoid a
            short wide syrk.
*/

namespace ARCH {

using namespace std::literals::string_literals;

const da_int SYRK_MIN_BLOCK_SIZE = 2048;
const da_int SYRK_MAX_NUM_BLOCKS = 512;

template <typename T>
da_status da_syrk(da_order order, da_uplo uplo, da_transpose transpose, da_int n,
                  da_int k, T alpha, T const *A, da_int ldA, T beta, T *C, da_int ldC) {

    // quick return if possible
    if (alpha == (T)0.0 && beta == (T)1.0) {
        return da_status_success;
    }

    da_int n_threads, n_blocks, block_size, final_block_size;

    n_threads = (da_int)omp_get_max_threads();

    if (omp_get_max_active_levels() == omp_get_level())
        n_threads = 1;

    da_int min_block_size = SYRK_MIN_BLOCK_SIZE;
    // Experiments suggest each block must be at least twice as tall as
    // wide to justify the tall skinny approach. If we can't occupy each available thread
    // with a block that tall, we just call regular syrk.

    da_int max_blocks = std::max(
        std::min((da_int)std::ceil((T)k / (2 * n)), SYRK_MAX_NUM_BLOCKS), (da_int)1);

    bool block_override = false;
    const char block_size_override[]{"syrk.block_size_override"};
    auto &settings = context::get_context()->get_hidden_settings();
    if (settings.find(block_size_override) != settings.end()) {
        // Override min_block_size
        std::string block_size_str = settings[block_size_override];

        min_block_size = static_cast<da_int>(std::stoi(block_size_str));
        // max_blocks just has to be something reasonably large so it doesn't
        // influence the desired blocking override
        max_blocks = 512;

        block_override = true;
    }

    if (k <= n || max_blocks == 1) {
        n_blocks = 1;
        block_size = std::max(k, n);
        final_block_size = block_size;
    } else {
        da_utils::tall_skinny_blocking_scheme(k, min_block_size, max_blocks, n, n_blocks,
                                              block_size, final_block_size);

        // If there aren't enough sufficiently tall blocks, or there is only one block, and an
        // override hasn't been set, then don't block
        if (((n_blocks < n_threads - 1) || (n_blocks == 1)) && !block_override) {
            n_blocks = 1;
            block_size = k;
            final_block_size = k;
        }
    }

    // Add telemetry of block size used
    context_set_hidden_settings("syrk.block_size"s, std::to_string(block_size));

    CBLAS_ORDER cblas_order = da_utils::da_order_to_cblas_order(order);
    CBLAS_UPLO cblas_uplo = da_utils::da_uplo_to_cblas_uplo(uplo);
    CBLAS_TRANSPOSE cblas_transpose =
        da_utils::da_transpose_to_cblas_transpose(transpose);

    if (n_blocks == 1) {
        da_blas::cblas_syrk(cblas_order, cblas_uplo, cblas_transpose, n, k, alpha, A, ldA,
                            beta, C, ldC);
        return da_status_success;
    }

    n_threads = std::min(n_threads, n_blocks);

    // main computation
    // Store blocked results in C_tmp
    // Each n by n result is contiguous in C_tmp
    // thread 0 will write directly to C
    std::vector<T> C_tmp;
    da_int n_tmp_blocks = n_threads - 1;
    da_int C_tmp_size = n_tmp_blocks * n * n;
    try {
        C_tmp.resize(C_tmp_size);
    } catch (std::bad_alloc const &) {
        return da_status_memory_error;
    }

    da_int A_block_offset;

    if (transpose == da_no_trans) {
        A_block_offset = (order == column_major) ? ldA * block_size : block_size;
    } else {
        A_block_offset = (order == column_major) ? block_size : ldA * block_size;
    }

    // Compute blockwise syrk
    da_int this_block_size = block_size;
    da_int this_block_ldC = n;
    T this_block_beta;
    da_int this_thread_counter = 0;
#pragma omp parallel for if (n_blocks > 1) num_threads(n_threads)                        \
    schedule(static) default(none)                                                       \
    firstprivate(this_block_size, this_thread_counter,                                   \
                     this_block_ldC) private(this_block_beta)                            \
    shared(C_tmp, n_blocks, final_block_size, cblas_order, cblas_uplo, cblas_transpose,  \
               n, alpha, beta, A, ldA, A_block_offset, ldC, C)
    for (da_int i = 0; i < n_blocks; i++) {
        if (i == n_blocks - 1) {
            this_block_size = final_block_size;
        }

        da_int thread_num = omp_get_thread_num();

        T *this_block_C;

        if (thread_num == 0) {
            this_block_C = C;
            this_block_beta = (this_thread_counter == 0) ? beta : (T)1.0;
            this_block_ldC = ldC;
        } else {
            this_block_C = &C_tmp[(thread_num - 1) * n * n];
            this_block_beta = (this_thread_counter == 0) ? (T)0.0 : (T)1.0;
        }

        da_blas::cblas_syrk(cblas_order, cblas_uplo, cblas_transpose, n, this_block_size,
                            alpha, &A[i * A_block_offset], ldA, this_block_beta,
                            this_block_C, this_block_ldC);

        this_thread_counter++;
    }

    // Accumulate blocks
    if (uplo == da_upper) {
        if (order == column_major) {
            // upper, col major
            for (da_int ii = 0; ii < n_tmp_blocks; ii++) {
                for (da_int j = 0; j < n; j++) {
#pragma omp simd
                    for (da_int i = 0; i <= j; i++) {
                        C[i + ldC * j] += C_tmp[ii * n * n + i + n * j];
                    }
                }
            }
        } else if (order == row_major) {
            // upper, row major
            for (da_int ii = 0; ii < n_tmp_blocks; ii++) {
                for (da_int i = 0; i < n; i++) {
#pragma omp simd
                    for (da_int j = i; j < n; j++) {
                        C[j + ldC * i] += C_tmp[ii * n * n + j + n * i];
                    }
                }
            }
        }
    } else if (uplo == da_lower) {
        if (order == column_major) {
            // lower, col major
            for (da_int ii = 0; ii < n_tmp_blocks; ii++) {
                for (da_int j = 0; j < n; j++) {
#pragma omp simd
                    for (da_int i = j; i < n; i++) {
                        C[i + ldC * j] += C_tmp[ii * n * n + i + n * j];
                    }
                }
            }
        } else if (order == row_major) {
            // lower, row major
            for (da_int ii = 0; ii < n_tmp_blocks; ii++) {
                for (da_int i = 0; i < n; i++) {
#pragma omp simd
                    for (da_int j = 0; j <= i; j++) {
                        C[j + ldC * i] += C_tmp[ii * n * n + j + n * i];
                    }
                }
            }
        }
    }
    return da_status_success;
}
} // namespace ARCH
