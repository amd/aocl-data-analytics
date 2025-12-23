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
#include "aoclda_types.h"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "da_std.hpp"
#include "macros.h"
#include "pairwise_distances.hpp"
#include "partitioning_infrastructure.hpp"
// Functor for SqEuclidean distance calculation

namespace ARCH {

namespace da_metrics {
namespace pairwise_distances {

// Functor for SqEuclidean distance calculation
template <typename T, da_int MR, da_int NR> struct SqEuclideanKernelFunctor_packed {
    inline void operator()(da_int m, da_int n, da_int k, const T *Atilde, const T *Btilde,
                           T *C, da_int ldc, T *C_temp) const {

        // Call microkernel if we have a full block
        if ((m == MR) && (n == NR)) {
#if defined(__AVX512F__)
            avx512::sqeuclidean_kernel_packed<T, MR, NR>(k, Atilde, Btilde, C, ldc);
#elif defined(__AVX2__)
            avx2::sqeuclidean_kernel_packed<T, MR, NR>(k, Atilde, Btilde, C, ldc);
#endif
        } else { //Call simple kernel if we have a partial block
            for (da_int j = 0; j < n; j++)
                for (da_int i = 0; i < m; i++)
                    ctemp_matrix(i, j) = c_matrix(i, j);

#if defined(__AVX512F__)
            avx512::sqeuclidean_kernel_packed<T, MR, NR>(k, Atilde, Btilde, C_temp, MR);
#elif defined(__AVX2__)
            avx2::sqeuclidean_kernel_packed<T, MR, NR>(k, Atilde, Btilde, C_temp, MR);
#endif

            for (da_int j = 0; j < n; j++)
                for (da_int i = 0; i < m; i++)
                    c_matrix(i, j) = ctemp_matrix(i, j);
        }
    }
};

// Functor for Euclidean distance calculation
template <typename T> struct SqrtPostOp {
    inline void operator()(da_int m, da_int n, T *C, da_int ldc) const {
        for (da_int j = 0; j < n; j++) {
            for (da_int i = 0; i < m; i++) {
                C[i + j * ldc] = std::sqrt(C[i + j * ldc]);
            }
        }
    }
};

template <typename T>
da_status euclidean(da_order order, da_int m, da_int n, da_int k, const T *X, da_int ldx,
                    const T *Y, da_int ldy, T *D, da_int ldd) {
    constexpr da_int MR = BlockSizes<T>::MR;
    constexpr da_int NR = BlockSizes<T>::NR;
    constexpr da_int MC = BlockSizes<T>::MC;
    constexpr da_int NC = BlockSizes<T>::NC;
    constexpr da_int KC = BlockSizes<T>::KC;

    SqEuclideanKernelFunctor_packed<T, MR, NR> kernel_op;
    SqrtPostOp<T> sqrt_post_op;

    if (order == row_major) {
        const T *Y_new = Y;
        // We want to compute the distance of X to itself
        // The sizes are copies so it's safe to update them
        if (!Y) {
            n = m;

            ldy = ldx;
            Y_new = X;
        }

        // Create temporary vectors X_col and Y_col
        std::vector<T> X_col(m * k), Y_col(n * k), D_col(m * n);
        // Transpose X_col and Y_col so that now the data of X and Y are
        // stored in column major order
        da_blas::omatcopy('T', k, m, 1.0, X, ldx, X_col.data(), m);
        da_blas::omatcopy('T', k, n, 1.0, Y_new, ldy, Y_col.data(), n);
        LoopFive_packed<T, MR, NR, MC, NC, KC>(m, n, k, X_col.data(), m, Y_col.data(), n,
                                               D_col.data(), m, kernel_op, sqrt_post_op);
        // Transpose D to return data in column major order
        da_blas::omatcopy('T', m, n, 1.0, D_col.data(), m, D, ldd);
    } else {
        const T *Y_new = Y;
        // We want to compute the distance of X to itself
        // The sizes are copies so it's safe to update them
        if (!Y) {
            n = m;
            ldy = ldx;
            Y_new = X;
        }
        LoopFive_packed<T, MR, NR, MC, NC, KC>(m, n, k, X, ldx, Y_new, ldy, D, ldd,
                                               kernel_op, sqrt_post_op);
    }
    return da_status_success;
}

template da_status euclidean<float>(da_order order, da_int m, da_int n, da_int k,
                                    const float *X, da_int ldx, const float *Y,
                                    da_int ldy, float *D, da_int ldd);

template da_status euclidean<double>(da_order order, da_int m, da_int n, da_int k,
                                     const double *X, da_int ldx, const double *Y,
                                     da_int ldy, double *D, da_int ldd);

} // namespace pairwise_distances
} // namespace da_metrics
} // namespace ARCH
