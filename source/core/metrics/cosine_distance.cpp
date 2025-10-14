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
#include "pairwise_distances.hpp"
#include "partitioning_infrastructure.hpp"

namespace ARCH {
namespace da_metrics {
namespace pairwise_distances {

// Functor for cosine distance calculation (packed version)
template <typename T, da_int MR, da_int NR> struct CosineKernelFunctor_packed {
    inline void operator()(da_int m, da_int n, da_int k, const T *Atilde, const T *Btilde,
                           T *C, da_int ldc, T *C_temp) const {

        // Call microkernel if we have a full block
        if ((m == MR) && (n == NR)) {
#if defined(__AVX512F__)
            avx512::cosine_kernel_packed<T, MR, NR>(k, Atilde, Btilde, C, ldc);
#elif defined(__AVX2__)
            avx2::cosine_kernel_packed<T, MR, NR>(k, Atilde, Btilde, C, ldc);
#endif
        } else { //Call simple kernel if we have a partial block
            for (da_int j = 0; j < n; j++)
                for (da_int i = 0; i < m; i++)
                    ctemp_matrix(i, j) = c_matrix(i, j);

#if defined(__AVX512F__)
            avx512::cosine_kernel_packed<T, MR, NR>(k, Atilde, Btilde, C_temp, MR);
#elif defined(__AVX2__)
            avx2::cosine_kernel_packed<T, MR, NR>(k, Atilde, Btilde, C_temp, MR);
#endif

            for (da_int j = 0; j < n; j++)
                for (da_int i = 0; i < m; i++)
                    c_matrix(i, j) = ctemp_matrix(i, j);
        }
    }
};

// Functor for Cosine distance calculation
template <typename T> struct CosinePostOp {
    std::vector<T> normX;
    std::vector<T> normY;
    bool X_is_Y = false;
    CosinePostOp(std::vector<T> &&normX, std::vector<T> &&normY, bool X_is_Y)
        : normX(std::move(normX)), normY(std::move(normY)), X_is_Y(X_is_Y) {}
    inline void operator()(da_int m, da_int n, T *C, da_int ldc) const {
        if (!X_is_Y) {
            for (da_int j = 0; j < n; j++) {
                for (da_int i = 0; i < m; i++) {
                    C[i + j * ldc] = 1.0 - C[i + j * ldc] / (normX[i] * normY[j]);
                }
            }
        } else {
            for (da_int j = 0; j < n; j++) {
                for (da_int i = 0; i < m; i++) {
                    C[i + j * ldc] = 1.0 - C[i + j * ldc] / (normX[i] * normY[j]);
                }
                C[j + j * ldc] = 0.0;
            }
        }
    }
};

template <typename T>
da_status cosine(da_order order, da_int m, da_int n, da_int k, const T *X, da_int ldx,
                 const T *Y, da_int ldy, T *D, da_int ldd) {

    bool X_is_Y = false;
    // We want to compute the distance of X to itself
    // The sizes are copies so it's safe to update them
    const T *Y_new = Y;
    if (!Y) {
        n = m;
        ldy = ldx;
        Y_new = X;
        X_is_Y = true;
    }
    // Compute the norms for all rows of X and Y
    std::vector<T> normsX(m);
    std::vector<T> normsY(n);
    const T eps = std::numeric_limits<T>::epsilon();
    // Open scope so that memory gets deallocated before calling the computational kernel
    if (order == column_major) {
        std::vector<T> X_row, Y_row, D_row;
        try {
            X_row.resize(m * k);
            Y_row.resize(n * k);
            D_row.resize(m * n);
        } catch (std::bad_alloc const &) {
            return da_status_memory_error;
        }
        // Transpose X and Y so that the data is stored in row major order
        da_blas::omatcopy('T', m, k, 1.0, X, ldx, X_row.data(), k);
        da_blas::omatcopy('T', n, k, 1.0, Y_new, ldy, Y_row.data(), k);
        // Now that everything is in row major order, compute the norms
        for (da_int i = 0; i < m; i++) {
            normsX[i] = da_blas::cblas_nrm2(k, X_row.data() + i * k, 1);
            if (normsX[i] <= eps) {
                normsX[i] = 1.0;
            }
        }
        for (da_int j = 0; j < n; j++) {
            normsY[j] = da_blas::cblas_nrm2(k, Y_row.data() + j * k, 1);
            if (normsY[j] <= eps) {
                normsY[j] = 1.0;
            }
        }
    } else {
        for (da_int i = 0; i < m; i++) {
            normsX[i] = da_blas::cblas_nrm2(k, X + i * ldx, 1);
            if (normsX[i] <= eps) {
                normsX[i] = 1.0;
            }
        }
        for (da_int j = 0; j < n; j++) {
            normsY[j] = da_blas::cblas_nrm2(k, Y_new + j * ldy, 1);
            if (normsY[j] <= eps) {
                normsY[j] = 1.0;
            }
        }
    }

    constexpr da_int MR = BlockSizes<T>::MR;
    constexpr da_int NR = BlockSizes<T>::NR;
    constexpr da_int MC = BlockSizes<T>::MC;
    constexpr da_int NC = BlockSizes<T>::NC;
    constexpr da_int KC = BlockSizes<T>::KC;

    CosineKernelFunctor_packed<T, MR, NR> kernel_op;

    // Constructor moves normsX and normsY into the functor
    CosinePostOp<T> post_op(std::move(normsX), std::move(normsY), X_is_Y);
    if (order == row_major) {
        // Create temporary vectors X_col and Y_col
        std::vector<T> X_col(m * k), Y_col(n * k), D_col(m * n);
        // Transpose X_col and Y_col so that now the data of X and Y are
        // stored in column major order
        da_blas::omatcopy('T', k, m, 1.0, X, ldx, X_col.data(), m);
        da_blas::omatcopy('T', k, n, 1.0, Y_new, ldy, Y_col.data(), n);
        LoopFive_packed<T, MR, NR, MC, NC, KC>(m, n, k, X_col.data(), m, Y_col.data(), n,
                                               D_col.data(), m, kernel_op, post_op);
        // Transpose D to return data in column major order
        da_blas::omatcopy('T', m, n, 1.0, D_col.data(), m, D, ldd);
    } else {
        LoopFive_packed<T, MR, NR, MC, NC, KC>(m, n, k, X, ldx, Y_new, ldy, D, ldd,
                                               kernel_op, post_op);
    }

    return da_status_success;
}

template da_status cosine<float>(da_order order, da_int m, da_int n, da_int k,
                                 const float *X, da_int ldx, const float *Y, da_int ldy,
                                 float *D, da_int ldd);

template da_status cosine<double>(da_order order, da_int m, da_int n, da_int k,
                                  const double *X, da_int ldx, const double *Y,
                                  da_int ldy, double *D, da_int ldd);
} // namespace pairwise_distances
} // namespace da_metrics
} // namespace ARCH