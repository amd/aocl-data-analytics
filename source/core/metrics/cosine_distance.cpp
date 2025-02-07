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
#include "pairwise_distances.hpp"

namespace ARCH {
namespace da_metrics {
namespace pairwise {

template <typename T>
da_status cosine(da_order order, da_int m, da_int n, da_int k, const T *X, da_int ldx,
                 const T *Y, da_int ldy, T *D, da_int ldd, bool compute_distance) {
    da_status status = da_status_success;
    const T *Y_new = Y;
    const T eps = std::numeric_limits<T>::epsilon();
    T normX, normY;
    // We want to compute the distance of X to itself
    // The sizes are copies so it's safe to update them
    if (Y == nullptr) {
        n = m;
        ldy = ldx;
        Y_new = X;
    }

    T *D_new = D;
    da_int ldd_new = ldd;
    const T *X_new = X;
    // Create temporary vectors X_row and Y_row
    std::vector<T> X_row, Y_row, D_row;
    if (order == column_major) {
        try {
            X_row.resize(m * k);
            Y_row.resize(n * k);
            D_row.resize(m * n);
        } catch (std::bad_alloc const &) {
            return da_status_memory_error;
        }
        // Transpose X and Y so that the data is stored in row major order
        da_blas::omatcopy('T', m, k, 1.0, X_new, ldx, X_row.data(), k);
        da_blas::omatcopy('T', n, k, 1.0, Y_new, ldy, Y_row.data(), k);
        D_new = D_row.data();
        X_new = X_row.data();
        Y_new = Y_row.data();
        ldx = k;
        ldy = k;
        ldd_new = n;
    }

    if (Y != nullptr) {
        // Go through the rows of D
        for (da_int i = 0; i < m; i++) {
            // Go through the columns of X (also updating the columns of D)
            for (da_int j = 0; j < n; j++) {
                D_new[i * ldd_new + j] =
                    da_blas::cblas_dot(k, X_new + i * ldx, 1, Y_new + j * ldy, 1);
                // Only compute the norms if Dij is nonzero
                if (std::abs(D_new[i * ldd_new + j]) > eps) {
                    normX = da_blas::cblas_nrm2(k, X_new + i * ldx, 1);
                    normY = da_blas::cblas_nrm2(k, Y_new + j * ldy, 1);
                } else {
                    normX = 1.0;
                    normY = 1.0;
                }
                D_new[i * ldd_new + j] = D_new[i * ldd_new + j] / (normX * normY);
                if (compute_distance)
                    D_new[i * ldd_new + j] = 1.0 - D_new[i * ldd_new + j];
            }
        }
    } else {
        // Go through the rows of D
        for (da_int i = 0; i < m; i++) {
            // Go through the columns of X (also updating the columns of D)
            // For the case where X==Y, the matrix is symmetric so we only need to iterate through half the columns
            for (da_int j = i + 1; j < n; j++) {
                D_new[i * ldd_new + j] =
                    da_blas::cblas_dot(k, X_new + i * ldx, 1, Y_new + j * ldy, 1);
                // Only compute the norms if Dij is nonzero
                if (std::abs(D_new[i * ldd_new + j]) > eps) {
                    normX = da_blas::cblas_nrm2(k, X_new + i * ldx, 1);
                    normY = da_blas::cblas_nrm2(k, Y_new + j * ldy, 1);
                } else {
                    normX = 1.0;
                    normY = 1.0;
                }
                D_new[i * ldd_new + j] = D_new[i * ldd_new + j] / (normX * normY);
                if (compute_distance)
                    D_new[i * ldd_new + j] = 1.0 - D_new[i * ldd_new + j];
                // Update the corresponding element in the lower triangular part
                D_new[j * ldd_new + i] = D_new[i * ldd_new + j];
            }
            D[i * ldd_new + i] = 0.0;
        }
    }
    if (order == column_major) {
        // Transpose D to return data in column major order
        da_blas::omatcopy('T', n, m, 1.0, D_new, ldd_new, D, ldd);
    }

    return status;
}

} // namespace pairwise

namespace pairwise_distances {

template <typename T>
da_status cosine(da_order order, da_int m, da_int n, da_int k, const T *X, da_int ldx,
                 const T *Y, da_int ldy, T *D, da_int ldd) {
    return da_metrics::pairwise::cosine(order, m, n, k, X, ldx, Y, ldy, D, ldd, true);
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