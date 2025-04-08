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

namespace ARCH {
namespace da_metrics {
namespace pairwise_distances {

template <typename T>
da_status minkowski(da_order order, da_int m, da_int n, da_int k, const T *X, da_int ldx,
                    const T *Y, da_int ldy, T *D, da_int ldd, T p) {
    da_status status = da_status_success;
    const T *Y_new = Y;
    // We want to compute the distance of X to itself
    // The sizes are copies so it's safe to update them
    if (Y == nullptr) {
        n = m;
        ldy = ldx;
        Y_new = X;
    }
    T elementwise = 0.0;
    T invp = 1.0 / p;
    if (order == column_major) {
        // Go through the columns of D
        for (da_int j = 0; j < n; j++) {
            // Fill the column Dj with zeros
            da_std::fill(D + j * ldd, D + j * ldd + m, 0.0);
            // Go through the rows of X (also updating the rows of D)
            for (da_int i = 0; i < m; i++) {
                // Go through the columns of both X and Y
                for (da_int l = 0; l < k; l++) {
                    elementwise = std::abs(X[i + l * ldx] - Y_new[j + l * ldy]);
                    elementwise = std::pow(elementwise, p);
                    D[i + j * ldd] += elementwise;
                }
                D[i + j * ldd] = std::pow(D[i + j * ldd], invp);
            }
        }
    } else {
        // Go through the rows of D
        for (da_int i = 0; i < m; i++) {
            // Fill the row Di with zeros
            da_std::fill(D + i * ldd, D + i * ldd + n, 0.0);
            // Go through the columns of X (also updating the columns of D)
            for (da_int j = 0; j < n; j++) {
                // Go through the columns of both X and Y
                for (da_int l = 0; l < k; l++) {
                    elementwise = std::abs(X[i * ldx + l] - Y_new[j * ldy + l]);
                    elementwise = std::pow(elementwise, p);
                    D[i * ldd + j] += elementwise;
                }
                D[i * ldd + j] = std::pow(D[i * ldd + j], invp);
            }
        }
    }

    return status;
}

template da_status minkowski<float>(da_order order, da_int m, da_int n, da_int k,
                                    const float *X, da_int ldx, const float *Y,
                                    da_int ldy, float *D, da_int ldd, float p);

template da_status minkowski<double>(da_order order, da_int m, da_int n, da_int k,
                                     const double *X, da_int ldx, const double *Y,
                                     da_int ldy, double *D, da_int ldd, double p);
} // namespace pairwise_distances
} // namespace da_metrics
} // namespace ARCH