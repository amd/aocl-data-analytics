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

#pragma once

#include "aoclda.h"
#include "aoclda_types.h"
#include "da_cblas.hh"
#include "da_error.hpp"
#include <iostream>
#include <vector>

/*
An important kernel used repeatedly in k-means and SVM computations, among others.
Given an m by k matrix X and an n by k matrix Y (both column major), computes the m by n matrix D, where
D_{ij} is the Euclidean distance between row i of X and row j of Y.
Computes the distance by forming the norms of the rows of X and Y and computing XY^T which is more efficient
Various options are available:
- the squared norms of the rows of X and Y can be supplied precomputed or not used at all:
    compute_X/Y_norms = 0: do not use at all - note this is risky if you want to set square to be false as the function doesn't test for negative outputs
    compute_X/Y_norms = 1: use precomputed versions
    compute_X/Y_norms = 2: compute them in this function
- the square of the Euclidean distances can be returned if square is set to true
- if X_is_Y is true then X and Y are taken to be the same matrix, so only X is referenced and syrk is used
  instead of gemm and only the upper triangle is referenced and stored. Need m=n, otherwise garbage will come out
*/
template <typename T>
inline void euclidean_distance(da_order order, da_int m, da_int n, da_int k, const T *X,
                               da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd,
                               T *X_norms, da_int compute_X_norms, T *Y_norms,
                               da_int compute_Y_norms, bool square, bool X_is_Y) {

    CBLAS_ORDER cblas_order =
        (order == column_major) ? CBLAS_ORDER::CblasColMajor : CBLAS_ORDER::CblasRowMajor;

    // If needed, compute the squared norms of the rows of X and Y
    if (compute_X_norms == 2) {
        for (da_int i = 0; i < m; i++) {
            X_norms[i] = 0.0;
        }
        if (order == column_major) {
            for (da_int j = 0; j < k; j++) {
                for (da_int i = 0; i < m; i++) {
                    X_norms[i] += X[i + j * ldx] * X[i + j * ldx];
                }
            }
        } else {
            for (da_int i = 0; i < m; i++) {
                for (da_int j = 0; j < k; j++) {
                    X_norms[i] += X[i * ldx + j] * X[i * ldx + j];
                }
            }
        }
    }

    if (compute_Y_norms == 2 && !(X_is_Y)) {
        for (da_int i = 0; i < n; i++) {
            Y_norms[i] = 0.0;
        }
        if (order == column_major) {
            for (da_int j = 0; j < k; j++) {
                for (da_int i = 0; i < n; i++) {
                    Y_norms[i] += Y[i + j * ldy] * Y[i + j * ldy];
                }
            }
        } else {
            for (da_int i = 0; i < n; i++) {
                for (da_int j = 0; j < k; j++) {
                    Y_norms[i] += Y[i * ldy + j] * Y[i * ldy + j];
                }
            }
        }
    }

    if (!X_is_Y) {
        // A few different cases to check depending on the boolean inputs

        if (compute_X_norms == 0 && compute_Y_norms == 0) {
            if (order == column_major) {
                for (da_int j = 0; j < n; j++) {
                    for (da_int i = 0; i < m; i++) {
                        D[i + j * ldd] = 0.0;
                    }
                }
            } else {
                for (da_int i = 0; i < m; i++) {
                    for (da_int j = 0; j < n; j++) {
                        D[i * ldd + j] = 0.0;
                    }
                }
            }
        } else if (compute_X_norms > 0 && compute_Y_norms == 0) {
            if (order == column_major) {
                for (da_int j = 0; j < n; j++) {
                    for (da_int i = 0; i < m; i++) {
                        D[i + j * ldd] = X_norms[i];
                    }
                }
            } else {
                for (da_int i = 0; i < m; i++) {
                    for (da_int j = 0; j < n; j++) {
                        D[i * ldd + j] = X_norms[i];
                    }
                }
            }
        } else if (compute_X_norms == 0 && compute_Y_norms > 0) {
            if (order == column_major) {
                for (da_int j = 0; j < n; j++) {
                    for (da_int i = 0; i < m; i++) {
                        D[i + j * ldd] = Y_norms[j];
                    }
                }
            } else {
                for (da_int i = 0; i < m; i++) {
                    for (da_int j = 0; j < n; j++) {
                        D[i * ldd + j] = Y_norms[j];
                    }
                }
            }
        } else {
            if (order == column_major) {
                for (da_int j = 0; j < n; j++) {
                    for (da_int i = 0; i < m; i++) {
                        D[i + j * ldd] = X_norms[i] + Y_norms[j];
                    }
                }
            } else {
                for (da_int i = 0; i < m; i++) {
                    for (da_int j = 0; j < n; j++) {
                        D[i * ldd + j] = X_norms[i] + Y_norms[j];
                    }
                }
            }
        }

        da_blas::cblas_gemm(cblas_order, CblasNoTrans, CblasTrans, m, n, k, -2.0, X, ldx,
                            Y, ldy, 1.0, D, ldd);

        if (!square) {
            if (order == column_major) {
                for (da_int j = 0; j < n; j++) {
                    for (da_int i = 0; i < m; i++) {
                        D[i + j * ldd] = std::sqrt(D[i + j * ldd]);
                    }
                }
            } else {
                for (da_int i = 0; i < m; i++) {
                    for (da_int j = 0; j < n; j++) {
                        D[i * ldd + j] = std::sqrt(D[i * ldd + j]);
                    }
                }
            }
        }
    } else {
        // Special case when computing upper triangle of symmetric distance matrix

        if (compute_X_norms == 0) {
            if (order == column_major) {
                for (da_int j = 0; j < m; j++) {
                    for (da_int i = 0; i <= j; i++) {
                        D[i + j * ldd] = 0.0;
                    }
                }
            } else {
                for (da_int j = 0; j < m; j++) {
                    for (da_int i = 0; i <= j; i++) {
                        D[i * ldd + j] = 0.0;
                    }
                }
            }
        } else {
            if (order == column_major) {
                for (da_int j = 0; j < m; j++) {
                    for (da_int i = 0; i <= j; i++) {
                        D[i + j * ldd] = X_norms[i] + X_norms[j];
                    }
                }
            } else {
                for (da_int j = 0; j < m; j++) {
                    for (da_int i = 0; i <= j; i++) {
                        D[i * ldd + j] = X_norms[i] + X_norms[j];
                    }
                }
            }
        }

        da_blas::cblas_syrk(cblas_order, CblasUpper, CblasNoTrans, m, k, -2.0, X, ldx,
                            1.0, D, ldd);

        if (compute_X_norms) {
            // Ensure diagonal entries are precisely zero

            if (order == column_major) {
                for (da_int j = 0; j < m; j++) {
                    if (!square) {
                        for (da_int i = 0; i < j; i++) {
                            D[i + j * ldd] = std::sqrt(D[i + j * ldd]);
                        }
                    }
                    D[j + j * ldd] = 0.0;
                }
            } else {
                for (da_int j = 0; j < m; j++) {
                    if (!square) {
                        for (da_int i = 0; i < j; i++) {
                            D[i * ldd + j] = std::sqrt(D[i * ldd + j]);
                        }
                    }
                    D[j + j * ldd] = 0.0;
                }
            }
        }
    }
}

namespace da_metrics {
namespace pairwise_distances {

template <typename T>
da_status euclidean(da_order order, da_int m, da_int n, da_int k, const T *X, da_int ldx,
                    const T *Y, da_int ldy, T *D, da_int ldd, bool square_distances) {
    da_status status = da_status_success;
    // Initialize X_is_Y.
    bool X_is_Y = false;
    // Allocate memory for compute_X_norms.
    std::vector<T> x_work, y_work;
    try {
        x_work.resize(m);
    } catch (std::bad_alloc const &) {
        return da_status_memory_error;
    }
    if (Y) {
        try {
            y_work.resize(n);
        } catch (std::bad_alloc const &) {
            return da_status_memory_error;
        }
    } else {
        // Y is null pointer, set X_is_Y to true
        X_is_Y = true;
    }
    euclidean_distance(order, m, n, k, X, ldx, Y, ldy, D, ldd, x_work.data(), 2,
                       y_work.data(), 2, square_distances, X_is_Y);
    // If X_is_Y only the upper triangular part of the symmetric matrix D is computed in euclidean_distance.
    // Update the lower part accordingly before returning.
    if (X_is_Y) {
        if (order == column_major) {
            for (da_int i = 0; i < m; i++)
                for (da_int j = 0; j < m; j++)
                    if (i > j)
                        D[i + j * ldd] = D[j + i * ldd];
        } else {
            for (da_int i = 0; i < m; i++)
                for (da_int j = 0; j < m; j++)
                    if (i > j)
                        D[j + i * ldd] = D[i + j * ldd];
        }
    }

    return status;
}
} // namespace pairwise_distances
} // namespace da_metrics