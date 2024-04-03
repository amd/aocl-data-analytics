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

#ifndef EUCLIDEAN_DISTANCE_HPP
#define EUCLIDEAN_DISTANCE_HPP
#include "aoclda_types.h"
#include "da_cblas.hh"

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
inline void euclidean_distance(da_int m, da_int n, da_int k, const T *X, da_int ldx,
                               const T *Y, da_int ldy, T *D, da_int ldd, T *X_norms,
                               da_int compute_X_norms, T *Y_norms, da_int compute_Y_norms,
                               bool square, bool X_is_Y) {

    // If needed, compute the squared norms of the rows of X and Y
    if (compute_X_norms == 2) {
        for (da_int i = 0; i < m; i++) {
            X_norms[i] = 0.0;
        }
        for (da_int j = 0; j < k; j++) {
            for (da_int i = 0; i < m; i++) {
                X_norms[i] += X[i + j * ldx] * X[i + j * ldx];
            }
        }
    }

    if (compute_Y_norms == 2 && !(X_is_Y)) {
        for (da_int i = 0; i < n; i++) {
            Y_norms[i] = 0.0;
        }
        for (da_int j = 0; j < k; j++) {
            for (da_int i = 0; i < n; i++) {
                Y_norms[i] += Y[i + j * ldy] * Y[i + j * ldy];
            }
        }
    }

    if (!X_is_Y) {
        // A few different cases to check depending on the boolean inputs

        if (compute_X_norms == 0 && compute_Y_norms == 0) {
            for (da_int j = 0; j < n; j++) {
                for (da_int i = 0; i < m; i++) {
                    D[i + j * ldd] = 0.0;
                }
            }
        } else if (compute_X_norms > 0 && compute_Y_norms == 0) {
            for (da_int j = 0; j < n; j++) {
                for (da_int i = 0; i < m; i++) {
                    D[i + j * ldd] = X_norms[i];
                }
            }
        } else if (compute_X_norms == 0 && compute_Y_norms > 0) {
            for (da_int j = 0; j < n; j++) {
                for (da_int i = 0; i < m; i++) {
                    D[i + j * ldd] = Y_norms[j];
                }
            }
        } else {
            for (da_int j = 0; j < n; j++) {
                for (da_int i = 0; i < m; i++) {
                    D[i + j * ldd] = X_norms[i] + Y_norms[j];
                }
            }
        }

        da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, k, -2.0, X,
                            ldx, Y, ldy, 1.0, D, ldd);

        if (!square) {
            for (da_int j = 0; j < n; j++) {
                for (da_int i = 0; i < m; i++) {
                    D[i + j * ldd] = std::sqrt(D[i + j * ldd]);
                }
            }
        }
    } else {
        // Special case when computing upper triangle of symmetric distance matrix

        if (compute_X_norms == 0) {
            for (da_int j = 0; j < m; j++) {
                for (da_int i = 0; i <= j; i++) {
                    D[i + j * ldd] = 0.0;
                }
            }
        } else {
            for (da_int j = 0; j < m; j++) {
                for (da_int i = 0; i <= j; i++) {
                    D[i + j * ldd] = X_norms[i] + X_norms[j];
                }
            }
        }

        da_blas::cblas_syrk(CblasColMajor, CblasUpper, CblasNoTrans, m, k, -2.0, X, ldx,
                            1.0, D, ldd);

        if (compute_X_norms) {
            // Ensure diagonal entries are precisely zero

            for (da_int j = 0; j < m; j++) {
                if (!square) {
                    for (da_int i = 0; i < j; i++) {
                        D[i + j * ldd] = std::sqrt(D[i + j * ldd]);
                    }
                }
                D[j + j * ldd] = 0.0;
            }
        }
    }
}

#endif
