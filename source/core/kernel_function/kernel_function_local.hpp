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

#pragma once

#include "aoclda.h"
#include "aoclda_types.h"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "euclidean_distance.hpp"
#include <iostream>
#include <vector>

/*
Internal functions that avoid input checking
Also for RBF we can avoid repeatable creation of work arrays
*/

/*
RBF kernel
Given an m by k matrix X and an n by k matrix Y (both column major), computes the m by n kernel matrix D
*/
template <typename T>
inline void rbf_kernel_local(da_order order, da_int m, da_int n, da_int k, const T *X,
                             T *X_norms, da_int ldx, const T *Y, T *Y_norms, da_int ldy,
                             T *D, da_int ldd, T gamma, bool X_is_Y) {
    T multiplier = -gamma;
    // Compute |x_i-y_j|^2
    euclidean_distance(order, m, n, k, X, ldx, Y, ldy, D, ldd, X_norms, 2, Y_norms, 2,
                       true, X_is_Y);
    // If X==Y then result of euclidean_distance() is upper triangular matrix D.
    // This loop is to make D symmetric matrix.
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
                        D[i * ldd + j] = D[j * ldd + i];
        }
    }
    // Exponentiate all the entries in the matrix
    if (order == column_major) {
        for (da_int i = 0; i < n; i++) {
            for (da_int j = 0; j < m; j++) {
                D[i * ldd + j] = exp(multiplier * D[i * ldd + j]);
            }
        }
    } else {
        for (da_int i = 0; i < m; i++) {
            for (da_int j = 0; j < n; j++) {
                D[i * ldd + j] = exp(multiplier * D[i * ldd + j]);
            }
        }
    }
}
/*
Linear kernel
*/
template <typename T>
inline void linear_kernel_local(da_order order, da_int m, da_int n, da_int k, const T *X,
                                da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd,
                                bool X_is_Y) {
    CBLAS_ORDER cblas_order =
        (order == column_major) ? CBLAS_ORDER::CblasColMajor : CBLAS_ORDER::CblasRowMajor;
    // Compute X*Y^t
    if (X_is_Y) {
        da_blas::cblas_syrk(cblas_order, CblasUpper, CblasNoTrans, m, k, 1.0, X, ldx, 0.0,
                            D, ldd);
        // After syrk D matrix is upper triangular, this loop is to make D symmetric
        if (order == column_major) {
            for (da_int i = 0; i < m; i++)
                for (da_int j = 0; j < m; j++)
                    if (i > j)
                        D[i + j * ldd] = D[j + i * ldd];
        } else {
            for (da_int i = 0; i < m; i++)
                for (da_int j = 0; j < m; j++)
                    if (i > j)
                        D[i * ldd + j] = D[j * ldd + i];
        }
    } else {
        da_blas::cblas_gemm(cblas_order, CblasNoTrans, CblasTrans, m, n, k, 1.0, X, ldx,
                            Y, ldy, 0.0, D, ldd);
    }
}

/*
Polynomial kernel
*/
template <typename T>
inline void polynomial_kernel_local(da_order order, da_int m, da_int n, da_int k,
                                    const T *X, da_int ldx, const T *Y, da_int ldy, T *D,
                                    da_int ldd, T gamma, da_int degree, T coef0,
                                    bool X_is_Y) {
    CBLAS_ORDER cblas_order =
        (order == column_major) ? CBLAS_ORDER::CblasColMajor : CBLAS_ORDER::CblasRowMajor;
    // Compute gamma*X*Y^t
    if (X_is_Y) {
        da_blas::cblas_syrk(cblas_order, CblasUpper, CblasNoTrans, m, k, gamma, X, ldx,
                            0.0, D, ldd);
        // After syrk D matrix is upper triangular, this loop is to make D symmetric
        if (order == column_major) {
            for (da_int i = 0; i < m; i++)
                for (da_int j = 0; j < m; j++)
                    if (i > j)
                        D[i + j * ldd] = D[j + i * ldd];
        } else {
            for (da_int i = 0; i < m; i++)
                for (da_int j = 0; j < m; j++)
                    if (i > j)
                        D[i * ldd + j] = D[j * ldd + i];
        }
    } else {
        da_blas::cblas_gemm(cblas_order, CblasNoTrans, CblasTrans, m, n, k, gamma, X, ldx,
                            Y, ldy, 0.0, D, ldd);
    }
    // Raise to the power and add constant
    if (order == column_major) {
        for (da_int i = 0; i < n; i++) {
            for (da_int j = 0; j < m; j++) {
                D[i * ldd + j] = pow(D[i * ldd + j] + coef0, degree);
            }
        }
    } else {
        for (da_int i = 0; i < m; i++) {
            for (da_int j = 0; j < n; j++) {
                D[i * ldd + j] = pow(D[i * ldd + j] + coef0, degree);
            }
        }
    }
}

/*
Sigmoid kernel
*/
template <typename T>
inline void sigmoid_kernel_local(da_order order, da_int m, da_int n, da_int k, const T *X,
                                 da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd,
                                 T gamma, T coef0, bool X_is_Y) {
    CBLAS_ORDER cblas_order =
        (order == column_major) ? CBLAS_ORDER::CblasColMajor : CBLAS_ORDER::CblasRowMajor;
    // Compute gamma*X*Y^t
    if (X_is_Y) {
        da_blas::cblas_syrk(cblas_order, CblasUpper, CblasNoTrans, m, k, gamma, X, ldx,
                            0.0, D, ldd);
        // After syrk D matrix is upper triangular, this loop is to make D symmetric
        if (order == column_major) {
            for (da_int i = 0; i < m; i++)
                for (da_int j = 0; j < m; j++)
                    if (i > j)
                        D[i + j * ldd] = D[j + i * ldd];
        } else {
            for (da_int i = 0; i < m; i++)
                for (da_int j = 0; j < m; j++)
                    if (i > j)
                        D[i * ldd + j] = D[j * ldd + i];
        }
    } else {
        da_blas::cblas_gemm(cblas_order, CblasNoTrans, CblasTrans, m, n, k, gamma, X, ldx,
                            Y, ldy, 0.0, D, ldd);
    }
    // Raise to the compute tanh and add constant
    if (order == column_major) {
        for (da_int i = 0; i < n; i++) {
            for (da_int j = 0; j < m; j++) {
                D[i * ldd + j] = tanh(D[i * ldd + j] + coef0);
            }
        }
    } else {
        for (da_int i = 0; i < m; i++) {
            for (da_int j = 0; j < n; j++) {
                D[i * ldd + j] = tanh(D[i * ldd + j] + coef0);
            }
        }
    }
}