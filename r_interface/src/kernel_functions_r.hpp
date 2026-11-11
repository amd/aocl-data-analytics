/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef KERNEL_FUNCTIONS_R_HPP
#define KERNEL_FUNCTIONS_R_HPP

#include <Rcpp.h>

#include "aoclda.h"
#include "internal_utilities_r.hpp"

using namespace Rcpp;

inline NumericMatrix da_rbf_kernel_wrapper(NumericMatrix X, Nullable<NumericMatrix> Y,
                                           double gamma = 1.0) {
    da_status status;
    da_int m, n, k_x, k_y, ldx, ldy, ldd;

    m = X.nrow();
    k_x = X.ncol();
    ldx = m;
    n = 0;
    ldy = 0;
    da_int ncols = m;
    const double *Y_ptr = nullptr;

    if (Y.isNotNull()) {
        const NumericMatrix &Y_mat = Y.get();
        n = Y_mat.nrow();
        k_y = Y_mat.ncol();
        ldy = n;
        ncols = n;
        if (k_x != k_y)
            stop("X and Y must have the same number of features");
        Y_ptr = Y_mat.begin();
    }

    NumericMatrix D(m, ncols);
    ldd = m;

    status = da_rbf_kernel_d(column_major, m, n, k_x, X.begin(), ldx, Y_ptr, ldy,
                             D.begin(), ldd, gamma);
    status_to_exception(status);
    return D;
}

inline NumericMatrix da_linear_kernel_wrapper(NumericMatrix X,
                                              Nullable<NumericMatrix> Y) {
    da_status status;
    da_int m, n, k_x, k_y, ldx, ldy, ldd;

    m = X.nrow();
    k_x = X.ncol();
    ldx = m;
    n = 0;
    ldy = 0;
    da_int ncols = m;
    const double *Y_ptr = nullptr;

    if (Y.isNotNull()) {
        const NumericMatrix &Y_mat = Y.get();
        n = Y_mat.nrow();
        k_y = Y_mat.ncol();
        ldy = n;
        ncols = n;
        if (k_x != k_y)
            stop("X and Y must have the same number of features");
        Y_ptr = Y_mat.begin();
    }

    NumericMatrix D(m, ncols);
    ldd = m;

    status = da_linear_kernel_d(column_major, m, n, k_x, X.begin(), ldx, Y_ptr, ldy,
                                D.begin(), ldd);
    status_to_exception(status);
    return D;
}

inline NumericMatrix da_polynomial_kernel_wrapper(NumericMatrix X,
                                                  Nullable<NumericMatrix> Y = R_NilValue,
                                                  da_int degree = 3, double gamma = 1.0,
                                                  double coef0 = 1.0) {
    da_status status;
    da_int m, n, k_x, k_y, ldx, ldy, ldd;

    m = X.nrow();
    k_x = X.ncol();
    ldx = m;
    n = 0;
    ldy = 0;
    da_int ncols = m;
    const double *Y_ptr = nullptr;

    if (Y.isNotNull()) {
        const NumericMatrix &Y_mat = Y.get();
        n = Y_mat.nrow();
        k_y = Y_mat.ncol();
        ldy = n;
        ncols = n;
        if (k_x != k_y)
            stop("X and Y must have the same number of features");
        Y_ptr = Y_mat.begin();
    }

    NumericMatrix D(m, ncols);
    ldd = m;

    status = da_polynomial_kernel_d(column_major, m, n, k_x, X.begin(), ldx, Y_ptr, ldy,
                                    D.begin(), ldd, gamma, degree, coef0);
    status_to_exception(status);
    return D;
}

inline NumericMatrix da_sigmoid_kernel_wrapper(NumericMatrix X,
                                               Nullable<NumericMatrix> Y = R_NilValue,
                                               double gamma = 1.0, double coef0 = 1.0) {
    da_status status;
    da_int m, n, k_x, k_y, ldx, ldy, ldd;

    m = X.nrow();
    k_x = X.ncol();
    ldx = m;
    n = 0;
    ldy = 0;
    da_int ncols = m;
    const double *Y_ptr = nullptr;

    if (Y.isNotNull()) {
        const NumericMatrix &Y_mat = Y.get();
        n = Y_mat.nrow();
        k_y = Y_mat.ncol();
        ldy = n;
        ncols = n;
        if (k_x != k_y)
            stop("X and Y must have the same number of features");
        Y_ptr = Y_mat.begin();
    }

    NumericMatrix D(m, ncols);
    ldd = m;

    status = da_sigmoid_kernel_d(column_major, m, n, k_x, X.begin(), ldx, Y_ptr, ldy,
                                 D.begin(), ldd, gamma, coef0);
    status_to_exception(status);
    return D;
}

#endif