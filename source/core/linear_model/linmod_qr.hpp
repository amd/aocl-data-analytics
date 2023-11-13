/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef LINMOD_QR_REG_HPP
#define LINMOD_QR_REG_HPP

#include "aoclda.h"
#include <vector>

// data for QR factorization used in standard linear least squares
template <typename T> struct qr_data {
    // A needs to be copied as lapack's dgeqr modifies the matrix
    std::vector<T> A, b, tau, work;
    da_int lwork = 0;

    // Constructors
    qr_data(da_int m, da_int n, T *Ai, T *bi, bool intercept, da_int ncoef) {

        // Copy A and b, starting with the first n columns of A
        A.resize(m * ncoef);
        for (da_int j = 0; j < n; j++) {
            for (da_int i = 0; i < m; i++) {
                A[j * m + i] = Ai[j * m + i];
            }
        }
        b.resize(m);
        for (da_int i = 0; i < m; i++)
            b[i] = bi[i];

        // add a column of 1 to A if intercept is required
        if (intercept) {
            for (da_int i = 0; i < m; i++)
                A[n * m + i] = 1.0;
        }

        // work arrays for the LAPACK QR factorization
        tau.resize(std::min(m, ncoef));
        lwork = ncoef;
        work.resize(lwork);
    };
};

#endif