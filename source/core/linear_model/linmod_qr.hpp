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

namespace da_linmod {
// data for QR factorization used in standard linear least squares
template <typename T> struct qr_data {
    // X needs to be copied as lapack's dgeqr modifies the matrix
    std::vector<T> X, y, tau, work;
    da_int lwork = 0;

    // Constructors
    qr_data(da_int nsamples, da_int nfeat, const T *Xi, const T *yi, bool intercept,
            da_int ncoef) {

        // Copy X and y, starting with the first nfeat columns of X
        X.resize(nsamples * ncoef);
        for (da_int j = 0; j < nfeat; j++) {
            for (da_int i = 0; i < nsamples; i++) {
                X[j * nsamples + i] = Xi[j * nsamples + i];
            }
        }
        y.resize(nsamples);
        for (da_int i = 0; i < nsamples; i++)
            y[i] = yi[i];

        // add a column of 1 to X if intercept is required
        if (intercept) {
            for (da_int i = 0; i < nsamples; i++)
                X[nfeat * nsamples + i] = 1.0;
        }

        // work arrays for the LAPACK QR factorization
        tau.resize(std::min(nsamples, ncoef));
        lwork = ncoef;
        work.resize(lwork);
    };
};
} // namespace da_linmod

#endif