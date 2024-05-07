/* ************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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
    std::vector<T> tau, work;
    da_int lwork = 0, n_col, n_row;

    // Constructors
    qr_data(da_int nsamples, da_int nfeat) {
        // work arrays for the LAPACK QR factorization
        /* Naming convention of n_col and n_row comes from the fact that in QR we are always
            dealing with tall matrix (if we don't, we transpose it so that we do). So it's more
            natural to call it this way rather than min_order/max_order */
        n_col = std::min(nsamples, nfeat);
        n_row = std::max(nsamples, nfeat);
        tau.resize(n_col);
        lwork = n_col;
        work.resize(lwork);
    };
};
} // namespace da_linmod

#endif