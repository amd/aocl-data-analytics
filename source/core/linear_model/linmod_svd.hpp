/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#ifndef LINMOD_SVD_REG_HPP
#define LINMOD_SVD_REG_HPP

#include "aoclda.h"
#include <vector>

namespace da_linmod {
// data for svd used in linear regression
template <typename T> struct svd_data {
    std::vector<T> S, U, Vt, temp, work;
    std::vector<da_int> iwork;
    da_int lwork = 0, min_order;
    T alpha = 1.0, beta = 0.0;

    // Constructors
    svd_data(da_int nsamples, da_int nfeat) {

        // work arrays for the SVD
        min_order = std::min(nsamples, nfeat);
        S.resize(min_order);
        U.resize(nsamples * min_order);
        Vt.resize(min_order * nfeat);
        temp.resize(min_order);
        lwork = 4 * min_order * min_order + 7 * min_order;
        iwork.resize(8 * min_order);
        work.resize(lwork);
    };
};
} // namespace da_linmod

#endif