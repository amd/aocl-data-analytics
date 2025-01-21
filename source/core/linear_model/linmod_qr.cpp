/* ************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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

#include "aoclda.h"
#include "convert_num.hpp"
#include "da_cblas.hh"
#include "linear_model.hpp"
#include "macros.h"
#include "sparse_overloads.hpp"
#include <vector>

namespace ARCH {

namespace da_linmod {

using namespace da_linmod_types;

// Data for QR factorization used in standard linear least squares
template <typename T> qr_data<T>::qr_data(da_int nsamples, da_int nfeat) {
    // Work arrays for the LAPACK QR factorization
    /* Naming convention of n_col and n_row comes from the fact that in QR we are always
            dealing with tall matrix (if we don't, we transpose it so that we do). So it's more
            natural to call it this way rather than min_order/max_order */
    n_col = std::min(nsamples, nfeat);
    n_row = std::max(nsamples, nfeat);
    tau.resize(n_col);
    lwork = n_col;
    work.resize(lwork);
};

template struct qr_data<float>;
template struct qr_data<double>;

} // namespace da_linmod

} // namespace ARCH