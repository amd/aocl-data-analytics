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

// Data for svd used in linear regression
template <typename T> svd_data<T>::svd_data(da_int nsamples, da_int nfeat) {

    // Work arrays for the SVD
    min_order = std::min(nsamples, nfeat);
    S.resize(min_order);
    U.resize(nsamples * min_order);
    Vt.resize(min_order * nfeat);
    temp.resize(min_order);
    lwork = 4 * min_order * min_order + 7 * min_order;
    iwork.resize(8 * min_order);
    work.resize(lwork);
};

template struct svd_data<float>;
template struct svd_data<double>;

} // namespace da_linmod

} // namespace ARCH