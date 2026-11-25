/* ************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#include "aoclda.h"
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
    lwork = 1;
    work.resize(lwork);

    da_int qlwork{-1};
    da_int info{1};
    T X[1];
    da::geqrf(&n_row, &n_col, X, &n_row, tau.data(), work.data(), &qlwork, &info);
    qlwork = work[0]; // get optimal size of work array
    if (info != 0 || qlwork < 0) {
        throw std::runtime_error(
            "encountered an unexpected error while quering work array size in the QR "
            "factorization (geqrf INFO=" +
            std::to_string(info) + ")");
    }
    work.resize(qlwork);
    lwork = qlwork;
};

template struct qr_data<float>;
template struct qr_data<double>;

} // namespace da_linmod

} // namespace ARCH