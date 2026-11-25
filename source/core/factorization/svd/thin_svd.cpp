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

#include "svd/thin_svd.hpp"
#include "aoclda.h"
#include "da_error.hpp"
#include "lapack_templates.hpp"
#include "macros.h"
#include <algorithm>
#include <new>
#include <vector>

namespace ARCH {

namespace da_svd {

template <typename T>
da_status thin_svd(T *A, da_int m, da_int n, da_int lda, T *U, da_int ldu, T *S, T *Vt,
                   da_int ldvt, da_errors::da_error_t &err) {
    if (A == nullptr || U == nullptr || S == nullptr || Vt == nullptr)
        return da_error(&err, da_status_invalid_pointer,
                        "Null pointer passed to thin_svd.");
    if (m < 1 || n < 1)
        return da_error(&err, da_status_invalid_input,
                        "Matrix dimensions m and n must both be >= 1.");

    const da_int k = std::min(m, n);

    if (lda < m)
        return da_error(&err, da_status_invalid_leading_dimension,
                        "Leading dimension lda must be >= m.");
    if (ldu < m)
        return da_error(&err, da_status_invalid_leading_dimension,
                        "Leading dimension ldu must be >= m.");
    if (ldvt < k)
        return da_error(&err, da_status_invalid_leading_dimension,
                        "Leading dimension ldvt must be >= min(m, n).");

    char JOBZ = 'S';
    da_int INFO = 0;
    da_int lwork = -1;
    T estworkspace[1];

    std::vector<da_int> iwork;
    try {
        iwork.resize(static_cast<std::size_t>(8) * static_cast<std::size_t>(k));
    } catch (std::bad_alloc const &) {
        return da_error(&err, da_status_memory_error,
                        "Memory allocation failed for thin_svd iwork.");
    }

    /* Workspace query. */
    da::gesdd(&JOBZ, &m, &n, A, &lda, S, U, &ldu, Vt, &ldvt, estworkspace, &lwork,
              iwork.data(), &INFO);
    if (INFO != 0)
        return da_error(&err, da_status_internal_error,
                        "LAPACK gesdd workspace query failed.");

    lwork = static_cast<da_int>(estworkspace[0]);
    std::vector<T> work;
    try {
        work.resize(static_cast<std::size_t>(lwork));
    } catch (std::bad_alloc const &) {
        return da_error(&err, da_status_memory_error,
                        "Memory allocation failed for thin_svd work buffer.");
    }

    INFO = 0;
    da::gesdd(&JOBZ, &m, &n, A, &lda, S, U, &ldu, Vt, &ldvt, work.data(), &lwork,
              iwork.data(), &INFO);

    if (INFO < 0)
        return da_error(&err, da_status_internal_error,
                        "LAPACK gesdd reported an invalid argument.");
    if (INFO > 0)
        return da_error(&err, da_status_numerical_difficulties,
                        "LAPACK gesdd failed to converge while computing the SVD.");

    return da_status_success;
}

template da_status thin_svd<float>(float *, da_int, da_int, da_int, float *, da_int,
                                   float *, float *, da_int, da_errors::da_error_t &);
template da_status thin_svd<double>(double *, da_int, da_int, da_int, double *, da_int,
                                    double *, double *, da_int, da_errors::da_error_t &);

} // namespace da_svd
} // namespace ARCH
