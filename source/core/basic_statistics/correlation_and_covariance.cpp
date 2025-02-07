/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclda.h"
#include "basic_statistics.hpp"
#include "da_cblas.hh"
#include "macros.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace ARCH {

namespace da_basic_statistics {

/* Correlation or covariance matrix of x */
template <typename T>
da_status cov_corr_matrix(da_order order, da_int n, da_int p, const T *x, da_int ldx,
                          da_int dof, T *mat, da_int ldmat, bool compute_corr) {

    da_status status = da_status_success;

    if (order == column_major && ldx < n)
        return da_status_invalid_leading_dimension;
    if (order == row_major && ldx < p)
        return da_status_invalid_leading_dimension;
    if (ldmat < p)
        return da_status_invalid_leading_dimension;
    if (n <= 1 || p < 1)
        return da_status_invalid_array_dimension;
    if (x == nullptr || mat == nullptr)
        return da_status_invalid_pointer;

    // We need a copy of x so we don't alter the input data
    T *x_copy = nullptr;
    da_int ldx_copy;
    CBLAS_ORDER cblas_order;
    try {
        x_copy = new T[n * p];
    } catch (std::bad_alloc const &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    }

    if (order == column_major) {
        ldx_copy = n;
        cblas_order = CBLAS_ORDER::CblasColMajor;
        for (da_int j = 0; j < p; j++) {
            for (da_int i = 0; i < n; i++) {
                x_copy[j * ldx_copy + i] = x[j * ldx + i];
            }
        }
    } else {
        ldx_copy = p;
        cblas_order = CBLAS_ORDER::CblasRowMajor;
        for (da_int i = 0; i < n; i++) {
            for (da_int j = 0; j < p; j++) {
                x_copy[i * ldx_copy + j] = x[i * ldx + j];
            }
        }
    }

    T *dummy = nullptr;

    da_int scale_factor = dof;
    if (dof < 0) {
        scale_factor = n;
    } else if (dof == 0) {
        scale_factor = n - 1;
    }

    // For correlation matrix we standardize completely. For covariance we just mean centre the columns.
    if (compute_corr) {
        // Use dof = 1 to avoid dividing standard deviations by n, only to have to multiply by n later
        status =
            standardize(order, da_axis_col, n, p, x_copy, ldx_copy, 1, 0, dummy, dummy);
    } else {
        std::vector<T> col_means(p);
        // Call standardize with col_means full of zeros so the means are computed
        status = standardize(order, da_axis_col, n, p, x_copy, ldx_copy, dof, 0,
                             col_means.data(), dummy);
    }
    if (status != da_status_success) {
        status = da_status_internal_error; // LCOV_EXCL_LINE
        goto exit;
    }

    // Form X^T X in mat
    da_blas::cblas_gemm(cblas_order, CblasTrans, CblasNoTrans, p, p, n, 1.0, x_copy,
                        ldx_copy, x_copy, ldx_copy, 0.0, mat, ldmat);

    // In this section, the same code works for both row- and column-major ordering
    if (!compute_corr && scale_factor > 1) {
        // Scale the covariance matrix
        for (da_int j = 0; j < p; j++) {
            for (da_int i = 0; i < p; i++) {
                mat[j * ldmat + i] /= scale_factor;
            }
        }
    } else if (compute_corr) {
        // Correlation matrix should have diagonals equal to 1 precisely
        for (da_int i = 0; i < p; i++) {
            mat[i * ldmat + i] = 1.0;
        }
    }

exit:
    delete[] x_copy;
    return status;
}

template <typename T>
da_status covariance_matrix(da_order order, da_int n, da_int p, const T *x, da_int ldx,
                            da_int dof, T *cov, da_int ldcov) {
    return cov_corr_matrix(order, n, p, x, ldx, dof, cov, ldcov, false);
}

template <typename T>
da_status correlation_matrix(da_order order, da_int n, da_int p, const T *x, da_int ldx,
                             T *corr, da_int ldcorr) {
    return cov_corr_matrix(order, n, p, x, ldx, 0, corr, ldcorr, true);
}

template da_status covariance_matrix<float>(da_order order, da_int n, da_int p,
                                            const float *x, da_int ldx, da_int dof,
                                            float *cov, da_int ldcov);
template da_status covariance_matrix<double>(da_order order, da_int n, da_int p,
                                             const double *x, da_int ldx, da_int dof,
                                             double *cov, da_int ldcov);
template da_status correlation_matrix<float>(da_order order, da_int n, da_int p,
                                             const float *x, da_int ldx, float *corr,
                                             da_int ldcorr);
template da_status correlation_matrix<double>(da_order order, da_int n, da_int p,
                                              const double *x, da_int ldx, double *corr,
                                              da_int ldcorr);
template da_status cov_corr_matrix<float>(da_order order, da_int n, da_int p,
                                          const float *x, da_int ldx, da_int dof,
                                          float *mat, da_int ldmat, bool compute_corr);
template da_status cov_corr_matrix<double>(da_order order, da_int n, da_int p,
                                           const double *x, da_int ldx, da_int dof,
                                           double *mat, da_int ldmat, bool compute_corr);

} // namespace da_basic_statistics

} // namespace ARCH