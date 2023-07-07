#ifndef CORRELATION_AND_COVARIANCE_HPP
#define CORRELATION_AND_COVARIANCE_HPP

#include "aoclda.h"
#include "da_cblas.hh"
#include "moment_statistics.hpp"
#include "statistical_utilities.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace da_basic_statistics {

/* Correlation or covariance matrix of x */
template <typename T>
da_status cov_corr_matrix(da_int n, da_int p, const T *x, da_int ldx, T *mat,
                          da_int ldmat, bool compute_corr) {

    da_status status = da_status_success;

    if (ldx < n || ldmat < p)
        return da_status_invalid_leading_dimension;
    if (n <= 1 || p < 1)
        return da_status_invalid_array_dimension;
    if (x == nullptr || mat == nullptr)
        return da_status_invalid_pointer;

    // We need a copy of x so we don't alter the input data
    T *x_copy = nullptr, *col_means = nullptr;
    try {
        x_copy = new T[n * p];
    } catch (std::bad_alloc const &) {
        return da_status_memory_error;
    }

    for (da_int j = 0; j < p; j++) {
        for (da_int i = 0; i < n; i++) {
            x_copy[j * n + i] = x[j * ldx + i];
        }
    }

    T *dummy = nullptr;

    // For correlation matrix we standardize completely. For covariance we just mean centre the columns.
    if (compute_corr) {
        status = standardize(da_axis_col, n, p, x_copy, n, dummy, dummy);
    } else {
        try {
            col_means = new T[p];
        } catch (std::bad_alloc const &) {
            status = da_status_memory_error; // LCOV_EXCL_LINE
            goto exit;                       // LCOV_EXCL_LINE
        }
        status = mean(da_axis_col, n, p, x_copy, n, col_means);
        status = standardize(da_axis_col, n, p, x_copy, n, col_means, dummy);
    }
    if (status != da_status_success) {
        status = da_status_internal_error; // LCOV_EXCL_LINE
        goto exit;
    }

    // Form X^T X in mat
    da_blas::cblas_gemm(CblasColMajor, CblasTrans, CblasNoTrans, p, p, n, 1.0, x_copy, n,
                        x_copy, n, 0.0, mat, ldmat);

    // Scale by n-1
    for (da_int j = 0; j < p; j++) {
        for (da_int i = 0; i < p; i++) {
            mat[j * ldmat + i] /= n - 1;
        }
    }

    // Correlation matrix should have diagonals equal to 1 precisely
    if (compute_corr) {
        for (da_int i = 0; i < p; i++) {
            mat[i * ldmat + i] = 1.0;
        }
    }

exit:
    delete[] x_copy;
    delete[] col_means;
    return status;
}

template <typename T>
da_status covariance_matrix(da_int n, da_int p, const T *x, da_int ldx, T *cov,
                            da_int ldcov) {
    return cov_corr_matrix(n, p, x, ldx, cov, ldcov, false);
}

template <typename T>
da_status correlation_matrix(da_int n, da_int p, const T *x, da_int ldx, T *corr,
                             da_int ldcorr) {
    return cov_corr_matrix(n, p, x, ldx, corr, ldcorr, true);
}

} // namespace da_basic_statistics

#endif