/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
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

#ifndef PCA_HPP
#define PCA_HPP

#include "../basic_statistics/moment_statistics.hpp"
#include "../basic_statistics/statistical_utilities.hpp"
#include "aoclda.h"
#include "aoclda_pca.h"
#include "basic_handle.hpp"
#include "callbacks.hpp"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "da_qr.hpp"
#include "lapack_templates.hpp"
#include "options.hpp"
#include "pca_options.hpp"
#include "pca_types.hpp"
#include <algorithm>
#include <iostream>
#include <string.h>
#include <vector>

namespace da_pca {

/* PCA class */
template <typename T> class da_pca : public basic_handle<T> {
  private:
    // n x p (samples x features)
    da_int n = 0;
    da_int p = 0;

    // User's data
    const T *A;
    da_int lda;

    // Set true when initialization is complete
    bool initdone = false;

    // Set true when PCA is computed successfully
    bool iscomputed = false;

    // Correlation or covariance based PCA
    da_int method = pca_method_cov;

    // SVD solver
    da_int solver = solver_gesdd;

    // Sign flip flag for consistency with sklearn results
    bool svd_flip_u_based = false;

    // Whether we are storing U
    bool store_U = false;

    // Number of principal components requested
    da_int npc = 1;

    // Degrees of freedom (bias) when computing variances, and associated divisor
    da_int dof = 0;
    da_int div = 0;

    // Actual number of principal components found - on output should be the same as npc unless dgesvdx gives unexpected behaviour
    da_int ns = 0;

    // Will we perform a QR decomposition prior to the SVD?
    bool qr = false;

    // Pointer to error trace
    da_errors::da_error_t *err = nullptr;

    // Arrays used by the SVD, and to store results
    std::vector<T> scores;   // U*Sigma
    std::vector<T> variance; // Sigma**2 / n-1
    std::vector<T> column_means, column_sdevs,
        column_sdevs_nonzero; // Store standardization data
    T total_variance = 0.0;   // Sum((MeanCentered A [][])**2)
    da_int n_components = 0, ldvt = 0, u_size = 0, ldu = 0;
    std::vector<T> u, sigma, vt, work, A_copy;
    std::vector<da_int> iwork;

  public:
    da_options::OptionRegistry opts;

    da_pca(da_errors::da_error_t &err) {
        this->err = &err;
        register_pca_options<T>(opts);
    };

    da_status init(da_int n, da_int p, const T *A, da_int lda);

    da_status compute();

    da_status transform(da_int m, da_int p, const T *X, da_int ldx, T *X_transform,
                        da_int ldx_transform);

    da_status inverse_transform(da_int k, da_int r, const T *X, da_int ldx,
                                T *X_inv_transform, da_int ldx_inv_transform);

    da_status get_result(da_result query, da_int *dim, T *result) {
        // Don't return anything if PCA has not been computed
        if (!iscomputed) {
            return da_warn(err, da_status_no_data,
                           "PCA has not yet been computed. Please call da_pca_compute_s "
                           "or da_pca_compute_d before extracting results.");
        }

        da_int rinfo_size = 3;

        if (result == nullptr) {
            return da_warn(err, da_status_invalid_array_dimension,
                           "The results array has not been allocated.");
        }

        switch (query) {
        case da_result::da_rinfo:
            if (*dim < rinfo_size) {
                *dim = rinfo_size;
                return da_warn(err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(rinfo_size) + ".");
            }
            result[0] = (T)n;
            result[1] = (T)p;
            result[2] = (T)ns;
            break;
        case da_result::da_pca_scores:
            if (store_U == false) {
                return da_error(
                    err, da_status_invalid_option,
                    "In order to return the scores, the option 'store U' must be set.");
            }
            if (*dim < n * ns) {
                *dim = n * ns;
                return da_warn(err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(n * ns) + ".");
            }
            // Compute Scores matrix, U * Sigma
            for (da_int j = 0; j < ns; j++) {
                for (da_int i = 0; i < n; i++) {
                    result[j * n + i] = sigma[j] * u[j * ldu + i];
                }
            }
            break;
        case da_result::da_pca_u:
            if (store_U == false) {
                return da_error(
                    err, da_status_invalid_option,
                    "In order to return U, the option 'store U' must be set.");
            }
            if (*dim < n * ns) {
                *dim = n * ns;
                return da_warn(err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(n * ns) + ".");
            }
            for (da_int j = 0; j < ns; j++) {
                for (da_int i = 0; i < n; i++) {
                    result[i + n * j] = u[i + ldu * j];
                }
            }
            break;
        case da_result::da_pca_principal_components:
            if (*dim < ns * p) {
                *dim = ns * p;
                return da_warn(err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(ns * p) + ".");
            }
            for (da_int j = 0; j < p; j++)
                for (da_int i = 0; i < ns; i++)
                    result[i + j * ns] = vt[i + j * ldvt];
            break;
        case da_result::da_pca_vt:
            if (*dim < npc * p) {
                *dim = npc * p;
                return da_warn(err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(npc * p) + ".");
            }
            for (da_int j = 0; j < p; j++) {
                for (da_int i = 0; i < npc; i++) {
                    result[i + npc * j] = vt[i + ldvt * j];
                }
            }
            break;
        case da_result::da_pca_variance:
            if (*dim < ns) {
                *dim = ns;
                return da_warn(this->err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(ns) + ".");
            }
            // Compute variance, proportional to (Sigma**2)
            for (da_int j = 0; j < ns; j++) {
                result[j] = sigma[j] * sigma[j] / div;
            }
            break;
        case da_result::da_pca_sigma:
            if (*dim < ns) {
                *dim = ns;
                return da_warn(this->err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(ns) + ".");
            }
            for (da_int i = 0; i < ns; i++)
                result[i] = sigma[i];
            break;
        case da_result::da_pca_column_means:
            if (method == pca_method_svd)
                return da_warn(err, da_status_unknown_query,
                               "Column means are only computed if the 'PCA method' "
                               "option is set to 'covariance' or 'correlation'.");
            if (*dim < p) {
                *dim = p;
                return da_warn(err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(p) + ".");
            }
            for (da_int i = 0; i < p; i++)
                result[i] = column_means[i];
            break;
        case da_result::da_pca_column_sdevs:
            if (method != pca_method_corr)
                return da_warn(err, da_status_unknown_query,
                               "Standard deviations are only computed if the 'PCA "
                               "method' option is set to 'correlation'.");
            if (*dim < p) {
                *dim = p;
                return da_warn(this->err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(p) + ".");
            }
            for (da_int i = 0; i < p; i++)
                result[i] = column_sdevs[i];
            break;
        case da_result::da_pca_total_variance:
            result[0] = total_variance;
            break;
        default:
            return da_warn(err, da_status_unknown_query,
                           "The requested result could not be found.");
        }
        return da_status_success;
    };

    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim,
                         [[maybe_unused]] da_int *result) {

        return da_warn(err, da_status_unknown_query,
                       "There are no integer results available for this API.");
    };
};

/* Store the user's data matrix in preparation for PCA computation */
template <typename T>
da_status da_pca<T>::init(da_int n, da_int p, const T *A, da_int lda) {

    // Check for illegal arguments and function calls
    if (n < 1)
        return da_error(err, da_status_invalid_input,
                        "The function was called with n_samples = " + std::to_string(n) +
                            ". Constraint: n_samples >= 1.");
    if (p < 1)
        return da_error(err, da_status_invalid_input,
                        "The function was called with n_features = " + std::to_string(p) +
                            ". Constraint: n_features >= 1.");
    if (lda < n)
        return da_error(err, da_status_invalid_input,
                        "The function was called with n_samples = " + std::to_string(n) +
                            " and lda = " + std::to_string(lda) +
                            ". Constraint: lda >= n_samples.");
    if (A == nullptr)
        return da_error(err, da_status_invalid_pointer, "The array A is null.");

    // Store dimensions of A
    this->n = n;
    this->p = p;
    this->A = A;
    this->lda = lda;

    qr = false;
    store_U = false;

    u.resize(0);
    sigma.resize(0);
    vt.resize(0);
    column_means.resize(0);
    column_sdevs.resize(0);
    column_sdevs_nonzero.resize(0);

    // Record that initialization is complete but computation has not yet been performed
    initdone = true;
    iscomputed = false;

    // Now that we have a data matrix we can re-register the n_components option with new constraints
    da_int npc, max_npc = std::min(n, p);
    opts.get("n_components", npc);

    reregister_pca_option<T>(opts, max_npc);

    opts.set("n_components", std::min(npc, max_npc));

    if (npc > max_npc)
        return da_warn(
            err, da_status_incompatible_options,
            "The requested number of principal components has been decreased from " +
                std::to_string(npc) + " to " + std::to_string(max_npc) +
                " due to the size (" + std::to_string(n) + " x " + std::to_string(p) +
                ") of the data array.");

    return da_status_success;
}

/* Compute the PCA */
template <typename T> da_status da_pca<T>::compute() {
    if (initdone == false)
        return da_error(err, da_status_no_data,
                        "No data has been passed to the handle. Please call "
                        "da_pca_set_data_s or da_pca_set_data_d.");

    // Read in options and store in class together with associated variables
    this->opts.get("n_components", npc);

    // If n_components was set to 0 it means find all the components
    if (npc == 0) {
        npc = std::min(n, p);
    }

    ns = npc;
    std::string opt_method;

    this->opts.get("PCA method", opt_method, method);

    da_int u_tmp;
    this->opts.get("store U", u_tmp);
    store_U = (u_tmp > 0) ? true : false;

    std::string svd_routine;
    this->opts.get("svd solver", svd_routine, solver);
    if (solver == solver_auto) {
        solver = (n > 3 * p && !(store_U))
                     ? solver_syevd
                     : solver_gesdd; // TODO switch where appropriate
    }
    if (solver == solver_syevd && store_U) {
        return da_error(err, da_status_incompatible_options,
                        "The 'store U' and 'syevd' options cannot be used together.");
    }

    std::string degrees_of_freedom;
    div = (n == 1) ? 1 : n - 1;
    this->opts.get("degrees of freedom", degrees_of_freedom);
    if (degrees_of_freedom == "biased") {
        dof = -1;
        div = n;
    }

    // Initialize some workspace arrays
    da_int iwork_size = 0, sigma_size = 0, A_copy_size = 0;
    ldu = n;
    if (solver == solver_gesvdx) {
        iwork_size = 12 * std::min(n, p);
        u_size = (store_U) ? n * npc : 0;
        ldvt = npc;
        sigma_size = 2 * std::min(n, p) + 1; // TODO bug in gesvdx being fixed at 4.2
        A_copy_size = n * p;
    } else if (solver == solver_gesvd) {
        iwork_size = 0;
        u_size = (store_U) ? n * std::min(n, p) : 0;
        ldvt = std::min(n, p);
        sigma_size = std::min(n, p);
        A_copy_size = n * p;
    } else if (solver == solver_gesdd) {
        iwork_size = 8 * std::min(n, p);
        u_size = n * std::min(n, p);
        ldvt = std::min(n, p);
        sigma_size = std::min(n, p);
        A_copy_size = n * p;
    } else if (solver == solver_syevd) {
        sigma_size = p;
        ldvt = p;
    }

    try {
        u.resize(u_size, 0.0);
        sigma.resize(sigma_size, 0.0);
        vt.resize(ldvt * p, 0.0);
        iwork.resize(iwork_size);
        A_copy.resize(A_copy_size);
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    if (solver == solver_syevd) {
        // Compute A^T A in vt
        da_blas::cblas_gemm(CblasColMajor, CblasTrans, CblasNoTrans, p, p, n, 1.0, A, lda,
                            A, lda, 0.0, vt.data(), ldvt);
    }

    // Depending on the chosen method standardize by column means and possible standard deviations
    // Note we don't use the standardize API here since are also copying the data in the same step

    switch (method) {
    case pca_method_cov:
        try {
            column_means.resize(p, 0.0);
        } catch (std::bad_alloc const &) {
            return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }
        da_basic_statistics::mean(da_axis_col, n, p, A, lda, column_means.data());

        if (solver == solver_syevd) {
#pragma omp simd
            for (da_int j = 0; j < p; j++) {
                for (da_int i = 0; i <= j; i++) {
                    vt[i + ldvt * j] -= 2 * n * column_means[j] * column_means[i];
                }
            }
        } else {
#pragma omp simd
            for (da_int j = 0; j < p; j++) {
                for (da_int i = 0; i < n; i++) {
                    A_copy[i + j * n] = A[i + lda * j] - column_means[j];
                }
            }
        }
        break;
    case pca_method_corr:
        try {
            column_means.resize(p, 0.0);
            column_sdevs.resize(p, 0.0);
            column_sdevs_nonzero.resize(p, 0.0);
        } catch (std::bad_alloc const &) {
            return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }
        da_basic_statistics::variance(da_axis_col, n, p, A, lda, dof, column_means.data(),
                                      column_sdevs.data());
        if (solver == solver_syevd) {
#pragma omp simd
            for (da_int j = 0; j < p; j++) {
                column_sdevs[j] = sqrt(column_sdevs[j]);
                column_sdevs_nonzero[j] =
                    (column_sdevs[j] == (T)0.0) ? (T)1.0 : column_sdevs[j];
                for (da_int i = 0; i <= j; i++) {
                    vt[i + ldvt * j] -= 2 * n * column_means[j] * column_means[i];
                    vt[i + ldvt * j] /=
                        (column_sdevs_nonzero[j] * column_sdevs_nonzero[i]);
                }
            }
        } else {
#pragma omp simd
            for (da_int j = 0; j < p; j++) {
                column_sdevs[j] = sqrt(column_sdevs[j]);
                column_sdevs_nonzero[j] =
                    (column_sdevs[j] == (T)0.0) ? (T)1.0 : column_sdevs[j];
                for (da_int i = 0; i < n; i++) {
                    A_copy[i + j * n] =
                        (A[i + lda * j] - column_means[j]) / column_sdevs_nonzero[j];
                }
            }
        }
        break;
    default:
        if (solver != solver_syevd) {
            // No standardization is required, just copy the input matrix into internal matrix buffer
#pragma omp simd
            for (da_int j = 0; j < p; j++) {
                for (da_int i = 0; i < n; i++) {
                    A_copy[i + j * n] = A[i + lda * j];
                }
            }
        }
        break;
    }

    // Compute and store the total variance of the (standardized) input matrix
    total_variance = 0.0;
    if (solver == solver_syevd) {
#pragma omp simd reduction(+ : total_variance)
        for (da_int i = 0; i < p; i++) {
            total_variance += vt[i + ldvt * i];
        }
    } else {
#pragma omp simd reduction(+ : total_variance)
        for (da_int i = 0; i < n * p; i++) {
            total_variance += A_copy[i] * A_copy[i];
        }
    }
    total_variance /= div;

    // Collect the dimensions and pointers we will supply to the SVD routines
    da_int m_svd = n;
    da_int n_svd = p;
    T *A_svd = A_copy.data();

    // If necessary, perform a QR decomposition before the SVD
    std::vector<T> tau, R_blocked, tau_R_blocked, R;
    da_int n_blocks, block_size, final_block_size;
    if (solver != solver_syevd && (T)n / (T)p > 1.2) {
        // The factor is a heuristic based on flop counts of QR and SVD routines
        qr = true;
        da_status status = da_qr(n, p, A_copy, n, tau, R_blocked, tau_R_blocked, R,
                                 n_blocks, block_size, final_block_size, store_U, err);
        if (status != da_status_success)
            return da_error(err, status, // LCOV_EXCL_LINE
                            "Failed to compute QR decomposition prior to SVD.");

        A_svd = R.data();
        m_svd = p;
        n_svd = p;
    }

    // Compute SVD of standardized data matrix

    // Some variables are common to all the SVD solvers
    da_int INFO = 0;
    T estworkspace[1];
    da_int lwork = -1;

    switch (solver) {
    case solver_gesvdx: {
        char JOBU = (store_U) ? 'V' : 'N';
        char JOBVT = 'V';
        char RANGE = 'I';
        T vl = 0.0;
        T vu = 0.0;
        da_int iu = npc;
        da_int il = 1;

        // Query gesvdx for optimal work space required
        da::gesvdx(&JOBU, &JOBVT, &RANGE, &m_svd, &n_svd, A_svd, &m_svd, &vl, &vu, &il,
                   &iu, &ns, sigma.data(), u.data(), &ldu, vt.data(), &ldvt, estworkspace,
                   &lwork, iwork.data(), &INFO);

        // Handle SVD Error
        if (INFO != 0) {
            return da_error(
                err, da_status_internal_error,
                "An internal error occurred while computing the PCA. Please check "
                "the input data for undefined values.");
        }

        // Allocate the workspace required
        lwork = (da_int)estworkspace[0];
        try {
            work.resize(lwork);
        } catch (std::bad_alloc const &) {
            return da_error(err, da_status_memory_error,
                            "Memory allocation failed."); // LCOV_EXCL_LINE
        }

        INFO = 0;

        /*Call gesvdx*/
        da::gesvdx(&JOBU, &JOBVT, &RANGE, &m_svd, &n_svd, A_svd, &m_svd, &vl, &vu, &il,
                   &iu, &ns, sigma.data(), u.data(), &ldu, vt.data(), &ldvt, work.data(),
                   &lwork, iwork.data(), &INFO);

        break;
    }

    case solver_gesvd: {
        char JOBU = 'S';
        char JOBVT = 'S';

        // Query gesvd for optimal work space required
        da::gesvd(&JOBU, &JOBVT, &m_svd, &n_svd, A_svd, &m_svd, sigma.data(), u.data(),
                  &ldu, vt.data(), &ldvt, estworkspace, &lwork, &INFO);

        // Handle error
        if (INFO != 0) {
            return da_error(
                err, da_status_internal_error,
                "An internal error occurred while computing the PCA. Please check "
                "the input data for undefined values.");
        }

        // Allocate the workspace required
        lwork = (da_int)estworkspace[0];
        try {
            work.resize(lwork);
        } catch (std::bad_alloc const &) {
            return da_status_memory_error; // LCOV_EXCL_LINE
        }

        INFO = 0;

        /*Call gesvd*/
        da::gesvd(&JOBU, &JOBVT, &m_svd, &n_svd, A_svd, &m_svd, sigma.data(), u.data(),
                  &ldu, vt.data(), &ldvt, work.data(), &lwork, &INFO);

        // Handle error
        if (INFO != 0) {
            return da_error(
                err, da_status_internal_error,
                "An internal error occurred while computing the PCA. Please check "
                "the input data for undefined values.");
        }
        break;
    }

    case solver_gesdd: {
        char JOBZ = 'S';

        // Query gesdd for optimal work space required
        da::gesdd(&JOBZ, &m_svd, &n_svd, A_svd, &m_svd, sigma.data(), u.data(), &ldu,
                  vt.data(), &ldvt, estworkspace, &lwork, iwork.data(), &INFO);

        // Handle error
        if (INFO != 0) {
            return da_error(
                err, da_status_internal_error,
                "An internal error occurred while computing the PCA. Please check "
                "the input data for undefined values.");
        }

        // Allocate the workspace required
        lwork = (da_int)estworkspace[0];
        try {
            work.resize(lwork);
        } catch (std::bad_alloc const &) {
            return da_status_memory_error; // LCOV_EXCL_LINE
        }

        INFO = 0;

        /*Call gesdd*/
        da::gesdd(&JOBZ, &m_svd, &n_svd, A_svd, &m_svd, sigma.data(), u.data(), &ldu,
                  vt.data(), &ldvt, work.data(), &lwork, iwork.data(), &INFO);

        // Handle error
        if (INFO != 0) {
            return da_error(
                err, da_status_internal_error,
                "An internal error occurred while computing the PCA. Please check "
                "the input data for undefined values.");
        }

        break;
    }

    case solver_syevd: {
        char JOB = 'V';
        char UPLO = 'U';
        da_int liwork = -1;
        da_int estiworkspace[1];

        // Query syevd for workspace requirements
        da::syevd(&JOB, &UPLO, &p, vt.data(), &ldvt, sigma.data(), estworkspace, &lwork,
                  estiworkspace, &liwork, &INFO);

        // Handle eigensolver error
        if (INFO != 0) {
            return da_error(
                err, da_status_internal_error,
                "An internal error occurred while computing the PCA. Please check "
                "the input data for undefined values.");
        }

        // Allocate the workspace required
        lwork = (da_int)estworkspace[0];
        liwork = estiworkspace[0];
        try {
            work.resize(lwork);
            iwork.resize(liwork);
        } catch (std::bad_alloc const &) {
            return da_status_memory_error; // LCOV_EXCL_LINE
        }

        INFO = 0;

        da::syevd(&JOB, &UPLO, &p, vt.data(), &ldvt, sigma.data(), work.data(), &lwork,
                  iwork.data(), &liwork, &INFO);

        // Handle error
        if (INFO != 0) {
            return da_error(
                err, da_status_internal_error,
                "An internal error occurred while computing the PCA. Please check "
                "the input data for undefined values.");
        }

        // Need to reverse order of eigenvalues, square root them and transpose/reverse vt
        std::reverse(sigma.begin(), sigma.end());

        for (da_int i = 0; i < npc; i++) {
            sigma[i] = (sigma[i] < 0) ? (T)0.0 : sqrt(sigma[i]);
        }

        for (da_int i = 0; i < p; i++) {
            std::reverse(vt.begin() + i * p, vt.begin() + (i + 1) * p);
        }

#pragma omp simd
        for (da_int j = 0; j < p; j++) {
            for (da_int i = 0; i < j; i++) {
                vt[i + ldvt * j] = vt[j + ldvt * i];
            }
        }

        break;
    }
    default:
        return da_error(
            err, da_status_internal_error,
            "An internal error occurred while computing the PCA. Please check "
            "the input data for undefined values.");
    }

    if (qr && store_U) {
        // Update the relevant ns columns of U with the reflectors used in the QR decomposition
        da_status status =
            da_qr_apply(p, A_copy, n, tau, R_blocked, tau_R_blocked, n_blocks, block_size,
                        final_block_size, ns, u, ldu, err);
        if (status != da_status_success)
            return da_error(err, status, // LCOV_EXCL_LINE
                            "Failed to update U following QR decomposition.");
    }

    // Go through the ns columns of U and find the max absolute value
    // If that value is negative, flip sign of that column of U and that row of VT

    if (store_U) {
        T colmax;
        for (da_int j = 0; j < ns; j++) {
            // Look at column j of U
            colmax = (T)0.0;
            for (da_int i = 0; i < n; i++) {
                colmax =
                    std::abs(u[i + ldu * j]) > std::abs(colmax) ? u[i + ldu * j] : colmax;
            }
            if (colmax < 0) {
                // Negate column j of U and row j of VT
                for (da_int i = 0; i < n; i++) {
                    u[i + ldu * j] = -u[i + ldu * j];
                }
                for (da_int i = 0; i < p; i++) {
                    vt[j + ldvt * i] = -vt[j + ldvt * i];
                }
            }
        }
    }

    n_components = ns;

    // Update flag to true
    iscomputed = true;
    return da_status_success;
}

template <typename T>
da_status da_pca<T>::transform(da_int m, da_int p, const T *X, da_int ldx, T *X_transform,
                               da_int ldx_transform) {

    if (!iscomputed) {
        return da_warn(err, da_status_no_data,
                       "The PCA has not been computed. Please call da_pca_compute_s or "
                       "da_pca_compute_d.");
    }

    /* Check for illegal arguments */
    if (m < 1)
        return da_error(err, da_status_invalid_input,
                        "The function was called with m_samples = " + std::to_string(m) +
                            ". Constraint: m_samples >= 1.");
    if (p != this->p)
        return da_error(err, da_status_invalid_input,
                        "The function was called with m_features = " + std::to_string(p) +
                            " but the PCA has been computed with " +
                            std::to_string(this->p) + " features.");
    if (ldx < m)
        return da_error(err, da_status_invalid_input,
                        "The function was called with m_samples = " + std::to_string(m) +
                            " and ldx = " + std::to_string(ldx) +
                            ". Constraint: ldx >= m_samples.");

    if (ldx_transform < m)
        return da_error(err, da_status_invalid_input,
                        "The function was called with m_samples = " + std::to_string(m) +
                            " and ldx_transform = " + std::to_string(ldx_transform) +
                            ". Constraint: ldx_transform >= m_samples.");

    if (X == nullptr)
        return da_error(err, da_status_invalid_pointer, "The array X is null.");

    if (X_transform == nullptr)
        return da_error(err, da_status_invalid_pointer, "The array X_transform is null.");

    // We need a copy of X to avoid changing the user's data
    std::vector<T> X_copy;
    try {
        X_copy.resize(p * m);
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error,
                        "Memory allocation failed."); // LCOV_EXCL_LINE
    }

    const T *X_gemm;
    da_int ldx_gemm;

    // Standardize the new data matrix based on the standardization used in the PCA computation
    switch (method) {
    case pca_method_cov:
#pragma omp simd
        for (da_int j = 0; j < p; j++) {
            for (da_int i = 0; i < m; i++) {
                X_copy[i + j * m] = X[i + ldx * j] - column_means[j];
            }
        }
        X_gemm = X_copy.data();
        ldx_gemm = m;
        break;
    case pca_method_corr:
#pragma omp simd
        for (da_int j = 0; j < p; j++) {
            for (da_int i = 0; i < m; i++) {
                X_copy[i + j * m] =
                    (X[i + ldx * j] - column_means[j]) / column_sdevs_nonzero[j];
            }
        }
        X_gemm = X_copy.data();
        ldx_gemm = m;
        break;
    default:
        // No standardization is required
        X_gemm = X;
        ldx_gemm = ldx;
        break;
    }

    // Compute X * VT^T and store in transformed_data
    da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, m, ns, p, 1.0, X_gemm,
                        ldx_gemm, vt.data(), ldvt, 0.0, X_transform, ldx_transform);

    return da_status_success;
}

template <typename T>
da_status da_pca<T>::inverse_transform(da_int k, da_int r, const T *X, da_int ldx,
                                       T *X_inv_transform, da_int ldx_inv_transform) {

    if (!iscomputed) {
        return da_warn(err, da_status_no_data,
                       "The PCA has not been computed. Please call da_pca_compute_s or "
                       "da_pca_compute_d.");
    }

    /* Check for illegal arguments */
    if (k < 1)
        return da_error(err, da_status_invalid_input,
                        "The function was called with k_samples = " + std::to_string(k) +
                            ". Constraint: k_samples >= 1.");
    if (r != ns)
        return da_error(err, da_status_invalid_input,
                        "The function was called with k_features = " + std::to_string(r) +
                            " but the PCA has been computed with " + std::to_string(ns) +
                            " components.");
    if (ldx < k)
        return da_error(err, da_status_invalid_input,
                        "The function was called with k_samples = " + std::to_string(k) +
                            " and ldy = " + std::to_string(ldx) +
                            ". Constraint: ldy >= k_samples.");

    if (ldx_inv_transform < k)
        return da_error(
            err, da_status_invalid_input,
            "The function was called with k_samples = " + std::to_string(k) +
                " and ldy_inv_transform = " + std::to_string(ldx_inv_transform) +
                ". Constraint: ldy_inv_transform >= k_samples.");

    if (X == nullptr)
        return da_error(err, da_status_invalid_pointer, "The array Y is null.");

    if (X_inv_transform == nullptr)
        return da_error(err, da_status_invalid_pointer,
                        "The array Y_inv_transform is null.");

    // Compute X * VT and store
    da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, k, p, r, 1.0, X, ldx,
                        vt.data(), ldvt, 0.0, X_inv_transform, ldx_inv_transform);

    // Undo the standardization used in the PCA computation
    switch (method) {
    case pca_method_cov:
        da_basic_statistics::standardize(da_axis_col, k, p, X_inv_transform,
                                         ldx_inv_transform, dof, 1, column_means.data(),
                                         (T *)nullptr);
        break;
    case pca_method_corr:
        da_basic_statistics::standardize(da_axis_col, k, p, X_inv_transform,
                                         ldx_inv_transform, dof, 1, column_means.data(),
                                         column_sdevs.data());
        break;
    default:
        // No standardization is required
        break;
    }

    return da_status_success;
}

} // namespace da_pca

#endif //PCA_HPP
