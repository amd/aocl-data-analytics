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
#include "lapack_templates.hpp"
#include "options.hpp"
#include "pca_options.hpp"
#include <iostream>
#include <string.h>

namespace da_pca {

/* PCA class */
template <typename T> class da_pca : public basic_handle<T> {
  private:
    // n x p (samples x features)
    da_int n = 0;
    da_int p = 0;

    // If we go on to transform another data matrix, store the number of rows here
    da_int m = 0;

    // If we go on to compute the inverse transform of another data matrix, store the number of rows here
    da_int k = 0;

    // Set true when initialization is complete
    bool initdone = false;

    // Set true when PCA is computed successfully
    bool iscomputed = false;

    // Correlation or covariance based PCA
    pca_method method = pca_method_cov;

    // Sign flip flag for consistency with sklearn results
    bool svd_flip_u_based = true;

    // Number of principal components requested
    da_int npc = 1;

    // Actual number of principal components found - on output should be the same as npc unless dgesvdx gives unexpected behaviour
    da_int ns = 0;

    // Pointer to error trace
    da_errors::da_error_t *err = nullptr;

    // Arrays used by the SVD, and to store results
    std::vector<T> scores;                     // U*Sigma
    std::vector<T> variance;                   // Sigma**2 / n-1
    std::vector<T> principal_components;       // Vt
    std::vector<T> column_means, column_sdevs; // Store standardization data
    std::vector<T> transformed_data;           // Used by the transform function
    std::vector<T> inverse_transformed_data;   // Used by the inverse_transform function
    T total_variance = 0.0;                    // Sum((MeanCentered A [][])**2)
    da_int n_components = 0;
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

    da_status transform(da_int m, da_int p, const T *X, da_int ldx);

    da_status inverse_transform(da_int k, da_int r, const T *X, da_int ldx);

    da_status get_result(da_result query, da_int *dim, T *result) {
        da_status status = da_status_success;

        // Don't return anything if PCA has not been computed
        if (!iscomputed) {
            return da_warn(err, da_status_no_data,
                           "PCA has not yet been computed. Please call da_pca_compute_s "
                           "or da_pca_compute_d before extracting results.");
        }

        da_int rinfo_size = 5;

        if (result == nullptr || *dim <= 0) {
            return da_warn(err, da_status_invalid_array_dimension,
                           "The results array has not been allocated, or an unsuitable "
                           "dimension has been provided.");
        }

        switch (query) {
        case da_result::da_rinfo:
            if (*dim < rinfo_size)
                return da_warn(err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(rinfo_size) + ".");
            result[0] = (T)n;
            result[1] = (T)p;
            result[2] = (T)ns;
            result[3] = (T)m;
            result[4] = (T)k;
            break;
        case da_result::da_pca_scores:
            if (*dim < n * ns)
                return da_warn(err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(n * ns) + ".");
            for (da_int i = 0; i < n * ns; i++)
                result[i] = scores[i];
            break;
        case da_result::da_pca_u:
            if (*dim < n * ns)
                return da_warn(err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(n * ns) + ".");
            for (da_int i = 0; i < n * ns; i++)
                result[i] = u[i];
            break;
        case da_result::da_pca_principal_components:
            if (*dim < ns * p)
                return da_warn(err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(ns * p) + ".");
            for (da_int i = 0; i < ns * p; i++)
                result[i] = principal_components[i];
            break;
        case da_result::da_pca_vt:
            if (*dim < npc * p)
                return da_warn(err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(npc * p) + ".");
            for (da_int i = 0; i < npc * p; i++)
                result[i] = vt[i];
            break;
        case da_result::da_pca_variance:
            if (*dim < ns)
                return da_warn(this->err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(ns) + ".");
            for (da_int i = 0; i < ns; i++)
                result[i] = variance[i];
            break;
        case da_result::da_pca_sigma:
            if (*dim < ns)
                return da_warn(this->err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(ns) + ".");
            for (da_int i = 0; i < ns; i++)
                result[i] = sigma[i];
            break;
        case da_result::da_pca_column_means:
            if (method == pca_method_svd)
                return da_warn(err, da_status_unknown_query,
                               "Column means are only computed if the 'PCA method' "
                               "option is set to 'covariance' or 'correlation'.");
            if (*dim < p)
                return da_warn(err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(p) + ".");
            for (da_int i = 0; i < p; i++)
                result[i] = column_means[i];
            break;
        case da_result::da_pca_column_sdevs:
            if (method != pca_method_corr)
                return da_warn(err, da_status_unknown_query,
                               "Standard deviations are only computed if the 'PCA "
                               "method' option is set to 'correlation'.");
            if (*dim < p)
                return da_warn(this->err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(p) + ".");
            for (da_int i = 0; i < p; i++)
                result[i] = column_sdevs[i];
            break;
        case da_result::da_pca_total_variance:
            result[0] = total_variance;
            break;
        case da_result::da_pca_transformed_data:
            if (m == 0)
                return da_warn(err, da_status_unknown_query,
                               "No data matrices have been transformed yet. Please call "
                               "da_pca_transform_s or da_pca_transform_d before "
                               "extracting transformed data.");
            if (*dim < m * ns)
                return da_warn(err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(ns * m) + ".");
            for (da_int i = 0; i < ns * m; i++)
                result[i] = transformed_data[i];
            break;
        case da_result::da_pca_inverse_transformed_data:
            if (k == 0)
                return da_warn(
                    err, da_status_unknown_query,
                    "No data matrices have been inverse transformed yet. Please call "
                    "da_pca_inverse_transform_s or da_pca_inverse_transform_d before "
                    "extracting inverse transformed data.");
            if (*dim < k * p)
                return da_warn(err, da_status_invalid_array_dimension,
                               "The array is too small. Please provide an array of at "
                               "least size: " +
                                   std::to_string(k * p) + ".");
            for (da_int i = 0; i < k * p; i++)
                result[i] = inverse_transformed_data[i];
            break;
        default:
            return da_warn(err, da_status_unknown_query,
                           "The requested result could not be found.");
        }
        return status;
    };

    da_status get_result([[maybe_unused]] da_result query, [[maybe_unused]] da_int *dim, [[maybe_unused]] da_int *result) {

        return da_warn(err, da_status_unknown_query,
                       "There are no integer results available for this API.");
    };
};

/* Store the user's data matrix in preparation for PCA computation */
template <typename T>
da_status da_pca<T>::init(da_int n, da_int p, const T *A, da_int lda) {

    da_status status = da_status_success;

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

    // Store dimensions of A and rezero other scalars in case handle is being reused
    this->n = n;
    this->p = p;
    this->m = 0;
    this->k = 0;

    A_copy.resize(n * p);

    // Copy the input matrix into internal matrix buffer
    for (da_int j = 0; j < p; j++) {
        for (da_int i = 0; i < n; i++) {
            A_copy[i + j * n] = A[i + lda * j];
        }
    }

    // Record that initialization is complete but computation has not yet been performed
    initdone = true;
    iscomputed = false;

    // Now that we have a data matrix we can reregister the n_components option with new constraints
    da_int npc, max_npc = std::min(n, p);
    opts.get("n_components", npc);
    reregister_pca_option<T>(opts, max_npc);

    if (npc > max_npc)
        return da_warn(
            err, da_status_incompatible_options,
            "The requested number of principal components has been decreased from " +
                std::to_string(npc) + " to " + std::to_string(max_npc) +
                " due to the size (" + std::to_string(n) + " x " + std::to_string(p) +
                ") of the data array.");

    return status;
}

/* Compute the PCA */
template <typename T> da_status da_pca<T>::compute() {

    if (initdone == false)
        return da_error(err, da_status_no_data,
                        "No data has been passed to the handle. Please call "
                        "da_pca_set_data_s or da_pca_set_data_d.");

    // Read in options and store in class
    this->opts.get("n_components", npc);
    std::string opt_method;
    this->opts.get("PCA method", opt_method);
    if (opt_method == "covariance")
        method = pca_method_cov;
    else if (opt_method == "correlation")
        method = pca_method_corr;
    else
        method = pca_method_svd;

    // Initialize some workspace arrays
    u.resize(n * npc);
    //TODO: without the 2 it falls over - potential bug in flame being investigated
    sigma.resize(2 * std::min(n, p) + 1);
    vt.resize(npc * p);
    iwork.resize(12 * std::min(n, p));

    // Depending on the chosen method standardize by column means and possible standard deviations
    column_means.resize(p);

    switch (method) {
    case pca_method_cov:
        da_basic_statistics::mean(da_axis_col, n, p, A_copy.data(), n,
                                  column_means.data());
        da_basic_statistics::standardize(da_axis_col, n, p, A_copy.data(), n,
                                         column_means.data(), (T *)nullptr);
        break;
    case pca_method_corr:
        column_sdevs.resize(p);
        da_basic_statistics::variance(da_axis_col, n, p, A_copy.data(), n,
                                      column_means.data(), column_sdevs.data());
        for (da_int i = 0; i < p; i++)
            column_sdevs[i] = std::sqrt(column_sdevs[i]);
        da_basic_statistics::standardize(da_axis_col, n, p, A_copy.data(), n,
                                         column_means.data(), column_sdevs.data());
        break;
    default:
        // No standardization is required
        break;
    }

    // Compute and store the total variance of the (standardized) input matrix
    da_int div = (n == 1) ? 1 : n - 1;
    total_variance = 0.0;
    for (da_int j = 0; j < p; j++) {
        for (da_int i = 0; i < n; i++) {
            total_variance += A_copy[i + n * j] * A_copy[i + n * j];
        }
    }

    total_variance /= div;

    // Compute SVD of standardized data matrix
    char JOBU = 'V';
    char JOBVT = 'V';
    char RANGE = 'I';
    da_int INFO = 0;
    T vl = 0.0;
    T vu = 0.0;
    T estworkspace[1];
    da_int iu = npc;
    da_int il = 1;

    ns = npc;

    // Query gesvdx for optimal work space required
    da_int lwork = -1;

    da::gesvdx(&JOBU, &JOBVT, &RANGE, &n, &p, A_copy.data(), &n, &vl, &vu, &il, &iu, &ns,
               sigma.data(), u.data(), &n, vt.data(), &npc, estworkspace, &lwork,
               iwork.data(), &INFO);

    // Handle SVD Error
    if (INFO != 0) {
        return da_error(err, da_status_internal_error,
                        "An internal error occured while computing the PCA. Please check "
                        "for input data for undefined values.");
    }

    // Allocate the workspace required
    lwork = (da_int)estworkspace[0];
    work.resize(lwork);

    INFO = -1;

    /*Call gesvdx*/
    da::gesvdx(&JOBU, &JOBVT, &RANGE, &n, &p, A_copy.data(), &n, &vl, &vu, &il, &iu, &ns,
               sigma.data(), u.data(), &n, vt.data(), &npc, work.data(), &lwork,
               iwork.data(), &INFO);

    // Handle SVD Error
    if (INFO != 0) {
        return da_error(err, da_status_internal_error,
                        "An internal error occured while computing the PCA. Please check "
                        "for input data for undefined values.");
    }

    // Go through the ns columns of U and find the max absolute value
    // If that value is negative, flip sign of that column of U and that row of VT

    T colmax;
    for (da_int j = 0; j < ns; j++) {
        // Look at column j of U
        colmax = (T)0.0;
        for (da_int i = 0; i < n; i++) {
            colmax = std::abs(u[i + n * j]) > std::abs(colmax) ? u[i + n * j] : colmax;
        }
        if (colmax < 0) {
            // Negate column j of U and row j of VT
            for (da_int i = 0; i < n; i++) {
                u[i + n * j] = -u[i + n * j];
            }
            for (da_int i = 0; i < p; i++) {
                vt[j + ns * i] = -vt[j + ns * i];
            }
        }
    }

    // Allocate space for the results
    principal_components.resize(ns * p);
    scores.resize(n * ns);
    variance.resize(ns);

    // Compute Scores matrix, U * Sigma
    for (da_int j = 0; j < ns; j++) {
        for (da_int i = 0; i < n; i++) {
            scores[j * n + i] = sigma[j] * u[j * n + i];
        }
    }

    // Save the principal components
    // Note careful use of ns vs npc: they should be identical, but this guards against unexpected dgesvdx behaviour
    for (da_int j = 0; j < p; j++) {
        for (da_int i = 0; i < ns; i++) {
            principal_components[j * ns + i] = vt[j * npc + i];
        }
    }

    // Compute variance, (Sigma**2) / (n_samples-1)
    for (da_int j = 0; j < ns; j++) {
        variance[j] = sigma[j] * sigma[j] / div;
    }

    n_components = ns;

    // Update flag to true
    iscomputed = true;

    return da_status_success;
}

template <typename T>
da_status da_pca<T>::transform(da_int m, da_int p, const T *X, da_int ldx) {

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

    this->m = m;

    // We need a copy of X to avoid changing the user's data
    std::vector<T> X_copy(p * m);
    for (da_int j = 0; j < p; j++) {
        for (da_int i = 0; i < m; i++) {
            X_copy[i + j * m] = X[i + ldx * j];
        }
    }

    // Standardize the new data matrix based on the standardization used in the PCA computation
    switch (method) {
    case pca_method_cov:
        da_basic_statistics::standardize(da_axis_col, m, p, X_copy.data(), m,
                                         column_means.data(), (T *)nullptr);
        break;
    case pca_method_corr:
        da_basic_statistics::standardize(da_axis_col, m, p, X_copy.data(), m,
                                         column_means.data(), column_sdevs.data());
        break;
    default:
        // No standardization is required
        break;
    }

    // Compute X * VT^T and store in transformed_data
    transformed_data.resize(m * ns);
    da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, m, ns, p, 1.0,
                        X_copy.data(), m, principal_components.data(), ns, 0.0,
                        transformed_data.data(), m);

    return da_status_success;
}

template <typename T>
da_status da_pca<T>::inverse_transform(da_int k, da_int r, const T *X, da_int ldx) {

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
                            " and ldx = " + std::to_string(ldx) +
                            ". Constraint: ldx >= k_samples.");

    this->k = k;

    // Compute X * VT and store in inverse_transformed_data
    inverse_transformed_data.resize(k * p);
    da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, k, p, r, 1.0, X, ldx,
                        principal_components.data(), ns, 0.0,
                        inverse_transformed_data.data(), k);

    // Undo the standardization used in the PCA computation
    switch (method) {
    case pca_method_cov:
        for (da_int j = 0; j < p; j++) {
            for (da_int i = 0; i < k; i++) {
                inverse_transformed_data[i + k * j] += column_means[j];
            }
        }
        break;
    case pca_method_corr:
        for (da_int j = 0; j < p; j++) {
            for (da_int i = 0; i < k; i++) {
                inverse_transformed_data[i + k * j] =
                    inverse_transformed_data[i + k * j] * column_sdevs[j] +
                    column_means[j];
            }
        }
        break;
    default:
        // No standardization is required
        break;
    }

    return da_status_success;
}

} // namespace da_pca

#endif //PCA_HPP
