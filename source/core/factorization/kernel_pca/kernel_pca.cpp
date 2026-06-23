/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

#include "kernel_pca.hpp"
#include "aoclda.h"
#include "aoclda_pca.h"
#include "basic_statistics.hpp"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "da_randsvd.hpp"
#include "da_std.hpp"
#include "da_utils.hpp"
#include "kernel_functions.hpp"
#include "kernel_pca_options.hpp"
#include "kernel_pca_types.hpp"
#include "lapack_templates.hpp"
#include "macros.h"
#include "options.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace ARCH {

namespace da_kernel_pca {

using namespace da_kernel_pca_types;
using namespace da_model_persistence;

template <typename T>
kernel_pca<T>::kernel_pca(da_errors::da_error_t &err) : basic_handle<T>(err) {
    register_kernel_pca_options<T>(this->opts, err);
}

template <typename T>
da_status kernel_pca<T>::get_result(da_result query, da_int *dim, T *result) {

    if (!this->model_trained)
        return da_warn(
            this->err, da_status_no_data,
            "Kernel PCA has not been computed. Please call da_kernel_pca_compute_? "
            "before querying results.");

    switch (query) {
    case da_rinfo: {
        if (*dim < 2) {
            *dim = 2;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The results array is too small. Please provide an array of "
                           "size at least 2.");
        }
        result[0] = static_cast<T>(n_samples);
        result[1] = static_cast<T>(n_features);
        return da_status_success;
    }

    case da_kernel_pca_gamma: {
        da_int required = 1;
        if (*dim < required) {
            *dim = required;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The array is too small. Please provide an array of at "
                           "least size: " +
                               std::to_string(required) + ".");
        }
        result[0] = gamma;
        return da_status_success;
    }

    case da_kernel_pca_scores: {
        da_int required = n_samples * n_components;
        if (*dim < required) {
            *dim = required;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The results array is too small. Please provide an array of "
                           "size at least " +
                               std::to_string(required) + ".");
        }
        if (this->order == da_order::column_major) {
            for (da_int j = 0; j < n_components; j++) {
                T sq = std::sqrt(eigenvalues[j]);
#pragma omp simd
                for (da_int i = 0; i < n_samples; i++) {
                    result[i + j * n_samples] = eigenvectors[i + j * n_samples] * sq;
                }
            }
        } else {
            std::vector<T> sqrt_eig;
            try {
                sqrt_eig.resize(n_components);
            } catch (std::bad_alloc const &) {
                return da_error(this->err, da_status_memory_error,
                                "Memory allocation failed.");
            }
            for (da_int j = 0; j < n_components; j++)
                sqrt_eig[j] = std::sqrt(eigenvalues[j]);

            for (da_int i = 0; i < n_samples; i++) {
                da_int row_offset = i * n_components;
#pragma omp simd
                for (da_int j = 0; j < n_components; j++)
                    result[row_offset + j] = eigenvectors[row_offset + j] * sqrt_eig[j];
            }
        }
        return da_status_success;
    }

    case da_kernel_pca_eigenvalues: {
        da_int required = n_components;
        if (*dim < required) {
            *dim = required;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The results array is too small. Please provide an array of "
                           "size at least " +
                               std::to_string(required) + ".");
        }
        std::copy(eigenvalues.begin(), eigenvalues.end(), result);
        return da_status_success;
    }

    case da_kernel_pca_eigenvectors: {
        da_int required = n_samples * n_components;
        if (*dim < required) {
            *dim = required;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The results array is too small. Please provide an array of "
                           "size at least " +
                               std::to_string(required) + ".");
        }
        std::copy(eigenvectors.begin(), eigenvectors.end(), result);
        return da_status_success;
    }

    case da_kernel_pca_dual_coef: {
        if (!inverse_fitted)
            return da_warn(this->err, da_status_no_data,
                           "Dual coefficients are only available when 'fit inverse "
                           "transform' is set to 1.");
        da_int required = n_samples * n_features;
        if (*dim < required) {
            *dim = required;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The results array is too small. Please provide an array of "
                           "size at least " +
                               std::to_string(required) + ".");
        }
        std::copy(dual_coef.begin(), dual_coef.end(), result);
        return da_status_success;
    }

    case da_kernel_pca_X_fit: {
        da_int required = n_samples * n_features;
        if (*dim < required) {
            *dim = required;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The results array is too small. Please provide an array of "
                           "size at least " +
                               std::to_string(required) + ".");
        }
        if (this->order == da_order::column_major) {
            for (da_int j = 0; j < n_features; j++)
                memcpy(result + j * n_samples, A_ptr + j * this->lda,
                       n_samples * sizeof(T));
        } else {
            for (da_int i = 0; i < n_samples; i++)
                memcpy(result + i * n_features, A_ptr + i * this->lda,
                       n_features * sizeof(T));
        }
        return da_status_success;
    }

    default:
        return da_warn(this->err, da_status_unknown_query,
                       "The requested result is not available for kernel PCA.");
    }
}

template <typename T>
da_status kernel_pca<T>::get_result(da_result query, da_int *dim, da_int *result) {
    da_status status = this->get_result_common(query, dim, result);
    if (status != da_status_unknown_query)
        return status;

    if (!this->model_trained)
        return da_warn(
            this->err, da_status_no_data,
            "Kernel PCA has not been computed. Please call da_kernel_pca_compute_? "
            "before querying results.");

    switch (query) {
    case da_kernel_pca_n_components:
        if (*dim < 1) {
            *dim = 1;
            return da_warn(this->err, da_status_invalid_array_dimension,
                           "The results array is too small. Please provide an array of "
                           "size at least 1.");
        }
        result[0] = n_components;
        return da_status_success;
    default:
        return da_warn(this->err, da_status_unknown_query,
                       "The requested result is not available for kernel PCA.");
    }
}

template <typename T>
da_status kernel_pca<T>::init(da_int n_samples_in, da_int n_features_in, const T *A,
                              da_int lda_in) {
    // Read options and user data
    std::string opt_order;
    da_int iorder;
    this->opts.get("storage order", opt_order, iorder);
    this->order = da_order(iorder);

    da_status status =
        this->check_2D_array(this->order, n_samples_in, n_features_in, A, lda_in,
                             "n_samples", "n_features", "A", "lda");
    if (status != da_status_success)
        return status;

    // Reset state
    A_copy.resize(0);
    A_ptr = nullptr;

    row_means.resize(0);
    grand_mean = static_cast<T>(0.0);
    eigenvalues.resize(0);
    eigenvectors.resize(0);
    dual_coef.resize(0);

    this->model_trained = false;
    inverse_fitted = false;

    this->n_samples = n_samples_in;
    this->n_features = n_features_in;

    std::string copy_data_str;
    da_int copy_data_int;
    this->opts.get("copy data", copy_data_str, copy_data_int);
    this->copy_data = (copy_data_int != 0);

    /* Store training data in its native layout.
     * When copy_data=true, copy in to A_copy.
     * When copy_data=false, store a pointer to the user's data. */
    if (this->copy_data) {
        try {
            A_copy.resize(n_samples_in * n_features_in);
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }

        if (this->order == da_order::column_major) {
            for (da_int j = 0; j < n_features_in; j++) {
                memcpy(A_copy.data() + j * n_samples_in, A + j * lda_in,
                       n_samples_in * sizeof(T));
            }

            this->lda = n_samples_in;
        } else {
            for (da_int i = 0; i < n_samples_in; i++) {
                memcpy(A_copy.data() + i * n_features_in, A + i * lda_in,
                       n_features_in * sizeof(T));
            }
            this->lda = n_features_in;
        }
        A_ptr = A_copy.data();
    } else {
        A_ptr = A;
        this->lda = lda_in;
    }

    // Now that we have a data matrix we can re-register the n_components option with new constraints
    da_int npc, max_npc = n_samples_in;
    this->opts.get("n_components", npc);

    reregister_kernel_pca_n_components<T>(this->opts, max_npc);

    this->opts.set("n_components", std::min(npc, max_npc));

    init_done = true;

    if (npc > max_npc)
        return da_warn(
            this->err, da_status_incompatible_options,
            "The requested number of principal components has been decreased from " +
                std::to_string(npc) + " to " + std::to_string(max_npc) +
                " due to the size (" + std::to_string(n_samples_in) + " x " +
                std::to_string(n_features_in) + ") of the data array.");

    return da_status_success;
}

template <typename T> da_status kernel_pca<T>::compute() {
    /*
    B. Schölkopf, A. Smola and K. -R. Müller, 
    "Nonlinear Component Analysis as a Kernel Eigenvalue Problem," 
    in Neural Computation, vol. 10, no. 5, pp. 1299-1319, 1 July 1998. [1]
    */

    // Check we have set data
    if (!init_done)
        return da_error(this->err, da_status_no_data,
                        "No data has been provided. Please call da_kernel_pca_set_data_? "
                        "before da_kernel_pca_compute_?.");

    da_int n_components_opt, kernel_int, fit_inv_int, remove_zero_int, solver_int;
    std::string kernel_str, fit_inv_str, remove_zero_str, solver_str;

    // Parse options into member variables
    // Assume all opts.get pass
    this->opts.get("n_components", n_components_opt);
    this->opts.get("kernel", kernel_str, kernel_int);
    this->opts.get("eigensolver", solver_str, solver_int);
    this->opts.get("gamma", this->gamma);
    this->opts.get("degree", this->degree);
    this->opts.get("coef0", this->coef0);
    this->opts.get("fit inverse transform", fit_inv_str, fit_inv_int);
    this->opts.get("remove zero eig", remove_zero_str, remove_zero_int);
    this->opts.get("alpha", this->alpha);
    da_int p_oversample, q_iter;
    std::string rand_normalizer;
    da_int rand_normalizer_int = 0;
    this->opts.get("n_oversamples", p_oversample);
    this->opts.get("power iterations", q_iter);
    this->opts.get("power normalization", rand_normalizer, rand_normalizer_int);
    power_normalizer_t normalizer_type =
        static_cast<power_normalizer_t>(rand_normalizer_int);

    // Validation
    this->solver = static_cast<da_kernel_pca_types::solver_type>(solver_int);
    if (this->solver == solver_rand_syevd && n_components_opt == 0) {
        return da_error(
            this->err, da_status_invalid_input,
            "n_components must be set to a non-zero value to use the randomized solver.");
    }

    this->kernel_type = static_cast<da_kernel_pca_types::pca_kernel>(kernel_int);
    this->fit_inverse = (fit_inv_int != 0);
    this->remove_zero = (remove_zero_int != 0);

    if (this->fit_inverse && this->kernel_type == pca_kernel::precomputed)
        return da_error(this->err, da_status_invalid_input,
                        "The 'fit inverse transform' option cannot be used with a "
                        "precomputed kernel.");

    if (this->kernel_type == pca_kernel::precomputed && n_features != n_samples)
        return da_error(this->err, da_status_invalid_input,
                        "For a precomputed kernel, n_features must equal n_samples (" +
                            std::to_string(n_samples) + "), but got " +
                            std::to_string(n_features) + ".");

    // Resolve gamma
    if (this->gamma < 0.0) {
        this->gamma = static_cast<T>(1.0) / static_cast<T>(n_features);
    }

    // Solver selection
    if (this->solver == solver_auto) {
        this->solver =
            (n_components_opt > 0 && n_components_opt * 10 < n_samples && n_samples > 500)
                ? solver_rand_syevd
                : solver_syevd;
    }

    // Build K_work (n_samples x n_samples, symmetric — layout-agnostic)
    std::vector<T> K_work;
    try {
        K_work.resize(n_samples * n_samples);
        row_means.resize(n_samples, 0.0);
        eigenvalues.resize(n_samples, 0.0);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    da_status status = da_status_success;
    switch (this->kernel_type) {
    case pca_kernel::linear:
        status = da_kernel_functions::linear_kernel<T>(
            this->order, n_samples, n_samples, n_features, A_ptr, this->lda, nullptr,
            n_samples, K_work.data(), n_samples);
        break;
    case pca_kernel::polynomial:
        status = da_kernel_functions::polynomial_kernel<T>(
            this->order, n_samples, n_samples, n_features, A_ptr, this->lda, nullptr,
            n_samples, K_work.data(), n_samples, this->gamma, this->degree, this->coef0);
        break;
    case pca_kernel::rbf:
        status = da_kernel_functions::rbf_kernel<T>(
            this->order, n_samples, n_samples, n_features, A_ptr, this->lda, nullptr,
            n_samples, K_work.data(), n_samples, this->gamma);
        break;
    case pca_kernel::sigmoid:
        status = da_kernel_functions::sigmoid_kernel<T>(
            this->order, n_samples, n_samples, n_features, A_ptr, this->lda, nullptr,
            n_samples, K_work.data(), n_samples, this->gamma, this->coef0);
        break;
    case pca_kernel::precomputed:
        // A_ptr points to the precomputed n_samples x n_samples kernel matrix; copy into K_work
        if (this->order == da_order::column_major) {
            for (da_int j = 0; j < n_features; j++) {
                memcpy(K_work.data() + j * n_samples, A_ptr + j * this->lda,
                       n_samples * sizeof(T));
            }
        } else {
            for (da_int i = 0; i < n_samples; i++) {
                memcpy(K_work.data() + i * n_features, A_ptr + i * this->lda,
                       n_features * sizeof(T));
            }
        }
        break;
    default:
        return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unknown kernel type.");             // LCOV_EXCL_LINE
    }

    if (status != da_status_success)
        return da_error(this->err, status, "Kernel computation failed.");

    // Center K_work in-place
    // Kt_ij = K_ij - row_means[i] - row_means[j] + grand_mean
    // grand_mean = mean(row_means)
    // row_means[i] = (1/n) * sum_j K_ij */
    // K_work is symmetric by construction if we compute it, and by assumption if user supplies it.
    // If the exact centering formula seems opaque, see Appendix B of [1]
    da_basic_statistics::mean(this->order, da_axis_row, n_samples, n_samples,
                              K_work.data(), n_samples, row_means.data());

    // Compute mean of all row_means
    da_basic_statistics::mean(row_major, da_axis_all, 1, n_samples, row_means.data(),
                              n_samples, &grand_mean);

    // Just update the upper triangular portion of K
    for (da_int j = 0; j < n_samples; j++) {
        da_int col_offset = j * n_samples;
        T correction_j = grand_mean - row_means[j];
#pragma omp simd
        for (da_int i = 0; i <= j; i++)
            K_work[i + col_offset] += correction_j - row_means[i];
    }

    // Eigendecomposition of K_work
    if (this->solver == solver_rand_syevd) {
        // Randomized path: da_random_syevd returns eigenvectors in V (descending order)
        // and eigenvalues in descending order — no reversal or K_work copy needed.
        try {
            eigenvectors.resize(n_samples * n_components_opt);
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error,
                            "Memory allocation failed.");
        }
        da_int seed;
        this->opts.get("seed", seed);
        status = da_random_syevd(n_samples, K_work.data(), n_samples, eigenvectors.data(),
                                 n_samples, eigenvalues.data(), n_components_opt,
                                 p_oversample, q_iter, seed, normalizer_type, *this->err);
        if (status != da_status_success)
            return status;
        sign_normalize_columns(eigenvectors.data(), n_samples, n_components_opt);
    } else {
        char JOB = 'V', UPLO = 'U';
        da_int lwork = -1, liwork = -1, INFO = 0;
        T estworkspace[1];
        da_int estiworkspace[1];

        // Workspace query
        da::syevd(&JOB, &UPLO, &n_samples, K_work.data(), &n_samples, eigenvalues.data(),
                  estworkspace, &lwork, estiworkspace, &liwork, &INFO);
        if (INFO != 0)
            return da_error(
                this->err, da_status_internal_error,
                "An internal error occurred while computing the kernel PCA. Please check "
                "the input data for undefined values.");

        lwork = static_cast<da_int>(estworkspace[0]);
        liwork = estiworkspace[0];

        std::vector<T> work;
        std::vector<da_int> iwork;
        try {
            work.resize(lwork);
            iwork.resize(liwork);
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }

        da::syevd(&JOB, &UPLO, &n_samples, K_work.data(), &n_samples, eigenvalues.data(),
                  work.data(), &lwork, iwork.data(), &liwork, &INFO);
        if (INFO != 0)
            return da_error(
                this->err, da_status_internal_error,
                "An internal error occurred while computing the kernel PCA. Please check "
                "the input data for undefined values.");

        // syevd returns eigenvalues ascending; K_work columns are the eigenvectors.

        // Reverse eigenvalues to descending order
        std::reverse(eigenvalues.begin(), eigenvalues.end());

        // Sign-normalize K_work columns before reversal (K_work is col-major from syevd)
        sign_normalize_columns(K_work.data(), n_samples, n_samples);
    }

    // Matching scikit-learn behaviour:
    // Always clamp near zero. Any sufficiently large -ve eigenvalues will throw an error.
    status = clamp_near_zero_eigenvalues(eigenvalues, n_components_opt);
    if (status != da_status_success)
        return status;

    // Only actually remove zeros (reduce nc) if requested
    // Determine nc: number of components to keep
    da_int nc = n_components_opt;
    if (nc == 0 || remove_zero) {
        // Find first occurence of 0
        auto it = std::lower_bound(eigenvalues.begin(), eigenvalues.end(), 0,
                                   std::greater<T>());
        da_int nnz = std::distance(eigenvalues.begin(), it);
        nc = (nc == 0) ? nnz : std::min(nnz, nc);
    }

    // Store the resolved number of components
    this->n_components = nc;

    if (this->solver == solver_rand_syevd) {
        // eigenvectors already written column-major by da_random_syevd; trim to resolved nc
        eigenvectors.resize(n_samples * n_components);
        if (this->order == da_order::row_major) {
            // Transpose in-place from column-major to row-major.
            da_blas::imatcopy('T', n_samples, n_components, (T)1, eigenvectors.data(),
                              n_samples, n_components);
        }
    } else {
        // Allocate memory for eigenvectors once we have computed nc
        try {
            eigenvectors.resize(n_samples * n_components);
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }

        // Copy reversed (descending order) eigenvectors from K_work into eigenvectors
        // in this->order layout. K_work column (n_samples-1-j) corresponds to eigenvalue j.
        if (this->order == da_order::column_major) {
            for (da_int j = 0; j < n_components; j++)
                memcpy(eigenvectors.data() + j * n_samples,
                       K_work.data() + (n_samples - 1 - j) * n_samples,
                       n_samples * sizeof(T));
        } else {
            // Write row-major eigenvectors: eigenvectors(i, j) = K_work col (n_samples-1-j), row i
            for (da_int i = 0; i < n_samples; i++) {
                da_int row_offset = i * n_components;
#pragma omp simd
                for (da_int j = 0; j < n_components; j++)
                    eigenvectors[row_offset + j] =
                        K_work[i + (n_samples - 1 - j) * n_samples];
            }
        }
    }

    // Trim eigenvalues
    eigenvalues.resize(n_components);

    // Optional inverse transform (kernel ridge regression)
    if (this->fit_inverse) {
        /*
        Bakir, Weston, Scholkopf. Learning to find pre-images.
        Advances in Neural Information Processing Systems, 16:449–456, 2004 [2]
        */

        da_std::fill(K_work.begin(), K_work.end(), 0.0);
        // Compute Z_train = V * sqrt(Lambda) in this->order layout
        // Z_train = fit_transform(A)
        da_int ldz = (this->order == da_order::column_major) ? n_samples : n_components;
        std::vector<T> Z_train;
        try {
            Z_train.resize(n_samples * n_components, 0.0);
            dual_coef.resize(n_samples * n_features);
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }

        if (this->order == da_order::column_major) {
            for (da_int j = 0; j < n_components; j++) {
                T sq = std::sqrt(eigenvalues[j]);
#pragma omp simd
                for (da_int i = 0; i < n_samples; i++)
                    Z_train[i + n_samples * j] = eigenvectors[i + n_samples * j] * sq;
            }
        } else {
            std::vector<T> sqrt_eig;
            try {
                sqrt_eig.resize(n_components);
            } catch (std::bad_alloc const &) {
                return da_error(this->err, da_status_memory_error,
                                "Memory allocation failed.");
            }
            for (da_int j = 0; j < n_components; j++)
                sqrt_eig[j] = std::sqrt(eigenvalues[j]);
            for (da_int i = 0; i < n_samples; i++) {
                da_int row_offset = i * n_components;
#pragma omp simd
                for (da_int j = 0; j < n_components; j++)
                    Z_train[row_offset + j] = eigenvectors[row_offset + j] * sqrt_eig[j];
            }
        }

        /* Build (n_samples x n_samples, symmetric) kernel matrix of Z_train with itself */
        switch (this->kernel_type) {
        case pca_kernel::linear:
            status = da_kernel_functions::linear_kernel<T>(
                this->order, n_samples, n_samples, n_components, Z_train.data(), ldz,
                nullptr, n_samples, K_work.data(), n_samples);
            break;
        case pca_kernel::polynomial:
            status = da_kernel_functions::polynomial_kernel<T>(
                this->order, n_samples, n_samples, n_components, Z_train.data(), ldz,
                nullptr, n_samples, K_work.data(), n_samples, this->gamma, this->degree,
                this->coef0);
            break;
        case pca_kernel::rbf:
            status = da_kernel_functions::rbf_kernel<T>(
                this->order, n_samples, n_samples, n_components, Z_train.data(), ldz,
                nullptr, n_samples, K_work.data(), n_samples, this->gamma);
            break;
        case pca_kernel::sigmoid:
            status = da_kernel_functions::sigmoid_kernel<T>(
                this->order, n_samples, n_samples, n_components, Z_train.data(), ldz,
                nullptr, n_samples, K_work.data(), n_samples, this->gamma, this->coef0);
            break;
        default:
            return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                            "Unexpected kernel type in inverse transform.");
        }

        if (status != da_status_success)
            return da_error(this->err, status,
                            "Kernel computation for inverse transform failed.");

            // Add ridge regularization: K_work += alpha * I
#pragma omp simd
        for (da_int i = 0; i < n_samples; i++)
            K_work[i + i * n_samples] += this->alpha;

        // Cholesky factorization (K_work is symmetric)
        char UPLO2 = 'U';
        da_int INFO2 = 0;
        da::potrf(&UPLO2, &n_samples, K_work.data(), &n_samples, &INFO2);
        if (INFO2 > 0) {
            return da_error(this->err, da_status_numerical_difficulties,
                            "Cholesky factorization failed. "
                            "The kernel matrix may not be positive definite. "
                            "Try increasing the \"alpha\" option.");
        } else if (INFO2 < 0) {
            return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                            "potrf failed with illegal argument.");
        }

        // Solve (K_work + alpha*I) dual_coef = A for dual_coef (n_samples x n_features)
        // potrs requires col-major RHS, so copy A into dual_coef in col-major order
        if (this->order == da_order::column_major) {
            for (da_int j = 0; j < n_features; j++)
                memcpy(dual_coef.data() + j * n_samples, A_ptr + j * this->lda,
                       n_samples * sizeof(T));
        } else {
            da_blas::omatcopy('T', n_features, n_samples, static_cast<T>(1), A_ptr,
                              this->lda, dual_coef.data(), n_samples);
        }

        da::potrs(&UPLO2, &n_samples, &n_features, K_work.data(), &n_samples,
                  dual_coef.data(), &n_samples, &INFO2);
        if (INFO2 != 0)
            return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                            "potrs triangular solve failed.");

        // Transpose solution back to row-major if needed
        if (this->order == da_order::row_major)
            da_blas::imatcopy('T', n_samples, n_features, static_cast<T>(1),
                              dual_coef.data(), n_samples, n_features);

        inverse_fitted = true;
    }

    this->model_trained = true;
    return da_status_success;
}

template <typename T> da_status kernel_pca<T>::check_options_update() {
    da_int kernel_int_check, degree_check;
    T gamma_check, coef0_check;
    std::string kernel_str_check;

    this->opts.get("kernel", kernel_str_check, kernel_int_check);
    this->opts.get("gamma", gamma_check);
    this->opts.get("degree", degree_check);
    this->opts.get("coef0", coef0_check);

    // Resolve gamma
    if (gamma_check < 0.0)
        gamma_check = static_cast<T>(1.0) / static_cast<T>(n_features);

    if (static_cast<da_kernel_pca_types::pca_kernel>(kernel_int_check) !=
            this->kernel_type ||
        degree_check != this->degree || gamma_check != this->gamma ||
        coef0_check != this->coef0)
        return da_error(this->err, da_status_incompatible_options,
                        "Kernel parameters have been changed since compute() was "
                        "called. Please call compute() again.");

    return da_status_success;
}

template <typename T>
da_status kernel_pca<T>::transform(da_int m_samples, da_int m_features, const T *X,
                                   da_int ldx, T *X_transform, da_int ldx_transform) {

    if (!this->model_trained)
        return da_error(
            this->err, da_status_no_data,
            "Kernel PCA has not been computed. Please call da_kernel_pca_compute_? "
            "before da_kernel_pca_transform_?.");

    da_status status = check_options_update();
    if (status != da_status_success)
        return status;

    // Validate dimensions
    if (this->kernel_type == pca_kernel::precomputed) {
        // If kernel is precomputed, user must supply the m x n_samples cross kernel matrix
        if (m_features != n_samples) {
            return da_error(this->err, da_status_invalid_input,
                            "For precomputed kernel, m_features must equal the number of "
                            "training samples (" +
                                std::to_string(n_samples) + "), but got " +
                                std::to_string(m_features) + ".");
        }
    } else {
        if (m_features != n_features)
            return da_error(this->err, da_status_invalid_input,
                            "da_kernel_pca_transform_? was called with m_features = " +
                                std::to_string(m_features) +
                                " but the model was computed with " +
                                std::to_string(n_features) + " features.");
    }

    status = this->check_2D_array(this->order, m_samples, m_features, X, ldx, "m_samples",
                                  "m_features", "X", "ldx");
    if (status != da_status_success)
        return status;

    status = this->check_2D_array(this->order, m_samples, n_components, X_transform,
                                  ldx_transform, "m_samples", "n_components",
                                  "X_transform", "ldx_transform");
    if (status != da_status_success)
        return status;

    const da_int m = m_samples;
    // Build cross-kernel K_new (m x n_samples) in this->order layout
    // K_new[i,j] = k(x_new_i, x_train_j)
    da_int ldk = (this->order == da_order::column_major) ? m : n_samples;
    std::vector<T> K_new, new_row_means;
    try {
        K_new.resize(m * n_samples);
        new_row_means.resize(m, static_cast<T>(0));
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    switch (this->kernel_type) {
    case pca_kernel::linear:
        status = da_kernel_functions::linear_kernel<T>(this->order, m, n_samples,
                                                       n_features, X, ldx, A_ptr,
                                                       this->lda, K_new.data(), ldk);
        break;
    case pca_kernel::polynomial:
        status = da_kernel_functions::polynomial_kernel<T>(
            this->order, m, n_samples, n_features, X, ldx, A_ptr, this->lda, K_new.data(),
            ldk, this->gamma, this->degree, this->coef0);
        break;
    case pca_kernel::rbf:
        status = da_kernel_functions::rbf_kernel<T>(this->order, m, n_samples, n_features,
                                                    X, ldx, A_ptr, this->lda,
                                                    K_new.data(), ldk, this->gamma);
        break;
    case pca_kernel::sigmoid:
        status = da_kernel_functions::sigmoid_kernel<T>(
            this->order, m, n_samples, n_features, X, ldx, A_ptr, this->lda, K_new.data(),
            ldk, this->gamma, this->coef0);
        break;
    case pca_kernel::precomputed:
        if (this->order == da_order::column_major) {
            for (da_int j = 0; j < n_samples; j++)
                memcpy(K_new.data() + j * m, X + j * ldx, m * sizeof(T));
        } else {
            for (da_int i = 0; i < m; i++)
                memcpy(K_new.data() + i * n_samples, X + i * ldx, n_samples * sizeof(T));
        }
        break;
    default:
        return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unknown kernel type.");             // LCOV_EXCL_LINE
    }

    if (status != da_status_success)
        return da_error(this->err, status, "Cross-kernel computation failed.");

    // Center K_new using training statistics
    // K̃_new[i,j] = K_new[i,j] - new_row_means[i] - row_means[j] + grand_mean
    // See Appendix B of [1] for centering details
    da_basic_statistics::mean(this->order, da_axis_row, m, n_samples, K_new.data(), ldk,
                              new_row_means.data());

    if (this->order == da_order::column_major) {
        for (da_int j = 0; j < n_samples; j++) {
            da_int col_offset = j * m;
            T correction_j = grand_mean - row_means[j];
#pragma omp simd
            for (da_int i = 0; i < m; i++)
                K_new[i + col_offset] += correction_j - new_row_means[i];
        }
    } else {
        for (da_int i = 0; i < m; i++) {
            da_int row_offset = i * n_samples;
            T correction_i = grand_mean - new_row_means[i];
#pragma omp simd
            for (da_int j = 0; j < n_samples; j++)
                K_new[row_offset + j] += correction_i - row_means[j];
        }
    }

    // X_transform = K_new * (eigenvectors / sqrt(eigenvalues))
    // Accumulate K_new * eigenvectors in X_transform then scale columns
    da_int ldev;
    CBLAS_ORDER cblas_order;
    if (this->order == column_major) {
        ldev = n_samples;
        cblas_order = CblasColMajor;
    } else {
        ldev = n_components;
        cblas_order = CblasRowMajor;
    }

    da_blas::cblas_gemm(cblas_order, CblasNoTrans, CblasNoTrans, m, n_components,
                        n_samples, static_cast<T>(1), K_new.data(), ldk,
                        eigenvectors.data(), ldev, static_cast<T>(0), X_transform,
                        ldx_transform);

    if (this->order == da_order::column_major) {
        for (da_int j = 0; j < n_components; j++) {
            T scale = (eigenvalues[j] > static_cast<T>(0))
                          ? static_cast<T>(1) / std::sqrt(eigenvalues[j])
                          : static_cast<T>(0);
#pragma omp simd
            for (da_int i = 0; i < m; i++)
                X_transform[i + ldx_transform * j] *= scale;
        }
    } else {
        std::vector<T> scale_vec;
        try {
            scale_vec.resize(n_components);
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error,
                            "Memory allocation failed.");
        }
        for (da_int j = 0; j < n_components; j++)
            scale_vec[j] = (eigenvalues[j] > static_cast<T>(0))
                               ? static_cast<T>(1) / std::sqrt(eigenvalues[j])
                               : static_cast<T>(0);
        for (da_int i = 0; i < m; i++) {
            da_int row_offset = i * ldx_transform;
#pragma omp simd
            for (da_int j = 0; j < n_components; j++)
                X_transform[row_offset + j] *= scale_vec[j];
        }
    }

    return da_status_success;
}

template <typename T>
da_status kernel_pca<T>::inverse_transform(da_int k, da_int nc_in, const T *Y, da_int ldy,
                                           T *Y_inv_transform, da_int ldy_inv_transform) {

    if (!this->model_trained)
        return da_error(
            this->err, da_status_no_data,
            "Kernel PCA has not been computed. Please call da_kernel_pca_compute_? "
            "before da_kernel_pca_inverse_transform_?.");

    if (this->kernel_type == pca_kernel::precomputed)
        return da_error(
            this->err, da_status_invalid_input,
            "Inverse transform is not supported with the precomputed kernel.");

    if (!inverse_fitted)
        return da_error(this->err, da_status_no_data,
                        "Inverse transform has not been fitted. Set option \"fit inverse "
                        "transform\" to 1 before calling da_kernel_pca_compute_?.");

    da_status status = check_options_update();
    if (status != da_status_success)
        return status;

    // Validate dimensions
    if (nc_in != n_components)
        return da_error(this->err, da_status_invalid_input,
                        "k_components = " + std::to_string(nc_in) +
                            " does not match the number of components (" +
                            std::to_string(n_components) + ") from compute().");

    status = this->check_2D_array(this->order, k, nc_in, Y, ldy, "k_samples",
                                  "k_components", "Y", "ldy");
    if (status != da_status_success)
        return status;

    status = this->check_2D_array(this->order, k, n_features, Y_inv_transform,
                                  ldy_inv_transform, "k_samples", "n_features",
                                  "Y_inv_transform", "ldy_inv_transform");
    if (status != da_status_success)
        return status;

    std::vector<T> Z_train, K_cross;
    da_int ldz = (this->order == da_order::column_major) ? n_samples : n_components;
    da_int ldk = (this->order == da_order::column_major) ? k : n_samples;
    try {
        Z_train.resize(n_samples * n_components);
        K_cross.resize(k * n_samples);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    // Recompute Z_train = V * sqrt(Lambda), shape (n_samples x nc), in this->order
    if (this->order == da_order::column_major) {
        for (da_int j = 0; j < n_components; j++) {
            T sq = std::sqrt(eigenvalues[j]);
#pragma omp simd
            for (da_int i = 0; i < n_samples; i++)
                Z_train[i + n_samples * j] = eigenvectors[i + n_samples * j] * sq;
        }
    } else {
        std::vector<T> sqrt_eig;
        try {
            sqrt_eig.resize(n_components);
        } catch (std::bad_alloc const &) {
            return da_error(this->err, da_status_memory_error,
                            "Memory allocation failed.");
        }
        for (da_int j = 0; j < n_components; j++)
            sqrt_eig[j] = std::sqrt(eigenvalues[j]);
        for (da_int i = 0; i < n_samples; i++) {
            da_int row_offset = i * n_components;
#pragma omp simd
            for (da_int j = 0; j < n_components; j++)
                Z_train[row_offset + j] = eigenvectors[row_offset + j] * sqrt_eig[j];
        }
    }

    // Build cross-kernel K_cross (k x n_samples) in this->order
    // K_cross[i,j] = k(y_i, z_train_j)
    switch (this->kernel_type) {
    case pca_kernel::linear:
        status = da_kernel_functions::linear_kernel<T>(
            this->order, k, n_samples, n_components, Y, ldy, Z_train.data(), ldz,
            K_cross.data(), ldk);
        break;
    case pca_kernel::polynomial:
        status = da_kernel_functions::polynomial_kernel<T>(
            this->order, k, n_samples, n_components, Y, ldy, Z_train.data(), ldz,
            K_cross.data(), ldk, this->gamma, this->degree, this->coef0);
        break;
    case pca_kernel::rbf:
        status = da_kernel_functions::rbf_kernel<T>(
            this->order, k, n_samples, n_components, Y, ldy, Z_train.data(), ldz,
            K_cross.data(), ldk, this->gamma);
        break;
    case pca_kernel::sigmoid:
        status = da_kernel_functions::sigmoid_kernel<T>(
            this->order, k, n_samples, n_components, Y, ldy, Z_train.data(), ldz,
            K_cross.data(), ldk, this->gamma, this->coef0);
        break;
    case pca_kernel::precomputed:
        // Should not reach
        return da_error(this->err, da_status_invalid_input, //LCOV_EXCL_LINE
                        "inverse_transform is not supported with precomputed kernel.");
    default:
        return da_error(this->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unknown kernel type.");             // LCOV_EXCL_LINE
    }

    if (status != da_status_success)
        return da_error(this->err, status,
                        "Cross-kernel computation for inverse transform failed.");

    // Multiply: X_reconstructed = K_cross * dual_coef, shape (k x n_features)
    da_int ldd;
    CBLAS_ORDER cblas_order;
    if (this->order == column_major) {
        cblas_order = CblasColMajor;
        ldd = n_samples;
    } else {
        cblas_order = CblasRowMajor;
        ldd = n_features;
    }

    da_blas::cblas_gemm(cblas_order, CblasNoTrans, CblasNoTrans, k, n_features, n_samples,
                        static_cast<T>(1), K_cross.data(), ldk, dual_coef.data(), ldd,
                        static_cast<T>(0), Y_inv_transform, ldy_inv_transform);

    return da_status_success;
}

template <typename T> da_status kernel_pca<T>::serialize(serialization_buffer &buffer) {

    da_status status = da_status_success;
    auto io_dispatch = [&buffer, &status](auto &data) -> void {
        if (status != da_status_success) {
            return;
        }
        status = buffer.dispatch_buffer_io(data);
        return;
    };

    io_dispatch(this->model_trained);
    io_dispatch(this->order);
    io_dispatch(this->n_samples);
    io_dispatch(this->n_features);
    io_dispatch(this->lda);
    io_dispatch(this->copy_data);
    io_dispatch(this->row_means);
    io_dispatch(this->grand_mean);
    io_dispatch(this->eigenvalues);
    io_dispatch(this->eigenvectors);
    io_dispatch(this->dual_coef);
    io_dispatch(this->init_done);
    io_dispatch(this->inverse_fitted);

    da_int kernel_type_int = static_cast<da_int>(this->kernel_type);
    io_dispatch(kernel_type_int);
    this->kernel_type = static_cast<da_kernel_pca_types::pca_kernel>(kernel_type_int);

    io_dispatch(this->degree);
    io_dispatch(this->n_components);
    io_dispatch(this->gamma);
    io_dispatch(this->coef0);
    io_dispatch(this->alpha);
    io_dispatch(this->fit_inverse);
    io_dispatch(this->remove_zero);

    if (status != da_status_success)
        return status;

    if (buffer.get_mode() == deserialize) {
        status = buffer.deserialize_data(this->A_copy);
    } else {
        status = buffer.serialize_user_data(this->A_ptr, this->order, this->n_samples,
                                            this->n_features, this->lda);
    }

    return status;
}

template <typename T> da_status kernel_pca<T>::save_model(serialization_buffer &buffer) {
    if (!this->model_trained)
        return da_error(this->err, da_status_no_data,
                        "The kernel PCA has not been computed. Please call "
                        "da_kernel_pca_compute_s or da_kernel_pca_compute_d.");
    da_status status = basic_handle<T>::save_model(buffer);
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure serializing model.");
    return status;
}

template <typename T> da_status kernel_pca<T>::load_model(serialization_buffer &buffer) {
    da_status status = basic_handle<T>::load_model(buffer);
    if (status != da_status_success)
        return da_error_trace(this->err, status, "Failure deserializing model.");
    // Restore A_ptr to point into the deserialized A_copy (packed, no stride padding).
    // After a successful save, A_copy is guaranteed non-empty, so A_copy.data() is valid.
    A_ptr = A_copy.data();
    // serialize_user_data strips stride; update lda to tight packed value.
    this->lda =
        (this->order == da_order::column_major) ? this->n_samples : this->n_features;
    return da_status_success;
}

/* Explicit instantiations */
template class kernel_pca<double>;
template class kernel_pca<float>;

} // namespace da_kernel_pca
} // namespace ARCH