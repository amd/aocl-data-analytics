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

#ifndef KERNEL_PCA_HPP
#define KERNEL_PCA_HPP

#include "aoclda.h"
#include "aoclda_pca.h"
#include "basic_handle.hpp"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "kernel_pca/kernel_pca_options.hpp"
#include "kernel_pca/kernel_pca_types.hpp"
#include "macros.h"
#include "model_persistence.hpp"
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace ARCH {

namespace da_kernel_pca {

/* Sign-normalize the columns of a column-major matrix A (rows x cols).
 * For each column, if the element with the largest absolute value is negative,
 * the entire column is negated. */
template <typename T> inline void sign_normalize_columns(T *A, da_int rows, da_int cols) {
    for (da_int j = 0; j < cols; j++) {
        da_int col_start = j * rows;
        da_int max_index = da_blas::cblas_iamax(rows, A + col_start, 1);
        T colmax = A[max_index + col_start];

        if (colmax < static_cast<T>(0.0)) {
            for (da_int i = 0; i < rows; i++)
                A[i + col_start] = -A[i + col_start];
        }
    }
}

/* Clamp near-zero eigenvalues.
 *
 * n_check: number of leading eigenvalues to error-check.
 *   0 means check all.
 *   When the user requests a specific n_components, pass that value so
 *   trailing eigenvalues that will be discarded do not trigger errors. */
template <typename T>
da_status clamp_near_zero_eigenvalues(std::vector<T> &eigenvalues, da_int n_check = 0) {
    if (eigenvalues.empty())
        return da_status_success;

    da_int n_total = static_cast<da_int>(eigenvalues.size());
    if (n_check <= 0 || n_check > n_total)
        n_check = n_total;

    /*
    This is a direct copy of scikit-learn logic
    Magic number stated in format: double (float)
    4 eigenvalue checks:
        1. Not all eigenvalues are negative.
        3. No negative eigenvalue has absolute value larger than 1e-10 (1e-6)
        3. No negative eigenvalue has absolute value larger than 1e-5 (5e-3) * \lambda_0
        If 1., 2. or 3. fail, return da_status_numerical_difficulties.
        Any negative eigenvalues that pass the tests are set to 0.
        4. Any positive eigenvalue with value smaller than 1e-12 (2e-7) * \lambda_0
            eigenvalue is set to 0.
    */
    T max_eval = eigenvalues[0];

    T pos_bound, neg_abs_bound, neg_rel_bound;
    if constexpr (std::is_same_v<T, double>) {
        pos_bound = 1e-12;
        neg_abs_bound = 1e-10;
        neg_rel_bound = 1e-5;
    } else {
        pos_bound = 2e-7;
        neg_abs_bound = 1e-6;
        neg_rel_bound = 5e-3;
    }

    pos_bound *= max_eval;
    neg_rel_bound *= max_eval;
    T strictest_neg_bound = -std::min(neg_abs_bound, neg_rel_bound);

    for (da_int idx = 0; idx < n_total; idx++) {
        T &ev = eigenvalues[idx];
        if (ev < 0) {
            if (idx < n_check && ev < strictest_neg_bound)
                return da_status_numerical_difficulties;

            ev = 0;
        } else if (ev > 0 && ev < pos_bound) {
            ev = 0;
        }
    }
    return da_status_success;
}

template <typename T> class kernel_pca : public basic_handle<T> {
  private:
    // n_samples x n_features (samples x features)
    da_int n_samples = 0;
    da_int n_features = 0;

    /* Training data.
     * When copy_data=true, A_copy owns the data and A_ptr points to it.
     * When copy_data=false, A_ptr points to the user's data directly. */
    std::vector<T> A_copy;
    const T *A_ptr = nullptr;
    da_int lda = 0;
    bool copy_data = true;

    // Kernel centering statistics -- computed in compute(), consumed in transform()
    std::vector<T> row_means;
    T grand_mean = 0;

    // Eigendecomposition output
    std::vector<T> eigenvalues;
    std::vector<T> eigenvectors; // V, shape n_samples x n_components

    // For inverse transform -- populated only when fit_inverse_transform = 1
    std::vector<T>
        dual_coef; // W: solution to (K_t + alpha*I)W = A, shape n_samples x n_features

    // State flags
    bool init_done = false;
    bool inverse_fitted = false;

    // Compute parameters
    da_kernel_pca_types::pca_kernel kernel_type = da_kernel_functions_types::linear;
    da_kernel_pca_types::solver_type solver = da_kernel_pca_types::solver_syevd;
    da_int degree = 3;
    da_int n_components = 0;
    T gamma = -1.0;
    T coef0 = 1.0;
    T alpha = 1.0;
    bool fit_inverse = false;
    bool remove_zero = false;

    // Check that kernel options have not been changed since compute()
    da_status check_options_update();

  public:
    kernel_pca(da_errors::da_error_t &err);

    ~kernel_pca() = default;

    da_status get_result(da_result query, da_int *dim, T *result) override;

    da_status get_result(da_result query, da_int *dim, da_int *result) override;

    da_status init(da_int n_samples_in, da_int n_features_in, const T *A, da_int lda);

    da_status compute();

    da_status transform(da_int m_samples, da_int m_features, const T *X, da_int ldx,
                        T *X_transform, da_int ldx_transform);

    da_status inverse_transform(da_int k, da_int nc_in, const T *Y, da_int ldy,
                                T *Y_inv_transform, da_int ldy_inv_transform);

    da_status serialize(da_model_persistence::serialization_buffer &buffer) override;
    da_status save_model(da_model_persistence::serialization_buffer &buffer) override;
    da_status load_model(da_model_persistence::serialization_buffer &buffer) override;
};

} // namespace da_kernel_pca
} // namespace ARCH

#endif // KERNEL_PCA_HPP