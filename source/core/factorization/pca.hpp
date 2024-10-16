/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc.
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
#include "aoclda_pca.h"
#include "basic_handle.hpp"
#include "da_error.hpp"
#include "macros.h"
#include "pca_types.hpp"
#include <vector>

namespace ARCH {

namespace da_pca {

using namespace da_pca_types;

template <typename T> class pca : public basic_handle<T> {
  private:
    /* n x p (samples x features) */
    da_int n = 0;
    da_int p = 0;

    /* User's data */
    const T *A;
    da_int lda;

    /*Utility pointer to column major allocated copy of user's data */
    T *A_temp = nullptr;

    /* Set true when initialization is complete */
    bool initdone = false;

    /* Set true when PCA is computed successfully */
    bool iscomputed = false;

    /* Correlation or covariance based PCA */
    da_int method = pca_method_cov;

    /* SVD solver */
    da_int solver = solver_gesdd;

    /* Sign flip flag for consistency with sklearn results */
    bool svd_flip_u_based = false;

    /* Whether we are storing U */
    bool store_U = false;

    /* Number of principal components requested */
    da_int npc = 1;

    /* Degrees of freedom (bias) when computing variances, and associated divisor */
    da_int dof = 0;
    da_int div = 0;

    /* Actual number of principal components found - on output should be the same as npc unless dgesvdx gives unexpected behaviour */
    da_int ns = 0;

    /* Will we perform a QR decomposition prior to the SVD? */
    bool qr = false;

    /* Arrays used by the SVD, and to store results */
    std::vector<T> scores;
    /* U*Sigma */
    std::vector<T> variance;
    /* Sigma**2 / n-1 */
    std::vector<T> column_means, column_sdevs, column_sdevs_nonzero;
    /* Store standardization data */
    T total_variance = 0.0;
    /* Sum((MeanCentered A [][])**2) */
    da_int n_components = 0, ldvt = 0, u_size = 0, ldu = 0;
    std::vector<T> u, sigma, vt, work, A_copy;
    std::vector<da_int> iwork;

  public:
    pca(da_errors::da_error_t &err);

    ~pca();

    da_status get_result(da_result query, da_int *dim, T *result);

    da_status get_result(da_result query, da_int *dim, da_int *result);

    da_status init(da_int n, da_int p, const T *A, da_int lda);

    da_status compute();

    da_status transform(da_int m, da_int p, const T *X, da_int ldx, T *X_transform,
                        da_int ldx_transform);

    da_status inverse_transform(da_int k, da_int r, const T *X, da_int ldx,
                                T *X_inv_transform, da_int ldx_inv_transform);
};
} // namespace da_pca
} // namespace ARCH