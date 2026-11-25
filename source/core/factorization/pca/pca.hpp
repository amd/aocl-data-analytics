/* ************************************************************************
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclda_pca.h"
#include "basic_handle.hpp"
#include "da_error.hpp"
#include "macros.h"
#include "model_persistence.hpp"
#include "pca_types.hpp"
#include <vector>

#ifndef PCA_HPP
#define PCA_HPP

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

    /* Correlation or covariance based PCA */
    da_int method = pca_method_cov;

    /* SVD solver */
    da_int solver = solver_gesdd;

    /* Whether we are storing U */
    bool store_U = false;

    /* Whether we are whitening */
    bool whiten = false;

    /* Number of principal components requested */
    da_int npc = 1;

    /* Degrees of freedom (bias) when computing variances, and associated divisor and sqrt */
    da_int dof = 0;
    da_int div = 0;
    T sqrt_div = 0;

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

    da_status get_result(da_result query, da_int *dim, T *result) override;

    da_status get_result(da_result query, da_int *dim, da_int *result) override;

    da_status init(da_int n, da_int p, const T *A, da_int lda);

    da_status compute();

    da_status transform(da_int m, da_int p, const T *X, da_int ldx, T *X_transform,
                        da_int ldx_transform);

    da_status inverse_transform(da_int k, da_int r, const T *X, da_int ldx,
                                T *X_inv_transform, da_int ldx_inv_transform);

    da_status serialize(da_model_persistence::serialization_buffer &buffer) override;
    da_status save_model(da_model_persistence::serialization_buffer &buffer) override;
    da_status load_model(da_model_persistence::serialization_buffer &buffer) override;
};
} // namespace da_pca
} // namespace ARCH

#endif // PCA_HPP
