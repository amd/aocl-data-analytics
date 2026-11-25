/* ************************************************************************
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
 * ************************************************************************ */

#ifndef NMF_HPP
#define NMF_HPP

#include "aoclda.h"
#include "aoclda_nmf.h"
#include "basic_handle.hpp"
#include "da_error.hpp"
#include "macros.h"
#include "model_persistence.hpp"
#include "nmf/nmf_types.hpp"
#include <vector>

namespace ARCH {

namespace da_nmf {

template <typename T> class nmf : public basic_handle<T> {
  private:
    /* n x p (samples x features) */
    da_int n_samples = 0;
    da_int n_features = 0;
    /* number of components */
    da_int n_components = 0;

    /* User's data; always the active internal matrix pointer. */
    const T *A = nullptr;
    da_int lda = 0;

    /* Utility pointer to column-major allocated copy of user's data (for row-major conversion) */
    T *A_temp = nullptr;

    /* Factorization outputs, written by compute() and surfaced via get_result.
     * W is n x n_components (column-major, lda = n).
     * H is n_components x p (column-major, lda = n_components). */
    std::vector<T> W;
    std::vector<T> H;

    /* For init='custom', nmf_set_data writes directly into W and H to avoid
     * an extra staging copy for large matrices. k_init tracks the uploaded
     * component count for option consistency checks in compute().
     * custom_init_set is reset by init() and consumed by compute(). */
    da_int k_init = 0;
    bool custom_w_set = false;
    bool custom_h_set = false;
    bool custom_init_set = false;

    /* Number of MU iterations actually performed. */
    da_int n_iter = 0;
    /* Frobenius reconstruction error ||A - W H||_F. */
    T reconstruction_err = T{0};

    /* Itializer variants:
     * nndsvd:   leaves degenerate entries at zero.
     * nndsvda:  replaces near-zero entries with mean(A).
     * nndsvdar: replaces near-zero entries with U(0, mean(A)*0.01) random values. */
    da_status do_nndsvd_init(da_nmf_types::nmf_init variant);

    /* Random initializer: fills W and H with |N(0,1)| samples scaled by
     * sqrt(mean(A) / n_components). */
    da_status do_random_init();

    /* Lee-Seung multiplicative-update iterations on the Frobenius loss.
     * Refines the W, H already filled by an initializer. Updates n_iter and
     * reconstruction_err on success. */
    da_status do_mu_iterations(da_int max_iter, T tol);

    /* Blocked computation of sqrt(||X - W H||_F).
     * W is (n_rows x n_components) column-major with leading dimension n_rows.
     * X is (n_rows x n_features) column-major with leading dimension ldx.
     * H is always this->H (n_components x n_features), fixed. */
    da_status fro_residual(da_int n_rows, const T *W, const T *X, da_int ldx,
                           double &result);

    da_status check_nonneg_matrix(const T *mat, da_int lda, da_int n_rows, da_int n_cols);

  public:
    nmf(da_errors::da_error_t &err);

    ~nmf();

    da_status get_result(da_result query, da_int *dim, T *result) override;

    da_status get_result(da_result query, da_int *dim, da_int *result) override;

    da_status init(da_int n, da_int p, const T *A, da_int lda);

    /* Load user-supplied factors for init='custom'. W and H can be uploaded
     * together or in two calls (one factor per call). Passing one null pointer
     * is accepted and returns a warning, but compute() requires both factors to
     * have been uploaded with the same k. Must be called after init().
     * Arrays are interpreted using the storage-order option on the handle and
     * any uploaded factor must be nonnegative. */
    da_status nmf_set_data(da_int n_samples, da_int n_components, const T *W_in,
                           da_int ldw, da_int n_features, const T *H_in, da_int ldh);

    /* Compute an NMF factorization A ~= W H using the configured options.
     * Requires init() to have been called, and for init='custom', custom factors
     * must be uploaded first. On success, updates W/H, n_iter and
     * reconstruction_err, and marks the model as trained. */
    da_status compute();

    /* Project a new matrix X (m x p) onto the learned component space.
     * Solves for nonnegative coefficients W (m x n_components)
     * while keeping the fitted components H fixed. */
    da_status transform(da_int m_samples, da_int m_features, const T *X, da_int ldx,
                        T *Xt, da_int ldxt);

    /* Map coefficient-space samples Y (k_samples x n_components) back to feature
     * space using the fitted components H, i.e. computes Y_inv_transform = Y H
     * with output shape (k_samples x n_features). */
    da_status inverse_transform(da_int k_samples, da_int k_components, const T *Xt,
                                da_int ldxt, T *X, da_int ldx);

    da_status serialize(da_model_persistence::serialization_buffer &buffer) override;
    da_status save_model(da_model_persistence::serialization_buffer &buffer) override;
    da_status load_model(da_model_persistence::serialization_buffer &buffer) override;
};

} // namespace da_nmf
} // namespace ARCH

#endif // NMF_HPP
