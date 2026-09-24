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

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "../datests_cblas.hh"
#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include "nmf_test_data.hpp"

template <typename T> class NMFTest : public testing::Test {};

using NMFTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(NMFTest, NMFTypes);

/* Compute ||A - W*H||_F for col-major or row-major layouts.
 * W is n*k, H is k*p, A is n*p with leading dimension lda. */
template <typename T>
T explicit_residual(const std::vector<T> &A, da_int n, da_int p, const std::vector<T> &W,
                    const std::vector<T> &H, da_int k, da_int lda = -1,
                    bool col_major = true) {
    std::vector<T> WH(static_cast<std::size_t>(n * p));
    T sq = T(0);
    if (col_major) {
        if (lda < n)
            lda = n;
        datest_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, p, k, T(1),
                                W.data(), n, H.data(), k, T(0), WH.data(), n);
        for (da_int j = 0; j < p; ++j)
            for (da_int i = 0; i < n; ++i) {
                const T d = A[i + j * lda] - WH[i + j * n];
                sq += d * d;
            }
    } else {
        if (lda < p)
            lda = p;
        datest_blas::cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, p, k, T(1),
                                W.data(), k, H.data(), p, T(0), WH.data(), p);
        for (da_int i = 0; i < n; ++i)
            for (da_int j = 0; j < p; ++j) {
                const T d = A[i * lda + j] - WH[i * p + j];
                sq += d * d;
            }
    }
    return std::sqrt(sq);
}

/* Passing a matrix that contains at least one negative element must be
 * rejected with da_status_invalid_input at set_data time (inside init()).
 * Col-major sub-cases (default order):
 *   - single negative entry in the interior
 *   - negative entry in the first element
 *   - all entries negative
 *   - lda > n (col-major): padding filled with -99, all logical entries
 *       positive, must be accepted (padding must not be read)
 *   - lda > p (row-major): same idea
 * Row-major sub-cases:
 *   - single interior negative entry
 *   - negative entry at [0,0]
 *   - all entries negative                                             */
TYPED_TEST(NMFTest, NegativeInputMatrixARejected) {
    using T = TypeParam;

    const da_int n = 4, p = 3, k = 2;

    /* All-positive baseline. */
    std::vector<T> A_pos = {T(1), T(2), T(3), T(4),  T(5),  T(6),
                            T(7), T(8), T(9), T(10), T(11), T(12)};

    /* Negative entry at position [0,0]. */
    {
        std::vector<T> A_neg = A_pos;
        A_neg[0] = T(-1);

        da_handle handle = nullptr;
        ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);
        ASSERT_EQ(da_options_set_int(handle, "n_components", k), da_status_success);
        EXPECT_EQ(da_nmf_init(handle, n, p, A_neg.data(), n), da_status_invalid_input);
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
    }

    /* Single interior negative entry. */
    {
        std::vector<T> A_neg = A_pos;
        A_neg[5] = T(-0.5); /* interior element */

        da_handle handle = nullptr;
        ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);
        ASSERT_EQ(da_options_set_int(handle, "n_components", k), da_status_success);
        EXPECT_EQ(da_nmf_init(handle, n, p, A_neg.data(), n), da_status_invalid_input);
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
    }

    /* All entries negative. */
    {
        std::vector<T> A_neg(n * p, T(-1));

        da_handle handle = nullptr;
        ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);
        ASSERT_EQ(da_options_set_int(handle, "n_components", k), da_status_success);
        EXPECT_EQ(da_nmf_init(handle, n, p, A_neg.data(), n), da_status_invalid_input);
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
    }

    /* Column-major, lda=6 (2 padding rows per column filled with -99).
     * All logical elements are positive, so set_data must succeed. */
    {
        const da_int lda = 6;
        std::vector<T> A(lda * p, T(-99));
        for (da_int j = 0; j < p; ++j)
            for (da_int i = 0; i < n; ++i)
                A[j * lda + i] = static_cast<T>(1 + i + j * n);

        da_handle handle = nullptr;
        ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);
        ASSERT_EQ(da_options_set_int(handle, "n_components", k), da_status_success);
        ASSERT_EQ(da_options_set_string(handle, "storage order", "column-major"),
                  da_status_success);
        EXPECT_EQ(da_nmf_init(handle, n, p, A.data(), lda), da_status_success);
        EXPECT_EQ(da_nmf_compute<T>(handle), da_status_success);
        da_handle_destroy(&handle);
    }

    /* Row-major, lda=5 (2 padding columns per row filled with -99).
     * All logical elements are positive, so set_data must succeed. */
    {
        const da_int lda = 5;
        std::vector<T> A(n * lda, T(-99));
        for (da_int i = 0; i < n; ++i)
            for (da_int j = 0; j < p; ++j)
                A[i * lda + j] = static_cast<T>(1 + j + i * p);

        da_handle handle = nullptr;
        ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);
        ASSERT_EQ(da_options_set_int(handle, "n_components", k), da_status_success);
        ASSERT_EQ(da_options_set_string(handle, "storage order", "row-major"),
                  da_status_success);
        EXPECT_EQ(da_nmf_init(handle, n, p, A.data(), lda), da_status_success);
        EXPECT_EQ(da_nmf_compute<T>(handle), da_status_success);
        da_handle_destroy(&handle);
    }

    /* Row-major, single interior negative entry at [1,1]. */
    {
        const da_int lda = p;
        std::vector<T> A_rm(static_cast<std::size_t>(n) * static_cast<std::size_t>(lda));
        for (da_int i = 0; i < n; ++i)
            for (da_int j = 0; j < p; ++j)
                A_rm[i * lda + j] = static_cast<T>(1 + j + i * p);
        A_rm[1 * lda + 1] = T(-0.5);

        da_handle handle = nullptr;
        ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);
        ASSERT_EQ(da_options_set_int(handle, "n_components", k), da_status_success);
        ASSERT_EQ(da_options_set_string(handle, "storage order", "row-major"),
                  da_status_success);
        EXPECT_EQ(da_nmf_init(handle, n, p, A_rm.data(), lda), da_status_invalid_input);
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
    }

    /* Row-major, negative entry at [0,0]. */
    {
        const da_int lda = p;
        std::vector<T> A_rm(static_cast<std::size_t>(n) * static_cast<std::size_t>(lda));
        for (da_int i = 0; i < n; ++i)
            for (da_int j = 0; j < p; ++j)
                A_rm[i * lda + j] = static_cast<T>(1 + j + i * p);
        A_rm[0] = T(-1);

        da_handle handle = nullptr;
        ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);
        ASSERT_EQ(da_options_set_int(handle, "n_components", k), da_status_success);
        ASSERT_EQ(da_options_set_string(handle, "storage order", "row-major"),
                  da_status_success);
        EXPECT_EQ(da_nmf_init(handle, n, p, A_rm.data(), lda), da_status_invalid_input);
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
    }

    /* Row-major, all entries negative. */
    {
        const da_int lda = p;
        std::vector<T> A_rm(static_cast<std::size_t>(n) * static_cast<std::size_t>(lda),
                            T(-1));

        da_handle handle = nullptr;
        ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);
        ASSERT_EQ(da_options_set_int(handle, "n_components", k), da_status_success);
        ASSERT_EQ(da_options_set_string(handle, "storage order", "row-major"),
                  da_status_success);
        EXPECT_EQ(da_nmf_init(handle, n, p, A_rm.data(), lda), da_status_invalid_input);
        da_handle_print_error_message(handle);
        da_handle_destroy(&handle);
    }
}

/* transform() must reject X with negative entries and must not read outside
 * the logical bounds when ldx > m_samples (col-major) or ldx > m_features (row-major).
 * Col-major sub-cases:
 *   - interior negative at [1,1]
 *   - negative at [0,0]
 *   - all entries negative
 *   - ldx > m: padding rows filled with -99, logical entries positive
 * Row-major sub-cases:
 *   - interior negative at [1,1]
 *   - negative at [0,0]
 *   - all entries negative
 *   - ldx > p: padding columns filled with -99, logical entries positive */
TYPED_TEST(NMFTest, TransformRejectsNegativeInput) {
    using T = TypeParam;

    da_int n = 4, p = 3;
    std::vector<T> A_cm = {T(1), T(2), T(3), T(4),  T(5),  T(6),
                           T(7), T(8), T(9), T(10), T(11), T(12)};
    constexpr da_int k = 2;
    const da_int m = 2; /* number of new samples for transform */

    /* ---- col-major model ---- */
    da_handle handle_cm = nullptr;
    ASSERT_EQ(da_handle_init<T>(&handle_cm, da_handle_nmf), da_status_success);
    ASSERT_EQ(da_options_set_int(handle_cm, "n_components", k), da_status_success);
    ASSERT_EQ(da_nmf_init(handle_cm, n, p, A_cm.data(), n), da_status_success);
    ASSERT_EQ(da_nmf_compute<T>(handle_cm), da_status_success);

    std::vector<T> W_out_cm(static_cast<std::size_t>(m) * static_cast<std::size_t>(k),
                            T(0));

    /* col-major, interior negative at [1,1] (index = col*m + row = 1*2+1 = 3). */
    {
        std::vector<T> X(static_cast<std::size_t>(m) * static_cast<std::size_t>(p), T(1));
        X[1 * m + 1] = T(-0.5);
        EXPECT_EQ(da_nmf_transform(handle_cm, m, p, X.data(), m, W_out_cm.data(), m),
                  da_status_invalid_input);
        da_handle_print_error_message(handle_cm);
    }

    /* col-major, negative at [0,0]. */
    {
        std::vector<T> X(static_cast<std::size_t>(m) * static_cast<std::size_t>(p), T(1));
        X[0] = T(-1);
        EXPECT_EQ(da_nmf_transform(handle_cm, m, p, X.data(), m, W_out_cm.data(), m),
                  da_status_invalid_input);
        da_handle_print_error_message(handle_cm);
    }

    /* col-major, all entries negative. */
    {
        std::vector<T> X(static_cast<std::size_t>(m) * static_cast<std::size_t>(p),
                         T(-1));
        EXPECT_EQ(da_nmf_transform(handle_cm, m, p, X.data(), m, W_out_cm.data(), m),
                  da_status_invalid_input);
        da_handle_print_error_message(handle_cm);
    }

    /* col-major, ldx > m: padding rows filled with -99, logical entries positive
     * - transform must succeed (padding must not be read). */
    {
        const da_int ldx = m + 2;
        std::vector<T> X(static_cast<std::size_t>(ldx) * static_cast<std::size_t>(p),
                         T(-99));
        for (da_int j = 0; j < p; ++j)
            for (da_int i = 0; i < m; ++i)
                X[j * ldx + i] = T(1);
        std::vector<T> W_out_d(static_cast<std::size_t>(m) * static_cast<std::size_t>(k),
                               T(0));
        EXPECT_EQ(da_nmf_transform(handle_cm, m, p, X.data(), ldx, W_out_d.data(), m),
                  da_status_success);
    }

    da_handle_destroy(&handle_cm);

    /* ---- row-major model ---- */
    /* Build the same n*p matrix in row-major order (lda = p). */
    da_handle handle_rm = nullptr;
    ASSERT_EQ(da_handle_init<T>(&handle_rm, da_handle_nmf), da_status_success);
    ASSERT_EQ(da_options_set_int(handle_rm, "n_components", k), da_status_success);
    ASSERT_EQ(da_options_set_string(handle_rm, "storage order", "row-major"),
              da_status_success);
    {
        std::vector<T> A_rm(static_cast<std::size_t>(n) * static_cast<std::size_t>(p));
        for (da_int i = 0; i < n; ++i)
            for (da_int j = 0; j < p; ++j)
                A_rm[i * p + j] = A_cm[j * n + i];
        ASSERT_EQ(da_nmf_init(handle_rm, n, p, A_rm.data(), p), da_status_success);
    }
    ASSERT_EQ(da_nmf_compute<T>(handle_rm), da_status_success);

    /* Row-major W_out: m rows * k cols, ldw = k. */
    std::vector<T> W_out_rm(static_cast<std::size_t>(m) * static_cast<std::size_t>(k),
                            T(0));

    /* row-major, interior negative at [1,1] (index = row*p + col = 1*p+1). */
    {
        std::vector<T> X(static_cast<std::size_t>(m) * static_cast<std::size_t>(p), T(1));
        X[1 * p + 1] = T(-0.5);
        EXPECT_EQ(da_nmf_transform(handle_rm, m, p, X.data(), p, W_out_rm.data(), k),
                  da_status_invalid_input);
        da_handle_print_error_message(handle_rm);
    }

    /* row-major, negative at [0,0]. */
    {
        std::vector<T> X(static_cast<std::size_t>(m) * static_cast<std::size_t>(p), T(1));
        X[0] = T(-1);
        EXPECT_EQ(da_nmf_transform(handle_rm, m, p, X.data(), p, W_out_rm.data(), k),
                  da_status_invalid_input);
        da_handle_print_error_message(handle_rm);
    }

    /* row-major, all entries negative. */
    {
        std::vector<T> X(static_cast<std::size_t>(m) * static_cast<std::size_t>(p),
                         T(-1));
        EXPECT_EQ(da_nmf_transform(handle_rm, m, p, X.data(), p, W_out_rm.data(), k),
                  da_status_invalid_input);
        da_handle_print_error_message(handle_rm);
    }

    /* row-major, ldx > p: padding columns filled with -99, logical entries positive
     * transform must succeed (padding must not be read). */
    {
        const da_int ldx = p + 2;
        std::vector<T> X(static_cast<std::size_t>(m) * static_cast<std::size_t>(ldx),
                         T(-99));
        for (da_int i = 0; i < m; ++i)
            for (da_int j = 0; j < p; ++j)
                X[i * ldx + j] = T(1);
        std::vector<T> W_out_h(static_cast<std::size_t>(m) * static_cast<std::size_t>(k),
                               T(0));
        EXPECT_EQ(da_nmf_transform(handle_rm, m, p, X.data(), ldx, W_out_h.data(), k),
                  da_status_success);
    }

    da_handle_destroy(&handle_rm);
}

/* Fit with padded A (lda > n), padded W0/H0 (ldw > n, ldh > k), and
 * transform/inverse_transform with padded input (ldx > m) and padded outputs
 * (ldxt > m, ldy > m). Verifies padding is never read or written. */
TYPED_TEST(NMFTest, MUWithCustomInitPaddedLda) {
    using T = TypeParam;
    da_int n = 4, p = 3;
    /* A: 4x3 col-major, lda=7; -99 in padding rows 4, 5 and 6. */
    const da_int lda = n + 3;
    std::vector<T> A(lda * p, T(-99));
    for (da_int j = 0; j < p; ++j)
        for (da_int i = 0; i < n; ++i)
            A[j * lda + i] = static_cast<T>(1 + i + j * n);
    const da_int k = 2;
    const da_int m = 2;

    /* W0: 4x2 col-major, ldw=6; -99 in padding rows 4 and 5. */
    const da_int ldw0 = n + 2;
    std::vector<T> W0 = {T(0.5), T(1.0), T(1.5), T(2.0), T(-99), T(-99),
                         T(1.0), T(0.5), T(1.0), T(0.5), T(-99), T(-99)};

    /* H0: 2x3 col-major, ldh=4; -99 in padding rows 2 and 3. */
    const da_int ldh0 = k + 2;
    std::vector<T> H0 = {T(1.0), T(0.5), T(-99), T(-99), T(1.0), T(0.5),
                         T(-99), T(-99), T(1.0), T(1.5), T(-99), T(-99)};

    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);
    ASSERT_EQ(da_options_set_int(handle, "n_components", k), da_status_success);
    ASSERT_EQ(da_options_set_string(handle, "init", "custom"), da_status_success);
    ASSERT_EQ(da_options_set_int(handle, "max_iter", 500), da_status_success);
    ASSERT_EQ(da_options_set(handle, "tol", T(1e-6)), da_status_success);
    ASSERT_EQ(da_nmf_init(handle, n, p, A.data(), lda), da_status_success);
    ASSERT_EQ(da_nmf_set_data<T>(handle, n, k, W0.data(), ldw0, p, H0.data(), ldh0),
              da_status_success);
    ASSERT_EQ(da_nmf_compute<T>(handle), da_status_success);

    std::vector<T> W(n * k), H(k * p);
    da_int wd = n * k, hd = k * p;
    ASSERT_EQ(da_handle_get_result<T>(handle, da_nmf_w, &wd, W.data()),
              da_status_success);
    ASSERT_EQ(da_handle_get_result<T>(handle, da_nmf_h, &hd, H.data()),
              da_status_success);
    for (T v : W)
        EXPECT_GE(v, T(0));
    for (T v : H)
        EXPECT_GE(v, T(0));

    /* X_test: 2x3 col-major, ldx=4 with -99 in padding rows 2 and 3. */
    const da_int ldx = m + 2;
    std::vector<T> X_test = {T(2),   T(3),   T(-99), T(-99), T(4),   T(5),
                             T(-99), T(-99), T(6),   T(7),   T(-99), T(-99)};

    /* Transform: ldxt=m+2; padding rows must not be written. */
    const da_int ldxt = m + 2;
    std::vector<T> Xt(static_cast<std::size_t>(ldxt * k), T(-99));
    ASSERT_EQ(da_nmf_transform(handle, m, p, X_test.data(), ldx, Xt.data(), ldxt),
              da_status_success);
    for (da_int j = 0; j < k; ++j)
        for (da_int i = m; i < ldxt; ++i)
            EXPECT_EQ(Xt[j * ldxt + i], T(-99));
    for (da_int j = 0; j < k; ++j)
        for (da_int i = 0; i < m; ++i)
            EXPECT_GE(Xt[j * ldxt + i], T(0));

    /* inverse_transform: ldy=m+2; padding rows must not be written. */
    const da_int ldy = m + 2;
    std::vector<T> Arec(static_cast<std::size_t>(ldy * p), T(-99));
    ASSERT_EQ(da_nmf_inverse_transform(handle, m, k, Xt.data(), ldxt, Arec.data(), ldy),
              da_status_success);
    for (da_int j = 0; j < p; ++j)
        for (da_int i = m; i < ldy; ++i)
            EXPECT_EQ(Arec[j * ldy + i], T(-99));

    /* Logical Arec must equal Xt * H to machine precision. */
    std::vector<T> Arec_ref(static_cast<std::size_t>(m * p));
    datest_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, p, k, T(1),
                            Xt.data(), ldxt, H.data(), k, T(0), Arec_ref.data(), m);
    T A_fro = T(0);
    for (da_int j = 0; j < p; ++j)
        for (da_int i = 0; i < n; ++i)
            A_fro += A[j * lda + i] * A[j * lda + i];
    A_fro = std::sqrt(A_fro);
    const T tol = std::sqrt(std::numeric_limits<T>::epsilon());
    for (da_int j = 0; j < p; ++j)
        for (da_int i = 0; i < m; ++i)
            EXPECT_NEAR(Arec[j * ldy + i], Arec_ref[j * m + i], tol);

    da_handle_destroy(&handle);
}
/* Test invalid inputs for custom init requirements. */
TYPED_TEST(NMFTest, CustomInitUploadRequirements) {
    using T = TypeParam;
    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);

    da_int n = 4, p = 3;
    std::vector<T> A = {T(1), T(2), T(3), T(4),  T(5),  T(6),
                        T(7), T(8), T(9), T(10), T(11), T(12)};
    const da_int k = 2;

    ASSERT_EQ(da_options_set_int(handle, "n_components", k), da_status_success);
    ASSERT_EQ(da_options_set_string(handle, "init", "custom"), da_status_success);
    ASSERT_EQ(da_nmf_init(handle, n, p, A.data(), n), da_status_success);

    /* No upload at all: compute must refuse. */
    EXPECT_EQ(da_nmf_compute<T>(handle), da_status_no_data);

    std::vector<T> W0(n * k, T(1));
    std::vector<T> H0(k * p, T(1));

    /* W only: partial upload returns a warning, compute still refuses. */
    EXPECT_EQ(da_nmf_set_data<T>(handle, n, k, W0.data(), n, p, (const T *)nullptr, k),
              da_status_missing_block);
    EXPECT_EQ(da_nmf_compute<T>(handle), da_status_no_data);

    /* Second W upload without H: H is still missing, compute still refuses. */
    EXPECT_EQ(da_nmf_set_data<T>(handle, n, k, W0.data(), n, p, (const T *)nullptr, k),
              da_status_missing_block);
    EXPECT_EQ(da_nmf_compute<T>(handle), da_status_no_data);

    /* H with wrong k: set_data must reject the mismatch against the already-uploaded W. */
    const da_int k_bad = 3;
    std::vector<T> H_bad(k_bad * p, T(1));
    EXPECT_EQ(da_nmf_set_data<T>(handle, n, k_bad, (const T *)nullptr, n, p, H_bad.data(),
                                 k_bad),
              da_status_invalid_input);

    /* H with correct k: completes the pair, compute succeeds. */
    EXPECT_EQ(da_nmf_set_data<T>(handle, n, k, (const T *)nullptr, n, p, H0.data(), k),
              da_status_success);
    EXPECT_EQ(da_nmf_compute<T>(handle), da_status_success);

    da_handle_destroy(&handle);
}
/* Check that MU correctly returns early. */
TYPED_TEST(NMFTest, MUStopsAtMaxIter) {
    using T = TypeParam;
    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);

    da_int n = 4, p = 3;
    std::vector<T> A = {T(1), T(2), T(3), T(4),  T(5),  T(6),
                        T(7), T(8), T(9), T(10), T(11), T(12)};
    ASSERT_EQ(da_options_set_int(handle, "n_components", 2), da_status_success);
    ASSERT_EQ(da_options_set_string(handle, "init", "nndsvda"), da_status_success);
    ASSERT_EQ(da_options_set_int(handle, "max_iter", 3), da_status_success);
    ASSERT_EQ(da_options_set(handle, "tol", T(0)), da_status_success);
    ASSERT_EQ(da_nmf_init(handle, n, p, A.data(), n), da_status_success);
    ASSERT_EQ(da_nmf_compute<T>(handle), da_status_success);

    da_int it = 0, d = 1;
    ASSERT_EQ(da_handle_get_result<da_int>(handle, da_nmf_n_iter, &d, &it),
              da_status_success);
    EXPECT_EQ(it, 3);

    da_handle_destroy(&handle);
}
/* ============================================================
 * Error-exit / wrong-API-usage tests.
 * ============================================================ */

/* Helper: print a descriptive failure message. */
static std::string NMFErrMsg(const std::string &what) {
    return "NMF error-exit test failed for: " + what;
}

/* ------------------------------------------------------------------
 * Wrong call order: functions called before their prerequisites.
 * ------------------------------------------------------------------ */
TYPED_TEST(NMFTest, ErrorExitWrongCallOrder) {
    using T = TypeParam;
    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);

    std::vector<T> dummy(12, T(1));
    std::vector<T> out(8, T(0));
    da_int dim = 8;

    /* --- compute before init --- */
    EXPECT_EQ(da_nmf_compute<T>(handle), da_status_no_data)
        << NMFErrMsg("compute before init");

    /* --- transform before init and compute --- */
    EXPECT_EQ(da_nmf_transform(handle, 4, 3, dummy.data(), 4, out.data(), 4),
              da_status_no_data)
        << NMFErrMsg("transform before compute");

    /* --- inverse_transform before init and compute --- */
    EXPECT_EQ(da_nmf_inverse_transform(handle, 4, 2, dummy.data(), 4, out.data(), 4),
              da_status_no_data)
        << NMFErrMsg("inverse_transform before compute");

    /* --- get_result (fp) before compute --- */
    T fp_result = T(0);
    dim = 1;
    EXPECT_EQ(
        da_handle_get_result<T>(handle, da_nmf_reconstruction_err, &dim, &fp_result),
        da_status_no_data)
        << NMFErrMsg("get_result reconstruction_err before compute");
    dim = 8;
    EXPECT_EQ(da_handle_get_result<T>(handle, da_nmf_w, &dim, out.data()),
              da_status_no_data)
        << NMFErrMsg("get_result W before compute");
    dim = 6;
    EXPECT_EQ(da_handle_get_result<T>(handle, da_nmf_h, &dim, out.data()),
              da_status_no_data)
        << NMFErrMsg("get_result H before compute");

    /* --- get_result (int) before compute --- */
    da_int int_result = 0;
    dim = 1;
    EXPECT_EQ(da_handle_get_result_int(handle, da_nmf_n_iter, &dim, &int_result),
              da_status_no_data)
        << NMFErrMsg("get_result n_iter before compute");
    dim = 1;
    EXPECT_EQ(da_handle_get_result_int(handle, da_nmf_n_components, &dim, &int_result),
              da_status_no_data)
        << NMFErrMsg("get_result n_components before compute");

    /* --- da_rinfo before compute --- */
    dim = 2;
    std::vector<T> rinfo(2, T(0));
    EXPECT_EQ(da_handle_get_result<T>(handle, da_rinfo, &dim, rinfo.data()),
              da_status_no_data)
        << NMFErrMsg("get_result da_rinfo before compute");

    /* --- set_data before init --- */
    EXPECT_EQ(da_nmf_set_data<T>(handle, 4, 2, dummy.data(), 4, 3, dummy.data(), 2),
              da_status_no_data)
        << NMFErrMsg("set_data before init");

    /* Now init, then check transform/inverse_transform still blocked before compute. */
    ASSERT_EQ(da_options_set_int(handle, "n_components", 2), da_status_success);
    ASSERT_EQ(da_nmf_init(handle, 4, 3, dummy.data(), 4), da_status_success);

    EXPECT_EQ(da_nmf_transform(handle, 4, 3, dummy.data(), 4, out.data(), 4),
              da_status_no_data)
        << NMFErrMsg("transform after init but before compute");
    EXPECT_EQ(da_nmf_inverse_transform(handle, 4, 2, dummy.data(), 4, out.data(), 4),
              da_status_no_data)
        << NMFErrMsg("inverse_transform after init but before compute");

    da_handle_destroy(&handle);
}

/* ------------------------------------------------------------------
 * null handle, wrong handle type, wrong precision.
 * ------------------------------------------------------------------ */
TYPED_TEST(NMFTest, BadHandleTests) {
    using T = TypeParam;
    std::vector<T> dummy(12, T(1));
    std::vector<T> out(8, T(0));

    /* null handle */
    EXPECT_EQ(da_nmf_init<T>(nullptr, 4, 3, dummy.data(), 4),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_nmf_compute<T>(nullptr), da_status_handle_not_initialized);
    EXPECT_EQ(da_nmf_set_data<T>(nullptr, 4, 2, dummy.data(), 4, 3, dummy.data(), 2),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_nmf_transform<T>(nullptr, 4, 3, dummy.data(), 4, out.data(), 4),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_nmf_inverse_transform<T>(nullptr, 4, 2, dummy.data(), 4, out.data(), 4),
              da_status_handle_not_initialized);

    /* wrong handle type (linmod handle passed to NMF functions) */
    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<T>(&handle, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_nmf_init<T>(handle, 4, 3, dummy.data(), 4),
              da_status_invalid_handle_type);
    EXPECT_EQ(da_nmf_compute<T>(handle), da_status_invalid_handle_type);
    EXPECT_EQ(da_nmf_set_data<T>(handle, 4, 2, dummy.data(), 4, 3, dummy.data(), 2),
              da_status_invalid_handle_type);
    EXPECT_EQ(da_nmf_transform<T>(handle, 4, 3, dummy.data(), 4, out.data(), 4),
              da_status_invalid_handle_type);
    EXPECT_EQ(da_nmf_inverse_transform<T>(handle, 4, 2, dummy.data(), 4, out.data(), 4),
              da_status_invalid_handle_type);
    da_handle_destroy(&handle);

    /* wrong precision: each NMF function rejects a handle of the opposite type */
    da_handle handle_d = nullptr;
    da_handle handle_s = nullptr;
    ASSERT_EQ(da_handle_init_d(&handle_d, da_handle_nmf), da_status_success);
    ASSERT_EQ(da_handle_init_s(&handle_s, da_handle_nmf), da_status_success);

    double Ad = 1.0;
    float As = 1.0f;
    double out_d = 0.0;
    float out_s = 0.0f;

    EXPECT_EQ(da_nmf_init_d(handle_s, 1, 1, &Ad, 1), da_status_wrong_type);
    EXPECT_EQ(da_nmf_init_s(handle_d, 1, 1, &As, 1), da_status_wrong_type);
    EXPECT_EQ(da_nmf_set_data_d(handle_s, 1, 1, &Ad, 1, 1, &Ad, 1), da_status_wrong_type);
    EXPECT_EQ(da_nmf_set_data_s(handle_d, 1, 1, &As, 1, 1, &As, 1), da_status_wrong_type);
    EXPECT_EQ(da_nmf_compute_d(handle_s), da_status_wrong_type);
    EXPECT_EQ(da_nmf_compute_s(handle_d), da_status_wrong_type);
    EXPECT_EQ(da_nmf_transform_d(handle_s, 1, 1, &Ad, 1, &out_d, 1),
              da_status_wrong_type);
    EXPECT_EQ(da_nmf_transform_s(handle_d, 1, 1, &As, 1, &out_s, 1),
              da_status_wrong_type);
    EXPECT_EQ(da_nmf_inverse_transform_d(handle_s, 1, 1, &Ad, 1, &out_d, 1),
              da_status_wrong_type);
    EXPECT_EQ(da_nmf_inverse_transform_s(handle_d, 1, 1, &As, 1, &out_s, 1),
              da_status_wrong_type);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}

/* ------------------------------------------------------------------
 * da_nmf_init: invalid dimension and pointer arguments.
 * ------------------------------------------------------------------ */
TYPED_TEST(NMFTest, ErrorExitInitInvalidArgs) {
    using T = TypeParam;
    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);

    std::vector<T> A(12, T(1));

    /* null data pointer */
    EXPECT_EQ(da_nmf_init(handle, 4, 3, (const T *)nullptr, 4), da_status_invalid_pointer)
        << NMFErrMsg("init null A");

    /* n_samples < 1 */
    EXPECT_EQ(da_nmf_init(handle, 0, 3, A.data(), 4), da_status_invalid_array_dimension)
        << NMFErrMsg("init n_samples = 0");
    EXPECT_EQ(da_nmf_init(handle, -1, 3, A.data(), 4), da_status_invalid_array_dimension)
        << NMFErrMsg("init n_samples = -1");

    /* n_features < 1 */
    EXPECT_EQ(da_nmf_init(handle, 4, 0, A.data(), 4), da_status_invalid_array_dimension)
        << NMFErrMsg("init n_features = 0");
    EXPECT_EQ(da_nmf_init(handle, 4, -1, A.data(), 4), da_status_invalid_array_dimension)
        << NMFErrMsg("init n_features = -1");

    /* lda too small (col-major: lda must be >= n_samples) */
    EXPECT_EQ(da_nmf_init(handle, 4, 3, A.data(), 3), da_status_invalid_leading_dimension)
        << NMFErrMsg("init lda < n_samples (col-major)");

    /* lda too small (row-major: lda must be >= n_features) */
    ASSERT_EQ(da_options_set_string(handle, "storage order", "row-major"),
              da_status_success);
    EXPECT_EQ(da_nmf_init(handle, 4, 3, A.data(), 2), da_status_invalid_leading_dimension)
        << NMFErrMsg("init lda < n_features (row-major)");

    da_handle_destroy(&handle);
}

/* ------------------------------------------------------------------
 * da_nmf_compute: invalid n_components option values.
 * ------------------------------------------------------------------ */
TYPED_TEST(NMFTest, ErrorExitComputeInvalidNComponents) {
    using T = TypeParam;
    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);

    std::vector<T> A(12, T(1)); /* 4x3 */
    ASSERT_EQ(da_nmf_init(handle, 4, 3, A.data(), 4), da_status_success);

    /* n_components = 0 is now accepted: it means "all components" (min(n_samples, n_features)) */
    ASSERT_EQ(da_options_set_int(handle, "n_components", 0), da_status_success);
    EXPECT_EQ(da_nmf_compute<T>(handle), da_status_success)
        << NMFErrMsg("compute n_components = 0 should succeed (auto)");

    /* n_components < 0 is rejected by the option registry */
    EXPECT_NE(da_options_set_int(handle, "n_components", -3), da_status_success)
        << NMFErrMsg("set n_components = -3 should be rejected");

    /* n_components > min(n_samples, n_features) = 3 */
    ASSERT_EQ(da_options_set_int(handle, "n_components", 5), da_status_success);
    EXPECT_EQ(da_nmf_compute<T>(handle), da_status_invalid_input)
        << NMFErrMsg("compute n_components > min(n,p)");

    da_handle_destroy(&handle);
}

/* ------------------------------------------------------------------
 * check_every option: boundary validation.
 * ------------------------------------------------------------------ */
TYPED_TEST(NMFTest, CheckEveryBoundaryValidation) {
    using T = TypeParam;
    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);

    EXPECT_NE(da_options_set_int(handle, "check_every", 0), da_status_success);
    EXPECT_NE(da_options_set_int(handle, "check_every", -1), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "check_every", 1), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "check_every", 100), da_status_success);

    da_handle_destroy(&handle);
}

/* ------------------------------------------------------------------
 * da_nmf_set_data: all invalid-argument paths.
 * ------------------------------------------------------------------ */
TYPED_TEST(NMFTest, ErrorExitSetDataInvalidArgs) {
    using T = TypeParam;
    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);

    const da_int n = 4, p = 3, k = 2;
    std::vector<T> A(n * p, T(1));
    std::vector<T> W(n * k, T(1));
    std::vector<T> H(k * p, T(1));

    ASSERT_EQ(da_options_set_int(handle, "n_components", k), da_status_success);
    ASSERT_EQ(da_nmf_init(handle, n, p, A.data(), n), da_status_success);

    /* n_samples mismatch */
    EXPECT_EQ(da_nmf_set_data<T>(handle, n + 1, k, W.data(), n + 1, p, H.data(), k),
              da_status_invalid_input)
        << NMFErrMsg("set_data n_samples mismatch");

    /* n_features mismatch */
    EXPECT_EQ(da_nmf_set_data<T>(handle, n, k, W.data(), n, p + 1, H.data(), k),
              da_status_invalid_input)
        << NMFErrMsg("set_data n_features mismatch");

    /* n_components = 0 */
    EXPECT_EQ(da_nmf_set_data<T>(handle, n, 0, W.data(), n, p, H.data(), 0),
              da_status_invalid_input)
        << NMFErrMsg("set_data n_components = 0");

    /* n_components > min(n, p) */
    const da_int k_big = std::min(n, p) + 1;
    std::vector<T> Wbig(n * k_big, T(1));
    std::vector<T> Hbig(k_big * p, T(1));
    EXPECT_EQ(da_nmf_set_data<T>(handle, n, k_big, Wbig.data(), n, p, Hbig.data(), k_big),
              da_status_invalid_input)
        << NMFErrMsg("set_data n_components > min(n,p)");

    /* both W and H null */
    EXPECT_EQ(
        da_nmf_set_data<T>(handle, n, k, (const T *)nullptr, n, p, (const T *)nullptr, k),
        da_status_invalid_input)
        << NMFErrMsg("set_data both W and H null");

    /* negative entry in W */
    std::vector<T> Wbad = W;
    Wbad[2] = T(-1);
    EXPECT_EQ(da_nmf_set_data<T>(handle, n, k, Wbad.data(), n, p, H.data(), k),
              da_status_invalid_input)
        << NMFErrMsg("set_data negative entry in W");

    /* negative entry in H */
    std::vector<T> Hbad = H;
    Hbad[1] = T(-0.5);
    EXPECT_EQ(da_nmf_set_data<T>(handle, n, k, W.data(), n, p, Hbad.data(), k),
              da_status_invalid_input)
        << NMFErrMsg("set_data negative entry in H");

    /* ldw too small (col-major: ldw >= n_samples) */
    EXPECT_EQ(da_nmf_set_data<T>(handle, n, k, W.data(), n - 1, p, H.data(), k),
              da_status_invalid_leading_dimension)
        << NMFErrMsg("set_data ldw < n_samples (col-major)");

    /* ldh too small (col-major: ldh >= n_components) */
    EXPECT_EQ(da_nmf_set_data<T>(handle, n, k, W.data(), n, p, H.data(), k - 1),
              da_status_invalid_leading_dimension)
        << NMFErrMsg("set_data ldh < n_components (col-major)");

    /* inconsistent k across two partial uploads */
    EXPECT_EQ(da_nmf_set_data<T>(handle, n, k, W.data(), n, p, (const T *)nullptr, k),
              da_status_missing_block);
    /* second call with a different k must be rejected */
    std::vector<T> W3(n * (k + 1), T(1));
    EXPECT_EQ(
        da_nmf_set_data<T>(handle, n, k + 1, W3.data(), n, p, (const T *)nullptr, k + 1),
        da_status_invalid_input)
        << NMFErrMsg("set_data inconsistent k in two partial uploads");

    /* Row-major leading-dimension constraints.
     * row-major W (n x k): ldw >= n_components (k).
     * row-major H (k x p): ldh >= n_features   (p). */
    {
        da_handle handle_rm = nullptr;
        ASSERT_EQ(da_handle_init<T>(&handle_rm, da_handle_nmf), da_status_success);
        ASSERT_EQ(da_options_set_int(handle_rm, "n_components", k), da_status_success);
        ASSERT_EQ(da_options_set_string(handle_rm, "storage order", "row-major"),
                  da_status_success);
        std::vector<T> A_rm(static_cast<std::size_t>(n * p));
        for (da_int i = 0; i < n; ++i)
            for (da_int j = 0; j < p; ++j)
                A_rm[i * p + j] = T(1);
        ASSERT_EQ(da_nmf_init(handle_rm, n, p, A_rm.data(), p), da_status_success);

        /* row-major W: ldw (= n_components in row-major W layout) must be >= k. */
        EXPECT_EQ(da_nmf_set_data<T>(handle_rm, n, k, W.data(), k - 1, p, H.data(), p),
                  da_status_invalid_leading_dimension)
            << NMFErrMsg("set_data ldw < n_components (row-major)");

        /* row-major H: ldh (= n_features in row-major H layout) must be >= p. */
        EXPECT_EQ(da_nmf_set_data<T>(handle_rm, n, k, W.data(), k, p, H.data(), p - 1),
                  da_status_invalid_leading_dimension)
            << NMFErrMsg("set_data ldh < n_features (row-major)");

        /* row-major W with a negative entry (interior at index [1, 0]). */
        std::vector<T> W_rm_bad(n * k, T(1));
        W_rm_bad[1 * k + 0] = T(-1); /* row 1, col 0 in row-major */
        EXPECT_EQ(da_nmf_set_data<T>(handle_rm, n, k, W_rm_bad.data(), k, p, H.data(), p),
                  da_status_invalid_input)
            << NMFErrMsg("set_data negative entry in W (row-major)");

        /* row-major H with a negative entry (interior at index [0, 1]). */
        std::vector<T> H_rm_bad(k * p, T(1));
        H_rm_bad[0 * p + 1] = T(-1); /* row 0, col 1 in row-major */
        EXPECT_EQ(da_nmf_set_data<T>(handle_rm, n, k, W.data(), k, p, H_rm_bad.data(), p),
                  da_status_invalid_input)
            << NMFErrMsg("set_data negative entry in H (row-major)");

        da_handle_destroy(&handle_rm);
    }

    da_handle_destroy(&handle);
}

/* ------------------------------------------------------------------
 * da_nmf_transform: invalid-argument paths.
 * ------------------------------------------------------------------ */
TYPED_TEST(NMFTest, ErrorExitTransformInvalidArgs) {
    using T = TypeParam;
    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);

    const da_int n = 4, p = 3, k = 2;
    std::vector<T> A(n * p, T(1));
    ASSERT_EQ(da_options_set_int(handle, "n_components", k), da_status_success);
    ASSERT_EQ(da_nmf_init(handle, n, p, A.data(), n), da_status_success);
    ASSERT_EQ(da_nmf_compute<T>(handle), da_status_success);

    const da_int m = 2;
    std::vector<T> X(m * p, T(1));
    std::vector<T> Xt(m * k, T(0));

    /* null X */
    EXPECT_EQ(da_nmf_transform(handle, m, p, (const T *)nullptr, m, Xt.data(), m),
              da_status_invalid_pointer)
        << NMFErrMsg("transform null X");

    /* null Xt */
    EXPECT_EQ(da_nmf_transform(handle, m, p, X.data(), m, (T *)nullptr, m),
              da_status_invalid_pointer)
        << NMFErrMsg("transform null Xt");

    /* m_samples < 1 */
    EXPECT_EQ(da_nmf_transform(handle, 0, p, X.data(), m, Xt.data(), m),
              da_status_invalid_array_dimension)
        << NMFErrMsg("transform m_samples = 0");
    EXPECT_EQ(da_nmf_transform(handle, -1, p, X.data(), m, Xt.data(), m),
              da_status_invalid_array_dimension)
        << NMFErrMsg("transform m_samples = -1");

    /* m_features != n_features */
    EXPECT_EQ(da_nmf_transform(handle, m, p + 1, X.data(), m, Xt.data(), m),
              da_status_invalid_input)
        << NMFErrMsg("transform m_features mismatch");

    /* ldx too small (col-major: ldx >= m_samples) */
    EXPECT_EQ(da_nmf_transform(handle, m, p, X.data(), m - 1, Xt.data(), m),
              da_status_invalid_leading_dimension)
        << NMFErrMsg("transform ldx < m_samples (col-major)");

    /* ldxt too small (col-major: ldxt >= m_samples) */
    EXPECT_EQ(da_nmf_transform(handle, m, p, X.data(), m, Xt.data(), m - 1),
              da_status_invalid_leading_dimension)
        << NMFErrMsg("transform ldxt < m_samples (col-major)");

    /* Row-major leading-dimension constraints.
     * row-major: ldx >= m_features (p),  ldxt >= n_components (k). */
    ASSERT_EQ(da_options_set_string(handle, "storage order", "row-major"),
              da_status_success);
    {
        std::vector<T> A_rm(static_cast<std::size_t>(n * p));
        for (da_int i = 0; i < n; ++i)
            for (da_int j = 0; j < p; ++j)
                A_rm[i * p + j] = T(1);
        ASSERT_EQ(da_nmf_init(handle, n, p, A_rm.data(), p), da_status_success);
        ASSERT_EQ(da_nmf_compute<T>(handle), da_status_success);

        std::vector<T> X_rm(static_cast<std::size_t>(m * p), T(1));
        std::vector<T> Xt_rm(static_cast<std::size_t>(m * k), T(0));

        /* ldx < m_features (row-major) */
        EXPECT_EQ(da_nmf_transform(handle, m, p, X_rm.data(), p - 1, Xt_rm.data(), k),
                  da_status_invalid_leading_dimension)
            << NMFErrMsg("transform ldx < m_features (row-major)");

        /* ldxt < n_components (row-major) */
        EXPECT_EQ(da_nmf_transform(handle, m, p, X_rm.data(), p, Xt_rm.data(), k - 1),
                  da_status_invalid_leading_dimension)
            << NMFErrMsg("transform ldxt < n_components (row-major)");
    }

    da_handle_destroy(&handle);
}

/* ------------------------------------------------------------------
 * da_nmf_inverse_transform: invalid-argument paths.
 * ------------------------------------------------------------------ */
TYPED_TEST(NMFTest, ErrorExitInverseTransformInvalidArgs) {
    using T = TypeParam;
    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);

    const da_int n = 4, p = 3, k = 2;
    std::vector<T> A(n * p, T(1));
    ASSERT_EQ(da_options_set_int(handle, "n_components", k), da_status_success);
    ASSERT_EQ(da_nmf_init(handle, n, p, A.data(), n), da_status_success);
    ASSERT_EQ(da_nmf_compute<T>(handle), da_status_success);

    const da_int m = 3;
    std::vector<T> Xt(m * k, T(1));
    std::vector<T> X(m * p, T(0));

    /* null Xt */
    EXPECT_EQ(da_nmf_inverse_transform(handle, m, k, (const T *)nullptr, m, X.data(), m),
              da_status_invalid_pointer)
        << NMFErrMsg("inverse_transform null Xt");

    /* null X */
    EXPECT_EQ(da_nmf_inverse_transform(handle, m, k, Xt.data(), m, (T *)nullptr, m),
              da_status_invalid_pointer)
        << NMFErrMsg("inverse_transform null X");

    /* k_samples < 1 */
    EXPECT_EQ(da_nmf_inverse_transform(handle, 0, k, Xt.data(), m, X.data(), m),
              da_status_invalid_array_dimension)
        << NMFErrMsg("inverse_transform k_samples = 0");
    EXPECT_EQ(da_nmf_inverse_transform(handle, -1, k, Xt.data(), m, X.data(), m),
              da_status_invalid_array_dimension)
        << NMFErrMsg("inverse_transform k_samples = -1");

    /* k_components != n_components */
    EXPECT_EQ(da_nmf_inverse_transform(handle, m, k + 1, Xt.data(), m, X.data(), m),
              da_status_invalid_input)
        << NMFErrMsg("inverse_transform k_components mismatch");

    /* ldxt too small (col-major: ldxt >= k_samples) */
    EXPECT_EQ(da_nmf_inverse_transform(handle, m, k, Xt.data(), m - 1, X.data(), m),
              da_status_invalid_leading_dimension)
        << NMFErrMsg("inverse_transform ldxt < k_samples (col-major)");

    /* ldx too small (col-major: ldx >= k_samples) */
    EXPECT_EQ(da_nmf_inverse_transform(handle, m, k, Xt.data(), m, X.data(), m - 1),
              da_status_invalid_leading_dimension)
        << NMFErrMsg("inverse_transform ldx < k_samples (col-major)");

    /* Row-major leading-dimension constraints.
     * row-major: ldxt >= n_components (k),  ldx >= n_features (p). */
    ASSERT_EQ(da_options_set_string(handle, "storage order", "row-major"),
              da_status_success);
    {
        std::vector<T> A_rm(static_cast<std::size_t>(n * p));
        for (da_int i = 0; i < n; ++i)
            for (da_int j = 0; j < p; ++j)
                A_rm[i * p + j] = T(1);
        ASSERT_EQ(da_nmf_init(handle, n, p, A_rm.data(), p), da_status_success);
        ASSERT_EQ(da_nmf_compute<T>(handle), da_status_success);

        std::vector<T> Xt_rm(static_cast<std::size_t>(m * k), T(1));
        std::vector<T> X_rm(static_cast<std::size_t>(m * p), T(0));

        /* ldxt < n_components (row-major) */
        EXPECT_EQ(
            da_nmf_inverse_transform(handle, m, k, Xt_rm.data(), k - 1, X_rm.data(), p),
            da_status_invalid_leading_dimension)
            << NMFErrMsg("inverse_transform ldxt < n_components (row-major)");

        /* ldx < n_features (row-major) */
        EXPECT_EQ(
            da_nmf_inverse_transform(handle, m, k, Xt_rm.data(), k, X_rm.data(), p - 1),
            da_status_invalid_leading_dimension)
            << NMFErrMsg("inverse_transform ldx < n_features (row-major)");
    }

    da_handle_destroy(&handle);
}

/* ------------------------------------------------------------------
 * da_handle_get_result: null pointer and undersized-array paths.
 * ------------------------------------------------------------------ */
TYPED_TEST(NMFTest, ErrorExitGetResultInvalidArgs) {
    using T = TypeParam;
    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);

    const da_int n = 4, p = 3, k = 2;
    std::vector<T> A(n * p, T(1));
    ASSERT_EQ(da_options_set_int(handle, "n_components", k), da_status_success);
    ASSERT_EQ(da_nmf_init(handle, n, p, A.data(), n), da_status_success);
    ASSERT_EQ(da_nmf_compute<T>(handle), da_status_success);

    /* null result pointer */
    da_int dim = n * k;
    EXPECT_EQ(da_handle_get_result<T>(handle, da_nmf_w, &dim, (T *)nullptr),
              da_status_invalid_input)
        << NMFErrMsg("get_result null result ptr (W)");

    dim = k * p;
    EXPECT_EQ(da_handle_get_result<T>(handle, da_nmf_h, &dim, (T *)nullptr),
              da_status_invalid_input)
        << NMFErrMsg("get_result null result ptr (H)");

    dim = 1;
    EXPECT_EQ(
        da_handle_get_result<T>(handle, da_nmf_reconstruction_err, &dim, (T *)nullptr),
        da_status_invalid_input)
        << NMFErrMsg("get_result null result ptr (reconstruction_err)");

    da_int int_result = 0;
    EXPECT_EQ(da_handle_get_result_int(handle, da_nmf_n_iter, &dim, nullptr),
              da_status_invalid_input)
        << NMFErrMsg("get_result null result ptr (n_iter)");

    /* null dim pointer */
    std::vector<T> buf(n * k, T(0));
    EXPECT_EQ(da_handle_get_result<T>(handle, da_nmf_w, nullptr, buf.data()),
              da_status_invalid_input)
        << NMFErrMsg("get_result null dim ptr (W)");

    /* undersized array: dim reported back and da_status_invalid_array_dimension returned */
    da_int small_dim = 1;
    EXPECT_EQ(da_handle_get_result<T>(handle, da_nmf_w, &small_dim, buf.data()),
              da_status_invalid_array_dimension)
        << NMFErrMsg("get_result undersized array for W");
    EXPECT_EQ(small_dim, n * k) << NMFErrMsg("get_result W dim not updated on undersize");

    small_dim = 1;
    EXPECT_EQ(da_handle_get_result<T>(handle, da_nmf_h, &small_dim, buf.data()),
              da_status_invalid_array_dimension)
        << NMFErrMsg("get_result undersized array for H");
    EXPECT_EQ(small_dim, k * p) << NMFErrMsg("get_result H dim not updated on undersize");

    small_dim = 0;
    EXPECT_EQ(da_handle_get_result<T>(handle, da_nmf_reconstruction_err, &small_dim,
                                      buf.data()),
              da_status_invalid_array_dimension)
        << NMFErrMsg("get_result undersized array for reconstruction_err");

    small_dim = 0;
    EXPECT_EQ(da_handle_get_result_int(handle, da_nmf_n_iter, &small_dim, &int_result),
              da_status_invalid_array_dimension)
        << NMFErrMsg("get_result undersized array for n_iter");

    /* da_nmf_n_components: value, undersized, null */
    da_int n_comp_result = 0;
    da_int comp_dim = 1;
    EXPECT_EQ(
        da_handle_get_result_int(handle, da_nmf_n_components, &comp_dim, &n_comp_result),
        da_status_success)
        << NMFErrMsg("get_result n_components value");
    EXPECT_EQ(n_comp_result, k) << NMFErrMsg("get_result n_components wrong value");

    small_dim = 0;
    EXPECT_EQ(
        da_handle_get_result_int(handle, da_nmf_n_components, &small_dim, &n_comp_result),
        da_status_invalid_array_dimension)
        << NMFErrMsg("get_result undersized array for n_components");

    EXPECT_EQ(da_handle_get_result_int(handle, da_nmf_n_components, &comp_dim, nullptr),
              da_status_invalid_input)
        << NMFErrMsg("get_result null result ptr (n_components)");

    /* da_rinfo: undersized array (dim=1 < 2) */
    small_dim = 1;
    EXPECT_EQ(da_handle_get_result<T>(handle, da_rinfo, &small_dim, buf.data()),
              da_status_invalid_array_dimension)
        << NMFErrMsg("get_result undersized array for da_rinfo");
    EXPECT_EQ(small_dim, 2) << NMFErrMsg(
        "get_result da_rinfo dim not updated on undersize");

    /* unknown query */
    dim = 1;
    EXPECT_EQ(da_handle_get_result<T>(handle, da_result(9999), &dim, buf.data()),
              da_status_unknown_query)
        << NMFErrMsg("get_result unknown query (fp)");
    EXPECT_EQ(da_handle_get_result_int(handle, da_result(9999), &dim, &int_result),
              da_status_unknown_query)
        << NMFErrMsg("get_result unknown query (int)");

    da_handle_destroy(&handle);
}

/* ============================================================
 * check_every option tests.
 * ============================================================ */

/* Verify that changing check_every does not affect the final result when
 * tol=0 (i.e. no early stopping).  With tol=0 check_every is irrelevant
 * to the number of iterations; convergence is driven only by max_iter.
 * Both runs must produce bit-identical output because the MU update rule
 * itself is unchanged. */
TYPED_TEST(NMFTest, CheckEveryDoesNotAffectResultWhenTolZero) {
    using T = TypeParam;
    da_int n = 4, p = 3;
    std::vector<T> A = {T(1), T(2), T(3), T(4),  T(5),  T(6),
                        T(7), T(8), T(9), T(10), T(11), T(12)};
    const da_int k = 2;

    /* Lambda runs NMF for different values of check_every. */
    auto fit_with_check_every =
        [&](da_int check_every_val) -> std::pair<std::vector<T>, std::vector<T>> {
        da_handle nmf_handle = nullptr;
        EXPECT_EQ(da_handle_init<T>(&nmf_handle, da_handle_nmf), da_status_success);
        EXPECT_EQ(da_options_set_int(nmf_handle, "n_components", k), da_status_success);
        EXPECT_EQ(da_options_set_string(nmf_handle, "init", "nndsvda"),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(nmf_handle, "max_iter", 20), da_status_success);
        EXPECT_EQ(da_options_set(nmf_handle, "tol", T(0)), da_status_success);
        EXPECT_EQ(da_options_set_int(nmf_handle, "random_state", 42), da_status_success);
        EXPECT_EQ(da_options_set_int(nmf_handle, "check_every", check_every_val),
                  da_status_success);
        EXPECT_EQ(da_nmf_init(nmf_handle, n, p, A.data(), n), da_status_success);
        EXPECT_EQ(da_nmf_compute<T>(nmf_handle), da_status_success);

        std::vector<T> W(n * k), H(k * p);
        da_int wd = n * k, hd = k * p;
        EXPECT_EQ(da_handle_get_result<T>(nmf_handle, da_nmf_w, &wd, W.data()),
                  da_status_success);
        EXPECT_EQ(da_handle_get_result<T>(nmf_handle, da_nmf_h, &hd, H.data()),
                  da_status_success);
        da_int n_iter = 0, nd = 1;
        EXPECT_EQ(da_handle_get_result_int(nmf_handle, da_nmf_n_iter, &nd, &n_iter),
                  da_status_success);
        std::cout << "  check_every=" << check_every_val << " n_iter=" << n_iter << "\n";
        da_handle_destroy(&nmf_handle);
        return {W, H};
    };
    /* Call lambda with different check_every values */
    auto [W1, H1] = fit_with_check_every(1);
    auto [W20, H20] = fit_with_check_every(20);
    /* Multithreaded BLAS may assign different reduction orders across calls, causing last-bit differences. */
    const T check_every_tol = T(100) * std::numeric_limits<T>::epsilon();
    for (da_int i = 0; i < n * k; ++i)
        EXPECT_NEAR(W1[i], W20[i], check_every_tol);
    for (da_int i = 0; i < k * p; ++i)
        EXPECT_NEAR(H1[i], H20[i], check_every_tol);
}

/* Verify that a larger check_every value causes the convergence check to
 * fire later, so more iterations are performed before the threshold is met.
 * We use a very tight tol so that early stopping would normally trigger at
 * iter ~= check_every. With check_every=1 the solver exits as soon as the
 * first check passes, with check_every=50 it must run at least 50 iters. */
TYPED_TEST(NMFTest, CheckEveryControlsCheckFrequency) {
    using T = TypeParam;
    da_int n = 4, p = 3;
    std::vector<T> A = {T(1), T(2), T(3), T(4),  T(5),  T(6),
                        T(7), T(8), T(9), T(10), T(11), T(12)};
    const da_int k = 2;

    /* Very loose tol so that early stopping triggers on the first check. */
    const T loose_tol = T(1.0e10);

    auto n_iter_with = [&](da_int check_every_val) -> da_int {
        da_handle nmf_handle = nullptr;
        EXPECT_EQ(da_handle_init<T>(&nmf_handle, da_handle_nmf), da_status_success);
        EXPECT_EQ(da_options_set_int(nmf_handle, "n_components", k), da_status_success);
        EXPECT_EQ(da_options_set_string(nmf_handle, "init", "nndsvda"),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(nmf_handle, "max_iter", 200), da_status_success);
        EXPECT_EQ(da_options_set(nmf_handle, "tol", loose_tol), da_status_success);
        EXPECT_EQ(da_options_set_int(nmf_handle, "check_every", check_every_val),
                  da_status_success);
        EXPECT_EQ(da_nmf_init(nmf_handle, n, p, A.data(), n), da_status_success);
        EXPECT_EQ(da_nmf_compute<T>(nmf_handle), da_status_success);
        da_int it = 0, d = 1;
        EXPECT_EQ(da_handle_get_result_int(nmf_handle, da_nmf_n_iter, &d, &it),
                  da_status_success);
        T err = T(0);
        d = 1;
        EXPECT_EQ(
            da_handle_get_result<T>(nmf_handle, da_nmf_reconstruction_err, &d, &err),
            da_status_success);
        std::cout << "  check_every=" << check_every_val << " n_iter=" << it
                  << " reconstruction_err=" << err << "\n";
        da_handle_destroy(&nmf_handle);
        return it;
    };

    const da_int it1 = n_iter_with(1);
    const da_int it20 = n_iter_with(20);

    /* With check_every=20 the solver cannot stop before iter 20, so it must
     * run at least as many iterations as with check_every=1. */
    EXPECT_GE(it20, it1);
    /* And specifically it must have run at least 20 iterations. */
    EXPECT_GE(it20, 20);
}

TYPED_TEST(NMFTest, TransformMultipleDataOnSameHandle) {
    using T = TypeParam;
    da_int n = 4, p = 3, k = 2;
    std::vector<T> A = {T(1), T(2), T(3), T(4),  T(5),  T(6),
                        T(7), T(8), T(9), T(10), T(11), T(12)};

    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);
    ASSERT_EQ(da_options_set_int(handle, "n_components", k), da_status_success);
    ASSERT_EQ(da_options_set_string(handle, "init", "nndsvda"), da_status_success);
    ASSERT_EQ(da_options_set_int(handle, "random_state", 42), da_status_success);
    ASSERT_EQ(da_nmf_init(handle, n, p, A.data(), n), da_status_success);
    ASSERT_EQ(da_nmf_compute<T>(handle), da_status_success);

    /* Check with an m1-by-k matrix. */
    const da_int m1 = 2;
    std::vector<T> X1 = {T(1), T(2), T(3), T(4), T(5), T(6)};
    std::vector<T> Xt1(m1 * k, T(-1));
    ASSERT_EQ(da_nmf_transform(handle, m1, p, X1.data(), m1, Xt1.data(), m1),
              da_status_success);
    for (T v : Xt1)
        EXPECT_GE(v, T(0));

    /* Check with an m2-by-k matrix. */
    const da_int m2 = 3;
    std::vector<T> X2 = {T(2), T(4), T(6), T(3), T(6), T(9), T(4), T(8), T(12)};
    std::vector<T> Xt2(m2 * k, T(-1));
    ASSERT_EQ(da_nmf_transform(handle, m2, p, X2.data(), m2, Xt2.data(), m2),
              da_status_success);
    for (T v : Xt2)
        EXPECT_GE(v, T(0));

    /* Re-transforming the same X with the same handle should give near-identical results. */
    std::vector<T> Xt1_repeat(m1 * k, T(-1));
    ASSERT_EQ(da_nmf_transform(handle, m1, p, X1.data(), m1, Xt1_repeat.data(), m1),
              da_status_success);
    const T repeat_tol = T(100) * std::numeric_limits<T>::epsilon();
    for (da_int i = 0; i < m1 * k; ++i)
        EXPECT_NEAR(Xt1[i], Xt1_repeat[i], repeat_tol);

    da_handle_destroy(&handle);
}

/* ============================================================
 * Parameterised functionality tests.
 *
 * test_nmf_functionality drives every test case in NMFParamType:
 *   - fits NMF on A
 *   - retrieves and validates W, H, reconstruction_err, n_iter, da_rinfo
 *   - validates inverse_transform(transform(X_test)) ~= expected_Xinv
 * ============================================================ */
template <typename T> void test_nmf_functionality(const NMFParamType<T> &param) {
    std::cout << "NMF functionality test: " << param.test_name << std::endl;

    da_handle handle = nullptr;
    ASSERT_EQ(da_handle_init<T>(&handle, da_handle_nmf), da_status_success);

    /* Storage order. */
    EXPECT_EQ(da_options_set_string(handle, "storage order", param.order.c_str()),
              da_status_success);

    /* Solver options. */
    EXPECT_EQ(da_options_set_int(handle, "n_components", param.n_components),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "init", param.init.c_str()),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "random_state", param.random_state),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "max_iter", param.max_iter), da_status_success);
    EXPECT_EQ(da_options_set(handle, "tol", param.nmf_tol), da_status_success);

    /* Fit. */
    ASSERT_EQ(
        da_nmf_init(handle, param.n_samples, param.n_features, param.A.data(), param.lda),
        da_status_success);
    if (param.init == "custom") {
        ASSERT_EQ(da_nmf_set_data<T>(handle, param.n_samples, param.n_components,
                                     param.W_init.data(), param.n_samples,
                                     param.n_features, param.H_init.data(),
                                     param.n_components),
                  da_status_success);
    }
    ASSERT_EQ(da_nmf_compute<T>(handle), da_status_success);

    /* Retrieve W (n_samples x n_components) and H (n_components x n_features). */
    da_int wdim = param.n_samples * param.n_components;
    da_int hdim = param.n_components * param.n_features;
    std::vector<T> W(static_cast<std::size_t>(wdim));
    std::vector<T> H(static_cast<std::size_t>(hdim));
    ASSERT_EQ(da_handle_get_result<T>(handle, da_nmf_w, &wdim, W.data()),
              da_status_success);
    ASSERT_EQ(da_handle_get_result<T>(handle, da_nmf_h, &hdim, H.data()),
              da_status_success);

    /* Retrieve reconstruction_err and n_iter. */
    T err = T(0);
    da_int d = 1;
    ASSERT_EQ(da_handle_get_result<T>(handle, da_nmf_reconstruction_err, &d, &err),
              da_status_success);
    da_int n_iter = 0;
    d = 1;
    ASSERT_EQ(da_handle_get_result_int(handle, da_nmf_n_iter, &d, &n_iter),
              da_status_success);

    /* Non-negativity of all factor entries. */
    for (T v : W)
        EXPECT_GE(v, T(0));
    for (T v : H)
        EXPECT_GE(v, T(0));

    /* Comparison against expected values. */
    if (!param.expected_W.empty()) {
        EXPECT_ARR_NEAR(wdim, W.data(), param.expected_W.data(), param.convergence_tol);
    }
    if (!param.expected_H.empty()) {
        EXPECT_ARR_NEAR(hdim, H.data(), param.expected_H.data(), param.convergence_tol);
    }
    if (param.expected_reconstruction_err.has_value()) {
        EXPECT_NEAR(err, param.expected_reconstruction_err.value(),
                    param.convergence_tol);
    }

    /* reconstruction_err must be finite; n_iter must not exceed max_iter. */
    EXPECT_TRUE(std::isfinite(err));
    EXPECT_LE(n_iter, param.max_iter);

    /* Convergence check: reconstruction_err / ||A||_F < convergence_tol. */
    T A_fro_sq = T(0);
    for (T v : param.A)
        A_fro_sq += v * v;
    const T A_fro = std::sqrt(A_fro_sq);
    /* For a zero data matrix A_fro = 0, check absolute error instead. */
    if (A_fro > T(0)) {
        EXPECT_GE(n_iter, 1);
        EXPECT_LT(err / A_fro, param.convergence_tol);
    } else
        EXPECT_NEAR(err, T(0), param.convergence_tol);

    /* Cross-check reconstruction_err vs explicit W*H residual (both orders). */
    if (A_fro > T(0)) {
        const bool col_major = (param.order == "column-major");
        const T err_explicit =
            explicit_residual<T>(param.A, param.n_samples, param.n_features, W, H,
                                 param.n_components, param.lda, col_major);
        const T abs_tol = T(100) * (std::sqrt(std::numeric_limits<T>::epsilon()) * A_fro +
                                    std::numeric_limits<T>::epsilon() * err);
        EXPECT_NEAR(err, err_explicit, abs_tol);
    }

    /* da_rinfo: {n_samples, n_features}. */
    if (!param.expected_rinfo.empty()) {
        da_int rsize = 2;
        std::vector<T> rinfo(2);
        EXPECT_EQ(da_handle_get_result<T>(handle, da_rinfo, &rsize, rinfo.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(2, rinfo.data(), param.expected_rinfo.data(),
                        T(10) * std::numeric_limits<T>::epsilon());
    }

    /* Transform test: non-negativity of Xt, Xinv == Xt*H to machine precision,
     * and Xinv ~= expected_Xinv if provided. */
    const da_int xt_size = param.m_test * param.n_components;
    const da_int xinv_size = param.m_test * param.n_features;
    std::vector<T> Xt(static_cast<std::size_t>(xt_size), T(-1));
    std::vector<T> Xinv(static_cast<std::size_t>(xinv_size), T(-1));

    ASSERT_EQ(da_nmf_transform<T>(handle, param.m_test, param.n_features,
                                  param.X_test.data(), param.ldx_test, Xt.data(),
                                  param.ldxt),
              da_status_success);
    for (T v : Xt)
        EXPECT_GE(v, T(0));

    ASSERT_EQ(da_nmf_inverse_transform<T>(handle, param.m_test, param.n_components,
                                          Xt.data(), param.ldxt, Xinv.data(),
                                          param.ldyinv),
              da_status_success);

    /* inverse_transform(Xt) must equal Xt * H to machine precision. */
    const bool col_major_xt = (param.order == "column-major");
    std::vector<T> Xinv_ref(static_cast<std::size_t>(param.m_test * param.n_features));
    const T gemm_tol = T(10) * std::numeric_limits<T>::epsilon() * A_fro;
    if (col_major_xt) {
        datest_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, param.m_test,
                                param.n_features, param.n_components, T(1), Xt.data(),
                                param.ldxt, H.data(), param.n_components, T(0),
                                Xinv_ref.data(), param.m_test);
        for (da_int j = 0; j < param.n_features; ++j)
            for (da_int i = 0; i < param.m_test; ++i)
                EXPECT_NEAR(Xinv[i + j * param.ldyinv], Xinv_ref[i + j * param.m_test],
                            gemm_tol);
    } else {
        datest_blas::cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, param.m_test,
                                param.n_features, param.n_components, T(1), Xt.data(),
                                param.ldxt, H.data(), param.n_features, T(0),
                                Xinv_ref.data(), param.n_features);
        for (da_int i = 0; i < param.m_test; ++i)
            for (da_int j = 0; j < param.n_features; ++j)
                EXPECT_NEAR(Xinv[i * param.ldyinv + j],
                            Xinv_ref[i * param.n_features + j], gemm_tol);
    }

    if (param.check_xinv_roundtrip) {
        EXPECT_ARR_NEAR(xinv_size, Xinv.data(), param.expected_Xinv.data(),
                        param.convergence_tol);
    }

    da_handle_destroy(&handle);
}

/* ============================================================
 * Parameterised test classes.
 * ============================================================ */

class DoubleNMFFunctionalityTest : public testing::TestWithParam<NMFParamType<double>> {};
class FloatNMFFunctionalityTest : public testing::TestWithParam<NMFParamType<float>> {};

template <typename T> void PrintTo(const NMFParamType<T> &param, ::std::ostream *os) {
    *os << param.test_name;
}

TEST_P(DoubleNMFFunctionalityTest, ParameterizedTest) {
    const NMFParamType<double> &p = GetParam();
    test_nmf_functionality(p);
}

TEST_P(FloatNMFFunctionalityTest, ParameterizedTest) {
    const NMFParamType<float> &p = GetParam();
    test_nmf_functionality(p);
}

template <typename T> std::vector<NMFParamType<T>> getNMFParams() {
    std::vector<NMFParamType<T>> params;
    GetNMFData(params);
    return params;
}

INSTANTIATE_TEST_SUITE_P(NMF_Functionality_Tests_Double, DoubleNMFFunctionalityTest,
                         ::testing::ValuesIn(getNMFParams<double>()));
INSTANTIATE_TEST_SUITE_P(NMF_Functionality_Tests_Float, FloatNMFFunctionalityTest,
                         ::testing::ValuesIn(getNMFParams<float>()));

/* ------------------------------------------------------------------
 * auto init: verify dispatch to nndsvda (k < min(n,p)) vs random (k == min(n,p))
 * by cross-checking against the explicit init.
 * ------------------------------------------------------------------ */
TYPED_TEST(NMFTest, AutoInitDispatch) {
    using T = TypeParam;
    da_int n = 4, p = 3;
    std::vector<T> A = {T(1), T(2), T(3), T(4),  T(5),  T(6),
                        T(7), T(8), T(9), T(10), T(11), T(12)};
    const da_int k_lt = 2; // k < min(4,3)=3 -> should dispatch to nndsvda
    const da_int k_eq = 3; // k == min(4,3)=3 -> should dispatch to random

    /* k < min(n,p): auto should match nndsvda with the same seed */
    auto fit_and_get_factors = [&](const char *init_str, da_int k) {
        da_handle nmf_handle = nullptr;
        EXPECT_EQ(da_handle_init<T>(&nmf_handle, da_handle_nmf), da_status_success);
        EXPECT_EQ(da_options_set_int(nmf_handle, "n_components", k), da_status_success);
        EXPECT_EQ(da_options_set_string(nmf_handle, "init", init_str), da_status_success);
        EXPECT_EQ(da_options_set_int(nmf_handle, "random_state", 0), da_status_success);
        EXPECT_EQ(da_options_set(nmf_handle, "tol", T(0)), da_status_success);
        EXPECT_EQ(da_options_set_int(nmf_handle, "max_iter", 20), da_status_success);
        EXPECT_EQ(da_nmf_init(nmf_handle, n, p, A.data(), n), da_status_success);
        EXPECT_EQ(da_nmf_compute<T>(nmf_handle), da_status_success);
        std::vector<T> W(n * k), H(k * p);
        da_int wd = n * k, hd = k * p;
        EXPECT_EQ(da_handle_get_result<T>(nmf_handle, da_nmf_w, &wd, W.data()),
                  da_status_success);
        EXPECT_EQ(da_handle_get_result<T>(nmf_handle, da_nmf_h, &hd, H.data()),
                  da_status_success);
        da_handle_destroy(&nmf_handle);
        return std::make_pair(W, H);
    };
    /* Compute NMF with different init options*/
    auto [W_auto_lt, H_auto_lt] = fit_and_get_factors("auto", k_lt);
    auto [W_nndsvda, H_nndsvda] = fit_and_get_factors("nndsvda", k_lt);
    /* Multithreaded BLAS may assign different reduction orders across calls, causing last-bit differences. */
    const T dispatch_tol = T(100) * std::numeric_limits<T>::epsilon();
    EXPECT_ARR_NEAR(n * k_lt, W_auto_lt.data(), W_nndsvda.data(), dispatch_tol);
    EXPECT_ARR_NEAR(k_lt * p, H_auto_lt.data(), H_nndsvda.data(), dispatch_tol);

    /* k == min(n,p): auto should match random with the same seed. */
    auto [W_auto_eq, H_auto_eq] = fit_and_get_factors("auto", k_eq);
    auto [W_random, H_random] = fit_and_get_factors("random", k_eq);
    EXPECT_ARR_NEAR(n * k_eq, W_auto_eq.data(), W_random.data(), dispatch_tol);
    EXPECT_ARR_NEAR(k_eq * p, H_auto_eq.data(), H_random.data(), dispatch_tol);
}

/* ------------------------------------------------------------------
 * Row-major: cross-check that col-major and row-major fits on the same
 * data produce identical W, H, reconstruction_err, transform, and
 * inverse_transform outputs.
 * ------------------------------------------------------------------ */
TYPED_TEST(NMFTest, RowMajorFitAndTransformCorrectness) {
    using T = TypeParam;
    da_int n = 4, p = 3;
    std::vector<T> A_col = {T(1), T(2), T(3), T(4),  T(5),  T(6),
                            T(7), T(8), T(9), T(10), T(11), T(12)};
    std::vector<T> A_row = {T(1), T(5), T(9),  T(2), T(6), T(10),
                            T(3), T(7), T(11), T(4), T(8), T(12)};
    const da_int k = 2;

    // Reference: column-major
    da_handle h_col = nullptr;
    ASSERT_EQ(da_handle_init<T>(&h_col, da_handle_nmf), da_status_success);
    ASSERT_EQ(da_options_set_int(h_col, "n_components", k), da_status_success);
    ASSERT_EQ(da_options_set_string(h_col, "init", "nndsvda"), da_status_success);
    ASSERT_EQ(da_options_set_int(h_col, "random_state", 5), da_status_success);
    ASSERT_EQ(da_options_set(h_col, "tol", T(0)), da_status_success);
    ASSERT_EQ(da_options_set_int(h_col, "max_iter", 30), da_status_success);
    ASSERT_EQ(da_nmf_init(h_col, n, p, A_col.data(), n), da_status_success);
    ASSERT_EQ(da_nmf_compute<T>(h_col), da_status_success);

    std::vector<T> W_col(n * k), H_col(k * p);
    da_int wd = n * k, hd = k * p;
    ASSERT_EQ(da_handle_get_result<T>(h_col, da_nmf_w, &wd, W_col.data()),
              da_status_success);
    ASSERT_EQ(da_handle_get_result<T>(h_col, da_nmf_h, &hd, H_col.data()),
              da_status_success);
    T err_col = T(0);
    da_int ed = 1;
    ASSERT_EQ(da_handle_get_result<T>(h_col, da_nmf_reconstruction_err, &ed, &err_col),
              da_status_success);

    // Row-major run
    da_handle h_row = nullptr;
    ASSERT_EQ(da_handle_init<T>(&h_row, da_handle_nmf), da_status_success);
    ASSERT_EQ(da_options_set_string(h_row, "storage order", "row-major"),
              da_status_success);
    ASSERT_EQ(da_options_set_int(h_row, "n_components", k), da_status_success);
    ASSERT_EQ(da_options_set_string(h_row, "init", "nndsvda"), da_status_success);
    ASSERT_EQ(da_options_set_int(h_row, "random_state", 5), da_status_success);
    ASSERT_EQ(da_options_set(h_row, "tol", T(0)), da_status_success);
    ASSERT_EQ(da_options_set_int(h_row, "max_iter", 30), da_status_success);
    ASSERT_EQ(da_nmf_init(h_row, n, p, A_row.data(), p), da_status_success);
    ASSERT_EQ(da_nmf_compute<T>(h_row), da_status_success);

    // get_result returns data in user's (row-major) order: W is (n*k) row-major
    // and H is (k*p) row-major.  Convert to column-major for comparison.
    std::vector<T> W_row_rm(n * k), H_row_rm(k * p);
    wd = n * k;
    hd = k * p;
    ASSERT_EQ(da_handle_get_result<T>(h_row, da_nmf_w, &wd, W_row_rm.data()),
              da_status_success);
    ASSERT_EQ(da_handle_get_result<T>(h_row, da_nmf_h, &hd, H_row_rm.data()),
              da_status_success);
    T err_row = T(0);
    ed = 1;
    ASSERT_EQ(da_handle_get_result<T>(h_row, da_nmf_reconstruction_err, &ed, &err_row),
              da_status_success);
    da_int n_iter_row = 0;
    da_int nd = 1;
    ASSERT_EQ(da_handle_get_result_int(h_row, da_nmf_n_components, &nd, &n_iter_row),
              da_status_success);
    EXPECT_EQ(n_iter_row, k);

    // Convert row-major W_row (n*k lda=k) to column-major for comparison
    std::vector<T> W_row_cm(n * k);
    for (da_int i = 0; i < n; ++i)
        for (da_int j = 0; j < k; ++j)
            W_row_cm[i + j * n] = W_row_rm[i * k + j];
    // Convert row-major H_row (k*p lda=p) to column-major for comparison
    std::vector<T> H_row_cm(k * p);
    for (da_int i = 0; i < k; ++i)
        for (da_int j = 0; j < p; ++j)
            H_row_cm[i + j * k] = H_row_rm[i * p + j];

    /* Multithreaded BLAS may assign different reduction orders across calls, causing last-bit differences. */
    const T cross_tol = T(100) * std::numeric_limits<T>::epsilon();
    EXPECT_NEAR(err_col, err_row, cross_tol);
    for (da_int i = 0; i < n * k; ++i)
        EXPECT_NEAR(W_col[i], W_row_cm[i], cross_tol);
    for (da_int i = 0; i < k * p; ++i)
        EXPECT_NEAR(H_col[i], H_row_cm[i], cross_tol);

    // transform: 2 new samples in row-major (lda=p=3) and column-major (lda=m=2)
    da_int m = 2;
    std::vector<T> X_col = {T(1), T(2), T(3), T(4), T(5), T(6)};
    std::vector<T> X_row = {T(1), T(3), T(5), T(2), T(4), T(6)};

    // Column-major transform output (m*k, lda=m=2)
    std::vector<T> Xt_col(m * k, T(0));
    ASSERT_EQ(da_nmf_transform<T>(h_col, m, p, X_col.data(), m, Xt_col.data(), m),
              da_status_success);

    // Row-major transform output (m*k, lda=k=2); lda=k since row-major
    std::vector<T> Xt_row_rm(m * k, T(0));
    ASSERT_EQ(da_nmf_transform<T>(h_row, m, p, X_row.data(), p, Xt_row_rm.data(), k),
              da_status_success);

    // Convert row-major Xt (m*k, lda=k) to column-major
    std::vector<T> Xt_row_cm(m * k);
    for (da_int i = 0; i < m; ++i)
        for (da_int j = 0; j < k; ++j)
            Xt_row_cm[i + j * m] = Xt_row_rm[i * k + j];

    for (da_int i = 0; i < m * k; ++i)
        EXPECT_NEAR(Xt_col[i], Xt_row_cm[i], cross_tol);

    // inverse_transform: row-major Xt -> row-major X_inv (m*p, lda=p)
    std::vector<T> Xinv_col(m * p, T(0));
    ASSERT_EQ(
        da_nmf_inverse_transform<T>(h_col, m, k, Xt_col.data(), m, Xinv_col.data(), m),
        da_status_success);

    std::vector<T> Xinv_row_rm(m * p, T(0));
    ASSERT_EQ(da_nmf_inverse_transform<T>(h_row, m, k, Xt_row_rm.data(), k,
                                          Xinv_row_rm.data(), p),
              da_status_success);

    // Convert row-major Xinv (m*p, lda=p) to column-major
    std::vector<T> Xinv_row_cm(m * p);
    for (da_int i = 0; i < m; ++i)
        for (da_int j = 0; j < p; ++j)
            Xinv_row_cm[i + j * m] = Xinv_row_rm[i * p + j];

    for (da_int i = 0; i < m * p; ++i)
        EXPECT_NEAR(Xinv_col[i], Xinv_row_cm[i], cross_tol);

    da_handle_destroy(&h_col);
    da_handle_destroy(&h_row);
}
