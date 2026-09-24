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

#ifndef NMF_TEST_DATA_HPP
#define NMF_TEST_DATA_HPP

#include "../utest_utils.hpp"
#include "gtest/gtest.h"

template <typename T> struct NMFParamType {
    std::string test_name;

    da_int n_samples = 0, n_features = 0, n_components = 0;
    std::vector<T> A;
    da_int lda = 0;
    std::string order = "column-major";

    std::string init = "nndsvda";
    da_int random_state = 42;
    da_int max_iter = 100;
    /* Tolerance used to stop the MU iterations. */
    T nmf_tol = std::sqrt(std::numeric_limits<T>::epsilon());

    /* Tolerance for convergence checks: err/||A||_F and array comparisons. */
    T convergence_tol = T(500) * std::sqrt(std::numeric_limits<T>::epsilon());

    std::vector<T> expected_rinfo;

    std::vector<T> expected_W;
    std::vector<T> expected_H;
    std::optional<T> expected_reconstruction_err;

    /* da_nmf_set_data is called before compute when W_init/H_init are non-empty. */
    std::vector<T> W_init;
    std::vector<T> H_init;

    /* All transform arrays follow the same storage order as A. */
    da_int m_test = 0;
    std::vector<T> X_test;
    da_int ldx_test = 0;
    da_int ldxt = 0;
    da_int ldyinv = 0;
    std::vector<T> expected_Xinv;
    bool check_xinv_roundtrip = false;
};

template <typename T> void Get1by1Data(std::vector<NMFParamType<T>> &params) {
    NMFParamType<T> param;
    param.test_name = "1 by 1 data matrix";
    param.n_samples = 1;
    param.n_features = 1;
    param.n_components = 1;
    std::vector<double> A{4.0};
    param.A = convert_vector<double, T>(A);
    param.lda = 1;
    param.order = "column-major";
    param.init = "random";
    param.max_iter = 100;
    param.random_state = 42;

    std::vector<double> expected_rinfo{1.0, 1.0};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);

    /* Transform roundtrip: inverse_transform(transform(A)) = A. */
    param.m_test = 1;
    param.X_test = param.A;
    param.ldx_test = 1;
    param.ldxt = 1;
    param.ldyinv = 1;
    param.expected_Xinv = param.A;
    param.check_xinv_roundtrip = true;

    params.push_back(param);
}

template <typename T> void Get1by5Data(std::vector<NMFParamType<T>> &params) {
    NMFParamType<T> param;
    param.test_name = "1 by 5 data matrix";
    param.n_samples = 1;
    param.n_features = 5;
    param.n_components = 1;
    std::vector<double> A{2.0, 0.5, 1.0, 1.5, 3.0};
    param.A = convert_vector<double, T>(A);
    param.lda = 1;
    param.order = "column-major";
    param.init = "random";
    param.max_iter = 200;
    param.random_state = 42;

    std::vector<double> expected_rinfo{1.0, 5.0};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);

    /* Transform roundtrip: inverse_transform(transform(A)) = A. */
    param.m_test = 1;
    param.X_test = param.A;
    param.ldx_test = 1;
    param.ldxt = 1;
    param.ldyinv = 1;
    param.expected_Xinv = param.A;
    param.check_xinv_roundtrip = true;

    params.push_back(param);
}

/* Test data: 5 x 1 data matrix.
 *
 * Five samples with a single non-negative feature, n_components = 1.  With one
 * feature the NMF problem is rank-1 and always has an exact factorization, so
 * the transform roundtrip inverse_transform(transform(A)) recovers A exactly.
 * Exact W/H values are not checked due to the scaling ambiguity. */
template <typename T> void Get5by1Data(std::vector<NMFParamType<T>> &params) {
    NMFParamType<T> param;
    param.test_name = "5 by 1 data matrix";
    param.n_samples = 5;
    param.n_features = 1;
    param.n_components = 1;
    std::vector<double> A{1.0, 2.0, 3.0, 4.0, 5.0};
    param.A = convert_vector<double, T>(A);
    param.lda = 5;
    param.order = "column-major";
    param.init = "random";
    param.max_iter = 200;
    param.random_state = 42;

    std::vector<double> expected_rinfo{5.0, 1.0};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);

    /* Transform roundtrip: inverse_transform(transform(A)) = A. */
    param.m_test = 5;
    param.X_test = param.A;
    param.ldx_test = 5;
    param.ldxt = 5;
    param.ldyinv = 5;
    param.expected_Xinv = param.A;
    param.check_xinv_roundtrip = true;

    params.push_back(param);
}

/* Test data: 4 x 4 diagonal data matrix.
 *
 * n_components = 4 admits an exact non-negative factorization.
 * nndsvdar is used because plain nndsvd leaves zeros frozen under MU updates.
 * Exact W/H values are not checked due to scaling ambiguity. */
template <typename T> void GetDiagonalData(std::vector<NMFParamType<T>> &params) {
    NMFParamType<T> param;
    param.test_name = "Diagonal data matrix";
    param.n_samples = 4;
    param.n_features = 4;
    param.n_components = 4;
    std::vector<double> A{1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0,
                          0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0};
    param.A = convert_vector<double, T>(A);
    param.lda = 4;
    param.order = "column-major";
    param.init = "nndsvdar";
    param.max_iter = 2000;
    param.random_state = 42;

    std::vector<double> expected_rinfo{4.0, 4.0};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);

    param.m_test = 4;
    param.X_test = param.A;
    param.ldx_test = 4;
    param.ldxt = 4;
    param.ldyinv = 4;
    param.expected_Xinv = param.A;
    param.check_xinv_roundtrip = true;

    params.push_back(param);
}

/* Test data: 4 x 4 identity data matrix.
 *
 * n_components = 4 admits an exact non-negative factorization (W = H = I).
 * nndsvdar is used for the same reason as GetDiagonalData.
 * Exact W/H values are not checked due to scaling ambiguity. */
template <typename T> void GetIdentityData(std::vector<NMFParamType<T>> &params) {
    NMFParamType<T> param;
    param.test_name = "Identity data matrix";
    param.n_samples = 4;
    param.n_features = 4;
    param.n_components = 4;
    std::vector<double> A{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                          0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    param.A = convert_vector<double, T>(A);
    param.lda = 4;
    param.order = "column-major";
    param.init = "nndsvdar";
    param.max_iter = 2000;
    param.random_state = 42;

    std::vector<double> expected_rinfo{4.0, 4.0};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);

    param.m_test = 4;
    param.X_test = param.A;
    param.ldx_test = 4;
    param.ldxt = 4;
    param.ldyinv = 4;
    param.expected_Xinv = param.A;
    param.check_xinv_roundtrip = true;

    params.push_back(param);
}

/* Test data: 4 x 4 zero data matrix.
 *
 * The only valid NMF factorization is W = 0, H = 0, so the reconstruction error
 * is exactly zero.  random init is used so that at least one MU iteration is
 * attempted; the numerator A*H^T = 0 drives W to zero in the first update. */
template <typename T> void GetZeroData(std::vector<NMFParamType<T>> &params) {
    NMFParamType<T> param;
    param.test_name = "Zero data matrix";
    param.n_samples = 4;
    param.n_features = 4;
    param.n_components = 2;
    std::vector<double> A(16, 0.0);
    param.A = convert_vector<double, T>(A);
    param.lda = 4;
    param.order = "column-major";
    param.init = "random";
    param.max_iter = 100;
    param.random_state = 42;

    std::vector<double> expected_rinfo{4.0, 4.0};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);

    param.expected_reconstruction_err = T(0);

    /* H = 0 after fitting on zeros, so any transform query returns Xt = 0. */
    param.m_test = 2;
    param.X_test = convert_vector<double, T>({1, 2, 3, 4, 5, 6, 7, 8});
    param.ldx_test = 2;
    param.ldxt = 2;
    param.ldyinv = 2;

    params.push_back(param);

    /* transform() on an all-zero query exercises the mean_x==0 branch. */
    NMFParamType<T> param2;
    param2.test_name = "Zero query to transform";
    param2.n_samples = 4;
    param2.n_features = 3;
    param2.n_components = 2;
    std::vector<double> A2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    param2.A = convert_vector<double, T>(A2);
    param2.lda = 4;
    param2.order = "column-major";
    param2.m_test = 3;
    param2.X_test = std::vector<T>(9, T(0));
    param2.ldx_test = 3;
    param2.ldxt = 3;
    param2.ldyinv = 3;
    std::vector<double> expected_rinfo2{4.0, 3.0};
    param2.expected_rinfo = convert_vector<double, T>(expected_rinfo2);
    /* factorization accuracy is not the focus here; only the transform output matters. */
    param2.convergence_tol = T(0.05);
    params.push_back(param2);
}

/* ============================================================
 * Square, tall-thin, short-fat, and subarray test data.
 *
 * Every matrix is generated as A = W_true * H_true using a block-dominant
 * design so that MU + nndsvda converges accurately within the iteration budget:
 *
 *   in-block  W[i,k] = 10*(k+1) + stride*(i - block_start_k)   (~10-52)
 *   out-block W[i,k] = (i+k)%3 + 1                              (1-3)
 *   in-block  H[k,j] = 10*(k+1) + stride*(j - block_start_k)   (~10-52)
 *   out-block H[k,j] = (k+j)%4 + 1                              (1-4)
 *
 * Different bases per component (10, 20, 30, 40) make block-swap bugs visible.
 * Different positions within a block have different values, making
 * index-within-block bugs visible.  The ~10:4 contrast gives nndsvda
 * a near-optimal starting point.
 *
 * Column-major convention for W_true and H_true:
 *   W_true[i + n_samples * k]     (n_samples x n_components)
 *   H_true[k + n_components * j]  (n_components x n_features)
 * ============================================================ */

/* Square 20x20, n_components=4, column-major, nndsvda init. */
template <typename T> void GetSquareData1(std::vector<NMFParamType<T>> &params) {
    const da_int n_samples = 20, n_features = 20, n_components = 4;
    /* W: block-dominant (5 rows/comp), in-block = 10*(k+1) + 2*(i-start), out-of-block = (i+k)%3+1
     * H: block-dominant (5 cols/comp), in-block = 10*(k+1) + 3*(j-start), out-of-block = (k+j)%4+1 */
    const std::vector<double> W_true{
        10, 12, 14, 16, 18, 3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,
        2,  3,  1,  2,  3,  20, 22, 24, 26, 28, 3,  1,  2,  3,  1,  2,  3,  1,  2,  3,
        3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  30, 32, 34, 36, 38, 3,  1,  2,  3,  1,
        1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  40, 42, 44, 46, 48};
    const std::vector<double> H_true{
        10, 2,  3,  4,  13, 3,  4,  1,  16, 4,  1,  2,  19, 1,  2,  3,  22, 2,  3,  4,
        2,  20, 4,  1,  3,  23, 1,  2,  4,  26, 2,  3,  1,  29, 3,  4,  2,  32, 4,  1,
        3,  4,  30, 2,  4,  1,  33, 3,  1,  2,  36, 4,  2,  3,  39, 1,  3,  4,  42, 2,
        4,  1,  2,  40, 1,  2,  3,  43, 2,  3,  4,  46, 3,  4,  1,  49, 4,  1,  2,  52};
    std::vector<double> A_col(n_samples * n_features, 0.0);
    for (da_int j = 0; j < n_features; ++j)
        for (da_int l = 0; l < n_components; ++l)
            for (da_int i = 0; i < n_samples; ++i)
                A_col[i + n_samples * j] +=
                    W_true[i + n_samples * l] * H_true[l + n_components * j];

    NMFParamType<T> param;
    param.test_name = "Square data matrix 1 (20x20), column-major";
    param.n_samples = n_samples;
    param.n_features = n_features;
    param.n_components = n_components;
    param.A = convert_vector<double, T>(A_col);
    param.lda = n_samples;
    param.order = "column-major";
    param.init = "nndsvda";
    param.max_iter = 2000;
    param.random_state = 42;

    std::vector<double> expected_rinfo{20.0, 20.0};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);

    param.convergence_tol = std::is_same<T, float>::value ? T(0.04) : T(2e-3);

    const da_int m_test_sq1 = 3;
    std::vector<double> X_test_sq1(n_features * m_test_sq1, 0.0);
    for (da_int j = 0; j < n_features; ++j)
        for (da_int i = 0; i < m_test_sq1; ++i)
            X_test_sq1[i + m_test_sq1 * j] = A_col[i + n_samples * j];
    param.m_test = m_test_sq1;
    param.X_test = convert_vector<double, T>(X_test_sq1);
    param.ldx_test = m_test_sq1;
    param.ldxt = m_test_sq1;
    param.ldyinv = m_test_sq1;

    params.push_back(param);
}

/* Square 20x20, n_components=4, column-major, nndsvda init. */
template <typename T> void GetSquareData2(std::vector<NMFParamType<T>> &params) {
    const da_int n_samples = 20, n_features = 20, n_components = 4;
    /* W: block-dominant (5 rows/comp), in-block = 10*(k+1) + 3*(i-start), out-of-block = (i+k)%3+1
     * H: block-dominant (5 cols/comp), in-block = 10*(k+1) + 2*(j-start), out-of-block = (k+j)%4+1 */
    const std::vector<double> W_true{
        10, 13, 16, 19, 22, 3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,
        2,  3,  1,  2,  3,  20, 23, 26, 29, 32, 3,  1,  2,  3,  1,  2,  3,  1,  2,  3,
        3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  30, 33, 36, 39, 42, 3,  1,  2,  3,  1,
        1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  40, 43, 46, 49, 52};
    const std::vector<double> H_true{
        10, 2,  3,  4,  12, 3,  4,  1,  14, 4,  1,  2,  16, 1,  2,  3,  18, 2,  3,  4,
        2,  20, 4,  1,  3,  22, 1,  2,  4,  24, 2,  3,  1,  26, 3,  4,  2,  28, 4,  1,
        3,  4,  30, 2,  4,  1,  32, 3,  1,  2,  34, 4,  2,  3,  36, 1,  3,  4,  38, 2,
        4,  1,  2,  40, 1,  2,  3,  42, 2,  3,  4,  44, 3,  4,  1,  46, 4,  1,  2,  48};
    std::vector<double> A_col(n_samples * n_features, 0.0);
    for (da_int j = 0; j < n_features; ++j)
        for (da_int l = 0; l < n_components; ++l)
            for (da_int i = 0; i < n_samples; ++i)
                A_col[i + n_samples * j] +=
                    W_true[i + n_samples * l] * H_true[l + n_components * j];

    NMFParamType<T> param;
    param.test_name = "Square data matrix 2 (20x20), column-major";
    param.n_samples = n_samples;
    param.n_features = n_features;
    param.n_components = n_components;
    param.A = convert_vector<double, T>(A_col);
    param.lda = n_samples;
    param.order = "column-major";
    param.init = "nndsvda";
    param.max_iter = 2000;
    param.random_state = 42;

    std::vector<double> expected_rinfo{20.0, 20.0};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);

    param.convergence_tol = std::is_same<T, float>::value ? T(0.04) : T(2e-3);

    const da_int m_test_sq2 = 3;
    std::vector<double> X_test_sq2(n_features * m_test_sq2, 0.0);
    for (da_int j = 0; j < n_features; ++j)
        for (da_int i = 0; i < m_test_sq2; ++i)
            X_test_sq2[i + m_test_sq2 * j] = A_col[i + n_samples * j];
    param.m_test = m_test_sq2;
    param.X_test = convert_vector<double, T>(X_test_sq2);
    param.ldx_test = m_test_sq2;
    param.ldxt = m_test_sq2;
    param.ldyinv = m_test_sq2;

    params.push_back(param);
}

/* Tall-thin 20x5, n_components=3, column-major, nndsvda init. */
template <typename T> void GetTallThinData1(std::vector<NMFParamType<T>> &params) {
    const da_int n_samples = 20, n_features = 5, n_components = 3;
    /* W: block-dominant (6-7 rows/comp), in-block = 10*(k+1) + 2*(i-start), out-of-block = (i+k)%3+1
     * H: block-dominant (1-2 cols/comp), in-block = 10*(k+1) + 4*(j-start), out-of-block = (k+j)%4+1 */
    const std::vector<double> W_true{
        10, 12, 14, 16, 18, 20, 1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,
        2,  3,  1,  2,  3,  1,  20, 22, 24, 26, 28, 30, 32, 3,  1,  2,  3,  1,  2,  3,
        3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  30, 32, 34, 36, 38, 40, 42};
    const std::vector<double> H_true{10, 2, 3, 2, 20, 4, 3, 24, 1, 4, 1, 30, 1, 2, 34};
    std::vector<double> A_col(n_samples * n_features, 0.0);
    for (da_int j = 0; j < n_features; ++j)
        for (da_int l = 0; l < n_components; ++l)
            for (da_int i = 0; i < n_samples; ++i)
                A_col[i + n_samples * j] +=
                    W_true[i + n_samples * l] * H_true[l + n_components * j];

    NMFParamType<T> param;
    param.test_name = "Tall thin data matrix 1 (20x5), column-major";
    param.n_samples = n_samples;
    param.n_features = n_features;
    param.n_components = n_components;
    param.A = convert_vector<double, T>(A_col);
    param.lda = n_samples;
    param.order = "column-major";
    param.init = "nndsvda";
    param.max_iter = 2000;
    param.random_state = 42;

    std::vector<double> expected_rinfo{20.0, 5.0};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);

    param.convergence_tol = std::is_same<T, float>::value ? T(0.04) : T(1e-3);

    const da_int m_test_tt1 = 3;
    std::vector<double> X_test_tt1(n_features * m_test_tt1, 0.0);
    for (da_int j = 0; j < n_features; ++j)
        for (da_int i = 0; i < m_test_tt1; ++i)
            X_test_tt1[i + m_test_tt1 * j] = A_col[i + n_samples * j];
    param.m_test = m_test_tt1;
    param.X_test = convert_vector<double, T>(X_test_tt1);
    param.ldx_test = m_test_tt1;
    param.ldxt = m_test_tt1;
    param.ldyinv = m_test_tt1;

    params.push_back(param);
}

/* Tall-thin 20x5, n_components=3, column-major, nndsvda init. */
template <typename T> void GetTallThinData2(std::vector<NMFParamType<T>> &params) {
    const da_int n_samples = 20, n_features = 5, n_components = 3;
    /* W: block-dominant (6-7 rows/comp), in-block = 10*(k+1) + 3*(i-start), out-of-block = (i+k)%3+1
     * H: block-dominant (1-2 cols/comp), in-block = 10*(k+1) + 4*(j-start), out-of-block = (k+j)%4+1 */
    const std::vector<double> W_true{
        10, 13, 16, 19, 22, 25, 1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,
        2,  3,  1,  2,  3,  1,  20, 23, 26, 29, 32, 35, 38, 3,  1,  2,  3,  1,  2,  3,
        3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  30, 33, 36, 39, 42, 45, 48};
    const std::vector<double> H_true{10, 2, 3, 2, 20, 4, 3, 24, 1, 4, 1, 30, 1, 2, 34};
    std::vector<double> A_col(n_samples * n_features, 0.0);
    for (da_int j = 0; j < n_features; ++j)
        for (da_int l = 0; l < n_components; ++l)
            for (da_int i = 0; i < n_samples; ++i)
                A_col[i + n_samples * j] +=
                    W_true[i + n_samples * l] * H_true[l + n_components * j];

    NMFParamType<T> param;
    param.test_name = "Tall thin data matrix 2 (20x5), column-major";
    param.n_samples = n_samples;
    param.n_features = n_features;
    param.n_components = n_components;
    param.A = convert_vector<double, T>(A_col);
    param.lda = n_samples;
    param.order = "column-major";
    param.init = "nndsvda";
    param.max_iter = 2000;
    param.random_state = 42;

    std::vector<double> expected_rinfo{20.0, 5.0};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);

    param.convergence_tol = std::is_same<T, float>::value ? T(0.04) : T(1e-3);

    const da_int m_test_tt2 = 3;
    std::vector<double> X_test_tt2(n_features * m_test_tt2, 0.0);
    for (da_int j = 0; j < n_features; ++j)
        for (da_int i = 0; i < m_test_tt2; ++i)
            X_test_tt2[i + m_test_tt2 * j] = A_col[i + n_samples * j];
    param.m_test = m_test_tt2;
    param.X_test = convert_vector<double, T>(X_test_tt2);
    param.ldx_test = m_test_tt2;
    param.ldxt = m_test_tt2;
    param.ldyinv = m_test_tt2;

    params.push_back(param);
}

/* Tall-thin 20x8, n_components=4, column-major, nndsvda init. */
template <typename T> void GetTallThinData3(std::vector<NMFParamType<T>> &params) {
    const da_int n_samples = 20, n_features = 8, n_components = 4;
    /* W: block-dominant (5 rows/comp), in-block = 10*(k+1) + 2*(i-start), out-of-block = (i+k)%3+1
     * H: block-dominant (2 cols/comp), in-block = 10*(k+1) + 3*(j-start), out-of-block = (k+j)%4+1 */
    const std::vector<double> W_true{
        10, 12, 14, 16, 18, 3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,
        2,  3,  1,  2,  3,  20, 22, 24, 26, 28, 3,  1,  2,  3,  1,  2,  3,  1,  2,  3,
        3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  30, 32, 34, 36, 38, 3,  1,  2,  3,  1,
        1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  40, 42, 44, 46, 48};
    const std::vector<double> H_true{10, 2, 3,  4, 13, 3,  4, 1,  3, 20, 1,
                                     2,  4, 23, 2, 3,  1,  2, 30, 4, 2,  3,
                                     33, 1, 3,  4, 1,  40, 4, 1,  2, 43};
    std::vector<double> A_col(n_samples * n_features, 0.0);
    for (da_int j = 0; j < n_features; ++j)
        for (da_int l = 0; l < n_components; ++l)
            for (da_int i = 0; i < n_samples; ++i)
                A_col[i + n_samples * j] +=
                    W_true[i + n_samples * l] * H_true[l + n_components * j];

    NMFParamType<T> param;
    param.test_name = "Tall thin data matrix 3 (20x8), column-major";
    param.n_samples = n_samples;
    param.n_features = n_features;
    param.n_components = n_components;
    param.A = convert_vector<double, T>(A_col);
    param.lda = n_samples;
    param.order = "column-major";
    param.init = "nndsvda";
    param.max_iter = 2000;
    param.random_state = 42;

    std::vector<double> expected_rinfo{20.0, 8.0};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);

    param.convergence_tol = std::is_same<T, float>::value ? T(0.04) : T(1e-3);

    const da_int m_test_tt3 = 3;
    std::vector<double> X_test_tt3(n_features * m_test_tt3, 0.0);
    for (da_int j = 0; j < n_features; ++j)
        for (da_int i = 0; i < m_test_tt3; ++i)
            X_test_tt3[i + m_test_tt3 * j] = A_col[i + n_samples * j];
    param.m_test = m_test_tt3;
    param.X_test = convert_vector<double, T>(X_test_tt3);
    param.ldx_test = m_test_tt3;
    param.ldxt = m_test_tt3;
    param.ldyinv = m_test_tt3;

    params.push_back(param);
}

/* Short-fat 5x20, n_components=3, column-major, nndsvda init. */
template <typename T> void GetShortFatData(std::vector<NMFParamType<T>> &params) {
    const da_int n_samples = 5, n_features = 20, n_components = 3;
    /* W: block-dominant (1-2 rows/comp), in-block = 10*(k+1) + 2*(i-start), out-of-block = (i+k)%3+1
     * H: block-dominant (6-7 cols/comp), in-block = 10*(k+1) + 3*(j-start), out-of-block = (k+j)%4+1 */
    const std::vector<double> W_true{10, 2, 3, 1, 2, 2, 20, 22, 2, 3, 3, 1, 2, 30, 32};
    const std::vector<double> H_true{
        10, 2,  3,  13, 3,  4,  16, 4,  1,  19, 1,  2,  22, 2,  3,  25, 3,  4,  3, 20,
        1,  4,  23, 2,  1,  26, 3,  2,  29, 4,  3,  32, 1,  4,  35, 2,  1,  38, 3, 2,
        3,  30, 3,  4,  33, 4,  1,  36, 1,  2,  39, 2,  3,  42, 3,  4,  45, 4,  1, 48};
    std::vector<double> A_col(n_samples * n_features, 0.0);
    for (da_int j = 0; j < n_features; ++j)
        for (da_int l = 0; l < n_components; ++l)
            for (da_int i = 0; i < n_samples; ++i)
                A_col[i + n_samples * j] +=
                    W_true[i + n_samples * l] * H_true[l + n_components * j];

    NMFParamType<T> param;
    param.test_name = "Short fat data matrix (5x20)";
    param.n_samples = n_samples;
    param.n_features = n_features;
    param.n_components = n_components;
    param.A = convert_vector<double, T>(A_col);
    param.lda = n_samples;
    param.order = "column-major";
    param.init = "nndsvda";
    param.max_iter = 2000;
    param.random_state = 42;

    std::vector<double> expected_rinfo{5.0, 20.0};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);

    param.convergence_tol = std::is_same<T, float>::value ? T(0.04) : T(1e-3);

    const da_int m_test_sf = 3;
    std::vector<double> X_test_sf(n_features * m_test_sf, 0.0);
    for (da_int j = 0; j < n_features; ++j)
        for (da_int i = 0; i < m_test_sf; ++i)
            X_test_sf[i + m_test_sf * j] = A_col[i + n_samples * j];
    param.m_test = m_test_sf;
    param.X_test = convert_vector<double, T>(X_test_sf);
    param.ldx_test = m_test_sf;
    param.ldxt = m_test_sf;
    param.ldyinv = m_test_sf;

    params.push_back(param);
}

/* Subarray: 20x5, n_components=3, column-major, lda=25 (5 padding rows per column). */
template <typename T> void GetSubarrayData1(std::vector<NMFParamType<T>> &params) {
    const da_int n_samples = 20, n_features = 5, n_components = 3;
    const da_int lda = 25;
    /* W: block-dominant (6-7 rows/comp), in-block = 10*(k+1) + 2*(i-start), out-of-block = (i+k)%3+1
     * H: block-dominant (1-2 cols/comp), in-block = 10*(k+1) + 4*(j-start), out-of-block = (k+j)%4+1 */
    const std::vector<double> W_true{
        10, 12, 14, 16, 18, 20, 1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,
        2,  3,  1,  2,  3,  1,  20, 22, 24, 26, 28, 30, 32, 3,  1,  2,  3,  1,  2,  3,
        3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  30, 32, 34, 36, 38, 40, 42};
    const std::vector<double> H_true{10, 2, 3, 2, 20, 4, 3, 24, 1, 4, 1, 30, 1, 2, 34};
    std::vector<double> A_col(n_samples * n_features, 0.0);
    for (da_int j = 0; j < n_features; ++j)
        for (da_int l = 0; l < n_components; ++l)
            for (da_int i = 0; i < n_samples; ++i)
                A_col[i + n_samples * j] +=
                    W_true[i + n_samples * l] * H_true[l + n_components * j];
    /* Embed into padded column-major array: lda rows per column. */
    std::vector<double> A_pad(n_features * lda, 0.0);
    for (da_int j = 0; j < n_features; ++j)
        for (da_int i = 0; i < n_samples; ++i)
            A_pad[i + lda * j] = A_col[i + n_samples * j];

    NMFParamType<T> param;
    param.test_name = "Subarray data matrix 1 (20x5, lda=25), column-major";
    param.n_samples = n_samples;
    param.n_features = n_features;
    param.n_components = n_components;
    param.A = convert_vector<double, T>(A_pad);
    param.lda = lda;
    param.order = "column-major";
    param.init = "nndsvda";
    param.max_iter = 2000;
    param.random_state = 42;

    std::vector<double> expected_rinfo{20.0, 5.0};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);

    param.convergence_tol = std::is_same<T, float>::value ? T(0.04) : T(1e-3);

    /* X_test: first 3 rows of A, col-major with ldx_test=5 (2 padding rows). */
    const da_int m_test1 = 3, ldx_test1 = m_test1 + 2;
    std::vector<double> X_test1(n_features * ldx_test1, 0.0);
    for (da_int j = 0; j < n_features; ++j)
        for (da_int i = 0; i < m_test1; ++i)
            X_test1[i + ldx_test1 * j] = A_col[i + n_samples * j];
    param.m_test = m_test1;
    param.X_test = convert_vector<double, T>(X_test1);
    param.ldx_test = ldx_test1;
    param.ldxt = m_test1;
    param.ldyinv = m_test1;

    params.push_back(param);
}

/* Subarray: 5x20, n_components=3, row-major, lda=25 (5 padding columns per row). */
template <typename T> void GetSubarrayData2(std::vector<NMFParamType<T>> &params) {
    const da_int n_samples = 5, n_features = 20, n_components = 3;
    const da_int lda = 25;
    /* W: block-dominant (1-2 rows/comp), in-block = 10*(k+1) + 2*(i-start), out-of-block = (i+k)%3+1
     * H: block-dominant (6-7 cols/comp), in-block = 10*(k+1) + 3*(j-start), out-of-block = (k+j)%4+1 */
    const std::vector<double> W_true{10, 2, 3, 1, 2, 2, 20, 22, 2, 3, 3, 1, 2, 30, 32};
    const std::vector<double> H_true{
        10, 2,  3,  13, 3,  4,  16, 4,  1,  19, 1,  2,  22, 2,  3,  25, 3,  4,  3, 20,
        1,  4,  23, 2,  1,  26, 3,  2,  29, 4,  3,  32, 1,  4,  35, 2,  1,  38, 3, 2,
        3,  30, 3,  4,  33, 4,  1,  36, 1,  2,  39, 2,  3,  42, 3,  4,  45, 4,  1, 48};
    std::vector<double> A_col(n_samples * n_features, 0.0);
    for (da_int j = 0; j < n_features; ++j)
        for (da_int l = 0; l < n_components; ++l)
            for (da_int i = 0; i < n_samples; ++i)
                A_col[i + n_samples * j] +=
                    W_true[i + n_samples * l] * H_true[l + n_components * j];
    /* Embed into padded row-major array: lda columns per row. */
    std::vector<double> A_pad(n_samples * lda, 0.0);
    for (da_int i = 0; i < n_samples; ++i)
        for (da_int j = 0; j < n_features; ++j)
            A_pad[i * lda + j] = A_col[i + n_samples * j];

    NMFParamType<T> param;
    param.test_name = "Subarray data matrix 2 (5x20, lda=25), row-major";
    param.n_samples = n_samples;
    param.n_features = n_features;
    param.n_components = n_components;
    param.A = convert_vector<double, T>(A_pad);
    param.lda = lda;
    param.order = "row-major";
    param.init = "nndsvda";
    param.max_iter = 2000;
    param.random_state = 42;

    std::vector<double> expected_rinfo{5.0, 20.0};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);

    param.convergence_tol = std::is_same<T, float>::value ? T(0.04) : T(1e-3);

    /* X_test: first 2 rows of A, row-major with ldx_test=23 (3 padding columns, distinct from lda=25). */
    const da_int m_test2 = 2;
    const da_int ldx_test2 = n_features + 3;
    std::vector<double> X_test2(m_test2 * ldx_test2, 0.0);
    for (da_int i = 0; i < m_test2; ++i)
        for (da_int j = 0; j < n_features; ++j)
            X_test2[i * ldx_test2 + j] = A_col[i + n_samples * j];
    param.m_test = m_test2;
    param.X_test = convert_vector<double, T>(X_test2);
    param.ldx_test = ldx_test2;
    param.ldxt = n_components; /* row-major Xt: lda = k */
    param.ldyinv = n_features; /* row-major Xinv: lda = n_features */

    params.push_back(param);
}

/* 4x3, k=2, custom init (W0/H0 supplied), transform roundtrip on a new 2x3 matrix. */
template <typename T> void GetCustomInitData(std::vector<NMFParamType<T>> &params) {
    NMFParamType<T> param;
    param.test_name = "Custom init 4x3 k=2";
    param.n_samples = 4;
    param.n_features = 3;
    param.n_components = 2;
    param.A = convert_vector<double, T>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    param.lda = 4;
    param.order = "column-major";
    param.init = "custom";
    param.W_init = convert_vector<double, T>({0.5, 1.0, 1.5, 2.0, 1.0, 0.5, 1.0, 0.5});
    param.H_init = convert_vector<double, T>({1.0, 0.5, 1.0, 0.5, 1.0, 1.5});
    param.max_iter = 500;
    param.nmf_tol = T(1e-6);
    /* X_test lies in the approximate column space of H so Xinv ≈ X_test. */
    param.convergence_tol = T(0.05);
    param.m_test = 2;
    param.X_test = convert_vector<double, T>({2, 3, 4, 5, 6, 7});
    param.ldx_test = 2;
    param.ldxt = 2;
    param.ldyinv = 2;
    param.expected_Xinv = param.X_test;
    param.check_xinv_roundtrip = true;
    params.push_back(param);
}

template <typename T> void GetNMFData(std::vector<NMFParamType<T>> &params) {
    Get1by1Data(params);
    Get1by5Data(params);
    Get5by1Data(params);
    GetDiagonalData(params);
    GetIdentityData(params);
    GetZeroData(params);
    GetSquareData1(params);
    GetSquareData2(params);
    GetTallThinData1(params);
    GetTallThinData2(params);
    GetTallThinData3(params);
    GetShortFatData(params);
    GetSubarrayData1(params);
    GetSubarrayData2(params);
    GetCustomInitData(params);
}

#endif /* NMF_TEST_DATA_HPP */
