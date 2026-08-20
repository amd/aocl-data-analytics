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

#ifndef AOCLDA_KERNEL_PCA_TEST_DATA_HPP
#define AOCLDA_KERNEL_PCA_TEST_DATA_HPP

#include "../datests_cblas.hh"
#include "../utest_utils.hpp"
#include "aoclda.h"
#include <limits>
#include <string>
#include <vector>

template <typename T> struct KernelPCAParamType {
    std::string test_name;

    // Input
    da_int n = 0;
    da_int p = 0;
    std::vector<T> A;
    da_int lda = 0;
    std::string order = "column-major";

    // Kernel options - defaults match registered option defaults
    std::string kernel = "linear";
    std::string solver = "syevd";
    da_int n_components = 0;
    T gamma = T(-1);
    da_int degree = 3;
    T coef0 = T(1);
    std::string fit_inverse_transform = "no";
    std::string remove_zero_eig = "no";
    T alpha = T(1);
    std::string copy_data =
        "yes"; // passed to "copy data" option; "no" = no copy, "yes" = copy

    // Expected outputs - layout matches `order`
    da_int expected_n_components = 0;
    std::vector<T> expected_eigenvalues;  // length nc, descending
    std::vector<T> expected_eigenvectors; // n x nc matrix
    std::vector<T> expected_scores;       // n x nc matrix
    std::vector<T> expected_rinfo;        // [n, p]
    T expected_gamma = T(-1);             // resolved gamma value
    std::vector<T> expected_X_fit;        // tight-packed training data, n x p

    // Transform test data - tested if X_transform_in.size() > 0
    da_int m = 0;                        // rows for transform input
    da_int p_transform = 0;              // cols (= p, or = n for precomputed)
    std::vector<T> X_transform_in;       // m x p_transform input matrix
    da_int ldx = 0;                      // leading dim of X_transform_in
    std::vector<T> expected_X_transform; // m x nc expected output
    da_int ldx_transform = 0;            // leading dim of expected_X_transform

    // Inverse transform test data - tested if Y_inv.size() > 0
    da_int k = 0;                            // rows for inverse transform input
    std::vector<T> Y_inv;                    // k x nc input matrix
    da_int ldy = 0;                          // leading dim of Y_inv
    std::vector<T> expected_Y_inv_transform; // k x p expected output
    da_int ldy_inv_transform = 0;            // leading dim of expected output

    T epsilon = (T)2000 * std::numeric_limits<T>::epsilon();
};

template <typename T>
void sign_correct_columns(da_int n, da_int nc, std::vector<T> &result,
                          const std::vector<T> &reference, const std::string &order,
                          da_int ld = 0) {
    if (order == "column-major") {
        da_int s = (ld > 0) ? ld : n;
        for (da_int j = 0; j < nc; j++) {
            da_int max_idx = 0;
            T max_abs = T(0);
            for (da_int i = 0; i < n; i++) {
                T v = std::abs(result[j * s + i]);
                if (v > max_abs) {
                    max_abs = v;
                    max_idx = i;
                }
            }
            if (result[j * s + max_idx] * reference[j * s + max_idx] < T(0)) {
                for (da_int i = 0; i < n; i++)
                    result[j * s + i] = -result[j * s + i];
            }
        }
    } else {
        da_int s = (ld > 0) ? ld : nc;
        for (da_int j = 0; j < nc; j++) {
            da_int max_idx = 0;
            T max_abs = T(0);
            for (da_int i = 0; i < n; i++) {
                T v = std::abs(result[i * s + j]);
                if (v > max_abs) {
                    max_abs = v;
                    max_idx = i;
                }
            }
            if (result[max_idx * s + j] * reference[max_idx * s + j] < T(0)) {
                for (da_int i = 0; i < n; i++)
                    result[i * s + j] = -result[i * s + j];
            }
        }
    }
}

/* ============================================================================
 * pad_array
 *
 * Increases the leading dimension of a contiguous 2-D array from
 * stride_in to stride_out (>= stride_in). The array has nblocks
 * contiguous blocks of stride_in elements; each block is copied into a
 * block of stride_out elements with zeros filling the padding positions.
 *
 * For column-major: stride_in = n (rows), nblocks = p (cols).
 * For row-major:    stride_in = p (cols), nblocks = n (rows).
 * ============================================================================ */
template <typename T>
std::vector<T> pad_array(const T *data, da_int stride_in, da_int nblocks,
                         da_int stride_out) {
    std::vector<T> padded(stride_out * nblocks, T(0));
    for (da_int j = 0; j < nblocks; j++)
        for (da_int i = 0; i < stride_in; i++)
            padded[j * stride_out + i] = data[j * stride_in + i];
    return padded;
}

/* ============================================================================
 * push_variants
 *
 * Takes a column-major, unpadded KernelPCAParamType<T> (the "base") and
 * generates 4 variants:
 *   1. Column-major, unpadded  ({base_name}_col)
 *   2. Column-major, padded    ({base_name}_col_pad)
 *   3. Row-major, unpadded     ({base_name}_row)
 *   4. Row-major, padded       ({base_name}_row_pad)
 *
 * Expected eigenvectors/eigenvalues/scores from get_result are NOT padded
 * (get_result always returns tight n*nc arrays). Only input data A,
 * transform I/O, and inverse transform I/O are padded.
 * ============================================================================ */
template <typename T>
void push_variants(std::vector<KernelPCAParamType<T>> &params,
                   const KernelPCAParamType<T> &base, da_int pad_amount) {

    da_int n = base.n;
    da_int p = base.p;
    da_int nc = base.expected_n_components;
    da_int m = base.m;
    da_int p_transform = base.p_transform;
    da_int k = base.k;

    // --- Variant 1: column-major, unpadded ---
    {
        KernelPCAParamType<T> v = base;
        v.test_name = base.test_name + "_col";
        v.expected_X_fit = base.A; // base.A is already tight col-major
        params.push_back(v);
    }

    // --- Variant 2: column-major, padded ---
    {
        KernelPCAParamType<T> v = base;
        v.test_name = base.test_name + "_col_pad";
        v.expected_X_fit = base.A; // get_result output is always tight; no padding
        v.lda = n + pad_amount;
        v.A = pad_array(base.A.data(), n, p, v.lda);
        if (base.X_transform_in.size() > 0) {
            v.ldx = m + pad_amount;
            v.ldx_transform = m + pad_amount;
            v.X_transform_in =
                pad_array(base.X_transform_in.data(), m, p_transform, v.ldx);
            v.expected_X_transform =
                pad_array(base.expected_X_transform.data(), m, nc, v.ldx_transform);
        }
        if (base.Y_inv.size() > 0) {
            v.ldy = k + pad_amount;
            v.ldy_inv_transform = k + pad_amount;
            v.Y_inv = pad_array(base.Y_inv.data(), k, nc, v.ldy);
            v.expected_Y_inv_transform = pad_array(base.expected_Y_inv_transform.data(),
                                                   k, p, v.ldy_inv_transform);
        }
        params.push_back(v);
    }

    // --- Variant 3: row-major, unpadded ---
    {
        KernelPCAParamType<T> v = base;
        v.test_name = base.test_name + "_row";
        v.order = "row-major";
        v.lda = p;
        v.A = base.A;
        datest_blas::imatcopy('T', n, p, T(1), v.A.data(), n, p);
        v.expected_X_fit = base.A; // tight row-major transposition of col-major base
        datest_blas::imatcopy('T', n, p, T(1), v.expected_X_fit.data(), n, p);
        if (base.expected_eigenvectors.size() > 0) {
            v.expected_eigenvectors = base.expected_eigenvectors;
            datest_blas::imatcopy('T', n, nc, T(1), v.expected_eigenvectors.data(), n,
                                  nc);
        }
        if (base.expected_scores.size() > 0) {
            v.expected_scores = base.expected_scores;
            datest_blas::imatcopy('T', n, nc, T(1), v.expected_scores.data(), n, nc);
        }
        if (base.X_transform_in.size() > 0) {
            v.ldx = p_transform;
            v.ldx_transform = nc;
            v.X_transform_in = base.X_transform_in;
            datest_blas::imatcopy('T', m, p_transform, T(1), v.X_transform_in.data(), m,
                                  p_transform);
            v.expected_X_transform = base.expected_X_transform;
            datest_blas::imatcopy('T', m, nc, T(1), v.expected_X_transform.data(), m, nc);
        }
        if (base.Y_inv.size() > 0) {
            v.ldy = nc;
            v.ldy_inv_transform = p;
            v.Y_inv = base.Y_inv;
            datest_blas::imatcopy('T', k, nc, T(1), v.Y_inv.data(), k, nc);
            v.expected_Y_inv_transform = base.expected_Y_inv_transform;
            datest_blas::imatcopy('T', k, p, T(1), v.expected_Y_inv_transform.data(), k,
                                  p);
        }
        params.push_back(v);
    }

    // --- Variant 4: row-major, padded ---
    {
        KernelPCAParamType<T> v = base;
        v.test_name = base.test_name + "_row_pad";
        v.order = "row-major";
        v.lda = p + pad_amount;
        v.A = base.A;
        datest_blas::imatcopy('T', n, p, T(1), v.A.data(), n, p);
        v.A = pad_array(v.A.data(), p, n, v.lda);

        v.expected_X_fit = base.A; // tight row-major transposition of col-major base
        datest_blas::imatcopy('T', n, p, T(1), v.expected_X_fit.data(), n, p);

        if (base.expected_eigenvectors.size() > 0) {
            v.expected_eigenvectors = base.expected_eigenvectors;
            datest_blas::imatcopy('T', n, nc, T(1), v.expected_eigenvectors.data(), n,
                                  nc);
        }
        if (base.expected_scores.size() > 0) {
            v.expected_scores = base.expected_scores;
            datest_blas::imatcopy('T', n, nc, T(1), v.expected_scores.data(), n, nc);
        }

        if (base.X_transform_in.size() > 0) {
            v.ldx = p_transform + pad_amount;
            v.ldx_transform = nc + pad_amount;
            v.X_transform_in = base.X_transform_in;
            datest_blas::imatcopy('T', m, p_transform, T(1), v.X_transform_in.data(), m,
                                  p_transform);
            v.X_transform_in = pad_array(v.X_transform_in.data(), p_transform, m, v.ldx);
            v.expected_X_transform = base.expected_X_transform;
            datest_blas::imatcopy('T', m, nc, T(1), v.expected_X_transform.data(), m, nc);
            v.expected_X_transform =
                pad_array(v.expected_X_transform.data(), nc, m, v.ldx_transform);
        }
        if (base.Y_inv.size() > 0) {
            v.ldy = nc + pad_amount;
            v.ldy_inv_transform = p + pad_amount;
            v.Y_inv = base.Y_inv;
            datest_blas::imatcopy('T', k, nc, T(1), v.Y_inv.data(), k, nc);
            v.Y_inv = pad_array(v.Y_inv.data(), nc, k, v.ldy);
            v.expected_Y_inv_transform = base.expected_Y_inv_transform;
            datest_blas::imatcopy('T', k, p, T(1), v.expected_Y_inv_transform.data(), k,
                                  p);
            v.expected_Y_inv_transform =
                pad_array(v.expected_Y_inv_transform.data(), p, k, v.ldy_inv_transform);
        }
        params.push_back(v);
    }
}

/* ============================================================================
 * Dataset functions
 *
 * Each adds KernelPCAParamType<T> entries to `params`.
 * ============================================================================ */

/* Dataset 1a: 4x3 zero-mean matrix, linear kernel, column-major.
 *
 * A = [[-1.5, -0.5,  1.0],
 *      [-0.5,  0.5, -1.0],
 *      [ 0.5, -0.5,  0.5],
 *      [ 1.5,  0.5, -0.5]]
 *
 * All columns sum to zero, so kernel centering is a no-op for the linear
 * kernel: K_tilde = A * A^T. */
template <typename T>
void add_linear_zero_mean_colmaj(std::vector<KernelPCAParamType<T>> &params) {
    KernelPCAParamType<T> p;
    p.test_name = "linear_zero_mean_colmaj";
    p.n = 4;
    p.p = 3;
    p.order = "column-major";
    p.kernel = "linear";
    p.remove_zero_eig = "yes";
    p.expected_n_components = 3;
    p.lda = 4;

    // Column-major (lda = 4): col 0, col 1, col 2
    p.A = {-1.5, -0.5, 0.5, 1.5, -0.5, 0.5, -0.5, 0.5, 1.0, -1.0, 0.5, -0.5};
    p.expected_rinfo = {T(p.n), T(p.p)};
    p.expected_gamma = T(1) / T(p.p);

    params.push_back(p);
}

/* Dataset 1b: same 4x3 zero-mean matrix, linear kernel, row-major.
 *
 * Numerically identical to dataset 1a but stored row-major (lda = p = 3). */
template <typename T>
void add_linear_zero_mean_rowmaj(std::vector<KernelPCAParamType<T>> &params) {
    KernelPCAParamType<T> p;
    p.test_name = "linear_zero_mean_rowmaj";
    p.n = 4;
    p.p = 3;
    p.order = "row-major";
    p.kernel = "linear";
    p.remove_zero_eig = "yes";
    p.expected_n_components = 3;
    p.lda = 3;

    // Row-major (lda = 3): row 0, row 1, row 2, row 3
    p.A = {-1.5, -0.5, 1.0, -0.5, 0.5, -1.0, 0.5, -0.5, 0.5, 1.5, 0.5, -0.5};
    p.expected_rinfo = {T(p.n), T(p.p)};
    p.expected_gamma = T(1) / T(p.p);

    params.push_back(p);
}

/* ============================================================================ *
 * Each function defines a column-major, unpadded base dataset and calls
 * push_variants to generate 4 layout variants (col, col_pad, row, row_pad).
 * ============================================================================ */

template <typename T> void add_linear_tall(std::vector<KernelPCAParamType<T>> &params) {
    KernelPCAParamType<T> base;
    base.test_name = "linear_tall";
    base.n = 6;
    base.p = 3;
    base.lda = 6;
    base.order = "column-major";
    base.kernel = "linear";
    base.fit_inverse_transform = "yes";
    base.n_components = 3;
    base.expected_n_components = 3;
    base.copy_data = "no";

    // clang-format off
    base.A = convert_vector<double, T>({
    1.0, 3.0, 0.5, -2.0, 1.5, -1.0,
    2.0, -1.0, 1.5, 0.0, -0.5, 1.0,
    -1.0, 0.5, 2.0, 1.0, -1.5, 0.0,
    });
    base.expected_rinfo = {T(base.n), T(base.p)};
    base.expected_gamma = T(1) / T(base.p);
    base.expected_eigenvalues = convert_vector<double, T>({
    1.83055715040013816e+01, 7.30102995247190556e+00, 5.72673187686005924e+00,
    });
    base.expected_eigenvectors = convert_vector<double, T>({
    8.62948669540869462e-02, 6.06456873889861248e-01, -2.05409257584859228e-01, -5.52530436625913945e-01,
    4.03508842196679263e-01, -3.38320888829854005e-01, 5.26237765855806838e-01, -4.89792827467531167e-01,
    -5.16718413116041830e-01, -1.33917212983749284e-01, 3.76485404837801518e-01, 2.37705282873712398e-01,
    5.45314575607418472e-01, -1.48924748101957716e-01, 5.35040989459531158e-01, -5.07508208089862967e-01,
    -3.64874355212676194e-01, -5.90482536624547596e-02,
    });
    base.expected_scores = convert_vector<double, T>({
    3.69212681586973235e-01, 2.59472639079286527e+00, -8.78843730717443372e-01, -2.36400207064011569e+00,
    1.72641301771466504e+00, -1.44750628873694298e+00, 1.42191644626689495e+00, -1.32344069891494143e+00,
    -1.39619475714320584e+00, -3.61849908795781716e-01, 1.01727930538722422e+00, 6.42289613199805709e-01,
    1.30497000988612277e+00, -3.56385724306790885e-01, 1.28038471102070939e+00, -1.21449713789624814e+00,
    -8.73165897681538516e-01, -1.41305961022259535e-01,
    });

    base.m = 3; base.p_transform = 3;
    base.ldx = 3; base.ldx_transform = 3;
    base.X_transform_in = convert_vector<double, T>({
    0.5, -0.5, 1.0, -1.0, 2.0, 0.0, 1.5, -1.0, 0.5,
    });
    base.expected_X_transform = convert_vector<double, T>({
    2.82450111998761681e-02, -9.81122101044141104e-01, 4.94629198892678312e-01, -1.68857555355862798e+00,
    1.75868547965963362e+00, -5.72179687098797740e-01, -1.08429359356415156e+00, 7.45342818834525689e-01,
    -1.97644813707366113e-01,
    });

    base.k = 6; base.ldy = 6; base.ldy_inv_transform = 6;
    base.Y_inv = convert_vector<double, T>({
    3.69212681586973235e-01, 2.59472639079286527e+00, -8.78843730717443372e-01, -2.36400207064011569e+00,
    1.72641301771466504e+00, -1.44750628873694298e+00, 1.42191644626689495e+00, -1.32344069891494143e+00,
    -1.39619475714320584e+00, -3.61849908795781716e-01, 1.01727930538722422e+00, 6.42289613199805709e-01,
    1.30497000988612277e+00, -3.56385724306790885e-01, 1.28038471102070939e+00, -1.21449713789624814e+00,
    -8.73165897681538516e-01, -1.41305961022259535e-01,
    });
    base.expected_Y_inv_transform = convert_vector<double, T>({
    4.48863636363635132e-01, 2.36297928262213919e+00, -6.77952999381563198e-02, -2.33219310451453232e+00,
    9.95439084724799250e-01, -1.40729359925788500e+00, 1.27840909090908750e+00, -1.36359771181199863e+00,
    8.64950525664811343e-01, -3.60138373531232348e-01, -8.92779839208410642e-01, 4.73156307977736668e-01,
    -1.04166666666666607e+00, 2.38945578231292755e-01, 1.62414965986394622e+00, 7.91241496598640293e-01,
    -1.49829931972789132e+00, -1.14370748299320424e-01,
    });
    // clang-format on

    push_variants(params, base, 2);
}

template <typename T> void add_linear_wide(std::vector<KernelPCAParamType<T>> &params) {
    KernelPCAParamType<T> base;
    base.test_name = "linear_wide";
    base.n = 4;
    base.p = 7;
    base.lda = 4;
    base.order = "column-major";
    base.kernel = "linear";
    base.solver = "auto";
    base.n_components = 2;
    base.fit_inverse_transform = "yes";
    base.expected_n_components = 2;

    // clang-format off
    base.A = convert_vector<double, T>({
    1.0, -0.5, 0.5, 0.0,
    0.5, 1.0, -1.0, 0.5,
    -1.0, 0.5, 1.5, -0.5,
    0.0, -1.0, 0.5, 1.0,
    1.5, 0.0, -0.5, -1.0,
    -0.5, 1.0, 0.0, 1.5,
    0.5, -1.5, 1.0, -0.5,
    });
    base.expected_rinfo = {T(base.n), T(base.p)};
    base.expected_gamma = T(1) / T(base.p);
    base.expected_eigenvalues = convert_vector<double, T>({
    8.60596352848875767e+00, 6.98136841918766926e+00,
    });
    base.expected_eigenvectors = convert_vector<double, T>({
    -4.59558492984263933e-01, 6.25517201171490900e-01, -5.21022203683143115e-01, 3.55063495495915926e-01,
    7.27740523679692330e-01, 1.28575754059560093e-01, -6.36992070381986575e-01, -2.19324207357265988e-01,
    });
    base.expected_scores = convert_vector<double, T>({
    -1.34815723651297903e+00, 1.83501241778065927e+00, -1.52846670228640669e+00, 1.04161152101872578e+00,
    1.92285633540952561e+00, 3.39726448134828829e-01, -1.68307823775763277e+00, -5.79504545786722169e-01,
    });

    base.m = 2; base.p_transform = 7;
    base.ldx = 2; base.ldx_transform = 2;
    base.X_transform_in = convert_vector<double, T>({
    -1.0, 0.5, 0.5, -1.5, 1.5, 0.0, -0.5, 1.0, 0.0, -1.0, 1.0, 0.5, -2.0, 1.5,
    });
    base.expected_X_transform = convert_vector<double, T>({
    1.98852464498372616e+00, -1.65797345436315657e+00, -5.48418458578109558e-01, -1.62789208583919276e+00,
    });

    base.k = 4; base.ldy = 4; base.ldy_inv_transform = 4;
    base.Y_inv = convert_vector<double, T>({
    -1.34815723651297903e+00, 1.83501241778065927e+00, -1.52846670228640669e+00, 1.04161152101872578e+00,
    1.92285633540952561e+00, 3.39726448134828829e-01, -1.68307823775763277e+00, -5.79504545786722169e-01,
    });
    base.expected_Y_inv_transform = convert_vector<double, T>({
    6.44819268463416750e-01, -5.40000536162334344e-01, 2.89902706700754353e-01, -3.94721439001836594e-01,
    1.98610049740104461e-01, 7.27931155526871043e-01, -1.07899760021793489e+00, 1.52456394950959112e-01,
    -8.83859919856601839e-01, -2.74396418813152965e-01, 9.28113906914982856e-01, 2.30142431754771892e-01,
    -2.05593916772187746e-01, -3.72499541164310555e-01, 6.19149193401816733e-01, -4.10557354653175929e-02,
    1.35997291467560988e+00, -2.56035700781084119e-01, -5.41983553439137844e-01, -5.61953660455387860e-01,
    -9.30618285885978058e-01, 7.14311405949195510e-01, -3.33436680072439928e-01, 5.49743560009222865e-01,
    5.41694258749428448e-01, -1.08612131166869141e+00, 1.06983790018621572e+00, -5.25410847266952641e-01,
    });
    // clang-format on

    push_variants(params, base, 2);
}

template <typename T> void add_poly_tall(std::vector<KernelPCAParamType<T>> &params) {
    KernelPCAParamType<T> base;
    base.test_name = "poly_tall";
    base.n = 6;
    base.p = 3;
    base.lda = 6;
    base.order = "column-major";
    base.kernel = "poly";
    base.n_components = 3;
    base.gamma = T(0.5);
    base.degree = 2;
    base.fit_inverse_transform = "yes";
    base.expected_n_components = 3;

    // clang-format off
    base.A = convert_vector<double, T>({
    1.0, 3.0, 0.5, -2.0, 1.5, -1.0,
    2.0, -1.0, 1.5, 0.0, -0.5, 1.0,
    -1.0, 0.5, 2.0, 1.0, -1.5, 0.0,
    });
    base.expected_rinfo = {T(base.n), T(base.p)};
    base.expected_gamma = base.gamma;
    base.expected_eigenvalues = convert_vector<double, T>({
    3.13876078230150242e+01, 1.53891964842798430e+01, 1.36514222371583678e+01,
    });
    base.expected_eigenvectors = convert_vector<double, T>({
    -2.73301497257292580e-01, 8.56978266303177860e-01, -3.20438971211782275e-01, -1.74561970802561417e-01,
    1.19689257418373773e-01, -2.08365084449915722e-01, 6.71666545903379375e-01, -1.18807267377822101e-01,
    -5.91475930190690224e-01, -2.54605289008694780e-01, 3.42941943313711306e-01, -4.97200026398837841e-02,
    -2.99718444324180933e-01, -1.26502171652715850e-01, -5.88779043511001299e-01, 6.95073126721668322e-01,
    7.87162727563856662e-02, 2.41210260009843552e-01,
    });
    base.expected_scores = convert_vector<double, T>({
    -1.53116192030838172e+00, 4.80119026446463604e+00, -1.79524794202049609e+00, -9.77977234333404777e-01,
    6.70554808766644417e-01, -1.16735797656899964e+00, 2.63488515091744402e+00, -4.66069817745446924e-01,
    -2.32030485229604055e+00, -9.98792778121419422e-01, 1.34532922560364687e+00, -1.95046928358184823e-01,
    -1.10739464613718219e+00, -4.67398087324324973e-01, -2.17541086606143974e+00, 2.56814444950496013e+00,
    2.90839555139325967e-01, 8.91219594878658872e-01,
    });

    base.m = 3; base.p_transform = 3;
    base.ldx = 3; base.ldx_transform = 3;
    base.X_transform_in = convert_vector<double, T>({
    0.5, -0.5, 1.0, -1.0, 2.0, 0.0, 1.5, -1.0, 0.5,
    });
    base.expected_X_transform = convert_vector<double, T>({
    -1.33944437804207808e-02, -1.48199198470137050e+00, 2.33091152752886478e-02, -6.76746000481215804e-01,
    1.53054680629574791e+00, -1.45303469742128516e-01, -6.01868651453384210e-02, -2.04510062494455080e-01,
    -3.29439161029442096e-01,
    });

    base.k = 6; base.ldy = 6; base.ldy_inv_transform = 6;
    base.Y_inv = convert_vector<double, T>({
    -1.53116192030838172e+00, 4.80119026446463604e+00, -1.79524794202049609e+00, -9.77977234333404777e-01,
    6.70554808766644417e-01, -1.16735797656899964e+00, 2.63488515091744402e+00, -4.66069817745446924e-01,
    -2.32030485229604055e+00, -9.98792778121419422e-01, 1.34532922560364687e+00, -1.95046928358184823e-01,
    -1.10739464613718219e+00, -4.67398087324324973e-01, -2.17541086606143974e+00, 2.56814444950496013e+00,
    2.90839555139325967e-01, 8.91219594878658872e-01,
    });
    base.expected_Y_inv_transform = convert_vector<double, T>({
    1.00071347448220460e+00, 2.98491683024966381e+00, 4.97273810833001095e-01, -1.97774350930994403e+00,
    1.23899294144788152e+00, -8.18122823954347544e-01, 1.93908154440707436e+00, -9.88637735487921998e-01,
    1.47730233574145520e+00, 8.37917667384892317e-02, -3.49269957374555629e-01, 6.95529495989266500e-01,
    -1.00545617085360872e+00, 4.88794377823636272e-01, 1.95442465916293262e+00, 9.40184197148184797e-01,
    -1.21074056415688158e+00, 9.16004108402117267e-02,
    });
    // clang-format on

    push_variants(params, base, 2);
}

template <typename T> void add_poly_wide(std::vector<KernelPCAParamType<T>> &params) {
    KernelPCAParamType<T> base;
    base.test_name = "poly_wide";
    base.n = 4;
    base.p = 7;
    base.lda = 4;
    base.order = "column-major";
    base.kernel = "poly";
    base.n_components = 2;
    base.gamma = T(0.5);
    base.degree = 2;
    base.fit_inverse_transform = "yes";
    base.expected_n_components = 2;

    // clang-format off
    base.A = convert_vector<double, T>({
    1.0, -0.5, 0.5, 0.0,
    0.5, 1.0, -1.0, 0.5,
    -1.0, 0.5, 1.5, -0.5,
    0.0, -1.0, 0.5, 1.0,
    1.5, 0.0, -0.5, -1.0,
    -0.5, 1.0, 0.0, 1.5,
    0.5, -1.5, 1.0, -0.5,
    });
    base.expected_rinfo = {T(base.n), T(base.p)};
    base.expected_gamma = base.gamma;
    base.expected_eigenvalues = convert_vector<double, T>({
    1.45440565982219958e+01, 1.22497396621108887e+01,
    });
    base.expected_eigenvectors = convert_vector<double, T>({
    -5.12615591339545040e-01, 6.68141841537933878e-01, -4.51070992583608388e-01, 2.95544742385219661e-01,
    -6.95156830833805528e-01, -8.36253377635032824e-02, 7.10726540143847618e-01, 6.80556284534614281e-02,
    });
    base.expected_scores = convert_vector<double, T>({
    -1.95494520546339401e+00, 2.54807054594420634e+00, -1.72023459522682032e+00, 1.12710925474600865e+00,
    -2.43302305411496533e+00, -2.92685572035178698e-01, 2.48751645764199969e+00, 2.38192168508145313e-01,
    });

    base.m = 2; base.p_transform = 7;
    base.ldx = 2; base.ldx_transform = 2;
    base.X_transform_in = convert_vector<double, T>({
    -1.0, 0.5, 0.5, -1.5, 1.5, 0.0, -0.5, 1.0, 0.0, -1.0, 1.0, 0.5, -2.0, 1.5,
    });
    base.expected_X_transform = convert_vector<double, T>({
    2.77547534039406818e+00, -1.00602936343485916e+00, -4.51360707381994242e-01, 1.96610557575330391e+00,
    });

    base.k = 4; base.ldy = 4; base.ldy_inv_transform = 4;
    base.Y_inv = convert_vector<double, T>({
    -1.95494520546339401e+00, 2.54807054594420634e+00, -1.72023459522682032e+00, 1.12710925474600865e+00,
    -2.43302305411496533e+00, -2.92685572035178698e-01, 2.48751645764199969e+00, 2.38192168508145313e-01,
    });
    base.expected_Y_inv_transform = convert_vector<double, T>({
    9.70246386768747193e-01, -4.45629491885175699e-01, 4.80683691272310254e-01, -8.14466006854964591e-02,
    4.87199045207279580e-01, 9.71882551111366677e-01, -9.66342119534914223e-01, 4.09878477273876496e-01,
    -9.68849622219502771e-01, 3.85635460027764165e-01, 1.46071011354652680e+00, -1.92421445924784634e-01,
    -6.07537325901057450e-03, -7.53257389754291462e-01, 4.67942830435398471e-01, 3.58125546428496311e-01,
    1.46080969496861912e+00, -1.46188343311383168e-01, -4.74859062843696667e-01, -5.09968833424246037e-01,
    -4.87802810574355261e-01, 1.12287360855257190e+00, -7.36594289430732557e-03, 9.13525196517682492e-01,
    4.83096728920565655e-01, -1.42089607816204455e+00, 9.62859309651216022e-01, -4.86597277764569902e-01,
    });
    // clang-format on

    push_variants(params, base, 2);
}

template <typename T> void add_rbf_tall(std::vector<KernelPCAParamType<T>> &params) {
    KernelPCAParamType<T> base;
    base.test_name = "rbf_tall";
    base.n = 6;
    base.p = 3;
    base.lda = 6;
    base.order = "column-major";
    base.kernel = "rbf";
    base.solver = "auto";
    base.gamma = T(0.5);
    base.fit_inverse_transform = "yes";
    base.n_components = 5;
    base.expected_n_components = 5;

    // clang-format off
    base.A = convert_vector<double, T>({
    1.0, 3.0, 0.5, -2.0, 1.5, -1.0,
    2.0, -1.0, 1.5, 0.0, -0.5, 1.0,
    -1.0, 0.5, 2.0, 1.0, -1.5, 0.0,
    });
    base.expected_rinfo = {T(base.n), T(base.p)};
    base.expected_gamma = base.gamma;
    base.expected_eigenvalues = convert_vector<double, T>({
    1.13889228957286082e+00, 1.00597570048067242e+00, 9.97984963380812107e-01, 9.49034172788480568e-01,
    7.72031940305863085e-01,
    });
    base.expected_eigenvectors = convert_vector<double, T>({
    -1.95028014286023982e-01, -3.94798609311985960e-01, -1.06080900507886405e-01, 5.77080518678823151e-01,
    -4.17431923304374508e-01, 5.36258928731447648e-01, 2.03277015442453690e-01, -4.26037216817467201e-01,
    8.01002270085189094e-01, -2.16518839920699063e-01, -2.88713023829341664e-01, -7.30102049601350500e-02,
    7.39328180274708813e-01, -5.00222506788661869e-01, -4.00884385569803159e-01, -7.07999594271337546e-02,
    1.88414024608524189e-01, 4.41646469023658159e-02, -4.40908483370571203e-01, -4.93806029526193124e-01,
    1.33445035924157157e-01, 7.60466819753996270e-02, 7.33549793901154268e-01, -8.32699890394634351e-03,
    -1.13883331147902001e-01, 4.35055193133711132e-02, -4.42681713273022498e-02, -6.65308400788504395e-01,
    4.61905597439004820e-02, 7.33763824206437154e-01,
    });
    base.expected_scores = convert_vector<double, T>({
    -2.08131744945377534e-01, -4.21324719727715846e-01, -1.13208366546269992e-01, 6.15853962141346578e-01,
    -4.45478742688867613e-01, 5.72289611766884421e-01, 2.03883472070575555e-01, -4.27308256208729254e-01,
    8.03391980179940735e-01, -2.17164802206682828e-01, -2.89574370236584555e-01, -7.32280235985198896e-02,
    7.38582917974258235e-01, -4.99718269311865082e-01, -4.00480283538559800e-01, -7.07285912011655915e-02,
    1.88224098303585108e-01, 4.41201277737471587e-02, -4.29525923643774010e-01, -4.81057858791155646e-01,
    1.29999998804345873e-01, 7.40834493948139133e-02, 7.14612362083490660e-01, -8.11202784772041552e-03,
    -1.00063985381965842e-01, 3.82262760030636708e-02, -3.88963828501703995e-02, -5.84575542530803571e-01,
    4.05854961249403692e-02, 6.44724138634935939e-01,
    });

    base.m = 3; base.p_transform = 3;
    base.ldx = 3; base.ldx_transform = 3;
    base.X_transform_in = convert_vector<double, T>({
    0.5, -0.5, 1.0, -1.0, 2.0, 0.0, 1.5, -1.0, 0.5,
    });
    base.expected_X_transform = convert_vector<double, T>({
    -3.23178684524370874e-02, 6.82903208331758355e-02, -8.32522890797626203e-02, 1.99334687074597637e-02,
    5.21438300158077322e-02, 1.86186052944305354e-02, -3.49148012957224835e-02, 2.46382377694858545e-01,
    -2.99402012077579008e-02, -5.42760827410980126e-03, -1.47711471467210448e-01, 2.75309134599570210e-02,
    -1.95356460243256425e-02, 2.11250004018720017e-01, 3.90323844419649887e-02,
    });

    base.k = 6; base.ldy = 6; base.ldy_inv_transform = 6;
    base.Y_inv = convert_vector<double, T>({
    -2.08131744945377534e-01, -4.21324719727715846e-01, -1.13208366546269992e-01, 6.15853962141346578e-01,
    -4.45478742688867613e-01, 5.72289611766884421e-01, 2.03883472070575555e-01, -4.27308256208729254e-01,
    8.03391980179940735e-01, -2.17164802206682828e-01, -2.89574370236584555e-01, -7.32280235985198896e-02,
    7.38582917974258235e-01, -4.99718269311865082e-01, -4.00480283538559800e-01, -7.07285912011655915e-02,
    1.88224098303585108e-01, 4.41201277737471587e-02, -4.29525923643774010e-01, -4.81057858791155646e-01,
    1.29999998804345873e-01, 7.40834493948139133e-02, 7.14612362083490660e-01, -8.11202784772041552e-03,
    -1.00063985381965842e-01, 3.82262760030636708e-02, -3.88963828501703995e-02, -5.84575542530803571e-01,
    4.05854961249403692e-02, 6.44724138634935939e-01,
    });
    base.expected_Y_inv_transform = convert_vector<double, T>({
    5.70410869347726335e-01, 1.35383625951921815e+00, 3.71077723430620565e-01, -6.29830732520272996e-01,
    7.83702250371070219e-01, -2.72153610068759488e-01, 9.49595944846236550e-01, -2.18934105710737875e-01,
    7.58774381441299051e-01, 1.98327530606759034e-01, -1.94650942724729735e-02, 5.66730786744574444e-01,
    -3.36029209124823147e-01, 2.42677256950412423e-01, 8.31193105855172809e-01, 4.42036204914346753e-01,
    -5.26291152015300079e-01, 9.08113776061106187e-02,
    });
    // clang-format on

    push_variants(params, base, 2);
}

template <typename T> void add_rbf_wide(std::vector<KernelPCAParamType<T>> &params) {
    KernelPCAParamType<T> base;
    base.test_name = "rbf_wide";
    base.n = 4;
    base.p = 7;
    base.lda = 4;
    base.order = "column-major";
    base.kernel = "rbf";
    base.n_components = 2;
    base.gamma = T(0.5);
    base.fit_inverse_transform = "yes";
    base.expected_n_components = 2;
    base.copy_data = "no";

    // clang-format off
    base.A = convert_vector<double, T>({
    1.0, -0.5, 0.5, 0.0,
    0.5, 1.0, -1.0, 0.5,
    -1.0, 0.5, 1.5, -0.5,
    0.0, -1.0, 0.5, 1.0,
    1.5, 0.0, -0.5, -1.0,
    -0.5, 1.0, 0.0, 1.5,
    0.5, -1.5, 1.0, -0.5,
    });
    base.expected_rinfo = {T(base.n), T(base.p)};
    base.expected_gamma = base.gamma;
    base.expected_eigenvalues = convert_vector<double, T>({
    1.00826477047597796e+00, 9.98878936390051986e-01,
    });
    base.expected_eigenvectors = convert_vector<double, T>({
    5.37581275041487561e-01, -5.24541070390345032e-01, 4.60265723685591988e-01, -4.73305928336734627e-01,
    -6.78529736494887570e-01, -7.19544445764139229e-02, 7.30774579612772479e-01, 1.97096014585293119e-02,
    });
    base.expected_scores = convert_vector<double, T>({
    5.39798196804770480e-01, -5.26704215887915272e-01, 4.62163805235498992e-01, -4.75257786152354256e-01,
    -6.78149292341537313e-01, -7.19141005114830728e-02, 7.30364842351047039e-01, 1.96985505019736037e-02,
    });

    base.m = 2; base.p_transform = 7;
    base.ldx = 2; base.ldx_transform = 2;
    base.X_transform_in = convert_vector<double, T>({
    -1.0, 0.5, 0.5, -1.5, 1.5, 0.0, -0.5, 1.0, 0.0, -1.0, 1.0, 0.5, -2.0, 1.5,
    });
    base.expected_X_transform = convert_vector<double, T>({
    -1.89310628559939020e-01, 8.08062747879385035e-02, -2.61928211023443741e-02, 1.26687750628180829e-01,
    });

    base.k = 4; base.ldy = 4; base.ldy_inv_transform = 4;
    base.Y_inv = convert_vector<double, T>({
    5.39798196804770480e-01, -5.26704215887915272e-01, 4.62163805235498992e-01, -4.75257786152354256e-01,
    -6.78149292341537313e-01, -7.19141005114830728e-02, 7.30364842351047039e-01, 1.96985505019736037e-02,
    });
    base.expected_Y_inv_transform = convert_vector<double, T>({
    4.46094439760114647e-01, -4.91215828764538023e-02, 2.60858100579093977e-01, -3.46438183638235336e-02,
    2.75219643220575239e-01, 4.47093540587533755e-01, -3.11854491368420972e-01, 4.05312217090820048e-01,
    -3.51344125667803320e-01, 1.74799206754564923e-02, 5.96075829715485117e-01, 6.43145066981611890e-02,
    2.75594491372645051e-02, 2.17981453016213855e-02, 2.56492876631446820e-01, 4.64128751605381584e-02,
    5.72833160984794421e-01, -2.20015401435306263e-01, -2.18968867975998127e-01, -2.43851107217672225e-01,
    -3.86210000590048105e-02, 7.36639409191907735e-01, 1.65612348705424711e-01, 7.31470968312991854e-01,
    1.38071044846583529e-01, -5.26658746675697120e-01, 3.47018371542623216e-01, -4.85798986779630959e-01,
    });
    // clang-format on

    push_variants(params, base, 2);
}

template <typename T> void add_sigmoid_tall(std::vector<KernelPCAParamType<T>> &params) {
    KernelPCAParamType<T> base;
    base.test_name = "sigmoid_tall";
    base.n = 6;
    base.p = 3;
    base.lda = 6;
    base.order = "column-major";
    base.kernel = "sigmoid";
    base.n_components = 2;
    base.gamma = T(0.0001);
    base.coef0 = T(0.0);
    base.fit_inverse_transform = "yes";
    base.expected_n_components = 2;

    // clang-format off
    base.A = convert_vector<double, T>({
    1.0, 3.0, 0.5, -2.0, 1.5, -1.0,
    2.0, -1.0, 1.5, 0.0, -0.5, 1.0,
    -1.0, 0.5, 2.0, 1.0, -1.5, 0.0,
    });
    base.expected_rinfo = {T(base.n), T(base.p)};
    base.expected_gamma = base.gamma;
    base.expected_eigenvalues = convert_vector<double, T>({
    1.83055691890268539e-03, 7.30102862710383868e-04,
    });
    base.expected_eigenvectors = convert_vector<double, T>({
    8.62949115563199365e-02, 6.06456808608454967e-01, -2.05409264444078393e-01, -5.52530455661531694e-01,
    4.03508895540004076e-01, -3.38320895599168892e-01, 5.26237723939191993e-01, -4.89792838636811112e-01,
    -5.16718462240296605e-01, -1.33917138640366307e-01, 3.76485365465003496e-01, 2.37705350113278202e-01,
    });
    base.expected_scores = convert_vector<double, T>({
    3.69212849071820720e-03, 2.59472594741807852e-02, -8.78843704494080245e-03, -2.36400200260481162e-02,
    1.72641313678057497e-02, -1.44750622617158247e-02, 1.42191620394502907e-02, -1.32344060897160304e-02,
    -1.39619476315220453e-02, -3.61849675073262506e-03, 1.01727910666608422e-02, 6.42289736585955912e-03,
    });

    base.m = 3; base.p_transform = 3;
    base.ldx = 3; base.ldx_transform = 3;
    base.X_transform_in = convert_vector<double, T>({
    0.5, -0.5, 1.0, -1.0, 2.0, 0.0, 1.5, -1.0, 0.5,
    });
    base.expected_X_transform = convert_vector<double, T>({
    2.82449857691371856e-04, -9.81121952271260013e-03, 4.94629266296353880e-03, -1.68857568707467577e-02,
    1.75868552039861945e-02, -5.72179857928879071e-03,
    });

    base.k = 6; base.ldy = 6; base.ldy_inv_transform = 6;
    base.Y_inv = convert_vector<double, T>({
    3.69212849071820720e-03, 2.59472594741807852e-02, -8.78843704494080245e-03, -2.36400200260481162e-02,
    1.72641313678057497e-02, -1.44750622617158247e-02, 1.42191620394502907e-02, -1.32344060897160304e-02,
    -1.39619476315220453e-02, -3.61849675073262506e-03, 1.01727910666608422e-02, 6.42289736585955912e-03,
    });
    base.expected_Y_inv_transform = convert_vector<double, T>({
    3.75352495440569067e-08, 4.49280876017674677e-07, -1.21939425553838631e-07, -3.83634805749934504e-07,
    2.67822293164069096e-07, -2.49064187422028001e-07, 1.11240925536065784e-08, -1.71531370428425792e-07,
    1.73981518425376572e-08, 1.21687385507875942e-07, -7.22509579655287595e-08, 9.35726984899345140e-08,
    -1.17527643296565869e-07, -6.05423046184767283e-08, 1.45260655306254886e-07, 1.61188735524285965e-07,
    -1.68647860202797168e-07, 4.02684172872988958e-08,
    });
    // clang-format on

    // bump tolerance a little for sigmoid
    base.epsilon = T(5000) * std::numeric_limits<T>::epsilon();
    push_variants(params, base, 2);
}

template <typename T> void add_sigmoid_wide(std::vector<KernelPCAParamType<T>> &params) {
    KernelPCAParamType<T> base;
    base.test_name = "sigmoid_wide";
    base.n = 4;
    base.p = 7;
    base.lda = 4;
    base.order = "column-major";
    base.kernel = "sigmoid";
    base.n_components = 2;
    base.gamma = T(0.0001);
    base.coef0 = T(0.0);
    base.fit_inverse_transform = "yes";
    base.expected_n_components = 2;

    // clang-format off
    base.A = convert_vector<double, T>({
    1.0, -0.5, 0.5, 0.0,
    0.5, 1.0, -1.0, 0.5,
    -1.0, 0.5, 1.5, -0.5,
    0.0, -1.0, 0.5, 1.0,
    1.5, 0.0, -0.5, -1.0,
    -0.5, 1.0, 0.0, 1.5,
    0.5, -1.5, 1.0, -0.5,
    });
    base.expected_rinfo = {T(base.n), T(base.p)};
    base.expected_gamma = base.gamma;
    base.expected_eigenvalues = convert_vector<double, T>({
    8.60596297989186642e-04, 6.98136797250674466e-04,
    });
    base.expected_eigenvectors = convert_vector<double, T>({
    -4.59558504341474050e-01, 6.25517189265766382e-01, -5.21022196798101689e-01, 3.55063511873809412e-01,
    7.27740516285634631e-01, 1.28575760456850596e-01, -6.36992081819385714e-01, -2.19324194923099541e-01,
    });
    base.expected_scores = convert_vector<double, T>({
    -1.34815722686049584e-02, 1.83501232436664580e-02, -1.52846663337158177e-02, 1.04161153586543216e-02,
    1.92285625435877870e-02, 3.39726454169762999e-03, -1.68307821413460970e-02, -5.79504494393932174e-03,
    });

    base.m = 2; base.p_transform = 7;
    base.ldx = 2; base.ldx_transform = 2;
    base.X_transform_in = convert_vector<double, T>({
    -1.0, 0.5, 0.5, -1.5, 1.5, 0.0, -0.5, 1.0, 0.0, -1.0, 1.0, 0.5, -2.0, 1.5,
    });
    base.expected_X_transform = convert_vector<double, T>({
    1.98852451121392382e-02, -1.65797334806282171e-02, -5.48418434158373484e-03, -1.62789208301876674e-02,
    });

    base.k = 4; base.ldy = 4; base.ldy_inv_transform = 4;
    base.Y_inv = convert_vector<double, T>({
    -1.34815722686049584e-02, 1.83501232436664580e-02, -1.52846663337158177e-02, 1.04161153586543216e-02,
    1.92285625435877870e-02, 3.39726454169762999e-03, -1.68307821413460970e-02, -5.79504494393932174e-03,
    });
    base.expected_Y_inv_transform = convert_vector<double, T>({
    5.83737153760978443e-08, -5.25025233125556742e-08, 3.09704748953004887e-08, -3.68416669588427383e-08,
    8.53236215334978572e-09, 6.80615353135395007e-08, -9.44171369916173236e-08, 1.78232395247280768e-08,
    -6.92949982481536333e-08, -2.36007813680800701e-08, 7.54923527082200149e-08, 1.74034269080136687e-08,
    -1.28577198259645543e-08, -3.45645765973631793e-08, 5.34430506572923263e-08, -6.02075423396461003e-09,
    1.13787673076812896e-07, -2.75718606966843695e-08, -3.73132016996316576e-08, -4.89026106804969253e-08,
    -8.35593130009902720e-08, 6.96474917896786218e-08, -3.71376450860782483e-08, 5.10494662973899316e-08,
    5.57198736438735519e-08, -1.03681353515576591e-07, 9.95428012598848881e-08, -5.15813213881819353e-08,
    });
    // clang-format on

    // bump tolerance a little for sigmoid
    base.epsilon = T(5000) * std::numeric_limits<T>::epsilon();
    push_variants(params, base, 2);
}

template <typename T>
void add_precomputed_square(std::vector<KernelPCAParamType<T>> &params) {
    KernelPCAParamType<T> base;
    base.test_name = "precomputed_square";
    base.n = 5;
    base.p = 5;
    base.lda = 5;
    base.order = "column-major";
    base.kernel = "precomputed";
    base.n_components = 3;
    base.expected_n_components = 3;

    // clang-format off
    base.A = convert_vector<double, T>({
    1.00000000000000000e+00, 1.19432968266719619e-01, 6.39278612067075702e-02, 4.72366552741014689e-01,
    1.70361979580257398e-03, 1.19432968266719619e-01, 1.00000000000000000e+00, 1.70361979580257398e-03,
    6.87289278790972236e-01, 6.39278612067075702e-02, 6.39278612067075702e-02, 1.70361979580257398e-03,
    1.00000000000000000e+00, 1.11089965382423061e-02, 8.04733010124613246e-04, 4.72366552741014689e-01,
    6.87289278790972236e-01, 1.11089965382423061e-02, 1.00000000000000000e+00, 2.66490973363554852e-02,
    1.70361979580257398e-03, 6.39278612067075702e-02, 8.04733010124613246e-04, 2.66490973363554852e-02,
    1.00000000000000000e+00,
    });
    base.expected_rinfo = {T(base.n), T(base.p)};
    base.expected_gamma = T(1) / T(base.p);
    base.expected_eigenvalues = convert_vector<double, T>({
    1.30935519359013486e+00, 1.02528252980439105e+00, 8.61130664749314767e-01,
    });
    base.expected_eigenvectors = convert_vector<double, T>({
    -1.61015148844806999e-01, -4.15181464190497618e-01, 5.68390845682762991e-01, -4.85280034656519976e-01,
    4.93085802009061269e-01, -3.38511806819172589e-01, 2.04741665273422246e-01, -5.74637504707371405e-01,
    -7.98511147375696555e-03, 7.16392757726878937e-01, 7.45094714157353066e-01, -5.11782671380051957e-01,
    -3.76798798008218638e-01, -5.20322370388089719e-02, 1.95518992269726188e-01,
    });
    base.expected_scores = convert_vector<double, T>({
    -1.84244900102900427e-01, -4.75079940882352947e-01, 6.50392930935710933e-01, -5.55291673787787321e-01,
    5.64223583837329401e-01, -3.42764313485112992e-01, 2.07313703467753435e-01, -5.81856307035801934e-01,
    -8.08542330656865892e-03, 7.25392340359730436e-01, 6.91426482869123227e-01, -4.74919611885669057e-01,
    -3.49658456443050270e-01, -4.82844207158856717e-02, 1.81436006175481418e-01,
    });

    base.m = 4; base.p_transform = 5;
    base.ldx = 4; base.ldx_transform = 4;
    base.X_transform_in = convert_vector<double, T>({
    6.87289278790972236e-01, 1.61634945881658741e-02, 3.24652467358349739e-01, 5.35261428518990279e-01,
    4.72366552741014689e-01, 1.73773943450445140e-01, 1.83156388887341787e-02, 1.73773943450445140e-01,
    2.66490973363554852e-02, 5.94621735647209420e-03, 6.87289278790972236e-01, 5.94621735647209420e-03,
    8.82496902584595344e-01, 1.19432968266719619e-01, 9.30144892106634924e-02, 5.35261428518990279e-01,
    1.11089965382423061e-02, 7.78800783071404878e-01, 3.18278079650966689e-03, 3.18278079650966689e-03,
    });
    base.expected_X_transform = convert_vector<double, T>({
    -4.56697623521052032e-01, 3.90219027706195820e-01, 4.18634774306407165e-01, -1.93403507535594599e-01,
    -1.42281585252488241e-01, 5.82620229941195000e-01, -4.87158223117837463e-01, -1.42953192358950376e-01,
    2.39818883308549030e-01, 7.85696196551257942e-02, -2.65934589069756443e-02, 3.08633415849249360e-01,
    });
    // clang-format on

    push_variants(params, base, 2);
}

template <typename T> void add_linear_ncomp2(std::vector<KernelPCAParamType<T>> &params) {
    KernelPCAParamType<T> base;
    base.test_name = "linear_ncomp2";
    base.n = 6;
    base.p = 3;
    base.lda = 6;
    base.order = "column-major";
    base.kernel = "linear";
    base.n_components = 2;
    base.fit_inverse_transform = "yes";
    base.expected_n_components = 2;

    // clang-format off
    base.A = convert_vector<double, T>({
    1.0, 3.0, 0.5, -2.0, 1.5, -1.0,
    2.0, -1.0, 1.5, 0.0, -0.5, 1.0,
    -1.0, 0.5, 2.0, 1.0, -1.5, 0.0,
    });
    base.expected_rinfo = {T(base.n), T(base.p)};
    base.expected_gamma = T(1) / T(base.p);
    base.expected_eigenvalues = convert_vector<double, T>({
    1.83055715040013816e+01, 7.30102995247189313e+00,
    });
    base.expected_eigenvectors = convert_vector<double, T>({
    8.62948669540870433e-02, 6.06456873889861026e-01, -2.05409257584859201e-01, -5.52530436625914168e-01,
    4.03508842196679263e-01, -3.38320888829853950e-01, 5.26237765855806394e-01, -4.89792827467530500e-01,
    -5.16718413116042830e-01, -1.33917212983747119e-01, 3.76485404837802295e-01, 2.37705282873712370e-01,
    });
    base.expected_scores = convert_vector<double, T>({
    3.69212681586973623e-01, 2.59472639079286438e+00, -8.78843730717443261e-01, -2.36400207064011658e+00,
    1.72641301771466504e+00, -1.44750628873694254e+00, 1.42191644626689251e+00, -1.32344069891493854e+00,
    -1.39619475714320740e+00, -3.61849908795775610e-01, 1.01727930538722555e+00, 6.42289613199805154e-01,
    });

    base.m = 3; base.p_transform = 3;
    base.ldx = 3; base.ldx_transform = 3;
    base.X_transform_in = convert_vector<double, T>({
    0.5, -0.5, 1.0, -1.0, 2.0, 0.0, 1.5, -1.0, 0.5,
    });
    base.expected_X_transform = convert_vector<double, T>({
    2.82450111998757726e-02, -9.81122101044140882e-01, 4.94629198892678201e-01, -1.68857555355862710e+00,
    1.75868547965963407e+00, -5.72179687098798184e-01,
    });

    base.k = 6; base.ldy = 6; base.ldy_inv_transform = 6;
    base.Y_inv = convert_vector<double, T>({
    3.69212681586973623e-01, 2.59472639079286438e+00, -8.78843730717443261e-01, -2.36400207064011658e+00,
    1.72641301771466504e+00, -1.44750628873694254e+00, 1.42191644626689251e+00, -1.32344069891493854e+00,
    -1.39619475714320740e+00, -3.61849908795775610e-01, 1.01727930538722555e+00, 6.42289613199805154e-01,
    });
    base.expected_Y_inv_transform = convert_vector<double, T>({
    3.43767330683011629e-02, 2.47617515235670638e+00, -4.74473376734707353e-01, -1.94644235627198237e+00,
    1.27277559533525642e+00, -1.36241174775357465e+00, 2.73078516332397492e-01, -1.08904315325969581e+00,
    -1.21439879829629019e-01, 5.75493170652321639e-01, -2.20105090389046937e-01, 5.82016436493651534e-01,
    -1.26919838360960147e+00, 3.01084213702732195e-01, 1.40090458186292222e+00, 1.00299856144637922e+00,
    -1.34605602342958375e+00, -8.97329499728498903e-02,
    });
    // clang-format on

    push_variants(params, base, 2);
}

template <typename T>
void add_rbf_remove_zero(std::vector<KernelPCAParamType<T>> &params) {
    KernelPCAParamType<T> base;
    base.test_name = "rbf_remove_zero";
    base.n = 6;
    base.p = 3;
    base.lda = 6;
    base.order = "column-major";
    base.kernel = "rbf";
    base.n_components = 3;
    base.gamma = T(1.0);
    base.fit_inverse_transform = "yes";
    base.remove_zero_eig = "yes";
    base.alpha = T(2.0);
    base.expected_n_components = 3;

    // clang-format off
    base.A = convert_vector<double, T>({
    1.0, 3.0, 0.5, -2.0, 1.5, -1.0,
    2.0, -1.0, 1.5, 0.0, -0.5, 1.0,
    -1.0, 0.5, 2.0, 1.0, -1.5, 0.0,
    });
    base.expected_rinfo = {T(base.n), T(base.p)};
    base.expected_gamma = base.gamma;
    base.expected_eigenvalues = convert_vector<double, T>({
    1.03237569475972335e+00, 1.00046974079009288e+00, 9.99979550391963068e-01,
    });
    base.expected_eigenvectors = convert_vector<double, T>({
    -2.61900507156718698e-01, -3.08209176055662837e-01, -2.66265632592976920e-01, 5.82941767470741334e-01,
    -3.16894798649134113e-01, 5.70328346983751233e-01, -2.13291978985384115e-02, -3.79986111237924473e-01,
    8.36444321590110507e-01, -3.23929532403670062e-02, -3.92900073071944911e-01, -9.83598614133514293e-03,
    7.84766137216844628e-01, -5.76019099660908807e-01, -2.22413049310047101e-01, -3.12942244125294297e-02,
    4.35336486724287736e-02, 1.42658749421237067e-03,
    });
    base.expected_scores = convert_vector<double, T>({
    -2.66106342047598809e-01, -3.13158677377432471e-01, -2.70541566610667417e-01, 5.92303172882272855e-01,
    -3.21983781543311587e-01, 5.79487194696737373e-01, -2.13342069075099713e-02, -3.80075348247647660e-01,
    8.36640754532923747e-01, -3.24005604928369884e-02, -3.92992342832875707e-01, -9.83829605205285840e-03,
    7.84758113095868515e-01, -5.76013209948393112e-01, -2.22410775168580377e-01, -3.12939044335820288e-02,
    4.35332035471272180e-02, 1.42657290756025403e-03,
    });

    base.m = 3; base.p_transform = 3;
    base.ldx = 3; base.ldx_transform = 3;
    base.X_transform_in = convert_vector<double, T>({
    0.5, -0.5, 1.0, -1.0, 2.0, 0.0, 1.5, -1.0, 0.5,
    });
    base.expected_X_transform = convert_vector<double, T>({
    -9.54066535544726042e-03, 2.26449841588555641e-02, -1.46245147539856343e-02, 1.36543578451729045e-03,
    -2.85462274708001527e-03, 6.27442194204481463e-04, -8.12570481435087848e-04, 8.28034080343089396e-02,
    -3.85619744286799966e-03,
    });

    base.k = 6; base.ldy = 6; base.ldy_inv_transform = 6;
    base.Y_inv = convert_vector<double, T>({
    -2.66106342047598809e-01, -3.13158677377432471e-01, -2.70541566610667417e-01, 5.92303172882272855e-01,
    -3.21983781543311587e-01, 5.79487194696737373e-01, -2.13342069075099713e-02, -3.80075348247647660e-01,
    8.36640754532923747e-01, -3.24005604928369884e-02, -3.92992342832875707e-01, -9.83829605205285840e-03,
    7.84758113095868515e-01, -5.76013209948393112e-01, -2.22410775168580377e-01, -3.12939044335820288e-02,
    4.35332035471272180e-02, 1.42657290756025403e-03,
    });
    base.expected_Y_inv_transform = convert_vector<double, T>({
    3.85156007850096904e-01, 1.00667331180335906e+00, 2.35157887060332416e-01, -4.88792712323935341e-01,
    8.17407760353011104e-01, -4.87899814407920684e-01, 6.42420545030316648e-01, -2.85042720274115979e-01,
    5.28023119915381423e-01, 2.79348459583695086e-01, 1.81938609080357187e-02, 3.00125976457415344e-01,
    -3.81166005240288541e-01, 2.09034131209851620e-02, 6.23821159709684769e-01, 2.10866032432175665e-01,
    -3.27464992872297411e-01, 2.06933190485916130e-01,
    });
    // clang-format on

    push_variants(params, base, 2);
}

/* 1x1: single sample, single feature.
 * Centered kernel is zero so eigenvalue=0, eigenvector=[[1]], scores=[[0]].
 * n_components must be set explicitly (auto would give nc=0).
 * Inverse transform skipped: zero eigenvalue makes scores all-zero (degenerate). */
template <typename T> void add_linear_1x1(std::vector<KernelPCAParamType<T>> &params) {
    KernelPCAParamType<T> base;
    base.test_name = "linear_1x1";
    base.n = 1;
    base.p = 1;
    base.lda = 1;
    base.order = "column-major";
    base.kernel = "linear";
    base.n_components = 1;
    base.expected_n_components = 1;

    // clang-format off
    base.A = convert_vector<double, T>({3.0});
    base.expected_rinfo = {T(base.n), T(base.p)};
    base.expected_gamma = T(1) / T(base.p);
    base.expected_eigenvalues = convert_vector<double, T>({0.0});
    base.expected_eigenvectors = convert_vector<double, T>({1.0});
    base.expected_scores = convert_vector<double, T>({0.0});

    base.m = 1; base.p_transform = 1;
    base.ldx = 1; base.ldx_transform = 1;
    base.X_transform_in = convert_vector<double, T>({2.0});
    base.expected_X_transform = convert_vector<double, T>({0.0});
    // clang-format on

    push_variants(params, base, 2);
}

/* nx1: 5 samples, 1 feature. Rank-1 linear kernel gives 1 non-zero eigenvalue;
 * Eigenvectors are skipped because the 2nd column is arbitrary for the zero
 * eigenvalue. Scores and transform are still deterministic: the zero eigenvalue
 * multiplied by any eigenvector produces a zero column. */
template <typename T> void add_linear_nx1(std::vector<KernelPCAParamType<T>> &params) {
    KernelPCAParamType<T> base;
    base.test_name = "linear_nx1";
    base.n = 5;
    base.p = 1;
    base.lda = 5;
    base.order = "column-major";
    base.kernel = "linear";
    base.n_components = 2;
    base.fit_inverse_transform = "yes";
    base.expected_n_components = 2;

    // clang-format off
    base.A = convert_vector<double, T>({1.0, 2.0, -1.0, 0.5, -0.5});
    base.expected_rinfo = {T(base.n), T(base.p)};
    base.expected_gamma = T(1) / T(base.p);
    base.expected_eigenvalues = convert_vector<double, T>({5.7, 0.0});
    base.expected_scores = convert_vector<double, T>({
    5.99999999999999978e-01, 1.60000000000000009e+00, -1.40000000000000013e+00, 9.99999999999999639e-02,
    -9.00000000000000133e-01, 0.0, 0.0, 0.0, 0.0, 0.0,
    });

    base.m = 2; base.p_transform = 1;
    base.ldx = 2; base.ldx_transform = 2;
    base.X_transform_in = convert_vector<double, T>({1.5, -2.0});
    base.expected_X_transform = convert_vector<double, T>({
    1.10000000000000031e+00, -2.40000000000000036e+00, 0.0, 0.0,
    });

    base.k = 5; base.ldy = 5; base.ldy_inv_transform = 5;
    base.Y_inv = convert_vector<double, T>({
    5.99999999999999978e-01, 1.60000000000000009e+00, -1.40000000000000013e+00, 9.99999999999999639e-02,
    -9.00000000000000133e-01, 0.0, 0.0, 0.0, 0.0, 0.0,
    });
    base.expected_Y_inv_transform = convert_vector<double, T>({
    5.10447761194029903e-01, 1.36119402985074656e+00, -1.19104477611940318e+00, 8.50746268656716181e-02,
    -7.65671641791045077e-01,
    });
    // clang-format on

    push_variants(params, base, 2);
}

/* 1xn: 1 sample, 5 features. Kernel matrix is 1x1, centered to zero.
 * Same degenerate behaviour as 1x1: eigenvalue=0, scores=0.
 * Inverse transform skipped: zero eigenvalue makes scores all-zero (degenerate). */
template <typename T> void add_linear_1xn(std::vector<KernelPCAParamType<T>> &params) {
    KernelPCAParamType<T> base;
    base.test_name = "linear_1xn";
    base.n = 1;
    base.p = 5;
    base.lda = 1;
    base.order = "column-major";
    base.kernel = "linear";
    base.n_components = 1;
    base.expected_n_components = 1;

    // clang-format off
    base.A = convert_vector<double, T>({1.0, -0.5, 2.0, 0.5, -1.0});
    base.expected_rinfo = {T(base.n), T(base.p)};
    base.expected_gamma = T(1) / T(base.p);
    base.expected_eigenvalues = convert_vector<double, T>({0.0});
    base.expected_eigenvectors = convert_vector<double, T>({1.0});
    base.expected_scores = convert_vector<double, T>({0.0});

    base.m = 1; base.p_transform = 5;
    base.ldx = 1; base.ldx_transform = 1;
    base.X_transform_in = convert_vector<double, T>({0.5, 1.0, -1.0, 0.0, 1.5});
    base.expected_X_transform = convert_vector<double, T>({0.0});
    // clang-format on

    push_variants(params, base, 2);
}

/* All-zeros tall: 6x3 input of all zeros. Kernel matrix is zero, all eigenvalues
 * are zero. Eigenvectors skipped: all eigenvalues are zero so all eigenvector
 * columns are arbitrary. Scores and transform are deterministically zero regardless
 * of eigenvectors. Non-zero transform input is used to verify that transform still
 * produces zeros when the model has no information.
 * Inverse transform skipped: all-zero scores make it degenerate. */
template <typename T>
void add_linear_zeros_tall(std::vector<KernelPCAParamType<T>> &params) {
    KernelPCAParamType<T> base;
    base.test_name = "linear_zeros_tall";
    base.n = 6;
    base.p = 3;
    base.lda = 6;
    base.order = "column-major";
    base.kernel = "linear";
    base.n_components = 2;
    base.expected_n_components = 2;

    // clang-format off
    base.A = convert_vector<double, T>({
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    });
    base.expected_rinfo = {T(base.n), T(base.p)};
    base.expected_gamma = T(1) / T(base.p);
    base.expected_eigenvalues = convert_vector<double, T>({0.0, 0.0});
    base.expected_scores = convert_vector<double, T>({
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    });

    base.m = 3; base.p_transform = 3;
    base.ldx = 3; base.ldx_transform = 3;
    base.X_transform_in = convert_vector<double, T>({
    0.5, -0.5, 1.0, -1.0, 2.0, 0.0, 1.5, -1.0, 0.5,
    });
    base.expected_X_transform = convert_vector<double, T>({
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    });
    // clang-format on

    push_variants(params, base, 2);
}

template <typename T> void GetKernelPCAData(std::vector<KernelPCAParamType<T>> &params) {
    add_linear_zero_mean_colmaj(params);
    add_linear_zero_mean_rowmaj(params);
    add_linear_tall(params);
    add_linear_wide(params);
    add_poly_tall(params);
    add_poly_wide(params);
    add_rbf_tall(params);
    add_rbf_wide(params);
    add_sigmoid_tall(params);
    add_sigmoid_wide(params);
    add_precomputed_square(params);
    add_linear_ncomp2(params);
    add_rbf_remove_zero(params);
    add_linear_1x1(params);
    add_linear_nx1(params);
    add_linear_1xn(params);
    add_linear_zeros_tall(params);
}

#endif // AOCLDA_KERNEL_PCA_TEST_DATA_HPP
