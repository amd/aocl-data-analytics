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

#ifndef DA_RANDSVD_HPP
#define DA_RANDSVD_HPP

#include "boost/random/mersenne_twister.hpp"
#include "boost/random/normal_distribution.hpp"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "da_qr.hpp"
#include "da_std.hpp"
#include "lapack_templates.hpp"
#include "macros.h"
#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

namespace ARCH {

// Power-iteration normalization strategy for the randomized rangefinder.
enum class power_normalizer_t : da_int { qr = 0, lu = 1, none = 2 };

// Fill a rows-by-cols matrix with i.i.d. standard normal (N(0,1)) values.
// seed == -1 uses std::random_device for a non-deterministic seed;
// any other value seeds the generator deterministically.
template <typename T>
void generate_random_normal_matrix(T *data, da_int rows, da_int cols, da_int seed) {
    boost::random::mt19937 rng;
    if (seed == -1) {
        std::random_device rd;
        rng.seed(rd());
    } else {
        rng.seed(seed);
    }
    boost::random::normal_distribution<T> normal_dist(T(0), T(1));
    da_int count = rows * cols;
    for (da_int i = 0; i < count; ++i)
        data[i] = normal_dist(rng);
}

// Multiply A (or A^T) by B, dispatching to cblas_symm when A is symmetric.
template <typename T>
void sketch_matmul(bool is_symmetric, CBLAS_TRANSPOSE trans, da_int m, da_int n,
                   da_int sketch_size, const T *A, da_int lda, const T *B, da_int ldb,
                   T *C, da_int ldc) {
    if (is_symmetric) {
        da_blas::cblas_symm(CblasColMajor, CblasLeft, CblasUpper, m, sketch_size, T{1}, A,
                            lda, B, ldb, T{0}, C, ldc);
    } else {
        // Derive common and output dim from trans
        da_int rows_C = (trans == CblasNoTrans) ? m : n;
        da_int k = (trans == CblasNoTrans) ? n : m;
        da_blas::cblas_gemm(CblasColMajor, trans, CblasNoTrans, rows_C, sketch_size, k,
                            T{1}, A, lda, B, ldb, T{0}, C, ldc);
    }
}

// Computes an orthonormal basis Q for the range of A using a randomized sketch.
// A: m-by-n, column-major, lda >= m.
// Q: pre-allocated, ldq >= m, ldq*sketch_size elements; output is m-by-sketch_size.
// q: No. of power iterations. Set to -1 auto-selects 7 iterations if sketch_size < 0.1*min(m,n), else 4.
// seed: random seed. Set to -1 for non-deterministic.
// is_symmetric: set true if A is symmetric.
template <typename T>
da_status da_random_rangefinder(da_int m, da_int n, const T *A, da_int lda, T *Q,
                                da_int ldq, da_int sketch_size, da_int q, da_int seed,
                                power_normalizer_t normalizer_type,
                                da_errors::da_error_t &err, bool is_symmetric = false) {
    da_int q_eff =
        (q == -1) ? ((sketch_size < static_cast<da_int>(0.1 * std::min(m, n))) ? 7 : 4)
                  : q;

    // da_qr performs required allocations of tau(_blocked), R(_blocked)
    std::vector<T> Omega, Y, Z, tau_blocked, R_blocked, tau, R, Q_buf;
    std::vector<da_int> ipiv;
    da_int n_blocks_qr = 0, block_sz_qr = 0, final_block_sz_qr = 0;
    try {
        Omega.resize(n * sketch_size);
        Y.resize(m * sketch_size);
        Z.resize(n * sketch_size);
        Q_buf.resize(std::max(m, n) * sketch_size);
        ipiv.resize(sketch_size);
    } catch (std::bad_alloc &) {
        return da_error(&err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    generate_random_normal_matrix(Omega.data(), n, sketch_size, seed);

    // Y = A * Omega  (initial sketch)
    sketch_matmul(is_symmetric, CblasNoTrans, m, n, sketch_size, A, lda, Omega.data(), n,
                  Y.data(), m);

    // QR-normalize Y in-place using the blocked parallel da_qr + da_qr_apply.
    // After the call Y holds the explicit orthonormal Q (rows x sketch_size).
    auto qr_normalize = [&](std::vector<T> &Y, da_int rows) -> da_status {
        // Clear possible stale data from previous da_qr call
        da_std::fill(tau_blocked.begin(), tau_blocked.end(), T{0});
        da_std::fill(R_blocked.begin(), R_blocked.end(), T{0});
        da_std::fill(tau.begin(), tau.end(), T{0});
        da_std::fill(R.begin(), R.end(), T{0});

        da_std::fill(Q_buf.begin(), Q_buf.begin() + rows * sketch_size, T{0});
        // Place identity in leading sketch_size by sketch_size block
#pragma omp simd
        for (da_int i = 0; i < sketch_size; ++i)
            Q_buf[i + i * rows] = T{1};

        // Compute QR factorization
        da_status status =
            da_qr(rows, sketch_size, Y, rows, tau_blocked, R_blocked, tau, R, n_blocks_qr,
                  block_sz_qr, final_block_sz_qr, true, &err);
        if (status != da_status_success)
            return status;

        // Accumulate Q factor in Q_buf
        status =
            da_qr_apply(sketch_size, Y, rows, tau_blocked, R_blocked, tau, n_blocks_qr,
                        block_sz_qr, final_block_sz_qr, sketch_size, Q_buf, rows, &err);
        if (status != da_status_success)
            return status; // LCOV_EXCL_LINE

        memcpy(Y.data(), Q_buf.data(),
               static_cast<size_t>(rows) * sketch_size * sizeof(T));
        return da_status_success;
    };

    // LU-normalize Y in-place: getrf, then extract unit lower-trapezoidal L
    // and apply the row permutation (LAPACK ipiv is 1-based).
    auto lu_normalize = [&](std::vector<T> &Y, da_int rows) -> da_status {
        da_int info = 0;
        da::getrf(&rows, &sketch_size, Y.data(), &rows, ipiv.data(), &info);
        if (info < 0)
            return da_error(&err, da_status_internal_error, // LCOV_EXCL_LINE
                            "getrf failed.");
        for (da_int j = 0; j < sketch_size; ++j) {
            Y[j + j * rows] = T{1};
            for (da_int i = 0; i < j; ++i)
                Y[i + j * rows] = T{0};
        }
        for (da_int j = sketch_size - 1; j >= 0; --j) {
            da_int piv = ipiv[j] - 1;
            if (piv != j) {
                for (da_int col = 0; col < sketch_size; ++col)
                    std::swap(Y[j + col * rows], Y[piv + col * rows]);
            }
        }
        return da_status_success;
    };

    auto normalizer = [&](std::vector<T> &Y, da_int rows) -> da_status {
        switch (normalizer_type) {
        case power_normalizer_t::lu:
            return lu_normalize(Y, rows);
        case power_normalizer_t::none:
            return da_status_success;
        default: // qr
            return qr_normalize(Y, rows);
        }
    };

    // Power iteration
    for (da_int iter = 0; iter < q_eff; ++iter) {
        da_status status = normalizer(Y, m);
        if (status != da_status_success)
            return status; // LCOV_EXCL_LINE

        sketch_matmul(is_symmetric, CblasTrans, m, n, sketch_size, A, lda, Y.data(), m,
                      Z.data(), n);

        status = normalizer(Z, n);
        if (status != da_status_success)
            return status; // LCOV_EXCL_LINE

        sketch_matmul(is_symmetric, CblasNoTrans, m, n, sketch_size, A, lda, Z.data(), n,
                      Y.data(), m);
    }

    // Final QR of Y: always orthonormal output regardless of loop normalizer used.
    da_status status = qr_normalize(Y, m);
    if (status != da_status_success)
        return status; // LCOV_EXCL_LINE

    for (da_int j = 0; j < sketch_size; ++j)
        memcpy(Q + j * ldq, Y.data() + j * m, static_cast<size_t>(m) * sizeof(T));

    return da_status_success;
}

// Rank-k truncated SVD of A using the randomized rangefinder.
// A: m-by-n, column-major, lda >= m.
// k: requested number of singular values
// sigma: size >= k, filled in descending order.
// U: m-by-k, ldu >= m; pass nullptr to skip computing U.
// Vt: k-by-n, ldvt >= k; receives the top-k right singular vectors.
// p: oversampling parameter >= 0; clamped so sketch_size = k+p <= min(m,n).
// q: power iterations (see rangefinder).
template <typename T>
da_status da_random_svd(da_int m, da_int n, const T *A, da_int lda, T *U, da_int ldu,
                        T *sigma, T *Vt, da_int ldvt, da_int k, da_int p, da_int q,
                        da_int seed, power_normalizer_t normalizer_type,
                        da_errors::da_error_t &err) {

    da_int sketch_size = k + std::min(p, std::min(m, n) - k);

    // Q: m x sketch_size orthonormal range basis;
    // B: sketch_size x n projected matrix;
    // U_hat: sketch_size x sketch_size left singular vectors of B (lifted to U via Q);
    // V_hat: n_svs x n right singular vectors of B; top-k rows copied to caller's Vt.
    // DGESDD JOBZ='S' writes n_svs = min(sketch_size, n) singular values/vectors into
    // internal buffers; only k values/vectors are copied back to the caller's outputs.
    da_int n_svs = std::min(sketch_size, n);
    std::vector<T> Q, B, U_hat, Vt_hat, sigma_int, work;
    std::vector<da_int> iwork;
    try {
        Q.resize(m * sketch_size);
        B.resize(sketch_size * n);
        U_hat.resize(sketch_size * sketch_size);
        Vt_hat.resize(sketch_size * n);
        sigma_int.resize(n_svs);
        iwork.resize(8 * sketch_size);
    } catch (std::bad_alloc &) {
        return da_error(&err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    da_status status = da_random_rangefinder(m, n, A, lda, Q.data(), m, sketch_size, q,
                                             seed, normalizer_type, err);
    if (status != da_status_success)
        return status; // LCOV_EXCL_LINE

    // B = Q^T * A  (sketch_size x n)
    da_blas::cblas_gemm(CblasColMajor, CblasTrans, CblasNoTrans, sketch_size, n, m, T{1},
                        Q.data(), m, A, lda, T{0}, B.data(), sketch_size);

    // SVD of the small matrix B (sketch_size x n) via gesdd into internal V_hat buffer.
    // Vt_hat is sketch_size-by-n; only the top-k rows are copied to the caller's Vt afterwards.
    char jobz = 'S';
    da_int info = 0, lwork = -1;
    T wq = T{0};
    da::gesdd(&jobz, &sketch_size, &n, B.data(), &sketch_size, sigma_int.data(),
              U_hat.data(), &sketch_size, Vt_hat.data(), &n_svs, &wq, &lwork,
              iwork.data(), &info);
    if (info != 0)
        return da_error(&err, da_status_internal_error, // LCOV_EXCL_LINE
                        "da_random_svd: gesdd failed (info=" + std::to_string(info) +
                            ").");

    lwork = static_cast<da_int>(wq);
    try {
        work.resize(lwork);
    } catch (std::bad_alloc &) {
        return da_error(&err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    da::gesdd(&jobz, &sketch_size, &n, B.data(), &sketch_size, sigma_int.data(),
              U_hat.data(), &sketch_size, Vt_hat.data(), &n_svs, work.data(), &lwork,
              iwork.data(), &info);
    if (info != 0)
        return da_error(&err, da_status_internal_error, // LCOV_EXCL_LINE
                        "da_random_svd: gesdd failed (info=" + std::to_string(info) +
                            ").");

    // Copy the top-k singular values into the caller's buffer
    for (da_int i = 0; i < k; ++i)
        sigma[i] = sigma_int[i];

    // Copy top-k rows of V_hat (n_svs-by-n, col-major) into caller's Vt (k-by-n, col-major)
    for (da_int j = 0; j < n; ++j)
        memcpy(Vt + j * ldvt, Vt_hat.data() + j * n_svs,
               static_cast<size_t>(k) * sizeof(T));

    // U = Q * U_hat[:, 0:k]  (m x k); GESDD returns singular vectors in descending order,
    // so the first k columns of U_hat are the k principal left singular vectors.
    if (U != nullptr) {
        da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, k, sketch_size,
                            T{1}, Q.data(), m, U_hat.data(), sketch_size, T{0}, U, ldu);
    }

    return da_status_success;
}

// Top-k eigendecomposition of a symmetric matrix using the randomized rangefinder.
// A: symmetric m-by-m, column-major, lda >= m; only the upper triangle is read.
// k: requested number of eigenvalues
// lambda: size >= k, filled in descending order.
// V: m-by-k, column-major, ldv >= m.
// p: oversampling parameter; clamped so sketch_size = k+p <= m. q: power iterations (see rangefinder).
template <typename T>
da_status da_random_syevd(da_int m, const T *A, da_int lda, T *V, da_int ldv, T *lambda,
                          da_int k, da_int p, da_int q, da_int seed,
                          power_normalizer_t normalizer_type,
                          da_errors::da_error_t &err) {

    da_int sketch_size = k + std::min(p, m - k);

    std::vector<T> Q, C, B, evals, work;
    std::vector<da_int> iwork;
    try {
        Q.resize(m * sketch_size);
        C.resize(m * sketch_size);
        B.resize(sketch_size * sketch_size);
        evals.resize(sketch_size);
    } catch (std::bad_alloc &) {
        return da_error(&err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    // is_symmetric=true: rangefinder uses cblas_symm(CblasUpper) for all A multiplications
    da_status status =
        da_random_rangefinder(m, m, A, lda, Q.data(), m, sketch_size, q, seed,
                              normalizer_type, err, /*is_symmetric=*/true);
    if (status != da_status_success)
        return status; // LCOV_EXCL_LINE

    // C = A * Q  (m x sketch_size)
    da_blas::cblas_symm(CblasColMajor, CblasLeft, CblasUpper, m, sketch_size, T{1}, A,
                        lda, Q.data(), m, T{0}, C.data(), m);

    // B = Q^T * C  (sketch_size x sketch_size)
    da_blas::cblas_gemm(CblasColMajor, CblasTrans, CblasNoTrans, sketch_size, sketch_size,
                        m, T{1}, Q.data(), m, C.data(), m, T{0}, B.data(), sketch_size);

    // Eigendecomposition of the small symmetric matrix B
    char job = 'V', uplo = 'U';
    da_int info = 0, lwork = -1, liwork = -1;
    T wq = T{0};
    da_int iwq = 0;
    da::syevd(&job, &uplo, &sketch_size, B.data(), &sketch_size, evals.data(), &wq,
              &lwork, &iwq, &liwork, &info);
    if (info != 0)
        return da_error(&err, da_status_internal_error, // LCOV_EXCL_LINE
                        "da_random_syevd: syevd failed (info=" + std::to_string(info) +
                            ").");

    lwork = static_cast<da_int>(wq);
    liwork = iwq;
    try {
        work.resize(lwork);
        iwork.resize(liwork);
    } catch (std::bad_alloc &) {
        return da_error(&err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    da::syevd(&job, &uplo, &sketch_size, B.data(), &sketch_size, evals.data(),
              work.data(), &lwork, iwork.data(), &liwork, &info);
    if (info != 0)
        return da_error(&err, da_status_internal_error, // LCOV_EXCL_LINE
                        "da_random_syevd: syevd failed (info=" + std::to_string(info) +
                            ").");

    // syevd returns eigenvalues/vectors in ascending order.
    // The top-k eigenvectors occupy columns sketch_size-k .. sketch_size-1 of B.
    // Reverse those k columns in-place so they become
    // descending: col sketch_size-k <- largest, col sketch_size-1 <- k-th largest.
    // The block's starting address is unchanged, so the GEMM below can use
    // B.data() + sketch_size*(sketch_size-k) to address the now-descending k columns.
    for (da_int i = 0; i < k / 2; ++i) {
        da_int col_lo = sketch_size - k + i;
        da_int col_hi = sketch_size - 1 - i;
        std::swap_ranges(B.data() + col_lo * sketch_size,
                         B.data() + col_lo * sketch_size + sketch_size,
                         B.data() + col_hi * sketch_size);
    }
    // V = Q * B[:, sketch_size-k : sketch_size]
    da_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, k, sketch_size,
                        T{1}, Q.data(), m,
                        B.data() + static_cast<size_t>(sketch_size) * (sketch_size - k),
                        sketch_size, T{0}, V, ldv);

    // Copy eigenvalues in descending order
    for (da_int i = 0; i < k; ++i)
        lambda[i] = evals[sketch_size - 1 - i];

    return da_status_success;
}

} // namespace ARCH

#endif // DA_RANDSVD_HPP
