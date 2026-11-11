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

#include "aoclda.h"
#include <cmath>
#include <iomanip>
#include <iostream>

// Basic NMF example
//
// Factorize a small non-negative integer matrix A (n_samples x n_features)
// into A ~= W * H with W (n_samples x n_components) and H (n_components x
// n_features), both non-negative, using NNDSVDa initialization and
// multiplicative-update iterations on the Frobenius loss.

int main() {

    da_handle handle = nullptr;

    std::cout << "-----------------------------------------------------------------------"
              << std::endl;
    std::cout << "Basic NMF" << std::endl;
    std::cout << "Non-negative matrix factorization of a 5x4 integer data matrix"
              << std::endl
              << std::endl;
    std::cout << std::fixed;
    std::cout.precision(4);

    int exit_code = 0;
    bool pass = true;

    // Input data: 5 samples (rows), 4 features (cols), non-negative integers
    // stored column-major (lda = n_samples = 5).
    double A[20] = {1, 2, 3, 4, 5, 2, 4, 6, 1, 3, 3, 6, 9, 5, 8, 4, 8, 12, 2, 4};

    const da_int n_samples = 5;
    const da_int n_features = 4;
    const da_int n_components = 2;
    const da_int lda = n_samples;

    pass = pass && (da_handle_init_d(&handle, da_handle_nmf) == da_status_success);
    pass = pass &&
           (da_nmf_init_d(handle, n_samples, n_features, A, lda) == da_status_success);

    // Options: 2 components, NNDSVDa init, MU solver (default), tight tol.
    pass = pass && (da_options_set_int(handle, "n_components", n_components) ==
                    da_status_success);
    pass =
        pass && (da_options_set_string(handle, "init", "nndsvda") == da_status_success);
    pass = pass && (da_options_set_int(handle, "max_iter", 200) == da_status_success);
    pass = pass && (da_options_set_real_d(handle, "tol", 1.0e-6) == da_status_success);

    pass = pass && (da_nmf_compute_d(handle) == da_status_success);

    // Pull W (n_samples x n_components) and H (n_components x n_features) out
    // of the handle. Both are returned column-major.
    da_int w_dim = n_samples * n_components;
    da_int h_dim = n_components * n_features;
    double *W = new double[w_dim];
    double *H = new double[h_dim];
    pass = pass &&
           (da_handle_get_result_d(handle, da_nmf_w, &w_dim, W) == da_status_success);
    pass = pass &&
           (da_handle_get_result_d(handle, da_nmf_h, &h_dim, H) == da_status_success);

    double recon_err = 0.0;
    da_int err_dim = 1;
    pass = pass && (da_handle_get_result_d(handle, da_nmf_reconstruction_err, &err_dim,
                                           &recon_err) == da_status_success);

    da_int n_iter = 0, niter_dim = 1;
    pass = pass && (da_handle_get_result_int(handle, da_nmf_n_iter, &niter_dim,
                                             &n_iter) == da_status_success);

    // transform / inverse_transform on held-out data.
    const da_int m_samples = 2;
    double X_new[8] = {2.0, 1.0, 3.0, 1.0, 5.0, 2.0, 4.0, 1.0};
    const da_int ldx = m_samples, ldx_t = m_samples, ldxi = m_samples;

    double *W_new = new double[m_samples * n_components];
    double *X_inv = new double[m_samples * n_features];
    pass = pass && (da_nmf_transform_d(handle, m_samples, n_features, X_new, ldx, W_new,
                                       ldx_t) == da_status_success);
    pass = pass && (da_nmf_inverse_transform_d(handle, m_samples, n_components, W_new,
                                               ldx_t, X_inv, ldxi) == da_status_success);

    if (pass) {
        std::cout << "NMF computed successfully" << std::endl << std::endl;

        std::cout << "W (" << n_samples << " x " << n_components << "):" << std::endl;
        for (da_int i = 0; i < n_samples; ++i) {
            for (da_int j = 0; j < n_components; ++j)
                std::cout << std::right << std::setw(10) << W[i + j * n_samples] << "  ";
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "H (" << n_components << " x " << n_features << "):" << std::endl;
        for (da_int i = 0; i < n_components; ++i) {
            for (da_int j = 0; j < n_features; ++j)
                std::cout << std::right << std::setw(10) << H[i + j * n_components]
                          << "  ";
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Iterations performed : " << n_iter << std::endl;
        std::cout << "Reconstruction error : " << recon_err << std::endl;
        std::cout << std::endl;

        std::cout << "transform(X_new) (" << m_samples << " x " << n_components
                  << "):" << std::endl;
        for (da_int i = 0; i < m_samples; ++i) {
            for (da_int j = 0; j < n_components; ++j)
                std::cout << std::right << std::setw(10) << W_new[i + j * m_samples]
                          << "  ";
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "inverse_transform(transform(X_new)) (" << m_samples << " x "
                  << n_features << "):" << std::endl;
        for (da_int i = 0; i < m_samples; ++i) {
            for (da_int j = 0; j < n_features; ++j)
                std::cout << std::right << std::setw(10) << X_inv[i + j * m_samples]
                          << "  ";
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // Check against expected results
        double W_exp[10] = {2.53493168366467181e-01, 5.06986336872657484e-01,
                            7.60479505409181078e-01, 9.98461672080685259e-01,
                            1.36345457779917467e+00, 6.24358008801556674e-01,
                            1.24871601749579764e+00, 1.87307402616674046e+00,
                            2.49784638880087076e-02, 3.45332681541548281e-01};
        double H_exp[8] = {3.78338129657690247e+00, 5.57914801716682673e-02,
                           1.30144992885377220e+00, 2.69191018718589703e+00,
                           5.08536708924910563e+00, 2.74693959618509709e+00,
                           1.62190560947065232e+00, 5.73693261405312338e+00};
        double recon_err_exp = 6.58869910477614784e-01;
        double W_new_exp[4] = {6.15852798532844692e-01, 3.06810842563463471e-01,
                               5.92125583024944335e-01, 1.19464213545056860e-01};
        double X_inv_exp[8] = {2.36304152213818153e+00, 1.16744748864283121e+00,
                               2.39545046987320376e+00, 7.20885882671875478e-01,
                               4.75837076340629395e+00, 1.88840673991105445e+00,
                               4.39583967741964443e+00, 1.18297636949894214e+00};

        // Tolerance is looser than for direct methods since NMF uses
        // multiplicative updates and floating-point differences accumulate across
        // iterations between the library's BLAS calls.
        double tol = 1.0e-4;
        double max_err = 0.0;
        for (da_int i = 0; i < w_dim; i++)
            max_err = std::max(max_err, std::abs(W[i] - W_exp[i]));
        for (da_int i = 0; i < h_dim; i++)
            max_err = std::max(max_err, std::abs(H[i] - H_exp[i]));
        max_err = std::max(max_err, std::abs(recon_err - recon_err_exp));
        for (da_int i = 0; i < m_samples * n_components; i++)
            max_err = std::max(max_err, std::abs(W_new[i] - W_new_exp[i]));
        for (da_int i = 0; i < m_samples * n_features; i++)
            max_err = std::max(max_err, std::abs(X_inv[i] - X_inv_exp[i]));

        if (max_err > tol) {
            std::cout << "Solution is not within the expected tolerance: " << max_err
                      << std::endl;
            exit_code = 1;
        }
    } else {
        std::cout << "Something went wrong during NMF computation." << std::endl;
        exit_code = 1;
    }

    delete[] W;
    delete[] H;
    delete[] W_new;
    delete[] X_inv;
    da_handle_destroy(&handle);

    std::cout << "-----------------------------------------------------------------------"
              << std::endl;

    return exit_code;
}
