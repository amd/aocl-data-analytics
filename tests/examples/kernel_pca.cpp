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
#include <iomanip>
#include <iostream>
#include <limits>

/* Basic Kernel PCA example
 *
 * This example computes a kernel principal component
 * analysis using an RBF kernel for a small data matrix.
 */

int main() {

    // Initialize the handle
    da_handle handle = nullptr;

    std::cout << "-----------------------------------------------------------------------"
              << std::endl;
    std::cout << "Basic Kernel PCA" << std::endl;
    std::cout << "Kernel PCA with RBF kernel for a 6x3 data matrix" << std::endl
              << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    int exit_code = 0;
    bool pass = true;

    // Input data (column-major, 6 samples x 3 features)
    double A[18] = {1.0,  3.0,  0.5, -2.0, 1.5,  -1.0, // column 0
                    2.0,  -1.0, 1.5, 0.0,  -0.5, 1.0,  // column 1
                    -1.0, 0.5,  2.0, 1.0,  -1.5, 0.0}; // column 2

    da_int n_samples = 6, n_features = 3, n_components = 2, lda = 6;

    // Create the handle and pass it the data matrix
    pass = pass && (da_handle_init_d(&handle, da_handle_kernel_pca) == da_status_success);
    pass = pass && (da_kernel_pca_set_data_d(handle, n_samples, n_features, A, lda) ==
                    da_status_success);

    // Set options: RBF kernel with gamma = 0.5
    pass = pass && (da_options_set_string(handle, "kernel", "rbf") == da_status_success);
    pass = pass && (da_options_set_real_d(handle, "gamma", 0.5) == da_status_success);
    pass = pass && (da_options_set_int(handle, "n_components", n_components) ==
                    da_status_success);

    // Compute the kernel PCA
    pass = pass && (da_kernel_pca_compute_d(handle) == da_status_success);

    // Extract eigenvalues from the handle
    da_int eigenvalues_dim = n_components;
    double *eigenvalues = new double[eigenvalues_dim];
    pass = pass &&
           (da_handle_get_result_d(handle, da_kernel_pca_eigenvalues, &eigenvalues_dim,
                                   eigenvalues) == da_status_success);

    // Transform another data matrix into the kernel PCA feature space
    double X[9] = {0.5,  -0.5, 1.0,  // column 0
                   -1.0, 2.0,  0.0,  // column 1
                   1.5,  -1.0, 0.5}; // column 2
    da_int m_samples = 3, m_features = 3, ldx = 3, ldx_transform = 3;
    double *X_transform = new double[m_samples * n_components];
    pass = pass &&
           (da_kernel_pca_transform_d(handle, m_samples, m_features, X, ldx, X_transform,
                                      ldx_transform) == da_status_success);

    // Check status (we could do this after every function call)
    if (pass) {
        std::cout << "Kernel PCA computed successfully" << std::endl << std::endl;

        std::cout << "Eigenvalues:" << std::endl;
        for (da_int i = 0; i < n_components; i++) {
            std::cout << "  lambda " << i + 1 << ": " << eigenvalues[i] << std::endl;
        }
        std::cout << std::endl;

        std::cout << "X_transform (" << m_samples << " x " << n_components
                  << "):" << std::endl;
        for (da_int j = 0; j < m_samples; j++) {
            for (da_int i = 0; i < n_components; i++) {
                std::cout << std::right << std::setw(12)
                          << X_transform[j + i * ldx_transform] << "  ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // Check against expected results
        // Note: eigenvector signs may differ from the reference; use std::abs for comparison
        double eigenvalues_exp[2] = {1.13889228957286082e+00, 1.00597570048067242e+00};
        double X_transform_exp[6] = {-3.23178684524370874e-02, 6.82903208331758355e-02,
                                     -8.32522890797626203e-02, 1.99334687074597637e-02,
                                     5.21438300158077322e-02,  1.86186052944305354e-02};

        double tol = 100.0 * std::numeric_limits<double>::epsilon();
        double err = 0.0;
        for (da_int i = 0; i < eigenvalues_dim; i++)
            err = std::max(err, std::abs(eigenvalues[i] - eigenvalues_exp[i]));
        for (da_int i = 0; i < m_samples * n_components; i++)
            err = std::max(
                err, std::abs(std::abs(X_transform[i]) - std::abs(X_transform_exp[i])));
        if (err > tol) {
            std::cout << "Solution is not within the expected tolerance: " << err
                      << std::endl;
            exit_code = 1;
        }
    } else {
        exit_code = 1;
    }

    // Clean up
    da_handle_destroy(&handle);
    delete[] eigenvalues;
    delete[] X_transform;

    std::cout << "-----------------------------------------------------------------------"
              << std::endl;

    return exit_code;
}
