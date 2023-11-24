/*
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iostream>

/* Basic PCA example
 *
 * This example computes a principal component
 * analysis for a small data matrix.
 */

int main() {

    // Initialize the handle
    da_handle handle = nullptr;

    std::cout << "-----------------------------------------------------------------------"
              << std::endl;
    std::cout << "Basic PCA" << std::endl;
    std::cout << "Principal component analysis for a 6x5 data matrix" << std::endl
              << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    int exit_code = 0;
    bool pass = true;

    // Input data
    double A[30] = {2.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 5.0, 2.0, 8.0,
                    3.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 8.0,
                    4.0, 6.0, 9.0, 5.0, 4.0, 3.0, 1.0, 4.0, 2.0, 2.0};

    da_int n_samples = 6, n_features = 5, n_components = 3, lda = 6;

    // Create the handle and pass it the data matrix
    pass = pass && (da_handle_init_d(&handle, da_handle_pca) == da_status_success);
    pass = pass && (da_pca_set_data_d(handle, n_samples, n_features, A, lda) ==
                    da_status_success);

    // Set options
    pass = pass && (da_options_set_string(handle, "PCA method", "covariance") ==
                    da_status_success);
    pass = pass && (da_options_set_int(handle, "n_components", n_components) ==
                    da_status_success);

    // Compute the PCA
    pass = pass && (da_pca_compute_d(handle) == da_status_success);

    // Transform another data matrix into the same feature space
    double X[15] = {7.0, 3.0, 3.0, 4.0, 2.0, 3.0, 2.0, 5.0,
                    2.0, 9.0, 6.0, 4.0, 3.0, 4.0, 1.0};
    da_int m_samples = 3, m_features = 5, ldx = 3, ldx_transform = 3;
    double *X_transform = new double[m_samples * n_components];
    pass = pass && (da_pca_transform_d(handle, m_samples, m_features, X, ldx, X_transform,
                                       ldx_transform) == da_status_success);

    // Extract results from the handle
    da_int principal_components_dim = n_components * n_features;
    da_int scores_dim = n_samples * n_components;
    double *principal_components = new double[principal_components_dim];
    double *scores = new double[scores_dim];

    pass = pass && (da_handle_get_result_d(handle, da_pca_principal_components,
                                           &principal_components_dim,
                                           principal_components) == da_status_success);
    pass = pass && (da_handle_get_result_d(handle, da_pca_scores, &scores_dim, scores) ==
                    da_status_success);

    // Check status (we could do this after every function call)
    if (pass) {
        std::cout << "PCA computed successfully" << std::endl << std::endl;

        std::cout << "Principal components:" << std::endl;
        for (da_int j = 0; j < n_features; j++) {
            for (da_int i = 0; i < n_components; i++) {
                std::cout << principal_components[n_components * i + j] << "  ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // Check against expected results
        double principal_components_exp[15] = {
            -0.14907884486130418,  -0.07220367025708045, -0.38718653977350936,
            -0.6612054163818867,   0.623738867070505,    -0.06907631947413592,
            -0.031706610956396264, 0.20952521660694667,  0.8854125206703791,
            -0.7289116905829763,   -0.6138062400926413,  0.1296593407398653,
            -0.09091387966203135,  0.4302063910917139,   -0.21106437194645863};
        double scores_exp[18] = {
            3.797261129593253,    -2.5006179943446254,  2.431393931595693,
            -3.383775820752579,   -2.0509494403116166,  1.7066881942198742,
            1.8917911630360351,   -0.14051085079306697, -0.48911894407452433,
            3.0345920645743383,   -2.9954589898464876,  -1.3012944428962916,
            -0.10695425296598449, 1.5602497256676358,   1.2837835252499912,
            -0.7771478863983585,  -0.5060720435855457,  -1.4538590679677388};
        double X_transform_exp[9] = {
            -3.250305270939447,  0.6691223004872521,   1.833601737126601,
            -2.1581247424555086, -0.21658703437771865, -0.2844305102179128,
            -1.9477723543266676, 1.7953216115607247,   -0.5561178355649032};

        double tol = 1.0e-14;
        double err = 0.0;
        for (da_int i = 0; i < principal_components_dim; i++)
            err = std::max(
                err, std::abs(principal_components[i] - principal_components_exp[i]));
        for (da_int i = 0; i < scores_dim; i++)
            err = std::max(err, std::abs(scores[i] - scores_exp[i]));
        for (da_int i = 0; i < m_samples * n_components; i++)
            err = std::max(err, std::abs(X_transform[i] - X_transform_exp[i]));
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
    delete[] principal_components;
    delete[] scores;
    delete[] X_transform;

    std::cout << "-----------------------------------------------------------------------"
              << std::endl;

    return exit_code;
}
