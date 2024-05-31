/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iostream>

/* Metrics example
 *
 * This example computes the euclidean distance matrix 
 * between two feature arrays
 */

int main() {
    std::cout << "-----------------------------------------------------------------------"
              << std::endl;
    std::cout << "Metrics" << std::endl;
    std::cout << "Euclidean distance matrix for a 3x2 and a 2x2 data matrix" << std::endl
              << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    int exit_code = 0;
    bool pass = true;

    // Feature data matrices
    double X[6] = {1.0, 3.0, 5.0, 2.0, 4.0, 6.0};
    double Y[4] = {7.0, 9.0, 8.0, 10.0};

    da_int m = 3, n = 2, k = 2;

    // Array used to store the distance matrix
    double D[6];

    // Compute the euclidean distance matrix
    pass = pass && (da_pairwise_distances_d(m, n, k, X, m, Y, n, D, m, da_euclidean,
                                            da_allow_infinite) == da_status_success);

    // Check status and print
    if (pass) {
        std::cout << "Euclidean distance matrix computed successfully" << std::endl
                  << std::endl;

        // Print computed matrix
        std::cout << "Distance matrix D:\n";
        for (da_int i = 0; i < m; i++) {
            for (da_int j = 0; j < n; j++)
                std::cout << "  " << D[i + j * m];
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // Check against expected results
        double sqrt2 = std::sqrt(2.0);
        double D_exp[6] = {6 * sqrt2, 4 * sqrt2, 2 * sqrt2,
                           8 * sqrt2, 6 * sqrt2, 4 * sqrt2};

        double tol = 1.0e-14;
        double err = 0.0;
        for (da_int i = 0; i < m * n; i++)
            err = std::max(err, std::abs(D[i] - D_exp[i]));

        if (err > tol) {
            std::cout << "Solution is not within expected tolerance. Maximum error is: "
                      << err << std::endl;
            exit_code = 1;
        }
    } else
        exit_code = 1;

    std::cout << "-----------------------------------------------------------------------"
              << std::endl;

    return exit_code;
}