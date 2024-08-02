/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <vector>

// This example demonstrates how to compute different kernel functions.
// We use small arrays X and Y, then compute each kernel's matrix D.

static void print_matrix(const double *M, da_int rows, da_int cols) {
    for (da_int i = 0; i < rows; i++) {
        for (da_int j = 0; j < cols; j++) {
            std::cout << std::setw(10) << M[i + j * rows] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {

    std::cout << "------------------------" << std::endl;
    std::cout << "Kernel Functions Example" << std::endl;
    std::cout << "------------------------" << std::endl;

    // Example data in row-major order
    // Let's have X be 3 samples (rows) by 2 features (columns).
    double X[6] = {1.0, 2.0, -1.0, 0.0, 2.0, -2.0};

    // Y is 2 samples, 2 features
    double Y[4] = {0.5, -0.5, 1.0, 3.0};

    da_order order = row_major;

    // Dimensions
    da_int m = 3; // rows of X
    da_int n = 2; // rows of Y
    da_int k = 2; // number of features in both X and Y

    // ldd depends on whether we compute m x n or m x m
    da_int ldx = k;
    da_int ldy = k;

    // We'll compute a D matrix of size m x n => 3x2
    std::vector<double> D(m * n, 0.0);
    da_int ldd = n;

    bool pass = true;
    int exit_code = 0;

    /*************************************************
     * LINEAR kernel
     *************************************************/
    std::cout << "Computing Linear Kernel (double) ..." << std::endl;

    da_status status = da_linear_kernel_d(order, m, n, k, X, ldx, Y, ldy, D.data(), ldd);

    if (status != da_status_success) {
        std::cout << "Error computing linear kernel" << std::endl;
        pass = false;
    } else {
        std::cout << "Resulting Linear Kernel Matrix (3 x 2):" << std::endl;
        print_matrix(D.data(), m, n);
    }
    std::cout << std::endl;

    /*************************************************
     * RBF kernel
     *************************************************/
    std::cout << "Computing RBF Kernel (double) ..." << std::endl;

    double gamma_rbf = 0.5;
    std::fill(D.begin(), D.end(), 0.0);

    status = da_rbf_kernel_d(order, m, n, k, X, ldx, Y, ldy, D.data(), ldd, gamma_rbf);

    if (status != da_status_success) {
        std::cout << "Error computing RBF kernel" << std::endl;
        pass = false;
    } else {
        std::cout << "Resulting RBF Kernel Matrix (3 x 2):" << std::endl;
        print_matrix(D.data(), m, n);
    }
    std::cout << std::endl;

    /*************************************************
     * Polynomial kernel
     *************************************************/
    // Now compute kernel matrix with itself
    std::vector<double> D_with_itself(m * m, 0.0);
    ldd = m;
    std::cout << "Computing Polynomial Kernel (double) ..." << std::endl;

    double gamma_poly = 1.0;
    double coef0 = 1.0;
    da_int degree = 2;

    status = da_polynomial_kernel_d(order, m, n, k, X, ldx, nullptr, ldy,
                                    D_with_itself.data(), ldd, gamma_poly, degree, coef0);

    if (status != da_status_success) {
        std::cout << "Error computing polynomial kernel" << std::endl;
        pass = false;
    } else {
        std::cout << "Resulting Polynomial Kernel Matrix (3 x 3):" << std::endl;
        print_matrix(D_with_itself.data(), m, m);
    }
    std::cout << std::endl;

    /*************************************************
     * Sigmoid kernel
     *************************************************/
    std::cout << "Computing Sigmoid Kernel (double) ..." << std::endl;

    double gamma_sig = 0.2;
    double coef0_sig = 0.0;
    std::fill(D_with_itself.begin(), D_with_itself.end(), 0.0);

    status = da_sigmoid_kernel_d(order, m, n, k, X, ldx, nullptr, ldy,
                                 D_with_itself.data(), ldd, gamma_sig, coef0_sig);

    if (status != da_status_success) {
        std::cout << "Error computing sigmoid kernel" << std::endl;
        pass = false;
    } else {
        std::cout << "Resulting Sigmoid Kernel Matrix (3 x 3):" << std::endl;
        print_matrix(D_with_itself.data(), m, m);
    }
    std::cout << std::endl;

    if (pass) {
        std::cout << "All kernel computations completed successfully." << std::endl;
    } else {
        exit_code = 1;
        std::cout << "Some kernel computations failed. Check above error messages."
                  << std::endl;
    }

    return exit_code;
}