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
#include <cmath>
#include <iostream>

/* Basic statistics example
 *
 * This example computes descriptive statistics
 * for a small dataset, together with the
 * correlation matrix and the standardized data
 */

int main() {

    std::cout << "-----------------------------------------------------------------------"
              << std::endl;
    std::cout << "Basic statistics" << std::endl;
    std::cout << "Descriptive statistics for a 4x5 data matrix" << std::endl << std::endl;
    std::cout << std::fixed;
    std::cout.precision(5);

    // Problem data
    double X[20]{1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 8.0,
                 4.0, 6.0, 9.0, 5.0, 4.0, 3.0, 1.0, 1.0, 2.0, 2.0};
    da_int n_rows = 4, n_cols = 5, ldx = 4, ldcov = 5, dof = 0, mode = 0;

    // Arrays for output data
    double harmonic_mean[5], mean[4], variance[4], kurtosis[4];
    double minimum[1], lower_hinge[1], median[1], upper_hinge[1], maximum[1];
    double cov[25];
    double *dummy = nullptr;

    // Arrays for expected outputs
    double harmonic_mean_exp[5]{1.92, 1.92, 3.84, 4.472049689440994, 1.3333333333333333};
    double mean_exp[4]{3.4, 3.8, 3., 3.2};
    double variance_exp[4]{9.04, 6.16, 0.8, 2.96};
    double kurtosis_exp[4]{-0.4210588143159213, -0.9675324675324677, -1.7500000000000002,
                           -1.005478451424398};
    double minimum_exp[1]{1}, lower_hinge_exp[1]{2}, median_exp[1]{3},
        upper_hinge_exp[1]{4}, maximum_exp[1]{9};
    double cov_exp[25]{// clang-format off
                     1.6666666666666665, -1.6666666666666665,  1.3333333333333333,
                     -3.1666666666666665,  0.6666666666666666, -1.6666666666666665,
                     1.6666666666666665, -1.3333333333333333,  3.1666666666666665,
                     -0.6666666666666666,  1.3333333333333333, -1.3333333333333333,
                     6.666666666666666,  -4.333333333333333,   0.0,
                     -3.1666666666666665,  3.1666666666666665 ,-4.333333333333333,
                     6.916666666666666,  -1.1666666666666665,  0.6666666666666666,
                     -0.6666666666666666,  0.0,                -1.1666666666666665,
                     0.3333333333333333 };
    // clang format on
    double X_exp[20]{-1.1618950038622251, -0.3872983346207417, 0.3872983346207417,
                     1.1618950038622251,  1.1618950038622251,  0.3872983346207417,
                     -0.3872983346207417, -1.1618950038622251, -1.1618950038622251,
                     1.1618950038622251,  -0.3872983346207417, 0.3872983346207417,
                     1.4258795636800752,  -0.0950586375786717, -0.4752931878933584,
                     -0.8555277382080452, -0.8660254037844387, -0.8660254037844387,
                     0.8660254037844387,  0.8660254037844387};

    int exit_code = 0;
    bool pass = true;

    // Compute column-wise harmonic means
    pass = pass && (da_harmonic_mean_d(da_axis_col, n_rows, n_cols, X, ldx,
                                       harmonic_mean) == da_status_success);

    // Compute row-wise mean, variance and kurtosis
    pass = pass && (da_kurtosis_d(da_axis_row, n_rows, n_cols, X, ldx, mean, variance,
                                  kurtosis) == da_status_success);

    // Compute overall max/min, median and hinges
    pass = pass && (da_five_point_summary_d(da_axis_all, n_rows, n_cols, X, ldx, minimum,
                                            lower_hinge, median, upper_hinge,
                                            maximum) == da_status_success);

    // Compute covariance matrix
    pass = pass && (da_covariance_matrix_d(n_rows, n_cols, X, ldx, dof, cov, ldcov) ==
                    da_status_success);

    // Standardize the original data matrix
    pass = pass && (da_standardize_d(da_axis_col, n_rows, n_cols, X, ldx, dof, mode,
                                     dummy, dummy) == da_status_success);

    // Check status (we could do this after every function call)
    if (pass) {
        std::cout << "Statistics computed successfully" << std::endl << std::endl;

        // Print computed statistics
        std::cout << "Column-wise harmonic means:  ";
        for (da_int i = 0; i < n_cols; i++)
            std::cout << "  " << harmonic_mean[i];
        std::cout << std::endl << std::endl;

        std::cout << "Row-wise means:  ";
        for (da_int i = 0; i < n_rows; i++)
            std::cout << "  " << mean[i];
        std::cout << std::endl << std::endl;

        std::cout << "Row-wise variances:  ";
        for (da_int i = 0; i < n_rows; i++)
            std::cout << "  " << variance[i];
        std::cout << std::endl << std::endl;

        std::cout << "Row-wise kurtoses:  ";
        for (da_int i = 0; i < n_rows; i++)
            std::cout << "  " << kurtosis[i];
        std::cout << std::endl << std::endl;

        std::cout << "Overall five-point summary statistics:  " << minimum[0] << "  "
                  << lower_hinge[0] << "  " << median[0] << "  " << upper_hinge[0] << "  "
                  << maximum[0] << std::endl
                  << std::endl;

        std::cout << "Covariance matrix:" << std::endl;
        for (da_int j = 0; j < n_cols; j++) {
            for (da_int i = 0; i < n_cols; i++) {
                std::cout << cov[ldcov * i + j] << "  ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Standardized data matrix:" << std::endl;
        for (da_int j = 0; j < n_rows; j++) {
            for (da_int i = 0; i < n_cols; i++) {
                std::cout << X[ldx * i + j] << "  ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // Check the outputs match the expected results
        double tol = 1.0e-14;
        double err = 0.0;
        for (da_int i = 0; i < n_cols; i++)
            err = std::max(err, std::abs(harmonic_mean[i] - harmonic_mean_exp[i]));
        for (da_int i = 0; i < n_rows; i++)
            err = std::max(err, std::abs(mean[i] - mean_exp[i]));
        for (da_int i = 0; i < n_rows; i++)
            err = std::max(err, std::abs(variance[i] - variance_exp[i]));
        for (da_int i = 0; i < n_rows; i++)
            err = std::max(err, std::abs(kurtosis[i] - kurtosis_exp[i]));
        err = std::max(err, std::abs(minimum[0] - minimum_exp[0]));
        err = std::max(err, std::abs(lower_hinge[0] - lower_hinge_exp[0]));
        err = std::max(err, std::abs(median[0] - median_exp[0]));
        err = std::max(err, std::abs(upper_hinge[0] - upper_hinge_exp[0]));
        err = std::max(err, std::abs(maximum[0] - maximum_exp[0]));
        for (da_int j = 0; j < n_cols; j++) {
            for (da_int i = 0; i < n_cols; i++) {
                err =
                    std::max(err, std::abs(cov[ldcov * j + i] - cov_exp[ldcov * j + i]));
            }
        }
        for (da_int j = 0; j < n_cols; j++) {
            for (da_int i = 0; i < n_rows; i++) {
                err = std::max(err, std::abs(X[ldx * j + i] - X_exp[ldx * j + i]));
            }
        }
        if (err > tol) {
            std::cout << "Solution is not within the expected tolerance: " << err
                      << std::endl;
            exit_code = 1;
        }

    } else {
        exit_code = 1;
    }

    std::cout << "-----------------------------------------------------------------------"
              << std::endl;

    return exit_code;
}