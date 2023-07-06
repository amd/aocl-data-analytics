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
    double x[20]{1, 2, 3, 4, 4, 3, 2, 1, 2, 8, 4, 6, 9, 5, 4, 3, 1, 1, 2, 2};
    da_int n = 4, p = 5, ldx = 4, ldcov = 5;

    // Arrays for output data
    double harmonic_mean[5], mean[4], variance[4], kurtosis[4];
    double minimum[1], lower_hinge[1], median[1], upper_hinge[1], maximum[1];
    double cov[25];
    double *dummy = nullptr;

    // Arrays for expected outputs
    double harmonic_mean_exp[5]{1.92, 1.92, 3.84, 4.472049689440994, 1.3333333333333333};
    double mean_exp[4]{3.4, 3.8, 3., 3.2};
    double variance_exp[4]{11.3, 7.7, 1.0, 3.7};
    double kurtosis_exp[4]{-0.4210588143159213, -0.9675324675324677, -1.7500000000000002,
                           -1.005478451424398};
    double minimum_exp[1]{1}, lower_hinge_exp[1]{2}, median_exp[1]{3},
        upper_hinge_exp[1]{4}, maximum_exp[1]{9};
    double cov_exp[25]{1.6666666666666665,
                       -1.6666666666666665,
                       1.3333333333333333,
                       -3.1666666666666665,
                       0.6666666666666666,
                       -1.6666666666666665,
                       1.6666666666666665,
                       -1.3333333333333333,
                       3.1666666666666665,
                       -0.6666666666666666,
                       1.3333333333333333,
                       -1.3333333333333333,
                       6.666666666666666,
                       -4.333333333333333,
                       0.0,
                       -3.1666666666666665,
                       3.1666666666666665,
                       -4.333333333333333,
                       6.916666666666666,
                       -1.1666666666666665,
                       0.6666666666666666,
                       -0.6666666666666666,
                       0.0,
                       -1.1666666666666665,
                       0.3333333333333333};
    double x_exp[20]{-1.1618950038622251, -0.3872983346207417, 0.3872983346207417,
                     1.1618950038622251,  1.1618950038622251,  0.3872983346207417,
                     -0.3872983346207417, -1.1618950038622251, -1.1618950038622251,
                     1.1618950038622251,  -0.3872983346207417, 0.3872983346207417,
                     1.4258795636800752,  -0.0950586375786717, -0.4752931878933584,
                     -0.8555277382080452, -0.8660254037844387, -0.8660254037844387,
                     0.8660254037844387,  0.8660254037844387};

    int exit_code = 0;
    bool pass = true;

    // Compute column-wise harmonic means
    pass = pass && (da_harmonic_mean_d(da_axis_col, n, p, x, ldx, harmonic_mean) ==
                    da_status_success);

    // Compute row-wise mean, variance and kurtosis
    pass = pass && (da_kurtosis_d(da_axis_row, n, p, x, ldx, mean, variance, kurtosis) ==
                    da_status_success);

    // Compute overall max/min, median and hinges
    pass = pass &&
           (da_five_point_summary_d(da_axis_all, n, p, x, ldx, minimum, lower_hinge,
                                    median, upper_hinge, maximum) == da_status_success);

    // Compute covariance matrix
    pass =
        pass && (da_covariance_matrix_d(n, p, x, ldx, cov, ldcov) == da_status_success);

    // Standardize the original data matrix
    pass = pass && (da_standardize_d(da_axis_col, n, p, x, ldx, dummy, dummy) ==
                    da_status_success);

    // Check status (we could do this after every function call)
    if (pass) {
        std::cout << "Statistics computed successfully" << std::endl << std::endl;

        // Print computed statistics
        std::cout << "Column-wise harmonic means:  ";
        for (da_int i = 0; i < p; i++)
            std::cout << "  " << harmonic_mean[i];
        std::cout << std::endl << std::endl;

        std::cout << "Row-wise means:  ";
        for (da_int i = 0; i < n; i++)
            std::cout << "  " << mean[i];
        std::cout << std::endl << std::endl;

        std::cout << "Row-wise variances:  ";
        for (da_int i = 0; i < n; i++)
            std::cout << "  " << variance[i];
        std::cout << std::endl << std::endl;

        std::cout << "Row-wise kurtoses:  ";
        for (da_int i = 0; i < n; i++)
            std::cout << "  " << kurtosis[i];
        std::cout << std::endl << std::endl;

        std::cout << "Overall five-point summary statistics:  " << minimum[0] << "  "
                  << lower_hinge[0] << "  " << median[0] << "  " << upper_hinge[0] << "  "
                  << maximum[0] << std::endl
                  << std::endl;

        std::cout << "Covariance matrix:" << std::endl;
        for (da_int j = 0; j < p; j++) {
            for (da_int i = 0; i < p; i++) {
                std::cout << cov[ldcov * i + j] << "  ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Standardized data matrix:" << std::endl;
        for (da_int j = 0; j < n; j++) {
            for (da_int i = 0; i < p; i++) {
                std::cout << x[ldx * i + j] << "  ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // Check the outputs match the expected results
        double tol = 1.0e-14;
        double err = 0.0;
        for (da_int i = 0; i < p; i++)
            err = std::max(err, std::abs(harmonic_mean[i] - harmonic_mean_exp[i]));
        for (da_int i = 0; i < n; i++)
            err = std::max(err, std::abs(mean[i] - mean_exp[i]));
        for (da_int i = 0; i < n; i++)
            err = std::max(err, std::abs(variance[i] - variance_exp[i]));
        for (da_int i = 0; i < n; i++)
            err = std::max(err, std::abs(kurtosis[i] - kurtosis_exp[i]));
        err = std::max(err, std::abs(minimum[0] - minimum_exp[0]));
        err = std::max(err, std::abs(lower_hinge[0] - lower_hinge_exp[0]));
        err = std::max(err, std::abs(median[0] - median_exp[0]));
        err = std::max(err, std::abs(upper_hinge[0] - upper_hinge_exp[0]));
        err = std::max(err, std::abs(maximum[0] - maximum_exp[0]));
        for (da_int j = 0; j < p; j++) {
            for (da_int i = 0; i < p; i++) {
                err =
                    std::max(err, std::abs(cov[ldcov * j + i] - cov_exp[ldcov * j + i]));
            }
        }
        for (da_int j = 0; j < p; j++) {
            for (da_int i = 0; i < n; i++) {
                err = std::max(err, std::abs(x[ldx * j + i] - x_exp[ldx * j + i]));
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