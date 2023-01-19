#include "aoclda.h"
#include <cmath>
#include <iostream>

int main() {
    int n = 4;
    double *x = nullptr;

    x = (double *)malloc(n * sizeof(double));

    // Starting point
    for (int i = 0; i < n; i++)
        x[i] = (double)i;

    double mean;

    // int err = 1;
    int err = da_mean_d(n, x, 1, &mean);

    if (!err) {
        std::cout << "mean = " << mean << std::endl;
    } else {
        std::cout << "error computing mean" << std::endl;
    }

    if (x)
        free(x);
}