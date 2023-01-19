#include "aoclda.h"
#include "da_mean.hpp"

da_status da_mean_s(int n, float *x, int incx, float *mean) {
    return da_mean(n, x, incx, mean);
}

da_status da_mean_d(int n, double *x, int incx, double *mean) {
    return da_mean(n, x, incx, mean);
}