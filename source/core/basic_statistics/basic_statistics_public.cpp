#include "aoclda.h"
#include "da_mean.hpp"

da_status da_mean_s(da_int n, float *x, da_int incx, float *mean) {
    return da_mean(n, x, incx, mean);
}

da_status da_mean_d(da_int n, double *x, da_int incx, double *mean) {
    return da_mean(n, x, incx, mean);
}