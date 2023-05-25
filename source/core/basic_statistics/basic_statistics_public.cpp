#include "aoclda.h"
#include "da_mean.hpp"

da_status da_mean_s(da_int n, float *x, da_int incx, float *mean) {
    return da_mean(n, x, incx, mean);
}

da_status da_mean_d(da_int n, double *x, da_int incx, double *mean) {
    return da_mean(n, x, incx, mean);
}

da_status da_colmean_s(da_int n, da_int p, float *x, da_int incx, float *mean) {
    return da_colmean(n, p, x, incx, mean);
}

da_status da_colmean_d(da_int n, da_int p, double *x, da_int incx, double *mean) {
    return da_colmean(n, p, x, incx, mean);
}
