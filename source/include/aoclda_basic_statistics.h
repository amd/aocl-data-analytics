#ifndef AOCLDA_BASICSTATS
#define AOCLDA_BASICSTATS

#include "aoclda_error.h"
#ifdef __cplusplus
extern "C" {
#endif

da_status da_mean_s(int n, float *x, int incx, float *mean);
da_status da_mean_d(int n, double *x, int incx, double *mean);

#ifdef __cplusplus
}
#endif

#endif