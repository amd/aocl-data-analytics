#ifndef AOCLDA_BASICSTATS
#define AOCLDA_BASICSTATS

#include "aoclda_error.h"
#include "aoclda_types.h"
#ifdef __cplusplus
extern "C" {
#endif

da_status da_mean_s(da_int n, float *x, da_int incx, float *mean);
da_status da_mean_d(da_int n, double *x, da_int incx, double *mean);

#ifdef __cplusplus
}
#endif

#endif