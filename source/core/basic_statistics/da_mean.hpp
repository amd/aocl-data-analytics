#ifndef AOCLDA_MEAN_HPP
#define AOCLDA_MEAN_HPP

#include "aoclda.h"

template <typename T>
da_status da_mean(da_int n, T *x, [[maybe_unused]] da_int incx, T *mean) {
    da_status err = da_status_success;

    *mean = (T)0.0;

    for (da_int i = 0; i < n; i++) {
        *mean += x[i];
    }

    *mean /= n;

    return err;
}

//Input X = nxp (rows x cols)
//Output colmean = 1xp
template <typename T>
da_status da_colmean(da_int n, da_int p, T *x, da_int incx, T *colmean) {
    da_status err = da_status_success;

    for (da_int i = 0; i < p; i++) {
        T mean = (T)0.0;

        for (da_int j = 0; j < n; j++) {
            mean += x[j * incx + i];
        }
        mean /= n;

        colmean[i] = mean;
    }

    return err;
}

#endif