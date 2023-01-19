#ifndef DA_MEAN_HPP
#define DA_MEAN_HPP

#include "aoclda.h"

template <typename T> da_status da_mean(int n, T *x, int incx, T *mean) {
    da_status err = da_status_success;

    *mean = (T)0.0;

    for (int i = 0; i < n; i++) {
        *mean += x[i];
    }

    *mean /= n;

    return err;
}

#endif