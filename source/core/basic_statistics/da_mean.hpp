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

#endif