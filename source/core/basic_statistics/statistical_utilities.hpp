#ifndef STATISTICAL_UTILITIES_HPP
#define STATISTICAL_UTILITIES_HPP

#include "aoclda.h"
#include "moment_statistics.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace da_basic_statistics {

/* Standardize a data array x by shifting and scaling */
template <typename T>
da_status standardize(da_axis axis, da_int n, da_int p, T *x, da_int ldx, T *shift,
                      T *scale) {

    da_status status = da_status_success;

    if (ldx < n)
        return da_status_invalid_leading_dimension;
    if (n < 1 || p < 1)
        return da_status_invalid_array_dimension;
    if (x == nullptr)
        return da_status_invalid_pointer;

    T *amean = nullptr, *var = nullptr;
    T **internal_shift, **internal_scale;

    internal_shift = &shift;
    internal_scale = &scale;

    // By computing length we can avoid duplicating some of the logic for the different axis types
    da_int length;
    switch (axis) {
    case da_axis_col:
        length = p;
        break;
    case da_axis_row:
        length = n;
        break;
    case da_axis_all:
        length = 1;
        break;
    default:
        return da_status_internal_error; // LCOV_EXCL_LINE
        break;
    }

    if (shift == nullptr && scale == nullptr) {
        // If both input arrays are null, get the mean and variance
        try {
            amean = new T[length];
            var = new T[length];
        } catch (std::bad_alloc const &) {
            return da_status_memory_error;
        }

        status = variance(axis, n, p, x, ldx, amean, var);
        if (status != da_status_success)
            goto exit;

        for (da_int i = 0; i < length; i++) {
            var[i] = std::sqrt(var[i]);
        }

        internal_scale = &var;
        internal_shift = &amean;
    } else if (shift == nullptr) {
        // Shift is null so set internal shift to 0
        try {
            amean = new T[length];
            std::fill(amean, amean + length, 0.0);
        } catch (std::bad_alloc const &) {
            return da_status_memory_error;
        }
        internal_shift = &amean;

    } else if (scale == nullptr) {
        // Scale is null so set internal scale to 1
        try {
            var = new T[length];
            std::fill(var, var + length, 1.0);
        } catch (std::bad_alloc const &) {
            return da_status_memory_error;
        }
        internal_scale = &var;
    }

    // FIXME post 4.2: investigate vectorization of these loops
    T tmp_scale;

    switch (axis) {
    case da_axis_col:
        for (da_int i = 0; i < p; i++) {
            tmp_scale = (*internal_scale)[i] != 0 ? (*internal_scale)[i] : (T)1.0;
            for (da_int j = 0; j < n; j++) {
                x[i * ldx + j] = (x[i * ldx + j] - (*internal_shift)[i]) / tmp_scale;
            }
        }
        break;
    case da_axis_row:
        for (da_int i = 0; i < p; i++) {
            for (da_int j = 0; j < n; j++) {
                tmp_scale = (*internal_scale)[j] != 0 ? (*internal_scale)[j] : (T)1.0;
                x[i * ldx + j] = (x[i * ldx + j] - (*internal_shift)[j]) / tmp_scale;
            }
        }
        break;
    case da_axis_all:
        tmp_scale = (*internal_scale)[0] != 0 ? (*internal_scale)[0] : (T)1.0;
        for (da_int i = 0; i < p; i++) {
            for (da_int j = 0; j < n; j++) {
                x[i * ldx + j] = (x[i * ldx + j] - (*internal_shift)[0]) / tmp_scale;
            }
        }

        break;
    default:
        status = da_status_internal_error; // LCOV_EXCL_LINE
        goto exit; // LCOV_EXCL_LINE
    }

exit:
    delete[] amean;
    delete[] var;
    return status;
}

} // namespace da_basic_statistics

#endif