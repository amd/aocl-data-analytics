/*
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef STATISTICAL_UTILITIES_HPP
#define STATISTICAL_UTILITIES_HPP

#include "aoclda.h"
#include "moment_statistics.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace da_basic_statistics {

template <typename T> bool is_zero(T *arr, da_int len) {

    T zero = (T)0.0;

    for (da_int i = 0; i < len; i++) {
        if (arr[i] != zero)
            return false;
    }
    return true;
}

/* Standardize a data array x by shifting and scaling */
template <typename T>
da_status standardize(da_axis axis, da_int n, da_int p, T *x, da_int ldx, da_int dof,
                      da_int mode, T *shift, T *scale) {

    da_status status = da_status_success;

    if (ldx < n)
        return da_status_invalid_leading_dimension;
    if (n < 1 || p < 1)
        return da_status_invalid_array_dimension;
    if (x == nullptr)
        return da_status_invalid_pointer;
    if (mode != 0 && mode != 1)
        return da_status_invalid_input;

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

    da_int scale_factor = dof;
    if (dof < 0) {
        scale_factor = length;
    } else if (dof == 0) {
        scale_factor = length - 1;
    }

    // These booleans will help us determine which route the computation should take
    bool shift_is_null = true;
    bool shift_is_zero = false;
    bool scale_is_null = true;
    bool scale_is_zero = false;

    if (shift != nullptr) {
        shift_is_null = false;
        shift_is_zero = is_zero(shift, length);
    }

    if (scale != nullptr) {
        scale_is_null = false;
        scale_is_zero = is_zero(scale, length);
    }

    if (shift_is_null && scale_is_null) {
        // If both input arrays are null, get the mean and variance
        try {
            amean = new T[length];
            var = new T[length];
        } catch (std::bad_alloc const &) {
            return da_status_memory_error; // LCOV_EXCL_LINE
        }

        status = variance(axis, n, p, x, ldx, dof, amean, var);
        if (status != da_status_success) {
            delete[] amean;
            delete[] var;
            return status;
        }

        for (da_int i = 0; i < length; i++) {
            var[i] = std::sqrt(var[i]);
        }

        internal_scale = &var;
        internal_shift = &amean;
    } else if (shift_is_null) {
        // Shift is null so set internal shift to 0
        try {
            amean = new T[length];
        } catch (std::bad_alloc const &) {
            return da_status_memory_error; // LCOV_EXCL_LINE
        }
        // If we need to compute standard deviations, do so
        if (scale_is_zero) {
            status = variance(axis, n, p, x, ldx, dof, amean, scale);
            if (status != da_status_success) {
                delete[] amean;
                delete[] var;
                return status;
            }
            for (da_int i = 0; i < length; i++) {
                scale[i] = std::sqrt(scale[i]);
            }
        }
        std::fill(amean, amean + length, 0.0);
        internal_shift = &amean;

    } else if (scale_is_null) {
        // Scale is null so set internal scale to 1
        try {
            var = new T[length];
        } catch (std::bad_alloc const &) {
            return da_status_memory_error; // LCOV_EXCL_LINE
        }
        // If we need to compute means, do so
        if (shift_is_zero) {
            status = mean(axis, n, p, x, ldx, shift);
            if (status != da_status_success) {
                delete[] amean;
                delete[] var;
                return status;
            }
        }
        std::fill(var, var + length, 1.0);
        internal_scale = &var;
    } else {
        // Shift and scale are both to be used, but one of both might contain only zeros
        if (shift_is_zero && scale_is_zero) {
            status = variance(axis, n, p, x, ldx, dof, shift, scale);
            if (status != da_status_success) {
                delete[] amean;
                delete[] var;
                return status;
            }
            for (da_int i = 0; i < length; i++) {
                scale[i] = std::sqrt(scale[i]);
            }
        } else if (shift_is_zero) {
            status = mean(axis, n, p, x, ldx, shift);
            if (status != da_status_success) {
                delete[] amean;
                delete[] var;
                return status;
            }
        } else if (scale_is_zero) {
            // Compute standard deviations about the supplied means
            status = moment(axis, n, p, x, ldx, 2, 1, shift, scale);
            if (status != da_status_success) {
                delete[] amean;
                delete[] var;
                return status;
            }
            for (da_int i = 0; i < length; i++) {
                scale[i] = std::sqrt(length * scale[i] / scale_factor);
            }
        }
    }

    // Some of the loops below could be investigated for vectorization
    T tmp_scale;

    switch (axis) {
    case da_axis_col:
        for (da_int i = 0; i < p; i++) {
            tmp_scale = (*internal_scale)[i] != 0 ? (*internal_scale)[i] : (T)1.0;
            if (mode == 0) {
                for (da_int j = 0; j < n; j++) {
                    x[i * ldx + j] = (x[i * ldx + j] - (*internal_shift)[i]) / tmp_scale;
                }
            } else {
                for (da_int j = 0; j < n; j++) {
                    x[i * ldx + j] = x[i * ldx + j] * tmp_scale + (*internal_shift)[i];
                }
            }
        }
        break;
    case da_axis_row:
        if (mode == 0) {
            for (da_int i = 0; i < p; i++) {
                for (da_int j = 0; j < n; j++) {
                    tmp_scale = (*internal_scale)[j] != 0 ? (*internal_scale)[j] : (T)1.0;
                    x[i * ldx + j] = (x[i * ldx + j] - (*internal_shift)[j]) / tmp_scale;
                }
            }
        } else {
            for (da_int i = 0; i < p; i++) {
                for (da_int j = 0; j < n; j++) {
                    tmp_scale = (*internal_scale)[j] != 0 ? (*internal_scale)[j] : (T)1.0;
                    x[i * ldx + j] = x[i * ldx + j] * tmp_scale + (*internal_shift)[j];
                }
            }
        }
        break;
    case da_axis_all:
        tmp_scale = (*internal_scale)[0] != 0 ? (*internal_scale)[0] : (T)1.0;
        if (mode == 0) {
            for (da_int i = 0; i < p; i++) {
                for (da_int j = 0; j < n; j++) {
                    x[i * ldx + j] = (x[i * ldx + j] - (*internal_shift)[0]) / tmp_scale;
                }
            }
        } else {
            for (da_int i = 0; i < p; i++) {
                for (da_int j = 0; j < n; j++) {
                    x[i * ldx + j] = x[i * ldx + j] * tmp_scale + (*internal_shift)[0];
                }
            }
        }

        break;
    default:
        delete[] amean;                  // LCOV_EXCL_LINE
        delete[] var;                    // LCOV_EXCL_LINE
        return da_status_internal_error; // LCOV_EXCL_LINE
    }

    delete[] amean;
    delete[] var;
    return status;
}

} // namespace da_basic_statistics

#endif