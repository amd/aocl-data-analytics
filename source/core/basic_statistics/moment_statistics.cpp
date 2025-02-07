/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclda.h"
#include "basic_statistics.hpp"
#include "macros.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace ARCH {

namespace da_basic_statistics {

/* Compute double/float raised to positive integer power efficiently by binary powering */
template <typename T> T power(T a, da_int exponent) {
    T result = (T)1.0;
    da_int current_exponent = exponent;
    T current_base = a;
    while (current_exponent > 0) {
        // If bit in unit place is 1, multiply result by appropriate repeated squared a
        if (current_exponent & 1)
            result *= current_base;
        current_base *= current_base;
        current_exponent >>= 1;
    }

    return result;
}

/* Arithmetic mean along specified axis */
template <typename T>
da_status mean(da_order order, da_axis axis_in, da_int n_in, da_int p_in, const T *x,
               da_int ldx, T *amean) {

    da_int n, p;
    da_axis axis;

    // If we are in row-major we can switch the axis and n and p and work as if we were in column-major
    da_status status = row_to_col_major(order, axis_in, n_in, p_in, ldx, axis, n, p);
    if (status != da_status_success)
        return status;

    if (x == nullptr || amean == nullptr)
        return da_status_invalid_pointer;

    T zero = (T)0.0;

    switch (axis) {
    case da_axis_row:
        std::fill(amean, amean + n, 0.0);

        for (da_int i = 0; i < p; i++) {
            for (da_int j = 0; j < n; j++) {
                amean[j] += x[j + ldx * i];
            }
        }

        for (da_int j = 0; j < n; j++)
            amean[j] /= p;
        break;

    case da_axis_col:
        for (da_int i = 0; i < p; i++) {
            T tmp = zero;
#pragma omp simd reduction(+ : tmp)
            for (da_int j = 0; j < n; j++) {
                tmp += x[j + ldx * i];
            }
            amean[i] = tmp / n;
        }
        break;

    case da_axis_all:
        amean[0] = zero;
        for (da_int i = 0; i < p; i++) {
            for (da_int j = 0; j < n; j++) {
                amean[0] += x[j + ldx * i];
            }
        }
        amean[0] /= (n * p);
        break;

    default:
        return da_status_internal_error; // LCOV_EXCL_LINE
        break;
    }
    return da_status_success;
}

/* Geometric mean computed using log and exp to avoid overflow. Care needed to deal with negative or zero entries */
template <typename T>
da_status geometric_mean(da_order order, da_axis axis_in, da_int n_in, da_int p_in,
                         const T *x, da_int ldx, T *gmean) {

    da_int n, p;
    da_axis axis;

    // If we are in row-major we can switch the axis and n and p and work as if we were in column-major
    da_status status = row_to_col_major(order, axis_in, n_in, p_in, ldx, axis, n, p);
    if (status != da_status_success)
        return status;

    if (x == nullptr || gmean == nullptr)
        return da_status_invalid_pointer;

    T zero = (T)0.0;

    switch (axis) {
    case da_axis_row:
        std::fill(gmean, gmean + n, 0.0);
        ;

        for (da_int i = 0; i < p; i++) {
            for (da_int j = 0; j < n; j++) {
                if (x[j + ldx * i] < zero)
                    return da_status_negative_data;
                gmean[j] += (x[j + ldx * i] == zero)
                                ? std::numeric_limits<T>::infinity() * (-1)
                                : log(x[j + ldx * i]);
            }
        }

        for (da_int j = 0; j < n; j++)
            gmean[j] = exp(gmean[j] / p);
        break;

    case da_axis_col:
        for (da_int i = 0; i < p; i++) {
            gmean[i] = zero;
            for (da_int j = 0; j < n; j++) {
                if (x[j + ldx * i] < zero)
                    return da_status_negative_data;
                gmean[i] += (x[j + ldx * i] == zero)
                                ? std::numeric_limits<T>::infinity() * (-1)
                                : log(x[j + ldx * i]);
            }
            gmean[i] = exp(gmean[i] / n);
        }
        break;

    case da_axis_all:
        gmean[0] = zero;
        for (da_int i = 0; i < p; i++) {
            for (da_int j = 0; j < n; j++) {
                if (x[j + ldx * i] < zero)
                    return da_status_negative_data;
                gmean[0] += (x[j + ldx * i] == zero)
                                ? std::numeric_limits<T>::infinity() * (-1)
                                : log(x[j + ldx * i]);
            }
        }
        gmean[0] = exp(gmean[0] / (n * p));
        break;

    default:
        return da_status_internal_error; // LCOV_EXCL_LINE
        break;
    }
    return da_status_success;
}

/* Harmonic mean along a specified axis */
template <typename T>
da_status harmonic_mean(da_order order, da_axis axis_in, da_int n_in, da_int p_in,
                        const T *x, da_int ldx, T *hmean) {

    da_int n, p;
    da_axis axis;

    // If we are in row-major we can switch the axis and n and p and work as if we were in column-major
    da_status status = row_to_col_major(order, axis_in, n_in, p_in, ldx, axis, n, p);
    if (status != da_status_success)
        return status;

    if (x == nullptr || hmean == nullptr)
        return da_status_invalid_pointer;

    T one = (T)1.0;
    T zero = (T)0.0;

    // Note: harmonic mean is defined for zero entries, but we need to guard against exceptions nonetheless
    switch (axis) {
    case da_axis_row:
        std::fill(hmean, hmean + n, 0.0);

        for (da_int i = 0; i < p; i++) {
            for (da_int j = 0; j < n; j++) {
                hmean[j] += (x[j + ldx * i] == zero) ? zero : one / x[j + ldx * i];
            }
        }

        for (da_int j = 0; j < n; j++)
            hmean[j] = (hmean[j] == zero) ? zero : p / hmean[j];
        break;

    case da_axis_col:
        for (da_int i = 0; i < p; i++) {
            hmean[i] = zero;
            for (da_int j = 0; j < n; j++) {
                hmean[i] += (x[j + ldx * i] == zero) ? zero : one / x[j + ldx * i];
            }
            hmean[i] = (hmean[i] == zero) ? zero : n / hmean[i];
        }
        break;

    case da_axis_all:
        hmean[0] = zero;
        for (da_int i = 0; i < p; i++) {
            for (da_int j = 0; j < n; j++) {
                hmean[0] += (x[j + ldx * i] == zero) ? zero : one / x[j + ldx * i];
            }
        }
        hmean[0] = (hmean[0] == zero) ? zero : (n * p) / hmean[0];
        break;

    default:
        return da_status_internal_error; // LCOV_EXCL_LINE
        break;
    }
    return da_status_success;
}

/* Mean and variance along specified axis */
template <typename T>
da_status variance(da_order order, da_axis axis_in, da_int n_in, da_int p_in, const T *x,
                   da_int ldx, da_int dof, T *amean, T *var) {

    da_int n, p;
    da_axis axis;

    // If we are in row-major we can switch the axis and n and p and work as if we were in column-major
    da_status status = row_to_col_major(order, axis_in, n_in, p_in, ldx, axis, n, p);
    if (status != da_status_success)
        return status;

    if (x == nullptr || amean == nullptr || var == nullptr)
        return da_status_invalid_pointer;

    mean(column_major, axis, n, p, x, ldx, amean);

    T zero = (T)0.0;
    da_int scale_factor = dof;

    // There is potential vectorization of the loops in the mean and variance computations here
    switch (axis) {
    case da_axis_row:
        std::fill(var, var + n, 0.0);

        for (da_int i = 0; i < p; i++) {
            for (da_int j = 0; j < n; j++) {
                var[j] += (x[j + ldx * i] - amean[j]) * (x[j + ldx * i] - amean[j]);
            }
        }

        if (dof < 0) {
            scale_factor = p;
        } else if (dof == 0) {
            scale_factor = p - 1;
        }

        if (scale_factor > 1) {
            for (da_int j = 0; j < n; j++)
                var[j] /= scale_factor;
        }
        break;

    case da_axis_col:
        for (da_int i = 0; i < p; i++) {
            T tmp = zero;
#pragma omp simd reduction(+ : tmp)
            for (da_int j = 0; j < n; j++) {
                tmp += (x[j + ldx * i] - amean[i]) * (x[j + ldx * i] - amean[i]);
            }
            var[i] = tmp;

            if (dof < 0) {
                scale_factor = n;
            } else if (dof == 0) {
                scale_factor = n - 1;
            }

            if (scale_factor > 1)
                var[i] /= scale_factor;
        }
        break;

    case da_axis_all:
        var[0] = zero;
        for (da_int i = 0; i < p; i++) {
            for (da_int j = 0; j < n; j++) {
                var[0] += (x[j + ldx * i] - amean[0]) * (x[j + ldx * i] - amean[0]);
            }
        }

        if (dof < 0) {
            scale_factor = n * p;
        } else if (dof == 0) {
            scale_factor = n * p - 1;
        }

        if (scale_factor > 1)
            var[0] /= scale_factor;

        break;
    default:
        return da_status_internal_error; // LCOV_EXCL_LINE
        break;
    }
    return da_status_success;
}

/* Mean, variance and skewness along specified axis */
template <typename T>
da_status skewness(da_order order, da_axis axis_in, da_int n_in, da_int p_in, const T *x,
                   da_int ldx, T *amean, T *var, T *skew) {

    da_int n, p;
    da_axis axis;

    // If we are in row-major we can switch the axis and n and p and work as if we were in column-major
    da_status status = row_to_col_major(order, axis_in, n_in, p_in, ldx, axis, n, p);
    if (status != da_status_success)
        return status;

    if (x == nullptr || amean == nullptr || var == nullptr || skew == nullptr)
        return da_status_invalid_pointer;

    mean(column_major, axis, n, p, x, ldx, amean);

    T zero = (T)0.0;

    switch (axis) {
    case da_axis_row: {
        std::fill(var, var + n, 0.0);
        std::fill(skew, skew + n, 0.0);

        for (da_int i = 0; i < p; i++) {
            T tmp, tmp2;
            for (da_int j = 0; j < n; j++) {
                tmp = x[j + ldx * i] - amean[j];
                tmp2 = tmp * tmp;
                var[j] += tmp2;
                skew[j] += tmp2 * tmp;
            }
        }

        T sqrtp = std::sqrt((T)p);

        for (da_int j = 0; j < n; j++) {
            skew[j] = (var[j] == zero) ? zero : skew[j] * sqrtp / pow(var[j], (T)1.5);
            var[j] /= p;
        }
        break;
    }
    case da_axis_col: {
        T sqrtn = std::sqrt((T)n);

        for (da_int i = 0; i < p; i++) {
            var[i] = zero;
            skew[i] = zero;
            T tmp, tmp2;
            for (da_int j = 0; j < n; j++) {
                tmp = x[j + ldx * i] - amean[i];
                tmp2 = tmp * tmp;
                var[i] += tmp2;
                skew[i] += tmp2 * tmp;
            }
            skew[i] = (var[i] == zero) ? zero : skew[i] * sqrtn / pow(var[i], (T)1.5);
            var[i] /= n;
        }
        break;
    }
    case da_axis_all:
        var[0] = zero;
        skew[0] = zero;
        T tmp, tmp2;
        for (da_int i = 0; i < p; i++) {
            for (da_int j = 0; j < n; j++) {
                tmp = x[j + ldx * i] - amean[0];
                tmp2 = tmp * tmp;
                var[0] += tmp2;
                skew[0] += tmp2 * tmp;
            }
        }
        skew[0] =
            (var[0] == zero) ? zero : skew[0] * (T)(sqrt(n * p)) / pow(var[0], (T)1.5);
        var[0] /= n * p;
        break;
    default:
        return da_status_internal_error; // LCOV_EXCL_LINE
        break;
    }
    return da_status_success;
}

/* Mean, variance and kurtosis along specified axis */
template <typename T>
da_status kurtosis(da_order order, da_axis axis_in, da_int n_in, da_int p_in, const T *x,
                   da_int ldx, T *amean, T *var, T *kurt) {

    da_int n, p;
    da_axis axis;

    // If we are in row-major we can switch the axis and n and p and work as if we were in column-major
    da_status status = row_to_col_major(order, axis_in, n_in, p_in, ldx, axis, n, p);
    if (status != da_status_success)
        return status;

    if (x == nullptr || amean == nullptr || var == nullptr || kurt == nullptr)
        return da_status_invalid_pointer;

    mean(column_major, axis, n, p, x, ldx, amean);

    T zero = (T)0.0;
    T three = (T)3.0;

    switch (axis) {
    case da_axis_row:
        std::fill(var, var + n, 0.0);
        std::fill(kurt, kurt + n, 0.0);

        for (da_int i = 0; i < p; i++) {
            T tmp, tmp2;
            for (da_int j = 0; j < n; j++) {
                tmp = x[j + ldx * i] - amean[j];
                tmp2 = tmp * tmp;
                var[j] += tmp2;
                kurt[j] += tmp2 * tmp2;
            }
        }

        for (da_int j = 0; j < n; j++) {
            kurt[j] = (var[j] == zero) ? -three : p * kurt[j] / (var[j] * var[j]) - three;
            var[j] /= p;
        }
        break;

    case da_axis_col:
        for (da_int i = 0; i < p; i++) {
            var[i] = zero;
            kurt[i] = zero;
            T tmp, tmp2;
            for (da_int j = 0; j < n; j++) {
                tmp = x[j + ldx * i] - amean[i];
                tmp2 = tmp * tmp;
                var[i] += tmp2;
                kurt[i] += tmp2 * tmp2;
            }
            kurt[i] = (var[i] == zero) ? -three : n * kurt[i] / (var[i] * var[i]) - three;
            var[i] /= n;
        }
        break;

    case da_axis_all:
        var[0] = zero;
        kurt[0] = zero;
        T tmp, tmp2;
        for (da_int i = 0; i < p; i++) {
            for (da_int j = 0; j < n; j++) {
                tmp = x[j + ldx * i] - amean[0];
                tmp2 = tmp * tmp;
                var[0] += tmp2;
                kurt[0] += tmp2 * tmp2;
            }
        }
        kurt[0] = (var[0] == zero) ? -three : n * p * kurt[0] / (var[0] * var[0]) - three;
        var[0] /= n * p;
        break;
    default:
        return da_status_internal_error; // LCOV_EXCL_LINE
        break;
    }
    return da_status_success;
}

/* kth moment along specified axis. Optionally use precomputed mean. */
template <typename T>
da_status moment(da_order order, da_axis axis_in, da_int n_in, da_int p_in, const T *x,
                 da_int ldx, da_int k, da_int use_precomputed_mean, T *amean, T *mom) {

    da_int n, p;
    da_axis axis;

    // If we are in row-major we can switch the axis and n and p and work as if we were in column-major
    da_status status = row_to_col_major(order, axis_in, n_in, p_in, ldx, axis, n, p);
    if (status != da_status_success)
        return status;

    if (k < 0)
        return da_status_invalid_input;
    if (x == nullptr || amean == nullptr || mom == nullptr)
        return da_status_invalid_pointer;

    if (!use_precomputed_mean)
        mean(column_major, axis, n, p, x, ldx, amean);

    T zero = (T)0.0;

    switch (axis) {
    case da_axis_row:
        std::fill(mom, mom + n, 0.0);

        for (da_int i = 0; i < p; i++) {
            for (da_int j = 0; j < n; j++) {
                mom[j] += power(x[j + ldx * i] - amean[j], k);
            }
        }

        for (da_int j = 0; j < n; j++)
            mom[j] /= p;
        break;

    case da_axis_col:
        for (da_int i = 0; i < p; i++) {
            mom[i] = zero;
            for (da_int j = 0; j < n; j++) {
                mom[i] += power(x[j + ldx * i] - amean[i], k);
            }
            mom[i] /= n;
        }
        break;

    case da_axis_all:
        mom[0] = zero;
        for (da_int i = 0; i < p; i++) {
            for (da_int j = 0; j < n; j++) {
                mom[0] += power(x[j + ldx * i] - amean[0], k);
            }
        }
        mom[0] /= (n * p);
        break;

    default:
        return da_status_internal_error; // LCOV_EXCL_LINE
        break;
    }
    return da_status_success;
}

// Explicit template instantiations
template double power<double>(double a, da_int exponent);
template float power<float>(float a, da_int exponent);
template da_status mean<double>(da_order order, da_axis axis_in, da_int n_in, da_int p_in,
                                const double *x, da_int ldx, double *amean);
template da_status mean<float>(da_order order, da_axis axis_in, da_int n_in, da_int p_in,
                               const float *x, da_int ldx, float *amean);
template da_status geometric_mean<double>(da_order order, da_axis axis_in, da_int n_in,
                                          da_int p_in, const double *x, da_int ldx,
                                          double *gmean);
template da_status geometric_mean<float>(da_order order, da_axis axis_in, da_int n_in,
                                         da_int p_in, const float *x, da_int ldx,
                                         float *gmean);
template da_status harmonic_mean<double>(da_order order, da_axis axis_in, da_int n_in,
                                         da_int p_in, const double *x, da_int ldx,
                                         double *hmean);
template da_status harmonic_mean<float>(da_order order, da_axis axis_in, da_int n_in,
                                        da_int p_in, const float *x, da_int ldx,
                                        float *hmean);
template da_status variance<double>(da_order order, da_axis axis_in, da_int n_in,
                                    da_int p_in, const double *x, da_int ldx, da_int dof,
                                    double *amean, double *var);
template da_status variance<float>(da_order order, da_axis axis_in, da_int n_in,
                                   da_int p_in, const float *x, da_int ldx, da_int dof,
                                   float *amean, float *var);
template da_status skewness<double>(da_order order, da_axis axis_in, da_int n_in,
                                    da_int p_in, const double *x, da_int ldx,
                                    double *amean, double *var, double *skew);
template da_status skewness<float>(da_order order, da_axis axis_in, da_int n_in,
                                   da_int p_in, const float *x, da_int ldx, float *amean,
                                   float *var, float *skew);
template da_status kurtosis<double>(da_order order, da_axis axis_in, da_int n_in,
                                    da_int p_in, const double *x, da_int ldx,
                                    double *amean, double *var, double *kurt);
template da_status kurtosis<float>(da_order order, da_axis axis_in, da_int n_in,
                                   da_int p_in, const float *x, da_int ldx, float *amean,
                                   float *var, float *kurt);
template da_status moment<double>(da_order order, da_axis axis_in, da_int n_in,
                                  da_int p_in, const double *x, da_int ldx, da_int k,
                                  da_int use_precomputed_mean, double *amean,
                                  double *mom);
template da_status moment<float>(da_order order, da_axis axis_in, da_int n_in,
                                 da_int p_in, const float *x, da_int ldx, da_int k,
                                 da_int use_precomputed_mean, float *amean, float *mom);

} // namespace da_basic_statistics

} // namespace ARCH