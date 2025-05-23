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

// Deal with some Windows compilation issues regarding max/min macros
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "aoclda.h"
#include "basic_statistics.hpp"
#include "macros.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace ARCH {

namespace da_basic_statistics {

/* This routine uses the partial sort routine std::nth_element to correctly place the kth element of x.
   It uses the index array to do the sorting, so x is not itself reordered. */
template <typename T>
da_status indexed_partial_sort(const T *x, da_int length, da_int stride, da_int *xindex,
                               da_int k, da_int dim1, bool two_d, T &stat) {
    try {
        if (two_d) {
            // Deal with special case of 2d array, in which case stride corresponds to ldx
            std::nth_element(xindex, xindex + k, xindex + length,
                             [x, stride, dim1](da_int i, da_int j) {
                                 return x[stride * (i / dim1) + i % dim1] <
                                        x[stride * (j / dim1) + j % dim1];
                             });
            stat = x[stride * (xindex[k] / dim1) + xindex[k] % dim1];

        } else {
            std::nth_element(xindex, xindex + k, xindex + length,
                             [x, stride](da_int i, da_int j) {
                                 return x[i * stride] < x[j * stride];
                             });
            stat = x[xindex[k] * stride];
        }
        return da_status_success;
    } catch (std::bad_alloc const &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    }
}

/* Compute the qth quantile of x along the specified axis */
template <typename T>
da_status quantile(da_order order, da_axis axis_in, da_int n_in, da_int p_in, const T *x,
                   da_int ldx, T q, T *quant, da_quantile_type quantile_type) {

    da_int n, p;
    da_axis axis;

    // If we are in row-major we can switch the axis and n and p and work as if we were in column-major
    da_status status = row_to_col_major(order, axis_in, n_in, p_in, ldx, axis, n, p);
    if (status != da_status_success)
        return status;

    if (q < 0 || q > 1)
        return da_status_invalid_input;
    if (x == nullptr || quant == nullptr)
        return da_status_invalid_pointer;

    // With a little bit of logic here we can deal with the different choices of axis all in one go
    da_int num_stats, length, stride, spacing, dim1, dim2;
    bool two_d;

    switch (axis) {
    case da_axis_col:
        num_stats = p;
        dim1 = n;
        stride = 1;
        spacing = ldx;
        dim2 = 1;
        two_d = false;
        break;
    case da_axis_row:
        num_stats = n;
        dim1 = p;
        stride = ldx;
        spacing = 1;
        dim2 = 1;
        two_d = false;
        break;
    case da_axis_all:
        dim2 = p;
        num_stats = 1;
        dim1 = n;
        spacing = 1;
        stride = ldx;
        two_d = true;
        break;

    default:
        return da_status_internal_error; // LCOV_EXCL_LINE
        break;
    }

    T h;
    length = dim1 * dim2;

    // Special case of statistics based on a single element
    if (length == 1) {
        for (da_int i = 0; i < num_stats; i++)
            quant[i] = x[i * spacing];
        return da_status_success;
    }

    // We could combine some of these, but this is perhaps clearer
    switch (quantile_type) {
    case da_quantile_type_1:
        h = length * q;
        break;
    case da_quantile_type_2:
        h = length * q + (T)0.5;
        break;
    case da_quantile_type_3:
        h = length * q - (T)0.5;
        break;
    case da_quantile_type_4:
        h = length * q;
        break;
    case da_quantile_type_5:
        h = length * q + (T)0.5;
        break;
    case da_quantile_type_6:
        h = (length + 1) * q;
        break;
    case da_quantile_type_7:
        h = (length - 1) * q + 1;
        break;
    case da_quantile_type_8: {
        T third = (T)1.0 / (T)3.0;
        h = (length + third) * q + third;
        break;
    }
    case da_quantile_type_9:
        h = (length + (T)0.25) * q + (T)3.0 / (T)8.0;
        break;

    default:
        return da_status_internal_error; // LCOV_EXCL_LINE
        break;
    }

    // Account for 0 array indexing in C++
    h -= (T)1.0;

    // Declaring this vector allows us to sort in place, moving elements of xindex, instead of x
    da_int *xindex;
    try {
        xindex = new da_int[length];
    } catch (std::bad_alloc const &) {
        return da_status_memory_error;
    }

    da_int izero = 0;

    for (da_int i = 0; i < num_stats; i++) {

        for (da_int j = 0; j < length; j++)
            xindex[j] = j;

        // There are 4 possibilities for the precise logic of forming the statistic here; we need to use std::clamp to guard against illegal array indexing
        switch (quantile_type) {
        case da_quantile_type_1: {
            da_int hceil = std::clamp((da_int)std::ceil(h), izero, length - 1);
            status = indexed_partial_sort(&x[i * spacing], length, stride, xindex, hceil,
                                          dim1, two_d, quant[i]);
            break;
        }
        case da_quantile_type_2: {
            da_int h1 = std::clamp((da_int)std::ceil(h - (T)0.5), izero, length - 1);
            da_int h2 = std::clamp((da_int)std::floor(h + (T)0.5), izero, length - 1);
            if (h1 == h2) {
                status = indexed_partial_sort(&x[i * spacing], length, stride, xindex, h1,
                                              dim1, two_d, quant[i]);
            } else {
                T tmp1, tmp2;
                status = indexed_partial_sort(&x[i * spacing], length, stride, xindex, h1,
                                              dim1, two_d, tmp1);
                // h2 = h1+1 so just find the minimum value of the upper part of the array now
                status = indexed_partial_sort(&x[i * spacing], length - h1 - 1, stride,
                                              &xindex[h1 + 1], izero, dim1, two_d, tmp2);
                quant[i] = (T)0.5 * (tmp1 + tmp2);
            }
            break;
        }
        case da_quantile_type_3: {
            da_int hint = std::clamp((da_int)std::nearbyint(h), izero, length - 1);
            status = indexed_partial_sort(&x[i * spacing], length, stride, xindex, hint,
                                          dim1, two_d, quant[i]);
            break;
        }
        default: {
            da_int hceil = std::clamp((da_int)std::ceil(h), izero, length - 1);
            da_int hfloor = std::clamp((da_int)std::floor(h), izero, length - 1);
            if (hceil == hfloor) {
                status = indexed_partial_sort(&x[i * spacing], length, stride, xindex,
                                              hfloor, dim1, two_d, quant[i]);
            } else {
                T tmp1, tmp2;
                status = indexed_partial_sort(&x[i * spacing], length, stride, xindex,
                                              hfloor, dim1, two_d, tmp1);
                // hceil = hfloor+1 so just find the minimum value of the upper part of the array now
                status =
                    indexed_partial_sort(&x[i * spacing], length - hfloor - 1, stride,
                                         &xindex[hfloor + 1], izero, dim1, two_d, tmp2);
                quant[i] = tmp1 + (h - hfloor) * (tmp2 - tmp1);
            }
            break;
        }
        }

        if (status != da_status_success) {
            delete[] xindex;
            return status;
        }
    }

    delete[] xindex;
    return da_status_success;
}

/* Compute min/max, hinges and median along specified axis */
template <typename T>
da_status five_point_summary(da_order order, da_axis axis_in, da_int n_in, da_int p_in,
                             const T *x, da_int ldx, T *minimum, T *lower_hinge,
                             T *median, T *upper_hinge, T *maximum) {

    da_int n, p;
    da_axis axis;

    // If we are in row-major we can switch the axis and n and p and work as if we were in column-major
    da_status status = row_to_col_major(order, axis_in, n_in, p_in, ldx, axis, n, p);
    if (status != da_status_success)
        return status;

    // Quantile enables user to choose a method, but for this simple routine we will use a default of type 6
    // Note, we are not directly calling quantile here because efficiencies are possible in the sorting stage due to computing multiple statistics

    if (x == nullptr || minimum == nullptr || lower_hinge == nullptr ||
        median == nullptr || upper_hinge == nullptr || maximum == nullptr)
        return da_status_invalid_pointer;

    // With a little bit of logic we can deal with the choice of axis all in one go
    da_int num_stats, length, stride, spacing, dim1, dim2;
    bool two_d;

    switch (axis) {
    case da_axis_col:
        num_stats = p;
        dim1 = n;
        stride = 1;
        spacing = ldx;
        dim2 = 1;
        two_d = false;
        break;
    case da_axis_row:
        num_stats = n;
        dim1 = p;
        stride = ldx;
        spacing = 1;
        dim2 = 1;
        two_d = false;
        break;
    case da_axis_all:
        dim2 = p;
        num_stats = 1;
        dim1 = n;
        spacing = 1;
        stride = ldx;
        two_d = true;
        break;

    default:
        return da_status_internal_error; // LCOV_EXCL_LINE
        break;
    }

    length = dim1 * dim2;

    // Special case of statistics based on a single element
    if (length == 1) {
        for (da_int i = 0; i < num_stats; i++) {
            median[i] = x[i * spacing];
            maximum[i] = x[i * spacing];
            minimum[i] = x[i * spacing];
            upper_hinge[i] = x[i * spacing];
            lower_hinge[i] = x[i * spacing];
        }
        return da_status_success;
    }

    T h_median = (length + 1) * (T)0.5 - (T)1.0;
    T h_upper = (length + 1) * (T)0.75 - (T)1.0;
    T h_lower = (length + 1) * (T)0.25 - (T)1.0;

    da_int izero = 0;

    da_int h_median_floor = std::clamp((da_int)std::floor(h_median), izero, length - 1);
    da_int h_median_ceil = std::clamp((da_int)std::ceil(h_median), izero, length - 1);
    da_int h_upper_floor = std::clamp((da_int)std::floor(h_upper), izero, length - 1);
    da_int h_upper_ceil = std::clamp((da_int)std::ceil(h_upper), izero, length - 1);
    da_int h_lower_floor = std::clamp((da_int)std::floor(h_lower), izero, length - 1);
    da_int h_lower_ceil = std::clamp((da_int)std::ceil(h_lower), izero, length - 1);
    da_int h_maximum = length - 1;
    da_int h_minimum = 0;

    // Declaring this vector allows us to sort in place, moving elements of xindex, instead of x
    da_int *xindex;
    try {
        xindex = new da_int[length];
    } catch (std::bad_alloc const &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    }

    for (da_int i = 0; i < num_stats; i++) {

        for (da_int j = 0; j < length; j++)
            xindex[j] = j;

        // Compute the median first
        T tmp1, tmp2;
        if (h_median_floor == h_median_ceil) {
            status = indexed_partial_sort(&x[i * spacing], length, stride, xindex,
                                          h_median_floor, dim1, two_d, median[i]);
        } else {
            status = indexed_partial_sort(&x[i * spacing], length, stride, xindex,
                                          h_median_floor, dim1, two_d, tmp1);
            // h_median_ceil = h_median_floor+1 so just find the minimum value of the upper part of the array now
            status =
                indexed_partial_sort(&x[i * spacing], length - h_median_floor - 1, stride,
                                     &xindex[(std::min)(h_median_floor + 1, length - 1)],
                                     izero, dim1, two_d, tmp2);
            median[i] = tmp1 + (h_median - h_median_floor) * (tmp2 - tmp1);
        }

        // We can now use the fact that we've got a partially ordered array to save a lot of work for the hinges and max/min
        if (h_lower_floor == h_lower_ceil) {
            status = indexed_partial_sort(&x[i * spacing], h_median_floor, stride, xindex,
                                          h_lower_ceil, dim1, two_d, lower_hinge[i]);
        } else {
            status = indexed_partial_sort(&x[i * spacing], h_median_floor, stride, xindex,
                                          h_lower_ceil, dim1, two_d, tmp2);
            // h_lower_ceil = h_lower_floor+1 so just find the maximum value of the lower part of the array now
            status = indexed_partial_sort(&x[i * spacing], h_lower_ceil, stride, xindex,
                                          h_lower_floor, dim1, two_d, tmp1);
            lower_hinge[i] = tmp1 + (h_lower - h_lower_floor) * (tmp2 - tmp1);
        }
        status = indexed_partial_sort(&x[i * spacing], h_lower_floor, stride, xindex,
                                      h_minimum, dim1, two_d, minimum[i]);

        if (h_upper_floor == h_upper_ceil) {
            status =
                indexed_partial_sort(&x[i * spacing], length - h_median_ceil - 1, stride,
                                     &xindex[(std::min)(h_median_ceil + 1, length - 1)],
                                     (std::max)(h_upper_floor - h_median_ceil - 1, izero),
                                     dim1, two_d, upper_hinge[i]);
        } else {
            if (h_median_ceil == h_upper_floor) {
                status =
                    indexed_partial_sort(&x[i * spacing], length - h_median_ceil, stride,
                                         &xindex[(std::min)(h_median_ceil, length - 1)],
                                         izero, dim1, two_d, tmp1);
            } else {
                status = indexed_partial_sort(
                    &x[i * spacing], length - h_median_ceil - 1, stride,
                    &xindex[(std::min)(h_median_ceil + 1, length - 1)],
                    (std::max)(h_upper_floor - h_median_ceil - 1, izero), dim1, two_d,
                    tmp1);
            }
            // h_upper_ceil = h_upper_floor+1 so just find the minimum value of the upper part of the array now
            status =
                indexed_partial_sort(&x[i * spacing], length - h_upper_floor - 1, stride,
                                     &xindex[(std::min)(h_upper_floor + 1, length - 1)],
                                     izero, dim1, two_d, tmp2);
            upper_hinge[i] = tmp1 + (h_upper - h_upper_floor) * (tmp2 - tmp1);
        }
        status = indexed_partial_sort(&x[i * spacing], length - h_upper_ceil - 1, stride,
                                      &xindex[(std::min)(h_upper_ceil + 1, length - 1)],
                                      (std::max)(h_maximum - h_upper_ceil - 1, izero),
                                      dim1, two_d, maximum[i]);

        if (status != da_status_success) {
            delete[] xindex;
            return status;
        }
    }

    delete[] xindex;
    return da_status_success;
}

template da_status quantile<float>(da_order order, da_axis axis_in, da_int n_in,
                                   da_int p_in, const float *x, da_int ldx, float q,
                                   float *quant, da_quantile_type quantile_type);
template da_status quantile<double>(da_order order, da_axis axis_in, da_int n_in,
                                    da_int p_in, const double *x, da_int ldx, double q,
                                    double *quant, da_quantile_type quantile_type);
template da_status five_point_summary<float>(da_order order, da_axis axis_in, da_int n_in,
                                             da_int p_in, const float *x, da_int ldx,
                                             float *minimum, float *lower_hinge,
                                             float *median, float *upper_hinge,
                                             float *maximum);
template da_status five_point_summary<double>(da_order order, da_axis axis_in,
                                              da_int n_in, da_int p_in, const double *x,
                                              da_int ldx, double *minimum,
                                              double *lower_hinge, double *median,
                                              double *upper_hinge, double *maximum);

} // namespace da_basic_statistics

} // namespace ARCH