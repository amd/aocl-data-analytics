/*
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "da_std.hpp"
#include "macros.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace ARCH {

namespace da_basic_statistics {

/*  Given two adjacent or equal indices (idx1 and idx2) within [0, length), partially sort x
    to retrieve the values at those positions. When idx1 != idx2, only one nth_element call is
    performed and the adjacent value is obtained via min/max on the
    already-partitioned portion. */
template <typename T>
da_status quick_selection(T *x, da_int length, da_int idx1, da_int idx2, T &val1,
                          T &val2) {
    if (idx1 > idx2)
        std::swap(idx1, idx2);

    if (idx2 - idx1 != 0 && idx2 - idx1 != 1)
        return da_status_internal_error;

    if (idx1 == idx2) {
        std::nth_element(x, x + idx1, x + length);
        val1 = x[idx1];
        val2 = x[idx1];
    } else {
        if (idx1 < length - 1 - idx2) {
            // Small quantile: find idx2 first, then max of left portion
            std::nth_element(x, x + idx2, x + length);
            val2 = x[idx2];
            val1 = *std::max_element(x, x + idx2);
        } else {
            // Large quantile: find idx1 first, then min of small right portion
            std::nth_element(x, x + idx1, x + length);
            val1 = x[idx1];
            val2 = *std::min_element(x + idx2, x + length);
        }
    }

    return da_status_success;
}

template <typename T>
da_status validate_quantile_parameters(da_order order, da_int n, da_int p, const T *x,
                                       da_int ldx, const T *q, da_int n_q, T *quantiles) {
    if (x == nullptr || q == nullptr || quantiles == nullptr)
        return da_status_invalid_pointer;

    if (n < 1 || p < 1)
        return da_status_invalid_array_dimension;

    if (n_q < 1)
        return da_status_invalid_input;

    if (order == column_major) {
        if (n > ldx)
            return da_status_invalid_leading_dimension;
    } else if (order == row_major) {
        if (p > ldx)
            return da_status_invalid_leading_dimension;
    }

    return da_status_success;
}

// Snap to nearest integer to correct float rounding noise
template <typename T> inline double denoise_cast(double val) {
    if constexpr (std::is_same_v<T, float>) {
        double r = std::round(val);
        double tol = std::min(0.25, std::max(1.0, std::abs(val)) *
                                        std::numeric_limits<float>::epsilon() * 2.0);
        if (std::abs(val - r) <= tol)
            val = r;
    }
    return val;
}

template <typename T>
inline da_status get_quantile_indices(const T *q, da_int n_q, da_int length,
                                      da_quantile_type quantile_type,
                                      std::vector<double> &h) {
    for (da_int i = 0; i < n_q; ++i) {
        if (q[i] < 0 || q[i] > 1)
            return da_status_invalid_input;

        double qi = static_cast<double>(q[i]);

        // We could combine some of these, but this is perhaps clearer
        switch (quantile_type) {
        case da_quantile_type_1:
            h[i] = denoise_cast<T>(length * qi);
            break;
        case da_quantile_type_2:
            h[i] = denoise_cast<T>(length * qi) + 0.5;
            break;
        case da_quantile_type_3:
            h[i] = denoise_cast<T>(length * qi) - 0.5;
            break;
        case da_quantile_type_4:
            h[i] = denoise_cast<T>(length * qi);
            break;
        case da_quantile_type_5:
            h[i] = denoise_cast<T>(length * qi) + 0.5;
            break;
        case da_quantile_type_6:
            h[i] = denoise_cast<T>((length + 1) * qi);
            break;
        case da_quantile_type_7:
            h[i] = denoise_cast<T>((length - 1) * qi) + 1;
            break;
        case da_quantile_type_8: {
            double third = 1.0 / 3.0;
            h[i] = denoise_cast<T>((length + third) * qi) + third;
            break;
        }
        case da_quantile_type_9:
            h[i] = denoise_cast<T>((length + 0.25) * qi) + 3.0 / 8.0;
            break;

        default:
            return da_status_internal_error; // LCOV_EXCL_LINE
            break;
        }
        h[i] -= 1.0;
    }
    return da_status_success;
}

/* Compute the qth quantile of x along the specified axis */
template <typename T>
da_status quantile(da_order order, da_axis axis, da_int n, da_int p, const T *x,
                   da_int ldx, const T *q, da_int n_q, T *quantiles,
                   da_quantile_type quantile_type) {

    // ** q range is validated inside get_quantile_indices() and not here
    da_status status =
        validate_quantile_parameters(order, n, p, x, ldx, q, n_q, quantiles);
    if (status != da_status_success)
        return status;

    // Set number of stats per quantile and length of working axis.
    da_int num_stats, length;

    // In case of mixed axis and order request, matrix will be transposed later to match axis.
    switch (axis) {
    case da_axis_col:
        num_stats = p;
        length = n;
        break;
    case da_axis_row:
        num_stats = n;
        length = p;
        break;
    case da_axis_all:
        num_stats = 1;
        length = n * p;
        break;
    default:
        return da_status_internal_error; // LCOV_EXCL_LINE
        break;
    }

    // Fractional indices. double is used to avoid non-adjacent h1 and h2 indices
    // for really big arrays.
    std::vector<double> h;
    try {
        h.resize(n_q);
    } catch (std::bad_alloc const &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    }

    status = get_quantile_indices(q, n_q, length, quantile_type, h);
    if (status != da_status_success)
        return status;

    // Sort quantile indices for faster selection while preserving output order.
    std::vector<da_int> sorted_h_idx(n_q);
    da_std::iota(sorted_h_idx.begin(), sorted_h_idx.end(), 0);
    std::sort(sorted_h_idx.begin(), sorted_h_idx.end(),
              [&h](da_int a, da_int b) { return h[a] < h[b]; });

    // Transpose x when quantiles axis doesn't match input order for faster computation
    bool transpose_x = ((order == column_major) && (axis == da_axis_row)) ||
                       ((order == row_major) && (axis == da_axis_col));

    // Early exit in case length == 1, where copying of X can be skipped
    if (length == 1) {
        for (da_int i = 0; i < num_stats; ++i) {
            da_int stride = transpose_x ? 1 : ldx;
            for (da_int j = 0; j < n_q; ++j) {
                da_int idx = order == row_major ? j * num_stats + i : i * n_q + j;
                quantiles[idx] = x[i * stride];
            }
        }
        return da_status_success;
    }

    // Create a copy of x (either partial (for axes col and row) or full (for axis all))
    // to work on in the selection step
    std::vector<T> copy_x;
    try {
        da_int copy_size = axis == da_axis_all ? (n * p) : length;
        copy_x.resize(copy_size);
    } catch (std::bad_alloc const &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    }

    da_int ldx_internal = ldx;

    if (axis == da_axis_all) {
        da_int dim1 = order == column_major ? n : p;
        da_int dim2 = order == column_major ? p : n;
        for (da_int i = 0; i < dim2; ++i) {
            std::copy(x + i * ldx, x + i * ldx + dim1, copy_x.begin() + i * dim1);
        }
        // Since copying only usable data, equal ldx to the proper dimension.
        ldx_internal = order == column_major ? n : p;
    }

    // Initial pointer to copy_x
    T *work = copy_x.data();
    da_int izero = 0;

    for (da_int i = 0; i < num_stats; ++i) {
        da_int last_ceil = 0;
        da_int last_floor = 0;
        da_int h1, h2;
        T tmp1 = 0;
        T tmp2 = 0;

        if (axis == da_axis_all) {
            work = copy_x.data() + i * ldx_internal;
        } else {
            if (!transpose_x) {
                std::copy(x + i * ldx_internal, x + i * ldx_internal + length, work);
            } else {
                for (da_int j = 0; j < length; ++j) {
                    work[j] = x[i + j * ldx_internal];
                }
            }
        }

        for (da_int j = 0; j < n_q; ++j) {

            da_int orig_h_idx = sorted_h_idx[j];

            switch (quantile_type) {
            case da_quantile_type_1: {
                h1 = std::clamp((da_int)std::ceil(h[orig_h_idx]), izero, length - 1);
                h2 = h1;
                break;
            }
            case da_quantile_type_2: {
                h1 =
                    std::clamp((da_int)std::ceil(h[orig_h_idx] - 0.5), izero, length - 1);
                h2 = std::clamp((da_int)std::floor(h[orig_h_idx] + 0.5), izero,
                                length - 1);
                break;
            }
            case da_quantile_type_3: {
                h1 = std::clamp((da_int)std::nearbyint(h[orig_h_idx]), izero, length - 1);
                h2 = h1;
                break;
            }
            default: {
                h1 = std::clamp((da_int)std::floor(h[orig_h_idx]), izero, length - 1);
                h2 = std::clamp((da_int)std::ceil(h[orig_h_idx]), izero, length - 1);
                break;
            }
            }

            // j == 0 make sure it runs on the first iterration when h1 and h2 are both 0.
            if (j == 0 || h1 != last_floor || h2 != last_ceil) {
                status = quick_selection(work + last_ceil, length - last_ceil,
                                         h1 - last_ceil, h2 - last_ceil, tmp1, tmp2);
                if (status != da_status_success)
                    return status;
            }

            da_int idx =
                order == row_major ? orig_h_idx * num_stats + i : i * n_q + orig_h_idx;
            if (h1 == h2) {
                quantiles[idx] = tmp1;
            } else if (quantile_type == da_quantile_type_2) {
                quantiles[idx] = (T)0.5 * (tmp1 + tmp2);
            } else if (quantile_type != da_quantile_type_1 &&
                       quantile_type != da_quantile_type_3) {
                quantiles[idx] = tmp1 + (h[orig_h_idx] - h1) * (tmp2 - tmp1);
            }

            last_floor = h1;
            last_ceil = h2;
        }
    }

    return da_status_success;
}

/* Compute min/max, hinges and median along specified axis */
template <typename T>
da_status five_point_summary(da_order order, da_axis axis, da_int n, da_int p, const T *x,
                             da_int ldx, T *minimum, T *lower_hinge, T *median,
                             T *upper_hinge, T *maximum) {

    if (x == nullptr || minimum == nullptr || lower_hinge == nullptr ||
        median == nullptr || upper_hinge == nullptr || maximum == nullptr)
        return da_status_invalid_pointer;

    // validate n and p here to make sure non-negative values are
    // passsed to the memory allocation later
    if (n < 1 || p < 1)
        return da_status_invalid_array_dimension;

    // Run quantiles() for the five points needed.
    std::vector<T> q = {T(0.0), T(0.25), T(0.5), T(0.75), T(1.0)};
    da_int n_q = 5;

    // Compute size of quantiles
    size_t quantiles_size = 0;
    da_int quantiles_axis_size = 0;
    if (axis == da_axis_col) {
        quantiles_size = p * n_q;
        quantiles_axis_size = p;
    } else if (axis == da_axis_row) {
        quantiles_size = n * n_q;
        quantiles_axis_size = n;
    } else {
        quantiles_size = n_q;
        quantiles_axis_size = 1;
    }

    std::vector<T> quantiles;
    try {
        quantiles.resize(quantiles_size);
    } catch (std::bad_alloc const &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    }

    da_status status = quantile(order, axis, n, p, x, ldx, q.data(), n_q,
                                quantiles.data(), da_quantile_type_6);
    if (status != da_status_success)
        return status;

    if (order == row_major) {
        std::copy(quantiles.begin(), quantiles.begin() + quantiles_axis_size, minimum);
        std::copy(quantiles.begin() + quantiles_axis_size,
                  quantiles.begin() + (quantiles_axis_size * 2), lower_hinge);
        std::copy(quantiles.begin() + (quantiles_axis_size * 2),
                  quantiles.begin() + (quantiles_axis_size * 3), median);
        std::copy(quantiles.begin() + (quantiles_axis_size * 3),
                  quantiles.begin() + (quantiles_axis_size * 4), upper_hinge);
        std::copy(quantiles.begin() + (quantiles_axis_size * 4),
                  quantiles.begin() + (quantiles_axis_size * 5), maximum);
    } else {
        for (da_int i = 0; i < quantiles_axis_size; ++i) {
            minimum[i] = quantiles[i * n_q];
            lower_hinge[i] = quantiles[i * n_q + 1];
            median[i] = quantiles[i * n_q + 2];
            upper_hinge[i] = quantiles[i * n_q + 3];
            maximum[i] = quantiles[i * n_q + 4];
        }
    }

    return da_status_success;
}

template da_status quantile<float>(da_order order, da_axis axis, da_int n, da_int p,
                                   const float *x, da_int ldx, const float *q, da_int n_q,
                                   float *quantiles, da_quantile_type quantile_type);
template da_status quantile<double>(da_order order, da_axis axis, da_int n, da_int p,
                                    const double *x, da_int ldx, const double *q,
                                    da_int n_q, double *quantiles,
                                    da_quantile_type quantile_type);
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