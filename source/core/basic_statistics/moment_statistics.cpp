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
#include "da_omp.hpp"
#include "da_std.hpp"
#include "macros.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

namespace ARCH {

namespace da_basic_statistics {

// Threshold for switching from pairwise to direct summation in recursive_sum.
#define RECURSIVE_SUM_THRESHOLD da_int(2048)

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

template <typename T> inline __attribute__((__always_inline__)) T arithmetic_sum(T x) {
    return x;
}

template <typename T> inline __attribute__((__always_inline__)) T log_sum(T x) {
    return (x == (T)0.0) ? (-1) * std::numeric_limits<T>::infinity() : log(x);
}

template <typename T> inline __attribute__((__always_inline__)) T inverse_sum(T x) {
    return (x == (T)0.0) ? (T)0.0 : (T)1.0 / x;
}

template <typename T>
inline __attribute__((__always_inline__)) T sum_of_squares(T x, T mean) {
    return (x - mean) * (x - mean);
}

template <typename T>
inline __attribute__((__always_inline__)) T sum_of_powers(T x, T mean, da_int k) {
    return power((x - mean), k);
}

// Mute warnings for failed vectorization due to log_sum and sum_of_powers calls
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif

// Computes the sum for a matrix (or a matrix subset) using recursive quasi-pairwise summation.
// inner_dim_size: Number of rows (column-major) or columns (row-major) in the matrix.
template <typename T, auto S>
T recursive_sum(da_int chunk_size, da_int inner_dim_size, const T *data, da_int stride,
                da_int element = 0, std::optional<T> opt_param1 = std::nullopt,
                std::optional<da_int> opt_param2 = std::nullopt) {
    if (chunk_size <= RECURSIVE_SUM_THRESHOLD) {
        T tmp_sum = 0;

        // Get the outer dimension index of the specific element
        da_int outer_dim_n = element / inner_dim_size;

        da_int initial_offset = element + (outer_dim_n * stride);
        data += initial_offset;

        // Number of elements (data points) left to process in current row/column
        da_int n_el_left =
            std::min(chunk_size, inner_dim_size * (outer_dim_n + 1) - element);

        while (chunk_size > 0) {
#pragma omp simd reduction(+ : tmp_sum)
            for (da_int i = 0; i < n_el_left; ++i) {
                if constexpr (std::is_invocable_v<decltype(S), T, T, da_int>) {
                    tmp_sum += S(data[i], opt_param1.value(), opt_param2.value());
                } else if constexpr (std::is_invocable_v<decltype(S), T, T>) {
                    tmp_sum += S(data[i], opt_param1.value());
                } else if constexpr (std::is_invocable_v<decltype(S), T>) {
                    tmp_sum += S(data[i]);
                }
            }

            data += (n_el_left + stride);
            chunk_size -= n_el_left;
            n_el_left = std::min(inner_dim_size, chunk_size);
        }
        return tmp_sum;
    }

    T sum1 = recursive_sum<T, S>(chunk_size / 2, inner_dim_size, data, stride, element,
                                 opt_param1, opt_param2);
    T sum2 =
        recursive_sum<T, S>(chunk_size - chunk_size / 2, inner_dim_size, data, stride,
                            element + (chunk_size / 2), opt_param1, opt_param2);

    return sum1 + sum2;
}

// Computes recursive quasi-pairwise sum for matrix data when the computation axis mismatches the storage order.
// chunk_size must be less than or equal to the outer dimension size.
// inner_dim_size must be less than or equal to the inner dimension size.
template <typename T, auto S>
da_status recursive_sum_vector(da_int chunk_size, da_int inner_dim_size, const T *data,
                               da_int ldx, T *&sum1,
                               std::optional<T *> opt_param1 = std::nullopt,
                               std::optional<da_int> opt_param2 = std::nullopt) {
    if (chunk_size <= RECURSIVE_SUM_THRESHOLD) {
        if (!sum1) {
            try {
                sum1 = new T[inner_dim_size]();
            } catch (std::bad_alloc const &) {
                return da_status_memory_error; // LCOV_EXCL_LINE
            }
        }

        for (da_int i = 0; i < chunk_size; ++i) {
            const T *current_data = &data[ldx * i];

#pragma omp simd
            for (da_int j = 0; j < inner_dim_size; ++j) {
                if constexpr (std::is_invocable_v<decltype(S), T, T, da_int>) {
                    sum1[j] +=
                        S(current_data[j], opt_param1.value()[j], opt_param2.value());
                } else if constexpr (std::is_invocable_v<decltype(S), T, T>) {
                    sum1[j] += S(current_data[j], opt_param1.value()[j]);
                } else if constexpr (std::is_invocable_v<decltype(S), T>) {
                    sum1[j] += S(current_data[j]);
                }
            }
        }
        return da_status_success;
    }

    da_status status = recursive_sum_vector<T, S>(chunk_size / 2, inner_dim_size, data,
                                                  ldx, sum1, opt_param1, opt_param2);
    if (status != da_status_success)
        return status;

    // Null-initialize sum2 to potentially reduce average memory usage
    T *sum2 = nullptr;
    status = recursive_sum_vector<T, S>(chunk_size - chunk_size / 2, inner_dim_size,
                                        data + (chunk_size / 2) * ldx, ldx, sum2,
                                        opt_param1, opt_param2);

    if (status != da_status_success) {
        delete[] sum2; // LCOV_EXCL_LINE
        return status; // LCOV_EXCL_LINE
    }

#pragma omp simd
    for (da_int i = 0; i < inner_dim_size; ++i) {
        sum1[i] += sum2[i];
    }

    delete[] sum2;
    return status;
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

template <typename T, auto S>
da_status full_sum(da_int outer_dim_size, da_int inner_dim_size, const T *x, da_int ldx,
                   T *asum, std::optional<T *> opt_param1 = std::nullopt,
                   std::optional<da_int> opt_param2 = std::nullopt) {
    asum[0] = 0.0;

    // full_sum splits the matrix as if it is a 1D array. Thus, for small matrices
    // (par_threshold > m*n) parallelization is not always beneficial.
    da_int par_threshold = 32768;

    std::optional<T> opt_param1_val = std::nullopt;
    if (opt_param1.has_value()) {
        opt_param1_val = opt_param1.value()[0];
    }

    // Parallelization and summation are done by treating the matrix as a long 1D array,
    // which reduces the number of calls to recursive_sum.
#pragma omp parallel default(none)                                                       \
    shared(outer_dim_size, inner_dim_size, x, ldx, asum, opt_param1_val,                 \
               opt_param2) if (par_threshold < (outer_dim_size * inner_dim_size))
    {
        da_int thread_id = omp_get_thread_num();
        da_int num_threads = omp_get_num_threads();

        da_int chunk_size = (outer_dim_size * inner_dim_size) / num_threads;
        da_int start_index = thread_id * chunk_size;

        // Add the remainder to the last thread
        if (thread_id == (num_threads - 1)) {
            chunk_size += ((outer_dim_size * inner_dim_size) % num_threads);
        }

        T local_sum =
            recursive_sum<T, S>(chunk_size, inner_dim_size, x, ldx - inner_dim_size,
                                start_index, opt_param1_val, opt_param2);

#pragma omp atomic
        asum[0] += local_sum;
    }

    return da_status_success;
}

template <typename T, auto S>
da_status sum_axis_aligned(da_int outer_dim_size, da_int inner_dim_size, const T *x,
                           da_int ldx, T *asum,
                           std::optional<T *> opt_param1 = std::nullopt,
                           std::optional<da_int> opt_param2 = std::nullopt) {

    // Used to extract opt_param1 values when it exists.
    std::optional<T> current_param1 = std::nullopt;

#pragma omp parallel for schedule(static) default(none)                                  \
    shared(outer_dim_size, inner_dim_size, x, ldx, asum, opt_param1, opt_param2)         \
    num_threads(std::min((da_int)omp_get_max_threads(), outer_dim_size))                 \
    firstprivate(current_param1)
    for (da_int i = 0; i < outer_dim_size; ++i) {
        if (opt_param1.has_value()) {
            current_param1 = opt_param1.value()[i];
        }
        asum[i] =
            recursive_sum<T, S>(inner_dim_size, inner_dim_size, x, ldx - inner_dim_size,
                                i * inner_dim_size, current_param1, opt_param2);
    }

    return da_status_success;
}

template <typename T, auto S>
da_status sum_axis_misaligned(da_int outer_dim_size, da_int inner_dim_size, const T *x,
                              da_int ldx, T *asum,
                              std::optional<T *> opt_param1 = std::nullopt,
                              std::optional<da_int> opt_param2 = std::nullopt) {
    da_std::fill(asum, asum + inner_dim_size, 0.0);

    da_int max_threads = inner_dim_size >= 1024 || outer_dim_size >= 1024
                             ? (da_int)omp_get_max_threads()
                             : 1;

    // Use RECURSIVE_SUM_THRESHOLD for block_size to get similar accuracy to aligned case.
    da_int inner_block_size = RECURSIVE_SUM_THRESHOLD;
    da_int n_inner_blocks = (inner_dim_size + inner_block_size - 1) / inner_block_size;

    da_int n_outer_blocks =
        std::min(max_threads, (outer_dim_size + RECURSIVE_SUM_THRESHOLD - 1) /
                                  RECURSIVE_SUM_THRESHOLD);
    da_int outer_block_size = outer_dim_size / n_outer_blocks;

    if ((n_outer_blocks * n_inner_blocks) < max_threads) {
        // Without making the inner_block_size smaller than 32,
        // try to split the inner dimension further until we get at least one block per thread.
        da_int condition1 = (inner_dim_size + 32 - 1) / 32;
        da_int condition2 = (max_threads + n_outer_blocks - 1) / n_outer_blocks;

        n_inner_blocks = std::min(condition1, condition2);
        inner_block_size = inner_dim_size / n_inner_blocks;
    }

    da_int n_threads = std::min(max_threads, (n_inner_blocks * n_outer_blocks));

    da_int threading_error = 0;

    std::vector<T *> local_sums;
    try {
        local_sums.resize(n_outer_blocks, nullptr);
        local_sums[0] = asum;
    } catch (std::bad_alloc const &) {
        return da_status_memory_error; // LCOV_EXCL_LINE
    }

    da_int current_outer_block_size, current_inner_block_size;

#pragma omp parallel shared(                                                             \
        n_outer_blocks, n_inner_blocks, inner_dim_size, outer_dim_size, x, ldx,          \
            threading_error, local_sums, opt_param1,                                     \
            opt_param2) private(current_outer_block_size,                                \
                                    current_inner_block_size) if (n_threads > 1)         \
    num_threads(n_threads)
    {

#pragma omp for schedule(static)
        for (da_int i = 1; i < n_outer_blocks; ++i) {
            try {
                local_sums[i] = new T[inner_dim_size]();
            } catch (std::bad_alloc const &) {
#pragma omp atomic write
                threading_error = 1; // LCOV_EXCL_LINE
            }
        }

        if (threading_error == 0) {
#pragma omp for collapse(2) schedule(static)
            for (da_int i = 0; i < n_outer_blocks; ++i) {
                for (da_int j = 0; j < n_inner_blocks; ++j) {
                    da_int inner_offset = j * inner_block_size;

                    current_outer_block_size = outer_block_size;
                    if (i == n_outer_blocks - 1)
                        current_outer_block_size =
                            outer_dim_size - (i * outer_block_size);

                    current_inner_block_size = inner_block_size;
                    if (j == n_inner_blocks - 1)
                        current_inner_block_size = inner_dim_size - inner_offset;

                    da_int start_index = i * outer_block_size * ldx + inner_offset;
                    T *this_sum = &(local_sums[i][inner_offset]);

                    std::optional<T *> current_opt_param1 = std::nullopt;
                    if (opt_param1.has_value()) {
                        current_opt_param1 = opt_param1.value() + inner_offset;
                    }

                    da_status sum_status = recursive_sum_vector<T, S>(
                        current_outer_block_size, current_inner_block_size,
                        x + start_index, ldx, this_sum, current_opt_param1, opt_param2);

                    if (sum_status != da_status_success) {
#pragma omp atomic write
                        threading_error = 1; // LCOV_EXCL_LINE
                    }
                }
            }
        }

        if (threading_error == 0) {
            // Binary tree reduction of the local sums into asum.
            for (da_int stride = 1; stride < n_outer_blocks; stride <<= 1) {

#pragma omp for schedule(static)
                for (da_int destination_idx = 0; destination_idx < n_outer_blocks;
                     destination_idx += (stride << 1)) {
                    da_int source_idx = destination_idx + stride;

                    if (source_idx < n_outer_blocks) {
                        T *sum_destination = local_sums[destination_idx];
                        T *sum_source = local_sums[source_idx];
#pragma omp simd
                        for (da_int k = 0; k < inner_dim_size; ++k) {
                            sum_destination[k] += sum_source[k];
                        }
                        delete[] sum_source;
                    }
                }
            }
        }
    }

    if (threading_error != 0) {
        for (da_int i = 1; i < n_outer_blocks; ++i) {
            delete[] local_sums[i]; // LCOV_EXCL_LINE
        }
        return da_status_memory_error; // LCOV_EXCL_LINE
    }

    return da_status_success;
}

template <typename T, auto S>
da_status da_sum(da_order order, da_axis axis, da_int m, da_int n, const T *x, da_int ldx,
                 T *asum, std::optional<T *> opt_param1 = std::nullopt,
                 std::optional<da_int> opt_param2 = std::nullopt) {

    da_status status;
    da_int outer_dim_size = order == column_major ? n : m;
    da_int inner_dim_size = order == column_major ? m : n;
    bool axis_aligned = (order == column_major && axis == da_axis_col) ||
                        (order == row_major && axis == da_axis_row);

    if (axis == da_axis_all) {
        status = full_sum<T, S>(outer_dim_size, inner_dim_size, x, ldx, asum, opt_param1,
                                opt_param2);
    } else if (axis_aligned) {
        status = sum_axis_aligned<T, S>(outer_dim_size, inner_dim_size, x, ldx, asum,
                                        opt_param1, opt_param2);
    } else {
        status = sum_axis_misaligned<T, S>(outer_dim_size, inner_dim_size, x, ldx, asum,
                                           opt_param1, opt_param2);
    }

    return status;
}

template <typename T>
da_status check_data(da_order order, da_int m, da_int n, const T *x, da_int ldx,
                     T *dest) {
    if (m < 1 || n < 1) {
        return da_status_invalid_array_dimension;
    }

    if (x == nullptr || dest == nullptr) {
        return da_status_invalid_pointer;
    }

    if (order == column_major) {
        if (ldx < m)
            return da_status_invalid_leading_dimension;
    } else {
        if (ldx < n)
            return da_status_invalid_leading_dimension;
    }
    return da_status_success;
}

/* Arithmetic mean along specified axis */
template <typename T>
da_status mean(da_order order, da_axis axis, da_int m, da_int n, const T *x, da_int ldx,
               T *amean) {

    da_status status = check_data(order, m, n, x, ldx, amean);
    if (status != da_status_success)
        return status;

    status = da_sum<T, arithmetic_sum<T>>(order, axis, m, n, x, ldx, amean);
    if (status != da_status_success)
        return status;

    if (axis == da_axis_all) {
        amean[0] /= (m * n);
    } else {
        da_int axis_size, divisor;
        if (axis == da_axis_col) {
            axis_size = n;
            divisor = m;
        } else {
            axis_size = m;
            divisor = n;
        }

#pragma omp simd
        for (da_int i = 0; i < axis_size; ++i) {
            amean[i] /= divisor;
        }
    }

    return da_status_success;
}

template <typename T>
da_status is_positive(da_order order, da_int m, da_int n, const T *x, da_int ldx) {
    da_int outer_dim = order == column_major ? n : m;
    da_int inner_dim = order == column_major ? m : n;
    T zero = (T)0.0;
    for (da_int i = 0; i < outer_dim; ++i) {
        for (da_int j = 0; j < inner_dim; ++j) {
            if (x[j + ldx * i] < zero)
                return da_status_negative_data;
        }
    }
    return da_status_success;
}

/* Geometric mean computed using log and exp to avoid overflow. Care needed to deal with negative or zero entries */
template <typename T>
da_status geometric_mean(da_order order, da_axis axis, da_int m, da_int n, const T *x,
                         da_int ldx, T *gmean) {

    da_status status = check_data(order, m, n, x, ldx, gmean);
    if (status != da_status_success)
        return status;

    status = is_positive(order, m, n, x, ldx);
    if (status != da_status_success)
        return status;

    status = da_sum<T, log_sum<T>>(order, axis, m, n, x, ldx, gmean);
    if (status != da_status_success)
        return status;

    if (axis == da_axis_all) {
        gmean[0] = exp(gmean[0] / (m * n));
    } else {
        da_int axis_size, divisor;
        if (axis == da_axis_col) {
            axis_size = n;
            divisor = m;
        } else {
            axis_size = m;
            divisor = n;
        }

        for (da_int i = 0; i < axis_size; ++i) {
            gmean[i] = exp(gmean[i] / divisor);
        }
    }
    return da_status_success;
}

/* Harmonic mean along a specified axis */
template <typename T>
da_status harmonic_mean(da_order order, da_axis axis, da_int m, da_int n, const T *x,
                        da_int ldx, T *hmean) {

    da_status status = check_data(order, m, n, x, ldx, hmean);
    if (status != da_status_success)
        return status;

    status = da_sum<T, inverse_sum<T>>(order, axis, m, n, x, ldx, hmean);
    if (status != da_status_success)
        return status;

    T zero = (T)0.0;

    if (axis == da_axis_all) {
        hmean[0] = (hmean[0] == zero) ? zero : (m * n) / hmean[0];
    } else {
        da_int axis_size, divisor;
        if (axis == da_axis_col) {
            axis_size = n;
            divisor = m;
        } else {
            axis_size = m;
            divisor = n;
        }

#pragma omp simd
        for (da_int i = 0; i < axis_size; ++i) {
            hmean[i] = (hmean[i] == zero) ? zero : divisor / hmean[i];
        }
    }

    return da_status_success;
}

/* Mean and variance along specified axis */
template <typename T>
da_status variance(da_order order, da_axis axis, da_int m, da_int n, const T *x,
                   da_int ldx, da_int dof, T *amean, T *var) {

    da_status status = check_data(order, m, n, x, ldx, amean);
    if (status != da_status_success)
        return status;

    if (var == nullptr)
        return da_status_invalid_pointer;

    status = mean(order, axis, m, n, x, ldx, amean);
    if (status != da_status_success)
        return status;

    status = da_sum<T, sum_of_squares<T>>(order, axis, m, n, x, ldx, var, amean);
    if (status != da_status_success)
        return status;

    da_int scale_factor = dof;
    da_int axis_size;

    if (axis == da_axis_all) {
        if (dof < 0) {
            scale_factor = m * n;
        } else if (dof == 0) {
            scale_factor = m * n - 1;
        }

        if (scale_factor > 1)
            var[0] /= scale_factor;
    } else {

        axis_size = axis == da_axis_col ? n : m;
        if (dof < 0) {
            scale_factor = axis == da_axis_col ? m : n;
        } else if (dof == 0) {
            scale_factor = axis == da_axis_col ? m - 1 : n - 1;
        }

        if (scale_factor > 1) {
#pragma omp simd
            for (da_int i = 0; i < axis_size; ++i) {
                var[i] /= scale_factor;
            }
        }
    }
    return da_status_success;
}

/* Mean, variance and skewness along specified axis */
template <typename T>
da_status skewness(da_order order, da_axis axis, da_int m, da_int n, const T *x,
                   da_int ldx, T *amean, T *var, T *skew) {

    da_status status = check_data(order, m, n, x, ldx, amean);
    if (status != da_status_success)
        return status;

    if (var == nullptr || skew == nullptr)
        return da_status_invalid_pointer;

    status = mean(order, axis, m, n, x, ldx, amean);
    if (status != da_status_success)
        return status;

    status = da_sum<T, sum_of_squares<T>>(order, axis, m, n, x, ldx, var, amean);
    if (status != da_status_success)
        return status;
    status = da_sum<T, sum_of_powers<T>>(order, axis, m, n, x, ldx, skew, amean, 3);
    if (status != da_status_success)
        return status;

    T zero = (T)0.0;

    if (axis == da_axis_all) {
        T sqrt_divisor = std::sqrt((T)(m * n));
        skew[0] = (var[0] == zero) ? zero : skew[0] * sqrt_divisor / pow(var[0], (T)1.5);
        var[0] /= (m * n);
    } else {
        da_int axis_size = axis == da_axis_col ? n : m;
        da_int divisor = axis == da_axis_col ? m : n;
        T sqrt_divisor = std::sqrt((T)divisor);

        for (da_int i = 0; i < axis_size; ++i) {
            skew[i] =
                (var[i] == zero) ? zero : skew[i] * sqrt_divisor / pow(var[i], (T)1.5);
            var[i] /= divisor;
        }
    }

    return da_status_success;
}

/* Mean, variance and kurtosis along specified axis */
template <typename T>
da_status kurtosis(da_order order, da_axis axis, da_int m, da_int n, const T *x,
                   da_int ldx, T *amean, T *var, T *kurt) {

    da_status status = check_data(order, m, n, x, ldx, amean);
    if (status != da_status_success)
        return status;

    if (var == nullptr || kurt == nullptr)
        return da_status_invalid_pointer;

    status = mean(order, axis, m, n, x, ldx, amean);
    if (status != da_status_success)
        return status;

    status = da_sum<T, sum_of_squares<T>>(order, axis, m, n, x, ldx, var, amean);
    if (status != da_status_success)
        return status;
    status = da_sum<T, sum_of_powers<T>>(order, axis, m, n, x, ldx, kurt, amean, 4);
    if (status != da_status_success)
        return status;

    T zero = (T)0.0;
    T three = (T)3.0;

    if (axis == da_axis_all) {
        kurt[0] =
            (var[0] == zero) ? -three : (kurt[0] * (m * n) / (var[0] * var[0])) - three;
        var[0] /= (m * n);
    } else {
        da_int axis_size = axis == da_axis_col ? n : m;
        da_int divisor = axis == da_axis_col ? m : n;

#pragma omp simd
        for (da_int i = 0; i < axis_size; ++i) {
            kurt[i] = (var[i] == zero) ? -three
                                       : (kurt[i] * divisor / (var[i] * var[i])) - three;
            var[i] /= divisor;
        }
    }

    return da_status_success;
}

/* kth moment along specified axis. Optionally use precomputed mean. */
template <typename T>
da_status moment(da_order order, da_axis axis, da_int m, da_int n, const T *x, da_int ldx,
                 da_int k, da_int use_precomputed_mean, T *amean, T *mom) {

    da_status status = check_data(order, m, n, x, ldx, amean);
    if (status != da_status_success)
        return status;

    if (k < 0)
        return da_status_invalid_input;
    if (mom == nullptr)
        return da_status_invalid_pointer;

    if (!use_precomputed_mean) {
        status = mean(order, axis, m, n, x, ldx, amean);
        if (status != da_status_success)
            return status;
    }

    if (k == 2) {
        status = da_sum<T, sum_of_squares<T>>(order, axis, m, n, x, ldx, mom, amean);
    } else {
        status = da_sum<T, sum_of_powers<T>>(order, axis, m, n, x, ldx, mom, amean, k);
    }

    if (status != da_status_success)
        return status;

    if (axis == da_axis_all) {
        mom[0] /= (m * n);
    } else {
        da_int axis_size = axis == da_axis_col ? n : m;
        da_int divisor = axis == da_axis_col ? m : n;

#pragma omp simd
        for (da_int i = 0; i < axis_size; ++i) {
            mom[i] /= divisor;
        }
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