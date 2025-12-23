/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "kernel_functions.hpp"
#include "aoclda.h"
#include "aoclda_types.h"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "da_syrk.hpp"
#include "da_utils.hpp"
#include "macros.h"
#include "miscellaneous.hpp"
#include "pairwise_distances.hpp"
#include <iostream>
#include <vector>

namespace ARCH {

namespace da_kernel_functions {

using namespace da_kernel_functions_types;
using namespace std::string_literals;

// Move the function pointer selection outside the loop

template <typename T>
exp_kernel_func_t<T> select_exp_kernel_function(vectorization_type vec_enum) {
    switch (vec_enum) {
    case scalar:
        return &exp_kernel<T, vectorization_type::scalar>;
    case avx:
        return &exp_kernel<T, vectorization_type::avx>;
    case avx2:
        return &exp_kernel<T, vectorization_type::avx2>;
    case avx512:
#ifdef __AVX512F__
        return &exp_kernel<T, vectorization_type::avx512>;
#else
        return &exp_kernel<T, vectorization_type::avx2>;
#endif
    default:
        return &exp_kernel<T, vectorization_type::scalar>;
    }
}

/*
Auxiliary function to check dimensions of given parameters (taken from pairwise_distances.hpp)
*/
template <typename T>
da_status check_input(da_order order, da_int m, da_int n, da_int k, const T *X,
                      da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd) {
    if (m < 1 || k < 1) {
        return da_status_invalid_array_dimension;
    }
    if (order == column_major && ldx < m)
        return da_status_invalid_leading_dimension;
    if (order == row_major && ldx < k)
        return da_status_invalid_leading_dimension;
    if (X == nullptr || D == nullptr)
        return da_status_invalid_pointer;
    // Check if Y is nullptr to continue with appropriate check of the leading dimension.
    if (Y != nullptr) {
        if (n < 1) {
            return da_status_invalid_array_dimension;
        }
        if (order == column_major) {
            if (ldy < n || ldd < m)
                return da_status_invalid_leading_dimension;
        } else {
            if (ldy < k || ldd < n)
                return da_status_invalid_leading_dimension;
        }
    } else {
        if (ldd < m)
            return da_status_invalid_leading_dimension;
    }
    return da_status_success;
}

/*
Auxiliary function to create work arrays
*/
template <typename T>
da_status create_work_arrays(da_int m, da_int &n, const T *Y, std::vector<T> &x_work,
                             std::vector<T> &y_work, bool &X_is_Y) {
    try {
        x_work.resize(m);
    } catch (std::bad_alloc const &) { // LCOV_EXCL_LINE
        return da_status_memory_error; // LCOV_EXCL_LINE
    }
    if (Y) {
        try {
            y_work.resize(n);
        } catch (std::bad_alloc const &) { // LCOV_EXCL_LINE
            return da_status_memory_error; // LCOV_EXCL_LINE
        }
    } else {
        // Y is null pointer, set X_is_Y to true
        X_is_Y = true;
        n = m;
    }
    return da_status_success;
}

/*
RBF kernel
Given an m by k matrix X and an n by k matrix Y (both column major), computes the m by n kernel matrix D
*/
template <typename T>
da_status rbf_kernel(da_order order, da_int m, da_int n, da_int k, const T *X, da_int ldx,
                     const T *Y, da_int ldy, T *D, da_int ldd, T gamma) {
    da_status status;
    status = check_input(order, m, n, k, X, ldx, Y, ldy, D, ldd);
    if (status != da_status_success)
        return status;
    if (gamma < 0)
        return da_status_invalid_input;
    // Initialize X_is_Y.
    bool X_is_Y = false;
    // Allocate memory for norm arrays.
    std::vector<T> x_work, y_work;
    status = create_work_arrays(m, n, Y, x_work, y_work, X_is_Y);
    if (status != da_status_success)
        return status;
    // Get vectorisation
    da_int first_dim = (order == column_major) ? m : n;
    vectorization_type vectorisation;
    select_simd_size<T>(first_dim, vectorisation);
    // Add telemetry
    context_set_hidden_settings("kf.setup"s,
                                "kernel.type="s + std::to_string(vectorisation));
    rbf_kernel_internal(order, m, n, k, X, x_work.data(), 2, ldx, Y, y_work.data(), 2,
                        ldy, D, ldd, gamma, X_is_Y, (da_int)vectorisation);
    return status;
}

/*
Linear kernel
*/
template <typename T>
da_status linear_kernel(da_order order, da_int m, da_int n, da_int k, const T *X,
                        da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd) {
    da_status status;
    status = check_input(order, m, n, k, X, ldx, Y, ldy, D, ldd);
    if (status != da_status_success)
        return status;
    // Initialize X_is_Y.
    bool X_is_Y = false;
    if (Y == nullptr) {
        X_is_Y = true;
        n = m;
    }
    linear_kernel_internal(order, m, n, k, X, ldx, Y, ldy, D, ldd, X_is_Y);
    return status;
}

/*
Polynomial kernel
*/
template <typename T>
da_status polynomial_kernel(da_order order, da_int m, da_int n, da_int k, const T *X,
                            da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd, T gamma,
                            da_int degree, T coef0) {
    da_status status;
    status = check_input(order, m, n, k, X, ldx, Y, ldy, D, ldd);
    if (status != da_status_success)
        return status;
    if (gamma < 0 || degree < 1)
        return da_status_invalid_input;
    // Initialize X_is_Y.
    bool X_is_Y = false;
    if (Y == nullptr) {
        X_is_Y = true;
        n = m;
    }
    // Get vectorisation
    da_int first_dim = (order == column_major) ? m : n;
    vectorization_type vectorisation;
    select_simd_size<T>(first_dim, vectorisation);
    // Add telemetry
    context_set_hidden_settings("kf.setup"s,
                                "kernel.type="s + std::to_string(vectorisation));
    polynomial_kernel_internal(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, degree,
                               coef0, X_is_Y, (da_int)vectorisation);
    return status;
}

/*
Sigmoid kernel
*/
template <typename T>
da_status sigmoid_kernel(da_order order, da_int m, da_int n, da_int k, const T *X,
                         da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd, T gamma,
                         T coef0) {
    da_status status;
    status = check_input(order, m, n, k, X, ldx, Y, ldy, D, ldd);
    if (status != da_status_success)
        return status;
    if (gamma < 0)
        return da_status_invalid_input;
    // Initialize X_is_Y.
    bool X_is_Y = false;
    if (Y == nullptr) {
        X_is_Y = true;
        n = m;
    }
    // Get vectorisation
    da_int first_dim = (order == column_major) ? m : n;
    vectorization_type vectorisation;
    select_simd_size<T>(first_dim, vectorisation);
    // Add telemetry
    context_set_hidden_settings("kf.setup"s,
                                "kernel.type="s + std::to_string(vectorisation));
    sigmoid_kernel_internal(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, coef0, X_is_Y,
                            (da_int)vectorisation);
    return status;
}

/*
Internal functions that avoid input checking
Also for RBF we can avoid repeatable creation of work arrays
*/

/*
Helper function to compute gemm/syrk and transpose
*/
template <typename T>
inline void fill_upper_triangular(da_order order, da_int m, T *D, da_int ldd) {
    if (order == column_major) {
        for (da_int i = 0; i < m; i++)
            for (da_int j = i + 1; j < m; j++)
                D[j + i * ldd] = D[i + j * ldd];
    } else {
        for (da_int i = 0; i < m; i++)
            for (da_int j = i + 1; j < m; j++)
                D[j * ldd + i] = D[i * ldd + j];
    }
};

template <typename T>
inline void kernel_setup(da_order order, da_int m, da_int n, da_int k, const T *X,
                         da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd, T gamma,
                         bool X_is_Y) {
    CBLAS_ORDER cblas_order = da_utils::da_order_to_cblas_order(order);
    // Compute X*Y^t
    if (X_is_Y) {
        da_syrk(order, da_upper, da_no_trans, m, k, gamma, X, ldx, (T)0.0, D, ldd);
        // After syrk D matrix is upper triangular, this loop is to make D symmetric
        fill_upper_triangular(order, m, D, ldd);
    } else {
        da_blas::cblas_gemm(cblas_order, CblasNoTrans, CblasTrans, m, n, k, gamma, X, ldx,
                            Y, ldy, 0.0, D, ldd);
    }
};

/*
RBF kernel
Given an m by k matrix X and an n by k matrix Y, computes the m by n kernel matrix D
*/
template <typename T>
void rbf_kernel_internal(da_order order, da_int m, da_int n, da_int k, const T *X,
                         T *X_norms, da_int compute_X_norms, da_int ldx, const T *Y,
                         T *Y_norms, da_int compute_Y_norms, da_int ldy, T *D, da_int ldd,
                         T gamma, bool X_is_Y, da_int vectorisation) {
    // For row-major order, parallelise over rows of X, for column-major order parallelise over rows of Y
    // Consequently, for row-major we fill D matrix row by row, for column-major we fill D matrix column by column.
    // first dim is the first dimension of the array we will be parallelising over.
    da_int first_dim = (order == column_major) ? n : m;
    da_int first_dim_D = (order == column_major) ? m : n;
    da_int n_blocks, block_rem;
    da_int block_size = std::min(first_dim, EUCLID_MAX_BLOCK_SIZE);
    da_utils::blocking_scheme(first_dim, block_size, n_blocks, block_rem);
    [[maybe_unused]] da_int n_threads = da_utils::get_n_threads_loop(n_blocks);
    vectorization_type vec_enum = (vectorization_type)vectorisation;
    auto exp_kernel_func = select_exp_kernel_function<T>(vec_enum);
    T multiplier = -gamma;
    if (n_threads > 1) {
// Compute |x_i-y_j|^2 in a parallel way
#pragma omp parallel for schedule(dynamic) num_threads(n_threads) default(none)          \
    shared(first_dim, block_size, n_blocks, block_rem, X, ldx, Y, ldy, D, ldd, X_norms,  \
               compute_X_norms, Y_norms, compute_Y_norms, m, n, k, order, X_is_Y,        \
               multiplier, exp_kernel_func, first_dim_D)
        for (da_int i = 0; i < n_blocks; i++) {
            da_int current_block_size =
                (i == n_blocks - 1 && block_rem > 0) ? block_rem : block_size;
            da_int Y_block_offset, X_block_offset, D_block_offset, Y_norm_offset,
                X_norm_offset, m_in, n_in;
            if (order == column_major) {
                Y_block_offset = i * block_size;
                X_block_offset = 0;
                D_block_offset = i * block_size * ldd;
                Y_norm_offset = i * block_size;
                X_norm_offset = 0;
                m_in = m;
                n_in = current_block_size;
            } else {
                Y_block_offset = 0;
                X_block_offset = i * block_size * ldx;
                D_block_offset = i * block_size;
                Y_norm_offset = 0;
                X_norm_offset = i * block_size;
                m_in = current_block_size;
                n_in = n;
            }
            ARCH::euclidean_gemm_distance(
                order, m_in, n_in, k, X + X_block_offset, ldx, Y + Y_block_offset, ldy,
                D + D_block_offset, ldd, X_norms + X_norm_offset, compute_X_norms,
                Y_norms + Y_norm_offset, compute_Y_norms, true, X_is_Y);
            exp_kernel_func(first_dim_D, current_block_size, D + D_block_offset, ldd,
                            multiplier);
        }
    } else {
        ARCH::euclidean_gemm_distance(order, m, n, k, X, ldx, Y, ldy, D, ldd, X_norms,
                                      compute_X_norms, Y_norms, compute_Y_norms, true,
                                      X_is_Y);
        exp_kernel_func(first_dim_D, first_dim, D, ldd, multiplier);
    }
    // If X==Y then result of euclidean_gemm_distance() is upper triangular matrix D.
    // This loop is to make D symmetric matrix.
    if (X_is_Y) {
        fill_upper_triangular(order, m, D, ldd);
    }
}

/*
Linear kernel
*/
template <typename T>
void linear_kernel_internal(da_order order, da_int m, da_int n, da_int k, const T *X,
                            da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd,
                            bool X_is_Y) {
    kernel_setup(order, m, n, k, X, ldx, Y, ldy, D, ldd, (T)1.0, X_is_Y);
}

/*
Polynomial kernel
*/
template <typename T>
void polynomial_kernel_internal(da_order order, da_int m, da_int n, da_int k, const T *X,
                                da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd,
                                T gamma, da_int degree, T coef0, bool X_is_Y,
                                [[maybe_unused]] da_int vectorisation) {
    kernel_setup(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, X_is_Y);
    // Raise to the power and add constant
    da_int first_dim = (order == column_major) ? m : n;
    da_int second_dim = (order == column_major) ? n : m;
    // We parallelise over blocks of second_dim to keep contiguity
    da_int n_blocks, block_rem;
    da_int block_size = std::min(second_dim, MATH_FUNC_MAX_BLOCK_SIZE);
    da_utils::blocking_scheme(second_dim, block_size, n_blocks, block_rem);
    [[maybe_unused]] da_int n_threads = da_utils::get_n_threads_loop(n_blocks);
    da_int block_size_local = block_size;
#pragma omp parallel firstprivate(block_size_local)                                      \
    shared(first_dim, second_dim, D, ldd, degree, coef0, n_blocks, block_rem,            \
               block_size) default(none) num_threads(n_threads)
    {
#pragma omp for schedule(dynamic)
        for (da_int i = 0; i < n_blocks; i++) {
            da_int block_offset;
            da_int current_block_size = block_size_local;
            if (i == n_blocks - 1 && block_rem > 0) {
                block_offset = (second_dim - block_rem) * first_dim;
                current_block_size = block_rem;
            } else {
                block_offset = i * (block_size * first_dim);
            }
            pow_kernel<T, scalar>(first_dim, current_block_size, D + block_offset, ldd,
                                  coef0, degree);
        }
    }
}

/*
Sigmoid kernel
*/
template <typename T>
void sigmoid_kernel_internal(da_order order, da_int m, da_int n, da_int k, const T *X,
                             da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd,
                             T gamma, T coef0, bool X_is_Y,
                             [[maybe_unused]] da_int vectorisation) {
    kernel_setup(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, X_is_Y);
    // Raise to the power and add constant
    da_int first_dim = (order == column_major) ? m : n;
    da_int second_dim = (order == column_major) ? n : m;
    // We parallelise over blocks of second_dim to keep contiguity
    da_int n_blocks, block_rem;
    da_int block_size = std::min(second_dim, MATH_FUNC_MAX_BLOCK_SIZE);
    da_utils::blocking_scheme(second_dim, block_size, n_blocks, block_rem);
    [[maybe_unused]] da_int n_threads = da_utils::get_n_threads_loop(n_blocks);
    da_int block_size_local = block_size;
#pragma omp parallel firstprivate(block_size_local)                                      \
    shared(first_dim, second_dim, D, ldd, coef0, n_blocks, block_rem,                    \
               block_size) default(none) num_threads(n_threads)
    {
#pragma omp for schedule(dynamic)
        for (da_int i = 0; i < n_blocks; i++) {
            da_int block_offset;
            da_int current_block_size = block_size_local;
            if (i == n_blocks - 1 && block_rem > 0) {
                block_offset = (second_dim - block_rem) * first_dim;
                current_block_size = block_rem;
            } else {
                block_offset = i * (block_size * first_dim);
            }
            tanh_kernel<T, scalar>(first_dim, current_block_size, D + block_offset, ldd,
                                   coef0);
        }
    }
}

template da_status check_input<float>(da_order order, da_int m, da_int n, da_int k,
                                      const float *X, da_int ldx, const float *Y,
                                      da_int ldy, float *D, da_int ldd);
template da_status check_input<double>(da_order order, da_int m, da_int n, da_int k,
                                       const double *X, da_int ldx, const double *Y,
                                       da_int ldy, double *D, da_int ldd);

template da_status create_work_arrays<float>(da_int m, da_int &n, const float *Y,
                                             std::vector<float> &x_work,
                                             std::vector<float> &y_work, bool &X_is_Y);
template da_status create_work_arrays<double>(da_int m, da_int &n, const double *Y,
                                              std::vector<double> &x_work,
                                              std::vector<double> &y_work, bool &X_is_Y);

template da_status rbf_kernel<float>(da_order order, da_int m, da_int n, da_int k,
                                     const float *X, da_int ldx, const float *Y,
                                     da_int ldy, float *D, da_int ldd, float gamma);
template da_status rbf_kernel<double>(da_order order, da_int m, da_int n, da_int k,
                                      const double *X, da_int ldx, const double *Y,
                                      da_int ldy, double *D, da_int ldd, double gamma);

template da_status linear_kernel<float>(da_order order, da_int m, da_int n, da_int k,
                                        const float *X, da_int ldx, const float *Y,
                                        da_int ldy, float *D, da_int ldd);
template da_status linear_kernel<double>(da_order order, da_int m, da_int n, da_int k,
                                         const double *X, da_int ldx, const double *Y,
                                         da_int ldy, double *D, da_int ldd);

template da_status polynomial_kernel<float>(da_order order, da_int m, da_int n, da_int k,
                                            const float *X, da_int ldx, const float *Y,
                                            da_int ldy, float *D, da_int ldd, float gamma,
                                            da_int degree, float coef0);
template da_status polynomial_kernel<double>(da_order order, da_int m, da_int n, da_int k,
                                             const double *X, da_int ldx, const double *Y,
                                             da_int ldy, double *D, da_int ldd,
                                             double gamma, da_int degree, double coef0);

template da_status sigmoid_kernel<float>(da_order order, da_int m, da_int n, da_int k,
                                         const float *X, da_int ldx, const float *Y,
                                         da_int ldy, float *D, da_int ldd, float gamma,
                                         float coef0);

template da_status sigmoid_kernel<double>(da_order order, da_int m, da_int n, da_int k,
                                          const double *X, da_int ldx, const double *Y,
                                          da_int ldy, double *D, da_int ldd, double gamma,
                                          double coef0);

template void rbf_kernel_internal<float>(da_order order, da_int m, da_int n, da_int k,
                                         const float *X, float *X_norms,
                                         da_int compute_X_norms, da_int ldx,
                                         const float *Y, float *Y_norms,
                                         da_int compute_Y_norms, da_int ldy, float *D,
                                         da_int ldd, float gamma, bool X_is_Y,
                                         da_int vectorisation);
template void rbf_kernel_internal<double>(da_order order, da_int m, da_int n, da_int k,
                                          const double *X, double *X_norms,
                                          da_int compute_X_norms, da_int ldx,
                                          const double *Y, double *Y_norms,
                                          da_int compute_Y_norms, da_int ldy, double *D,
                                          da_int ldd, double gamma, bool X_is_Y,
                                          da_int vectorisation);

template void linear_kernel_internal<float>(da_order order, da_int m, da_int n, da_int k,
                                            const float *X, da_int ldx, const float *Y,
                                            da_int ldy, float *D, da_int ldd,
                                            bool X_is_Y);
template void linear_kernel_internal<double>(da_order order, da_int m, da_int n, da_int k,
                                             const double *X, da_int ldx, const double *Y,
                                             da_int ldy, double *D, da_int ldd,
                                             bool X_is_Y);

template void polynomial_kernel_internal<float>(da_order order, da_int m, da_int n,
                                                da_int k, const float *X, da_int ldx,
                                                const float *Y, da_int ldy, float *D,
                                                da_int ldd, float gamma, da_int degree,
                                                float coef0, bool X_is_Y,
                                                da_int vectorisation);
template void polynomial_kernel_internal<double>(da_order order, da_int m, da_int n,
                                                 da_int k, const double *X, da_int ldx,
                                                 const double *Y, da_int ldy, double *D,
                                                 da_int ldd, double gamma, da_int degree,
                                                 double coef0, bool X_is_Y,
                                                 da_int vectorisation);

template void sigmoid_kernel_internal<float>(da_order order, da_int m, da_int n, da_int k,
                                             const float *X, da_int ldx, const float *Y,
                                             da_int ldy, float *D, da_int ldd,
                                             float gamma, float coef0, bool X_is_Y,
                                             da_int vectorisation);
template void sigmoid_kernel_internal<double>(da_order order, da_int m, da_int n,
                                              da_int k, const double *X, da_int ldx,
                                              const double *Y, da_int ldy, double *D,
                                              da_int ldd, double gamma, double coef0,
                                              bool X_is_Y, da_int vectorisation);

template void fill_upper_triangular<float>(da_order order, da_int m, float *D,
                                           da_int ldd);
template void fill_upper_triangular<double>(da_order order, da_int m, double *D,
                                            da_int ldd);

template void kernel_setup<float>(da_order order, da_int m, da_int n, da_int k,
                                  const float *X, da_int ldx, const float *Y, da_int ldy,
                                  float *D, da_int ldd, float gamma, bool X_is_Y);
template void kernel_setup<double>(da_order order, da_int m, da_int n, da_int k,
                                   const double *X, da_int ldx, const double *Y,
                                   da_int ldy, double *D, da_int ldd, double gamma,
                                   bool X_is_Y);

template exp_kernel_func_t<double>
select_exp_kernel_function<double>(vectorization_type vec_enum);
template exp_kernel_func_t<float>
select_exp_kernel_function<float>(vectorization_type vec_enum);

} // namespace da_kernel_functions

} // namespace ARCH