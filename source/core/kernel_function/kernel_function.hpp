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

#pragma once

#include "aoclda.h"
#include "aoclda_types.h"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "euclidean_distance.hpp"
#include "kernel_function_local.hpp"
#include <iostream>
#include <vector>

/*
Auxiliary function to check dimensions of given parameters (taken from pairwise_distances.hpp)
*/
template <typename T>
inline da_status check_input(da_order order, da_int m, da_int n, da_int k, const T *X,
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
inline da_status create_work_arrays(da_int m, da_int &n, const T *Y,
                                    std::vector<T> &x_work, std::vector<T> &y_work,
                                    bool &X_is_Y) {
    try {
        x_work.resize(m);
    } catch (std::bad_alloc const &) {
        return da_status_memory_error;
    }
    if (Y) {
        try {
            y_work.resize(n);
        } catch (std::bad_alloc const &) {
            return da_status_memory_error;
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
    rbf_kernel_local(order, m, n, k, X, x_work.data(), ldx, Y, y_work.data(), ldy, D, ldd,
                     gamma, X_is_Y);
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
    linear_kernel_local(order, m, n, k, X, ldx, Y, ldy, D, ldd, X_is_Y);
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
    polynomial_kernel_local(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, degree, coef0,
                            X_is_Y);
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
    sigmoid_kernel_local(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, coef0, X_is_Y);
    return status;
}