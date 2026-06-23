/*
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "context.hpp"
#include "da_error.hpp"
#include "dynamic_dispatch.hpp"
#include "macros.h"

da_errors::error_bypass_t *nosave_kernel(nullptr);

template <typename T>
da_status da_rbf_kernel(da_order order, da_int m, da_int n, da_int k, const T *X,
                        da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd, T gamma) {
    DISPATCHER(nosave_kernel, return (da_kernel_functions::rbf_kernel(
                                  order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma)));
}

template <typename T>
da_status da_linear_kernel(da_order order, da_int m, da_int n, da_int k, const T *X,
                           da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd) {
    DISPATCHER(nosave_kernel, return (da_kernel_functions::linear_kernel(
                                  order, m, n, k, X, ldx, Y, ldy, D, ldd)));
}

template <typename T>
da_status da_polynomial_kernel(da_order order, da_int m, da_int n, da_int k, const T *X,
                               da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd,
                               T gamma, da_int degree, T coef0) {
    DISPATCHER(nosave_kernel,
               return (da_kernel_functions::polynomial_kernel(
                   order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, degree, coef0)));
}

template <typename T>
da_status da_sigmoid_kernel(da_order order, da_int m, da_int n, da_int k, const T *X,
                            da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd, T gamma,
                            T coef0) {
    DISPATCHER(nosave_kernel, return (da_kernel_functions::sigmoid_kernel(
                                  order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, coef0)));
}

template da_status da_rbf_kernel<float>(da_order, da_int, da_int, da_int, const float *,
                                        da_int, const float *, da_int, float *, da_int,
                                        float);
template da_status da_rbf_kernel<double>(da_order, da_int, da_int, da_int, const double *,
                                         da_int, const double *, da_int, double *, da_int,
                                         double);
template da_status da_linear_kernel<float>(da_order, da_int, da_int, da_int,
                                           const float *, da_int, const float *, da_int,
                                           float *, da_int);
template da_status da_linear_kernel<double>(da_order, da_int, da_int, da_int,
                                            const double *, da_int, const double *,
                                            da_int, double *, da_int);
template da_status da_polynomial_kernel<float>(da_order, da_int, da_int, da_int,
                                               const float *, da_int, const float *,
                                               da_int, float *, da_int, float, da_int,
                                               float);
template da_status da_polynomial_kernel<double>(da_order, da_int, da_int, da_int,
                                                const double *, da_int, const double *,
                                                da_int, double *, da_int, double, da_int,
                                                double);
template da_status da_sigmoid_kernel<float>(da_order, da_int, da_int, da_int,
                                            const float *, da_int, const float *, da_int,
                                            float *, da_int, float, float);
template da_status da_sigmoid_kernel<double>(da_order, da_int, da_int, da_int,
                                             const double *, da_int, const double *,
                                             da_int, double *, da_int, double, double);