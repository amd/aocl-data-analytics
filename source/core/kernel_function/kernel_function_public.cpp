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

#include "aoclda.h"
#include "kernel_function.hpp"

da_status da_rbf_kernel_d(da_order order, da_int m, da_int n, da_int k, const double *X,
                          da_int ldx, const double *Y, da_int ldy, double *D, da_int ldd,
                          double gamma) {
    return rbf_kernel(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma);
}

da_status da_rbf_kernel_s(da_order order, da_int m, da_int n, da_int k, const float *X,
                          da_int ldx, const float *Y, da_int ldy, float *D, da_int ldd,
                          float gamma) {
    return rbf_kernel(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma);
}

da_status da_linear_kernel_d(da_order order, da_int m, da_int n, da_int k,
                             const double *X, da_int ldx, const double *Y, da_int ldy,
                             double *D, da_int ldd) {
    return linear_kernel(order, m, n, k, X, ldx, Y, ldy, D, ldd);
}

da_status da_linear_kernel_s(da_order order, da_int m, da_int n, da_int k, const float *X,
                             da_int ldx, const float *Y, da_int ldy, float *D,
                             da_int ldd) {
    return linear_kernel(order, m, n, k, X, ldx, Y, ldy, D, ldd);
}

da_status da_polynomial_kernel_d(da_order order, da_int m, da_int n, da_int k,
                                 const double *X, da_int ldx, const double *Y, da_int ldy,
                                 double *D, da_int ldd, double gamma, da_int degree,
                                 double coef0) {
    return polynomial_kernel(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, degree,
                             coef0);
}

da_status da_polynomial_kernel_s(da_order order, da_int m, da_int n, da_int k,
                                 const float *X, da_int ldx, const float *Y, da_int ldy,
                                 float *D, da_int ldd, float gamma, da_int degree,
                                 float coef0) {
    return polynomial_kernel(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, degree,
                             coef0);
}

da_status da_sigmoid_kernel_d(da_order order, da_int m, da_int n, da_int k,
                              const double *X, da_int ldx, const double *Y, da_int ldy,
                              double *D, da_int ldd, double gamma, double coef0) {
    return sigmoid_kernel(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, coef0);
}

da_status da_sigmoid_kernel_s(da_order order, da_int m, da_int n, da_int k,
                              const float *X, da_int ldx, const float *Y, da_int ldy,
                              float *D, da_int ldd, float gamma, float coef0) {
    return sigmoid_kernel(order, m, n, k, X, ldx, Y, ldy, D, ldd, gamma, coef0);
}