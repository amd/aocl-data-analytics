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
#include "aoclda_types.h"
#include "da_error.hpp"
#include "macros.h"
#include <vector>

namespace ARCH {

/*
Auxiliary function to check dimensions of given parameters (taken from pairwise_distances.hpp)
*/

template <typename T>
inline da_status check_input(da_order order, da_int m, da_int n, da_int k, const T *X,
                             da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd);
/*
Auxiliary function to create work arrays
*/
template <typename T>
inline da_status create_work_arrays(da_int m, da_int &n, const T *Y,
                                    std::vector<T> &x_work, std::vector<T> &y_work,
                                    bool &X_is_Y);
/*
RBF kernel
Given an m by k matrix X and an n by k matrix Y (both column major), computes the m by n kernel matrix D
*/
template <typename T>
da_status rbf_kernel(da_order order, da_int m, da_int n, da_int k, const T *X, da_int ldx,
                     const T *Y, da_int ldy, T *D, da_int ldd, T gamma);
/*
Linear kernel
*/
template <typename T>
da_status linear_kernel(da_order order, da_int m, da_int n, da_int k, const T *X,
                        da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd);
/*
Polynomial kernel
*/
template <typename T>
da_status polynomial_kernel(da_order order, da_int m, da_int n, da_int k, const T *X,
                            da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd, T gamma,
                            da_int degree, T coef0);
/*
Sigmoid kernel
*/
template <typename T>
da_status sigmoid_kernel(da_order order, da_int m, da_int n, da_int k, const T *X,
                         da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd, T gamma,
                         T coef0);
/*
RBF kernel
Given an m by k matrix X and an n by k matrix Y (both column major), computes the m by n kernel matrix D
*/
template <typename T>
void rbf_kernel_internal(da_order order, da_int m, da_int n, da_int k, const T *X,
                         T *X_norms, da_int ldx, const T *Y, T *Y_norms, da_int ldy, T *D,
                         da_int ldd, T gamma, bool X_is_Y);
/*
Linear kernel
*/
template <typename T>
void linear_kernel_internal(da_order order, da_int m, da_int n, da_int k, const T *X,
                            da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd,
                            bool X_is_Y);
/*
Polynomial kernel
*/
template <typename T>
void polynomial_kernel_internal(da_order order, da_int m, da_int n, da_int k, const T *X,
                                da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd,
                                T gamma, da_int degree, T coef0, bool X_is_Y);
/*
Sigmoid kernel
*/
template <typename T>
void sigmoid_kernel_internal(da_order order, da_int m, da_int n, da_int k, const T *X,
                             da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd,
                             T gamma, T coef0, bool X_is_Y);
/*
Helper function to transpose upper trainagular matrix to a symmetric
*/
template <typename T>
inline void fill_upper_traingular(da_order order, da_int m, T *D, da_int ldd);

/*
Helper function to compute gemm/syrk
*/
template <typename T>
inline void kernel_setup(da_order order, da_int m, da_int n, da_int k, const T *X,
                         da_int ldx, const T *Y, da_int ldy, T *D, da_int ldd, T gamma,
                         bool X_is_Y);

} // namespace ARCH