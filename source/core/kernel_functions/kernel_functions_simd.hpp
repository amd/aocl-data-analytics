/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#include "kt.hpp"
#include "macros.h"

namespace ARCH {

namespace da_kernel_functions {

using namespace kernel_templates;

/* first_dim represents the dimension we iterate over first, for example in column-major it is number of rows.
second_dim represents the dimension we iterate over second, for example in column-major it is number of columns.
This is to prevent creating switch-case for row/column major data. */

template <typename T>
void exp_kernel_scalar(da_int first_dim, da_int second_dim, T *data, da_int ldd,
                       T multiplier, const T *first_dim_norms, const T *second_dim_norms);

template <typename T>
void pow_kernel_scalar(da_int first_dim, da_int second_dim, T *data, da_int ldd, T coef0,
                       da_int degree);

template <typename T>
void tanh_kernel_scalar(da_int first_dim, da_int second_dim, T *data, da_int ldd,
                        T coef0);

template <bsz SZ, typename SUF>
void exp_kt(da_int first_dim, da_int second_dim, SUF *data, da_int ldd, SUF multiplier,
            const SUF *first_dim_norms, const SUF *second_dim_norms);

template <bsz SZ, typename SUF>
void pow_kt(da_int first_dim, da_int second_dim, SUF *data, da_int ldd, SUF coef0,
            da_int degree);

template <bsz SZ, typename SUF>
void tanh_kt(da_int first_dim, da_int second_dim, SUF *data, da_int ldd, SUF coef0);

} // namespace da_kernel_functions

} // namespace ARCH