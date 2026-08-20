/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
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