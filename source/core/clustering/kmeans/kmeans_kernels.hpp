/* ************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

#include "aoclda_types.h"
#include "kt.hpp"
#include "macros.h"

namespace ARCH {

namespace da_kmeans {

using namespace kernel_templates;

template <class T>
void elkan_iteration_kernel_scalar(da_int, T *, da_int, T *, T *, da_int *, da_int);

template <kernel_templates::bsz SZ, typename T>
void elkan_iteration_kt(da_int, T *, da_int, T *, T *, da_int *, da_int);

template <typename T, vectorization_type U>
void elkan_iteration_kernel(da_int, T *, da_int, T *, T *, da_int *, da_int);

template <class T> T elkan_reduction_kernel_scalar(da_int, const T *, T *);

template <kernel_templates::bsz SZ, typename T>
T elkan_reduction_kt(da_int, const T *, T *);

template <typename T, vectorization_type U>
T elkan_reduction_kernel(da_int, const T *, T *);

template <class T>
void lloyd_iteration_kernel_scalar(bool, da_int, T *, da_int *, da_int *, T *, da_int,
                                   da_int);

template <typename T, vectorization_type U>
void lloyd_iteration_kernel(bool, da_int, T *, da_int *, da_int *, T *, da_int, da_int);

} // namespace da_kmeans
} // namespace ARCH
