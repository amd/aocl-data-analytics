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

#ifndef TSNE_KERNELS_HPP
#define TSNE_KERNELS_HPP

#include "aoclda.h"
#include "kt.hpp"
#include <cstdint>

namespace ARCH::da_tsne {

template <typename T, int8_t D>
void attractive_forces_scalar_impl(const T *emb_i, const da_int *col_idx, const T *P_vals,
                                   const T *embedding, T exaggeration, da_int start,
                                   da_int end, T *grad_i);

template <kernel_templates::bsz SZ, typename SUF, int8_t D>
void attractive_forces_kt(const SUF *emb_i, const da_int *col_idx, const SUF *P_vals,
                          const SUF *embedding, SUF exaggeration, da_int start,
                          da_int end, SUF *grad_i);

template <kernel_templates::bsz SZ, typename SUF>
void attractive_forces_multi_d2(const SUF *emb_i, const da_int *col_idx,
                                const SUF *P_vals, const SUF *embedding, SUF exaggeration,
                                da_int start, da_int end, SUF *grad_i);
} // namespace ARCH::da_tsne

#endif // TSNE_KERNELS_HPP
