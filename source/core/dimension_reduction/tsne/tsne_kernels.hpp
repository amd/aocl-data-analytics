/* ************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
