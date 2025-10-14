/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "macros.h"
#include <algorithm>
#include <vector>

namespace ARCH {

namespace da_decision_forest {
da_status compress_count_occurences(std::vector<da_int> &indices,
                                    std::vector<da_int> &count);

template <typename T>
void swap(std::vector<da_int> &a, std::vector<T> &b, da_int i, da_int j);

template <typename T>
void multi_range_heap_sort(std::vector<da_int> &indices, std::vector<T> &values,
                           da_int start_idx, da_int n_elem);

template <typename T>
void multi_range_intro_sort(std::vector<da_int> &indices, std::vector<T> &values,
                            da_int start_idx, da_int n_elem, da_int max_depth);

template <class T>
void bucket_sort_samples(std::vector<da_int> &indices, std::vector<T> &values,
                         da_int n_cat, da_int start_idx, da_int n_elem, da_int ldbuck,
                         std::vector<da_int> &buckets, std::vector<da_int> &bucket_idx);

} // namespace da_decision_forest
} // namespace ARCH