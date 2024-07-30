/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef DA_UTILITIES_HPP
#define DA_UTILITIES_HPP

#include "aoclda.h"
#include <type_traits>

namespace da_utils {

/* Convert number into char array, appropriately depending on its type */
template <typename T, size_t U>
constexpr da_status convert_num_to_char(T num, char character[U]) {
    static_assert(std::is_arithmetic_v<T>,
                  "Error in convert_num_to_char function. T must be numerical "
                  "value");
    if constexpr (std::is_same_v<T, float>)
        sprintf(character, "%9.2e", num);
    else if constexpr (std::is_same_v<T, double>)
        sprintf(character, "%9.2e", num);
    else if constexpr (std::is_same_v<T, int>)
        sprintf(character, "%d", num);
    else if constexpr (std::is_same_v<T, long int>)
        sprintf(character, "%ld", num);
    else if constexpr (std::is_same_v<T, long long int>)
        sprintf(character, "%lld", num);
    return da_status_success;
}

void blocking_scheme(da_int n_samples, da_int block_size, da_int &n_blocks,
                     da_int &block_rem);

da_int get_n_threads_loop(da_int loop_size);

} // namespace da_utils

#endif
