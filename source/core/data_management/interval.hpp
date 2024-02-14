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

#ifndef INTERVAL_HPP
#define INTERVAL_HPP

#include "aoclda.h"
#include <algorithm>

namespace da_interval {
struct interval {
    da_int lower = -1, upper = -1;

    interval() = default;
    interval(da_int first, da_int second) : lower(first), upper(second) {}

    bool operator==(const interval &i) { return lower == i.lower && upper == i.upper; };
    bool operator!=(const interval &i) { return lower != i.lower || upper != i.upper; };
    bool contains(da_int val) const { return val >= lower && val <= upper; };
    interval intersect(const interval &i) const {
        da_int lb = std::max(lower, i.lower);
        da_int ub = std::min(upper, i.upper);
        return {lb, ub};
    };
    bool is_valid_idx(da_int max_val) {
        if (lower > upper)
            return false;
        if (lower < 0 || upper >= max_val)
            return false;
        return true;
    }
};
} // namespace da_interval
#endif