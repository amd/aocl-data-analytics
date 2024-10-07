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

#include "interval_set.hpp"
#include "aoclda.h"
#include "interval.hpp"
#include <set>

namespace da_interval {
da_status interval_set::insert(interval bounds) {
    da_int lb = bounds.lower, ub = bounds.upper;
    if (ub < lb)
        return da_status_invalid_input;

    // Find intervals with bigger upper bound and smaller lower bound
    inter_set::iterator it_upper = iset.begin();
    inter_set::iterator it_lower = iset.begin();
    while (it_upper != iset.end() && it_upper->upper < ub) {
        it_upper++;
        if (it_upper != iset.end() && it_upper->lower <= lb)
            it_lower = it_upper;
    }
    if (it_lower != iset.end() && it_lower->lower > lb)
        it_lower = iset.end();

    // Do nothing if bounds is entirely contained into one of the existing intervals
    if ((it_lower != iset.end() && it_lower->lower <= lb && it_lower->upper >= ub) ||
        (it_upper != iset.end() && it_upper->lower <= lb && it_upper->upper >= ub))
        return da_status_success;

    // Every interval between it_lower and it_upper in included in the inserted interval
    inter_set::iterator it = it_lower;
    if (it != iset.end())
        it++;
    else
        it = iset.begin();
    while (it != iset.end() && it != it_upper) {
        it = iset.erase(it);
    }

    // Merge the lower and upper intervals with the inserted one if necessary
    if (it_lower != iset.end() && it_lower->upper >= lb - 1) {
        lb = it_lower->lower;
        iset.erase(it_lower);
    }
    if (it_upper != iset.end() && it_upper->lower <= ub + 1) {
        ub = it_upper->upper;
        iset.erase(it_upper);
    }

    iset.insert({lb, ub});
    return da_status_success;
}

bool interval_set::find(da_int key, interval &inc) {
    if (iset.empty())
        return false;
    interval key_interval{key, key};
    auto it_ub = iset.upper_bound(key_interval);

    if (it_ub != iset.end() && it_ub->contains(key)) {
        inc = *it_ub;
        return true;
    } else if (it_ub == iset.end() || (it_ub->lower > key && it_ub != iset.begin()))
        it_ub--;

    if (it_ub->contains(key)) {
        inc = *it_ub;
        return true;
    }
    return false;
}

interval_set::iterator interval_set::find(da_int key) {
    if (iset.empty())
        return this->end();
    interval key_interval{key, key};
    auto it_ub = iset.upper_bound(key_interval);

    if (it_ub != iset.end() && it_ub->contains(key)) {
        return iterator(it_ub);
    } else if (it_ub == iset.end() || (it_ub->lower > key && it_ub != iset.begin()))
        it_ub--;

    if (it_ub->contains(key)) {
        return iterator(it_ub);
    }
    return this->end();
}

da_status interval_set::erase(interval bounds) {
    da_int lb = bounds.lower, ub = bounds.upper;
    if (ub < lb)
        return da_status_invalid_input;

    // Find intervals with bigger upper bound and smaller lower bound
    inter_set::iterator it_upper = iset.begin();
    inter_set::iterator it_lower = iset.begin();
    while (it_upper != iset.end() && it_upper->upper < ub) {
        it_upper++;
        if (it_upper != iset.end() && it_upper->lower <= lb)
            it_lower = it_upper;
    }
    if (it_lower != iset.end() && it_lower->lower > lb)
        it_lower = iset.end();

    // Every interval between it_lower and it_upper in included in the inserted interval
    bool is_included = it_lower == it_upper;
    if (!is_included) {
        inter_set::iterator it = it_lower;
        if (it != iset.end())
            it++;
        else
            it = iset.begin();
        while (it != iset.end() && it != it_upper) {
            it = iset.erase(it);
        }
    }

    // Remove bounds from the lower and upper interval
    da_int new_lb, new_ub;
    if (it_lower != iset.end()) {
        new_lb = it_lower->lower;
        new_ub = it_lower->upper;
        if (it_lower->upper >= lb) {
            iset.erase(it_lower);
            if (lb >= new_lb + 1)
                iset.insert({new_lb, lb - 1});
            if (ub <= new_ub - 1)
                iset.insert({ub + 1, new_ub});
        }
    }
    if (!is_included && it_upper != iset.end()) {
        new_lb = it_upper->lower;
        new_ub = it_upper->upper;
        if (it_upper->lower <= ub) {
            iset.erase(it_upper);
            if (lb >= new_lb + 1)
                iset.insert({new_lb, lb - 1});
            if (ub <= new_ub - 1)
                iset.insert({ub + 1, new_ub});
        }
    }
    return da_status_success;
}

void interval_set::clear() { iset.clear(); }

} // namespace da_interval