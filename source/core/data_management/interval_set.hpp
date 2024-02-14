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

#ifndef INTERVAL_SET_HPP
#define INTERVAL_SET_HPP
#include "aoclda.h"
#include "interval.hpp"
#include <set>

namespace da_interval {

class interval_set {

    struct comp_interval_set {
        constexpr bool operator()(const interval &p1, const interval &p2) const {
            bool res = p1.lower < p2.lower;
            if (p1.lower == p2.lower)
                res = p1.upper < p2.upper;
            return res;
        }
    };
    using inter_set = std::set<interval, comp_interval_set>;
    using set_it = typename inter_set::iterator;

    inter_set iset;
    set_it it;

  public:
    class iterator {
        set_it it;

      public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = set_it;
        using pointer = set_it *;
        using reference = set_it &;

        iterator() {}
        iterator(set_it new_it) : it(new_it) {}
        const interval &operator*() const { return *it; }
        const interval *operator->() { return &(*it); }
        iterator &operator++() {
            ++it;
            return *this;
        }
        iterator operator++(int) {
            // returns dereferencable iterator
            iterator res = *this;
            ++(*this);
            return res;
        }
        friend bool operator==(const iterator &i1, const iterator &i2) {
            return i1.it == i2.it;
        };
        friend bool operator!=(const iterator &i1, const iterator &i2) {
            return i1.it != i2.it;
        };
    };

    iterator begin() {
        it = iset.begin();
        return iterator(it);
    }
    iterator end() {
        it = iset.end();
        return iterator(it);
    }
    bool empty() { return iset.empty(); }

    da_status insert(interval bounds);

    bool find(da_int key, interval &inc);
    iterator find(da_int key);

    da_status erase(interval bounds);

    void clear();
};
} // namespace da_interval

#endif