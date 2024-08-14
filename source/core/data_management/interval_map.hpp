/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef INTERVAL_MAP_HPP
#define INTERVAL_MAP_HPP

#include "aoclda.h"
#include "interval.hpp"
#include <iostream>
#include <map>
#include <set>

/* Simplistic implementation of interval maps using std::maps
 *
 * Intervals are represented as pair of integers and serve as keys.
 * All are closed
 *
 * intervals CANNOT overlap: da_status_invalid_input will be returned if trying to create overlaps
 */

namespace da_interval {

template <class T> class interval_map {

    struct comp_interval_map {
        constexpr bool operator()(const interval &p1, const interval &p2) const {
            bool res = p1.lower < p2.lower;
            // Unintuitive: interval included in another is considered bigger
            if (p1.lower == p2.lower)
                res = p1.upper > p2.upper;
            return res;
        }
    };

    template <class U>
    using inter_map = typename std::map<interval, U, comp_interval_map>;
    using map_it = typename inter_map<T>::iterator;

    inter_map<T> imap;
    map_it it;

  public:
    class iterator {
        map_it it;

      public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = map_it;
        using pointer = map_it *;
        using reference = map_it &;

        iterator() {}
        iterator(map_it new_it) : it(new_it) {}
        std::pair<const interval, T> &operator*() const { return *it; }
        std::pair<const interval, T> *operator->() { return &(*it); }
        iterator &operator++() {
            ++it;
            return *this;
        }
        iterator operator++(int) {
            // Returns dereferencable iterator
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
        friend interval_map;
    };

    iterator begin() {
        it = imap.begin();
        return iterator(it);
    }
    iterator end() {
        it = imap.end();
        return iterator(it);
    }

    bool empty() { return imap.empty(); }

    da_status insert(interval bounds, T val) {
        da_int lb = bounds.lower, ub = bounds.upper;
        if (ub < lb)
            return da_status_invalid_input;

        if (find(lb) != imap.end() || find(ub) != imap.end())
            // lb or ub are already in the intervals => trying to create an overlap
            return da_status_invalid_input;

        imap.insert(std::make_pair(bounds, val));
        return da_status_success;
    }

    /* Try to find the key in the interval map
     * If found, the value as well as the lower and upper bounds of the containing interval
     * are returned on the interface
     */
    bool find(da_int key, T &val, da_int &lb, da_int &ub) {
        if (imap.empty())
            return false;
        bool found = false;
        interval key_pair(key, key);
        auto it_lb = imap.lower_bound(key_pair);
        if (it_lb == imap.end() || key_pair != it_lb->first)
            --it_lb;
        if (it_lb != imap.end()) {
            lb = it_lb->first.lower;
            ub = it_lb->first.upper;
            if (lb <= key && key <= ub) {
                val = it_lb->second;
                found = true;
            }
        }
        return found;
    }
    iterator find(da_int key) {
        if (imap.empty()) {
            return iterator(imap.end());
        }
        interval key_pair(key, key);
        auto it_lb = imap.lower_bound(key_pair);
        if (it_lb == imap.end() || key_pair != it_lb->first)
            --it_lb;
        iterator it(it_lb);
        if (it == this->end() || key < it->first.lower || key > it->first.upper)
            it = this->end();
        return it;
    }

    /* Returns an iterator to the biggest interval smaller than the key.
     * If no such element exists the iterator points to the closest bigger interval
     */
    iterator closest_interval(da_int key) {
        if (imap.empty()) {
            return imap.end();
        }
        interval key_pair(key, key);
        auto it_lb = imap.lower_bound(key_pair);
        if (it_lb == imap.end() || key_pair != it_lb->first)
            --it_lb;
        iterator it(it_lb);
        return it;
    }

    /* Erases the interval containing key or all intervals between first
     * and last iterators (last excluded)
     */
    iterator erase(da_int key) {
        iterator it = find(key);
        if (it != imap.end())
            it = iterator(imap.erase(it.it));
        return it;
    }
    iterator erase(iterator pos) {
        iterator it = imap.end();
        if (pos != imap.end()) {
            auto it_imap = imap.erase(pos.it);
            it = iterator(it_imap);
        }
        return it;
    }
    iterator erase(iterator first, iterator last) {
        iterator it = imap.end();
        if (first != imap.end()) {
            auto it_imap = imap.erase(first.it, last.it);
            it = iterator(it_imap);
        }
        return it;
    }
};

} // namespace da_interval
#endif