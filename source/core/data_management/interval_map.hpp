#ifndef INTERVAL_MAP_HPP
#define INTERVAL_MAP_HPP

#include "aoclda.h"
#include <map>

/* Simplistic implementation of interval maps using std::maps
 * 
 * Intervals are represented as pair of integers and serve as keys. 
 * All are closed
 * 
 * intervals CANNOT overlap: da_status_invalid_input will be returned if trying to create overlaps  
 * 
 * removal of intervals not supported for now
 */

namespace da_interval_map {
using pair = std::pair<da_int, da_int>;

struct comp_ipair {
    constexpr bool operator()(const pair &p1, const pair &p2) const {
        bool res = p1.first < p2.first;
        // Unintuitive: interval included in another is considered bigger
        // A bit hacky, TODO review
        if (p1.first == p2.first)
            res = p1.second > p2.second;
        return res;
    }
};

template <class T> class interval_map {

    template <class U> using inter_map = typename std::map<pair, U, comp_ipair>;
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
        std::pair<const pair, T> &operator*() const { return *it; }
        std::pair<const pair, T> *operator->() { return &(*it); }
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

    da_status insert(da_int lb, da_int ub, T val) {
        if (ub < lb)
            return da_status_invalid_input;

        if (find(lb) != imap.end() || find(ub) != imap.end())
            // lb or ub are already in the intervals => trying to create an overlap
            return da_status_invalid_input;

        imap.insert(std::make_pair(pair(lb, ub), val));
        return da_status_success;
    }

    /* try to find the key in the interval map
     * If found the value as well as the lower and upper bounds of the containing interval 
     * are returned on the interface
     */
    bool find(da_int key, T &val, da_int &lb, da_int &ub) {
        if (imap.empty())
            return false;
        bool found = false;
        pair key_pair(key, key);
        auto it_lb = imap.lower_bound(key_pair);
        if (it_lb == imap.end() || key_pair != it_lb->first)
            --it_lb;
        if (it_lb != imap.end()) {
            lb = it_lb->first.first;
            ub = it_lb->first.second;
            if (lb <= key && key <= ub) {
                val = it_lb->second;
                found = true;
            }
        }
        return found;
    }
    iterator find(da_int key) {
        if (imap.empty()) {
            return imap.end();
        }
        pair key_pair(key, key);
        auto it_lb = imap.lower_bound(key_pair);
        if (it_lb == imap.end() || key_pair != it_lb->first)
            --it_lb;
        iterator it(it_lb);
        if (it == this->end() || key < it->first.first || key > it->first.second)
            it = this->end();
        return it;
    }

    /* erases the interval containing key or all intervals between first 
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

} // namespace da_interval_map
#endif