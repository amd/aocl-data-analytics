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

#ifndef DA_CACHE_HPP
#define DA_CACHE_HPP

#include "aoclda.h"
#include "da_error.hpp"
#include "da_utils.hpp"
#include "macros.h"
#include <da_omp.hpp>
#include <list>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

/**
  * LRU (Least Recently Used) Cache implementation
  */
namespace da_cache {
template <typename T> class LRUCache {
  private:
    // Maximum number of columns the cache can hold
    da_int capacity_;
    // Length of each column (number of rows)
    da_int len_;

    // List to track usage order (front = most recently used)
    std::list<std::pair<da_int, T *>> items_;

    // Map for O(1) access to items in the list
    std::unordered_map<da_int, typename std::list<std::pair<da_int, T *>>::iterator>
        cache_;
    T *data_ = nullptr;
    // n_cols_free - number of columns that we can still cache without overwriting
    da_int n_cols_free;
    // free_col_idx - index of the next free column (it can work without it but, but left it for readability)
    da_int free_col_idx = 0;
    // Temprorary vector to hold pointers to free memory locations in cache each time we call put()
    std::vector<T *> free_pointers;

    da_errors::da_error_t *err = nullptr;

  public:
    // Flag to indicate if the cache is being used
    bool active_;

    LRUCache(da_errors::da_error_t &err) : err(&err) {}

    ~LRUCache() {
        if (data_) {
            delete[] data_;
        }
    }

    da_int get_capacity() const { return capacity_; }

    /**
      * Get value for key and mark it as most recently used
      * @param key The key to look up
      * @return Pointer to the value if found, nullptr otherwise
      */
    T *get(const da_int &key) {
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            return nullptr; // Key not found
        }

        // Move the accessed item to the front of the list
        items_.splice(items_.begin(), items_, it->second);
        return it->second->second;
    }

    /**
      * Add or update a key-value pair
      * @param key The array of keys to insert or update
      * @param value The array of pointers to first value to associated with the key
      */
    da_status put(const std::vector<da_int> &key, std::vector<T *> &value) {
        // PART 1 - Copy values into the cache memory (performance critical)
        da_int n_requested = key.size();
        // If cache capacity is smaller than idx_to_compute_count, we put in cache only
        // first cache_capacity pointers
        get_free_pointers(n_requested, free_pointers);
        da_int n_failed = 0;
        [[maybe_unused]] da_int n_threads = ARCH::da_utils::get_n_threads_loop(64);
        // Copy values to the free pointers
#pragma omp parallel for if (n_requested > 16) schedule(dynamic) default(none)           \
    shared(n_requested, free_pointers, value, len_) reduction(+ : n_failed)              \
    num_threads(n_threads)
        for (da_int i = 0; i < n_requested; i++) {
            if (value[i] == nullptr) {
                n_failed++;
                continue;
            }
            for (da_int j = 0; j < len_; j++) {
                free_pointers[i][j] = value[i][j];
            }
        }
        if (n_failed > 0) {
            return da_error(this->err, da_status_invalid_pointer,
                            "One or more pointers in the value array were invalid. ");
        }
        // PART 2 - Update the doubly linked list to keep track of the order of usage (LRU policy)
        // Check if cache is full
        for (da_int i = 0; i < n_requested; i++) {
            if ((da_int)items_.size() >= capacity_) {
                // Remove the least recently used item (back of the list)
                da_int lru_key = items_.back().first;
                items_.pop_back();
                cache_.erase(lru_key);
            }

            // Add new item to the front
            items_.emplace_front(key[i], free_pointers[i]);
            cache_[key[i]] = items_.begin();
        }
        return da_status_success;
    }

    /**
      * Set size of cache
      * @param capacity Size of cache in "number of rows" it can hold
      * @param len Number of columns for a single row that is being held
      */
    da_status set_size(const da_int &capacity, const da_int &len) {
        capacity_ = capacity;
        len_ = len;
        active_ = (capacity_ > 0);
        if (active_) {
            try {
                data_ = new T[capacity * len];
                free_pointers.resize(capacity);
            } catch (std::bad_alloc &) {
                return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                                "Failed to allocate memory for cache.");
            }
        }
        n_cols_free = capacity;
        free_col_idx = 0;
        return da_status_success;
    }

  private:
    /**
     * Retrieves a specified number of free memory pointers from the cache.
     * 
     * This function iterates from last free memory column in the cache.
     * If insufficient free columns are found, it will overwrite the least recently used cache columns
     * to get requested number of pointers.
     * 
     * @param n The number of free pointers requested. On output, it will be set to the actual number of pointers retrieved.
     * @param free_pointers Vector to store the retrieved free pointers. Must be pre-sized
     *                     to accommodate at least n pointers.
     */
    void get_free_pointers(da_int &n, std::vector<T *> &free_pointers) {
        // If cache capacity is smaller than the number of pointers we requested, we put in cache only
        // the first pointers to fill the capacity. Update the requested number.
        n = std::min(capacity_, n);
        da_int free_ptrs_needed = n;

        da_int idx = 0;
        // Retrieve free columns until we run out or satisfy the request
        while (free_ptrs_needed > 0 && n_cols_free > 0) {
            free_pointers[idx++] = &data_[free_col_idx * len_];
            ++free_col_idx;
            --n_cols_free;
            --free_ptrs_needed;
        }

        // If we still need pointers, overwrite least recently used cache items
        auto it = items_.rbegin();
        while (free_ptrs_needed > 0 && it != items_.rend()) {
            free_pointers[idx++] = it->second;
            ++it;
            --free_ptrs_needed;
        }
    }
};
} // namespace da_cache

#endif