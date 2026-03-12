/*
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
    da_int capacity_ = 0;
    // Length of each column (number of rows)
    da_int len_ = 0;

    // List to track usage order (front = most recently used)
    std::list<std::pair<da_int, T *>> items_;

    // Map for O(1) access to items in the list
    std::unordered_map<da_int, typename std::list<std::pair<da_int, T *>>::iterator>
        cache_;
    T *data_ = nullptr;
    // n_cols_free - number of columns that we can still cache without overwriting
    da_int n_cols_free = 0;
    // free_col_idx - index of the next free column (it can work without it but, but left it for readability)
    da_int free_col_idx = 0;

    da_errors::da_error_t *err = nullptr;

    omp_lock_t cache_lock_;

  public:
    // Flag to indicate if the cache is being used
    bool active_ = false;

    LRUCache(da_errors::da_error_t &err) : err(&err) { omp_init_lock(&cache_lock_); }

    ~LRUCache() {
        if (data_) {
            delete[] data_;
        }
        omp_destroy_lock(&cache_lock_);
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
      * Add or update a key-value pair from contiguous column data
      * @param key The array of keys to insert or update
      * @param data Pointer to the start of the first column
      * @param stride Number of elements between successive columns
      */
    da_status put(const std::vector<da_int> &key, const T *data, da_int stride) {
        da_int n_requested = (da_int)key.size();
        // PART 1 - Get the pointers to free data from cache data_ buffer
        std::vector<T *> free_pointers;
        da_int target = std::min(n_requested, capacity_);
        free_pointers.reserve(target);
        omp_set_lock(&cache_lock_);
        // Use free slots first
        while ((da_int)free_pointers.size() < target && n_cols_free > 0) {
            free_pointers.emplace_back(&data_[free_col_idx * len_]);
            ++free_col_idx;
            --n_cols_free;
        }
        // Evict LRU items if needed and point to freed space
        while ((da_int)free_pointers.size() < target && !items_.empty()) {
            auto it = std::prev(items_.end());
            free_pointers.emplace_back(it->second);
            cache_.erase(it->first);
            items_.erase(it);
        }
        omp_unset_lock(&cache_lock_);
        da_int actual = (da_int)free_pointers.size();

        // PART 2 - Copy values directly from contiguous data into free pointers
        da_int n_failed = 0;
        [[maybe_unused]] da_int n_threads = ARCH::da_utils::get_n_threads_loop(64);
#pragma omp parallel for if (actual > 16) schedule(dynamic) default(none)                \
    shared(actual, free_pointers, data, stride, len_) reduction(+ : n_failed)            \
    num_threads(n_threads)
        for (da_int i = 0; i < actual; i++) {
            if (data == nullptr || free_pointers[i] == nullptr) {
                n_failed++;
                continue;
            }
            const T *src = data + i * stride;
            for (da_int j = 0; j < len_; j++) {
                free_pointers[i][j] = src[j];
            }
        }
        if (n_failed > 0) {
            return da_error(this->err, da_status_invalid_pointer,
                            "One or more pointers in the value array were invalid. ");
        }
        // PART 3 - Update the doubly linked list to keep track of the order of usage (LRU policy)
        omp_set_lock(&cache_lock_);
        for (da_int i = 0; i < actual; i++) {
            // Add new item to the front
            items_.emplace_front(key[i], free_pointers[i]);
            cache_[key[i]] = items_.begin();
        }
        omp_unset_lock(&cache_lock_);
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
            } catch (std::bad_alloc &) {
                return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                                "Failed to allocate memory for cache.");
            }
        }
        n_cols_free = capacity;
        free_col_idx = 0;
        return da_status_success;
    }
};
} // namespace da_cache

#endif