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

#include "../utest_utils.hpp"
#include "da_cache.hpp"
#include "da_error.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <list>
#include <vector>

template <typename T> class da_cache_internal_test : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using Types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(da_cache_internal_test, Types);

TYPED_TEST(da_cache_internal_test, get_capacity) {
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    da_cache::LRUCache<TypeParam> cache(err);
    EXPECT_EQ(cache.set_size(10, 5), da_status_success);
    EXPECT_EQ(cache.get_capacity(), 10);
}

TYPED_TEST(da_cache_internal_test, get_empty_cache) {
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    da_cache::LRUCache<TypeParam> cache(err);
    EXPECT_EQ(cache.set_size(10, 5), da_status_success);
    EXPECT_EQ(cache.get(0), nullptr);
    EXPECT_EQ(cache.get(10), nullptr);
    EXPECT_EQ(cache.get(100), nullptr);
    EXPECT_EQ(cache.get(-1), nullptr);
}

TYPED_TEST(da_cache_internal_test, cache_functionality) {
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    da_cache::LRUCache<TypeParam> cache(err);
    // We will first store 2 floating points
    EXPECT_EQ(cache.set_size(5, 1), da_status_success);
    std::vector<da_int> keys_1 = {0, 1};
    TypeParam value_1 = 0.1, value_2 = 11.2;
    std::vector<TypeParam *> values_1 = {&value_1, &value_2};
    cache.put(keys_1, values_1);
    EXPECT_EQ(*cache.get(0), *values_1[0]);
    EXPECT_EQ(*cache.get(1), *values_1[1]);
    // Now we will add 3 more values, which will fill up the capacity
    std::vector<da_int> keys_2 = {2, 3, 4};
    TypeParam value_3 = 22.2, value_4 = 33.3, value_5 = 44.4;
    std::vector<TypeParam *> values_2 = {&value_3, &value_4, &value_5};
    cache.put(keys_2, values_2);
    EXPECT_EQ(*cache.get(2), *values_2[0]);
    EXPECT_EQ(*cache.get(3), *values_2[1]);
    EXPECT_EQ(*cache.get(4), *values_2[2]);
    // Now we add one more value, which will evict the least recently used item (0)
    std::vector<da_int> keys_3 = {5};
    TypeParam value_6 = 55.5;
    std::vector<TypeParam *> values_3 = {&value_6};
    cache.put(keys_3, values_3);
    EXPECT_EQ(*cache.get(5), *values_3[0]);
    // The item with key 0 should have been evicted
    EXPECT_EQ(cache.get(0), nullptr);
    // The item with key 1 should still be there
    EXPECT_EQ(*cache.get(1), *values_1[1]);
}

TYPED_TEST(da_cache_internal_test, cache_overwrite) {
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    da_cache::LRUCache<TypeParam> cache(err);
    EXPECT_EQ(cache.set_size(5, 1), da_status_success);
    // We will first store 5 floating points
    std::vector<da_int> keys_1 = {0, 1, 2, 3, 4};
    TypeParam value_1 = 0.1, value_2 = 11.2, value_3 = 22.2, value_4 = 33.3,
              value_5 = 44.4;
    std::vector<TypeParam *> values_1 = {&value_1, &value_2, &value_3, &value_4,
                                         &value_5};
    cache.put(keys_1, values_1);
    EXPECT_EQ(*cache.get(0), *values_1[0]);
    EXPECT_EQ(*cache.get(1), *values_1[1]);
    EXPECT_EQ(*cache.get(2), *values_1[2]);
    EXPECT_EQ(*cache.get(3), *values_1[3]);
    EXPECT_EQ(*cache.get(4), *values_1[4]);
    // Now we will add 5 more values, under the same keys, which will overwrite the existing values
    TypeParam value_6 = 55.5, value_7 = 66.6, value_8 = 77.7, value_9 = 88.8,
              value_10 = 99.9;
    std::vector<TypeParam *> values_2 = {&value_6, &value_7, &value_8, &value_9,
                                         &value_10};
    cache.put(keys_1, values_2);
    EXPECT_EQ(*cache.get(0), *values_2[0]);
    EXPECT_EQ(*cache.get(1), *values_2[1]);
    EXPECT_EQ(*cache.get(2), *values_2[2]);
    EXPECT_EQ(*cache.get(3), *values_2[3]);
    EXPECT_EQ(*cache.get(4), *values_2[4]);
    // Now we will put 5 more values into new keys, which will evict the first 5 items
    std::vector<da_int> keys_2 = {5, 6, 7, 8, 9};
    TypeParam value_11 = 101.1, value_12 = 112.2, value_13 = 123.3, value_14 = 134.4,
              value_15 = 145.5;
    std::vector<TypeParam *> values_3 = {&value_11, &value_12, &value_13, &value_14,
                                         &value_15};
    cache.put(keys_2, values_3);
    EXPECT_EQ(*cache.get(5), *values_3[0]);
    EXPECT_EQ(*cache.get(6), *values_3[1]);
    EXPECT_EQ(*cache.get(7), *values_3[2]);
    EXPECT_EQ(*cache.get(8), *values_3[3]);
    EXPECT_EQ(*cache.get(9), *values_3[4]);
    // The items with keys 0, 1, 2, 3, 4 should be nullptr
    EXPECT_EQ(cache.get(0), nullptr);
    EXPECT_EQ(cache.get(1), nullptr);
    EXPECT_EQ(cache.get(2), nullptr);
    EXPECT_EQ(cache.get(3), nullptr);
    EXPECT_EQ(cache.get(4), nullptr);
}

TYPED_TEST(da_cache_internal_test, put_more_than_capacity) {
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    da_cache::LRUCache<TypeParam> cache(err);
    EXPECT_EQ(cache.set_size(5, 1), da_status_success);
    // Try to put 10 items into cache with capacity 5
    // The cache should only hold the first 5 items
    std::vector<da_int> keys_1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    TypeParam value_1 = 0.1, value_2 = 0.2, value_3 = 0.3, value_4 = 0.4, value_5 = 0.5,
              value_6 = 0.6, value_7 = 0.7, value_8 = 0.8, value_9 = 0.9, value_10 = 1.0;
    std::vector<TypeParam *> values_1 = {&value_1, &value_2, &value_3, &value_4,
                                         &value_5, &value_6, &value_7, &value_8,
                                         &value_9, &value_10};
    cache.put(keys_1, values_1);
    EXPECT_EQ(*cache.get(0), *values_1[0]);
    EXPECT_EQ(*cache.get(1), *values_1[1]);
    EXPECT_EQ(*cache.get(2), *values_1[2]);
    EXPECT_EQ(*cache.get(3), *values_1[3]);
    EXPECT_EQ(*cache.get(4), *values_1[4]);
    // The items with keys 5, 6, 7, 8, 9 should be nullptr
    EXPECT_EQ(cache.get(5), nullptr);
    EXPECT_EQ(cache.get(6), nullptr);
    EXPECT_EQ(cache.get(7), nullptr);
    EXPECT_EQ(cache.get(8), nullptr);
    EXPECT_EQ(cache.get(9), nullptr);
}

TYPED_TEST(da_cache_internal_test, lru_access_order) {
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    da_cache::LRUCache<TypeParam> cache(err);
    EXPECT_EQ(cache.set_size(3, 1), da_status_success);

    // Fill cache with 3 items
    std::vector<da_int> keys = {1, 2, 3};
    TypeParam value_1 = 1.0, value_2 = 2.0, value_3 = 3.0;
    std::vector<TypeParam *> values = {&value_1, &value_2, &value_3};
    cache.put(keys, values);

    // Access item 1 to make it most recently used
    EXPECT_EQ(*cache.get(1), value_1);

    // Add new item, should evict item 2 (oldest unaccessed)
    std::vector<da_int> new_keys = {4};
    TypeParam value_4 = 4.0;
    std::vector<TypeParam *> new_values = {&value_4};
    cache.put(new_keys, new_values);

    // Item 2 should be evicted, others should remain
    EXPECT_EQ(cache.get(2), nullptr);
    EXPECT_EQ(*cache.get(1), value_1);
    EXPECT_EQ(*cache.get(3), value_3);
    EXPECT_EQ(*cache.get(4), value_4);
}

TYPED_TEST(da_cache_internal_test, capacity_one) {
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    da_cache::LRUCache<TypeParam> cache(err);
    EXPECT_EQ(cache.set_size(1, 1), da_status_success);

    // Add first item
    std::vector<da_int> keys_1 = {1};
    TypeParam value_1 = 1.0;
    std::vector<TypeParam *> values_1 = {&value_1};
    cache.put(keys_1, values_1);
    EXPECT_EQ(*cache.get(1), value_1);

    // Add second item, should evict first
    std::vector<da_int> keys_2 = {2};
    TypeParam value_2 = 2.0;
    std::vector<TypeParam *> values_2 = {&value_2};
    cache.put(keys_2, values_2);
    EXPECT_EQ(cache.get(1), nullptr);
    EXPECT_EQ(*cache.get(2), value_2);
}

TYPED_TEST(da_cache_internal_test, put_empty_vectors) {
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    da_cache::LRUCache<TypeParam> cache(err);
    EXPECT_EQ(cache.set_size(5, 1), da_status_success);

    // Put empty vectors should not crash or change cache state
    std::vector<da_int> keys = {1, 2};
    std::vector<TypeParam *> values = {nullptr, nullptr};
    da_status status = cache.put(keys, values);
    EXPECT_EQ(status, da_status_invalid_pointer);

    // Cache should still be empty
    EXPECT_EQ(cache.get(0), nullptr);
    EXPECT_EQ(cache.get(1), nullptr);
}

TYPED_TEST(da_cache_internal_test, sequential_eviction_pattern) {
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    da_cache::LRUCache<TypeParam> cache(err);
    EXPECT_EQ(cache.set_size(3, 1), da_status_success);

    // Fill cache
    for (da_int i = 0; i < 3; i++) {
        std::vector<da_int> keys = {i};
        TypeParam value = static_cast<TypeParam>(i);
        std::vector<TypeParam *> values = {&value};
        cache.put(keys, values);
    }

    // Add more items one by one and verify LRU eviction
    for (da_int i = 3; i < 6; i++) {
        std::vector<da_int> keys = {i};
        TypeParam value = static_cast<TypeParam>(i);
        std::vector<TypeParam *> values = {&value};
        cache.put(keys, values);

        // The item that was added (i-3) iterations ago should be evicted
        EXPECT_EQ(cache.get(i - 3), nullptr);
        // Recent items should still be there
        EXPECT_NE(cache.get(i - 2), nullptr);
        EXPECT_NE(cache.get(i - 1), nullptr);
        EXPECT_NE(cache.get(i), nullptr);
    }
}