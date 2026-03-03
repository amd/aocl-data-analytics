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
    TypeParam values_1[] = {(TypeParam)0.1, (TypeParam)11.2};
    cache.put(keys_1, values_1, 1);
    EXPECT_EQ(*cache.get(0), values_1[0]);
    EXPECT_EQ(*cache.get(1), values_1[1]);
    // Now we will add 3 more values, which will fill up the capacity
    std::vector<da_int> keys_2 = {2, 3, 4};
    TypeParam values_2[] = {(TypeParam)22.2, (TypeParam)33.3, (TypeParam)44.4};
    cache.put(keys_2, values_2, 1);
    EXPECT_EQ(*cache.get(2), values_2[0]);
    EXPECT_EQ(*cache.get(3), values_2[1]);
    EXPECT_EQ(*cache.get(4), values_2[2]);
    // Now we add one more value, which will evict the least recently used item (0)
    std::vector<da_int> keys_3 = {5};
    TypeParam values_3[] = {(TypeParam)55.5};
    cache.put(keys_3, values_3, 1);
    EXPECT_EQ(*cache.get(5), values_3[0]);
    // The item with key 0 should have been evicted
    EXPECT_EQ(cache.get(0), nullptr);
    // The item with key 1 should still be there
    EXPECT_EQ(*cache.get(1), values_1[1]);
}

TYPED_TEST(da_cache_internal_test, cache_overwrite) {
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    da_cache::LRUCache<TypeParam> cache(err);
    EXPECT_EQ(cache.set_size(5, 1), da_status_success);
    // We will first store 5 floating points
    std::vector<da_int> keys_1 = {0, 1, 2, 3, 4};
    TypeParam values_1[] = {(TypeParam)0.1, (TypeParam)11.2, (TypeParam)22.2,
                            (TypeParam)33.3, (TypeParam)44.4};
    cache.put(keys_1, values_1, 1);
    EXPECT_EQ(*cache.get(0), values_1[0]);
    EXPECT_EQ(*cache.get(1), values_1[1]);
    EXPECT_EQ(*cache.get(2), values_1[2]);
    EXPECT_EQ(*cache.get(3), values_1[3]);
    EXPECT_EQ(*cache.get(4), values_1[4]);
    // Now we will add 5 more values, under the same keys, which will overwrite the existing values
    TypeParam values_2[] = {(TypeParam)55.5, (TypeParam)66.6, (TypeParam)77.7,
                            (TypeParam)88.8, (TypeParam)99.9};
    cache.put(keys_1, values_2, 1);
    EXPECT_EQ(*cache.get(0), values_2[0]);
    EXPECT_EQ(*cache.get(1), values_2[1]);
    EXPECT_EQ(*cache.get(2), values_2[2]);
    EXPECT_EQ(*cache.get(3), values_2[3]);
    EXPECT_EQ(*cache.get(4), values_2[4]);
    // Now we will put 5 more values into new keys, which will evict the first 5 items
    std::vector<da_int> keys_2 = {5, 6, 7, 8, 9};
    TypeParam values_3[] = {(TypeParam)101.1, (TypeParam)112.2, (TypeParam)123.3,
                            (TypeParam)134.4, (TypeParam)145.5};
    cache.put(keys_2, values_3, 1);
    EXPECT_EQ(*cache.get(5), values_3[0]);
    EXPECT_EQ(*cache.get(6), values_3[1]);
    EXPECT_EQ(*cache.get(7), values_3[2]);
    EXPECT_EQ(*cache.get(8), values_3[3]);
    EXPECT_EQ(*cache.get(9), values_3[4]);
    // The items with keys 0, 1, 2, 3, 4 should be nullptr
    EXPECT_EQ(cache.get(0), nullptr);
    EXPECT_EQ(cache.get(1), nullptr);
    EXPECT_EQ(cache.get(2), nullptr);
    EXPECT_EQ(cache.get(3), nullptr);
    EXPECT_EQ(cache.get(4), nullptr);
}

TYPED_TEST(da_cache_internal_test, put_with_larger_stride) {
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    da_cache::LRUCache<TypeParam> cache(err);
    EXPECT_EQ(cache.set_size(4, 2), da_status_success);
    // We will store 3 columns with a stride larger than len
    std::vector<da_int> keys_1 = {10, 11, 12};
    da_int stride = 5;
    TypeParam values_1[] = {
        (TypeParam)1.1, (TypeParam)1.2, (TypeParam)9.9, (TypeParam)9.9, (TypeParam)9.9,
        (TypeParam)2.1, (TypeParam)2.2, (TypeParam)8.8, (TypeParam)8.8, (TypeParam)8.8,
        (TypeParam)3.1, (TypeParam)3.2, (TypeParam)7.7, (TypeParam)7.7, (TypeParam)7.7,
    };
    cache.put(keys_1, values_1, stride);

    EXPECT_EQ(cache.get(10)[0], values_1[0]);
    EXPECT_EQ(cache.get(10)[1], values_1[1]);
    EXPECT_EQ(cache.get(11)[0], values_1[5]);
    EXPECT_EQ(cache.get(11)[1], values_1[6]);
    EXPECT_EQ(cache.get(12)[0], values_1[10]);
    EXPECT_EQ(cache.get(12)[1], values_1[11]);
}

TYPED_TEST(da_cache_internal_test, overwrite_with_larger_stride) {
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    da_cache::LRUCache<TypeParam> cache(err);
    EXPECT_EQ(cache.set_size(3, 2), da_status_success);
    // First store 2 columns with a larger stride
    std::vector<da_int> keys_1 = {1, 2};
    da_int stride = 4;
    TypeParam values_1[] = {
        (TypeParam)10.1, (TypeParam)10.2, (TypeParam)1.0, (TypeParam)1.0,
        (TypeParam)20.1, (TypeParam)20.2, (TypeParam)2.0, (TypeParam)2.0,
    };
    cache.put(keys_1, values_1, stride);
    EXPECT_EQ(cache.get(1)[0], values_1[0]);
    EXPECT_EQ(cache.get(1)[1], values_1[1]);
    EXPECT_EQ(cache.get(2)[0], values_1[4]);
    EXPECT_EQ(cache.get(2)[1], values_1[5]);

    // Now overwrite the same keys with new values using the same stride
    TypeParam values_2[] = {
        (TypeParam)110.1, (TypeParam)110.2, (TypeParam)3.0, (TypeParam)3.0,
        (TypeParam)120.1, (TypeParam)120.2, (TypeParam)4.0, (TypeParam)4.0,
    };
    cache.put(keys_1, values_2, stride);
    EXPECT_EQ(cache.get(1)[0], values_2[0]);
    EXPECT_EQ(cache.get(1)[1], values_2[1]);
    EXPECT_EQ(cache.get(2)[0], values_2[4]);
    EXPECT_EQ(cache.get(2)[1], values_2[5]);
}

TYPED_TEST(da_cache_internal_test, put_more_than_capacity) {
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    da_cache::LRUCache<TypeParam> cache(err);
    EXPECT_EQ(cache.set_size(5, 1), da_status_success);
    // Try to put 10 items into cache with capacity 5
    // The cache should only hold the first 5 items
    std::vector<da_int> keys_1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    TypeParam values_1[] = {
        (TypeParam)0.1, (TypeParam)0.2, (TypeParam)0.3, (TypeParam)0.4, (TypeParam)0.5,
        (TypeParam)0.6, (TypeParam)0.7, (TypeParam)0.8, (TypeParam)0.9, (TypeParam)1.0};
    cache.put(keys_1, values_1, 1);
    EXPECT_EQ(*cache.get(0), values_1[0]);
    EXPECT_EQ(*cache.get(1), values_1[1]);
    EXPECT_EQ(*cache.get(2), values_1[2]);
    EXPECT_EQ(*cache.get(3), values_1[3]);
    EXPECT_EQ(*cache.get(4), values_1[4]);
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
    TypeParam values[] = {(TypeParam)1.0, (TypeParam)2.0, (TypeParam)3.0};
    cache.put(keys, values, 1);

    // Access item 1 to make it most recently used
    EXPECT_EQ(*cache.get(1), values[0]);

    // Add new item, should evict item 2 (oldest unaccessed)
    std::vector<da_int> new_keys = {4};
    TypeParam new_values[] = {(TypeParam)4.0};
    cache.put(new_keys, new_values, 1);

    // Item 2 should be evicted, others should remain
    EXPECT_EQ(cache.get(2), nullptr);
    EXPECT_EQ(*cache.get(1), values[0]);
    EXPECT_EQ(*cache.get(3), values[2]);
    EXPECT_EQ(*cache.get(4), new_values[0]);
}

TYPED_TEST(da_cache_internal_test, capacity_one) {
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    da_cache::LRUCache<TypeParam> cache(err);
    EXPECT_EQ(cache.set_size(1, 1), da_status_success);

    // Add first item
    std::vector<da_int> keys_1 = {1};
    TypeParam values_1[] = {(TypeParam)1.0};
    cache.put(keys_1, values_1, 1);
    EXPECT_EQ(*cache.get(1), values_1[0]);

    // Add second item, should evict first
    std::vector<da_int> keys_2 = {2};
    TypeParam values_2[] = {(TypeParam)2.0};
    cache.put(keys_2, values_2, 1);
    EXPECT_EQ(cache.get(1), nullptr);
    EXPECT_EQ(*cache.get(2), values_2[0]);
}

TYPED_TEST(da_cache_internal_test, put_null_data) {
    da_errors::da_error_t err(da_errors::action_t::DA_RECORD);
    da_cache::LRUCache<TypeParam> cache(err);
    EXPECT_EQ(cache.set_size(5, 1), da_status_success);

    // Put null data should return error
    std::vector<da_int> keys = {1, 2};
    da_status status = cache.put(keys, nullptr, 1);
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
        cache.put(keys, &value, 1);
    }

    // Add more items one by one and verify LRU eviction
    for (da_int i = 3; i < 6; i++) {
        std::vector<da_int> keys = {i};
        TypeParam value = static_cast<TypeParam>(i);
        cache.put(keys, &value, 1);

        // The item that was added (i-3) iterations ago should be evicted
        EXPECT_EQ(cache.get(i - 3), nullptr);
        // Recent items should still be there
        EXPECT_NE(cache.get(i - 2), nullptr);
        EXPECT_NE(cache.get(i - 1), nullptr);
        EXPECT_NE(cache.get(i), nullptr);
    }
}