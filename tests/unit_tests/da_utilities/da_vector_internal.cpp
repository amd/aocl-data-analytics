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
#include "da_vector.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <list>
#include <vector>

template <typename T> class da_vector_internal_test : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using Types = ::testing::Types<float, double, da_int>;
TYPED_TEST_SUITE(da_vector_internal_test, Types);

TYPED_TEST(da_vector_internal_test, push_back) {

    da_vector::da_vector<TypeParam> vec;
    EXPECT_EQ(vec.size(), 0);
    EXPECT_EQ(vec.capacity(), INIT_CAPACITY);

    da_vector::da_vector<TypeParam> vec2(INIT_CAPACITY + 2);
    EXPECT_EQ(vec2.size(), INIT_CAPACITY + 2);
    EXPECT_EQ(vec2.capacity(), INIT_CAPACITY * 2);

    da_int test_size = 123;
    for (da_int i = 0; i < test_size; i++) {
        vec.push_back((TypeParam)i);
        EXPECT_EQ(vec.size(), i + 1);
        EXPECT_EQ(vec[i], (TypeParam)i);
    }
}

TYPED_TEST(da_vector_internal_test, append) {
    da_vector::da_vector<TypeParam> vec1, vec2, vec3;
    da_int size_vec1 = 56, size_vec3 = 100;
    for (da_int i = 0; i < size_vec1; i++) {
        vec1.push_back((TypeParam)i);
    }
    for (da_int i = 0; i < size_vec3; i++) {
        vec3.push_back((TypeParam)(i + size_vec1));
    }

    vec1.append(vec2);
    vec1.append(vec3);

    EXPECT_EQ(vec1.size(), size_vec1 + size_vec3);

    for (da_int i = 0; i < size_vec1 + size_vec3; i++) {
        EXPECT_EQ(vec1[i], (TypeParam)i);
    }

    vec2.append(vec3);

    EXPECT_EQ(vec2.size(), size_vec3);

    for (da_int i = 0; i < size_vec3; i++) {
        EXPECT_EQ(vec2[i], vec3[i]);
    }
}

TYPED_TEST(da_vector_internal_test, append_std_vec) {
    da_vector::da_vector<TypeParam> vec1;
    std::vector<TypeParam> vec2, vec3;
    da_int size_vec1 = 56, size_vec3 = 100;
    for (da_int i = 0; i < size_vec1; i++) {
        vec1.push_back((TypeParam)i);
    }
    for (da_int i = 0; i < size_vec3; i++) {
        vec3.push_back((TypeParam)(i + size_vec1));
    }

    vec1.append(vec2);
    vec1.append(vec3);

    EXPECT_EQ(vec1.size(), size_vec1 + size_vec3);

    for (da_int i = 0; i < size_vec1 + size_vec3; i++) {
        EXPECT_EQ(vec1[i], (TypeParam)i);
    }
}