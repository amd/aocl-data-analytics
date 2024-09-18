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

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "interval.hpp"
#include "interval_set.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <string>

using namespace da_interval;

TEST(intervalSet, invalidInput) {
    interval_set iset;

    interval b1{50, 45};
    EXPECT_EQ(iset.insert(b1), da_status_invalid_input);
}

TEST(intervalSet, insert) {
    interval_set iset;

    // set: [10, 20]
    EXPECT_EQ(iset.insert({10, 20}), da_status_success);

    // fully contained insertion: do nothing
    EXPECT_EQ(iset.insert({10, 12}), da_status_success);
    auto it = iset.begin();
    EXPECT_EQ(it->lower, 10);
    EXPECT_EQ(it->upper, 20);
    it++;
    EXPECT_EQ(it, iset.end());

    // add another discontinued interval
    // iset = [10, 20]; [30, 35]
    EXPECT_EQ(iset.insert({30, 35}), da_status_success);
    it = iset.begin();
    it++;
    EXPECT_EQ(it->lower, 30);
    EXPECT_EQ(it->upper, 35);

    // Add contiguous interval from the beginning [5, 9]
    // iset = [5, 20]; [30, 35]
    EXPECT_EQ(iset.insert({5, 9}), da_status_success);
    it = iset.begin();
    EXPECT_EQ(it->lower, 5);
    EXPECT_EQ(it->upper, 20);

    // Add partially contained interval from the beginning [4, 5]
    // iset = [4, 20]; [30, 35]
    EXPECT_EQ(iset.insert({4, 6}), da_status_success);
    it = iset.begin();
    EXPECT_EQ(it->lower, 4);
    EXPECT_EQ(it->upper, 20);

    // Add discontiguous interval from the middle [22, 24]
    // iset = [4, 20]; [22, 24]; [30, 35]
    EXPECT_EQ(iset.insert({22, 24}), da_status_success);
    it = iset.begin();
    it++;
    EXPECT_EQ(it->lower, 22);
    EXPECT_EQ(it->upper, 24);

    // iset = [4, 20]; [22, 26]; [30,35]
    EXPECT_EQ(iset.insert({25, 26}), da_status_success);
    it = iset.begin();
    it++;
    EXPECT_EQ(it->lower, 22);
    EXPECT_EQ(it->upper, 26);

    // iset = [4, 26]; [30, 35]
    EXPECT_EQ(iset.insert({21, 21}), da_status_success);
    it = iset.begin();
    EXPECT_EQ(it->lower, 4);
    EXPECT_EQ(it->upper, 26);
    it++;
    EXPECT_EQ(it->lower, 30);
    EXPECT_EQ(it->upper, 35);

    // iset = [4, 26]; [30, 36]; [40, 45]
    EXPECT_EQ(iset.insert({32, 36}), da_status_success);
    EXPECT_EQ(iset.insert({40, 45}), da_status_success);
    it = iset.begin();
    it++;
    EXPECT_EQ(it->lower, 30);
    EXPECT_EQ(it->upper, 36);
    it++;
    EXPECT_EQ(it->lower, 40);
    EXPECT_EQ(it->upper, 45);

    // iset = [0, 45]
    EXPECT_EQ(iset.insert({0, 41}), da_status_success);
    it = iset.begin();
    EXPECT_EQ(it->lower, 0);
    EXPECT_EQ(it->upper, 45);

    // iset = [0, 45]; [50, 52]; [54, 54]
    EXPECT_EQ(iset.insert({50, 52}), da_status_success);
    it = iset.begin();
    it++;
    EXPECT_EQ(it->lower, 50);
    EXPECT_EQ(it->upper, 52);
    EXPECT_EQ(iset.insert({54, 54}), da_status_success);

    // iset = [0, 45]; [49, 55]
    EXPECT_EQ(iset.insert({49, 55}), da_status_success);
    it = iset.begin();
    it++;
    EXPECT_EQ(it->lower, 49);
    EXPECT_EQ(it->upper, 55);
    it++;
    EXPECT_EQ(it, iset.end());
}

TEST(intervalSet, find) {

    interval_set iset;

    // test empty interval set
    interval inc{42, 42};
    EXPECT_FALSE(iset.find(0, inc));
    EXPECT_EQ(iset.find(1), iset.end());

    // insert [0,2] and [5,7] into the map
    interval b1{0, 2};
    interval b2{5, 7};
    EXPECT_EQ(iset.insert(b1), da_status_success);
    EXPECT_EQ(iset.insert(b2), da_status_success);

    // Try to find element outside of all intervals
    EXPECT_FALSE(iset.find(-1, inc));
    EXPECT_FALSE(iset.find(3, inc));
    EXPECT_FALSE(iset.find(8, inc));

    // Check for valid values
    EXPECT_TRUE(iset.find(5, inc));
    EXPECT_EQ(inc.lower, b2.lower);
    EXPECT_EQ(inc.upper, b2.upper);
    EXPECT_TRUE(iset.find(0, inc));
    EXPECT_EQ(inc.lower, b1.lower);
    EXPECT_EQ(inc.upper, b1.upper);
    EXPECT_TRUE(iset.find(2, inc));
    EXPECT_EQ(inc.lower, b1.lower);
    EXPECT_EQ(inc.upper, b1.upper);
    EXPECT_TRUE(iset.find(6, inc));
    EXPECT_EQ(inc.lower, b2.lower);
    EXPECT_EQ(inc.upper, b2.upper);

    // find iterator of absent elements
    EXPECT_EQ(iset.find(-1), iset.end());
    EXPECT_EQ(iset.find(8), iset.end());
    EXPECT_EQ(iset.find(3), iset.end());
    EXPECT_EQ(iset.find(4), iset.end());

    // find valid iterator
    interval_set::iterator it1 = iset.begin();                // [0,2]
    interval_set::iterator it2 = interval_set::iterator(it1); // [5,7]
    it2++;
    EXPECT_EQ(iset.find(0), it1);
    EXPECT_EQ(iset.find(1), it1);
    EXPECT_EQ(iset.find(5), it2);
    EXPECT_EQ(iset.find(6), it2);
    EXPECT_EQ(iset.find(7), it2);
}

TEST(intervalSet, erase) {
    interval_set iset;

    // iset = [0,5]; [7,8]; [10;12]
    EXPECT_EQ(iset.insert({0, 3}), da_status_success);
    EXPECT_EQ(iset.insert({1, 5}), da_status_success);
    EXPECT_EQ(iset.insert({7, 8}), da_status_success);
    EXPECT_EQ(iset.insert({10, 12}), da_status_success);

    // iset = [0,3]; [5,5]; [7,8]; [10,12]
    EXPECT_EQ(iset.erase({4, 4}), da_status_success);
    auto it = iset.begin();
    EXPECT_EQ(it->lower, 0);
    EXPECT_EQ(it->upper, 3);
    it++;
    EXPECT_EQ(it->lower, 5);
    EXPECT_EQ(it->upper, 5);
    it++;
    EXPECT_EQ(it->lower, 7);
    EXPECT_EQ(it->upper, 8);
    it++;
    EXPECT_EQ(it->lower, 10);
    EXPECT_EQ(it->upper, 12);
    it++;
    EXPECT_EQ(it, iset.end());

    // iset = [0,3]; [5,5]; [7,8]; [10,10]; [12,12]
    EXPECT_EQ(iset.erase({11, 11}), da_status_success);
    it = iset.find(10);
    it++;
    EXPECT_EQ(it->lower, 12);
    EXPECT_EQ(it->upper, 12);

    // iset = [0,3]; [12,12]
    EXPECT_EQ(iset.erase({4, 10}), da_status_success);
    it = iset.begin();
    EXPECT_EQ(it->lower, 0);
    EXPECT_EQ(it->upper, 3);
    it++;
    EXPECT_EQ(it->lower, 12);
    EXPECT_EQ(it->upper, 12);
    it++;
    EXPECT_EQ(it, iset.end());

    // iset = empty
    EXPECT_EQ(iset.erase({0, 12}), da_status_success);
    EXPECT_TRUE(iset.empty());

    // iset = [0, 10]; [15, 20]; [22, 30]
    EXPECT_EQ(iset.insert({0, 10}), da_status_success);
    EXPECT_EQ(iset.insert({15, 20}), da_status_success);
    EXPECT_EQ(iset.insert({22, 30}), da_status_success);
    //iset = [0,9]; [25, 30]
    EXPECT_EQ(iset.erase({10, 24}), da_status_success);
    it = iset.begin();
    EXPECT_EQ(it->lower, 0);
    EXPECT_EQ(it->upper, 9);
    it++;
    EXPECT_EQ(it->lower, 25);
    EXPECT_EQ(it->upper, 30);
}
