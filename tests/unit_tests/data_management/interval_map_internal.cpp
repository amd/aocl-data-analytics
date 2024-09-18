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

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "interval_map.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <string>

using namespace da_interval;

TEST(intervalMap, invalidInput) {
    interval_map<double> imap;
    da_int lb, ub;
    double d;

    // invalid bounds
    interval bounds = {2, 0};
    EXPECT_EQ(imap.insert(bounds, 1.0), da_status_invalid_input);

    // find a key in empty map
    EXPECT_EQ(imap.find(0, d, lb, ub), false);
    EXPECT_EQ(imap.find(0), imap.end());

    // Insert correct interval [0,2]
    bounds = {0, 2};
    EXPECT_EQ(imap.insert(bounds, 1.0), da_status_success);

    // erase the end iterator
    imap.erase(imap.end());

    // find values outside of the inserted intervals
    EXPECT_EQ(imap.find(-1, d, lb, ub), false);
    EXPECT_EQ(imap.find(3, d, lb, ub), false);
    EXPECT_EQ(imap.find(-1), imap.end());
    EXPECT_EQ(imap.find(3), imap.end());
    EXPECT_EQ(imap.find(1, d, lb, ub), true);
    EXPECT_EQ(d, 1.0);
    EXPECT_EQ(lb, 0);
    EXPECT_EQ(ub, 2);
    auto it = imap.find(1);
    EXPECT_EQ(it->second, 1.0);
    EXPECT_EQ(it->first.lower, 0);
    EXPECT_EQ(it->first.upper, 2);

    // overlapping intervals
    EXPECT_EQ(imap.insert(interval{1, 3}, 1.0), da_status_invalid_input);
    EXPECT_EQ(imap.insert(interval{2, 3}, 1.0), da_status_invalid_input);
    EXPECT_EQ(imap.insert(interval{-1, 0}, 1.0), da_status_invalid_input);
    EXPECT_EQ(imap.insert(interval{0, 0}, 1.0), da_status_invalid_input);
    EXPECT_EQ(imap.insert(interval{2, 2}, 1.0), da_status_invalid_input);

    // add disjointed interval and try to find a value between them
    EXPECT_EQ(imap.insert(interval{5, 10}, 2.0), da_status_success);
    EXPECT_EQ(imap.find(4, d, lb, ub), false);
}

TEST(intervalMap, positive) {

    interval_map<char> imap;
    char c;
    da_int lb, ub;

    EXPECT_EQ(imap.insert(interval{0, 2}, 'a'), da_status_success);
    EXPECT_EQ(imap.insert(interval{4, 9}, 'b'), da_status_success);
    EXPECT_EQ(imap.find(0, c, lb, ub), true);
    EXPECT_EQ(c, 'a');
    EXPECT_EQ(lb, 0);
    EXPECT_EQ(ub, 2);
    auto it = imap.find(0);
    EXPECT_EQ(it->second, 'a');
    EXPECT_EQ(imap.find(2, c, lb, ub), true);
    EXPECT_EQ(c, 'a');
    EXPECT_EQ(lb, 0);
    EXPECT_EQ(ub, 2);
    EXPECT_EQ(imap.find(5, c, lb, ub), true);
    EXPECT_EQ(c, 'b');
    EXPECT_EQ(lb, 4);
    EXPECT_EQ(ub, 9);
    it = imap.find(5);
    EXPECT_EQ(it->second, 'b');
    EXPECT_EQ(imap.find(9, c, lb, ub), true);
    EXPECT_EQ(c, 'b');
    EXPECT_EQ(lb, 4);
    EXPECT_EQ(ub, 9);
    it = imap.find(9);
    EXPECT_EQ(it->second, 'b');

    EXPECT_EQ(imap.insert(interval{15, 20}, 'c'), da_status_success);
    EXPECT_EQ(imap.find(17, c, lb, ub), true);
    EXPECT_EQ(c, 'c');
    EXPECT_EQ(lb, 15);
    EXPECT_EQ(ub, 20);
    it = imap.find(17);
    EXPECT_EQ(it->second, 'c');
}

TEST(intervalMap, erase) {
    interval_map<char> imap;

    // insert intervals
    // [0,2] [4,9] [10,11] [12, 22] [24, 28] [30, 35] [55, 60]
    EXPECT_EQ(imap.insert(interval{0, 2}, 'a'), da_status_success);
    EXPECT_EQ(imap.insert(interval{4, 9}, 'b'), da_status_success);
    EXPECT_EQ(imap.insert(interval{10, 11}, 'c'), da_status_success);
    EXPECT_EQ(imap.insert(interval{12, 22}, 'd'), da_status_success);
    EXPECT_EQ(imap.insert(interval{55, 60}, 'g'), da_status_success);
    EXPECT_EQ(imap.insert(interval{30, 35}, 'f'), da_status_success);
    EXPECT_EQ(imap.insert(interval{24, 28}, 'e'), da_status_success);

    // erase a few intervals
    // leaves: [0,2] [24, 28] [30, 35] [55, 60]
    EXPECT_EQ(imap.erase(13)->second, 'e');
    EXPECT_EQ(imap.find(15), imap.end());
    interval_map<char>::iterator it1, it2;
    it1 = imap.find(9);
    it2 = imap.find(27);
    imap.erase(it1, it2);
    EXPECT_EQ(imap.find(5), imap.end());
    EXPECT_EQ(imap.find(10), imap.end());
    EXPECT_EQ(imap.find(25)->second, 'e');
    EXPECT_EQ(imap.find(35)->second, 'f');
    EXPECT_EQ(imap.find(55)->second, 'g');

    // try to erase invalid keys or iterators
    EXPECT_EQ(imap.erase(12), imap.end());
    EXPECT_EQ(imap.erase(imap.end(), imap.end()), imap.end());

    // erase all intervals from [30, 35]
    // [0,2] [24, 28]
    it1 = imap.find(31);
    it2 = imap.end();
    imap.erase(it1, it2);
    EXPECT_EQ(imap.find(35), imap.end());
    EXPECT_EQ(imap.find(59), imap.end());
    EXPECT_EQ(imap.find(1)->second, 'a');
    EXPECT_EQ(imap.find(28)->second, 'e');

    // erase [24, 28] with single iterator
    it1 = imap.find(26);
    imap.erase(it1);
    EXPECT_EQ(imap.find(28), imap.end());
}

TEST(intervalMap, iterator) {
    interval_map<char> imap;

    EXPECT_EQ(imap.insert(interval{0, 2}, 'a'), da_status_success);
    EXPECT_EQ(imap.insert(interval{4, 9}, 'b'), da_status_success);
    EXPECT_EQ(imap.insert(interval{10, 10}, 'c'), da_status_success);
    EXPECT_EQ(imap.insert(interval{12, 20}, 'd'), da_status_success);

    char vals[4] = {'a', 'b', 'c', 'd'};
    da_int i = 0;
    for (auto it = imap.begin(); it != imap.end(); ++it) {
        EXPECT_EQ(it->second, vals[i]);
        i++;
    }
    EXPECT_EQ(i, 4);
    i = 0;
    for (auto it = imap.begin(); it != imap.end(); it++) {
        EXPECT_EQ((*it).second, vals[i]);
        i++;
    }
    EXPECT_EQ(i, 4);
}

TEST(interval, intersect) {
    interval i1 = {1, 3}, i2 = {2, 4};
    interval res = i1.intersect(i2);
    EXPECT_EQ(res.lower, 2);
    EXPECT_EQ(res.upper, 3);
    res = i2.intersect(i1);
    EXPECT_EQ(res.lower, 2);
    EXPECT_EQ(res.upper, 3);
    i1 = {-1, -3};
    res = i1.intersect(i2);
    EXPECT_EQ(res.lower, 2);
    EXPECT_EQ(res.upper, -3);
}