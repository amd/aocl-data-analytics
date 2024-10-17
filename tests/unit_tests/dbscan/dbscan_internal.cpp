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

#include <iostream>
#include <limits>
#include <list>
#include <numeric>
#include <stdio.h>
#include <string.h>

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "da_error.hpp"
#include "da_vector.hpp"
#include "radius_neighbors.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

template <typename T> class DBSCANTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(DBSCANTest, FloatTypes);

TYPED_TEST(DBSCANTest, radius_neighbors_small) {

    da_int n_samples = 10;
    da_int n_features = 2;
    da_int lda = 10;
    TypeParam eps = 1.5;
    std::vector<double> A_double = {0.0,  -5.0, -6.0, 0.1,  0.1,  10.0, -0.1,
                                    -0.1, -5.5, -5.0, 0.0,  -5.0, -6.0, 0.1,
                                    -0.1, 10.0, 0.1,  -0.1, -5.5, -6.0};

    std::vector<TypeParam> A = convert_vector<double, TypeParam>(A_double);

    std::vector<da_vector::da_vector<da_int>> neighbors(n_samples);

    std::vector<da_vector::da_vector<da_int>> neighbors_exp(n_samples);
    neighbors_exp[0].append(std::vector<da_int>{3, 4, 6, 7});
    neighbors_exp[1].append(std::vector<da_int>{2, 8, 9});
    neighbors_exp[2].append(std::vector<da_int>{1, 8, 9});
    neighbors_exp[3].append(std::vector<da_int>{0, 4, 6, 7});
    neighbors_exp[4].append(std::vector<da_int>{0, 3, 6, 7});
    neighbors_exp[6].append(std::vector<da_int>{0, 3, 4, 7});
    neighbors_exp[7].append(std::vector<da_int>{0, 3, 4, 6});
    neighbors_exp[8].append(std::vector<da_int>{1, 2, 9});
    neighbors_exp[9].append(std::vector<da_int>{1, 2, 8});

    da_errors::da_error_t *err =
        new da_errors::da_error_t(da_errors::action_t::DA_RECORD);

    EXPECT_EQ(da_radius_neighbors::radius_neighbors(n_samples, n_features, A.data(), lda,
                                                    eps, neighbors, err),
              da_status_success);

    // In-place sort to allow for different ordering of stored indices in neighbors
    for (da_int i = 0; i < n_samples; i++) {
        std::sort(neighbors[i].data(), neighbors[i].data() + neighbors[i].size());
        EXPECT_EQ(neighbors[i].size(), neighbors_exp[i].size());
        for (da_int j = 0; j < (da_int)neighbors[i].size(); j++) {
            EXPECT_EQ((neighbors[i])[j], (neighbors_exp[i])[j]);
        }
    }

    delete err;
}

TYPED_TEST(DBSCANTest, radius_neighbors_large) {

    da_int n_samples = 800;
    da_int n_features = 1;
    da_int lda = 800;
    TypeParam eps = 1.1;

    std::vector<TypeParam> A(n_samples);
    std::iota(A.begin(), A.end(), 0);

    std::vector<da_vector::da_vector<da_int>> neighbors(n_samples);

    std::vector<da_vector::da_vector<da_int>> neighbors_exp(n_samples);
    for (da_int i = 1; i < n_samples - 1; i++) {
        neighbors_exp[i].append(std::vector<da_int>{i - 1, i + 1});
    }
    neighbors_exp[0].push_back(1);
    neighbors_exp[n_samples - 1].push_back(n_samples - 2);

    da_errors::da_error_t *err =
        new da_errors::da_error_t(da_errors::action_t::DA_RECORD);

    EXPECT_EQ(da_radius_neighbors::radius_neighbors(n_samples, n_features, A.data(), lda,
                                                    eps, neighbors, err),
              da_status_success);

    // In-place sort to allow for different ordering of stored indices in neighbors
    for (da_int i = 0; i < n_samples; i++) {
        std::sort(neighbors[i].data(), neighbors[i].data() + neighbors[i].size());
        EXPECT_EQ(neighbors[i].size(), neighbors_exp[i].size());
        for (da_int j = 0; j < (da_int)neighbors[i].size(); j++) {
            EXPECT_EQ((neighbors[i])[j], (neighbors_exp[i])[j]);
        }
    }

    delete err;
}