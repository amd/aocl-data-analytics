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
#include "binary_tree.hpp"
#include "da_error.hpp"
#include "da_vector.hpp"
#include "radius_neighbors.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

template <typename T> class RNTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(RNTest, FloatTypes);

TYPED_TEST(RNTest, radius_neighbors_small) {

    da_int n_samples = 10;
    da_int n_features = 2;
    da_int lda = 10;
    TypeParam eps = 1.5;
    std::vector<double> A_double = {0.0,  -5.0, -6.0, 0.1,  0.1,  10.0, -0.1,
                                    -0.1, -5.5, -5.0, 0.0,  -5.0, -6.0, 0.1,
                                    -0.1, 10.0, 0.1,  -0.1, -5.5, -6.0};

    std::vector<TypeParam> A = convert_vector<double, TypeParam>(A_double);

    std::vector<da_vector::da_vector<da_int>> neighbors_brute(n_samples);
    std::vector<da_vector::da_vector<da_int>> neighbors_kd_tree(n_samples);
    std::vector<da_vector::da_vector<da_int>> neighbors_ball_tree(n_samples);

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

    EXPECT_EQ(TEST_ARCH::da_radius_neighbors::radius_neighbors_brute(
                  n_samples, n_features, A.data(), lda, eps, da_euclidean, (TypeParam)0.0,
                  neighbors_brute, err),
              da_status_success);

    // In-place sort to allow for different ordering of stored indices in neighbors
    for (da_int i = 0; i < n_samples; i++) {
        std::sort(neighbors_brute[i].data(),
                  neighbors_brute[i].data() + neighbors_brute[i].size());
        EXPECT_EQ(neighbors_brute[i].size(), neighbors_exp[i].size());
        for (da_int j = 0; j < (da_int)neighbors_brute[i].size(); j++) {
            EXPECT_EQ((neighbors_brute[i])[j], (neighbors_exp[i])[j]);
        }
    }

    da_int leaf_size = 2;
    auto tree = TEST_ARCH::da_binary_tree::kd_tree<TypeParam>(
        n_samples, n_features, A.data(), lda, leaf_size, da_euclidean, (TypeParam)2.0);

    EXPECT_EQ(tree.radius_neighbors(n_samples, n_features, nullptr, 0, eps,
                                    neighbors_kd_tree, err),
              da_status_success);

    for (da_int i = 0; i < n_samples; i++) {
        std::sort(neighbors_kd_tree[i].data(),
                  neighbors_kd_tree[i].data() + neighbors_kd_tree[i].size());
        EXPECT_EQ(neighbors_kd_tree[i].size(), neighbors_exp[i].size());
        for (da_int j = 0; j < (da_int)neighbors_kd_tree[i].size(); j++) {
            EXPECT_EQ((neighbors_kd_tree[i])[j], (neighbors_exp[i])[j]);
        }
    }

    auto tree2 = TEST_ARCH::da_binary_tree::ball_tree<TypeParam>(
        n_samples, n_features, A.data(), lda, leaf_size, da_euclidean, (TypeParam)2.0);

    EXPECT_EQ(tree2.radius_neighbors(n_samples, n_features, nullptr, 0, eps,
                                     neighbors_ball_tree, err),
              da_status_success);

    for (da_int i = 0; i < n_samples; i++) {
        std::sort(neighbors_ball_tree[i].data(),
                  neighbors_ball_tree[i].data() + neighbors_ball_tree[i].size());
        EXPECT_EQ(neighbors_ball_tree[i].size(), neighbors_exp[i].size());
        for (da_int j = 0; j < (da_int)neighbors_ball_tree[i].size(); j++) {
            EXPECT_EQ((neighbors_ball_tree[i])[j], (neighbors_exp[i])[j]);
        }
    }

    delete err;
}

TYPED_TEST(RNTest, radius_neighbors_small_cosine) {

    // Compute radius neighbors with cosine distance
    // Other distance metrics are tested in the main DBSCAN tests; this is just to make sure we can go down the correct code path and get the expected outputs

    da_int n_samples = 5;
    da_int n_features = 2;
    da_int lda = 5;
    TypeParam eps = 0.5;
    std::vector<double> A_double = {0.1, -0.3, 2.1, 0.3, 0.5, 0.1, 0.2, 0.3, 0.3, 0.6};

    std::vector<TypeParam> A = convert_vector<double, TypeParam>(A_double);

    std::vector<da_vector::da_vector<da_int>> neighbors_brute(n_samples);

    std::vector<da_vector::da_vector<da_int>> neighbors_exp(n_samples);
    neighbors_exp[0].append(std::vector<da_int>{2, 3, 4});
    // neighbors_exp[1] is left empty
    neighbors_exp[2].append(std::vector<da_int>{0, 3, 4});
    neighbors_exp[3].append(std::vector<da_int>{0, 2, 4});
    neighbors_exp[4].append(std::vector<da_int>{0, 2, 3});

    da_errors::da_error_t *err =
        new da_errors::da_error_t(da_errors::action_t::DA_RECORD);

    EXPECT_EQ(TEST_ARCH::da_radius_neighbors::radius_neighbors_brute(
                  n_samples, n_features, A.data(), lda, eps, da_cosine, (TypeParam)0.0,
                  neighbors_brute, err),
              da_status_success);

    // In-place sort to allow for different ordering of stored indices in neighbors
    for (da_int i = 0; i < n_samples; i++) {
        std::sort(neighbors_brute[i].data(),
                  neighbors_brute[i].data() + neighbors_brute[i].size());
        EXPECT_EQ(neighbors_brute[i].size(), neighbors_exp[i].size());
        for (da_int j = 0; j < (da_int)neighbors_brute[i].size(); j++) {
            EXPECT_EQ((neighbors_brute[i])[j], (neighbors_exp[i])[j]);
        }
    }

    delete err;
}

TYPED_TEST(RNTest, radius_neighbors_large) {

    da_int n_samples = 800;
    da_int n_features = 1;
    da_int lda = n_samples;
    TypeParam eps = 1.1;

    std::vector<TypeParam> A(n_samples);
    std::iota(A.begin(), A.end(), 0);

    std::vector<da_vector::da_vector<da_int>> neighbors_brute(n_samples);
    std::vector<da_vector::da_vector<da_int>> neighbors_kd_tree_euc(n_samples);
    std::vector<da_vector::da_vector<da_int>> neighbors_kd_tree_mink(n_samples);
    std::vector<da_vector::da_vector<da_int>> neighbors_kd_tree_man(n_samples);
    std::vector<da_vector::da_vector<da_int>> neighbors_ball_tree_euc(n_samples);
    std::vector<da_vector::da_vector<da_int>> neighbors_ball_tree_mink(n_samples);
    std::vector<da_vector::da_vector<da_int>> neighbors_ball_tree_man(n_samples);

    std::vector<da_vector::da_vector<da_int>> neighbors_exp(n_samples);
    for (da_int i = 1; i < n_samples - 1; i++) {
        neighbors_exp[i].append(std::vector<da_int>{i - 1, i + 1});
    }
    neighbors_exp[0].push_back(1);
    neighbors_exp[n_samples - 1].push_back(n_samples - 2);

    da_errors::da_error_t *err =
        new da_errors::da_error_t(da_errors::action_t::DA_RECORD);

    EXPECT_EQ(TEST_ARCH::da_radius_neighbors::radius_neighbors_brute(
                  n_samples, n_features, A.data(), lda, eps, da_euclidean, (TypeParam)0.0,
                  neighbors_brute, err),
              da_status_success);

    // In-place sort to allow for different ordering of stored indices in neighbors
    for (da_int i = 0; i < n_samples; i++) {
        std::sort(neighbors_brute[i].data(),
                  neighbors_brute[i].data() + neighbors_brute[i].size());
        EXPECT_EQ(neighbors_brute[i].size(), neighbors_exp[i].size());
        for (da_int j = 0; j < (da_int)neighbors_brute[i].size(); j++) {
            EXPECT_EQ((neighbors_brute[i])[j], (neighbors_exp[i])[j]);
        }
    }

    da_int leaf_size = 5;
    auto tree = TEST_ARCH::da_binary_tree::kd_tree<TypeParam>(
        n_samples, n_features, A.data(), lda, leaf_size, da_euclidean, (TypeParam)0.0);

    EXPECT_EQ(tree.radius_neighbors(n_samples, n_features, nullptr, 0, eps,
                                    neighbors_kd_tree_euc, err),
              da_status_success);

    for (da_int i = 0; i < n_samples; i++) {
        std::sort(neighbors_kd_tree_euc[i].data(),
                  neighbors_kd_tree_euc[i].data() + neighbors_kd_tree_euc[i].size());
        EXPECT_EQ(neighbors_kd_tree_euc[i].size(), neighbors_exp[i].size());
        for (da_int j = 0; j < (da_int)neighbors_kd_tree_euc[i].size(); j++) {
            EXPECT_EQ((neighbors_kd_tree_euc[i])[j], (neighbors_exp[i])[j]);
        }
    }

    auto tree2 = TEST_ARCH::da_binary_tree::kd_tree<TypeParam>(
        n_samples, n_features, A.data(), lda, leaf_size, da_minkowski,
        (TypeParam)2.00001);

    EXPECT_EQ(tree2.radius_neighbors(n_samples, n_features, nullptr, 0, eps,
                                     neighbors_kd_tree_mink, err),
              da_status_success);

    for (da_int i = 0; i < n_samples; i++) {
        std::sort(neighbors_kd_tree_mink[i].data(),
                  neighbors_kd_tree_mink[i].data() + neighbors_kd_tree_mink[i].size());
        EXPECT_EQ(neighbors_kd_tree_mink[i].size(), neighbors_exp[i].size());
        for (da_int j = 0; j < (da_int)neighbors_kd_tree_mink[i].size(); j++) {
            EXPECT_EQ((neighbors_kd_tree_mink[i])[j], (neighbors_exp[i])[j]);
        }
    }

    auto tree3 = TEST_ARCH::da_binary_tree::kd_tree<TypeParam>(
        n_samples, n_features, A.data(), lda, leaf_size, da_manhattan, (TypeParam)0.0);

    EXPECT_EQ(tree3.radius_neighbors(n_samples, n_features, nullptr, 0, eps,
                                     neighbors_kd_tree_man, err),
              da_status_success);

    for (da_int i = 0; i < n_samples; i++) {
        std::sort(neighbors_kd_tree_man[i].data(),
                  neighbors_kd_tree_man[i].data() + neighbors_kd_tree_man[i].size());
        EXPECT_EQ(neighbors_kd_tree_man[i].size(), neighbors_exp[i].size());
        for (da_int j = 0; j < (da_int)neighbors_kd_tree_man[i].size(); j++) {
            EXPECT_EQ((neighbors_kd_tree_man[i])[j], (neighbors_exp[i])[j]);
        }
    }

    auto tree4 = TEST_ARCH::da_binary_tree::ball_tree<TypeParam>(
        n_samples, n_features, A.data(), lda, leaf_size, da_euclidean, (TypeParam)0.0);

    EXPECT_EQ(tree4.radius_neighbors(n_samples, n_features, nullptr, 0, eps,
                                     neighbors_ball_tree_euc, err),
              da_status_success);

    for (da_int i = 0; i < n_samples; i++) {
        std::sort(neighbors_ball_tree_euc[i].data(),
                  neighbors_ball_tree_euc[i].data() + neighbors_ball_tree_euc[i].size());
        EXPECT_EQ(neighbors_ball_tree_euc[i].size(), neighbors_exp[i].size());
        for (da_int j = 0; j < (da_int)neighbors_ball_tree_euc[i].size(); j++) {
            EXPECT_EQ((neighbors_ball_tree_euc[i])[j], (neighbors_exp[i])[j]);
        }
    }

    auto tree5 = TEST_ARCH::da_binary_tree::ball_tree<TypeParam>(
        n_samples, n_features, A.data(), lda, leaf_size, da_minkowski, (TypeParam)2.0001);

    EXPECT_EQ(tree5.radius_neighbors(n_samples, n_features, nullptr, 0, eps,
                                     neighbors_ball_tree_mink, err),
              da_status_success);

    for (da_int i = 0; i < n_samples; i++) {
        std::sort(neighbors_ball_tree_mink[i].data(),
                  neighbors_ball_tree_mink[i].data() +
                      neighbors_ball_tree_mink[i].size());
        EXPECT_EQ(neighbors_ball_tree_mink[i].size(), neighbors_exp[i].size());
        for (da_int j = 0; j < (da_int)neighbors_ball_tree_mink[i].size(); j++) {
            EXPECT_EQ((neighbors_ball_tree_mink[i])[j], (neighbors_exp[i])[j]);
        }
    }

    auto tree6 = TEST_ARCH::da_binary_tree::ball_tree<TypeParam>(
        n_samples, n_features, A.data(), lda, leaf_size, da_manhattan, (TypeParam)0.0);

    EXPECT_EQ(tree6.radius_neighbors(n_samples, n_features, nullptr, 0, eps,
                                     neighbors_ball_tree_man, err),
              da_status_success);

    for (da_int i = 0; i < n_samples; i++) {
        std::sort(neighbors_ball_tree_man[i].data(),
                  neighbors_ball_tree_man[i].data() + neighbors_ball_tree_man[i].size());
        EXPECT_EQ(neighbors_ball_tree_man[i].size(), neighbors_exp[i].size());
        for (da_int j = 0; j < (da_int)neighbors_ball_tree_man[i].size(); j++) {
            EXPECT_EQ((neighbors_ball_tree_man[i])[j], (neighbors_exp[i])[j]);
        }
    }

    delete err;
}

TYPED_TEST(RNTest, radius_neighbors_large_eps) {

    // Extra test for code coverage of ball tree code path with large eps
    da_int n_samples = 80;
    da_int n_features = 1;
    da_int lda = n_samples;
    TypeParam eps = 200;

    std::vector<TypeParam> A(n_samples);
    std::iota(A.begin(), A.end(), 0);

    std::vector<da_vector::da_vector<da_int>> neighbors(n_samples);

    std::vector<da_vector::da_vector<da_int>> neighbors_exp(n_samples);
    for (da_int i = 0; i < n_samples; i++) {
        for (da_int j = 0; j < n_samples; j++) {
            if (i != j) {
                neighbors_exp[i].push_back(j);
            }
        }
    }

    da_errors::da_error_t *err =
        new da_errors::da_error_t(da_errors::action_t::DA_RECORD);

    da_int leaf_size = 5;

    auto tree = TEST_ARCH::da_binary_tree::ball_tree<TypeParam>(
        n_samples, n_features, A.data(), lda, leaf_size, da_euclidean, (TypeParam)0.0);

    EXPECT_EQ(
        tree.radius_neighbors(n_samples, n_features, nullptr, 0, eps, neighbors, err),
        da_status_success);

    for (da_int i = 0; i < n_samples; i++) {
        std::sort(neighbors[i].data(), neighbors[i].data() + neighbors[i].size());
        EXPECT_EQ(neighbors[i].size(), neighbors_exp[i].size());
        for (da_int j = 0; j < (da_int)neighbors[i].size(); j++) {
            EXPECT_EQ((neighbors[i])[j], (neighbors_exp[i])[j]);
        }
    }

    delete err;
}

TYPED_TEST(RNTest, radius_neighbors_tiny_eps) {

    // Extra test for code coverage of ball tree code path with tiny eps
    da_int n_samples = 80;
    da_int n_features = 1;
    da_int lda = n_samples;
    TypeParam eps = 0.02;

    std::vector<TypeParam> A(n_samples);
    std::iota(A.begin(), A.end(), 0);

    std::vector<da_vector::da_vector<da_int>> neighbors(n_samples);

    da_errors::da_error_t *err =
        new da_errors::da_error_t(da_errors::action_t::DA_RECORD);

    da_int leaf_size = 5;

    auto tree = TEST_ARCH::da_binary_tree::ball_tree<TypeParam>(
        n_samples, n_features, A.data(), lda, leaf_size, da_euclidean, (TypeParam)0.0);

    EXPECT_EQ(
        tree.radius_neighbors(n_samples, n_features, nullptr, 0, eps, neighbors, err),
        da_status_success);

    for (da_int i = 0; i < n_samples; i++) {
        EXPECT_EQ(neighbors[i].size(), 0);
    }

    delete err;
}