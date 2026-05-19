/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../utest_utils.hpp"
#include "../persistence_test_utils.hpp"
#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

// Test parameters for radius neighbors serialization tests
struct radius_nn_serial_params {
    std::string test_name;
    std::string algorithm;
    std::string metric;
    std::string weights;
    da_int leaf_size;
    double minkowski_p;
    double radius;
    std::string order;
    da_int ldx_padding;
};

void PrintTo(const radius_nn_serial_params &param, ::std::ostream *os) {
    *os << param.test_name;
}

// Test configurations covering different codepaths
// Defaults: algorithm="auto", metric="euclidean", weights="uniform", leaf_size=30, minkowski_p=2.0
const std::vector<radius_nn_serial_params> radius_nn_serialization_params = {
    {"brute_manhattan_distance", "brute", "manhattan", "distance", 30, 2.0, 5.0,
     "column-major", 2},
    {"brute_minkowski_uniform", "brute", "minkowski", "uniform", 30, 3.0, 4.0,
     "row-major", 1},
    {"kdtree_euclidean_leaf20", "kd tree", "euclidean", "uniform", 20, 2.0, 3.2,
     "column-major", 0},
    {"balltree_euclidean_distance", "ball tree", "euclidean", "distance", 25, 2.0, 3.2,
     "row-major", 0},
    {"balltree_minkowski_leaf15", "ball tree", "minkowski", "uniform", 15, 1.5, 4.5,
     "column-major", 3},
};

class RadiusNNSerializationTest : public testing::TestWithParam<radius_nn_serial_params> {
  protected:
    // Problem dimensions
    da_int n_samples = 6;
    da_int n_features = 3;
    da_int n_queries = 3;

    std::vector<double> X_train;
    std::vector<double> X_test;
    da_int ldx_train = 0;
    da_int ldx_test = 0;

    std::string model_file;

    void SetUp() override {
        const radius_nn_serial_params &pr = GetParam();
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_case = test_info->name();
        std::replace(test_case.begin(), test_case.end(), '/', '_');
        model_file = model_persistence_test_utils::get_test_file_dir() + "/radius_nn_" +
                     pr.test_name + "_" + test_case + ".bin";

        bool is_colmajor = (pr.order == "column-major");

        // Training data (column-major order)
        std::vector<double> X_train_colmajor = {-1.0, -2.0, -3.0, 1.0, 2.0, 3.0,
                                                -1.0, -1.0, -2.0, 3.0, 5.0, -1.0,
                                                2.0,  3.0,  -1.0, 1.0, 1.0, 2.0};

        // Test data (column-major order)
        std::vector<double> X_test_colmajor = {-2.0, -1.0, 2.0,  2.0, -2.0,
                                               1.0,  3.0,  -1.0, -3.0};

        // Sentinel value helps detect if padding is accidentally used
        const double sentinel = -999.0;

        if (is_colmajor) {
            // Column-major: ldx >= n_samples (rows)
            ldx_train = n_samples + pr.ldx_padding;
            ldx_test = n_queries + pr.ldx_padding;

            X_train.resize(ldx_train * n_features, sentinel);
            for (da_int col = 0; col < n_features; col++) {
                for (da_int row = 0; row < n_samples; row++) {
                    X_train[row + col * ldx_train] =
                        X_train_colmajor[row + col * n_samples];
                }
            }

            X_test.resize(ldx_test * n_features, sentinel);
            for (da_int col = 0; col < n_features; col++) {
                for (da_int row = 0; row < n_queries; row++) {
                    X_test[row + col * ldx_test] = X_test_colmajor[row + col * n_queries];
                }
            }
        } else {
            // Row-major: ldx >= n_features (cols)
            ldx_train = n_features + pr.ldx_padding;
            ldx_test = n_features + pr.ldx_padding;

            // Source is column-major, transpose while copying to row-major with padding
            X_train.resize(n_samples * ldx_train, sentinel);
            for (da_int row = 0; row < n_samples; row++) {
                for (da_int col = 0; col < n_features; col++) {
                    X_train[col + row * ldx_train] =
                        X_train_colmajor[row + col * n_samples];
                }
            }

            X_test.resize(n_queries * ldx_test, sentinel);
            for (da_int row = 0; row < n_queries; row++) {
                for (da_int col = 0; col < n_features; col++) {
                    X_test[col + row * ldx_test] = X_test_colmajor[row + col * n_queries];
                }
            }
        }
    }

    void TearDown() override { std::remove(model_file.c_str()); }
};

template <typename T>
void radius_nn_serialization_test(const radius_nn_serial_params &pr, da_int n_samples,
                                  da_int n_features, da_int n_queries,
                                  const std::vector<double> &X_train_d, da_int ldx_train,
                                  const std::vector<double> &X_test_d, da_int ldx_test,
                                  const std::string &model_file) {

    // Convert data to correct type
    std::vector<T> X_train(X_train_d.begin(), X_train_d.end());
    std::vector<T> X_test(X_test_d.begin(), X_test_d.end());
    T radius = static_cast<T>(pr.radius);

    // Result arrays for original model
    std::vector<da_int> count_orig(n_queries + 1);
    std::vector<da_int> offsets_orig(n_queries + 1);
    da_int total_neighbors_orig = 0;
    std::vector<da_int> indices_orig;
    std::vector<T> distances_orig;

    // ==================== ORIGINAL MODEL BLOCK ====================
    {
        da_handle handle_orig = nullptr;
        EXPECT_EQ(da_handle_init<T>(&handle_orig, da_handle_nn), da_status_success);

        EXPECT_EQ(da_options_set_string(handle_orig, "algorithm", pr.algorithm.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle_orig, "metric", pr.metric.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle_orig, "storage order", pr.order.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle_orig, "weights", pr.weights.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle_orig, "leaf size", pr.leaf_size),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle_orig, "minkowski parameter", (T)pr.minkowski_p),
                  da_status_success);

        EXPECT_EQ(
            da_nn_set_data(handle_orig, n_samples, n_features, X_train.data(), ldx_train),
            da_status_success);

        // Compute radius neighbors
        da_int return_distance = 1;
        da_int sort_results = 1;
        EXPECT_EQ(da_nn_radius_neighbors(handle_orig, n_queries, n_features,
                                         X_test.data(), ldx_test, radius, return_distance,
                                         sort_results),
                  da_status_success);

        // Get counts
        da_int array_size = n_queries + 1;
        EXPECT_EQ(da_handle_get_result_int(handle_orig, da_nn_radius_neighbors_count,
                                           &array_size, count_orig.data()),
                  da_status_success);

        // Get offsets
        EXPECT_EQ(da_handle_get_result_int(handle_orig, da_nn_radius_neighbors_offsets,
                                           &array_size, offsets_orig.data()),
                  da_status_success);

        total_neighbors_orig = count_orig[n_queries];

        // Get indices and distances from original handle for comparison
        indices_orig.resize(total_neighbors_orig);
        distances_orig.resize(total_neighbors_orig);

        if (total_neighbors_orig > 0) {
            da_int size = total_neighbors_orig;
            EXPECT_EQ(da_handle_get_result_int(handle_orig,
                                               da_nn_radius_neighbors_indices, &size,
                                               indices_orig.data()),
                      da_status_success);
            EXPECT_EQ(da_handle_get_result(handle_orig, da_nn_radius_neighbors_distances,
                                           &size, distances_orig.data()),
                      da_status_success);
        }

        // Save the model
        EXPECT_EQ(da_handle_save_model(handle_orig, model_file.c_str()),
                  da_status_success);

        // Destroy the original handle
        da_handle_destroy(&handle_orig);
    }

    // ==================== LOADED MODEL BLOCK ====================
    {
        da_handle handle_loaded = nullptr;
        EXPECT_EQ(da_handle_load_model(&handle_loaded, model_file.c_str()),
                  da_status_success);

        // Compute radius neighbors from loaded model
        da_int return_distance = 1;
        da_int sort_results = 1;
        EXPECT_EQ(da_nn_radius_neighbors(handle_loaded, n_queries, n_features,
                                         X_test.data(), ldx_test, radius, return_distance,
                                         sort_results),
                  da_status_success);

        // Result arrays for loaded model
        std::vector<da_int> count_loaded(n_queries + 1);
        std::vector<da_int> offsets_loaded(n_queries + 1);

        // Get counts from loaded
        da_int array_size = n_queries + 1;
        EXPECT_EQ(da_handle_get_result_int(handle_loaded, da_nn_radius_neighbors_count,
                                           &array_size, count_loaded.data()),
                  da_status_success);

        // Get offsets from loaded
        EXPECT_EQ(da_handle_get_result_int(handle_loaded, da_nn_radius_neighbors_offsets,
                                           &array_size, offsets_loaded.data()),
                  da_status_success);

        da_int total_neighbors_loaded = count_loaded[n_queries];

        // ==================== COMPARE RESULTS ====================
        // Compare counts
        EXPECT_EQ(total_neighbors_orig, total_neighbors_loaded);
        EXPECT_ARR_EQ(n_queries + 1, count_orig.data(), count_loaded.data(), 1, 1, 0, 0);
        // Compare offsets
        EXPECT_ARR_EQ(n_queries + 1, offsets_orig.data(), offsets_loaded.data(), 1, 1, 0,
                      0);

        // If there are neighbors, compare indices and distances
        if (total_neighbors_orig > 0) {
            std::vector<da_int> indices_loaded(total_neighbors_loaded);
            std::vector<T> distances_loaded(total_neighbors_loaded);

            da_int size = total_neighbors_loaded;
            EXPECT_EQ(da_handle_get_result_int(handle_loaded,
                                               da_nn_radius_neighbors_indices, &size,
                                               indices_loaded.data()),
                      da_status_success);
            EXPECT_EQ(da_handle_get_result(handle_loaded,
                                           da_nn_radius_neighbors_distances, &size,
                                           distances_loaded.data()),
                      da_status_success);

            // Compare indices and distances between original and loaded models
            EXPECT_ARR_EQ(total_neighbors_orig, indices_orig.data(),
                          indices_loaded.data(), 1, 1, 0, 0);
            EXPECT_ARR_EQ(total_neighbors_orig, distances_orig.data(),
                          distances_loaded.data(), 1, 1, 0, 0);

            // Verify stability - run again and compare with loaded results
            EXPECT_EQ(da_nn_radius_neighbors(handle_loaded, n_queries, n_features,
                                             X_test.data(), ldx_test, radius,
                                             return_distance, sort_results),
                      da_status_success);

            std::vector<da_int> count_again(n_queries + 1);
            da_int array_size = n_queries + 1;
            EXPECT_EQ(da_handle_get_result_int(handle_loaded,
                                               da_nn_radius_neighbors_count, &array_size,
                                               count_again.data()),
                      da_status_success);
            EXPECT_ARR_EQ(n_queries + 1, count_loaded.data(), count_again.data(), 1, 1, 0,
                          0);
        }

        // ==================== VERIFY OPTIONS ====================
        // Verify that options are preserved after serialization
        char algo_loaded[64];
        da_int algo_len = 64;
        EXPECT_EQ(
            da_options_get_string(handle_loaded, "algorithm", algo_loaded, &algo_len),
            da_status_success);
        EXPECT_STREQ(algo_loaded, pr.algorithm.c_str());

        char metric_loaded[64];
        da_int metric_len = 64;
        EXPECT_EQ(
            da_options_get_string(handle_loaded, "metric", metric_loaded, &metric_len),
            da_status_success);
        EXPECT_STREQ(metric_loaded, pr.metric.c_str());

        char weights_loaded[64];
        da_int weights_len = 64;
        EXPECT_EQ(
            da_options_get_string(handle_loaded, "weights", weights_loaded, &weights_len),
            da_status_success);
        EXPECT_STREQ(weights_loaded, pr.weights.c_str());

        char order_loaded[64];
        da_int order_len = 64;
        EXPECT_EQ(da_options_get_string(handle_loaded, "storage order", order_loaded,
                                        &order_len),
                  da_status_success);
        EXPECT_STREQ(order_loaded, pr.order.c_str());

        da_int leaf_size_loaded = 0;
        EXPECT_EQ(da_options_get_int(handle_loaded, "leaf size", &leaf_size_loaded),
                  da_status_success);
        EXPECT_EQ(leaf_size_loaded, pr.leaf_size);

        T minkowski_p_loaded = 0;
        EXPECT_EQ(
            da_options_get(handle_loaded, "minkowski parameter", &minkowski_p_loaded),
            da_status_success);
        EXPECT_EQ(minkowski_p_loaded, (T)pr.minkowski_p);

        da_handle_destroy(&handle_loaded);
    }
}

TEST_P(RadiusNNSerializationTest, double) {
    const radius_nn_serial_params &pr = GetParam();
    radius_nn_serialization_test<double>(pr, n_samples, n_features, n_queries, X_train,
                                         ldx_train, X_test, ldx_test, model_file);
}

TEST_P(RadiusNNSerializationTest, float) {
    const radius_nn_serial_params &pr = GetParam();
    radius_nn_serialization_test<float>(pr, n_samples, n_features, n_queries, X_train,
                                        ldx_train, X_test, ldx_test, model_file);
}

INSTANTIATE_TEST_SUITE_P(RadiusNNSerializationSuite, RadiusNNSerializationTest,
                         testing::ValuesIn(radius_nn_serialization_params));

// ==================== ERROR HANDLING TESTS ====================

class RadiusNNSerializationErrorTest : public testing::Test {
  protected:
    std::string model_file =
        model_persistence_test_utils::get_test_file_dir() + "/radius_nn_error_test.bin";
    void TearDown() override { std::remove(model_file.c_str()); }
};

TEST_F(RadiusNNSerializationErrorTest, SaveBeforeSetDataFails) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_nn), da_status_success);
    EXPECT_EQ(da_handle_save_model(handle, model_file.c_str()), da_status_no_data);
    da_handle_destroy(&handle);
}