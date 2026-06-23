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
#include <string>
#include <vector>

// Test parameters for kNN classification serialization tests
// Defaults: algorithm="auto", metric="euclidean", weights="uniform", n_neighbors=5, leaf_size=30, minkowski_p=2.0
struct knn_class_serial_params {
    std::string test_name;
    std::string algorithm;
    std::string metric;
    std::string weights;
    da_int n_neighbors;
    da_int leaf_size;
    double minkowski_p;
    da_nn_search_mode mode;
    double radius;
    std::string order;
    da_int ldx_padding;
};

void PrintTo(const knn_class_serial_params &param, ::std::ostream *os) {
    *os << param.test_name;
}

// Test configurations covering different codepaths
// First test uses non-default values for all options
const std::vector<knn_class_serial_params> knn_class_serialization_params = {
    // k-NN search mode tests
    {"brute_manhattan_distance", "brute", "manhattan", "distance", 4, 30, 2.0,
     knn_search_mode, 0.0, "column-major", 2},
    {"brute_euclidean_uniform", "brute", "euclidean", "uniform", 3, 30, 2.0,
     knn_search_mode, 0.0, "row-major", 1},
    {"kdtree_euclidean_uniform", "kd tree", "euclidean", "uniform", 3, 40, 2.0,
     knn_search_mode, 0.0, "column-major", 0},
    {"kdtree_minkowski_uniform", "kd tree", "minkowski", "uniform", 3, 30, 3.0,
     knn_search_mode, 0.0, "row-major", 0},
    {"balltree_euclidean_uniform", "ball tree", "euclidean", "uniform", 3, 50, 2.0,
     knn_search_mode, 0.0, "column-major", 1},
    {"balltree_euclidean_distance", "ball tree", "euclidean", "distance", 3, 30, 2.0,
     knn_search_mode, 0.0, "row-major", 2},
    // Radius search mode tests (use larger radius to ensure all queries find neighbors)
    {"radius_brute_euclidean_uniform", "brute", "euclidean", "uniform", 0, 30, 2.0,
     radius_search_mode, 8.0, "column-major", 3},
    {"radius_kdtree_manhattan_distance", "kd tree", "manhattan", "distance", 0, 35, 2.0,
     radius_search_mode, 12.0, "row-major", 1},
    {"radius_balltree_minkowski_uniform", "ball tree", "minkowski", "uniform", 0, 25, 1.5,
     radius_search_mode, 10.0, "column-major", 0},
};

class KNNClassSerializationTest : public testing::TestWithParam<knn_class_serial_params> {
  protected:
    std::string model_file;

    void TearDown() override { std::remove(model_file.c_str()); }

    // Problem dimensions
    da_int n_samples = 6;
    da_int n_features = 3;
    da_int n_queries = 3;

    std::vector<double> X_train;
    std::vector<da_int> y_train;
    std::vector<double> X_test;
    da_int ldx_train = 0;
    da_int ldx_test = 0;

    void SetUp() override {
        const knn_class_serial_params &pr = GetParam();
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();

        std::string test_case = test_info->name();
        std::replace(test_case.begin(), test_case.end(), '/', '_');
        model_file = model_persistence_test_utils::get_test_file_dir() + "/knn_class_" +
                     pr.test_name + "_" + test_case + ".bin";

        bool is_colmajor = (pr.order == "column-major");

        // Training data (column-major order)
        std::vector<double> X_train_colmajor = {-1.0, -2.0, -3.0, 1.0, 2.0, 3.0,
                                                -1.0, -1.0, -2.0, 3.0, 5.0, -1.0,
                                                2.0,  3.0,  -1.0, 1.0, 1.0, 2.0};
        y_train = {1, 2, 0, 1, 2, 2};

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
};

template <typename T>
void knn_class_serialization_test(const knn_class_serial_params &pr, da_int n_samples,
                                  da_int n_features, da_int n_queries,
                                  const std::vector<double> &X_train_d,
                                  const std::vector<da_int> &y_train, da_int ldx_train,
                                  const std::vector<double> &X_test_d, da_int ldx_test,
                                  const std::string &model_file) {

    // Convert data to correct type
    std::vector<T> X_train(X_train_d.begin(), X_train_d.end());
    std::vector<T> X_test(X_test_d.begin(), X_test_d.end());

    // Result arrays for original model (only allocate for k-NN mode)
    da_int k_size = (pr.mode == knn_search_mode) ? pr.n_neighbors : 1;
    std::vector<da_int> k_ind_orig(k_size * n_queries);
    std::vector<T> k_dist_orig(k_size * n_queries);
    std::vector<da_int> y_pred_orig(n_queries);
    da_int n_classes_orig = 0;

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
        if (pr.mode == knn_search_mode) {
            EXPECT_EQ(
                da_options_set_int(handle_orig, "number of neighbors", pr.n_neighbors),
                da_status_success);
        } else {
            EXPECT_EQ(da_options_set(handle_orig, "radius", (T)pr.radius),
                      da_status_success);
        }
        EXPECT_EQ(da_options_set_int(handle_orig, "leaf size", pr.leaf_size),
                  da_status_success);
        EXPECT_EQ(da_options_set(handle_orig, "minkowski parameter", (T)pr.minkowski_p),
                  da_status_success);

        EXPECT_EQ(
            da_nn_set_data(handle_orig, n_samples, n_features, X_train.data(), ldx_train),
            da_status_success);
        EXPECT_EQ(da_nn_set_labels<T>(handle_orig, n_samples, y_train.data()),
                  da_status_success);

        // Compute k-neighbors (only for k-NN mode)
        if (pr.mode == knn_search_mode) {
            EXPECT_EQ(da_nn_kneighbors(handle_orig, n_queries, n_features, X_test.data(),
                                       ldx_test, k_ind_orig.data(), k_dist_orig.data(),
                                       pr.n_neighbors, 1),
                      da_status_success);
        }

        // Get number of classes
        EXPECT_EQ(da_nn_classes<T>(handle_orig, &n_classes_orig, nullptr),
                  da_status_success);

        // Get predictions
        EXPECT_EQ(da_nn_classifier_predict(handle_orig, n_queries, n_features,
                                           X_test.data(), ldx_test, y_pred_orig.data(),
                                           pr.mode),
                  da_status_success);

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

        // Result arrays for loaded model
        std::vector<da_int> k_ind_loaded(k_size * n_queries);
        std::vector<T> k_dist_loaded(k_size * n_queries);
        std::vector<da_int> y_pred_loaded(n_queries);
        da_int n_classes_loaded = 0;

        // Compute k-neighbors from loaded model (only for k-NN mode)
        if (pr.mode == knn_search_mode) {
            EXPECT_EQ(da_nn_kneighbors(handle_loaded, n_queries, n_features,
                                       X_test.data(), ldx_test, k_ind_loaded.data(),
                                       k_dist_loaded.data(), pr.n_neighbors, 1),
                      da_status_success);
        }

        // Get number of classes from loaded model
        EXPECT_EQ(da_nn_classes<T>(handle_loaded, &n_classes_loaded, nullptr),
                  da_status_success);

        // Get predictions from loaded model
        EXPECT_EQ(da_nn_classifier_predict(handle_loaded, n_queries, n_features,
                                           X_test.data(), ldx_test, y_pred_loaded.data(),
                                           pr.mode),
                  da_status_success);

        // ==================== COMPARE RESULTS ====================

        // Compare indices and distances (only for k-NN mode)
        if (pr.mode == knn_search_mode) {
            EXPECT_ARR_EQ(pr.n_neighbors * n_queries, k_ind_orig.data(),
                          k_ind_loaded.data(), 1, 1, 0, 0);
            EXPECT_ARR_EQ(pr.n_neighbors * n_queries, k_dist_orig.data(),
                          k_dist_loaded.data(), 1, 1, 0, 0);
        }

        // Compare number of classes
        EXPECT_EQ(n_classes_orig, n_classes_loaded);

        // Compare predictions
        EXPECT_ARR_EQ(n_queries, y_pred_orig.data(), y_pred_loaded.data(), 1, 1, 0, 0);

        // Test multiple predictions work after loading (stability check)
        std::vector<da_int> y_pred_again(n_queries);
        EXPECT_EQ(da_nn_classifier_predict(handle_loaded, n_queries, n_features,
                                           X_test.data(), ldx_test, y_pred_again.data(),
                                           pr.mode),
                  da_status_success);
        EXPECT_ARR_EQ(n_queries, y_pred_orig.data(), y_pred_again.data(), 1, 1, 0, 0);

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

        if (pr.mode == knn_search_mode) {
            da_int n_neighbors_loaded = 0;
            EXPECT_EQ(da_options_get_int(handle_loaded, "number of neighbors",
                                         &n_neighbors_loaded),
                      da_status_success);
            EXPECT_EQ(n_neighbors_loaded, pr.n_neighbors);
        } else {
            T radius_loaded = 0;
            EXPECT_EQ(da_options_get(handle_loaded, "radius", &radius_loaded),
                      da_status_success);
            EXPECT_NEAR(radius_loaded, (T)pr.radius, std::numeric_limits<T>::epsilon());
        }

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

TEST_P(KNNClassSerializationTest, double) {
    const knn_class_serial_params &pr = GetParam();
    knn_class_serialization_test<double>(pr, n_samples, n_features, n_queries, X_train,
                                         y_train, ldx_train, X_test, ldx_test,
                                         model_file);
}

TEST_P(KNNClassSerializationTest, float) {
    const knn_class_serial_params &pr = GetParam();
    knn_class_serialization_test<float>(pr, n_samples, n_features, n_queries, X_train,
                                        y_train, ldx_train, X_test, ldx_test, model_file);
}

INSTANTIATE_TEST_SUITE_P(KNNClassSerializationSuite, KNNClassSerializationTest,
                         testing::ValuesIn(knn_class_serialization_params));

// ==================== ERROR HANDLING TESTS ====================

class KNNClassSerializationErrorTest : public testing::Test {
  protected:
    std::string model_file =
        model_persistence_test_utils::get_test_file_dir() + "/knn_class_error_test.bin";
    void TearDown() override { std::remove(model_file.c_str()); }
};

TEST_F(KNNClassSerializationErrorTest, SaveBeforeSetDataFails) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_nn), da_status_success);
    EXPECT_EQ(da_handle_save_model(handle, model_file.c_str()), da_status_no_data);
    da_handle_destroy(&handle);
}
