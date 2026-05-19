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

// Test parameters for kmeans serialization tests
struct kmeans_serial_params {
    std::string test_name;
    std::string algorithm;
    std::string initialization_method;
    da_int n_clusters;
    da_int max_iter;
    da_int seed;
};

void PrintTo(const kmeans_serial_params &param, ::std::ostream *os) {
    *os << param.test_name;
}

// Test configurations covering different codepaths
// Defaults: algorithm="lloyd", init="random", n_clusters=1, n_init=10, max_iter=300, seed=0
const std::vector<kmeans_serial_params> kmeans_serialization_params = {
    {"elkan_kmeanspp_3c", "elkan", "k-means++", 3, 150, 77},
    {"lloyd_random_2c", "lloyd", "random", 2, 200, 42},
    {"macqueen_kmeanspp_2c", "macqueen", "k-means++", 2, 100, 123},
    {"hartigan_wong_random_3c", "hartigan-wong", "random", 3, 250, 456},
};

class KMeansSerializationTest : public testing::TestWithParam<kmeans_serial_params> {
  protected:
    // Problem dimensions
    da_int n_samples = 15;
    da_int n_features = 2;
    da_int lda = 15;
    da_int n_samples_test = 4;
    da_int ldx = 4;

    std::vector<double> A;
    std::vector<double> X_test;

    std::string model_file;

    void SetUp() override {
        const kmeans_serial_params &pr = GetParam();
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_case = test_info->name();
        std::replace(test_case.begin(), test_case.end(), '/', '_');
        model_file = model_persistence_test_utils::get_test_file_dir() + "/kmeans_" +
                     pr.test_name + "_" + test_case + ".bin";

        // Training data: 3 clusters for more interesting clustering
        // Cluster 0 around (2, 2), Cluster 1 around (-2, -2), Cluster 2 around (2, -2)
        // clang-format off
        A = {
            2.1,  1.9,  2.0,  2.2,  1.8,
           -2.1, -1.9, -2.0, -2.2, -1.8,
            2.1,  1.9,  2.0,  2.2,  1.8,
            2.0,  2.1,  1.9,  2.2,  1.8,
           -2.0, -2.1, -1.9, -2.2, -1.8,
           -2.0, -2.1, -1.9, -2.2, -1.8
        };
        // clang-format on

        // Test data for predict
        X_test = {2.0, -2.0, 2.0, 0.0, 2.0, -2.0, -2.0, 0.0};
    }

    void TearDown() override { std::remove(model_file.c_str()); }
};

template <typename T>
void kmeans_serialization_test(const kmeans_serial_params &pr, da_int n_samples,
                               da_int n_features, da_int lda, da_int n_samples_test,
                               da_int ldx, const std::vector<double> &A_d,
                               const std::vector<double> &X_test_d,
                               const std::string &model_file) {

    // Convert data to correct type
    std::vector<T> A(A_d.begin(), A_d.end());
    std::vector<T> X_test(X_test_d.begin(), X_test_d.end());

    // Result arrays for original model
    std::vector<T> centres_orig(pr.n_clusters * n_features);
    std::vector<da_int> labels_orig(n_samples);
    std::vector<da_int> predict_orig(n_samples_test);
    std::vector<T> transform_orig(n_samples_test * pr.n_clusters);

    // ==================== ORIGINAL MODEL BLOCK ====================
    {
        da_handle handle_orig = nullptr;
        EXPECT_EQ(da_handle_init<T>(&handle_orig, da_handle_kmeans), da_status_success);

        EXPECT_EQ(da_options_set_string(handle_orig, "algorithm", pr.algorithm.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle_orig, "initialization method",
                                        pr.initialization_method.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle_orig, "n_clusters", pr.n_clusters),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle_orig, "max_iter", pr.max_iter),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle_orig, "seed", pr.seed), da_status_success);
        EXPECT_EQ(da_options_set_int(handle_orig, "n_init", 1), da_status_success);

        EXPECT_EQ(da_kmeans_set_data(handle_orig, n_samples, n_features, A.data(), lda),
                  da_status_success);

        EXPECT_EQ(da_kmeans_compute<T>(handle_orig), da_status_success);

        // Get cluster centres
        da_int centres_dim = pr.n_clusters * n_features;
        EXPECT_EQ(da_handle_get_result(handle_orig, da_kmeans_cluster_centres,
                                       &centres_dim, centres_orig.data()),
                  da_status_success);

        // Get labels
        da_int labels_dim = n_samples;
        EXPECT_EQ(da_handle_get_result_int(handle_orig, da_kmeans_labels, &labels_dim,
                                           labels_orig.data()),
                  da_status_success);

        // Predict on test data
        EXPECT_EQ(da_kmeans_predict(handle_orig, n_samples_test, n_features,
                                    X_test.data(), ldx, predict_orig.data()),
                  da_status_success);

        // Transform test data
        EXPECT_EQ(da_kmeans_transform(handle_orig, n_samples_test, n_features,
                                      X_test.data(), ldx, transform_orig.data(),
                                      n_samples_test),
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
        std::vector<T> centres_loaded(pr.n_clusters * n_features);
        std::vector<da_int> labels_loaded(n_samples);
        std::vector<da_int> predict_loaded(n_samples_test);
        std::vector<T> transform_loaded(n_samples_test * pr.n_clusters);

        // Get cluster centres from loaded model
        da_int centres_dim = pr.n_clusters * n_features;
        EXPECT_EQ(da_handle_get_result(handle_loaded, da_kmeans_cluster_centres,
                                       &centres_dim, centres_loaded.data()),
                  da_status_success);

        // Get labels from loaded model
        da_int labels_dim = n_samples;
        EXPECT_EQ(da_handle_get_result_int(handle_loaded, da_kmeans_labels, &labels_dim,
                                           labels_loaded.data()),
                  da_status_success);

        // Predict from loaded model
        EXPECT_EQ(da_kmeans_predict(handle_loaded, n_samples_test, n_features,
                                    X_test.data(), ldx, predict_loaded.data()),
                  da_status_success);

        // Transform from loaded model
        EXPECT_EQ(da_kmeans_transform(handle_loaded, n_samples_test, n_features,
                                      X_test.data(), ldx, transform_loaded.data(),
                                      n_samples_test),
                  da_status_success);

        // ==================== COMPARE RESULTS ====================

        // Compare cluster centres
        EXPECT_ARR_EQ(pr.n_clusters * n_features, centres_orig.data(),
                      centres_loaded.data(), 1, 1, 0, 0);

        // Compare labels
        EXPECT_ARR_EQ(n_samples, labels_orig.data(), labels_loaded.data(), 1, 1, 0, 0);

        // Compare predictions
        EXPECT_ARR_EQ(n_samples_test, predict_orig.data(), predict_loaded.data(), 1, 1, 0,
                      0);

        // Compare transforms
        EXPECT_ARR_EQ(n_samples_test * pr.n_clusters, transform_orig.data(),
                      transform_loaded.data(), 1, 1, 0, 0);

        // Test multiple predictions work after loading (stability check)
        std::vector<da_int> predict_again(n_samples_test);
        EXPECT_EQ(da_kmeans_predict(handle_loaded, n_samples_test, n_features,
                                    X_test.data(), ldx, predict_again.data()),
                  da_status_success);
        EXPECT_ARR_EQ(n_samples_test, predict_orig.data(), predict_again.data(), 1, 1, 0,
                      0);

        // ==================== VERIFY OPTIONS ====================
        // Verify that options are preserved after serialization
        char algo_loaded[64];
        da_int algo_len = 64;
        EXPECT_EQ(
            da_options_get_string(handle_loaded, "algorithm", algo_loaded, &algo_len),
            da_status_success);
        EXPECT_STREQ(algo_loaded, pr.algorithm.c_str());

        char init_loaded[64];
        da_int init_len = 64;
        EXPECT_EQ(da_options_get_string(handle_loaded, "initialization method",
                                        init_loaded, &init_len),
                  da_status_success);
        EXPECT_STREQ(init_loaded, pr.initialization_method.c_str());

        da_int n_clusters_loaded = 0;
        EXPECT_EQ(da_options_get_int(handle_loaded, "n_clusters", &n_clusters_loaded),
                  da_status_success);
        EXPECT_EQ(n_clusters_loaded, pr.n_clusters);

        da_int max_iter_loaded = 0;
        EXPECT_EQ(da_options_get_int(handle_loaded, "max_iter", &max_iter_loaded),
                  da_status_success);
        EXPECT_EQ(max_iter_loaded, pr.max_iter);

        da_int seed_loaded = 0;
        EXPECT_EQ(da_options_get_int(handle_loaded, "seed", &seed_loaded),
                  da_status_success);
        EXPECT_EQ(seed_loaded, pr.seed);

        da_int n_init_loaded = 0;
        EXPECT_EQ(da_options_get_int(handle_loaded, "n_init", &n_init_loaded),
                  da_status_success);
        EXPECT_EQ(n_init_loaded, 1);

        da_handle_destroy(&handle_loaded);
    }
}

TEST_P(KMeansSerializationTest, double) {
    const kmeans_serial_params &pr = GetParam();
    kmeans_serialization_test<double>(pr, n_samples, n_features, lda, n_samples_test, ldx,
                                      A, X_test, model_file);
}

TEST_P(KMeansSerializationTest, float) {
    const kmeans_serial_params &pr = GetParam();
    kmeans_serialization_test<float>(pr, n_samples, n_features, lda, n_samples_test, ldx,
                                     A, X_test, model_file);
}

INSTANTIATE_TEST_SUITE_P(KMeansSerializationSuite, KMeansSerializationTest,
                         testing::ValuesIn(kmeans_serialization_params));

// ==================== ERROR HANDLING TESTS ====================

class KMeansSerializationErrorTest : public testing::Test {
  protected:
    std::string model_file;
    void SetUp() override {
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_name = test_info->name();
        model_file = model_persistence_test_utils::get_test_file_dir() +
                     "/kmeans_error_" + test_name + ".bin";
    }
    void TearDown() override { std::remove(model_file.c_str()); }
};

TEST_F(KMeansSerializationErrorTest, SaveBeforeComputeFails) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_kmeans), da_status_success);
    EXPECT_EQ(da_handle_save_model(handle, model_file.c_str()), da_status_no_data);
    da_handle_destroy(&handle);
}

TEST_F(KMeansSerializationErrorTest, FitAfterLoadWithoutSetDataFails) {
    // Train and save a model
    da_handle handle_train = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle_train, da_handle_kmeans), da_status_success);
    std::vector<double> X_train = {1.0, 2.0, 1.5, 2.5, 5.0, 6.0, 5.5, 6.5};
    EXPECT_EQ(da_kmeans_set_data_d(handle_train, 4, 2, X_train.data(), 4),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle_train, "n_clusters", 2), da_status_success);
    EXPECT_EQ(da_kmeans_compute_d(handle_train), da_status_success);
    EXPECT_EQ(da_handle_save_model(handle_train, model_file.c_str()), da_status_success);
    da_handle_destroy(&handle_train);

    // Load model and try to fit without calling set_data
    da_handle handle_load = nullptr;
    EXPECT_EQ(da_handle_load_model(&handle_load, model_file.c_str()), da_status_success);
    EXPECT_EQ(da_kmeans_compute_d(handle_load), da_status_no_data);
    da_handle_destroy(&handle_load);
}