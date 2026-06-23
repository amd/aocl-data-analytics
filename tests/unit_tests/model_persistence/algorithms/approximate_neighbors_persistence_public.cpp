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

// Test parameters for ANN serialization tests
struct ann_serial_params {
    std::string test_name;
    std::string metric;
    std::string order;
    da_int ldx_padding;
    da_int seed;
    bool compare_centroids;
    da_int k;
};

void PrintTo(const ann_serial_params &param, ::std::ostream *os) {
    *os << param.test_name;
}

// Test configurations covering different codepaths
const std::vector<ann_serial_params> ann_serialization_params = {
    {"sqeuclidean_colmajor", "sqeuclidean", "column-major", 3, 123, true, 3},
    {"sqeuclidean_rowmajor", "sqeuclidean", "row-major", 2, 123, true, 2},
    {"euclidean_colmajor", "euclidean", "column-major", 0, 456, true, 3},
    {"inner_product_colmajor", "inner product", "column-major", 0, 789, false, 2},
    {"cosine_colmajor", "cosine", "column-major", 1, 321, false, 3},
};

// Fixed algorithm parameters
constexpr da_int ANN_N_LIST = 4;
constexpr da_int ANN_N_PROBE = 4;
constexpr da_int ANN_KMEANS_ITER = 15;
constexpr double ANN_TRAIN_FRACTION = 0.75; // Non-default to test serialization

class ANNSerializationTest : public testing::TestWithParam<ann_serial_params> {
  protected:
    // Problem dimensions
    da_int n_samples = 16;
    da_int n_features = 2;
    da_int n_queries = 3;

    // Padded data arrays and strides (computed in SetUp based on test params)
    std::vector<double> X_train;
    std::vector<double> X_test;
    da_int ldx_train = 0;
    da_int ldx_test = 0;

    std::string model_file;

    void SetUp() override {
        const ann_serial_params &pr = GetParam();
        const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_case = test_info->name();
        std::replace(test_case.begin(), test_case.end(), '/', '_');
        model_file = model_persistence_test_utils::get_test_file_dir() + "/ann_" +
                     pr.test_name + "_" + test_case + ".bin";
        bool is_colmajor = (pr.order == "column-major");

        // Training data (column-major order): 4 clusters of 4 points each
        std::vector<double> X_train_colmajor = {
            0.0, 1.1,  0.0,  1.0,  6.0,  7.2,  6.1,  7.0,  0.0,  1.0, 0.1,
            1.1, 10.0, 11.1, 10.0, 11.0, -0.1, 0.0,  1.1,  1.0,  0.0, 0.1,
            1.0, 1.1,  10.0, 10.2, 11.0, 11.1, 10.0, 10.0, 11.2, 11.0};

        // Query points (column-major)
        std::vector<double> X_test_colmajor = {3.5, 0.4, 5.6, 0.4, 5.0, 5.1};

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
void ann_serialization_test(const ann_serial_params &pr, da_int n_samples,
                            da_int n_features, da_int n_queries,
                            const std::vector<double> &X_train_src,
                            const std::vector<double> &X_test_src, da_int ldx_train,
                            da_int ldx_test, const std::string &model_file) {

    // Convert source data to test precision
    std::vector<T> X_train(X_train_src.begin(), X_train_src.end());
    std::vector<T> X_test(X_test_src.begin(), X_test_src.end());

    // Result arrays for original model
    std::vector<da_int> k_ind_orig(n_queries * pr.k);
    std::vector<T> k_dist_orig(n_queries * pr.k);
    da_int centroids_dim = ANN_N_LIST * n_features;
    std::vector<T> centroids_orig(centroids_dim);
    std::vector<da_int> list_sizes_orig(ANN_N_LIST);
    da_int rinfo_size = 4;
    std::vector<T> rinfo_orig(rinfo_size);

    // ==================== ORIGINAL MODEL BLOCK ====================
    {
        da_handle handle_orig = nullptr;
        EXPECT_EQ(da_handle_init<T>(&handle_orig, da_handle_approx_nn),
                  da_status_success);

        EXPECT_EQ(da_options_set_string(handle_orig, "algorithm", "ivfflat"),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle_orig, "metric", pr.metric.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle_orig, "storage order", pr.order.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle_orig, "number of neighbors", pr.k),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle_orig, "n_list", ANN_N_LIST),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle_orig, "n_probe", ANN_N_PROBE),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle_orig, "k-means_iter", ANN_KMEANS_ITER),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle_orig, "seed", pr.seed), da_status_success);
        EXPECT_EQ(da_options_set(handle_orig, "train fraction", (T)ANN_TRAIN_FRACTION),
                  da_status_success);

        EXPECT_EQ(da_approx_nn_set_training_data(handle_orig, n_samples, n_features,
                                                 X_train.data(), ldx_train),
                  da_status_success);
        EXPECT_EQ(da_approx_nn_train_and_add<T>(handle_orig), da_status_success);

        // Query k-nearest neighbors
        EXPECT_EQ(da_approx_nn_kneighbors(handle_orig, n_queries, n_features,
                                          X_test.data(), ldx_test, k_ind_orig.data(),
                                          k_dist_orig.data(), pr.k, true),
                  da_status_success);

        // Get centroids (if applicable)
        if (pr.compare_centroids) {
            da_int dim = centroids_dim;
            EXPECT_EQ(da_handle_get_result(handle_orig, da_approx_nn_cluster_centroids,
                                           &dim, centroids_orig.data()),
                      da_status_success);
        }

        // Get list sizes
        da_int ls_dim = ANN_N_LIST;
        EXPECT_EQ(da_handle_get_result(handle_orig, da_approx_nn_list_sizes, &ls_dim,
                                       list_sizes_orig.data()),
                  da_status_success);

        // Get rinfo
        da_int ri_dim = rinfo_size;
        EXPECT_EQ(da_handle_get_result(handle_orig, da_rinfo, &ri_dim, rinfo_orig.data()),
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
        std::vector<da_int> k_ind_loaded(n_queries * pr.k);
        std::vector<T> k_dist_loaded(n_queries * pr.k);
        std::vector<T> centroids_loaded(centroids_dim);
        std::vector<da_int> list_sizes_loaded(ANN_N_LIST);
        std::vector<T> rinfo_loaded(rinfo_size);

        // Query from loaded model
        EXPECT_EQ(da_approx_nn_kneighbors(handle_loaded, n_queries, n_features,
                                          X_test.data(), ldx_test, k_ind_loaded.data(),
                                          k_dist_loaded.data(), pr.k, true),
                  da_status_success);

        // Get centroids from loaded model
        if (pr.compare_centroids) {
            da_int dim = centroids_dim;
            EXPECT_EQ(da_handle_get_result(handle_loaded, da_approx_nn_cluster_centroids,
                                           &dim, centroids_loaded.data()),
                      da_status_success);
        }

        // Get list sizes from loaded model
        da_int ls_dim = ANN_N_LIST;
        EXPECT_EQ(da_handle_get_result(handle_loaded, da_approx_nn_list_sizes, &ls_dim,
                                       list_sizes_loaded.data()),
                  da_status_success);

        // Get rinfo from loaded model
        da_int ri_dim = rinfo_size;
        EXPECT_EQ(
            da_handle_get_result(handle_loaded, da_rinfo, &ri_dim, rinfo_loaded.data()),
            da_status_success);

        // ==================== COMPARE RESULTS ====================

        // Compare neighbor indices
        EXPECT_ARR_EQ(n_queries * pr.k, k_ind_orig.data(), k_ind_loaded.data(), 1, 1, 0,
                      0);

        // Compare neighbor distances
        EXPECT_ARR_EQ(n_queries * pr.k, k_dist_orig.data(), k_dist_loaded.data(), 1, 1, 0,
                      0);

        // Compare centroids (if applicable)
        if (pr.compare_centroids) {
            EXPECT_ARR_EQ(centroids_dim, centroids_orig.data(), centroids_loaded.data(),
                          1, 1, 0, 0);
        }

        // Compare list sizes
        EXPECT_ARR_EQ(ANN_N_LIST, list_sizes_orig.data(), list_sizes_loaded.data(), 1, 1,
                      0, 0);

        // Compare rinfo (skip last element kmeans_iter for inner_product/cosine)
        da_int rinfo_cmp_size = pr.compare_centroids ? rinfo_size : (rinfo_size - 1);
        EXPECT_ARR_EQ(rinfo_cmp_size, rinfo_orig.data(), rinfo_loaded.data(), 1, 1, 0, 0);

        // Test multiple queries work after loading (stability check)
        std::vector<da_int> k_ind_again(n_queries * pr.k);
        std::vector<T> k_dist_again(n_queries * pr.k);
        EXPECT_EQ(da_approx_nn_kneighbors(handle_loaded, n_queries, n_features,
                                          X_test.data(), ldx_test, k_ind_again.data(),
                                          k_dist_again.data(), pr.k, true),
                  da_status_success);
        EXPECT_ARR_EQ(n_queries * pr.k, k_ind_orig.data(), k_ind_again.data(), 1, 1, 0,
                      0);
        EXPECT_ARR_EQ(n_queries * pr.k, k_dist_orig.data(), k_dist_again.data(), 1, 1, 0,
                      0);

        // ==================== VERIFY OPTIONS ====================
        // Verify that options are preserved after serialization
        char metric_loaded[64];
        da_int metric_len = 64;
        EXPECT_EQ(
            da_options_get_string(handle_loaded, "metric", metric_loaded, &metric_len),
            da_status_success);
        EXPECT_STREQ(metric_loaded, pr.metric.c_str());

        char order_loaded[64];
        da_int order_len = 64;
        EXPECT_EQ(da_options_get_string(handle_loaded, "storage order", order_loaded,
                                        &order_len),
                  da_status_success);
        EXPECT_STREQ(order_loaded, pr.order.c_str());

        da_int k_loaded = 0;
        EXPECT_EQ(da_options_get_int(handle_loaded, "number of neighbors", &k_loaded),
                  da_status_success);
        EXPECT_EQ(k_loaded, pr.k);

        da_int n_list_loaded = 0;
        EXPECT_EQ(da_options_get_int(handle_loaded, "n_list", &n_list_loaded),
                  da_status_success);
        EXPECT_EQ(n_list_loaded, ANN_N_LIST);

        da_int n_probe_loaded = 0;
        EXPECT_EQ(da_options_get_int(handle_loaded, "n_probe", &n_probe_loaded),
                  da_status_success);
        EXPECT_EQ(n_probe_loaded, ANN_N_PROBE);

        da_int kmeans_iter_loaded = 0;
        EXPECT_EQ(da_options_get_int(handle_loaded, "k-means_iter", &kmeans_iter_loaded),
                  da_status_success);
        EXPECT_EQ(kmeans_iter_loaded, ANN_KMEANS_ITER);

        da_int seed_loaded = 0;
        EXPECT_EQ(da_options_get_int(handle_loaded, "seed", &seed_loaded),
                  da_status_success);
        EXPECT_EQ(seed_loaded, pr.seed);

        T train_fraction_loaded = 0;
        EXPECT_EQ(da_options_get(handle_loaded, "train fraction", &train_fraction_loaded),
                  da_status_success);
        EXPECT_EQ(train_fraction_loaded, (T)ANN_TRAIN_FRACTION);

        da_handle_destroy(&handle_loaded);
    }
}

TEST_P(ANNSerializationTest, double) {
    const ann_serial_params &pr = GetParam();
    ann_serialization_test<double>(pr, n_samples, n_features, n_queries, X_train, X_test,
                                   ldx_train, ldx_test, model_file);
}

TEST_P(ANNSerializationTest, float) {
    const ann_serial_params &pr = GetParam();
    ann_serialization_test<float>(pr, n_samples, n_features, n_queries, X_train, X_test,
                                  ldx_train, ldx_test, model_file);
}

INSTANTIATE_TEST_SUITE_P(ANNSerializationSuite, ANNSerializationTest,
                         testing::ValuesIn(ann_serialization_params));

// ==================== ERROR HANDLING TESTS ====================

class ANNSerializationErrorTest : public testing::Test {
  protected:
    std::string model_file =
        model_persistence_test_utils::get_test_file_dir() + "/ann_error_test.bin";
    void TearDown() override { std::remove(model_file.c_str()); }
};

TEST_F(ANNSerializationErrorTest, SaveBeforeTrainFails) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_approx_nn), da_status_success);
    EXPECT_EQ(da_handle_save_model(handle, model_file.c_str()), da_status_no_data);
    da_handle_destroy(&handle);
}
