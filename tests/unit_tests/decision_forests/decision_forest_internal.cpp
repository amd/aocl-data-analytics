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

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "da_omp.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

TEST(decision_forest_internal, parallel_split) {
    // Load training data from CSV
    std::string input_data_fname =
        std::string(DATA_DIR) + "/df_data/gen_500x20_4class_data.csv";
    da_datastore csv_store = nullptr;
    EXPECT_EQ(da_datastore_init(&csv_store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(csv_store, "datastore precision",
                                              prec_name<double>()),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(csv_store, input_data_fname.c_str()),
              da_status_success);
    da_int ncols, nrows;
    EXPECT_EQ(da_data_get_n_cols(csv_store, &ncols), da_status_success);
    EXPECT_EQ(da_data_get_n_rows(csv_store, &nrows), da_status_success);
    da_int nfeat = ncols - 1;
    da_int nsamples = nrows;
    EXPECT_EQ(da_data_select_columns(csv_store, "features", 0, ncols - 2),
              da_status_success);
    EXPECT_EQ(da_data_select_columns(csv_store, "response", ncols - 1, ncols - 1),
              da_status_success);
    std::vector<double> X(nfeat * nsamples);
    std::vector<da_int> y(nsamples);
    EXPECT_EQ(da_data_extract_selection(csv_store, "features", column_major, X.data(),
                                        nsamples),
              da_status_success);
    EXPECT_EQ(da_data_extract_selection(csv_store, "response", column_major, y.data(),
                                        nsamples),
              da_status_success);
    da_datastore_destroy(&csv_store);

    // Load test data from CSV
    input_data_fname = std::string(DATA_DIR) + "/df_data/gen_500x20_4class_test.csv";
    csv_store = nullptr;
    EXPECT_EQ(da_datastore_init(&csv_store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(csv_store, "datastore precision",
                                              prec_name<double>()),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(csv_store, input_data_fname.c_str()),
              da_status_success);
    EXPECT_EQ(da_data_get_n_cols(csv_store, &ncols), da_status_success);
    EXPECT_EQ(da_data_get_n_rows(csv_store, &nrows), da_status_success);
    EXPECT_EQ(da_data_select_columns(csv_store, "features", 0, ncols - 2),
              da_status_success);
    EXPECT_EQ(da_data_select_columns(csv_store, "response", ncols - 1, ncols - 1),
              da_status_success);
    da_int nsamples_test = nrows;
    std::vector<double> X_test(nfeat * nsamples_test);
    std::vector<da_int> y_test(nsamples_test);
    EXPECT_EQ(da_data_extract_selection(csv_store, "features", column_major,
                                        X_test.data(), nsamples_test),
              da_status_success);
    EXPECT_EQ(da_data_extract_selection(csv_store, "response", column_major,
                                        y_test.data(), nsamples_test),
              da_status_success);
    da_datastore_destroy(&csv_store);

    // Train the forest with parallel splits (8 threads)
    da_handle forest_handle = nullptr;
    EXPECT_EQ(da_handle_init<double>(&forest_handle, da_handle_decision_forest),
              da_status_success);
    EXPECT_EQ(da_forest_set_training_data(forest_handle, nsamples, nfeat, 0, X.data(),
                                          nsamples, y.data()),
              da_status_success);
    da_int n_tree = 3;
    EXPECT_EQ(da_options_set(forest_handle, "number of trees", n_tree),
              da_status_success);
    EXPECT_EQ(da_options_set(forest_handle, "features selection", "all"),
              da_status_success);
    da_int max_tree_threads = 8;
    EXPECT_EQ(da_options_set(forest_handle, "maximum tree threads", max_tree_threads),
              da_status_success);
    EXPECT_EQ(da_forest_fit<double>(forest_handle), da_status_success);

    // Check accuracy on the test data
    double accuracy;
    EXPECT_EQ(da_forest_score(forest_handle, nsamples_test, nfeat, X_test.data(),
                              nsamples_test, y_test.data(), &accuracy),
              da_status_success);
    EXPECT_GT(accuracy, (double)0.8);

    // Check that n_threads_split reflects the requested thread count via rinfo
    double rinfo[6];
    da_int dim = 6;
    da_int n_omp_threads = omp_get_max_threads();
    da_int thread_add = n_omp_threads % n_tree == 0 ? 0 : 1;
    da_int threads_per_tree = std::min(
        max_tree_threads, std::max(n_omp_threads / n_tree + thread_add, (da_int)1));
    EXPECT_EQ(da_handle_get_result(forest_handle, da_result::da_rinfo, &dim, rinfo),
              da_status_success);
    da_int expected_threads = std::min(max_tree_threads, threads_per_tree);
    EXPECT_EQ((da_int)rinfo[5], expected_threads);

    da_handle_destroy(&forest_handle);
}
