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
#include "da_handle.hpp"
#include "da_omp.hpp"
#include "forest/decision_forest.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

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
    ASSERT_EQ(da_forest_select_model<double>(forest_handle, df_classifier),
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

// Regression test for the sample-blocking path in forest inference. Forcing a
// tiny per-buffer element budget via the hidden setting
// "decision_forest.inference_block_elems" drives block_nsamp down to 1 (see
// decision_forest<T>::inference_block_nsamp), so the inference loops run with
// n_blocks == nsamp (>= 2). The blocked results must reproduce the default
// single-block results: bit-identical for predict/score and within tolerance
// for the normalized probabilities. Both storage orders are exercised.
TEST(decision_forest_internal, inference_blocking) {
    const std::string block_key = "decision_forest.inference_block_elems";

    // Make sure we start from the default (single-block) budget
    ASSERT_EQ(da_debug_set(block_key.c_str(), ""), da_status_success);

    for (const std::string &order :
         {std::string("column-major"), std::string("row-major")}) {
        bool row = order == "row-major";
        da_order storage_order = row ? row_major : column_major;

        std::string input_data_fname =
            std::string(DATA_DIR) + "/df_data/gen_500x20_4class_data.csv";
        std::vector<double> X;
        std::vector<da_int> y;
        da_int nsamples, nfeat;
        ASSERT_TRUE(da_test::read_features_response_csv(input_data_fname, X, nsamples,
                                                        nfeat, storage_order, y));

        input_data_fname = std::string(DATA_DIR) + "/df_data/gen_500x20_4class_test.csv";
        std::vector<double> X_test;
        std::vector<da_int> y_test;
        da_int nsamples_test;
        ASSERT_TRUE(da_test::read_features_response_csv(
            input_data_fname, X_test, nsamples_test, nfeat, storage_order, y_test));
        ASSERT_GE(nsamples_test, 2); // needed so block_nsamp == 1 gives n_blocks >= 2

        da_int n_class = *std::max_element(y.begin(), y.end()) + 1;
        da_int ldx_train = row ? nfeat : nsamples;
        da_int ldx_test = row ? nfeat : nsamples_test;
        da_int ldy = row ? n_class : nsamples_test;

        da_handle forest_handle = nullptr;
        ASSERT_EQ(da_handle_init<double>(&forest_handle, da_handle_decision_forest),
                  da_status_success);
        ASSERT_EQ(da_forest_select_model<double>(forest_handle, df_classifier),
                  da_status_success);
        ASSERT_EQ(da_options_set_string(forest_handle, "storage order", order.c_str()),
                  da_status_success);
        ASSERT_EQ(da_options_set(forest_handle, "number of trees", (da_int)20),
                  da_status_success);
        ASSERT_EQ(da_options_set(forest_handle, "seed", (da_int)42), da_status_success);
        ASSERT_EQ(da_options_set(forest_handle, "features selection", "all"),
                  da_status_success);
        ASSERT_EQ(da_forest_set_training_data(forest_handle, nsamples, nfeat, n_class,
                                              X.data(), ldx_train, y.data()),
                  da_status_success);
        ASSERT_EQ(da_forest_fit<double>(forest_handle), da_status_success);

        // Runs predict / predict_proba / predict_log_proba / score with whatever
        // budget is currently set
        auto run_inference = [&](std::vector<da_int> &pred, std::vector<double> &proba,
                                 std::vector<double> &log_proba, double &acc) {
            pred.assign(nsamples_test, -1);
            proba.assign((size_t)nsamples_test * n_class, 0.0);
            log_proba.assign((size_t)nsamples_test * n_class, 0.0);
            acc = -1.0;
            ASSERT_EQ(da_forest_predict(forest_handle, nsamples_test, nfeat,
                                        X_test.data(), ldx_test, pred.data()),
                      da_status_success);
            ASSERT_EQ(da_forest_predict_proba(forest_handle, nsamples_test, nfeat,
                                              X_test.data(), ldx_test, proba.data(),
                                              n_class, ldy),
                      da_status_success);
            ASSERT_EQ(da_forest_predict_log_proba(forest_handle, nsamples_test, nfeat,
                                                  X_test.data(), ldx_test,
                                                  log_proba.data(), n_class, ldy),
                      da_status_success);
            ASSERT_EQ(da_forest_score(forest_handle, nsamples_test, nfeat, X_test.data(),
                                      ldx_test, y_test.data(), &acc),
                      da_status_success);
        };

        // Baseline: default budget => single block
        std::vector<da_int> pred_base;
        std::vector<double> proba_base, log_proba_base;
        double acc_base;
        run_inference(pred_base, proba_base, log_proba_base, acc_base);

        // Blocked: budget of 1 element forces block_nsamp == 1, hence
        // n_blocks == nsamples_test (>= 2)
        ASSERT_EQ(da_debug_set(block_key.c_str(), "1"), da_status_success);
        std::vector<da_int> pred_blk;
        std::vector<double> proba_blk, log_proba_blk;
        double acc_blk;
        run_inference(pred_blk, proba_blk, log_proba_blk, acc_blk);
        ASSERT_EQ(da_debug_set(block_key.c_str(), ""),
                  da_status_success); // restore default

        // predict and score must be bit-identical; probabilities within tolerance
        EXPECT_EQ(pred_base, pred_blk) << "order=" << order;
        EXPECT_EQ(acc_base, acc_blk) << "order=" << order;
        for (size_t k = 0; k < proba_base.size(); k++)
            EXPECT_NEAR(proba_base[k], proba_blk[k], 1.0e-13)
                << "order=" << order << " k=" << k;
        for (size_t k = 0; k < log_proba_base.size(); k++) {
            if (std::isfinite(log_proba_base[k]) && std::isfinite(log_proba_blk[k]))
                EXPECT_NEAR(log_proba_base[k], log_proba_blk[k], 1.0e-13)
                    << "order=" << order << " k=" << k;
            else
                EXPECT_EQ(log_proba_base[k], log_proba_blk[k])
                    << "order=" << order << " k=" << k;
        }

        da_handle_destroy(&forest_handle);
    }

    // Leave the context in its default state for subsequent tests
    ASSERT_EQ(da_debug_set(block_key.c_str(), ""), da_status_success);
}
