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
 */

#include "aoclda.h"
#include "aoclda_cpp_overloads.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

template <class T> struct option_t {
    std::string name{""};
    T value;
};

// return precision as a string literal to set CSV options
template <typename T> constexpr const char *prec_name();
template <> constexpr const char *prec_name<float>() { return "single"; }
template <> constexpr const char *prec_name<double>() { return "double"; }

template <typename T> constexpr const char *type_opt_name();
template <> constexpr const char *type_opt_name<float>() { return "float"; }
template <> constexpr const char *type_opt_name<double>() { return "double"; }

template <typename T>
void test_forest_positive(std::string csvname, std::vector<option_t<da_int>> iopts,
                          std::vector<option_t<std::string>> sopts,
                          std::vector<option_t<T>> ropts, T target_score) {

    // Create main handle and set options
    da_handle forest_handle = nullptr;
    EXPECT_EQ(da_handle_init<T>(&forest_handle, da_handle_decision_forest),
              da_status_success);
    for (auto &op : sopts)
        EXPECT_EQ(da_options_set_string(forest_handle, op.name.c_str(), op.value.c_str()),
                  da_status_success);
    for (auto &op : ropts)
        EXPECT_EQ(da_options_set(forest_handle, op.name.c_str(), op.value),
                  da_status_success);
    for (auto &op : iopts)
        EXPECT_EQ(da_options_set_int(forest_handle, op.name.c_str(), op.value),
                  da_status_success);
    // EXPECT_EQ(da_options_set_string(forest_handle, "print options", "yes"),
    //           da_status_success);

    ////////////////////////
    // Get the training data
    ////////////////////////
    std::string input_data_fname = std::string(DATA_DIR) + "/" + csvname + "_data.csv";
    da_datastore csv_store = nullptr;
    EXPECT_EQ(da_datastore_init(&csv_store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(csv_store, "CSV datastore precision",
                                              prec_name<T>()),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(csv_store, input_data_fname.c_str()),
              da_status_success);

    da_int ncols, nrows;
    EXPECT_EQ(da_data_get_n_cols(csv_store, &ncols), da_status_success);
    EXPECT_EQ(da_data_get_n_rows(csv_store, &nrows), da_status_success);
    // The first ncols-1 columns contain the feature matrix; the last one the response vector
    // Create the selections in the data store
    EXPECT_EQ(da_data_select_columns(csv_store, "features", 0, ncols - 2),
              da_status_success);
    EXPECT_EQ(da_data_select_columns(csv_store, "response", ncols - 1, ncols - 1),
              da_status_success);

    da_int nfeat = ncols - 1;
    da_int nsamples = nrows;
    // Extract the selections
    std::vector<T> X(nfeat * nsamples);
    std::vector<da_int> y(nsamples);
    EXPECT_EQ(da_data_extract_selection(csv_store, "features", X.data(), nsamples),
              da_status_success);
    EXPECT_EQ(da_data_extract_selection(csv_store, "response", y.data(), nsamples),
              da_status_success);
    da_datastore_destroy(&csv_store);

    ///////////////////
    // Create the model
    ///////////////////
    EXPECT_EQ(da_forest_set_training_data(forest_handle, nsamples, nfeat, 0, X.data(),
                                          nsamples, y.data()),
              da_status_success);
    // compute regression
    EXPECT_EQ(da_forest_fit<T>(forest_handle), da_status_success);

    ////////////////////////
    // Get the test data
    ////////////////////////
    input_data_fname = std::string(DATA_DIR) + "/" + csvname + "_test.csv";
    csv_store = nullptr;
    EXPECT_EQ(da_datastore_init(&csv_store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(csv_store, "CSV datastore precision",
                                              prec_name<T>()),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(csv_store, input_data_fname.c_str()),
              da_status_success);

    EXPECT_EQ(da_data_get_n_cols(csv_store, &ncols), da_status_success);
    EXPECT_EQ(da_data_get_n_rows(csv_store, &nrows), da_status_success);
    EXPECT_EQ(da_data_select_columns(csv_store, "features", 0, ncols - 2),
              da_status_success);
    EXPECT_EQ(da_data_select_columns(csv_store, "response", ncols - 1, ncols - 1),
              da_status_success);

    nfeat = ncols - 1;
    nsamples = nrows;
    // Extract the selections
    std::vector<T> X_test(nfeat * nsamples);
    std::vector<da_int> y_test(nsamples);
    EXPECT_EQ(da_data_extract_selection(csv_store, "features", X_test.data(), nsamples),
              da_status_success);
    EXPECT_EQ(da_data_extract_selection(csv_store, "response", y_test.data(), nsamples),
              da_status_success);
    da_datastore_destroy(&csv_store);

    //////////////////////////////////////
    // Check that the score is good enough
    //////////////////////////////////////
    T accuracy;
    EXPECT_EQ(da_forest_score(forest_handle, nsamples, nfeat, X_test.data(), nsamples,
                              y_test.data(), &accuracy),
              da_status_success);
    EXPECT_GT(accuracy, target_score);
    std::cout << "Accuracy on the test data: " << accuracy << std::endl;

    // Check that the prediction finds the same score
    std::vector<da_int> y_pred(nsamples);
    EXPECT_EQ(da_forest_predict(forest_handle, nsamples, nfeat, X_test.data(), nsamples,
                                y_pred.data()),
              da_status_success);
    da_int count_correct = 0;
    for (da_int i = 0; i < nsamples; i++) {
        if (y_pred[i] == y_test[i])
            count_correct++;
    }
    EXPECT_NEAR((T)count_correct / (T)nsamples, accuracy, (T)1.0e-05);

    // Check log_proba is consistent with prediction
    da_int nclass = *std::max_element(y.begin(), y.end()) + 1;
    std::vector<T> y_proba(nsamples * nclass);
    EXPECT_EQ(da_forest_predict_proba(forest_handle, nsamples, nfeat, X_test.data(),
                                      nsamples, y_proba.data(), nclass, nsamples),
              da_status_success);
    count_correct = 0;
    for (da_int i = 0; i < nsamples; i++) {
        T max_prob = 0.0;
        da_int best_class = -1;
        for (da_int j = 0; j < nclass; j++) {
            if (y_proba[j * nsamples + i] > max_prob) {
                max_prob = y_proba[j * nsamples + i];
                best_class = j;
            }
        }
        if (best_class == y_pred[i])
            count_correct += 1;
    }

    EXPECT_GT((T)count_correct / (T)nsamples, (T)0.9);

    //////////////
    // Print rinfo
    //////////////
    // T rinfo[100];
    // da_int dim = 100;
    // EXPECT_EQ(da_handle_get_result(forest_handle, da_result::da_rinfo, &dim, rinfo),
    //           da_status_success);
    // std::cout << "Tree depth: " << rinfo[4] << std::endl;

    da_handle_destroy(&forest_handle);
}
