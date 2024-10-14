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
void test_decision_tree_positive(std::string csvname, std::vector<option_t<da_int>> iopts,
                                 std::vector<option_t<std::string>> sopts,
                                 std::vector<option_t<T>> ropts, T target_score) {

    // Create main handle and set options
    da_handle tree_handle = nullptr;
    EXPECT_EQ(da_handle_init<T>(&tree_handle, da_handle_decision_tree),
              da_status_success);
    for (auto &op : sopts)
        EXPECT_EQ(da_options_set_string(tree_handle, op.name.c_str(), op.value.c_str()),
                  da_status_success);
    for (auto &op : ropts)
        EXPECT_EQ(da_options_set(tree_handle, op.name.c_str(), op.value),
                  da_status_success);
    for (auto &op : iopts)
        EXPECT_EQ(da_options_set_int(tree_handle, op.name.c_str(), op.value),
                  da_status_success);
    // EXPECT_EQ(da_options_set_string(tree_handle, "print options", "yes"),
    //           da_status_success);

    ////////////////////////
    // Get the training data
    ////////////////////////
    std::string input_data_fname =
        std::string(DATA_DIR) + "/df_data/" + csvname + "_data.csv";
    da_datastore csv_store = nullptr;
    EXPECT_EQ(da_datastore_init(&csv_store), da_status_success);
    EXPECT_EQ(
        da_datastore_options_set_string(csv_store, "datastore precision", prec_name<T>()),
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
    EXPECT_EQ(da_data_extract_selection(csv_store, "features", column_major, X.data(),
                                        nsamples),
              da_status_success);
    EXPECT_EQ(da_data_extract_selection(csv_store, "response", column_major, y.data(),
                                        nsamples),
              da_status_success);
    da_datastore_destroy(&csv_store);

    ///////////////////
    // Create the model
    ///////////////////
    EXPECT_EQ(da_tree_set_training_data(tree_handle, nsamples, nfeat, 0, X.data(),
                                        nsamples, y.data()),
              da_status_success);
    // compute regression
    EXPECT_EQ(da_tree_fit<T>(tree_handle), da_status_success);

    ////////////////////////
    // Get the test data
    ////////////////////////
    input_data_fname = std::string(DATA_DIR) + "/df_data/" + csvname + "_test.csv";
    csv_store = nullptr;
    EXPECT_EQ(da_datastore_init(&csv_store), da_status_success);
    EXPECT_EQ(
        da_datastore_options_set_string(csv_store, "datastore precision", prec_name<T>()),
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
    EXPECT_EQ(da_data_extract_selection(csv_store, "features", column_major,
                                        X_test.data(), nsamples),
              da_status_success);
    EXPECT_EQ(da_data_extract_selection(csv_store, "response", column_major,
                                        y_test.data(), nsamples),
              da_status_success);
    da_datastore_destroy(&csv_store);

    //////////////////////////////////////
    // Check that the score is good enough
    //////////////////////////////////////
    T accuracy;
    EXPECT_EQ(da_tree_score(tree_handle, nsamples, nfeat, X_test.data(), nsamples,
                            y_test.data(), &accuracy),
              da_status_success);
    EXPECT_GT(accuracy, target_score);
    std::cout << "Accuracy on the test data: " << accuracy << std::endl;

    //////////////
    // Print rinfo
    //////////////
    T rinfo[100];
    da_int dim = 100;
    EXPECT_EQ(da_handle_get_result(tree_handle, da_result::da_rinfo, &dim, rinfo),
              da_status_success);
    std::cout << "Tree depth: " << rinfo[4] << std::endl;

    da_handle_destroy(&tree_handle);
}
