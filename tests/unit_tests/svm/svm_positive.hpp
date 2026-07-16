/*
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

template <class T> struct option_t {
    std::string name{""};
    T value;
};

// Helper to define precision to which we expect the results to match
template <typename T> T expected_precision(T scale = (T)1.0);
template <> double expected_precision<double>(double scale) {
    return da_numeric::tolerance<double>::safe_tol() * scale *
           1e2; // safe_tol is 2e-8, svm was trained with 1e-6 tolerance
}
template <> float expected_precision<float>(float scale) {
    return da_numeric::tolerance<float>::safe_tol() * scale;
}

std::string get_model_name(da_svm_model model) {
    switch (model) {
    case svc:
        return "svc";
    case svr:
        return "svr";
    case nusvc:
        return "nusvc";
    case nusvr:
        return "nusvr";
    default:
        return "error";
    }
}

template <typename T>
void test_svm_positive(std::string csvname, da_svm_model model,
                       std::vector<option_t<da_int>> iopts,
                       std::vector<option_t<std::string>> sopts,
                       std::vector<option_t<T>> ropts, T target_score,
                       T check_tol_scale) {

    // Create main handle and set options
    da_handle svm_handle = nullptr;
    EXPECT_EQ(da_handle_init<T>(&svm_handle, da_handle_svm), da_status_success);
    for (auto &op : sopts)
        EXPECT_EQ(da_options_set_string(svm_handle, op.name.c_str(), op.value.c_str()),
                  da_status_success);
    for (auto &op : ropts)
        EXPECT_EQ(da_options_set(svm_handle, op.name.c_str(), op.value),
                  da_status_success);
    for (auto &op : iopts)
        EXPECT_EQ(da_options_set_int(svm_handle, op.name.c_str(), op.value),
                  da_status_success);
    std::string kernel_str;
    da_int lvalue = 0;
    EXPECT_EQ(da_options_get_string(svm_handle, "kernel", kernel_str.data(), &lvalue),
              da_status_invalid_input);
    kernel_str.resize(lvalue);
    EXPECT_EQ(da_options_get_string(svm_handle, "kernel", kernel_str.data(), &lvalue),
              da_status_success);
    if (lvalue > 0)
        kernel_str.resize(lvalue - 1);
    ////////////////////////
    // Get the training data
    ////////////////////////
    std::string input_data_fname =
        std::string(DATA_DIR) + "/svm_data/" + csvname + "_train.csv";
    std::vector<T> train_data;
    da_int nrows, ncols;
    ASSERT_TRUE(
        da_test::read_csv_data(input_data_fname, train_data, nrows, ncols, column_major))
        << "Failed to read training data: " << input_data_fname;
    // The first ncols-1 columns contain the feature matrix; the last one the response vector
    da_int nfeat = ncols - 1;
    da_int nsamples = nrows;
    std::vector<T> X(train_data.begin(), train_data.begin() + nfeat * nsamples);
    std::vector<T> y(train_data.begin() + nfeat * nsamples, train_data.end());

    ///////////////////
    // Create the model
    ///////////////////
    EXPECT_EQ(da_svm_select_model<T>(svm_handle, model), da_status_success);
    EXPECT_EQ(da_svm_set_data(svm_handle, nsamples, nfeat, X.data(), nsamples, y.data()),
              da_status_success);
    // Train SVM
    EXPECT_EQ(da_svm_compute<T>(svm_handle), da_status_success);
    da_handle_print_error_message(svm_handle);

    //////////////////////////
    // Check dual coefficients
    //////////////////////////
    da_int n_SV;
    da_int one = 1, size = 100;
    T rinfo[100];
    EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_rinfo, &size, rinfo),
              da_status_success);
    da_int nclass = rinfo[2];
    EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_n_support_vectors, &one,
                                   &n_SV),
              da_status_success);
    da_int dim = (nclass - 1) * n_SV;
    std::vector<T> dual_coeffs(dim);
    EXPECT_EQ(da_handle_get_result(svm_handle, da_result::da_svm_dual_coef, &dim,
                                   dual_coeffs.data()),
              da_status_success);
    std::string coef_fname = std::string(DATA_DIR) + "/svm_data/" +
                             get_model_name(model) + "/" + csvname + "_" + kernel_str +
                             "_dual.csv";
    std::vector<T> coef_exp;
    da_int n_rows{0}, n_cols{0};
    ASSERT_TRUE(da_test::read_csv_data(coef_fname, coef_exp, n_rows, n_cols))
        << "Failed to read dual coefficients: " << coef_fname;
    da_int coef_total = n_rows * n_cols;
    EXPECT_EQ(coef_total, dim) << "Number of coefficients to check does not match";
    EXPECT_ARR_NEAR(dim, dual_coeffs, coef_exp, expected_precision<T>(check_tol_scale))
        << "Checking coefficients (solution)";

    ////////////////////////
    // Get the test data
    ////////////////////////
    input_data_fname = std::string(DATA_DIR) + "/svm_data/" + csvname + "_test.csv";
    std::vector<T> test_data;
    ASSERT_TRUE(
        da_test::read_csv_data(input_data_fname, test_data, nrows, ncols, column_major))
        << "Failed to read test data: " << input_data_fname;
    nfeat = ncols - 1;
    nsamples = nrows;
    std::vector<T> X_test(test_data.begin(), test_data.begin() + nfeat * nsamples);
    std::vector<T> y_test(test_data.begin() + nfeat * nsamples, test_data.end());

    ////////////////////////////////////////////////
    // Check decision function (only classification)
    ////////////////////////////////////////////////
    if (model == svc || model == nusvc) {
        if (nclass > 2)
            dim = nsamples * nclass;
        else
            dim = nsamples;
        std::vector<T> decision_values(dim);
        EXPECT_EQ(da_svm_decision_function(svm_handle, nsamples, nfeat, X_test.data(),
                                           nsamples, ovr, decision_values.data(),
                                           nsamples),
                  da_status_success);
        std::string dec_fname = std::string(DATA_DIR) + "/svm_data/" +
                                get_model_name(model) + "/" + csvname + "_" + kernel_str +
                                "_dec.csv";
        std::vector<T> dec_exp;
        n_rows = 0, n_cols = 0;
        ASSERT_TRUE(da_test::read_csv_data(dec_fname, dec_exp, n_rows, n_cols))
            << "Failed to read decision function values: " << dec_fname;
        da_int dec_total = n_rows * n_cols;
        EXPECT_EQ(dec_total, dim) << "Number of coefficients to check does not match";
        EXPECT_ARR_NEAR(dim, decision_values, dec_exp,
                        expected_precision<T>(check_tol_scale))
            << "Checking decision function values (solution)";
    }
    //////////////////////////
    // Check prediction
    //////////////////////////
    std::vector<T> predictions(nsamples);
    EXPECT_EQ(da_svm_predict(svm_handle, nsamples, nfeat, X_test.data(), nsamples,
                             predictions.data()),
              da_status_success);
    std::string pred_fname = std::string(DATA_DIR) + "/svm_data/" +
                             get_model_name(model) + "/" + csvname + "_" + kernel_str +
                             "_pred.csv";
    std::vector<T> pred_exp;
    n_rows = 0, n_cols = 0;
    ASSERT_TRUE(da_test::read_csv_data(pred_fname, pred_exp, n_rows, n_cols))
        << "Failed to read expected predictions: " << pred_fname;
    da_int pred_total = n_rows * n_cols;
    EXPECT_EQ(pred_total, nsamples) << "Number of coefficients to check does not match";
    EXPECT_ARR_NEAR(nsamples, predictions, pred_exp,
                    expected_precision<T>(check_tol_scale))
        << "Checking test labels (solution)";

    //////////////////////////////////////
    // Check that the score is good enough
    //////////////////////////////////////
    T accuracy;
    EXPECT_EQ(da_svm_score(svm_handle, nsamples, nfeat, X_test.data(), nsamples,
                           y_test.data(), &accuracy),
              da_status_success);
    EXPECT_NEAR(accuracy, target_score, 1e-2);
    std::cout << "Accuracy on the test data: " << accuracy << std::endl;

    da_handle_destroy(&svm_handle);
}
