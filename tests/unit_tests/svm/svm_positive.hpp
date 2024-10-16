/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

// Helper to define precision to which we expect the results to match
template <typename T> T expected_precision(T scale = (T)1.0);
template <> double expected_precision<double>(double scale) {
    return da_numeric::tolerance<double>::safe_tol() * scale;
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
    kernel_str.resize(lvalue - 1);
    ////////////////////////
    // Get the training data
    ////////////////////////
    std::string input_data_fname =
        std::string(DATA_DIR) + "/svm_data/" + csvname + "_train.csv";
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
    std::vector<T> y(nsamples);
    EXPECT_EQ(da_data_extract_selection(csv_store, "features", column_major, X.data(),
                                        nsamples),
              da_status_success);
    EXPECT_EQ(da_data_extract_selection(csv_store, "response", column_major, y.data(),
                                        nsamples),
              da_status_success);

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
    T *coef_exp{nullptr};
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
    da_int n_rows{0}, n_cols{0};
    if (FILE *file = fopen(coef_fname.c_str(), "r")) {
        // read the expected coefficients
        fclose(file);
        EXPECT_EQ(da_read_csv(csv_store, coef_fname.c_str(), &coef_exp, &n_rows, &n_cols,
                              nullptr),
                  da_status_success);
        EXPECT_EQ(n_cols, dim) << "Number of coefficients to check does not match";
        // Check coefficients
        EXPECT_ARR_NEAR(dim, dual_coeffs, coef_exp,
                        expected_precision<T>(check_tol_scale))
            << "Checking coefficients (solution)";
        free(coef_exp);
    } else {
        da_handle_destroy(&svm_handle);
        da_datastore_destroy(&csv_store);
        FAIL() << "Check of coefficients was requested but the solution file "
               << coef_fname << " could not be opened.";
    }
    da_datastore_destroy(&csv_store);

    ////////////////////////
    // Get the test data
    ////////////////////////
    input_data_fname = std::string(DATA_DIR) + "/svm_data/" + csvname + "_test.csv";
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
    std::vector<T> y_test(nsamples);
    EXPECT_EQ(da_data_extract_selection(csv_store, "features", column_major,
                                        X_test.data(), nsamples),
              da_status_success);
    EXPECT_EQ(da_data_extract_selection(csv_store, "response", column_major,
                                        y_test.data(), nsamples),
              da_status_success);

    ////////////////////////////////////////////////
    // Check decision function (only classification)
    ////////////////////////////////////////////////
    if (model == svc || model == nusvc) {
        T *dec_exp{nullptr};
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
        n_rows = 0, n_cols = 0;
        if (FILE *file = fopen(dec_fname.c_str(), "r")) {
            // read the expected decision function values
            fclose(file);
            EXPECT_EQ(da_read_csv(csv_store, dec_fname.c_str(), &dec_exp, &n_rows,
                                  &n_cols, nullptr),
                      da_status_success);
            EXPECT_EQ(n_cols, dim) << "Number of coefficients to check does not match";
            // Check decision function values
            EXPECT_ARR_NEAR(dim, decision_values, dec_exp,
                            expected_precision<T>(check_tol_scale))
                << "Checking decision function values (solution)";
            free(dec_exp);
        } else {
            da_handle_destroy(&svm_handle);
            da_datastore_destroy(&csv_store);
            FAIL() << "Check of decision function values was requested but the solution "
                      "file "
                   << dec_fname << " could not be opened.";
        }
    }
    //////////////////////////
    // Check prediction
    //////////////////////////
    T *pred_exp{nullptr};
    std::vector<T> predictions(nsamples);
    EXPECT_EQ(da_svm_predict(svm_handle, nsamples, nfeat, X_test.data(), nsamples,
                             predictions.data()),
              da_status_success);
    std::string pred_fname = std::string(DATA_DIR) + "/svm_data/" +
                             get_model_name(model) + "/" + csvname + "_" + kernel_str +
                             "_pred.csv";
    n_rows = 0, n_cols = 0;
    if (FILE *file = fopen(pred_fname.c_str(), "r")) {
        // read the expected test labels
        fclose(file);
        EXPECT_EQ(da_read_csv(csv_store, pred_fname.c_str(), &pred_exp, &n_rows, &n_cols,
                              nullptr),
                  da_status_success);
        EXPECT_EQ(n_cols, nsamples) << "Number of coefficients to check does not match";
        // Check test labels
        EXPECT_ARR_NEAR(nsamples, predictions, pred_exp,
                        expected_precision<T>(check_tol_scale))
            << "Checking test labels (solution)";
        free(pred_exp);
    } else {
        da_handle_destroy(&svm_handle);
        da_datastore_destroy(&csv_store);
        FAIL() << "Check of test labels was requested but the solution file "
               << pred_fname << " could not be opened.";
    }

    da_datastore_destroy(&csv_store);

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
