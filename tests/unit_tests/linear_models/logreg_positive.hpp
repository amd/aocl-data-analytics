/*
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, \OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef LOGREG_POSITIVE_HPP
#define LOGREG_POSITIVE_HPP

#ifndef NO_FORTRAN

#include "../datests_cblas.hh"
#include "../utest_utils.hpp"
#include "aoclda.h"
#include "gtest/gtest.h"
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>

template <class T> struct option_t {
    std::string name = "";
    T value;
};

// Return precision as a string literal to set CSV options
template <typename T> constexpr const char *prec_name();
template <> constexpr const char *prec_name<float>() { return "single"; }
template <> constexpr const char *prec_name<double>() { return "double"; }

template <typename T> constexpr const char *prec_name_float();
template <> constexpr const char *prec_name_float<float>() { return "float"; }
template <> constexpr const char *prec_name_float<double>() { return "double"; }

// Helper to define precision to which we expect the results to match
template <typename T> T expected_precision(T scale = (T)1.0);
template <> double expected_precision<double>(double scale) { return scale * 1.0e-3; }

template <> float expected_precision<float>(float scale) { return scale * 0.5f; }

template <typename T>
void test_logreg_positive(std::string csvname, std::vector<option_t<da_int>> iopts,
                          std::vector<option_t<std::string>> sopts,
                          std::vector<option_t<T>> ropts) {

    // Create main handle and set options
    da_handle linmod_handle = nullptr;
    EXPECT_EQ(da_handle_init<T>(&linmod_handle, da_handle_linmod), da_status_success);
    for (auto &op : sopts)
        EXPECT_EQ(da_options_set_string(linmod_handle, op.name.c_str(), op.value.c_str()),
                  da_status_success);
    for (auto &op : ropts)
        EXPECT_EQ(da_options_set(linmod_handle, op.name.c_str(), op.value),
                  da_status_success);
    for (auto &op : iopts)
        EXPECT_EQ(da_options_set_int(linmod_handle, op.name.c_str(), op.value),
                  da_status_success);
    EXPECT_EQ(da_options_set_string(linmod_handle, "print options", "yes"),
              da_status_success);

    da_int intercept_int;
    EXPECT_EQ(da_options_get_int(linmod_handle, "intercept", &intercept_int),
              da_status_success);
    bool intercept = (bool)intercept_int;

    // No regularization
    T alpha = 0, lambda = 0;
    EXPECT_EQ(da_options_set(linmod_handle, "alpha", alpha), da_status_success);
    EXPECT_EQ(da_options_set(linmod_handle, "lambda", lambda), da_status_success);

    ///////////////
    // Get the data
    ///////////////
    std::string input_data_fname =
        std::string(DATA_DIR) + "/linmod_data/logistic/" + csvname + "_data.csv";
    da_datastore csv_store = nullptr;
    EXPECT_EQ(da_datastore_init(&csv_store), da_status_success);
    EXPECT_EQ(
        da_datastore_options_set_string(csv_store, "datastore precision", prec_name<T>()),
        da_status_success);
    EXPECT_EQ(
        da_datastore_options_set_string(csv_store, "datatype", prec_name_float<T>()),
        da_status_success);
    EXPECT_EQ(da_data_load_from_csv(csv_store, input_data_fname.c_str()),
              da_status_success);

    da_int ncols, nrows;
    EXPECT_EQ(da_data_get_n_cols(csv_store, &ncols), da_status_success);
    EXPECT_EQ(da_data_get_n_rows(csv_store, &nrows), da_status_success);

    // The first ncols-1 columns contain the feature matrix; the last one the response vector.
    // Create the selections in the data store
    EXPECT_EQ(da_data_select_columns(csv_store, "features", 0, ncols - 2),
              da_status_success);
    EXPECT_EQ(da_data_select_columns(csv_store, "response", ncols - 1, ncols - 1),
              da_status_success);

    // Extract the selections
    T *A = nullptr, *b = nullptr;
    A = new T[(ncols - 1) * nrows];
    b = new T[nrows];
    EXPECT_EQ(da_data_extract_selection(csv_store, "features", column_major, A, nrows),
              da_status_success);
    EXPECT_EQ(da_data_extract_selection(csv_store, "response", column_major, b, nrows),
              da_status_success);

    ///////////////////
    // Create the model
    ///////////////////
    EXPECT_EQ(da_linmod_select_model<T>(linmod_handle, linmod_model_logistic),
              da_status_success);
    EXPECT_EQ(da_linmod_define_features(linmod_handle, nrows, ncols - 1, A, nrows, b),
              da_status_success);

    // Compute regression
    EXPECT_EQ(da_linmod_fit<T>(linmod_handle), da_status_success);

    ////////////////////
    // Check the results
    ////////////////////
    // Check the coefficients if reference file is present
    std::string intercept_suff = "";
    if (!intercept)
        intercept_suff = "_noint";
    std::string coef_fname = std::string(DATA_DIR) + "/linmod_data/logistic/" + csvname +
                             intercept_suff + "_coeffs.csv";
    if (FILE *file = fopen(coef_fname.c_str(), "r")) {
        std::fclose(file); // Read the expected coefficients
        T *coef_exp = nullptr;
        da_int mc, nc;
        EXPECT_EQ(
            da_read_csv(csv_store, coef_fname.c_str(), &coef_exp, &mc, &nc, nullptr),
            da_status_success);

        // Read the computed coefficients
        T *coef = new T[nc];
        EXPECT_EQ(
            da_handle_get_result(linmod_handle, da_result::da_linmod_coef, &nc, coef),
            da_status_success);

        // Check coefficients
        EXPECT_ARR_NEAR(nc, coef, coef_exp, expected_precision<T>());
        delete[] coef;
        free(coef_exp);
    }

    // Check predictions if test data is present
    std::string test_set_fname =
        std::string(DATA_DIR) + "/linmod_data/logistic/" + csvname + "_test.csv";
    if (FILE *file = fopen(test_set_fname.c_str(), "r")) {
        std::fclose(file);
        da_datastore test_store = nullptr;
        EXPECT_EQ(da_datastore_init(&test_store), da_status_success);
        EXPECT_EQ(da_datastore_options_set_string(test_store, "datastore precision",
                                                  prec_name<T>()),
                  da_status_success);
        EXPECT_EQ(
            da_datastore_options_set_string(test_store, "datatype", prec_name_float<T>()),
            da_status_success);
        EXPECT_EQ(da_data_load_from_csv(test_store, test_set_fname.c_str()),
                  da_status_success);

        da_int ncols_test, nrows_test;
        EXPECT_EQ(da_data_get_n_cols(test_store, &ncols_test), da_status_success);
        EXPECT_EQ(da_data_get_n_rows(test_store, &nrows_test), da_status_success);

        // The first ncols_test-1 columns contain the feature matrix; the last one the response vector.
        // Create the selections in the data store
        EXPECT_EQ(da_data_select_columns(test_store, "features", 0, ncols_test - 2),
                  da_status_success);
        EXPECT_EQ(da_data_select_columns(test_store, "response", ncols_test - 1,
                                         ncols_test - 1),
                  da_status_success);

        // Extract the selections
        T *A_test = new T[(ncols_test - 1) * nrows_test];
        T *b_test = new T[nrows_test];
        EXPECT_EQ(da_data_extract_selection(test_store, "features", column_major, A_test,
                                            nrows_test),
                  da_status_success);
        EXPECT_EQ(da_data_extract_selection(test_store, "response", column_major, b_test,
                                            nrows_test),
                  da_status_success);

        // Check that the model evaluates the classes correctly
        T *predictions = new T[nrows_test];
        da_linmod_evaluate_model(linmod_handle, nrows_test, ncols_test - 1, A_test,
                                 nrows_test, predictions);
        std::cout << "Predictions: " << std::endl;
        for (da_int i = 0; i < nrows_test; i++)
            std::cout << predictions[i] << " ";
        std::cout << std::endl;
        EXPECT_ARR_NEAR(nrows_test, predictions, b_test, (T)0.1);

        // Free temp memory
        delete[] A_test;
        delete[] b_test;
        delete[] predictions;
        da_datastore_destroy(&test_store);
    }

    //////////////
    // Free memory
    //////////////
    delete[] A;
    delete[] b;
    da_datastore_destroy(&csv_store);
    da_handle_destroy(&linmod_handle);
}

#endif

#endif
