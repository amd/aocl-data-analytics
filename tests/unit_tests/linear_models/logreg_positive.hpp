/*
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Helper to define precision to which we expect the results to match
template <typename T> T expected_precision(T scale = (T)1.0);
template <> double expected_precision<double>(double scale) { return scale * 1.0e-3; }

template <> float expected_precision<float>(float scale) { return scale * 0.5f; }

template <typename T>
void test_logreg_positive(std::string csvname, std::vector<option_t<da_int>> iopts,
                          std::vector<option_t<std::string>> sopts,
                          std::vector<option_t<T>> ropts, bool check_coeff = true,
                          bool check_predict = true) {

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

    // Get storage order to pass to data storage handle
    da_int order_int, order_str_len{64};
    char order_str[64];
    EXPECT_EQ(da_options_get_string_key(linmod_handle, "storage order", order_str,
                                        &order_str_len, &order_int),
              da_status_success);
    da_order order = static_cast<da_order>(order_int);

    // No regularization
    T alpha = 0, lambda = 0;
    EXPECT_EQ(da_options_set(linmod_handle, "alpha", alpha), da_status_success);
    EXPECT_EQ(da_options_set(linmod_handle, "lambda", lambda), da_status_success);

    ///////////////
    // Get the data
    ///////////////
    std::string input_data_fname =
        std::string(DATA_DIR) + "/linmod_data/logistic/" + csvname + "_data.csv";
    std::vector<T> train_data;
    da_int nrows, ncols;
    ASSERT_TRUE(da_test::read_csv_data(input_data_fname, train_data, nrows, ncols, order))
        << "Failed to read training data: " << input_data_fname;

    // The first ncols-1 columns contain the feature matrix; the last one the response vector.
    da_int nfeat = ncols - 1;
    da_int nsamples = nrows;
    T *A = train_data.data();
    da_int ldA = (order == da_order::column_major) ? nrows : ncols;
    // b is the response (last) column extracted into a contiguous vector
    std::vector<T> b(nsamples);
    for (da_int i = 0; i < nsamples; ++i)
        b[i] = (order == da_order::column_major) ? train_data[nfeat * nsamples + i]
                                                 : train_data[i * ncols + nfeat];

    ///////////////////
    // Create the model
    ///////////////////
    EXPECT_EQ(da_linmod_select_model<T>(linmod_handle, linmod_model_logistic),
              da_status_success);
    EXPECT_EQ(
        da_linmod_define_features(linmod_handle, nrows, ncols - 1, A, ldA, b.data()),
        da_status_success);

    // Check that the model is not trained before fitting
    da_int dim = 1;
    da_int trained = 0;
    EXPECT_EQ(da_handle_get_result(linmod_handle, da_result::da_trained, &dim, &trained),
              da_status_success);
    EXPECT_EQ(trained, 0);

    // Compute regression
    EXPECT_EQ(da_linmod_fit<T>(linmod_handle), da_status_success);

    ////////////////////
    // Check the results
    ////////////////////
    // check that the model is trained
    EXPECT_EQ(da_handle_get_result(linmod_handle, da_result::da_trained, &dim, &trained),
              da_status_success);
    EXPECT_EQ(trained, 1);
    // Check the coefficients if requested
    std::string intercept_suff = "";
    if (!intercept)
        intercept_suff = "_noint";
    if (check_coeff) {
        std::string coef_fname = std::string(DATA_DIR) + "/linmod_data/logistic/" +
                                 csvname + intercept_suff + "_coeffs.csv";
        // Read the expected coefficients
        std::vector<T> coef_exp;
        da_int mc{0}, nc{0};
        ASSERT_TRUE(da_test::read_csv_data(coef_fname, coef_exp, mc, nc))
            << "Failed to read expected coefficients: " << coef_fname;

        // Read the computed coefficients
        std::vector<T> coef(nc);
        EXPECT_EQ(da_handle_get_result(linmod_handle, da_result::da_linmod_coef, &nc,
                                       coef.data()),
                  da_status_success);

        // Check coefficients
        EXPECT_ARR_NEAR(nc, coef, coef_exp, expected_precision<T>());
    }

    // Check score and predictions
    T *b_pred = new T[nrows];
    T score{0};
    EXPECT_EQ(da_linmod_evaluate_model(linmod_handle, nrows, ncols - 1, A, ldA, b_pred,
                                       b.data(), &score),
              da_status_success);
    T score_pred{0};
    for (da_int i = 0; i < nrows; ++i) {
        score_pred += b[i] == b_pred[i];
    }
    score_pred /= nrows;
    EXPECT_NEAR(score, score_pred, 2 * std::numeric_limits<T>::epsilon());
    delete[] b_pred;

    // Check predictions if requested
    if (check_predict) {
        std::string test_set_fname =
            std::string(DATA_DIR) + "/linmod_data/logistic/" + csvname + "_test.csv";
        std::vector<T> test_data;
        da_int nrows_test, ncols_test;
        ASSERT_TRUE(da_test::read_csv_data(test_set_fname, test_data, nrows_test,
                                           ncols_test, order))
            << "Failed to read test data: " << test_set_fname;

        // The first ncols_test-1 columns contain the feature matrix; the last one the response vector.
        da_int nfeat_test = ncols_test - 1;
        da_int nsamples_test = nrows_test;
        T *A_test = test_data.data();
        da_int ldA_test = (order == da_order::column_major) ? nrows_test : ncols_test;
        std::vector<T> b_test(nsamples_test);
        for (da_int i = 0; i < nsamples_test; ++i)
            b_test[i] = (order == da_order::column_major)
                            ? test_data[nfeat_test * nsamples_test + i]
                            : test_data[i * ncols_test + nfeat_test];

        // Check that the model evaluates the classes correctly
        std::vector<T> predictions(nsamples_test);
        da_linmod_evaluate_model(linmod_handle, nsamples_test, nfeat_test, A_test,
                                 ldA_test, predictions.data());
        std::cout << "Predictions: " << std::endl;
        for (da_int i = 0; i < nsamples_test; i++)
            std::cout << predictions[i] << " ";
        std::cout << std::endl;
        EXPECT_ARR_NEAR(nsamples_test, predictions, b_test, (T)0.1);
    }

    //////////////
    // Free memory
    //////////////
    da_handle_destroy(&linmod_handle);
}

#endif

#endif
