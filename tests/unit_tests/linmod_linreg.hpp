/*
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclda.h"
#include "da_cblas.hh"
#include "utest_utils.hpp"
#include "gtest/gtest.h"
#include <filesystem>
#include <iostream>
#include <string>
#include <type_traits>

template <class T> struct option_t {
    std::string name = "";
    T value;
};

// return precision as a string litteral to set CSV options
template <typename T> constexpr const char *prec_name();
template <> constexpr const char *prec_name<float>() { return "single"; }
template <> constexpr const char *prec_name<double>() { return "double"; }

// Helper to define precision to which we expect the results match
template <typename T> T expected_precision(T scale = (T)1.0);
template <> double expected_precision<double>(double scale) { return scale * 1.0e-3; }

template <> float expected_precision<float>(float scale) { return scale * 0.5f; }

template <typename T>
void test_linreg_positive(std::string csvname, std::vector<option_t<da_int>> iopts,
                          std::vector<option_t<std::string>> sopts,
                          std::vector<option_t<T>> ropts) {

    // Create main handle and set options
    da_handle linmod_handle = nullptr;
    EXPECT_EQ(da_handle_init<T>(&linmod_handle, da_handle_linmod), da_status_success);
    for (auto &op : sopts)
        EXPECT_EQ(da_options_set_string(linmod_handle, op.name.c_str(), op.value.c_str()),
                  da_status_success);
    for (auto &op : ropts)
        EXPECT_EQ(da_options_set_real(linmod_handle, op.name.c_str(), op.value),
                  da_status_success);
    for (auto &op : iopts)
        EXPECT_EQ(da_options_set_int(linmod_handle, op.name.c_str(), op.value),
                  da_status_success);
    EXPECT_EQ(da_options_set_string(linmod_handle, "print options", "yes"),
              da_status_success);

    da_int intercept_int;
    EXPECT_EQ(da_options_get_int(linmod_handle, "linmod intercept", &intercept_int),
              da_status_success);
    bool intercept = (bool)intercept_int;

    ///////////////
    // Get the data
    ///////////////
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

    // Extract the selections
    T *A = nullptr, *b = nullptr;
    A = new T[(ncols - 1) * nrows];
    b = new T[nrows];
    EXPECT_EQ(da_data_extract_selection(csv_store, "features", nrows, A),
              da_status_success);
    EXPECT_EQ(da_data_extract_selection(csv_store, "response", nrows, b),
              da_status_success);

    ///////////////////
    // Create the model
    ///////////////////
    EXPECT_EQ(da_linmod_select_model<T>(linmod_handle, linmod_model_mse),
              da_status_success);
    EXPECT_EQ(da_linreg_define_features(linmod_handle, nrows, ncols - 1, A, b),
              da_status_success);

    // compute regression
    EXPECT_EQ(da_linreg_fit<T>(linmod_handle), da_status_success);

    ////////////////////
    // Check the results
    ////////////////////
    // Check the coefficients if reference file is present
    std::string intercept_suff = "";
    if (!intercept)
        intercept_suff = "_noint";
    std::string coef_fname =
        std::string(DATA_DIR) + "/" + csvname + intercept_suff + "_coeffs.csv";
    if (std::filesystem::exists(coef_fname)) {
        // read the expected coefficients
        T *coef_exp = nullptr;
        da_int mc, nc;
        EXPECT_EQ(
            da_read_csv(csv_store, coef_fname.c_str(), &coef_exp, &mc, &nc, nullptr),
            da_status_success);

        // read the computed coefficients
        T *coef = new T[nc];
        EXPECT_EQ(
            da_handle_get_result(linmod_handle, da_result::da_linmod_coeff, &nc, coef),
            da_status_success);

        // Check coefficients
        EXPECT_ARR_NEAR(nc, coef, coef_exp, expected_precision<T>());
        delete[] coef;
        free(coef_exp);
    }

    // Check that rinfo contains the correct values
    da_int n_rinfo = 100;
    T rinfo[100];
    T rinfo_exp[100];
    for (da_int i = 0; i < 100; i++)
        rinfo_exp[i] = (T)0.0;
    rinfo_exp[0] = (T)(ncols - 1);
    rinfo_exp[1] = (T)nrows;
    rinfo_exp[2] = intercept ? (T)(ncols - 1) : ncols;
    rinfo_exp[3] = intercept ? (T)1 : (T)0;
    EXPECT_EQ(da_options_get_real(linmod_handle, "linmod alpha", &rinfo_exp[4]),
              da_status_success);
    EXPECT_EQ(da_options_get_real(linmod_handle, "linmod lambda", &rinfo_exp[4]),
              da_status_success);
    EXPECT_EQ(da_handle_get_result(linmod_handle, da_result::da_rinfo, &n_rinfo, rinfo),
              da_status_success);

    //////////////
    // Free memory
    //////////////
    delete[] A;
    delete[] b;
    da_datastore_destroy(&csv_store);
    da_handle_destroy(&linmod_handle);
}
