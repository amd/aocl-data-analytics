/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <fstream>
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

template <typename T> constexpr const char *type_opt_name();
template <> constexpr const char *type_opt_name<float>() { return "float"; }
template <> constexpr const char *type_opt_name<double>() { return "double"; }

// Helper to define precision to which we expect the results match
template <typename T> T expected_precision(T scale = (T)1.0);
template <> double expected_precision<double>(double scale) { return scale * 1.0e-3; }

template <> float expected_precision<float>(float scale) { return scale * 0.5f; }

template <typename T>
void test_linreg_positive(std::string csvname, std::vector<option_t<da_int>> iopts,
                          std::vector<option_t<std::string>> sopts,
                          std::vector<option_t<T>> ropts, bool check_coeff,
                          T check_tol_scale, bool double_call = false) {

    // Double call: calls a second time the solver with the solution point and
    // makes sure no iterations are performed.

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

    ///////////////
    // Get the data
    ///////////////
    std::string input_data_fname = std::string(DATA_DIR) + "/" + csvname + "_data.csv";
    da_datastore csv_store = nullptr;
    EXPECT_EQ(da_datastore_init(&csv_store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(csv_store, "CSV datastore precision",
                                              prec_name<T>()),
              da_status_success);
    EXPECT_EQ(
        da_datastore_options_set_string(csv_store, "CSV datatype", type_opt_name<T>()),
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
    EXPECT_EQ(da_data_extract_selection(csv_store, "features", A, nrows),
              da_status_success);
    EXPECT_EQ(da_data_extract_selection(csv_store, "response", b, nrows),
              da_status_success);

    ///////////////////
    // Create the model
    ///////////////////
    EXPECT_EQ(da_linmod_select_model<T>(linmod_handle, linmod_model_mse),
              da_status_success);
    EXPECT_EQ(da_linmod_define_features(linmod_handle, nrows, ncols - 1, A, b),
              da_status_success);

    // compute regression
    EXPECT_EQ(da_linmod_fit<T>(linmod_handle), da_status_success);

    ////////////////////
    // Check the results
    ////////////////////
    // Check the coefficients if reference file is present
    T *coef_exp = nullptr;
    da_int ncoef = intercept ? ncols : ncols - 1;
    std::vector<T> coef(ncoef, -9.87654321);
    // read the computed coefficients
    EXPECT_EQ(da_handle_get_result(linmod_handle, da_result::da_linmod_coef, &ncoef,
                                   coef.data()),
              da_status_success);

    da_int mc{0}, nc{0};
    std::string intercept_suff = "";
    if (!intercept)
        intercept_suff = "_noint";
    std::string coef_fname =
        std::string(DATA_DIR) + "/" + csvname + intercept_suff + "_coeffs.csv";
    if (FILE *file = fopen(coef_fname.c_str(), "r")) {
        // read the expected coefficients
        std::fclose(file);
        EXPECT_EQ(
            da_read_csv(csv_store, coef_fname.c_str(), &coef_exp, &mc, &nc, nullptr),
            da_status_success);
        EXPECT_EQ(nc, ncoef) << "Number of coefficients to check does not match";
        // Check coefficients
        EXPECT_ARR_NEAR(nc, coef, coef_exp, expected_precision<T>(check_tol_scale));
        free(coef_exp);
    } else if (check_coeff) {
        FAIL() << "Check of coefficients was requested but the solution file "
               << coef_fname << " could not be opened.";
    }

    // Check that info contains the correct values
    da_int linfo = 100;
    T info[100];
    std::vector<T> info_exp(100, T(0));
    info_exp[0] = (T)(ncols - 1);
    info_exp[1] = (T)nrows;
    info_exp[2] = intercept ? (T)(ncols - 1) : ncols;
    info_exp[3] = intercept ? (T)1 : (T)0;
    EXPECT_EQ(da_options_get(linmod_handle, "alpha", &info_exp[4]), da_status_success);
    EXPECT_EQ(da_options_get(linmod_handle, "lambda", &info_exp[5]), da_status_success);
    EXPECT_EQ(da_handle_get_result(linmod_handle, da_result::da_rinfo, &linfo, info),
              da_status_success);

    // TODO compare info
    da_int lsolver{15};
    char solver[15];
    EXPECT_EQ(da_options_set(linmod_handle, "lambda", info_exp[5]), da_status_success);

    // EXPECT_EQ(da_options_get(linmod_handle, "optim method", &solver[0], &lsolver), da_status_success);
    // std::string sol(solver);
    // if (double_call && (solver == "coord"s || solver == )){
    // Problem has been solved once and coef holds the solution
    // set any option to retriger
    // EXPECT_EQ(da_linmod_fit_start<T>(linmod_handle, nc, coef.data()), da_status_success);
    // EXPECT_EQ(da_handle_get_result(linmod_handle, da_result::da_rinfo, &linfo, info),
    //       da_status_success);
    // todo compare info
    // }

    //////////////
    // Free memory
    //////////////
    delete[] A;
    delete[] b;
    da_datastore_destroy(&csv_store);
    da_handle_destroy(&linmod_handle);
}
