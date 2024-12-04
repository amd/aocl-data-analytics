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
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef LINREG_POSITIVE_HPP
#define LINREG_POSITIVE_HPP

// Deal with some Windows compilation issues regarding max/min macros
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "linmod_functions.hpp"
#include "gtest/gtest.h"
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
using namespace std::literals::string_literals;

template <typename T>
void test_linreg_positive(std::string csvname, std::vector<option_t<da_int>> iopts,
                          std::vector<option_t<std::string>> sopts,
                          std::vector<option_t<T>> ropts, bool check_coeff,
                          bool check_predict, T check_tol_scale) {

    // get template instantiation type (either single or double)
    const bool single = std::is_same_v<T, float>; // otherwise assume double

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
    std::string input_data_fname =
        std::string(DATA_DIR) + "/linmod_data/linear_reg/" + csvname + "_data.csv";
    da_datastore csv_store = nullptr;
    EXPECT_EQ(da_datastore_init(&csv_store), da_status_success);
    EXPECT_EQ(
        da_datastore_options_set_string(csv_store, "datastore precision", prec_name<T>()),
        da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(csv_store, "datatype", type_opt_name<T>()),
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

    da_int nfeat = ncols - 1;
    da_int nsamples = nrows;
    // Extract the selections
    T *A = nullptr, *b = nullptr;
    A = new T[nfeat * nsamples];
    b = new T[nsamples];
    EXPECT_EQ(da_data_extract_selection(csv_store, "features", column_major, A, nsamples),
              da_status_success);
    EXPECT_EQ(da_data_extract_selection(csv_store, "response", column_major, b, nsamples),
              da_status_success);

    ///////////////////
    // Create the model
    ///////////////////
    EXPECT_EQ(da_linmod_select_model<T>(linmod_handle, linmod_model_mse),
              da_status_success);
    EXPECT_EQ(da_linmod_define_features(linmod_handle, nsamples, nfeat, A, b),
              da_status_success);

    // Compute regression
    EXPECT_EQ(da_linmod_fit<T>(linmod_handle), da_status_success);

    // Check that info contains the correct values
    da_int linfo = 100;
    T info[100];
    EXPECT_EQ(da_handle_get_result(linmod_handle, da_result::da_rinfo, &linfo, info),
              da_status_success);

    char cmethod[100];
    da_int lmethod = 100;
    EXPECT_EQ(da_options_get(linmod_handle, "optim method", cmethod, &lmethod),
              da_status_success);
    std::string method{cmethod};
    EXPECT_STRNE(method.c_str(), "auto");
    bool infochk = (method == "lbfgs"s || method == "coord"s || method == "bfgs"s ||
                    method == "lbfgsb"s);

    if (infochk) { // Assumes that initial iterate is not solution and that problem does not have residual=0 at x=0
        // info_objective is checked later
        const T iter = info[da_optim_info_t::info_iter];
        // lbfgs timer may be broken for windows
#if defined(WIN32)
        EXPECT_GE(info[da_optim_info_t::info_time], 0);
#else
        EXPECT_GT(info[da_optim_info_t::info_time], 0);
#endif
        EXPECT_GT(info[da_optim_info_t::info_nevalf], 0);
        if (method == "coord"s) {
            EXPECT_GT(iter, 1);
            EXPECT_GE(info[da_optim_info_t::info_inorm], 0);
            EXPECT_GE(info[da_optim_info_t::info_inorm_init], 0);
            EXPECT_GT(info[da_optim_info_t::info_nevalf], 0);
            EXPECT_GE(info[da_optim_info_t::info_ncheap], std::max(T(1), iter - T(1)));
        } else {
            EXPECT_GT(iter, 0);
            EXPECT_GT(info[da_optim_info_t::info_nevalf], 0);
            EXPECT_GE(info[da_optim_info_t::info_grad_norm], 0);
        }
    }

    ////////////////////
    // Check the results
    ////////////////////
    // Check the coefficients if reference file is present
    T *coef_exp{nullptr};
    da_int ncoef = intercept ? nfeat + 1 : nfeat;
    std::vector<T> coef(ncoef, -9.87654321);
    // read the computed coefficients
    EXPECT_EQ(da_handle_get_result(linmod_handle, da_result::da_linmod_coef, &ncoef,
                                   coef.data()),
              da_status_success);

    da_int mc{0}, nc{0};
    std::string intercept_suff = "";
    if (!intercept)
        intercept_suff = "_noint";
    std::string coef_fname = std::string(DATA_DIR) + "/linmod_data/linear_reg/" +
                             csvname + intercept_suff + "_coeffs.csv";
    if (FILE *file = fopen(coef_fname.c_str(), "r")) {
        // read the expected coefficients
        fclose(file);
        EXPECT_EQ(
            da_read_csv(csv_store, coef_fname.c_str(), &coef_exp, &mc, &nc, nullptr),
            da_status_success);
        EXPECT_EQ(nc, ncoef) << "Number of coefficients to check does not match";
        // Check coefficients
        EXPECT_ARR_NEAR(nc, coef, coef_exp, expected_precision<T>(check_tol_scale))
            << "Checking coefficients (solution)";
        free(coef_exp);
    } else if (check_coeff) {
        FAIL() << "Check of coefficients was requested but the solution file "
               << coef_fname << " could not be opened.";
    }

    da_datastore_destroy(&csv_store);

    // Predict

    // Check that solver found the same solution
    // A is the training set and b is the predicted y of the trained model:
    // beta = y ~ x, then b = predict(beta, x)
    std::string solution_fname = std::string(DATA_DIR) + "/linmod_data/linear_reg/" +
                                 csvname + intercept_suff + "_solution.csv";
    if (FILE *file = fopen(solution_fname.c_str(), "r")) {
        std::fclose(file);
        // read the expected prediction
        da_datastore sol_store = nullptr;
        EXPECT_EQ(da_datastore_init(&sol_store), da_status_success);
        EXPECT_EQ(da_datastore_options_set_string(sol_store, "datastore precision",
                                                  prec_name<T>()),
                  da_status_success);
        EXPECT_EQ(
            da_datastore_options_set_string(sol_store, "datatype", type_opt_name<T>()),
            da_status_success);
        EXPECT_EQ(da_data_load_from_csv(sol_store, solution_fname.c_str()),
                  da_status_success);

        da_int scols, srows;
        EXPECT_EQ(da_data_get_n_cols(sol_store, &scols), da_status_success);
        EXPECT_EQ(da_data_get_n_rows(sol_store, &srows), da_status_success);
        EXPECT_EQ(scols, nsamples);
        EXPECT_EQ(srows, 1);

        // The first ncols-1 columns contain the feature matrix; the last one the response vector.
        // Create the selections in the data store
        EXPECT_EQ(da_data_select_columns(sol_store, "solution", 0, nsamples - 1),
                  da_status_success);
        std::vector<T> sol(nsamples);
        T *sol_exp{nullptr};
        T loss{T(-1)};
        EXPECT_EQ(da_read_csv(sol_store, solution_fname.c_str(), &sol_exp, &srows, &scols,
                              nullptr),
                  da_status_success);
        EXPECT_EQ(da_linmod_evaluate_model(linmod_handle, nsamples, nfeat, A, sol.data(),
                                           b, &loss),
                  da_status_success);

        // Check predictions
        EXPECT_ARR_NEAR(nsamples, sol, sol_exp, expected_precision<T>(check_tol_scale));

        // Check loss with info from solver (objective function)
        if (infochk) {
            if (single) {
                // EXPECT_FLOAT_EQ(loss, info[da_optim_info_t::info_objective])
                EXPECT_NEAR(loss, info[da_optim_info_t::info_objective], 1.0e-5)
                    << "Objective function (LOSS) mismatch!";
            } else {
                // EXPECT_DOUBLE_EQ(loss, info[da_optim_info_t::info_objective])
                EXPECT_NEAR(loss, info[da_optim_info_t::info_objective], 1.0e-12)
                    << "Objective function (LOSS) mismatch!";
            }
        }
        free(sol_exp);
        sol_exp = nullptr;
        da_datastore_destroy(&sol_store);
    } else if (check_predict) {
        FAIL() << "Check of predictions was requested but the data file "
               << solution_fname << " could not be opened.";
    }

    delete[] b;
    b = nullptr;

    delete[] A;
    A = nullptr;

    // Check predictions on a random data (A) not used for training
    // A is the new data set and b is the predicted y of the trained model:
    // beta = y ~ x, then b = predict(beta, newx)
    std::string predict_fname = std::string(DATA_DIR) + "/linmod_data/linear_reg/" +
                                csvname + intercept_suff + "_predict_data.csv";
    if (FILE *file = fopen(predict_fname.c_str(), "r")) {
        fclose(file);

        EXPECT_EQ(da_datastore_init(&csv_store), da_status_success);
        EXPECT_EQ(da_datastore_options_set_string(csv_store, "datastore precision",
                                                  prec_name<T>()),
                  da_status_success);
        EXPECT_EQ(
            da_datastore_options_set_string(csv_store, "datatype", type_opt_name<T>()),
            da_status_success);

        EXPECT_EQ(da_data_load_from_csv(csv_store, predict_fname.c_str()),
                  da_status_success);

        EXPECT_EQ(da_data_get_n_cols(csv_store, &ncols), da_status_success);
        EXPECT_EQ(da_data_get_n_rows(csv_store, &nrows), da_status_success);

        // The first ncols-1 columns contain the feature matrix; the last one the response vector.
        // Create the selections in the data store
        EXPECT_EQ(da_data_select_columns(csv_store, "features", 0, ncols - 2),
                  da_status_success);
        EXPECT_EQ(da_data_select_columns(csv_store, "response", ncols - 1, ncols - 1),
                  da_status_success);

        nfeat = ncols - 1;
        nsamples = nrows;
        // Extract the selections
        A = new T[nfeat * nsamples];
        b = new T[nsamples];
        EXPECT_EQ(
            da_data_extract_selection(csv_store, "features", column_major, A, nsamples),
            da_status_success);
        EXPECT_EQ(
            da_data_extract_selection(csv_store, "response", column_major, b, nsamples),
            da_status_success);

        da_datastore_destroy(&csv_store);

        // Check that solver found the same solution
        // A is the training set and b is the predicted y of the trained model:
        // beta = y ~ x, then b = predict(beta, x)
        std::vector<T> pred(nsamples);
        EXPECT_EQ(
            da_linmod_evaluate_model(linmod_handle, nsamples, nfeat, A, pred.data()),
            da_status_success);
        EXPECT_ARR_NEAR(nsamples, pred.data(), b, expected_precision<T>(check_tol_scale));
        T loss;
        EXPECT_EQ(da_linmod_evaluate_model(linmod_handle, nsamples, nfeat, A, pred.data(),
                                           b, &loss),
                  da_status_success);

        delete[] A;
        A = nullptr;

        delete[] b;
        b = nullptr;

    } else if (check_predict) {
        FAIL() << "Check of predictions was requested but the data file " << predict_fname
               << " could not be opened.";
    }

    //////////////
    // Free memory
    //////////////
    da_handle_destroy(&linmod_handle);
}

#endif
