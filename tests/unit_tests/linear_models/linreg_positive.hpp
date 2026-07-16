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

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "linmod_functions.hpp"
#include "gtest/gtest.h"
#include <iostream>
#include <string>
#include <type_traits>
using namespace std::literals::string_literals;

const double infd{std::numeric_limits<double>::infinity()};
const float infs{std::numeric_limits<float>::infinity()};

typedef struct linregParam_t {
    std::string test_name; // name of the ctest test
    std::string data_name; // name of the files to read in
    std::vector<option_t<da_int>> iopts;
    std::vector<option_t<std::string>> sopts;
    std::vector<option_t<float>> fopts;
    std::vector<option_t<double>> dopts;
    // check the solution
    bool check_coeff{true};
    // check the prediction
    bool check_predict{true};
    // scale to pass to expected_precision<T>(T scale=1.0)
    float check_tol_scale{1.0f};
    // check dual-gap [0] = float [1] = double
    float dual_gap[2]{-1.0f, -1.0f};
    bool initial_guess{false}; // use initial guess for the coefficients?
} linregParam;

template <typename T>
void test_linreg_positive(std::string csvname, std::vector<option_t<da_int>> iopts,
                          std::vector<option_t<std::string>> sopts,
                          std::vector<option_t<T>> ropts, bool check_coeff,
                          bool check_predict, T check_tol_scale, T dual_gap,
                          bool initial_guess = false) {

    // get template instantiation type (either single or double)
    const bool single = std::is_same_v<T, float>; // otherwise assume double
    const da_int debug{0};
    if (debug) {
        EXPECT_EQ(da_debug_set("debug", std::to_string(debug).c_str()), da_status_success)
            << "Failed to set debug level to " << debug;
    }

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

    // check to see if we are using initial guess
    std::vector<T> coef0;
    da_int ncoef0{0};
    if (initial_guess) {
        std::string intercept_suff = "";
        if (!intercept)
            intercept_suff = "_noint";
        std::string coef_fname = std::string(DATA_DIR) + "/linmod_data/linear_reg/" +
                                 csvname + intercept_suff + "_coeffs0.csv";
        da_int mc0{0};
        ASSERT_TRUE(da_test::read_csv_data(coef_fname, coef0, mc0, ncoef0))
            << "Initial guess coefficients was requested but the file " << coef_fname
            << " could not be read.";
    }

    std::string input_data_fname =
        std::string(DATA_DIR) + "/linmod_data/linear_reg/" + csvname + "_data.csv";
    std::vector<T> train_data;
    da_int nrows, ncols;
    ASSERT_TRUE(da_test::read_csv_data(input_data_fname, train_data, nrows, ncols))
        << "Failed to read training data: " << input_data_fname;
    // The first ncols-1 columns contain the feature matrix; the last one the response vector.
    da_int nfeat = ncols - 1;
    da_int nsamples = nrows;
    std::vector<T> A(train_data.begin(), train_data.begin() + nfeat * nsamples);
    std::vector<T> b(train_data.begin() + nfeat * nsamples, train_data.end());

    ///////////////////
    // Create the model
    ///////////////////
    EXPECT_EQ(da_linmod_select_model<T>(linmod_handle, linmod_model_mse),
              da_status_success);
    EXPECT_EQ(da_linmod_define_features(linmod_handle, nsamples, nfeat, A.data(),
                                        nsamples, b.data()),
              da_status_success);

    // if coef0 is provided, set it as the initial guess
    if (initial_guess) {
        EXPECT_EQ(da_linmod_fit_start(linmod_handle, ncoef0, coef0.data()),
                  da_status_success);
    } else {
        // Compute regression
        EXPECT_EQ(da_linmod_fit<T>(linmod_handle), da_status_success);
    }

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
        const T iter = info[da_linmod_info_t::linmod_info_iter];
        // lbfgs timer may be broken for windows
        /* Deactivate timer check as some problems return 0 time
#if defined(WIN32)
        EXPECT_GE(info[da_linmod_info_t::linmod_info_time], 0);
#else
        EXPECT_GT(info[da_linmod_info_t::linmod_info_time], 0);
#endif
        */
        EXPECT_GT(info[da_linmod_info_t::linmod_info_nevalf], 0);
        if (method == "coord"s) {
            EXPECT_GE(info[da_linmod_info_t::linmod_info_inorm], 0);
            EXPECT_GE(info[da_linmod_info_t::linmod_info_inorm_init], 0);
            EXPECT_GE(info[da_linmod_info_t::linmod_info_ncheap],
                      std::max(T(1), iter - T(1)));
            if (dual_gap >= T(0)) {
                EXPECT_LT(info[da_linmod_info_t::linmod_info_optim], dual_gap)
                    << "Coord: Dual gap size unexpectedly LARGE!";
            }
            // make sure the dual gap is valid...
            // relax condition for being a metric
            const T dual_relax_zero{50 * std::numeric_limits<T>::epsilon()};
            EXPECT_GE(info[da_linmod_info_t::linmod_info_optim], -dual_relax_zero);
        } else {
            EXPECT_GE(info[da_linmod_info_t::linmod_info_grad_norm], 0);
        }
        EXPECT_GT(iter, 0);
    }

    ////////////////////
    // Check the results
    ////////////////////
    // Check the coefficients if reference file is present
    da_int ncoef = intercept ? nfeat + 1 : nfeat;
    std::vector<T> coef(ncoef, -9.87654321);
    // read the computed coefficients
    EXPECT_EQ(da_handle_get_result(linmod_handle, da_result::da_linmod_coef, &ncoef,
                                   coef.data()),
              da_status_success);

    std::string intercept_suff = "";
    if (!intercept)
        intercept_suff = "_noint";

    if (check_coeff) {
        std::string coef_fname = std::string(DATA_DIR) + "/linmod_data/linear_reg/" +
                                 csvname + intercept_suff + "_coeffs.csv";
        std::vector<T> coef_exp;
        da_int mc{0}, nc{0};
        ASSERT_TRUE(da_test::read_csv_data(coef_fname, coef_exp, mc, nc))
            << "Failed to read expected coefficients: " << coef_fname;
        EXPECT_EQ(nc, ncoef) << "Number of coefficients to check does not match";
        EXPECT_ARR_NEAR(nc, coef, coef_exp, expected_precision<T>(check_tol_scale))
            << "Checking coefficients (solution)";
    }

    if (check_predict) {
        // Check that solver found the same solution
        // A is the training set and b is the predicted y of the trained model:
        // beta = y ~ x, then b = predict(beta, x)
        std::string solution_fname = std::string(DATA_DIR) + "/linmod_data/linear_reg/" +
                                     csvname + intercept_suff + "_solution.csv";
        std::vector<T> sol_exp;
        da_int srows{0}, scols{0};
        ASSERT_TRUE(da_test::read_csv_data(solution_fname, sol_exp, srows, scols))
            << "Failed to read solution data: " << solution_fname;
        EXPECT_EQ(scols, nsamples);
        EXPECT_EQ(srows, 1);

        std::vector<T> sol(nsamples);
        T loss{T(-1)};
        EXPECT_EQ(da_linmod_evaluate_model(linmod_handle, nsamples, nfeat, A.data(),
                                           nsamples, sol.data(), b.data(), &loss),
                  da_status_success);

        EXPECT_ARR_NEAR(nsamples, sol, sol_exp, expected_precision<T>(check_tol_scale));

        // Check loss with info from solver (objective function)
        if (infochk) {
            if (single) {
                EXPECT_NEAR(loss, info[da_linmod_info_t::linmod_info_objective], 1.0e-5)
                    << "Objective function (LOSS) mismatch!";
            } else {
                EXPECT_NEAR(loss, info[da_linmod_info_t::linmod_info_objective], 1.0e-12)
                    << "Objective function (LOSS) mismatch!";
            }
        }

        // Check predictions on a random data (A) not used for training
        // A is the new data set and b is the predicted y of the trained model:
        // beta = y ~ x, then b = predict(beta, newx)
        std::string predict_fname = std::string(DATA_DIR) + "/linmod_data/linear_reg/" +
                                    csvname + intercept_suff + "_predict_data.csv";
        std::vector<T> predict_data;
        ASSERT_TRUE(da_test::read_csv_data(predict_fname, predict_data, nrows, ncols))
            << "Failed to read predict data: " << predict_fname;
        nfeat = ncols - 1;
        nsamples = nrows;
        std::vector<T> A_pred(predict_data.begin(),
                              predict_data.begin() + nfeat * nsamples);
        std::vector<T> b_pred(predict_data.begin() + nfeat * nsamples,
                              predict_data.end());

        std::vector<T> pred(nsamples);
        EXPECT_EQ(da_linmod_evaluate_model(linmod_handle, nsamples, nfeat, A_pred.data(),
                                           nsamples, pred.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(nsamples, pred.data(), b_pred.data(),
                        expected_precision<T>(check_tol_scale));
        EXPECT_EQ(da_linmod_evaluate_model(linmod_handle, nsamples, nfeat, A_pred.data(),
                                           nsamples, pred.data(), b_pred.data(), &loss),
                  da_status_success);
    }

    //////////////
    // Free memory
    //////////////
    da_handle_destroy(&linmod_handle);
}

#endif
