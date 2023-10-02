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
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <string>
#include <type_traits>

using namespace testing;

template <class T> struct option_t {
    std::string name = "";
    T value;
};

// Helper to define precision to which we expect the results match
template <typename T> T expected_precision(T scale = (T)1.0);
template <> double expected_precision<double>(double scale) { return scale * 1.0e-3; }

template <> float expected_precision<float>(float scale) { return scale * 0.5f; }

//template <typename T> T log_loss(T y, T p) { return -y * log(p) - (1 - y) * log(1 - p); }
//template <typename T> T logistic(T x) { return 1 / (1 + exp(-x)); }

template <typename T>
void objgrd_mse(da_int n, da_int m, T *x, std::vector<T> &grad, const T *A, const T *b,
                bool intercept) {

    T alpha = 1.0, beta = 0.0;
    std::vector<T> y;
    y.resize(m);
    da_int aux = intercept ? 1 : 0;
    da_blas::cblas_gemv(CblasColMajor, CblasNoTrans, m, n - aux, alpha, A, m, x, 1, beta,
                        y.data(), 1);
    if (intercept) {
        for (da_int i = 0; i < m; i++)
            y[i] += x[n - 1];
    }
    alpha = -1.0;
    da_blas::cblas_axpy(m, alpha, b, 1, y.data(), 1);

    alpha = 2.0;
    da_blas::cblas_gemv(CblasColMajor, CblasTrans, m, n - aux, alpha, A, m, y.data(), 1,
                        beta, grad.data(), 1);
    if (intercept) {
        grad[n - 1] = 0.0;
        for (da_int i = 0; i < m; i++)
            grad[n - 1] += (T)2.0 * y[i];
    }
}

template <typename T>
void objgrd_logistic(da_int n, da_int m, T *x, std::vector<T> &grad, const T *A,
                     const T *b, bool intercept) {
    /* gradient of log loss of the logistic function 
     * g_j = sum_i{A_ij*(b[i]-logistic(A_i^t x + x[n-1]))}
     */
    std::vector<T> y;

    // Comput A*x[0:n-2] = y
    da_int aux = intercept ? 1 : 0;
    T alpha = 1.0, beta = 0.0;
    y.resize(m);
    da_blas::cblas_gemv(CblasColMajor, CblasNoTrans, m, n - aux, alpha, A, m, x, 1, beta,
                        y.data(), 1);

    std::fill(grad.begin(), grad.end(), 0);
    T lin_comb;
    for (da_int i = 0; i < m; i++) {
        lin_comb = intercept ? x[n - 1] + y[i] : y[i];
        for (da_int j = 0; j < n - aux; j++)
            grad[j] += (logistic(lin_comb) - b[i]) * A[m * j + i];
    }
    if (intercept) {
        grad[n - 1] = 0.0;
        for (da_int i = 0; i < m; i++) {
            lin_comb = x[n - 1] + y[i];
            grad[n - 1] += (logistic(lin_comb) - b[i]);
        }
    }
}

template <typename T>
void objgrd(linmod_model mod, da_int n, da_int m, T *x, std::vector<T> &grad, const T *A,
            const T *b, bool intercept) {
    switch (mod) {
    case (linmod_model_mse):
        objgrd_mse(n, m, x, grad, A, b, intercept);
        break;

    case (linmod_model_logistic):
        objgrd_logistic(n, m, x, grad, A, b, intercept);
        break;

    default:
        FAIL() << "unexpected gradient function";
    }
}

template <typename T>
void test_linmod_positive(std::string csvname, linmod_model mod,
                          std::vector<option_t<da_int>> iopts,
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

    da_int intercept_int;
    EXPECT_EQ(da_options_get_int(linmod_handle, "linmod intercept", &intercept_int),
              da_status_success);
    bool intercept = (bool)intercept_int;

    T alpha = 0, lambda = 0;
    EXPECT_EQ(da_options_set_real(linmod_handle, "linmod alpha", alpha),
              da_status_success);
    EXPECT_EQ(da_options_set_real(linmod_handle, "linmod lambda", lambda),
              da_status_success);

    // get problem data and expected results
    // DATA_DIR is defined in the build system, it should point to the tests/data/linmod_data
    std::string A_file = std::string(DATA_DIR) + "/" + csvname + "_A.csv";
    std::string b_file = std::string(DATA_DIR) + "/" + csvname + "_b.csv";
    std::string modname, coef_file;
    switch (mod) {
    case linmod_model_mse:
        modname = "mse";
        break;
    case linmod_model_logistic:
        modname = "log";
        break;
    default:
        da_handle_destroy(&linmod_handle);
        FAIL() << "Unknown model\n";
        break;
    }
    coef_file = std::string(DATA_DIR) + "/" + csvname + "_" + modname;
    if (!intercept)
        coef_file.append("_noint");
    coef_file.append("_coeffs.csv");

    // Read features
    da_datastore csv_store = nullptr;
    EXPECT_EQ(da_datastore_init(&csv_store), da_status_success);
    T *a = nullptr, *b = nullptr;

    da_int n = 0, m = 0;
    EXPECT_EQ(da_read_csv(csv_store, A_file.c_str(), &a, &n, &m, nullptr),
              da_status_success);
    da_int nb, mb;
    EXPECT_EQ(da_read_csv(csv_store, b_file.c_str(), &b, &mb, &nb, nullptr),
              da_status_success);
    EXPECT_EQ(m, nb); // b is stored in one row
    da_int nc = intercept ? n + 1 : n;
    /* expected results not tested ? check gradient of solution instead */
    //da_int nc, mc;
    //EXPECT_EQ(da_read_csv(csv_store, coef_file.c_str(), &coef_exp, &mc, &nc),
    //          da_status_success);
    // EXPECT_EQ(n, nc); // TODO add check once the intersect has been solved

    EXPECT_EQ(da_linmod_select_model<T>(linmod_handle, mod), da_status_success);
    EXPECT_EQ(da_linreg_define_features(linmod_handle, n, m, a, b), da_status_success);

    // This should be options
    EXPECT_EQ(da_options_set_int(linmod_handle, "linmod intercept", intercept),
              da_status_success);

    EXPECT_EQ(da_options_set_string(linmod_handle, "print options", "yes"),
              da_status_success);

    // compute regression
    EXPECT_EQ(da_linreg_fit<T>(linmod_handle), da_status_success);

    // Check that RINFO containt data
    T rinfo[100], rexp[100]{0};
    da_int dim = 100;
    rexp[0] = (T)n;
    rexp[1] = (T)m;
    rexp[2] = (T)nc;
    rexp[3] = (T)intercept_int;
    rexp[4] = alpha;
    rexp[5] = lambda;
    // Extract and compare solution
    T *coef = new T[nc];
    da_int ncc = 0; // query the correct size
    if constexpr (std::is_same_v<T, double>) {
        EXPECT_EQ(
            da_handle_get_result_d(linmod_handle, da_result::da_linmod_coeff, &ncc, coef),
            da_status_invalid_array_dimension);
        EXPECT_EQ(
            da_handle_get_result_d(linmod_handle, da_result::da_linmod_coeff, &ncc, coef),
            da_status_success);
        EXPECT_EQ(da_handle_get_result_d(linmod_handle, da_result::da_rinfo, &dim, rinfo),
                  da_status_success);
    } else {
        EXPECT_EQ(
            da_handle_get_result_s(linmod_handle, da_result::da_linmod_coeff, &ncc, coef),
            da_status_invalid_array_dimension);
        EXPECT_EQ(
            da_handle_get_result_s(linmod_handle, da_result::da_linmod_coeff, &ncc, coef),
            da_status_success);
        EXPECT_EQ(da_handle_get_result_s(linmod_handle, da_result::da_rinfo, &dim, rinfo),
                  da_status_success);
    }
    // Don't check for entries 6 and 7 of rinfo
    rinfo[6] = rinfo[7] = 0;
    EXPECT_ARR_EQ(100, rexp, rinfo, 1, 1, 0, 0);

    std::vector<T> X(n);
    std::fill(X.begin(), X.end(), 1.0);
    T pred[1];
    EXPECT_EQ(da_linmod_evaluate_model(linmod_handle, n, 1, X.data(), pred),
              da_status_success);
    // TODO check model evaluation

    // Check that the gradient is close enough to 0
    std::vector<T> grad;
    grad.resize(nc);
    objgrd(mod, nc, m, coef, grad, a, b, intercept);
    EXPECT_THAT(grad,
                Each(AllOf(Gt(-expected_precision<T>()), Lt(expected_precision<T>()))));

    free(a);
    free(b);
    delete[] coef;
    da_datastore_destroy(&csv_store);
    da_handle_destroy(&linmod_handle);

    return;
}