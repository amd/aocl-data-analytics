#include "aoclda.h"
#include "utest_utils.hpp"
#include "gtest/gtest.h"
#include <iostream>
#include <string>

// Helper to define precision to which we expect the results match
template <typename T> T expected_precision(T scale = (T)1.0);
template <> double expected_precision<double>(double scale) { return scale * 1.0e-4; }

template <> float expected_precision<float>(float scale) { return scale * 5.0e-02f; }

template <typename T> void test_linmod_positive(std::string csvname, linreg_model mod) {

    // get problem data and expected results
    // DATA_DIR is defined in the build system, it should point to the tests/data/linmod_data
    std::string A_file = std::string(DATA_DIR) + "/" + csvname + "_A.csv";
    std::string b_file = std::string(DATA_DIR) + "/" + csvname + "_b.csv";
    std::string modname, coef_file;
    switch (mod) {
    case linreg_model_mse:
        modname = "mse";
        break;
    case linreg_model_logistic:
        modname = "log";
        break;
    default:
        FAIL() << "Unknown model\n";
        break;
    }
    coef_file = std::string(DATA_DIR) + "/" + csvname + "_" + modname + "_coeffs.csv";

    // Read features
    da_handle csv_handle = nullptr;
    ASSERT_EQ(da_handle_init_s(&csv_handle, da_handle_csv_opts), da_status_success);
    T *a = nullptr, *b = nullptr, *coef_exp = nullptr;

    da_int n = 0, m = 0;
    ASSERT_EQ(da_read_csv(csv_handle, A_file.c_str(), &a, &m, &n), da_status_success);
    da_int nb, mb;
    ASSERT_EQ(da_read_csv(csv_handle, b_file.c_str(), &b, &mb, &nb), da_status_success);
    ASSERT_EQ(m, nb); // b is stored in one row
    da_int nc, mc;
    ASSERT_EQ(da_read_csv(csv_handle, coef_file.c_str(), &coef_exp, &mc, &nc),
              da_status_success);
    // ASSERT_EQ(n, nc); // TODO add check once the intersect has been solved

    // Create problem
    da_handle linreg_handle = nullptr;
    ASSERT_EQ(da_linreg_init<T>(&linreg_handle), da_status_success);
    ASSERT_EQ(da_linreg_select_model<T>(linreg_handle, mod), da_status_success);
    ASSERT_EQ(da_linreg_define_features(linreg_handle, n, m, a, b), da_status_success);

    // compute regression
    EXPECT_EQ(da_linreg_fit<T>(linreg_handle), da_status_success);

    // Extract and compare solution
    T *coef = new T[nc];
    da_int ncc = nc;
    EXPECT_EQ(da_linreg_get_coef(linreg_handle, &ncc, coef), da_status_success);
    EXPECT_ARR_NEAR(ncc, coef_exp, coef, expected_precision<T>());

    if (a)
        free(a);
    if (b)
        free(b);
    if (coef_exp)
        free(coef_exp);
    if (coef)
        delete[] coef;
    da_handle_destroy(&csv_handle);
    da_handle_destroy(&linreg_handle);

    return;
}