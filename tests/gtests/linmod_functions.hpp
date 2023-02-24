#include "aoclda.h"
#include "da_cblas.hh"
#include "utest_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <string>

using namespace testing;
// Helper to define precision to which we expect the results match
template <typename T> T expected_precision(T scale = (T)1.0);
template <> double expected_precision<double>(double scale) { return scale * 5.0e-4; }

template <> float expected_precision<float>(float scale) { return scale * 0.5f; }

template <typename T> T log_loss(T y, T p) { return -y * log(p) - (1 - y) * log(1 - p); }
template <typename T> T logistic(T x) { return 1 / (1 + exp(-x)); }

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
            grad[n - 1] += 2.0 * y[i];
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
            lin_comb = intercept ? x[n - 1] + y[i] : y[i];
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
        FAIL() << "unexpected gardient function";
    }
}

template <typename T>
void test_linmod_positive(std::string csvname, linmod_model mod, bool intercept) {

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
        FAIL() << "Unknown model\n";
        break;
    }
    coef_file = std::string(DATA_DIR) + "/" + csvname + "_" + modname;
    if (!intercept)
        coef_file.append("_noint");
    coef_file.append("_coeffs.csv");

    // Read features
    da_handle csv_handle = nullptr;
    ASSERT_EQ(da_handle_init_d(&csv_handle, da_handle_csv_opts), da_status_success);
    T *a = nullptr, *b = nullptr, *coef_exp = nullptr;

    da_int n = 0, m = 0;
    ASSERT_EQ(da_read_csv(csv_handle, A_file.c_str(), &a, &n, &m), da_status_success);
    da_int nb, mb;
    ASSERT_EQ(da_read_csv(csv_handle, b_file.c_str(), &b, &mb, &nb), da_status_success);
    ASSERT_EQ(m, nb); // b is stored in one row
    da_int nc = intercept ? n + 1 : n;
    /* expected results not tested ? check gradient of solution instead */
    //da_int nc, mc;
    //ASSERT_EQ(da_read_csv(csv_handle, coef_file.c_str(), &coef_exp, &mc, &nc),
    //          da_status_success);
    // ASSERT_EQ(n, nc); // TODO add check once the intersect has been solved

    // Create problem
    da_handle linreg_handle = nullptr;
    ASSERT_EQ(da_linreg_init<T>(&linreg_handle), da_status_success);
    ASSERT_EQ(da_linreg_select_model<T>(linreg_handle, mod), da_status_success);
    ASSERT_EQ(da_linreg_define_features(linreg_handle, n, m, a, b), da_status_success);

    // This should be options
    EXPECT_EQ(da_linmod_set_intercept<T>(linreg_handle, intercept), da_status_success);

    // compute regression
    EXPECT_EQ(da_linreg_fit<T>(linreg_handle), da_status_success);

    // Extract and compare solution
    T *coef = new T[nc];
    da_int ncc = nc;
    EXPECT_EQ(da_linreg_get_coef(linreg_handle, &ncc, coef), da_status_success);
    //EXPECT_ARR_NEAR(ncc, coef_exp, coef, expected_precision<T>());

    // Check that the gradient is close enough to 0
    std::vector<T> grad;
    grad.resize(nc);
    objgrd(mod, nc, m, coef, grad, a, b, intercept);
    EXPECT_THAT(grad,
                Each(AllOf(Gt(-expected_precision<T>()), Lt(expected_precision<T>()))));

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