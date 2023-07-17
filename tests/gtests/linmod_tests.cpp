#include "aoclda.h"
#include "da_handle.hpp"
#include "linmod_functions.hpp"
#include "options.hpp"
#include "utest_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {
typedef struct {
    std::string test_name;
    std::string data_name;
    linmod_model mod;
    std::vector<option_t<da_int>> iopts;
    std::vector<option_t<std::string>> sopts;
    std::vector<option_t<float>> fopts;
    std::vector<option_t<double>> dopts;
} linmodParamType;

const linmodParamType linmodPosValuesD[] = {
    {"trivialMSENoint", "trivial", linmod_model_mse, {}, {}, {}, {}},
    {"trivialMSENointLbfgs",
     "trivial",
     linmod_model_mse,
     {},
     {{"linmod optim method", "lbfgs"}},
     {},
     {}},
    {"trivialMSEI", "trivial", linmod_model_mse, {{"linmod intercept", 1}}, {}, {}, {}},
    {"trivialMSEILbfgs",
     "trivial",
     linmod_model_mse,
     {{"linmod intercept", 1}},
     {{"linmod optim method", "lbfgs"}},
     {},
     {}},
    {"studyLogI", "study", linmod_model_logistic, {{"linmod intercept", 1}}, {}, {}, {}},
    {"studyLogNoint", "study", linmod_model_logistic, {}, {}, {}, {}},
    {"lrsetLogI", "lrset", linmod_model_logistic, {{"linmod intercept", 1}}, {}, {}, {}},
    {"lrsetLogNoint", "lrset", linmod_model_logistic, {}, {}, {}, {}}};
const linmodParamType linmodPosValuesF[] = {linmodPosValuesD[0], linmodPosValuesD[2]};

// Data Driven (parametrized) Tests
class linmodTestPosD : public testing::TestWithParam<linmodParamType> {};
class linmodTestPosF : public testing::TestWithParam<linmodParamType> {};

// Teach GTest how to print the param type
// in this case use only user's unique testname
// It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
void PrintTo(const linmodParamType &param, ::std::ostream *os) { *os << param.test_name; }

// Positive (da_status_success) tests with double type
TEST_P(linmodTestPosD, Double) {
    // Inside a test, access the test parameter with the GetParam() method
    // of the TestWithParam<T> class:
    const linmodParamType &param = GetParam();
    test_linmod_positive<double>(param.data_name, param.mod, param.iopts, param.sopts,
                                 param.dopts);
}
// Positive (da_status_success) tests with float type
TEST_P(linmodTestPosF, Float) {
    // Inside a test, access the test parameter with the GetParam() method
    // of the TestWithParam<T> class:
    const linmodParamType &param = GetParam();
    test_linmod_positive<float>(param.data_name, param.mod, param.iopts, param.sopts,
                                param.fopts);
}

INSTANTIATE_TEST_SUITE_P(linmodPosSuiteD, linmodTestPosD,
                         testing::ValuesIn(linmodPosValuesD));
INSTANTIATE_TEST_SUITE_P(linmodPosSuiteF, linmodTestPosF,
                         testing::ValuesIn(linmodPosValuesF));

/* simple errors tests */
TEST(linmod, badHandle) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_linmod_select_model<double>(handle, linmod_model_mse),
              da_status_memory_error);
    EXPECT_EQ(da_linmod_select_model<float>(handle, linmod_model_logistic),
              da_status_memory_error);

    da_int n = 1, m = 1;
    float *af = 0, *bf = 0;
    double *ad = 0, *bd = 0;
    EXPECT_EQ(da_linreg_define_features(handle, n, m, af, bf), da_status_memory_error);
    EXPECT_EQ(da_linreg_define_features(handle, n, m, ad, bd), da_status_memory_error);

    EXPECT_EQ(da_linmod_d_fit(handle), da_status_memory_error);
    EXPECT_EQ(da_linmod_s_fit(handle), da_status_memory_error);

    da_int nc = 1;
    float *xf = 0;
    double *xd = 0;
    EXPECT_EQ(da_linmod_get_coef(handle, &nc, xf), da_status_invalid_pointer);
    EXPECT_EQ(da_linmod_get_coef(handle, &nc, xd), da_status_invalid_pointer);

    float *predf = 0;
    double *predd = 0;
    EXPECT_EQ(da_linmod_evaluate_model(handle, n, m, xf, predf), da_status_memory_error);
    EXPECT_EQ(da_linmod_evaluate_model(handle, n, m, xd, predd), da_status_memory_error);
}

TEST(linmod, wrongType) {
    da_handle handle_d = nullptr;
    da_handle handle_s = nullptr;

    EXPECT_EQ(da_handle_init<double>(&handle_d, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_handle_init<float>(&handle_s, da_handle_linmod), da_status_success);

    EXPECT_EQ(da_linmod_select_model<double>(handle_s, linmod_model_mse),
              da_status_wrong_type);
    EXPECT_EQ(da_linmod_select_model<float>(handle_d, linmod_model_logistic),
              da_status_wrong_type);

    da_int n = 1, m = 1;
    float *af = 0, *bf = 0;
    double *ad = 0, *bd = 0;
    EXPECT_EQ(da_linreg_define_features(handle_d, n, m, af, bf), da_status_wrong_type);
    EXPECT_EQ(da_linreg_define_features(handle_s, n, m, ad, bd), da_status_wrong_type);

    EXPECT_EQ(da_linmod_d_fit(handle_s), da_status_wrong_type);
    EXPECT_EQ(da_linmod_s_fit(handle_d), da_status_wrong_type);

    da_int nc = 1;
    float *xf = 0;
    double *xd = 0;
    EXPECT_EQ(da_linmod_get_coef(handle_d, &nc, xf), da_status_wrong_type);
    EXPECT_EQ(da_linmod_get_coef(handle_s, &nc, xd), da_status_wrong_type);

    float *predf = 0;
    double *predd = 0;
    EXPECT_EQ(da_linmod_evaluate_model(handle_d, n, m, xf, predf), da_status_wrong_type);
    EXPECT_EQ(da_linmod_evaluate_model(handle_s, n, m, xd, predd), da_status_wrong_type);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}

TEST(linmod, invalidInput) {
    // problem data
    da_int m = 5, n = 2;
    double Ad[10] = {1, 2, 3, 4, 5, 1, 3, 5, 1, 1};
    double bd[5] = {1, 1, 1, 1, 1};
    da_int nx = 2;
    double xd[2];
    float As[10] = {1, 2, 3, 4, 5, 1, 3, 5, 1, 1};
    float bs[5] = {1, 1, 1, 1, 1};
    float xs[2];

    // Initialize and compute the linear regression
    da_handle handle_d = nullptr;
    da_handle handle_s = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle_d, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_d_select_model(handle_d, linmod_model_mse), da_status_success);
    EXPECT_EQ(da_handle_init_s(&handle_s, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_s_select_model(handle_s, linmod_model_mse), da_status_success);

    // define features
    EXPECT_EQ(da_linmod_d_define_features(handle_d, 0, m, Ad, bd),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_d_define_features(handle_d, n, 0, Ad, bd),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_d_define_features(handle_d, n, m, nullptr, bd),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_d_define_features(handle_d, n, m, Ad, nullptr),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_d_define_features(handle_d, n, m, Ad, bd), da_status_success);

    EXPECT_EQ(da_linmod_s_define_features(handle_s, 0, m, As, bs),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_s_define_features(handle_s, n, 0, As, bs),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_s_define_features(handle_s, n, m, nullptr, bs),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_s_define_features(handle_s, n, m, As, nullptr),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_s_define_features(handle_s, n, m, As, bs), da_status_success);

    // comput regression
    EXPECT_EQ(da_linmod_d_fit(handle_d), da_status_success);
    EXPECT_EQ(da_linmod_s_fit(handle_s), da_status_success);

    // get coefficients
    nx = -1;
    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_linmod_coeff, &nx, xd),
              da_status_invalid_array_dimension);
    nx = -1;
    EXPECT_EQ(da_handle_get_result_s(handle_s, da_result::da_linmod_coeff, &nx, xs),
              da_status_invalid_array_dimension);
    nx = 2;
    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_linmod_coeff, &nx, nullptr),
              da_status_invalid_input);
    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_linmod_coeff, &nx, xd),
              da_status_success);
    EXPECT_EQ(da_handle_get_result_s(handle_s, da_result::da_linmod_coeff, &nx, nullptr),
              da_status_invalid_input);
    EXPECT_EQ(da_handle_get_result_s(handle_s, da_result::da_linmod_coeff, &nx, xs),
              da_status_success);

    // evaluate models
    double X[2] = {1., 2.};
    double pred[1];
    EXPECT_EQ(da_linmod_evaluate_model(handle_d, 3, 1, X, pred), da_status_invalid_input);
    EXPECT_EQ(da_linmod_evaluate_model(handle_d, n, 1, nullptr, pred),
              da_status_invalid_pointer);
    EXPECT_EQ(da_linmod_evaluate_model(handle_d, n, 1, X, nullptr),
              da_status_invalid_pointer);
    EXPECT_EQ(da_linmod_evaluate_model(handle_d, n, 0, X, pred), da_status_invalid_input);
    float Xs[2] = {1., 2.};
    float preds[1];
    EXPECT_EQ(da_linmod_evaluate_model(handle_s, 3, 1, Xs, preds),
              da_status_invalid_input);
    EXPECT_EQ(da_linmod_evaluate_model(handle_s, n, 1, nullptr, preds),
              da_status_invalid_pointer);
    EXPECT_EQ(da_linmod_evaluate_model(handle_s, n, 1, Xs, nullptr),
              da_status_invalid_pointer);
    EXPECT_EQ(da_linmod_evaluate_model(handle_s, n, 0, Xs, preds),
              da_status_invalid_input);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}

TEST(linmod, modOutOfDate) {
    // problem data
    da_int m = 5, n = 2;
    double Ad[10] = {1, 2, 3, 4, 5, 1, 3, 5, 1, 1};
    double bd[5] = {1, 1, 1, 1, 1};
    da_int nx = 2;
    double xd[2];
    float As[10] = {1, 2, 3, 4, 5, 1, 3, 5, 1, 1};
    float bs[5] = {1, 1, 1, 1, 1};
    float xs[2];

    da_handle handle_d = nullptr;
    da_handle handle_s = nullptr;

    EXPECT_EQ(da_handle_init<double>(&handle_d, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_handle_init<float>(&handle_s, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_d_define_features(handle_d, n, m, Ad, bd), da_status_success);
    EXPECT_EQ(da_linmod_s_define_features(handle_s, n, m, As, bs), da_status_success);

    // Model was not yet fitted or out-of-date request of coefficients
    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_linmod_coeff, &nx, xd),
              da_status_unknown_query);
    EXPECT_EQ(da_handle_get_result_s(handle_s, da_result::da_linmod_coeff, &nx, xs),
              da_status_unknown_query);

    // Out of date request of model
    double X[2] = {1., 2.};
    double pred[1];
    float Xs[2] = {1., 2.};
    float preds[1];
    EXPECT_EQ(da_linmod_evaluate_model(handle_d, n, 1, X, pred), da_status_out_of_date);
    EXPECT_EQ(da_linmod_evaluate_model(handle_s, n, 1, Xs, preds), da_status_out_of_date);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}

TEST(linmod, incompatibleOptions) {
    // problem data
    da_int m = 5, n = 2;
    double Ad[10] = {1, 2, 3, 4, 5, 1, 3, 5, 1, 1};
    double bd[5] = {1, 1, 1, 1, 1};
    da_handle handle_d = nullptr;

    EXPECT_EQ(da_handle_init<double>(&handle_d, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_d_define_features(handle_d, n, m, Ad, bd), da_status_success);
    EXPECT_EQ(da_options_set_string(handle_d, "linmod optim method", "QR"),
              da_status_success);
    EXPECT_EQ(da_linmod_d_select_model(handle_d, linmod_model_logistic),
              da_status_success);

    // QR factorization should not be compatible with logistic regression
    EXPECT_EQ(da_linmod_d_fit(handle_d), da_status_incompatible_options);

    da_handle_destroy(&handle_d);
}

TEST(linmod, GetResultNegative) {
    // Test public interfaces (under linmod context)
    da_handle handle_d = nullptr;
    da_handle handle_s = nullptr;
    double dv[2];
    float sv[2];
    da_int iv[2];
    da_int dim;

    // handle is null
    EXPECT_EQ(da_handle_get_result_d(nullptr, da_result::da_rinfo, &dim, dv),
              da_status_invalid_pointer);
    EXPECT_EQ(da_handle_get_result_s(nullptr, da_result::da_rinfo, &dim, sv),
              da_status_invalid_pointer);
    EXPECT_EQ(da_handle_get_result_int(nullptr, da_result::da_rinfo, &dim, iv),
              da_status_invalid_pointer);

    // handle valid but not initialized with any solver
    EXPECT_EQ(da_handle_init_d(&handle_d, da_handle_type::da_handle_uninitialized),
              da_status_success);
    EXPECT_EQ(da_handle_init_s(&handle_s, da_handle_type::da_handle_uninitialized),
              da_status_success);

    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_rinfo, &dim, dv),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_handle_get_result_s(handle_s, da_result::da_rinfo, &dim, sv),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_handle_get_result_int(handle_d, da_result::da_rinfo, &dim, iv),
              da_status_handle_not_initialized);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);

    // handle valid but no problem solved -> empty data
    EXPECT_EQ(da_handle_init_d(&handle_d, da_handle_type::da_handle_linmod),
              da_status_success);
    EXPECT_EQ(da_handle_init_s(&handle_s, da_handle_type::da_handle_linmod),
              da_status_success);

    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_rinfo, &dim, dv),
              da_status_unknown_query);
    EXPECT_EQ(da_handle_get_result_s(handle_s, da_result::da_rinfo, &dim, sv),
              da_status_unknown_query);
    // enable this test once a handle provides integer data
    // EXPECT_EQ(da_handle_get_result_int(handle_s, da_result::da_integer_type, &dim, iv),
    //           da_status_unknown_query);

    // handle valid but get_result is of different precision than handle
    EXPECT_EQ(da_handle_get_result_d(handle_s, da_result::da_rinfo, &dim, dv),
              da_status_wrong_type);
    EXPECT_EQ(da_handle_get_result_s(handle_d, da_result::da_rinfo, &dim, sv),
              da_status_wrong_type);
    // No need to test get_result_int since it cannot fail on this test.

    // handle valid but query is for a different handle group (linmod vs. pca)
    EXPECT_EQ(da_handle_get_result_d(handle_d, da_result::da_pca_scores, &dim, dv),
              da_status_unknown_query);
    EXPECT_EQ(da_handle_get_result_s(handle_s, da_result::da_pca_scores, &dim, sv),
              da_status_unknown_query);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}

} // namespace
