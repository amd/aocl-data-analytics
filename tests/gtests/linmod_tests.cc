#include "linmod_functions.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {
typedef struct {
    std::string test_name;
    std::string data_name;
    linmod_model mod;
    /* This should be options */
    std::vector<option_t<bool>> bopts;
    std::vector<option_t<std::string>> sopts;
    std::vector<option_t<float>> fopts;
    std::vector<option_t<double>> dopts;
    std::vector<option_t<int>> iopts;
    // Some tests cannot run to sufficient accuracy in float
    bool skip_float = false;
} linmodParamType;

const linmodParamType linmodPosValues[] = {
    {"trivialMSENoint", "trivial", linmod_model_mse, {}, {}, {}, {}, {}, false},
    {"trivialMSEI",
     "trivial",
     linmod_model_mse,
     {{"linmod intercept", true}},
     {},
     {},
     {},
     {},
     false},
    {"studyLogI",
     "study",
     linmod_model_logistic,
     {{"linmod intercept", true}},
     {},
     {},
     {},
     {},
     true},
    {"studyLogNoint", "study", linmod_model_logistic, {}, {}, {}, {}, {}, true},
    {"lrsetLogI",
     "lrset",
     linmod_model_logistic,
     {{"linmod intercept", true}},
     {},
     {},
     {},
     {},
     true},
    {"lrsetLogNoint", "lrset", linmod_model_logistic, {}, {}, {}, {}, {}, true}};

// Data Driven (parametrized) Tests
class linmodTestPos : public testing::TestWithParam<linmodParamType> {};

// Teach GTest how to print the param type
// in this case use only user's unique testname
// It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
void PrintTo(const linmodParamType &param, ::std::ostream *os) { *os << param.test_name; }

// Positive (da_status_success) tests with double type
TEST_P(linmodTestPos, Double) {
    // Inside a test, access the test parameter with the GetParam() method
    // of the TestWithParam<T> class:
    const linmodParamType &param = GetParam();
    test_linmod_positive<double>(param.data_name, param.mod, param.bopts, param.sopts,
                                 param.dopts, param.iopts);
}
// Positive (da_status_success) tests with float type
TEST_P(linmodTestPos, Float) {
    // Inside a test, access the test parameter with the GetParam() method
    // of the TestWithParam<T> class:
    const linmodParamType &param = GetParam();
    if (param.skip_float)
        GTEST_SKIP() << "Single precision is not enough to achieve good enough accuracy";

    test_linmod_positive<float>(param.data_name, param.mod, param.bopts, param.sopts,
                                param.fopts, param.iopts);
}

INSTANTIATE_TEST_SUITE_P(linmodPosSuite, linmodTestPos,
                         testing::ValuesIn(linmodPosValues));
} // namespace