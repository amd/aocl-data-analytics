#include "linmod_functions.hpp"
#include "gtest/gtest.h"

namespace {
typedef struct {
    std::string test_name;
    std::string data_name;
    linreg_model mod;
    // Some tests cannot run to sufficient accuracy in float
    bool skip_float = false;
} linmodParamType;

const linmodParamType linmodPosValues[] = {
    {"trivialMSE", "trivial", linreg_model_mse},
    {"studyLog", "study", linreg_model_logistic, true}};

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
    test_linmod_positive<double>(param.data_name, param.mod);
}
// Positive (da_status_success) tests with float type
TEST_P(linmodTestPos, Float) {
    // Inside a test, access the test parameter with the GetParam() method
    // of the TestWithParam<T> class:
    const linmodParamType &param = GetParam();
    if (param.skip_float)
        GTEST_SKIP() << "Single precision is not enough to achieve good enough accuracy";

    test_linmod_positive<float>(param.data_name, param.mod);
}

INSTANTIATE_TEST_SUITE_P(linmodPosSuite, linmodTestPos,
                         testing::ValuesIn(linmodPosValues));
} // namespace