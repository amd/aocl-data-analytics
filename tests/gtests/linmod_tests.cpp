#include "linmod_functions.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {
typedef struct {
    std::string test_name;
    std::string data_name;
    linmod_model mod;
    std::vector<option_t<bool>> bopts;
    std::vector<option_t<std::string>> sopts;
    std::vector<option_t<float>> fopts;
    std::vector<option_t<double>> dopts;
    std::vector<option_t<int>> iopts;
} linmodParamType;

const linmodParamType linmodPosValuesD[] = {
    {"trivialMSENoint", "trivial", linmod_model_mse, {}, {}, {}, {}, {}},
    {"trivialMSEI",
     "trivial",
     linmod_model_mse,
     {{"linmod intercept", true}},
     {},
     {},
     {},
     {}},
    {"studyLogI",
     "study",
     linmod_model_logistic,
     {{"linmod intercept", true}},
     {},
     {},
     {},
     {}},
    {"studyLogNoint", "study", linmod_model_logistic, {}, {}, {}, {}, {}},
    {"lrsetLogI",
     "lrset",
     linmod_model_logistic,
     {{"linmod intercept", true}},
     {},
     {},
     {},
     {}},
    {"lrsetLogNoint", "lrset", linmod_model_logistic, {}, {}, {}, {}, {}}};
const linmodParamType linmodPosValuesF[] = {linmodPosValuesD[0], linmodPosValuesD[1]};

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
    test_linmod_positive<double>(param.data_name, param.mod, param.bopts, param.sopts,
                                 param.dopts, param.iopts);
}
// Positive (da_status_success) tests with float type
TEST_P(linmodTestPosF, Float) {
    // Inside a test, access the test parameter with the GetParam() method
    // of the TestWithParam<T> class:
    const linmodParamType &param = GetParam();
    test_linmod_positive<float>(param.data_name, param.mod, param.bopts, param.sopts,
                                param.fopts, param.iopts);
}

INSTANTIATE_TEST_SUITE_P(linmodPosSuiteD, linmodTestPosD,
                         testing::ValuesIn(linmodPosValuesD));
INSTANTIATE_TEST_SUITE_P(linmodPosSuiteF, linmodTestPosF,
                         testing::ValuesIn(linmodPosValuesF));
} // namespace
