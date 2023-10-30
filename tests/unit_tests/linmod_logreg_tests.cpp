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
#include "da_handle.hpp"
#include "linmod_logreg.hpp"
#include "options.hpp"
#include "utest_utils.hpp"
#include "gtest/gtest.h"

typedef struct {
    std::string test_name; // name of the ctest test
    std::string data_name; // name of the files to read in
    std::vector<option_t<da_int>> iopts;
    std::vector<option_t<std::string>> sopts;
    std::vector<option_t<float>> fopts;
    std::vector<option_t<double>> dopts;
} logregParam;

// Test parameters for the logistic retgression
const logregParam logregPosValuesD[] = {
    {"lrsetNoIntercept", "lrset", {}, {}, {}, {}},
    {"lrsetIntercept", "lrset", {{"linmod intercept", 1}}, {}, {}, {}},
    {"studyNoIntercept", "study", {}, {}, {}, {}},
    {"studyIntercept", "study", {{"linmod intercept", 1}}, {}, {}, {}},
    {"multinomialNoIntercept", "multinomial", {}, {}, {}, {}},
    {"multinomialIntercept", "multinomial", {{"linmod intercept", 1}}, {}, {}, {}},
    {"usrdataIntercept", "usrdata", {{"linmod intercept", 1}}, {}, {}, {}},
};
const logregParam logregPosValuesF[] = {
    {"multinomialNoIntercept", "multinomial", {}, {}, {}, {}},
    {"multinomialIntercept", "multinomial", {{"linmod intercept", 1}}, {}, {}, {}},
};

// Data  Tests
class logregPosD : public testing::TestWithParam<logregParam> {};
class logregPosF : public testing::TestWithParam<logregParam> {};

// Teach GTest how to print the param type
// in this case use only user's unique testname
// It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
void PrintTo(const logregParam &param, ::std::ostream *os) { *os << param.test_name; }

// Positive tests with double type
TEST_P(logregPosD, Double) {
    const logregParam &param = GetParam();
    test_logreg_positive<double>(param.data_name, param.iopts, param.sopts, param.dopts);
}

// Positive tests with float type
TEST_P(logregPosF, Float) {
    const logregParam &param = GetParam();
    test_logreg_positive<float>(param.data_name, param.iopts, param.sopts, param.fopts);
}

INSTANTIATE_TEST_SUITE_P(logregPosSuiteD, logregPosD,
                         testing::ValuesIn(logregPosValuesD));
INSTANTIATE_TEST_SUITE_P(logregPosSuiteF, logregPosF,
                         testing::ValuesIn(logregPosValuesF));
