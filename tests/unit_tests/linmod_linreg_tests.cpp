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
#include "linmod_linreg.hpp"
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
} linregParam;

const linregParam linregPosValuesD[] = {
    {"trivialNoint", "trivial", {}, {}, {}, {}},
    {"trivialNointLbfgs", "trivial", {}, {{"linmod optim method", "lbfgs"}}, {}, {}},
    {"trivialIntercept", "trivial", {{"linmod intercept", 1}}, {}, {}, {}},
    {"trivialILbfgs",
     "trivial",
     {{"linmod intercept", 1}},
     {{"linmod optim method", "lbfgs"}},
     {},
     {}},
};
const linregParam linregPosValuesF[] = {linregPosValuesD[0], linregPosValuesD[2]};

// Data  Tests
class linregPosD : public testing::TestWithParam<linregParam> {};
class linregPosF : public testing::TestWithParam<linregParam> {};

// Teach GTest how to print the param type
// in this case use only user's unique testname
// It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
void PrintTo(const linregParam &param, ::std::ostream *os) { *os << param.test_name; }

// Positive tests with double type
TEST_P(linregPosD, Double) {
    const linregParam &param = GetParam();
    test_linreg_positive<double>(param.data_name, param.iopts, param.sopts, param.dopts);
}

// Positive tests with float type
TEST_P(linregPosF, Float) {
    const linregParam &param = GetParam();
    test_linreg_positive<float>(param.data_name, param.iopts, param.sopts, param.fopts);
}

INSTANTIATE_TEST_SUITE_P(linregPosSuiteD, linregPosD,
                         testing::ValuesIn(linregPosValuesD));
INSTANTIATE_TEST_SUITE_P(linregPosSuiteF, linregPosF,
                         testing::ValuesIn(linregPosValuesF));