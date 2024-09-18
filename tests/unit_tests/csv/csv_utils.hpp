/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef CSV_TESTS_UTILS_HPP
#define CSV_TESTS_UTILS_HPP

#include "aoclda.h"
#include "gtest/gtest.h"
#include <cmath>

// Overload some EXPECTs nan checks and freeing functions for potential char* data
template <typename T> void EXPECT_EQ_overload(T d1, T d2) { EXPECT_EQ(d1, d2); }
void EXPECT_EQ_overload(const char *d1, const char *d2);

template <typename T> bool check_nan(T num) {
    return std::isnan(static_cast<double>(num));
}

bool check_nan([[maybe_unused]] char *num);
bool check_nan([[maybe_unused]] const char *num);

template <typename T> struct CSVParamType {
    std::string filename;
    da_int expected_rows;
    da_int expected_columns;
    std::vector<T> expected_data;
    std::vector<std::string> expected_char_data;
    std::vector<std::string> expected_headings;
    da_status expected_status;
    std::string datatype;
};

template <typename T> T GetExpectedData(CSVParamType<T> *params, da_int i) {
    return params->expected_data[i];
}

const char *GetExpectedData(CSVParamType<char *> *params, da_int i);

template <typename T> void GetBasicDataColMajor(CSVParamType<T> *params);

template <typename T> void GetMissingData(CSVParamType<T> *params);

template <typename T> void GetBasicData(CSVParamType<T> *params);

#endif