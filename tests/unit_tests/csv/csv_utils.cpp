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

#include "csv_utils.hpp"
#include "gtest/gtest.h"

void EXPECT_EQ_overload(const char *d1, const char *d2) { EXPECT_STREQ(d1, d2); }
bool check_nan([[maybe_unused]] char *num) { return false; }
bool check_nan([[maybe_unused]] const char *num) { return false; }

const char *GetExpectedData(CSVParamType<char *> *params, da_int i) {
    return params->expected_char_data[i].c_str();
}

template <> void GetBasicDataColMajor<double>(CSVParamType<double> *params) {

    params->filename = "csv_test_float";
    params->expected_rows = 3;
    params->expected_columns = 5;
    params->expected_data = {1.1, -1,     0.0,     1e3, -3.2, 0.0, 4.1e-3, -4.5e4,
                             0.0, 0.03e6, -5.6e-7, 0.0, 2,    -10, 4.5e5};
    params->expected_headings = {"one", "cat two", "three", "FOUR", "Five"};
    params->expected_status = da_status_success;
    params->datatype = "double";
}

template <> void GetBasicDataColMajor<float>(CSVParamType<float> *params) {

    params->filename = "csv_test_float";
    params->expected_rows = 3;
    params->expected_columns = 5;
    std::vector<float> data{1.1f, -1.f,    0.0f,     1e3f, -3.2f, 0.0f,  4.1e-3f, -4.5e4f,
                            0.0f, 0.03e6f, -5.6e-7f, 0.0f, 2.f,   -10.f, 4.5e5f};
    params->expected_data = data;
    std::vector<std::string> headings = {"one", "cat two", "three", "FOUR", "Five"};
    params->expected_headings = headings;
    params->expected_status = da_status_success;
    params->datatype = "float";
}

template <> void GetBasicDataColMajor<int64_t>(CSVParamType<int64_t> *params) {

    params->filename = "csv_test_int64";
    params->expected_rows = 3;
    params->expected_columns = 4;
    std::vector<int64_t> data{1,
                              -0,
                              +345,
                              5,
                              -43,
                              -9223372036854775807,
                              3,
                              9223372036854775807,
                              -9223372036854775806,
                              0,
                              9223372036854775806,
                              67};
    params->expected_data = data;
    std::vector<std::string> headings = {"one", "two", "three", "four"};
    params->expected_headings = headings;
    params->expected_status = da_status_success;
    params->datatype = "integer";
}

template <> void GetBasicDataColMajor<int32_t>(CSVParamType<int32_t> *params) {

    params->filename = "csv_test_int32";
    params->expected_rows = 3;
    params->expected_columns = 4;
    std::vector<int32_t> data{1, -0,  +2147483646, 5, 43,         184,
                              3, +92, -2147483647, 0, 2147483647, 67};
    params->expected_data = data;
    std::vector<std::string> headings = {"one", "two", "three", "four"};
    params->expected_headings = headings;
    params->expected_status = da_status_success;
    params->datatype = "integer";
}

template <> void GetBasicDataColMajor<char *>(CSVParamType<char *> *params) {

    params->filename = "csv_test_char";
    params->expected_rows = 3;
    params->expected_columns = 4;
    std::vector<std::string> data = {"lorem",       "amet", "sed",   "ipsum",
                                     "consectetur", "do",   "dolor", "adipiscing",
                                     "eiusmod",     "sit",  "edit",  "tempor"};
    params->expected_char_data = data;
    std::vector<std::string> headings = {"this", "is", "the", "header"};
    params->expected_headings = headings;
    params->expected_status = da_status_success;
    params->datatype = "string";
}

template <> void GetBasicDataColMajor<uint8_t>(CSVParamType<uint8_t> *params) {

    params->filename = "csv_test_bool";
    params->expected_rows = 2;
    params->expected_columns = 4;
    std::vector<uint8_t> data{1, 1, 1, 0, 1, 0, 1, 0};
    params->expected_data = data;
    std::vector<std::string> headings = {"one", "two", "three", "four"};
    params->expected_headings = headings;
    params->expected_status = da_status_success;
    params->datatype = "boolean";
}

template <> void GetBasicData<double>(CSVParamType<double> *params) {

    params->filename = "csv_test_float";
    params->expected_rows = 3;
    params->expected_columns = 5;
    params->expected_data = {1.1,     1e3, 4.1e-3, 0.03e6, 2,   -1,  -3.2,  -4.5e4,
                             -5.6e-7, -10, 0.0,    0.0,    0.0, 0.0, 4.5e+5};
    params->expected_headings = {"one", "cat two", "three", "FOUR", "Five"};
    params->expected_status = da_status_success;
    params->datatype = "double";
}

template <> void GetBasicData<float>(CSVParamType<float> *params) {

    params->filename = "csv_test_float";
    params->expected_rows = 3;
    params->expected_columns = 5;
    std::vector<float> data{1.1f, 1e3f,  4.1e-3f, 0.03e6f,  2.f,
                            -1.f, -3.2f, -4.5e4f, -5.6e-7f, -10.f,
                            0.0f, 0.0f,  0.0f,    0.0f,     4.5e+5f};
    params->expected_data = data;
    std::vector<std::string> headings = {"one", "cat two", "three", "FOUR", "Five"};
    params->expected_headings = headings;
    params->expected_status = da_status_success;
    params->datatype = "float";
}

template <> void GetBasicData<int64_t>(CSVParamType<int64_t> *params) {

    params->filename = "csv_test_int64";
    params->expected_rows = 3;
    params->expected_columns = 4;
    std::vector<int64_t> data{1,
                              5,
                              3,
                              0,
                              -0,
                              -43,
                              9223372036854775807,
                              9223372036854775806,
                              +345,
                              -9223372036854775807,
                              -9223372036854775806,
                              67};
    params->expected_data = data;
    std::vector<std::string> headings = {"one", "two", "three", "four"};
    params->expected_headings = headings;
    params->expected_status = da_status_success;
    params->datatype = "integer";
}

template <> void GetBasicData<int32_t>(CSVParamType<int32_t> *params) {

    params->filename = "csv_test_int32";
    params->expected_rows = 3;
    params->expected_columns = 4;
    std::vector<int32_t> data{1,   5,          3,           0,   0,           43,
                              +92, 2147483647, +2147483646, 184, -2147483647, 67};
    params->expected_data = data;
    std::vector<std::string> headings = {"one", "two", "three", "four"};
    params->expected_headings = headings;
    params->expected_status = da_status_success;
    params->datatype = "integer";
}

template <> void GetBasicData<char *>(CSVParamType<char *> *params) {

    params->filename = "csv_test_char";
    params->expected_rows = 3;
    params->expected_columns = 4;
    std::vector<std::string> data = {"lorem", "ipsum",       "dolor",      "sit",
                                     "amet",  "consectetur", "adipiscing", "edit",
                                     "sed",   "do",          "eiusmod",    "tempor"};
    params->expected_char_data = data;
    std::vector<std::string> headings = {"this", "is", "the", "header"};
    params->expected_headings = headings;
    params->expected_status = da_status_success;
    params->datatype = "string";
}

template <> void GetBasicData<uint8_t>(CSVParamType<uint8_t> *params) {

    params->filename = "csv_test_bool";
    params->expected_rows = 2;
    params->expected_columns = 4;
    std::vector<uint8_t> data{1, 1, 1, 1, 1, 0, 0, 0};
    params->expected_data = data;
    std::vector<std::string> headings = {"one", "two", "three", "four"};
    params->expected_headings = headings;
    params->expected_status = da_status_success;
    params->datatype = "boolean";
}

template <> void GetMissingData<double>(CSVParamType<double> *params) {

    params->filename = "csv_test_float_missing_data";
    params->expected_rows = 3;
    params->expected_columns = 5;
    std::vector<double> data{1.1,     NAN, 4.1e-3, 0.03e6, 2,   -1,  -3.2,  -4.5e4,
                             -5.6e-7, NAN, NAN,    0.0,    0.0, 0.0, 4.5e+5};
    params->expected_data = data;
    params->expected_status = da_status_missing_data;
    params->datatype = "double";
}

template <> void GetMissingData<float>(CSVParamType<float> *params) {

    params->filename = "csv_test_float_missing_data";
    params->expected_rows = 3;
    params->expected_columns = 5;
    params->expected_data = {1.1f,     NAN, 4.1e-3f, 0.03e6f, 2.f,  -1.f, -3.2f,  -4.5e4f,
                             -5.6e-7f, NAN, NAN,     0.0f,    0.0f, 0.0f, 4.5e+5f};
    params->expected_status = da_status_missing_data;
    params->datatype = "float";
}

template <> void GetMissingData<int64_t>(CSVParamType<int64_t> *params) {

    params->filename = "csv_test_int64_missing_data";
    params->expected_rows = 3;
    params->expected_columns = 4;
    std::vector<int64_t> data{1,
                              5,
                              3,
                              INT64_MAX,
                              -0,
                              -43,
                              9223372036854775807,
                              9223372036854775806,
                              INT64_MAX,
                              -9223372036854775807,
                              -9223372036854775806,
                              67};
    params->expected_data = data;
    params->expected_status = da_status_missing_data;
    params->datatype = "integer";
}

template <> void GetMissingData<int32_t>(CSVParamType<int32_t> *params) {

    params->filename = "csv_test_int32_missing_data";
    params->expected_rows = 3;
    params->expected_columns = 4;
    std::vector<int32_t> data{1,   5,   3,         INT32_MAX, -0,   -43,
                              922, 922, INT32_MAX, -922,      -922, 67};
    params->expected_data = data;
    params->expected_status = da_status_missing_data;
    params->datatype = "integer";
}

template <> void GetMissingData<char *>(CSVParamType<char *> *params) {

    params->filename = "csv_test_char_missing_data";
    params->expected_rows = 3;
    params->expected_columns = 4;
    std::vector<std::string> data = {"lorem",       "",      "dolor", "sit", "amet",
                                     "consectetur", "",      "edit",  "sed", "do",
                                     "eiusmod",     "tempor"};
    params->expected_char_data = data;
    params->expected_status = da_status_success;
    params->datatype = "string";
}

template <> void GetMissingData<uint8_t>(CSVParamType<uint8_t> *params) {

    params->filename = "csv_test_bool_missing_data";
    params->expected_rows = 2;
    params->expected_columns = 4;
    std::vector<uint8_t> data{1, 1, UINT8_MAX, 1, 1, 0, UINT8_MAX, 0};
    params->expected_data = data;
    params->expected_status = da_status_missing_data;
    params->datatype = "boolean";
}
