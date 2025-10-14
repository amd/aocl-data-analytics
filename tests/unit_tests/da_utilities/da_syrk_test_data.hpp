/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "../utest_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <limits>
#include <list>
#include <random>
#include <string.h>

template <typename T>
void fill_with_uniform_random(std::vector<T> &vec, da_int seed = 474) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    std::generate(vec.begin(), vec.end(), [&]() { return dist(gen); });
}

template <typename T> struct SYRKParamType {

    std::string test_name;

    // n is always the output dim - so C is always n by n
    da_int n = 0;
    da_int k = 0;
    std::vector<T> A;
    da_int ldA = 0;
    da_int ldC = 0;
    da_order order = column_major;
    da_uplo uplo = da_upper;
    da_transpose transpose = da_trans;
    T alpha = 1.0;
    T beta = 0.0;
    T c_init_value = 1.54;
    da_status expected_status = da_status_success;
    T epsilon = 100 * std::numeric_limits<T>::epsilon();
    std::string block_size_override;
};

template <typename T> void Get1by1Data(std::vector<SYRKParamType<T>> &params) {
    // Test with a 1 x 1 data matrix
    SYRKParamType<T> param;
    param.test_name = "1 by 1 data matrix";
    param.n = 1;
    param.k = 1;
    std::vector<double> A{1.23};
    param.A = convert_vector<double, T>(A);
    param.ldA = 1;
    param.ldC = 1;
    params.push_back(param);
}

template <typename T> void Get1ColumnData(std::vector<SYRKParamType<T>> &params) {
    // Test with a 1 column matrix
    SYRKParamType<T> param;
    param.test_name = "1 column matrix";
    param.n = 1;
    param.k = 6;
    std::vector<double> A{1.23, 4.56, 1.1, 2.4, 6.3, -0.5};
    param.A = convert_vector<double, T>(A);
    param.ldA = 6;
    param.ldC = 1;
    param.block_size_override = "2";
    params.push_back(param);
}

template <typename T> void GetTallSkinnyData1(std::vector<SYRKParamType<T>> &params) {
    // Tall skinny data - col major, upper storage, zero beta
    SYRKParamType<T> param;
    param.test_name = "Tall skinny data 1";
    param.n = 2;
    param.k = 700;

    std::vector<double> A(param.n * param.k);
    fill_with_uniform_random(A);
    param.A = convert_vector<double, T>(A);

    param.ldA = 700;
    param.ldC = 2;
    param.epsilon = 1000 * std::numeric_limits<T>::max();
    param.block_size_override = "3";
    params.push_back(param);
}

template <typename T> void GetTallSkinnyData2(std::vector<SYRKParamType<T>> &params) {
    // Tall skinny data - col major, lower storage, zero beta
    SYRKParamType<T> param;
    param.test_name = "Tall skinny data 2";
    param.n = 6;
    param.k = 80;

    std::vector<double> A(param.n * param.k);
    fill_with_uniform_random(A);
    param.A = convert_vector<double, T>(A);

    param.ldA = 80;
    param.ldC = 6;
    param.block_size_override = "15";
    param.uplo = da_lower;
    params.push_back(param);
}

template <typename T> void GetTallSkinnyData3(std::vector<SYRKParamType<T>> &params) {
    // Tall skinny data - row major, upper storage, zero beta
    SYRKParamType<T> param;
    param.test_name = "Tall skinny data 3";
    param.n = 6;
    param.k = 80;

    std::vector<double> A(param.n * param.k);
    fill_with_uniform_random(A);
    param.A = convert_vector<double, T>(A);

    param.ldA = 6;
    param.ldC = 6;
    param.order = row_major;
    param.block_size_override = "40";
    params.push_back(param);
}

template <typename T> void GetTallSkinnyData4(std::vector<SYRKParamType<T>> &params) {
    // Tall skinny data - row major, lower storage, zero beta
    SYRKParamType<T> param;
    param.test_name = "Tall skinny data 4";
    param.n = 5;
    param.k = 50;

    std::vector<double> A(param.n * param.k);
    fill_with_uniform_random(A);
    param.A = convert_vector<double, T>(A);

    param.ldA = 5;
    param.ldC = 5;
    param.order = row_major;
    param.uplo = da_lower;
    param.block_size_override = "9";
    params.push_back(param);
}

template <typename T> void GetTallSkinnyData5(std::vector<SYRKParamType<T>> &params) {
    // Tall skinny data - col major, upper storage, non-zero beta
    SYRKParamType<T> param;
    param.test_name = "Tall skinny data 5";
    param.n = 6;
    param.k = 80;

    std::vector<double> A(param.n * param.k);
    fill_with_uniform_random(A);
    param.A = convert_vector<double, T>(A);

    param.ldA = 80;
    param.ldC = 6;
    param.block_size_override = "9";
    param.beta = 1.5;
    params.push_back(param);
}

template <typename T> void GetTallSkinnyData6(std::vector<SYRKParamType<T>> &params) {
    // Tall skinny data - col major, lower storage, non-zero beta
    SYRKParamType<T> param;
    param.test_name = "Tall skinny data 6";
    param.n = 6;
    param.k = 80;

    std::vector<double> A(param.n * param.k);
    fill_with_uniform_random(A);
    param.A = convert_vector<double, T>(A);

    param.ldA = 80;
    param.ldC = 6;
    param.block_size_override = "15";
    param.uplo = da_lower;
    param.beta = -3.0;
    params.push_back(param);
}

template <typename T> void GetTallSkinnyData7(std::vector<SYRKParamType<T>> &params) {
    // Tall skinny data - row major, upper storage, non-zero beta
    SYRKParamType<T> param;
    param.test_name = "Tall skinny data 7";
    param.n = 2;
    param.k = 20;

    std::vector<double> A(param.n * param.k);
    fill_with_uniform_random(A);
    param.A = convert_vector<double, T>(A);

    param.ldA = 2;
    param.ldC = 2;
    param.order = row_major;
    param.block_size_override = "5";
    param.beta = 1.6;
    params.push_back(param);
}

template <typename T> void GetTallSkinnyData8(std::vector<SYRKParamType<T>> &params) {
    // Tall skinny data - row major, lower storage, non-zero beta
    SYRKParamType<T> param;
    param.test_name = "Tall skinny data 8";
    param.n = 5;
    param.k = 50;

    std::vector<double> A(param.n * param.k);
    fill_with_uniform_random(A);
    param.A = convert_vector<double, T>(A);

    param.ldA = 5;
    param.ldC = 5;
    param.order = row_major;
    param.uplo = da_lower;
    param.block_size_override = "9";
    param.beta = 4;
    params.push_back(param);
}

template <typename T> void GetShortWideData1(std::vector<SYRKParamType<T>> &params) {
    // Short wide data - col major, upper storage, zero beta
    SYRKParamType<T> param;
    param.test_name = "Short Wide data 1";
    param.n = 2;
    param.k = 700;

    std::vector<double> A(param.n * param.k);
    fill_with_uniform_random(A);
    param.A = convert_vector<double, T>(A);

    param.ldA = 2;
    param.ldC = 2;
    param.block_size_override = "3";
    param.transpose = da_no_trans;
    param.epsilon = 1000 * std::numeric_limits<T>::max();
    params.push_back(param);
}

template <typename T> void GetShortWideData2(std::vector<SYRKParamType<T>> &params) {
    // Short wide data - col major, lower storage, zero beta
    SYRKParamType<T> param;
    param.test_name = "Short Wide data 2";
    param.n = 6;
    param.k = 80;

    std::vector<double> A(param.n * param.k);
    fill_with_uniform_random(A);
    param.A = convert_vector<double, T>(A);

    param.ldA = 6;
    param.ldC = 6;
    param.block_size_override = "15";
    param.uplo = da_lower;
    param.transpose = da_no_trans;
    params.push_back(param);
}

template <typename T> void GetShortWideData3(std::vector<SYRKParamType<T>> &params) {
    // Short wide data - row major, upper storage, zero beta
    SYRKParamType<T> param;
    param.test_name = "Short Wide data 3";
    param.n = 6;
    param.k = 80;

    std::vector<double> A(param.n * param.k);
    fill_with_uniform_random(A);
    param.A = convert_vector<double, T>(A);

    param.ldA = 80;
    param.ldC = 6;
    param.order = row_major;
    param.block_size_override = "40";
    param.transpose = da_no_trans;
    params.push_back(param);
}

template <typename T> void GetShortWideData4(std::vector<SYRKParamType<T>> &params) {
    // Short Wide data - row major, lower storage, zero beta
    SYRKParamType<T> param;
    param.test_name = "Short Wide data 4";
    param.n = 5;
    param.k = 50;

    std::vector<double> A(param.n * param.k);
    fill_with_uniform_random(A);
    param.A = convert_vector<double, T>(A);

    param.ldA = 50;
    param.ldC = 5;
    param.order = row_major;
    param.uplo = da_lower;
    param.block_size_override = "9";
    param.transpose = da_no_trans;
    params.push_back(param);
}

template <typename T> void GetShortWideData5(std::vector<SYRKParamType<T>> &params) {
    // Short Wide data - col major, upper storage, non-zero beta
    SYRKParamType<T> param;
    param.test_name = "Short Wide data 5";
    param.n = 6;
    param.k = 80;

    std::vector<double> A(param.n * param.k);
    fill_with_uniform_random(A);
    param.A = convert_vector<double, T>(A);

    param.ldA = 6;
    param.ldC = 6;
    param.block_size_override = "9";
    param.beta = 1.5;
    param.transpose = da_no_trans;
    params.push_back(param);
}

template <typename T> void GetShortWideData6(std::vector<SYRKParamType<T>> &params) {
    // Short Wide data - col major, lower storage, non-zero beta
    SYRKParamType<T> param;
    param.test_name = "Short Wide data 6";
    param.n = 6;
    param.k = 80;

    std::vector<double> A(param.n * param.k);
    fill_with_uniform_random(A);
    param.A = convert_vector<double, T>(A);

    param.ldA = 6;
    param.ldC = 6;
    param.block_size_override = "15";
    param.uplo = da_lower;
    param.beta = -3.0;
    param.transpose = da_no_trans;
    params.push_back(param);
}

template <typename T> void GetShortWideData7(std::vector<SYRKParamType<T>> &params) {
    // Short Wide data - row major, upper storage, non-zero beta
    SYRKParamType<T> param;
    param.test_name = "Short Wide data 7";
    param.n = 2;
    param.k = 20;

    std::vector<double> A(param.n * param.k);
    fill_with_uniform_random(A);
    param.A = convert_vector<double, T>(A);

    param.ldA = 20;
    param.ldC = 2;
    param.order = row_major;
    param.block_size_override = "5";
    param.beta = 1.6;
    param.transpose = da_no_trans;
    params.push_back(param);
}

template <typename T> void GetShortWideData8(std::vector<SYRKParamType<T>> &params) {
    // Short Wide data - row major, lower storage, non-zero beta
    SYRKParamType<T> param;
    param.test_name = "Short Wide data 8";
    param.n = 5;
    param.k = 50;

    std::vector<double> A(param.n * param.k);
    fill_with_uniform_random(A);
    param.A = convert_vector<double, T>(A);

    param.ldA = 50;
    param.ldC = 5;
    param.order = row_major;
    param.uplo = da_lower;
    param.block_size_override = "9";
    param.beta = 4;
    param.transpose = da_no_trans;
    params.push_back(param);
}

template <typename T> void GetSubarrayInputData1(std::vector<SYRKParamType<T>> &params) {
    // subarray data - col major, tall skinny
    SYRKParamType<T> param;
    param.test_name = "Subarray input data 1";
    param.n = 2;
    param.k = 10;
    std::vector<double> A{1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                          9.0,  10.0, 0.0,  0.0,  -0.5, -1.0, -1.5, -2.0,
                          -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, 0.0,  0.0};

    param.A = convert_vector<double, T>(A);
    param.ldA = 12;
    param.ldC = 2;
    param.block_size_override = "3";
    params.push_back(param);
}

template <typename T> void GetSubarrayInputData2(std::vector<SYRKParamType<T>> &params) {
    // subarray data - row major, tall skinny
    SYRKParamType<T> param;
    param.test_name = "Subarray input data 2";
    param.n = 2;
    param.k = 10;
    std::vector<double> A{1.0,  2.0,  0.0,  0.0,  3.0,  4.0,  0.0,  0.0,  5.0,  6.0,
                          0.0,  0.0,  7.0,  8.0,  0.0,  0.0,  9.0,  10.0, 0.0,  0.0,
                          -0.5, -1.0, 0.0,  0.0,  -1.5, -2.0, 0.0,  0.0,  -2.5, -3.0,
                          0.0,  0.0,  -3.5, -4.0, 0.0,  0.0,  -4.5, -5.0, 0.0,  0.0};

    param.A = convert_vector<double, T>(A);
    param.ldA = 4;
    param.ldC = 2;
    param.block_size_override = "3";
    param.order = row_major;
    params.push_back(param);
}

template <typename T> void GetSubarrayInputData3(std::vector<SYRKParamType<T>> &params) {
    // subarray data - col major, short Wide
    SYRKParamType<T> param;
    param.test_name = "Subarray input data 3";
    param.n = 2;
    param.k = 10;
    std::vector<double> A{1.0,  2.0,  0.0,  0.0,  3.0,  4.0,  0.0,  0.0,  5.0,  6.0,
                          0.0,  0.0,  7.0,  8.0,  0.0,  0.0,  9.0,  10.0, 0.0,  0.0,
                          -0.5, -1.0, 0.0,  0.0,  -1.5, -2.0, 0.0,  0.0,  -2.5, -3.0,
                          0.0,  0.0,  -3.5, -4.0, 0.0,  0.0,  -4.5, -5.0, 0.0,  0.0};

    param.A = convert_vector<double, T>(A);
    param.ldA = 4;
    param.ldC = 2;
    param.transpose = da_no_trans;
    param.block_size_override = "5";
    params.push_back(param);
}

template <typename T> void GetSubarrayInputData4(std::vector<SYRKParamType<T>> &params) {
    // subarray data - row major, short Wide
    SYRKParamType<T> param;
    param.test_name = "Subarray input data 4";
    param.n = 2;
    param.k = 10;
    std::vector<double> A{1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                          9.0,  10.0, 0.0,  0.0,  -0.5, -1.0, -1.5, -2.0,
                          -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, 0.0,  0.0};

    param.A = convert_vector<double, T>(A);
    param.ldA = 12;
    param.ldC = 2;
    param.order = row_major;
    param.transpose = da_no_trans;
    param.block_size_override = "7";
    params.push_back(param);
}

template <typename T> void GetSubarrayOutputData1(std::vector<SYRKParamType<T>> &params) {
    // subarray data, output is also writing to subarray - col major, tall skinny
    SYRKParamType<T> param;
    param.test_name = "Subarray output data 1";
    param.n = 2;
    param.k = 10;
    std::vector<double> A{1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                          9.0,  10.0, 0.0,  0.0,  -0.5, -1.0, -1.5, -2.0,
                          -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, 0.0,  0.0};

    param.A = convert_vector<double, T>(A);
    param.ldA = 12;
    param.ldC = 4;
    param.block_size_override = "3";
    params.push_back(param);
}

template <typename T> void GetSubarrayOutputData2(std::vector<SYRKParamType<T>> &params) {
    // subarray data, output is also writing to subarray - row major, tall skinny
    SYRKParamType<T> param;
    param.test_name = "Subarray output  data 2";
    param.n = 2;
    param.k = 10;
    std::vector<double> A{1.0,  2.0,  0.0,  0.0,  3.0,  4.0,  0.0,  0.0,  5.0,  6.0,
                          0.0,  0.0,  7.0,  8.0,  0.0,  0.0,  9.0,  10.0, 0.0,  0.0,
                          -0.5, -1.0, 0.0,  0.0,  -1.5, -2.0, 0.0,  0.0,  -2.5, -3.0,
                          0.0,  0.0,  -3.5, -4.0, 0.0,  0.0,  -4.5, -5.0, 0.0,  0.0};

    param.A = convert_vector<double, T>(A);
    param.ldA = 4;
    param.ldC = 4;
    param.block_size_override = "3";
    param.order = row_major;
    params.push_back(param);
}

template <typename T> void GetSubarrayOutputData3(std::vector<SYRKParamType<T>> &params) {
    // subarray data, output is also writing to subarray - col major, short Wide
    SYRKParamType<T> param;
    param.test_name = "Subarray output data 3";
    param.n = 2;
    param.k = 10;
    std::vector<double> A{1.0,  2.0,  0.0,  0.0,  3.0,  4.0,  0.0,  0.0,  5.0,  6.0,
                          0.0,  0.0,  7.0,  8.0,  0.0,  0.0,  9.0,  10.0, 0.0,  0.0,
                          -0.5, -1.0, 0.0,  0.0,  -1.5, -2.0, 0.0,  0.0,  -2.5, -3.0,
                          0.0,  0.0,  -3.5, -4.0, 0.0,  0.0,  -4.5, -5.0, 0.0,  0.0};

    param.A = convert_vector<double, T>(A);
    param.ldA = 4;
    param.ldC = 6;
    param.transpose = da_no_trans;
    param.block_size_override = "5";
    params.push_back(param);
}

template <typename T> void GetSubarrayOutputData4(std::vector<SYRKParamType<T>> &params) {
    // subarray data, output is also writing to subarray - row major, short Wide
    SYRKParamType<T> param;
    param.test_name = "Subarray output data 4";
    param.n = 2;
    param.k = 10;
    std::vector<double> A{1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                          9.0,  10.0, 0.0,  0.0,  -0.5, -1.0, -1.5, -2.0,
                          -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, 0.0,  0.0};

    param.A = convert_vector<double, T>(A);
    param.ldA = 12;
    param.ldC = 3;
    param.order = row_major;
    param.transpose = da_no_trans;
    param.block_size_override = "7";
    params.push_back(param);
}

template <typename T> void GetSYRKData(std::vector<SYRKParamType<T>> &params) {
    Get1by1Data(params);
    Get1ColumnData(params);
    GetTallSkinnyData1(params);
    GetTallSkinnyData2(params);
    GetTallSkinnyData3(params);
    GetTallSkinnyData4(params);
    GetTallSkinnyData5(params);
    GetTallSkinnyData6(params);
    GetTallSkinnyData7(params);
    GetTallSkinnyData8(params);
    GetShortWideData1(params);
    GetShortWideData2(params);
    GetShortWideData3(params);
    GetShortWideData4(params);
    GetShortWideData5(params);
    GetShortWideData6(params);
    GetShortWideData7(params);
    GetShortWideData8(params);
    GetSubarrayInputData1(params);
    GetSubarrayInputData2(params);
    GetSubarrayInputData3(params);
    GetSubarrayInputData4(params);
    GetSubarrayOutputData1(params);
    GetSubarrayOutputData2(params);
    GetSubarrayOutputData3(params);
    GetSubarrayOutputData4(params);
}
