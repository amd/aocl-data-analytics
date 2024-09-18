/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclda.h"
#include "euclidean_distance.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <list>
#include <string>

template <typename T> class EDTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

template <typename T> struct EDDataType {

    std::string name;

    da_int m = 0, n = 0, k = 0, ldx = 0, ldy = 0, ldd;

    std::vector<T> X, Y, X_norms, Y_norms;

    T tol;
};

template <typename T> struct EDParamType {

    bool X_is_Y = false, square = false;

    da_int compute_X_norms = 0, compute_Y_norms = 0;
};

template <typename T>
void check_answer(EDDataType<T> data, EDParamType<T> params, T *D, T *X_norms,
                  T *Y_norms) {
    // This function assumes any X_norms or Y_norms supplied to the Euclidean distance routine actually correspond to the relevant norms
    // It computes Euclidean distances explicitly, instead of via the shortcut used in the library,
    // with a few inefficient checks and things to ensure we are computing the right thing depending on the parameters

    da_int m = data.m;
    da_int n = data.n;
    da_int ldx = data.ldx;
    da_int ldy = data.ldy;
    T *X = data.X.data();
    T *Y = data.Y.data();

    std::vector<T> X_norms_exp(data.m, 0);
    if (params.compute_X_norms) {
        for (da_int j = 0; j < data.k; j++) {
            for (da_int i = 0; i < m; i++) {
                X_norms_exp[i] += X[i + j * ldx] * X[i + j * ldx];
            }
        }
    } else {
        for (da_int i = 0; i < m; i++)
            X_norms_exp[i] = data.X_norms[i];
    }

    std::vector<T> Y_norms_exp(data.n, 0);
    if (params.compute_Y_norms) {
        for (da_int j = 0; j < data.k; j++) {
            for (da_int i = 0; i < n; i++) {
                Y_norms_exp[i] += Y[i + j * ldy] * Y[i + j * ldy];
            }
        }
    } else {
        for (da_int i = 0; i < n; i++)
            Y_norms_exp[i] = data.Y_norms[i];
    }

    EXPECT_ARR_NEAR(m, X_norms, X_norms_exp.data(), data.tol);
    EXPECT_ARR_NEAR(n, Y_norms, Y_norms_exp.data(), data.tol);

    // From this point on (but not before) is X_is_Y then point Y to X
    if (params.X_is_Y) {
        n = m;
        Y = data.X.data();
        ldy = ldx;
    }

    std::vector<T> D_exp(data.ldd * n, 0.0);

    for (da_int j = 0; j < n; j++) {
        for (da_int i = 0; i < m; i++) {
            for (da_int k = 0; k < data.k; k++) {
                D_exp[i + data.ldd * j] +=
                    (X[i + k * ldx] - Y[j + k * ldy]) * (X[i + k * ldx] - Y[j + k * ldy]);
            }
        }
    }

    if (params.compute_X_norms == 0) {
        for (da_int j = 0; j < n; j++) {
            for (da_int i = 0; i < m; i++) {
                D_exp[i + data.ldd * j] -= X_norms_exp[i];
                if (params.X_is_Y)
                    D_exp[i + data.ldd * j] -= X_norms_exp[j];
            }
        }
    }

    if (params.compute_Y_norms == 0 && !(params.X_is_Y)) {
        for (da_int j = 0; j < n; j++) {
            for (da_int i = 0; i < m; i++) {
                D_exp[i + data.ldd * j] -= Y_norms_exp[j];
            }
        }
    }

    if (params.square == false) {
        for (da_int j = 0; j < n; j++) {
            for (da_int i = 0; i < m; i++) {
                D_exp[i + data.ldd * j] = std::sqrt(D_exp[i + m * j]);
            }
        }
    }

    if (params.X_is_Y) {
        // Set lower triangle to 0
        for (da_int j = 0; j < n; j++) {
            for (da_int i = j + 1; i < m; i++) {
                D_exp[i + data.ldd * j] = 0.0;
            }
        }
    }

    EXPECT_ARR_NEAR(data.ldd * n, D, D_exp.data(), data.tol);
}

template <typename T> void Get1by1Data(std::vector<EDDataType<T>> &data) {
    EDDataType<T> test;

    test.name = "1 by 1";
    test.m = 1;
    test.n = 1;
    test.k = 1;
    test.ldx = 1;
    test.ldy = 1;
    test.ldd = 1;

    std::vector<double> VecX{2.1}, VecY{4.3}, VecX2{4.41}, VecY2{18.49};
    test.X = convert_vector<double, T>(VecX);
    test.X_norms = convert_vector<double, T>(VecX2);
    test.Y = convert_vector<double, T>(VecY);
    test.Y_norms = convert_vector<double, T>(VecY2);

    test.tol = 100 * std::numeric_limits<T>::epsilon();

    data.push_back(test);
}

template <typename T> void GetSingleRowData(std::vector<EDDataType<T>> &data) {
    EDDataType<T> test;

    test.name = "Single row";
    test.m = 1;
    test.n = 1;
    test.k = 4;
    test.ldx = 1;
    test.ldy = 1;
    test.ldd = 1;

    std::vector<double> VecX{1.0, 2.0, -3.0, 1.0}, VecY{0.0, 1.0, 4.0, -2.0}, VecX2{15.0},
        VecY2{21.0};
    test.X = convert_vector<double, T>(VecX);
    test.X_norms = convert_vector<double, T>(VecX2);
    test.Y = convert_vector<double, T>(VecY);
    test.Y_norms = convert_vector<double, T>(VecY2);

    test.tol = 100 * std::numeric_limits<T>::epsilon();

    data.push_back(test);
}

template <typename T> void GetSingleColData(std::vector<EDDataType<T>> &data) {
    EDDataType<T> test;

    test.name = "Single column";
    test.m = 3;
    test.n = 2;
    test.k = 1;
    test.ldx = 3;
    test.ldy = 2;
    test.ldd = 3;

    std::vector<double> VecX{1.0, 2.0, -3.0}, VecY{4.0, -2.0}, VecX2{1.0, 4.0, 9.0},
        VecY2{16.0, 4.0};
    test.X = convert_vector<double, T>(VecX);
    test.X_norms = convert_vector<double, T>(VecX2);
    test.Y = convert_vector<double, T>(VecY);
    test.Y_norms = convert_vector<double, T>(VecY2);

    test.tol = 100 * std::numeric_limits<T>::epsilon();

    data.push_back(test);
}

template <typename T> void GetTypicalData(std::vector<EDDataType<T>> &data) {
    EDDataType<T> test;

    test.name = "Typical data";
    test.m = 3;
    test.n = 2;
    test.k = 3;
    test.ldx = 3;
    test.ldy = 2;
    test.ldd = 3;

    std::vector<double> VecX{1.0, 2.0, -3.0, 1.0, -1.0, 2.0, 0.0, -2.0, 4.0},
        VecY{4.0, -2.0, 3.0, 3.0, -1.0, 2.0}, VecX2{2.0, 9.0, 29.0}, VecY2{26.0, 17.0};
    test.X = convert_vector<double, T>(VecX);
    test.X_norms = convert_vector<double, T>(VecX2);
    test.Y = convert_vector<double, T>(VecY);
    test.Y_norms = convert_vector<double, T>(VecY2);

    test.tol = 100 * std::numeric_limits<T>::epsilon();

    data.push_back(test);
}

template <typename T> void GetSubarrayData(std::vector<EDDataType<T>> &data) {
    EDDataType<T> test;

    test.name = "Subarray data";
    test.m = 3;
    test.n = 2;
    test.k = 3;
    test.ldx = 4;
    test.ldy = 4;
    test.ldd = 3;

    std::vector<double> VecX{1.0, 2.0, -3.0, 0.0,  1.0, -1.0,
                             2.0, 0.0, 0.0,  -2.0, 4.0, 0.0},
        VecY{4.0, -2.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, -1.0, 2.0, 0.0, 0.0},
        VecX2{2.0, 9.0, 29.0}, VecY2{26.0, 17.0};
    test.X = convert_vector<double, T>(VecX);
    test.X_norms = convert_vector<double, T>(VecX2);
    test.Y = convert_vector<double, T>(VecY);
    test.Y_norms = convert_vector<double, T>(VecY2);

    test.tol = 100 * std::numeric_limits<T>::epsilon();

    data.push_back(test);
}

template <typename T> void GetEDData(std::vector<EDDataType<T>> &data) {

    Get1by1Data(data);
    GetSingleRowData(data);
    GetSingleColData(data);
    GetTypicalData(data);
    GetSubarrayData(data);
}

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(EDTest, FloatTypes);

TYPED_TEST(EDTest, Euclidean_Distance) {

    std::vector<EDDataType<TypeParam>> data;
    GetEDData(data);

    std::vector<EDParamType<TypeParam>> params;

    // Various combinations of parameters we want to try
    EDParamType<TypeParam> param1{
        false, true, 0,
        0}; //X_is_Y=false, square = true, compute_X_norm = 0, compute_Y_norm = 0
    params.push_back(param1);

    EDParamType<TypeParam> param2{
        true, true, 0,
        0}; //X_is_Y=true, square = true, compute_X_norm = 0, compute_Y_norm = 0
    params.push_back(param2);

    EDParamType<TypeParam> param3{
        true, true, 1,
        1}; //X_is_Y=true, square = true, compute_X_norm = 1, compute_Y_norm = 1
    params.push_back(param3);

    EDParamType<TypeParam> param4{
        false, false, 1,
        1}; //X_is_Y=false, square = false, compute_X_norm = 1, compute_Y_norm = 1
    params.push_back(param4);

    EDParamType<TypeParam> param5{
        false, false, 2,
        2}; //X_is_Y=false, square = false, compute_X_norm = 2, compute_Y_norm = 2
    params.push_back(param5);

    EDParamType<TypeParam> param6{
        false, true, 0,
        2}; //X_is_Y=false, square = true, compute_X_norm = 0, compute_Y_norm = 2
    params.push_back(param6);

    EDParamType<TypeParam> param7{
        false, true, 2,
        0}; //X_is_Y=false, square = true, compute_X_norm = 2, compute_Y_norm = 0
    params.push_back(param7);

    EDParamType<TypeParam> param8{
        true, true, 2,
        0}; //X_is_Y=true, square = true, compute_X_norm = 2, compute_Y_norm = 2
    params.push_back(param8);

    da_int count = 0;
    for (auto &test : data) {
        for (auto &param : params) {
            count++;
            std::cout << "Test " << std::to_string(count) << ": " << test.name
                      << " with {X_is_y, square, compute_X_norm, compute_Y_norm} = {"
                      << param.X_is_Y << ", " << param.square << ", "
                      << param.compute_X_norms << ", " << param.compute_Y_norms << "}"
                      << std::endl;

            da_int D_cols = (param.X_is_Y) ? test.m : test.n;
            std::vector<TypeParam> D(test.ldd * D_cols, 0.0);
            std::vector<TypeParam> X_norms(test.X_norms);
            std::vector<TypeParam> Y_norms(test.Y_norms);

            euclidean_distance(test.m, test.n, test.k, test.X.data(), test.ldx,
                               test.Y.data(), test.ldy, D.data(), test.ldd,
                               X_norms.data(), param.compute_X_norms, Y_norms.data(),
                               param.compute_Y_norms, param.square, param.X_is_Y);

            check_answer(test, param, D.data(), X_norms.data(), Y_norms.data());
        }
    }
}