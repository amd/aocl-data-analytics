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

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <list>

template <typename T> class CorrelationCovarianceTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

template <typename T> struct CovCorrParamType {
    da_int n = 0;
    da_int p = 0;
    da_int ldx = 0;
    da_int dof = 0;
    da_int ldcov = 0;
    da_int ldcorr = 0;
    std::vector<T> x;
    std::vector<T> expected_cov;
    std::vector<T> expected_corr;

    da_status expected_status = da_status_success;
    T epsilon = std::numeric_limits<T>::epsilon();
};

template <typename T> void GetSubarrayData(std::vector<CovCorrParamType<T>> &params) {
    // Test with data stored in a subarray
    CovCorrParamType<T> param;
    param.n = 5;
    param.p = 6;
    param.ldx = param.n + 1;
    param.ldcov = param.p + 1;
    param.ldcorr = param.p + 1;

    std::vector<double> x{3, 7, 4, 2, 7, 0, 0,  0,  4,  7,  2,  0, -1, -4, 5, -3, 0, 0,
                          6, 8, 5, 4, 4, 0, -5, -5, -5, -5, -7, 0, 1,  2,  3, 4,  5, 0};
    param.x = convert_vector<double, T>(x);

    std::vector<double> expected_cov{
        5.3,  -3.95, -0.8, 1.45, -1.2, 0.75, 0, -3.95, 8.8,   1.7,  -3.55, 0.3, 2.75, 0,
        -0.8, 1.7,   12.3, -2.2, -0.3, 0.75, 0, 1.45,  -3.55, -2.2, 2.8,   0.7, -2.,  0,
        -1.2, 0.3,   -0.3, 0.7,  0.8,  -1.,  0, 0.75,  2.75,  0.75, -2.,   -1., 2.5,  0};
    param.expected_cov = convert_vector<double, T>(expected_cov);
    std::vector<double> expected_corr{1.,
                                      -0.578386069999205,
                                      -0.0990830796106615,
                                      0.3764012454470947,
                                      -0.5827715174143585,
                                      0.2060408459230335,
                                      0,
                                      -0.578386069999205,
                                      1.,
                                      0.1634011202231184,
                                      -0.715167880572525,
                                      0.1130667542166614,
                                      0.5863019699779287,
                                      0,
                                      -0.0990830796106615,
                                      0.1634011202231184,
                                      1.,
                                      -0.3748789971250484,
                                      -0.0956365069595008,
                                      0.1352504452001148,
                                      0,
                                      0.3764012454470947,
                                      -0.715167880572525,
                                      -0.3748789971250484,
                                      1.,
                                      0.4677071733467426,
                                      -0.7559289460184544,
                                      0,
                                      -0.5827715174143586,
                                      0.1130667542166614,
                                      -0.0956365069595008,
                                      0.4677071733467427,
                                      1.,
                                      -0.7071067811865475,
                                      0,
                                      0.2060408459230335,
                                      0.5863019699779286,
                                      0.1352504452001148,
                                      -0.7559289460184545,
                                      -0.7071067811865476,
                                      1.,
                                      0};
    param.expected_corr = convert_vector<double, T>(expected_corr);

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    param.expected_status = da_status_success;

    params.push_back(param);
}

template <typename T> void GetStandardData(std::vector<CovCorrParamType<T>> &params) {
    // Test with standard data
    CovCorrParamType<T> param;
    param.n = 5;
    param.p = 6;
    param.dof = -3;
    param.ldx = param.n;
    param.ldcov = param.p;
    param.ldcorr = param.p;

    std::vector<double> x{3, 7, 4, 2, 7, 0,  0,  4,  7,  2,  -1, -4, 5, -3, 0,
                          6, 8, 5, 4, 4, -5, -5, -5, -5, -7, 1,  2,  3, 4,  5};
    param.x = convert_vector<double, T>(x);

    std::vector<double> expected_cov{
        4.24,  -3.16, -0.64, 1.16,  -0.96, 0.6,  -3.16, 7.04,  1.36,  -2.84, 0.24, 2.2,
        -0.64, 1.36,  9.84,  -1.76, -0.24, 0.6,  1.16,  -2.84, -1.76, 2.24,  0.56, -1.6,
        -0.96, 0.24,  -0.24, 0.56,  0.64,  -0.8, 0.6,   2.2,   0.6,   -1.6,  -0.8, 2.0};
    param.expected_cov = convert_vector<double, T>(expected_cov);
    std::vector<double> expected_corr{1.,
                                      -0.578386069999205,
                                      -0.0990830796106615,
                                      0.3764012454470947,
                                      -0.5827715174143585,
                                      0.2060408459230335,
                                      -0.578386069999205,
                                      1.,
                                      0.1634011202231184,
                                      -0.715167880572525,
                                      0.1130667542166614,
                                      0.5863019699779287,
                                      -0.0990830796106615,
                                      0.1634011202231184,
                                      1.,
                                      -0.3748789971250484,
                                      -0.0956365069595008,
                                      0.1352504452001148,
                                      0.3764012454470947,
                                      -0.715167880572525,
                                      -0.3748789971250484,
                                      1.,
                                      0.4677071733467426,
                                      -0.7559289460184544,
                                      -0.5827715174143586,
                                      0.1130667542166614,
                                      -0.0956365069595008,
                                      0.4677071733467427,
                                      1.,
                                      -0.7071067811865475,
                                      0.2060408459230335,
                                      0.5863019699779286,
                                      0.1352504452001148,
                                      -0.7559289460184545,
                                      -0.7071067811865476,
                                      1.};
    param.expected_corr = convert_vector<double, T>(expected_corr);

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    param.expected_status = da_status_success;

    params.push_back(param);
}

template <typename T> void GetZeroData(std::vector<CovCorrParamType<T>> &params) {
    // Test with zero data
    CovCorrParamType<T> param;
    param.n = 5;
    param.p = 3;
    param.ldx = param.n;
    param.ldcov = param.p;
    param.ldcorr = param.p;

    std::vector<double> x{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    param.x = convert_vector<double, T>(x);

    std::vector<double> expected_cov{0, 0, 0, 0, 0, 0, 0, 0, 0};
    param.expected_cov = convert_vector<double, T>(expected_cov);
    std::vector<double> expected_corr{1, 0, 0, 0, 1, 0, 0, 0, 1};
    param.expected_corr = convert_vector<double, T>(expected_corr);

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    param.expected_status = da_status_success;

    params.push_back(param);
}

template <typename T> void GetSingleColumnData(std::vector<CovCorrParamType<T>> &params) {
    // Test with a single column
    CovCorrParamType<T> param;
    param.n = 5;
    param.p = 1;
    param.ldx = param.n;
    param.ldcov = param.p;
    param.ldcorr = param.p;

    std::vector<double> x{2.1, 4.3, 5.6, 0.3, -1.3};
    param.x = convert_vector<double, T>(x);

    std::vector<double> expected_cov{7.96};
    param.expected_cov = convert_vector<double, T>(expected_cov);
    std::vector<double> expected_corr{1};
    param.expected_corr = convert_vector<double, T>(expected_corr);

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    param.expected_status = da_status_success;

    params.push_back(param);
}

template <typename T> void GetSmallData(std::vector<CovCorrParamType<T>> &params) {
    // Test with a small problem
    CovCorrParamType<T> param;
    param.n = 3;
    param.p = 2;
    param.dof = 5;
    param.ldx = param.n;
    param.ldcov = param.p;
    param.ldcorr = param.p;

    std::vector<double> x{0, 1, 2, 2, 1, 0};
    param.x = convert_vector<double, T>(x);

    std::vector<double> expected_cov{0.4, -0.4, -0.4, 0.4};
    param.expected_cov = convert_vector<double, T>(expected_cov);
    std::vector<double> expected_corr{1, -1, -1, 1};
    param.expected_corr = convert_vector<double, T>(expected_corr);

    param.epsilon = 10 * std::numeric_limits<T>::epsilon();

    param.expected_status = da_status_success;

    params.push_back(param);
}

template <typename T> void GetCovCorrData(std::vector<CovCorrParamType<T>> &params) {

    GetStandardData(params);
    GetZeroData(params);
    GetSubarrayData(params);
    GetSingleColumnData(params);
    GetSmallData(params);
}

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(CorrelationCovarianceTest, FloatTypes);

TYPED_TEST(CorrelationCovarianceTest, CorrelationCovarianceFunctionality) {

    std::vector<CovCorrParamType<TypeParam>> params;
    GetCovCorrData(params);

    for (auto &param : params) {
        std::vector<TypeParam> cov(param.ldcov * param.p);
        std::vector<TypeParam> corr(param.ldcorr * param.p);
        std::vector<TypeParam> xcov(param.x);
        std::vector<TypeParam> xcorr(param.x);

        EXPECT_EQ(da_covariance_matrix(param.n, param.p, xcov.data(), param.ldx,
                                       param.dof, cov.data(), param.ldcov),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.ldx * param.p, param.x.data(), xcov.data(), param.epsilon);
        EXPECT_ARR_NEAR(param.ldcov * param.p, param.expected_cov.data(), cov.data(),
                        param.epsilon);
        EXPECT_EQ(da_correlation_matrix(param.n, param.p, xcorr.data(), param.ldx,
                                        corr.data(), param.ldcorr),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.ldx * param.p, param.x.data(), xcorr.data(), param.epsilon);
        EXPECT_ARR_NEAR(param.ldcorr * param.p, param.expected_corr.data(), corr.data(),
                        param.epsilon);
    }
}

TYPED_TEST(CorrelationCovarianceTest, IllegalArgsCorrelationCovariance) {

    std::vector<double> x_d{4.7, 1.2, -0.3, 4.5};
    std::vector<TypeParam> x = convert_vector<double, TypeParam>(x_d);
    da_int n = 2, p = 2, ldx = 2, ldmat = 2, dof = 0;
    std::vector<TypeParam> mat(4, 0);

    // Test with illegal value of ldx
    da_int ldx_illegal = 1;
    EXPECT_EQ(da_covariance_matrix(n, p, x.data(), ldx_illegal, dof, mat.data(), ldmat),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_correlation_matrix(n, p, x.data(), ldx_illegal, mat.data(), ldmat),
              da_status_invalid_leading_dimension);

    // Test with illegal p
    da_int p_illegal = 0;
    EXPECT_EQ(da_covariance_matrix(n, p_illegal, x.data(), ldx, dof, mat.data(), ldmat),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_correlation_matrix(n, p_illegal, x.data(), ldx, mat.data(), ldmat),
              da_status_invalid_array_dimension);

    // Test with illegal n
    da_int n_illegal = 1;
    EXPECT_EQ(da_covariance_matrix(n_illegal, p, x.data(), ldx, dof, mat.data(), ldmat),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_correlation_matrix(n_illegal, p, x.data(), ldx, mat.data(), ldmat),
              da_status_invalid_array_dimension);

    // Test with illegal ldmat
    da_int ldmat_illegal = 1;
    EXPECT_EQ(da_covariance_matrix(n, p, x.data(), ldx, dof, mat.data(), ldmat_illegal),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_correlation_matrix(n, p, x.data(), ldx, mat.data(), ldmat_illegal),
              da_status_invalid_leading_dimension);

    // Test illegal pointers
    TypeParam *matrixnull = nullptr;
    EXPECT_EQ(da_covariance_matrix(n, p, matrixnull, ldx, dof, mat.data(), ldmat),
              da_status_invalid_pointer);
    EXPECT_EQ(da_correlation_matrix(n, p, matrixnull, ldx, mat.data(), ldmat),
              da_status_invalid_pointer);
    EXPECT_EQ(da_covariance_matrix(n, p, x.data(), ldx, dof, matrixnull, ldmat),
              da_status_invalid_pointer);
    EXPECT_EQ(da_correlation_matrix(n, p, x.data(), ldx, matrixnull, ldmat),
              da_status_invalid_pointer);
}