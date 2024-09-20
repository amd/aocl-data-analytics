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
#include <list>
#include <random>

// Add all available distance metrics in a list so that we can run all
// tests for each metric type.
static std::list<std::tuple<std::string, da_metric>> MetricType = {
    {"da_euclidean", da_euclidean}, {"da_sqeuclidean", da_sqeuclidean}};

template <typename T> class PairwiseDistanceTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

template <typename T> struct PairwiseDistanceParamType {

    std::string name;
    da_int m = 1, n = 1, k = 1, ldx = 1, ldy = 1, ldd = 1;
    std::vector<T> X, Y;
    da_metric metric = da_euclidean;
    da_data_types data_type = da_allow_infinite;
    da_status expected_status = da_status_success;
    T tol = std::numeric_limits<T>::epsilon();
    // Set constructor to initialize data in test bodies as simply as possible.
    PairwiseDistanceParamType(){};
    PairwiseDistanceParamType(da_int m, da_int n, da_int k, da_int ldx, da_int ldy,
                              da_int ldd, std::tuple<std::string, da_metric> metric)
        : m(m), n(n), k(k), ldx(ldx), ldy(ldy), ldd(ldd), metric(std::get<1>(metric)) {
        name = "m=" + std::to_string(m) + ", n=" + std::to_string(n) +
               ", k=" + std::to_string(k) + ", ldx=" + std::to_string(ldx) +
               ", ldy=" + std::to_string(ldy) + ", ldd=" + std::to_string(ldd) +
               ", metric=" + std::get<0>(metric);
        this->generateRandomData();
    };

    void generateRandomData() {
        // Initialize random number generator
        std::mt19937 generator(94);
        std::uniform_real_distribution<T> distr(T(-10.0), T(10.0));
        // Initialize with NaNs so that if leading dimension is larger than m
        // and the wrong element is being accessed, testing fails.
        // X is an mxk matrix, with leading dimension ldx
        X.resize(ldx * k, std::numeric_limits<T>::quiet_NaN());
        // X is an nxk matrix, with leading dimension ldx
        Y.resize(ldy * k, std::numeric_limits<T>::quiet_NaN());
        // Initialize matrices with random data
        for (auto j = 0; j < k; j++) {
            for (auto i = 0; i < m; i++)
                X[i + j * ldx] = distr(generator);
            for (auto i = 0; i < n; i++)
                Y[i + j * ldy] = distr(generator);
        }
    }
};

template <typename T>
void InitGenericData(std::vector<PairwiseDistanceParamType<T>> &data, da_int m, da_int n,
                     da_int k, da_int ldx, da_int ldy, da_int ldd) {
    // Iterate through the list of metrics and register all tests accordingly.
    for (auto const &i : MetricType) {
        PairwiseDistanceParamType<T> test(m, n, k, ldx, ldy, ldd, i);
        test.tol = 5000 * test.tol;
        data.push_back(test);
    }
}

template <typename T>
void InitGenericData(std::vector<PairwiseDistanceParamType<T>> &data, da_int m, da_int n,
                     da_int k) {
    // Iterate through the list of metrics and register all tests accordingly.
    for (auto const &i : MetricType) {
        PairwiseDistanceParamType<T> test(m, n, k, m, n, m, i);
        test.tol = 5000 * test.tol;
        data.push_back(test);
    }
}

// Returns the appropriate solution
template <typename T>
std::vector<T> reference_distance(PairwiseDistanceParamType<T> &data) {
    std::vector<T> D;
    if (data.ldy > 0) {
        D.resize(data.ldd * data.n, T{0.0});
        for (auto i = 0; i < data.m; i++) {
            for (auto ii = 0; ii < data.n; ii++) {
                for (auto j = 0; j < data.k; j++) {
                    D[i + ii * data.ldd] +=
                        (data.X[i + j * data.ldx] - data.Y[ii + j * data.ldy]) *
                        (data.X[i + j * data.ldx] - data.Y[ii + j * data.ldy]);
                }
                if (data.metric == da_euclidean)
                    D[i + ii * data.ldd] = std::sqrt(D[i + ii * data.ldd]);
            }
        }
    } else {
        D.resize(data.ldd * data.m, T{0.0});
        for (auto i = 0; i < data.m; i++) {
            for (auto ii = 0; ii < data.m; ii++) {
                for (auto j = 0; j < data.k; j++) {
                    D[i + ii * data.ldd] +=
                        (data.X[i + j * data.ldx] - data.X[ii + j * data.ldx]) *
                        (data.X[i + j * data.ldx] - data.X[ii + j * data.ldx]);
                }
                if (data.metric == da_euclidean)
                    D[i + ii * data.ldd] = std::sqrt(D[i + ii * data.ldd]);
            }
        }
    }
    return D;
}

// Returns the appropriate solution
template <typename T> std::vector<T> test_distance(PairwiseDistanceParamType<T> &data) {
    std::vector<T> D;
    if (data.ldy > 0) {
        D.resize(data.ldd * data.n, T{0.0});
        auto status = da_pairwise_distances(column_major, data.m, data.n, data.k,
                                            data.X.data(), data.ldx, data.Y.data(),
                                            data.ldy, D.data(), data.ldd, data.metric);
        EXPECT_EQ(status, da_status_success);
    } else {
        D.resize(data.ldd * data.m, T{0.0});
        auto status = da_pairwise_distances(column_major, data.m, data.n, data.k,
                                            data.X.data(), data.ldx, nullptr, data.ldy,
                                            D.data(), data.ldd, data.metric);
        EXPECT_EQ(status, da_status_success);
    }

    return D;
}

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(PairwiseDistanceTest, FloatTypes);

TYPED_TEST(PairwiseDistanceTest, RowMajor) {
    std::vector<TypeParam> X{1.0, 2.0, -3.0, 1.0, -1.0, 2.0, 0.0, -2.0, 4.0},
        Y{4.0, -2.0, 3.0, 3.0, -1.0, 2.0}, D{0., 0., 0., 0., 0., 0.},
        D_exp{61., 38., 11., 4., 17., 14.};
    da_int m = 3, n = 2, k = 3, ldx = 3, ldy = 3, ldd = 2;

    auto status = da_pairwise_distances(row_major, m, n, k, X.data(), ldx, Y.data(), ldy,
                                        D.data(), ldd, da_sqeuclidean);
    EXPECT_EQ(status, da_status_success);
    EXPECT_ARR_NEAR(6, D.data(), D_exp.data(),
                    100 * std::numeric_limits<TypeParam>::epsilon());
}

TYPED_TEST(PairwiseDistanceTest, AccuracyTesting_XY) {
    std::vector<PairwiseDistanceParamType<TypeParam>> data;
    // Test for 1by1
    InitGenericData(data, 1, 1, 1);
    // Test for 2by2
    InitGenericData(data, 2, 2, 2);
    // Test for X Y, being a column
    InitGenericData(data, 25, 17, 1);
    // Test for X being a row, Y being a matrix
    InitGenericData(data, 1, 27, 4);
    // Test for X being a row, Y being a row
    InitGenericData(data, 1, 1, 4);
    // Generic test where both X and Y are matrices
    InitGenericData(data, 22, 18, 5);
    // Test for X being a column, Y being a row, varying the leading dimensions
    InitGenericData(data, 50, 2, 1, 52, 4, 56);
    // Test for X being a row, Y being a row, varying the leading dimensions
    InitGenericData(data, 1, 27, 4, 2, 29, 1);
    // Generic test where both X and Y are matrices, varying the leading dimensions
    InitGenericData(data, 23, 19, 8, 25, 32, 26);
    for (auto &dat : data) {
        auto D_ref = reference_distance(dat);
        auto D = test_distance(dat);
        for (da_int i = 0; i < (da_int)D.size(); i++)
            EXPECT_ARR_NEAR((da_int)D.size(), D.data(), D_ref.data(), dat.tol)
                << "\n"
                << "Test with parameters: " << dat.name << " FAILED" << std::endl;
        std::cout << "Testing for data = " << dat.name << std::endl;
    }
}

TYPED_TEST(PairwiseDistanceTest, AccuracyTesting_XX) {
    std::vector<PairwiseDistanceParamType<TypeParam>> data;
    // Test for 1by1
    InitGenericData(data, 1, 0, 1);
    // Test for 2by2
    InitGenericData(data, 2, 0, 2);
    // Test for X being a column
    InitGenericData(data, 25, 0, 1);
    // Test for X being a row
    InitGenericData(data, 1, 0, 4);
    // Generic test where X is a matrix
    InitGenericData(data, 22, 0, 5);
    // Test for X being a column, varying the leading dimensions
    InitGenericData(data, 50, 0, 1, 52, 0, 56);
    // Test for X being a row, varying the leading dimensions
    InitGenericData(data, 1, 0, 14, 2, 0, 1);
    // Generic test where X is a matrix, varying the leading dimensions
    InitGenericData(data, 23, 0, 8, 25, 0, 26);
    for (auto &dat : data) {
        auto D_ref = reference_distance(dat);
        auto D = test_distance(dat);
        for (da_int i = 0; i < (da_int)D.size(); i++)
            ASSERT_ARR_NEAR((da_int)D.size(), D.data(), D_ref.data(), dat.tol)
                << "\n"
                << "Test with parameters: " << dat.name << " FAILED" << std::endl;
        std::cout << "Testing for data = " << dat.name << std::endl;
    }
}

std::string ErrorExits_print(std::string param) {
    std::string ss = "Test for invalid value of " + param + " failed.";
    return ss;
}
// Error checking happens at the pairwise distance level, before calling into a specific distance metric
// such that euclidean, manhattan, etc. Thus, testing only for "euclidean" as an input argument suffices.
TYPED_TEST(PairwiseDistanceTest, ErrorExits) {
    PairwiseDistanceParamType<TypeParam> param;
    // In the following tests, we use nullptr as D, so that we get a memory error if D is accessed.
    // Use Y_tmp instead of Y to ensure it's not nullptr and that we test all options.
    std::vector<TypeParam> X(1), Y_tmp(1), D(1);
    // Test for invalid pointer X.
    EXPECT_EQ(da_pairwise_distances(column_major, param.m, param.n, param.k, nullptr,
                                    param.ldx, Y_tmp.data(), param.ldy, D.data(),
                                    param.ldd, param.metric, param.data_type),
              da_status_invalid_pointer)
        << ErrorExits_print("X");
    // Test for invalid pointer D.
    EXPECT_EQ(da_pairwise_distances(column_major, param.m, param.n, param.k, X.data(),
                                    param.ldx, Y_tmp.data(), param.ldy, nullptr,
                                    param.ldd, param.metric, param.data_type),
              da_status_invalid_pointer)
        << ErrorExits_print("D");
    // Test for invalid value of m.
    EXPECT_EQ(da_pairwise_distances(column_major, -1, param.n, param.k, X.data(),
                                    param.ldx, Y_tmp.data(), param.ldy, D.data(),
                                    param.ldd, param.metric, param.data_type),
              da_status_invalid_array_dimension)
        << ErrorExits_print("m");
    // Test for invalid value of n.
    EXPECT_EQ(da_pairwise_distances(column_major, param.m, 0, param.k, X.data(),
                                    param.ldx, Y_tmp.data(), param.ldy, D.data(),
                                    param.ldd, param.metric, param.data_type),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n");
    // Test for invalid value of k.
    EXPECT_EQ(da_pairwise_distances(column_major, param.m, param.n, -2, X.data(),
                                    param.ldx, Y_tmp.data(), param.ldy, D.data(),
                                    param.ldd, param.metric, param.data_type),
              da_status_invalid_array_dimension)
        << ErrorExits_print("k");
    // Test for invalid value of ldx
    EXPECT_EQ(da_pairwise_distances(column_major, param.m, param.n, param.k, X.data(), -1,
                                    Y_tmp.data(), param.ldy, D.data(), param.ldd,
                                    param.metric, param.data_type),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx");
    // Test for invalid value of ldy.
    EXPECT_EQ(da_pairwise_distances(column_major, param.m, param.n, param.k, X.data(),
                                    param.ldx, Y_tmp.data(), -1, D.data(), param.ldd,
                                    param.metric, param.data_type),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldy");
    // Test for invalid value of ldd.
    EXPECT_EQ(da_pairwise_distances(column_major, param.m, param.n, param.k, X.data(),
                                    param.ldx, Y_tmp.data(), param.ldy, D.data(), -1,
                                    param.metric, param.data_type),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldd");
    // Test for invalid value of metric.
    EXPECT_EQ(da_pairwise_distances(column_major, param.m, param.n, param.k, X.data(),
                                    param.ldx, Y_tmp.data(), param.ldy, D.data(),
                                    param.ldd, da_manhattan, param.data_type),
              da_status_not_implemented)
        << ErrorExits_print("metric");
    // Test for invalid value of data_type.
    EXPECT_EQ(da_pairwise_distances(column_major, param.m, param.n, param.k, X.data(),
                                    param.ldx, Y_tmp.data(), param.ldy, D.data(),
                                    param.ldd, param.metric, da_allow_NaN),
              da_status_not_implemented)
        << ErrorExits_print("data_type");
    // Test for invalid value of m, row-major.
    EXPECT_EQ(da_pairwise_distances(row_major, -1, param.n, param.k, X.data(), param.ldx,
                                    Y_tmp.data(), param.ldy, D.data(), param.ldd,
                                    param.metric, param.data_type),
              da_status_invalid_array_dimension)
        << ErrorExits_print("m");
    // Test for invalid value of n, row-major.
    EXPECT_EQ(da_pairwise_distances(row_major, param.m, 0, param.k, X.data(), param.ldx,
                                    Y_tmp.data(), param.ldy, D.data(), param.ldd,
                                    param.metric, param.data_type),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n");
    // Test for invalid value of k, row-major.
    EXPECT_EQ(da_pairwise_distances(row_major, param.m, param.n, -2, X.data(), param.ldx,
                                    Y_tmp.data(), param.ldy, D.data(), param.ldd,
                                    param.metric, param.data_type),
              da_status_invalid_array_dimension)
        << ErrorExits_print("k");
    // Test for invalid value of ldx, row-major
    EXPECT_EQ(da_pairwise_distances(row_major, param.m, param.n, param.k, X.data(), -1,
                                    Y_tmp.data(), param.ldy, D.data(), param.ldd,
                                    param.metric, param.data_type),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx");
    // Test for invalid value of ldy, row-major.
    EXPECT_EQ(da_pairwise_distances(row_major, param.m, param.n, param.k, X.data(),
                                    param.ldx, Y_tmp.data(), -1, D.data(), param.ldd,
                                    param.metric, param.data_type),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldy");
    // Test for invalid value of ldd, row-major.
    EXPECT_EQ(da_pairwise_distances(row_major, param.m, param.n, param.k, X.data(),
                                    param.ldx, Y_tmp.data(), param.ldy, D.data(), -1,
                                    param.metric, param.data_type),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldd");
}
