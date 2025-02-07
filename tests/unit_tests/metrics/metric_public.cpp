/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
// tests for each metric type. This is only used when we have a different implementation
// to compare against.
static std::list<std::tuple<std::string, da_metric>> MetricType = {
    {"da_euclidean", da_euclidean}, {"da_sqeuclidean", da_sqeuclidean}};

// Add all available distance metrics in a list so that we can run all
// tests for each metric type. This is only used when we have a different implementation
// to compare against.
static std::list<std::tuple<std::string, da_metric>> MetricExactResultsType = {
    {"da_euclidean", da_euclidean},
    {"da_l2", da_l2},
    {"da_sqeuclidean", da_sqeuclidean},
    {"da_manhattan", da_manhattan},
    {"da_l1", da_l1},
    {"da_cityblock", da_cityblock},
    {"da_cosine", da_cosine},
    {"da_minkowski", da_minkowski}};

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
                    if ((data.metric == da_euclidean) ||
                        (data.metric == da_sqeuclidean)) {
                        D[i + ii * data.ldd] +=
                            (data.X[i + j * data.ldx] - data.Y[ii + j * data.ldy]) *
                            (data.X[i + j * data.ldx] - data.Y[ii + j * data.ldy]);
                    } else {
                        throw std::runtime_error("Error in metric_public.cpp");
                    }
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
                    if ((data.metric == da_euclidean) ||
                        (data.metric == da_sqeuclidean)) {
                        D[i + ii * data.ldd] +=
                            (data.X[i + j * data.ldx] - data.X[ii + j * data.ldx]) *
                            (data.X[i + j * data.ldx] - data.X[ii + j * data.ldx]);
                    } else {
                        throw std::runtime_error("Error in metric_public.cpp");
                    }
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
    // Mikowski parameter
    T p = 2.0;
    if (data.ldy > 0) {
        D.resize(data.ldd * data.n, T{0.0});
        auto status = da_pairwise_distances(column_major, data.m, data.n, data.k,
                                            data.X.data(), data.ldx, data.Y.data(),
                                            data.ldy, D.data(), data.ldd, p, data.metric);
        EXPECT_EQ(status, da_status_success);
    } else {
        D.resize(data.ldd * data.m, T{0.0});
        auto status = da_pairwise_distances(column_major, data.m, data.n, data.k,
                                            data.X.data(), data.ldx, nullptr, data.ldy,
                                            D.data(), data.ldd, p, data.metric);
        EXPECT_EQ(status, da_status_success);
    }

    return D;
}

template <typename T>
void transposeMatrix(da_int m, da_int n, da_int ldx, da_int ldy, std::vector<T> &X,
                     std::vector<T> &Y) {
    for (da_int i = 0; i < m; i++) {
        for (da_int j = 0; j < n; j++) {
            Y[i + j * ldy] = X[i * ldx + j];
        }
    }
}

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(PairwiseDistanceTest, FloatTypes);

TYPED_TEST(PairwiseDistanceTest, zero_data) {
    da_int m = 3, n = 2, k = 3, ldx = m, ldy = n, ldd = m;
    std::vector<TypeParam> X(ldx * k, 0.0), Y(ldy * k, 0.0);
    std::string name;
    TypeParam p = 3.0;

    // Generate tests for XY
    // Iterate through the list of metrics and register all tests accordingly.
    for (auto const &metric : MetricExactResultsType) {
        // Expected distance is zero. Initialize D to nonzero to ensure
        // the values are being updated by da_pairwise_distances()
        std::vector<TypeParam> D_exp(ldd * n, 0.0), D(ldd * n, 999.);
        // Only for cosine D_exp is a vector of ones (1 - cosine_similarity)
        if (std::get<1>(metric) == da_cosine)
            std::fill(D_exp.begin(), D_exp.end(), 1.0);
        name = "m=" + std::to_string(m) + ", n=" + std::to_string(n) +
               ", k=" + std::to_string(k) + ", ldx=" + std::to_string(ldx) +
               ", ldy=" + std::to_string(ldy) + ", ldd=" + std::to_string(ldd) +
               ", metric=" + std::get<0>(metric) + ", XY";
        std::cout << "Testing for data = " << name << std::endl;
        auto status =
            da_pairwise_distances(column_major, m, n, k, X.data(), ldx, Y.data(), ldy,
                                  D.data(), ldd, p, std::get<1>(metric));
        EXPECT_EQ(status, da_status_success);

        EXPECT_ARR_EQ(da_int(D.size()), D.data(), D_exp.data(), 1, 1, 0, 0);
    }

    // Generate tests for XX
    // Iterate through the list of metrics and register all tests accordingly.
    for (auto const &metric : MetricExactResultsType) {
        // Expected distance is zero. Initialize D to nonzero to ensure
        // the values are being updated by da_pairwise_distances()
        std::vector<TypeParam> D_exp(ldd * m, 0.0), D(ldd * m, 999.);
        // Only for cosine D_exp is a vector of ones (1 - cosine_similarity)
        // apart from the diagonal
        if (std::get<1>(metric) == da_cosine) {
            std::fill(D_exp.begin(), D_exp.end(), 1.0);
            for (auto i = 0; i < m; i++)
                D_exp[i + i * ldd] = 0.0;
        }
        name = "m=" + std::to_string(m) + ", k=" + std::to_string(k) +
               ", ldx=" + std::to_string(ldx) + ", ldd=" + std::to_string(ldd) +
               ", metric=" + std::get<0>(metric) + ", XX";
        std::cout << "Testing for data = " << name << std::endl;
        auto status = da_pairwise_distances(column_major, m, n, k, X.data(), ldx, nullptr,
                                            0, D.data(), ldd, p, std::get<1>(metric));
        EXPECT_EQ(status, da_status_success);

        EXPECT_ARR_EQ(da_int(D.size()), D.data(), D_exp.data(), 1, 1, 0, 0);
    }
}

TYPED_TEST(PairwiseDistanceTest, Minkowski_equivalents) {
    da_int m = 3, n = 2, k = 3, ldx = m, ldy = n, ldd = m;
    std::vector<TypeParam> X{1.0, 2.0, -3.0, 1.0, -1.0, 2.0, 0.0, -2.0, 4.0},
        Y{4.0, -2.0, 3.0, 3.0, -1.0, 2.0}, D(ldd * n, 0.);
    std::vector<TypeParam> D_exp(ldd * n);
    std::string name;
    TypeParam p = 1.0;
    std::cout << "Testing for data = "
              << "p = " << std::to_string(1) << std::endl;
    auto status = da_pairwise_distances(column_major, m, n, k, X.data(), ldx, Y.data(),
                                        ldy, D.data(), ldd, p, da_minkowski);
    EXPECT_EQ(status, da_status_success);
    status = da_pairwise_distances(column_major, m, n, k, X.data(), ldx, Y.data(), ldy,
                                   D_exp.data(), ldd, p, da_manhattan);
    EXPECT_EQ(status, da_status_success);

    // Check that Minkowski with p=1 gives exactly the same results Manhattan
    EXPECT_ARR_EQ(da_int(D.size()), D.data(), D_exp.data(), 1, 1, 0, 0);

    p = 2.0;
    std::cout << "Testing for data = "
              << "p = " << std::to_string(2) << std::endl;
    status = da_pairwise_distances(column_major, m, n, k, X.data(), ldx, Y.data(), ldy,
                                   D.data(), ldd, p, da_minkowski);
    EXPECT_EQ(status, da_status_success);
    status = da_pairwise_distances(column_major, m, n, k, X.data(), ldx, Y.data(), ldy,
                                   D_exp.data(), ldd, p, da_euclidean);
    EXPECT_EQ(status, da_status_success);

    // Check that Minkowski with p=1 gives exactly the same results Manhattan
    EXPECT_ARR_EQ(da_int(D.size()), D.data(), D_exp.data(), 1, 1, 0, 0);
}

TYPED_TEST(PairwiseDistanceTest, FixedData_XY) {
    da_int m = 3, n = 2, k = 3, ldx_r = k, ldy_r = k, ldd_r = n, ldx_c = m, ldy_c = n,
           ldd_c = m;
    std::vector<TypeParam> X_row{1.0, 2.0, -3.0, 1.0, -1.0, 2.0, 0.0, -2.0, 4.0},
        Y_row{4.0, -2.0, 3.0, 3.0, -1.0, 2.0}, D_row(m * ldd_r, 0.), D_col(ldd_c * n, 0.);
    std::vector<TypeParam> D_exp_row(m * ldd_r), D_exp_col(ldd_c * n);
    TypeParam p = 4.5;
    std::string name;
    std::vector<TypeParam> X_col(ldx_c * k), Y_col(ldy_c * k);
    transposeMatrix(m, k, ldx_r, ldx_c, X_row, X_col);
    transposeMatrix(n, k, ldy_r, ldy_c, Y_row, Y_col);
    // Test for row major order
    // Iterate through the list of metrics and register all tests accordingly.
    for (auto const &metric : MetricExactResultsType) {
        name = "m=" + std::to_string(m) + ", n=" + std::to_string(n) +
               ", k=" + std::to_string(k) + ", ldx=" + std::to_string(ldx_r) +
               ", ldy=" + std::to_string(ldy_r) + ", ldd=" + std::to_string(ldd_r) +
               ", metric=" + std::get<0>(metric) + ", order=row_major";
        if ((std::get<1>(metric) == da_sqeuclidean) || (std::get<1>(metric) == da_l2) ||
            (std::get<1>(metric) == da_euclidean)) {
            D_exp_row = {61., 38., 11., 4., 17., 14.};
        } else if ((std::get<1>(metric) == da_manhattan) ||
                   (std::get<1>(metric) == da_l1) ||
                   (std::get<1>(metric) == da_cityblock)) {
            D_exp_row = {13., 10., 5., 2., 5., 6.};
        } else if ((std::get<1>(metric) == da_cosine)) {
            D_exp_row = {1.4466625002869187, 1.3571428571428572, 0.0902823477053158,
                         0.1271284390560302, 0.3356361611700803, 0.4023856953328032};
        } else if ((std::get<1>(metric) == da_minkowski)) {
            D_exp_row = {6.254413691636758, 5.124034794569624, 3.009451461921273, 2,
                         4.001734793974489, 3.1055785043917186};
        } else {
            throw std::runtime_error("Error in metric_public.cpp");
        }

        if ((std::get<1>(metric) == da_euclidean)) {
            for (auto &d : D_exp_row)
                d = std::sqrt(d);
        }

        transposeMatrix(m, n, ldd_r, ldd_c, D_exp_row, D_exp_col);
        std::cout << "Testing for data = " << name << std::endl;
        auto status =
            da_pairwise_distances(row_major, m, n, k, X_row.data(), ldx_r, Y_row.data(),
                                  ldy_r, D_row.data(), ldd_r, p, std::get<1>(metric));
        EXPECT_EQ(status, da_status_success);
        EXPECT_ARR_NEAR(da_int(D_row.size()), D_row.data(), D_exp_row.data(),
                        100 * std::numeric_limits<TypeParam>::epsilon());

        // Test for column major order
        name = "m=" + std::to_string(m) + ", n=" + std::to_string(n) +
               ", k=" + std::to_string(k) + ", ldx=" + std::to_string(ldx_c) +
               ", ldy=" + std::to_string(ldy_c) + ", ldd=" + std::to_string(ldd_c) +
               ", metric=" + std::get<0>(metric) + ", order=column_major";
        std::cout << "Testing for data = " << name << std::endl;
        status = da_pairwise_distances(column_major, m, n, k, X_col.data(), ldx_c,
                                       Y_col.data(), ldy_c, D_col.data(), ldd_c, p,
                                       std::get<1>(metric));
        EXPECT_EQ(status, da_status_success);
        EXPECT_ARR_NEAR(da_int(D_col.size()), D_col.data(), D_exp_col.data(),
                        100 * std::numeric_limits<TypeParam>::epsilon());
    }
}

TYPED_TEST(PairwiseDistanceTest, FixedData_XX) {
    da_int m = 4, k = 3, ldx_r = k, ldd_r = m, ldx_c = m, ldd_c = m;
    std::vector<TypeParam> X_row{1.0, 4.0,  -3.0, 2.0,  -1.0, 3.0,
                                 1.0, -2.0, 5.0,  -3.0, 1.0,  3.0},
        D_row(m * ldd_r, 0.), D_col(ldd_c * m, 0.);
    std::vector<TypeParam> D_exp_row(m * ldd_r, 0.), D_exp_col(ldd_c * m, 0.);
    TypeParam p = 1.5;
    std::string name;
    std::vector<TypeParam> X_col(ldx_c * k);
    transposeMatrix(m, k, ldx_r, ldx_c, X_row, X_col);
    // Test for row major order
    // Iterate through the list of metrics and register all tests accordingly.
    for (auto const &metric : MetricExactResultsType) {
        name = "m=" + std::to_string(m) + ", k=" + std::to_string(k) +
               ", ldx=" + std::to_string(ldx_r) + ", ldd=" + std::to_string(ldd_r) +
               ", metric=" + std::get<0>(metric) + ", order=row_major";
        if ((std::get<1>(metric) == da_sqeuclidean) ||
            (std::get<1>(metric) == da_euclidean)) {
            D_exp_row = {0.,   62., 100., 61., 62., 0.,  6.,  29.,
                         100., 6.,  0.,   29., 61., 29., 29., 0.};
        } else if ((std::get<1>(metric) == da_manhattan)) {
            D_exp_row = {0.,  12., 14., 13., 12., 0., 4., 7.,
                         14., 4.,  0.,  9.,  13., 7., 9., 0.};
        } else if ((std::get<1>(metric) == da_cosine)) {
            D_exp_row = {0.,
                         1.576556660197055,
                         1.7877263614433763,
                         1.359937016532678,
                         1.576556660197055,
                         0.,
                         0.0728949306988934,
                         0.8773721321030069,
                         1.7877263614433763,
                         0.0728949306988934,
                         0.,
                         0.5811460917083046,
                         1.359937016532678,
                         0.8773721321030069,
                         0.5811460917083046,
                         0.};
        } else if ((std::get<1>(metric) == da_minkowski)) {
            D_exp_row = {0.,
                         8.972707819822075,
                         11.168500752960059,
                         9.19738630386655,
                         8.972707819822075,
                         0.,
                         2.8567382778502783,
                         5.811210513661822,
                         11.168500752960059,
                         2.8567382778502783,
                         0.,
                         6.356105477264618,
                         9.19738630386655,
                         5.811210513661822,
                         6.356105477264618,
                         0.};
        } else {
            throw std::runtime_error("Error in metric_public.cpp");
        }

        if ((std::get<1>(metric) == da_euclidean)) {
            for (auto &d : D_exp_row)
                d = std::sqrt(d);
        }

        transposeMatrix(m, m, ldd_r, ldd_c, D_exp_row, D_exp_col);
        auto status =
            da_pairwise_distances(row_major, m, 1, k, X_row.data(), ldx_r, nullptr, 0,
                                  D_row.data(), ldd_r, p, std::get<1>(metric));

        EXPECT_EQ(status, da_status_success);
        EXPECT_ARR_NEAR(da_int(D_row.size()), D_row.data(), D_exp_row.data(),
                        100 * std::numeric_limits<TypeParam>::epsilon());

        // Test for column major order
        name = "m=" + std::to_string(m) + ", k=" + std::to_string(k) +
               ", ldx=" + std::to_string(ldx_c) + ", ldd=" + std::to_string(ldd_c) +
               ", metric=" + std::get<0>(metric) + ", order=column_major";
        status =
            da_pairwise_distances(column_major, m, 1, k, X_col.data(), ldx_c, nullptr, 0,
                                  D_col.data(), ldd_c, p, std::get<1>(metric));
        EXPECT_EQ(status, da_status_success);
        EXPECT_ARR_NEAR(da_int(D_col.size()), D_col.data(), D_exp_col.data(),
                        100 * std::numeric_limits<TypeParam>::epsilon());
    }
}

TYPED_TEST(PairwiseDistanceTest, FixedData_XY_ld) {
    da_int ldx_inc = 2, ldy_inc = 1, ldd_inc = 3;
    da_int m = 3, n = 2, k = 3, ldx_r = k + ldx_inc, ldy_r = k + ldy_inc,
           ldd_r = n + ldd_inc, ldx_c = m + ldx_inc, ldy_c = n + ldy_inc,
           ldd_c = m + ldd_inc;
    std::vector<TypeParam> X_row{1.0,  2.0,  -3.0, 123., 123., 1.0,  -1.0, 2.0,
                                 123., 123., 0.0,  -2.0, 4.0,  123., 123.},
        Y_row{4.0, -2.0, 3.0, 456., 3.0, -1.0, 2.0, 456.}, D_row(m * ldd_r, 0.),
        D_col(ldd_c * n, 0.);
    std::vector<TypeParam> D_exp_row(m * ldd_r), D_exp_col(ldd_c * n);
    TypeParam p = 4.5;
    std::string name;
    std::vector<TypeParam> X_col(ldx_c * k, 123.), Y_col(ldy_c * k, 456.);

    transposeMatrix(m, k, ldx_r, ldx_c, X_row, X_col);
    transposeMatrix(n, k, ldy_r, ldy_c, Y_row, Y_col);
    // Test for row major order
    // Iterate through the list of metrics and register all tests accordingly.
    for (auto const &metric : MetricExactResultsType) {
        name = "m=" + std::to_string(m) + ", n=" + std::to_string(n) +
               ", k=" + std::to_string(k) + ", ldx=" + std::to_string(ldx_r) +
               ", ldy=" + std::to_string(ldy_r) + ", ldd=" + std::to_string(ldd_r) +
               ", metric=" + std::get<0>(metric) + ", order=row_major";
        if ((std::get<1>(metric) == da_sqeuclidean) || (std::get<1>(metric) == da_l2) ||
            (std::get<1>(metric) == da_euclidean)) {
            D_exp_row = {61., 38., 0., 0., 0., 11., 4., 0., 0., 0., 17., 14., 0., 0., 0.};
        } else if ((std::get<1>(metric) == da_manhattan) ||
                   (std::get<1>(metric) == da_l1) ||
                   (std::get<1>(metric) == da_cityblock)) {
            D_exp_row = {13., 10., 0., 0., 0., 5., 2., 0., 0., 0., 5., 6., 0., 0., 0.};
        } else if ((std::get<1>(metric) == da_cosine)) {
            D_exp_row = {1.4466625002869187, 1.3571428571428572, 0., 0., 0.,
                         0.0902823477053158, 0.1271284390560302, 0., 0., 0.,
                         0.3356361611700803, 0.4023856953328032, 0., 0., 0.};
        } else if ((std::get<1>(metric) == da_minkowski)) {
            D_exp_row = {6.254413691636758,
                         5.124034794569624,
                         0.,
                         0.,
                         0.,
                         3.009451461921273,
                         2,
                         0.,
                         0.,
                         0.,
                         4.001734793974489,
                         3.1055785043917186,
                         0.,
                         0.,
                         0.};
        } else {
            throw std::runtime_error("Error in metric_public.cpp");
        }

        if ((std::get<1>(metric) == da_euclidean)) {
            for (auto &d : D_exp_row)
                d = std::sqrt(d);
        }

        transposeMatrix(m, n, ldd_r, ldd_c, D_exp_row, D_exp_col);
        std::cout << "Testing for data = " << name << std::endl;
        auto status =
            da_pairwise_distances(row_major, m, n, k, X_row.data(), ldx_r, Y_row.data(),
                                  ldy_r, D_row.data(), ldd_r, p, std::get<1>(metric));
        EXPECT_EQ(status, da_status_success);

        EXPECT_ARR_NEAR(da_int(D_row.size()), D_row.data(), D_exp_row.data(),
                        100 * std::numeric_limits<TypeParam>::epsilon());

        // Test for column major order
        name = "m=" + std::to_string(m) + ", n=" + std::to_string(n) +
               ", k=" + std::to_string(k) + ", ldx=" + std::to_string(ldx_c) +
               ", ldy=" + std::to_string(ldy_c) + ", ldd=" + std::to_string(ldd_c) +
               ", metric=" + std::get<0>(metric) + ", order=column_major";
        std::cout << "Testing for data = " << name << std::endl;

        status = da_pairwise_distances(column_major, m, n, k, X_col.data(), ldx_c,
                                       Y_col.data(), ldy_c, D_col.data(), ldd_c, p,
                                       std::get<1>(metric));

        EXPECT_EQ(status, da_status_success);
        EXPECT_ARR_NEAR(da_int(D_col.size()), D_col.data(), D_exp_col.data(),
                        100 * std::numeric_limits<TypeParam>::epsilon());
    }
}

TYPED_TEST(PairwiseDistanceTest, FixedData_XX_ld) {
    da_int ldx_inc = 2, ldd_inc = 3;
    da_int m = 4, k = 3, ldx_r = k + ldx_inc, ldd_r = m + ldd_inc, ldx_c = m + ldx_inc,
           ldd_c = m + ldd_inc;
    std::vector<TypeParam> X_row{1.0,  4.0,  -3.0, 123., 123., 2.0, -1.0,
                                 3.0,  123., 123., 1.0,  -2.0, 5.0, 123.,
                                 123., -3.0, 1.0,  3.0,  123., 123.},
        D_row(m * ldd_r, 0.), D_col(ldd_c * m, 0.);
    std::vector<TypeParam> D_exp_row(m * ldd_r, 0.), D_exp_col(ldd_c * m, 0.);
    TypeParam p = 1.5;
    std::string name;
    std::vector<TypeParam> X_col(ldx_c * k, 123.);
    transposeMatrix(m, k, ldx_r, ldx_c, X_row, X_col);
    // Test for row major order
    // Iterate through the list of metrics and register all tests accordingly.
    for (auto const &metric : MetricExactResultsType) {
        name = "m=" + std::to_string(m) + ", k=" + std::to_string(k) +
               ", ldx=" + std::to_string(ldx_r) + ", ldd=" + std::to_string(ldd_r) +
               ", metric=" + std::get<0>(metric) + ", order=row_major";
        if ((std::get<1>(metric) == da_sqeuclidean) ||
            (std::get<1>(metric) == da_euclidean)) {
            D_exp_row = {0.,  62., 100., 61., 0.,   0., 0., 62., 0., 6.,
                         29., 0.,  0.,   0.,  100., 6., 0., 29., 0., 0.,
                         0.,  61., 29.,  29., 0.,   0., 0., 0.};
        } else if ((std::get<1>(metric) == da_manhattan)) {
            D_exp_row = {0.,  12., 14., 13., 0., 0., 0., 12., 0., 4., 7., 0., 0., 0.,
                         14., 4.,  0.,  9.,  0., 0., 0., 13., 7., 9., 0., 0., 0., 0.};
        } else if ((std::get<1>(metric) == da_cosine)) {
            D_exp_row = {0.,
                         1.576556660197055,
                         1.7877263614433763,
                         1.359937016532678,
                         0.,
                         0.,
                         0.,
                         1.576556660197055,
                         0.,
                         0.0728949306988934,
                         0.8773721321030069,
                         0.,
                         0.,
                         0.,
                         1.7877263614433763,
                         0.0728949306988934,
                         0.,
                         0.5811460917083046,
                         0.,
                         0.,
                         0.,
                         1.359937016532678,
                         0.8773721321030069,
                         0.5811460917083046,
                         0.,
                         0.,
                         0.,
                         0.};
        } else if ((std::get<1>(metric) == da_minkowski)) {
            D_exp_row = {0.,
                         8.972707819822075,
                         11.168500752960059,
                         9.19738630386655,
                         0.,
                         0.,
                         0.,
                         8.972707819822075,
                         0.,
                         2.8567382778502783,
                         5.811210513661822,
                         0.,
                         0.,
                         0.,
                         11.168500752960059,
                         2.8567382778502783,
                         0.,
                         6.356105477264618,
                         0.,
                         0.,
                         0.,
                         9.19738630386655,
                         5.811210513661822,
                         6.356105477264618,
                         0.,
                         0.,
                         0.,
                         0.};
        } else {
            throw std::runtime_error("Error in metric_public.cpp");
        }

        if ((std::get<1>(metric) == da_euclidean)) {
            for (auto &d : D_exp_row)
                d = std::sqrt(d);
        }

        transposeMatrix(m, m, ldd_r, ldd_c, D_exp_row, D_exp_col);
        auto status =
            da_pairwise_distances(row_major, m, 1, k, X_row.data(), ldx_r, nullptr, 0,
                                  D_row.data(), ldd_r, p, std::get<1>(metric));

        EXPECT_EQ(status, da_status_success);
        EXPECT_ARR_NEAR(da_int(D_row.size()), D_row.data(), D_exp_row.data(),
                        100 * std::numeric_limits<TypeParam>::epsilon());

        // Test for column major order
        name = "m=" + std::to_string(m) + ", k=" + std::to_string(k) +
               ", ldx=" + std::to_string(ldx_c) + ", ldd=" + std::to_string(ldd_c) +
               ", metric=" + std::get<0>(metric) + ", order=column_major";
        status =
            da_pairwise_distances(column_major, m, 1, k, X_col.data(), ldx_c, nullptr, 0,
                                  D_col.data(), ldd_c, p, std::get<1>(metric));
        EXPECT_EQ(status, da_status_success);
        EXPECT_ARR_NEAR(da_int(D_col.size()), D_col.data(), D_exp_col.data(),
                        100 * std::numeric_limits<TypeParam>::epsilon());
    }
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
    TypeParam p = 0.0;
    // Test for invalid minkowski parameter.
    EXPECT_EQ(da_pairwise_distances(column_major, param.m, param.n, param.k, X.data(),
                                    param.ldx, Y_tmp.data(), param.ldy, D.data(),
                                    param.ldd, p, da_minkowski),
              da_status_invalid_input)
        << ErrorExits_print("Minkowski parameter");
    // Test for invalid pointer X.
    EXPECT_EQ(da_pairwise_distances(column_major, param.m, param.n, param.k, nullptr,
                                    param.ldx, Y_tmp.data(), param.ldy, D.data(),
                                    param.ldd, p, param.metric),
              da_status_invalid_pointer)
        << ErrorExits_print("X");
    // Test for invalid pointer D.
    EXPECT_EQ(da_pairwise_distances(column_major, param.m, param.n, param.k, X.data(),
                                    param.ldx, Y_tmp.data(), param.ldy, nullptr,
                                    param.ldd, p, param.metric),
              da_status_invalid_pointer)
        << ErrorExits_print("D");
    // Test for invalid value of m.
    EXPECT_EQ(da_pairwise_distances(column_major, -1, param.n, param.k, X.data(),
                                    param.ldx, Y_tmp.data(), param.ldy, D.data(),
                                    param.ldd, p, param.metric),
              da_status_invalid_array_dimension)
        << ErrorExits_print("m");
    // Test for invalid value of n.
    EXPECT_EQ(da_pairwise_distances(column_major, param.m, 0, param.k, X.data(),
                                    param.ldx, Y_tmp.data(), param.ldy, D.data(),
                                    param.ldd, p, param.metric),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n");
    // Test for invalid value of k.
    EXPECT_EQ(da_pairwise_distances(column_major, param.m, param.n, -2, X.data(),
                                    param.ldx, Y_tmp.data(), param.ldy, D.data(),
                                    param.ldd, p, param.metric),
              da_status_invalid_array_dimension)
        << ErrorExits_print("k");
    // Test for invalid value of ldx
    EXPECT_EQ(da_pairwise_distances(column_major, param.m, param.n, param.k, X.data(), -1,
                                    Y_tmp.data(), param.ldy, D.data(), param.ldd, p,
                                    param.metric),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx");
    // Test for invalid value of ldy.
    EXPECT_EQ(da_pairwise_distances(column_major, param.m, param.n, param.k, X.data(),
                                    param.ldx, Y_tmp.data(), -1, D.data(), param.ldd, p,
                                    param.metric),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldy");
    // Test for invalid value of ldd.
    EXPECT_EQ(da_pairwise_distances(column_major, param.m, param.n, param.k, X.data(),
                                    param.ldx, Y_tmp.data(), param.ldy, D.data(), -1, p,
                                    param.metric),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldd");
    // Test for invalid value of ldd when Y is nullptr.
    EXPECT_EQ(da_pairwise_distances(column_major, param.m, param.n, param.k, X.data(),
                                    param.ldx, nullptr, param.ldy, D.data(), -1, p,
                                    param.metric),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldd");
    // Test for invalid value of metric.
    EXPECT_EQ(da_pairwise_distances(column_major, param.m, param.n, param.k, X.data(),
                                    param.ldx, Y_tmp.data(), param.ldy, D.data(),
                                    param.ldd, p, da_metric(999)),
              da_status_not_implemented)
        << ErrorExits_print("metric");
    // Test for invalid value of m, row-major.
    EXPECT_EQ(da_pairwise_distances(row_major, -1, param.n, param.k, X.data(), param.ldx,
                                    Y_tmp.data(), param.ldy, D.data(), param.ldd, p,
                                    param.metric),
              da_status_invalid_array_dimension)
        << ErrorExits_print("m");
    // Test for invalid value of n, row-major.
    EXPECT_EQ(da_pairwise_distances(row_major, param.m, 0, param.k, X.data(), param.ldx,
                                    Y_tmp.data(), param.ldy, D.data(), param.ldd, p,
                                    param.metric),
              da_status_invalid_array_dimension)
        << ErrorExits_print("n");
    // Test for invalid value of k, row-major.
    EXPECT_EQ(da_pairwise_distances(row_major, param.m, param.n, -2, X.data(), param.ldx,
                                    Y_tmp.data(), param.ldy, D.data(), param.ldd, p,
                                    param.metric),
              da_status_invalid_array_dimension)
        << ErrorExits_print("k");
    // Test for invalid value of ldx, row-major
    EXPECT_EQ(da_pairwise_distances(row_major, param.m, param.n, param.k, X.data(), -1,
                                    Y_tmp.data(), param.ldy, D.data(), param.ldd, p,
                                    param.metric),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldx");
    // Test for invalid value of ldy, row-major.
    EXPECT_EQ(da_pairwise_distances(row_major, param.m, param.n, param.k, X.data(),
                                    param.ldx, Y_tmp.data(), -1, D.data(), param.ldd, p,
                                    param.metric),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldy");
    // Test for invalid value of ldd, row-major.
    EXPECT_EQ(da_pairwise_distances(row_major, param.m, param.n, param.k, X.data(),
                                    param.ldx, Y_tmp.data(), param.ldy, D.data(), -1, p,
                                    param.metric),
              da_status_invalid_leading_dimension)
        << ErrorExits_print("ldd");
}
