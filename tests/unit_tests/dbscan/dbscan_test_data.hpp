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

#include <limits>
#include <list>
#include <numeric>
#include <random>
#include <string.h>

#include "../utest_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

template <typename T> struct DBSCANParamType {

    std::string test_name;

    da_int n_samples = 0;
    da_int n_features = 0;
    std::vector<T> A;
    da_int lda = 0;
    da_int min_samples = 5;
    T eps = 0.5;
    T power = 2.0;

    std::vector<da_int> expected_labels;
    std::vector<da_int> expected_core_sample_indices;
    std::vector<T> expected_rinfo;
    da_int expected_n_clusters = 0;
    da_int expected_n_core_samples = 0;
    da_int leaf_size = 30;

    std::string algorithm = "brute";
    std::string order = "column-major";
    std::string metric = "euclidean";

    da_status expected_status = da_status_success;
};

template <typename T> void Get1by1BaseData(DBSCANParamType<T> &param) {
    param.test_name = "1 by 1 data matrix";

    param.n_samples = 1;
    param.n_features = 1;
    std::vector<double> A{2.1};
    param.A = convert_vector<double, T>(A);
    param.lda = 1;
    param.min_samples = 1;
    std::vector<double> expected_rinfo_double{1, 1, 1, 0.5, 1, 30, 2, 1, 1};
    param.expected_core_sample_indices = std::vector<da_int>{0};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo_double);
    param.expected_labels = std::vector<da_int>{0};
    param.expected_n_core_samples = 1;
    param.expected_n_clusters = 1;
}

template <typename T> void GetZeroBaseData(DBSCANParamType<T> &param) {
    param.test_name = "Data matrix full of zeros";

    param.n_samples = 25;
    param.n_features = 5;
    T zero = 0;
    param.A = std::vector<T>(param.n_samples * param.n_features, zero);
    param.lda = param.n_samples;
    param.min_samples = 2;
    std::vector<double> expected_rinfo_double{25, 5, 25, 0.5, 2, 30, 2, 25, 1};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo_double);
    param.expected_labels = std::vector<da_int>(param.n_samples, 0);
    param.expected_n_core_samples = 25;
    param.expected_n_clusters = 1;
    param.expected_core_sample_indices = std::vector<da_int>(25);
    std::iota(param.expected_core_sample_indices.begin(),
              param.expected_core_sample_indices.end(), (da_int)0);
}

template <typename T> void Get30by3BaseData(DBSCANParamType<T> &param) {
    param.test_name = "30 by 3 data matrix containing 3 clusters";

    param.n_samples = 30;
    param.n_features = 3;
    std::vector<double> A(90);

    std::random_device rd;  // Random number generator seed
    std::mt19937 gen(rd()); // Initialize Mersenne Twister random number generator
    std::uniform_real_distribution<> dis(
        -0.1, 0.1); // Create uniform distribution in range [-0.1, 0.1]

    for (da_int i = 0; i < 90; i++) {
        A[i] = dis(gen);
        if (i % 3 == 0) {
            A[i] += 2.0;
        } else if (i % 3 == 1) {
            A[i] -= 2.0;
        }
    }
    param.A = convert_vector<double, T>(A);
    // A now contains three clusters centered on (2, 2, 2), (-2, -2, -2), and (0, 0, 0)
    param.lda = 30;
    param.min_samples = 3;

    std::vector<double> expected_rinfo_double{30, 3, 30, 0.5, 3, 30, 2, 30, 3};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo_double);
    param.expected_labels =
        std::vector<da_int>{0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
                            0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
    param.expected_core_sample_indices = std::vector<da_int>(30);
    std::iota(param.expected_core_sample_indices.begin(),
              param.expected_core_sample_indices.end(), (da_int)0);
    param.expected_n_core_samples = 30;
    param.expected_n_clusters = 3;
}

template <typename T> void Get25by2BaseData(DBSCANParamType<T> &param) {
    param.test_name = "25 by 2 data matrix containing 3 clusters and some noise";

    param.n_samples = 25;
    param.n_features = 2;
    std::vector<double> A{
        0,     1.0,   1.0,   -1.0,  -1.0, 10.0, -7.0, -6.0, -8.0, -6.5, -5.0, -8.0, -6.5,
        -5.0,  -12.0, -12.1, -11.9, 15.0, -5.0, -5.1, -4.9, -5.0, -5.1, -5.1, -5.0, 0,
        1.0,   -1.0,  1.0,   -1.0,  10.0, 2.0,  2.0,  3.0,  3.1,  3.0,  1.0,  1.0,  1.0,
        -12.0, -12.1, -11.9, 0.0,   -5.0, -5.1, -4.9, -5.0, -5.1, -5.1, -5.0};

    param.A = convert_vector<double, T>(A);
    // A contains three clusters, some noise points and some non-core points
    param.lda = 25;
    param.min_samples = 4;
    param.eps = 1.5;

    std::vector<double> expected_rinfo_double{25, 2, 25, 1.5, 4, 30, 2, 11, 3};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo_double);
    param.expected_labels =
        std::vector<da_int>{0, 0,  0,  0,  0,  -1, 1, 1, 1, 1, 1, 1, 1,
                            1, -1, -1, -1, -1, 2,  2, 2, 2, 2, 2, 2, 2};
    param.expected_core_sample_indices =
        std::vector<da_int>{0, 6, 7, 12, 18, 19, 20, 21, 22, 23, 24};
    param.expected_n_core_samples = 11;
    param.expected_n_clusters = 3;
}

// The tests below will be expanded when further options are added to the DBSCAN API

template <typename T> void GetRowMajorData(std::vector<DBSCANParamType<T>> &params) {
    // Tests with a data matrix in row-major order
    DBSCANParamType<T> param;
    Get25by2BaseData(param);
    param.test_name = "25 by 2 data matrix in row-major order";
    param.order = "row-major";
    param.lda = 2;
    da_switch_order_in_place(column_major, param.n_samples, param.n_features,
                             param.A.data(), param.n_samples, param.lda);
    param.expected_rinfo[2] = 2;
    params.push_back(param);
}

template <typename T> void Get1by1Data(std::vector<DBSCANParamType<T>> &params) {
    // Tests with a 1 x 1 data matrix
    DBSCANParamType<T> param;
    Get1by1BaseData(param);
    params.push_back(param);
    param.algorithm = "brute serial";
    param.test_name = "1 by 1 data matrix with serial DBSCAN";
    params.push_back(param);
}

template <typename T> void GetZeroData(std::vector<DBSCANParamType<T>> &params) {
    // Tests with a data matrix full of zeros
    DBSCANParamType<T> param;
    GetZeroBaseData(param);
    params.push_back(param);
    param.algorithm = "brute serial";
    param.test_name = "Empty data matrix with serial DBSCAN";
    params.push_back(param);
}

template <typename T> void Get30by3Data(std::vector<DBSCANParamType<T>> &params) {
    // Tests with a 30 x 3 data matrix
    DBSCANParamType<T> param;
    Get30by3BaseData(param);
    params.push_back(param);
    param.algorithm = "brute serial";
    param.test_name = "30 by 3 data matrix with serial DBSCAN";
    params.push_back(param);
}

template <typename T> void Get25by2Data(std::vector<DBSCANParamType<T>> &params) {
    // Tests with a 25 x 2 data matrix
    DBSCANParamType<T> param;
    Get25by2BaseData(param);
    params.push_back(param);
    param.algorithm = "brute serial";
    param.test_name = "25 by 2 data matrix with serial DBSCAN";
    params.push_back(param);
    Get25by2BaseData(param);
    param.test_name = "25 by 2 data matrix with tiny eps";
    param.eps = (T)0.0001;
    param.expected_labels =
        std::vector<da_int>{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    param.expected_n_clusters = 0;
    param.expected_n_core_samples = 0;
    param.expected_rinfo[3] = 0.0001;
    param.expected_rinfo[7] = 0;
    param.expected_rinfo[8] = 0;
    params.push_back(param);
    param.algorithm = "brute serial";
    param.test_name = "25 by 2 data matrix with tiny eps and serial DBSCAN";
    params.push_back(param);
    Get25by2BaseData(param);
    param.test_name = "25 by 2 data matrix stored in a subarray";
    std::vector<double> A{0,    1.0,  1.0,  -1.0, -1.0, 10.0,  -7.0,  -6.0,  -8.0,
                          -6.5, -5.0, -8.0, -6.5, -5.0, -12.0, -12.1, -11.9, 15.0,
                          -5.0, -5.1, -4.9, -5.0, -5.1, -5.1,  -5.0,  0,     0,
                          0,    1.0,  -1.0, 1.0,  -1.0, 10.0,  2.0,   2.0,   3.0,
                          3.1,  3.0,  1.0,  1.0,  1.0,  -12.0, -12.1, -11.9, 0.0,
                          -5.0, -5.1, -4.9, -5.0, -5.1, -5.1,  -5.0,  0,     0};
    param.lda = 27;
    param.A = convert_vector<double, T>(A);
    param.expected_rinfo[2] = 27;
    params.push_back(param);
}

template <typename T> void GetDBSCANData(std::vector<DBSCANParamType<T>> &params) {
    Get1by1Data(params);
    Get30by3Data(params);
    Get25by2Data(params);
    GetZeroData(params);
    GetRowMajorData(params);
}
