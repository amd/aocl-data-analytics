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

#include <limits>
#include <list>
#include <string.h>

#include "../utest_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

template <typename T> struct KMeansParamType {

    std::string test_name;

    bool is_random = false;

    da_int n_samples = 0;
    da_int n_features = 0;
    std::vector<T> A;
    da_int lda = 0;

    std::vector<T> C;
    da_int ldc = 0;
    da_int m_samples = 0;
    da_int m_features = 0;
    std::vector<T> X;
    da_int ldx = 0;
    std::vector<T> X_transform;
    da_int ldx_transform = 0;

    da_int k_samples = 0;
    da_int k_features = 0;
    std::vector<T> Y;
    da_int ldy = 0;
    std::vector<da_int> Y_labels;

    da_int n_clusters = 1;
    da_int n_init = 1;
    da_int max_iter = 50;
    da_int seed = 0;
    T convergence_tolerance = (T)1e-4;
    std::string initialization_method;
    std::string algorithm;
    std::string order = "column-major";

    std::vector<T> expected_rinfo;
    std::vector<T> expected_centres;
    std::vector<da_int> expected_labels;
    std::vector<T> expected_X_transform;
    std::vector<da_int> expected_Y_labels;

    da_status expected_status = da_status_success;
    T tol = 10 * std::numeric_limits<T>::epsilon();
    T max_allowed_inertia = (T)0.0;
};

template <typename T> void Get1by1BaseData(KMeansParamType<T> &param) {
    param.test_name = "1 by 1 data matrix";

    param.n_samples = 1;
    param.n_features = 1;
    std::vector<double> A{2.1};
    param.A = convert_vector<double, T>(A);
    param.lda = 1;

    std::vector<double> C{2.3};
    param.C = convert_vector<double, T>(C);
    param.ldc = 1;

    param.m_samples = 1;
    param.m_features = 1;
    std::vector<double> X{3.3};
    param.X = convert_vector<double, T>(X);
    param.ldx = 1;
    std::vector<double> X_transform{0.0};
    param.X_transform = convert_vector<double, T>(X_transform);
    param.ldx_transform = 1;

    param.k_samples = 1;
    param.k_features = 1;
    std::vector<double> Y{1.3};
    param.Y = convert_vector<double, T>(Y);
    param.ldy = 1;
    std::vector<da_int> Y_labels{0};
    param.Y_labels = Y_labels;

    param.n_clusters = 1;
    param.n_init = 1;
    param.max_iter = 30;
    param.seed = 78;
    param.convergence_tolerance = (T)1.0e-4;
    param.initialization_method = "k-means++";
    param.algorithm = "elkan";

    std::vector<double> expected_rinfo{1.0, 1.0, 1.0, 0.0, 0.0};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);
    std::vector<double> expected_centres{2.1};
    param.expected_centres = convert_vector<double, T>(expected_centres);
    std::vector<da_int> expected_labels{0};
    param.expected_labels = expected_labels;
    std::vector<double> expected_X_transform{1.2};
    param.expected_X_transform = convert_vector<double, T>(expected_X_transform);
    std::vector<da_int> expected_Y_labels{0};
    param.expected_Y_labels = expected_Y_labels;

    param.tol = 10 * std::numeric_limits<T>::epsilon();
    param.expected_status = da_status_success;
}

template <typename T> void GetZeroBaseData(KMeansParamType<T> &param) {
    param.test_name = "Data matrix full of zeros";

    param.n_samples = 5;
    param.n_features = 3;
    std::vector<double> A{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    param.A = convert_vector<double, T>(A);
    param.lda = 5;

    std::vector<double> C{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    param.C = convert_vector<double, T>(C);
    param.ldc = 2;

    param.m_samples = 3;
    param.m_features = 3;
    std::vector<double> X{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    param.X = convert_vector<double, T>(X);
    param.ldx = 3;
    std::vector<double> X_transform{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    param.X_transform = convert_vector<double, T>(X_transform);
    param.ldx_transform = 3;

    param.k_samples = 2;
    param.k_features = 3;
    std::vector<double> Y{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    param.Y = convert_vector<double, T>(Y);
    param.ldy = 2;
    std::vector<da_int> Y_labels{0, 0, 0};
    param.Y_labels = Y_labels;

    param.n_clusters = 2;
    param.n_init = 1;
    param.max_iter = 3;
    param.seed = -1;
    param.convergence_tolerance = (T)1.0e-4;
    param.initialization_method = "supplied";
    param.algorithm = "hartigan-wong";

    std::vector<double> expected_rinfo{5.0, 3.0, 2.0, 0.0, 0.0};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);
    std::vector<double> expected_centres{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    param.expected_centres = convert_vector<double, T>(expected_centres);
    std::vector<da_int> expected_labels{0, 0, 0, 0, 0};
    param.expected_labels = expected_labels;
    std::vector<double> expected_X_transform{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    param.expected_X_transform = convert_vector<double, T>(expected_X_transform);
    std::vector<da_int> expected_Y_labels{0, 0};
    param.expected_Y_labels = expected_Y_labels;

    param.tol = 100 * std::sqrt(std::numeric_limits<T>::epsilon());
    param.max_allowed_inertia = (T)0.0;
    param.expected_status = da_status_success;
}

template <typename T> void Get3ClustersBaseData(KMeansParamType<T> &param) {
    param.test_name = "Data matrix in three distinct clusters";

    param.n_samples = 10;
    param.n_features = 2;
    std::vector<double> A{1.0, 1.1, 0.5,  0.49, -2.0, -2.0, 0.53, 0.9,  1.2, -1.8,
                          1.0, 1.2, -2.0, -1.9, 0.5,  0.51, -2.1, 0.95, 0.8, 0.6};
    param.A = convert_vector<double, T>(A);
    param.lda = 10;

    std::vector<double> C{0.5, 0.7, -1.3, 0.5, -1.7, 0.2};
    param.C = convert_vector<double, T>(C);
    param.ldc = 3;

    param.m_samples = 4;
    param.m_features = 2;
    std::vector<double> X{0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, -1.0};
    param.X = convert_vector<double, T>(X);
    param.ldx = 4;
    std::vector<double> X_transform{0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    param.X_transform = convert_vector<double, T>(X_transform);
    param.ldx_transform = 4;

    param.k_samples = 3;
    param.k_features = 2;
    std::vector<double> Y{0.5, 0.5, -1.0, 0.5, -1.0, 0.0};
    param.Y = convert_vector<double, T>(Y);
    param.ldy = 3;
    std::vector<da_int> Y_labels{0, 0, 0};
    param.Y_labels = Y_labels;

    param.n_clusters = 3;
    param.n_init = 1;
    param.max_iter = 50;
    param.seed = 78;
    param.convergence_tolerance = (T)1.0e-4;
    param.initialization_method = "supplied";
    param.algorithm = "hartigan-wong";

    std::vector<double> expected_rinfo{10.0, 2.0, 3.0, 1.0, 0.185475};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);
    std::vector<double> expected_centres{1.05,   0.506667, -1.93333333333,
                                         0.9875, -2.0,     0.53666666666};
    param.expected_centres = convert_vector<double, T>(expected_centres);
    std::vector<da_int> expected_labels{0, 0, 1, 1, 2, 2, 1, 0, 0, 2};
    param.expected_labels = expected_labels;
    std::vector<double> expected_X_transform{
        1.4414077320453085, 1.0500744021258683, 0.9887650125282548, 2.2478114355968564,
        2.0631798542810347, 3.042484364973978,  2.0599460618612757, 1.1210312712458608,
        2.0064368639179446, 1.9880783574989047, 2.9820220581939925, 2.469640099735629};
    param.expected_X_transform = convert_vector<double, T>(expected_X_transform);
    std::vector<da_int> expected_Y_labels{0, 1, 2};
    param.expected_Y_labels = expected_Y_labels;

    param.tol = 100 * std::sqrt(std::numeric_limits<T>::epsilon());
    param.max_allowed_inertia = (T)0.185475;
    param.expected_status = da_status_success;
}

template <typename T> void GetRowMajorBaseData(KMeansParamType<T> &param) {
    param.test_name =
        "Data matrix in three distinct clusters, stored in row major format";

    param.n_samples = 10;
    param.n_features = 2;
    std::vector<double> A{1.0,  1.0,  1.1,  1.2,  0.5, -2.0, 0.49, -1.9, -2.0, 0.5,
                          -2.0, 0.51, 0.53, -2.1, 0.9, 0.95, 1.2,  0.8,  -1.8, 0.6};
    param.A = convert_vector<double, T>(A);
    param.lda = 2;

    std::vector<double> C{0.5, 0.5, 0.7, -1.7, -1.3, 0.2};
    param.C = convert_vector<double, T>(C);
    param.ldc = 2;

    param.m_samples = 4;
    param.m_features = 2;
    std::vector<double> X{0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0};
    param.X = convert_vector<double, T>(X);
    param.ldx = 2;
    std::vector<double> X_transform{0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    param.X_transform = convert_vector<double, T>(X_transform);
    param.ldx_transform = 3;

    param.k_samples = 3;
    param.k_features = 2;
    std::vector<double> Y{0.5, 0.5, 0.5, -1.0, -1.0, 0.0};
    param.Y = convert_vector<double, T>(Y);
    param.ldy = 2;
    std::vector<da_int> Y_labels{0, 0, 0};
    param.Y_labels = Y_labels;

    param.n_clusters = 3;
    param.n_init = 1;
    param.max_iter = 50;
    param.seed = 78;
    param.convergence_tolerance = (T)1.0e-4;
    param.initialization_method = "supplied";
    param.algorithm = "hartigan-wong";
    param.order = "row-major";

    std::vector<double> expected_rinfo{10.0, 2.0, 3.0, 1.0, 0.185475};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);
    std::vector<double> expected_centres{1.05, 0.9875,         0.506667,
                                         -2.0, -1.93333333333, 0.53666666666};
    param.expected_centres = convert_vector<double, T>(expected_centres);
    std::vector<da_int> expected_labels{0, 0, 1, 1, 2, 2, 1, 0, 0, 2};
    param.expected_labels = expected_labels;
    std::vector<double> expected_X_transform{
        1.4414077320453085, 2.0631798542810347, 2.0064368639179446, 1.0500744021258683,
        3.042484364973978,  1.9880783574989047, 0.9887650125282548, 2.0599460618612757,
        2.9820220581939925, 2.2478114355968564, 1.1210312712458608, 2.469640099735629};
    param.expected_X_transform = convert_vector<double, T>(expected_X_transform);
    std::vector<da_int> expected_Y_labels{0, 1, 2};
    param.expected_Y_labels = expected_Y_labels;

    param.tol = 100 * std::sqrt(std::numeric_limits<T>::epsilon());
    param.max_allowed_inertia = (T)0.185475;
    param.expected_status = da_status_success;
}

template <typename T> void GetSubarrayBaseData(KMeansParamType<T> &param) {
    param.test_name = "Data matrix in three distinct clusters but stored in subarrays";

    param.n_samples = 10;
    param.n_features = 2;
    std::vector<double> A{1.0, 1.1,  0.5,  0.49, -2.0, -2.0, 0.53, 0.9,
                          1.2, -1.8, 0.0,  0.0,  1.0,  1.2,  -2.0, -1.9,
                          0.5, 0.51, -2.1, 0.95, 0.8,  0.6,  0.0,  0.0};
    param.A = convert_vector<double, T>(A);
    param.lda = 12;

    std::vector<double> C{0.5, 0.7, -1.3, 0.0, 0.5, -1.7, 0.2, 0.0};
    param.C = convert_vector<double, T>(C);
    param.ldc = 4;

    param.m_samples = 4;
    param.m_features = 2;
    std::vector<double> X{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0};
    param.X = convert_vector<double, T>(X);
    param.ldx = 6;
    std::vector<double> X_transform{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    param.X_transform = convert_vector<double, T>(X_transform);
    param.ldx_transform = 5;

    param.k_samples = 3;
    param.k_features = 2;
    std::vector<double> Y{0.5, 0.5, -1.0, 0.0, 0.5, -1.0, 0.0, 0.0};
    param.Y = convert_vector<double, T>(Y);
    param.ldy = 4;
    std::vector<da_int> Y_labels{0, 0, 0};
    param.Y_labels = Y_labels;

    param.n_clusters = 3;
    param.n_init = 1;
    param.max_iter = 30;
    param.seed = 78;
    param.convergence_tolerance = (T)1.0e-4;
    param.initialization_method = "supplied";
    param.algorithm = "hartigan-wong";

    std::vector<double> expected_rinfo{10.0, 2.0, 3.0, 1.0, 0.185475};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);
    std::vector<double> expected_centres{1.05,   0.506667, -1.93333333333,
                                         0.9875, -2.0,     0.53666666666};
    param.expected_centres = convert_vector<double, T>(expected_centres);
    std::vector<da_int> expected_labels{0, 0, 1, 1, 2, 2, 1, 0, 0, 2};
    param.expected_labels = expected_labels;
    std::vector<double> expected_X_transform{1.4414077320453085,
                                             1.0500744021258683,
                                             0.9887650125282548,
                                             2.2478114355968564,
                                             0.0,
                                             2.0631798542810347,
                                             3.042484364973978,
                                             2.0599460618612757,
                                             1.1210312712458608,
                                             0.0,
                                             2.0064368639179446,
                                             1.9880783574989047,
                                             2.9820220581939925,
                                             2.469640099735629,
                                             0.0};
    param.expected_X_transform = convert_vector<double, T>(expected_X_transform);
    std::vector<da_int> expected_Y_labels{0, 1, 2};
    param.expected_Y_labels = expected_Y_labels;

    param.tol = 100 * std::sqrt(std::numeric_limits<T>::epsilon());
    param.max_allowed_inertia = (T)0.185475;
    param.expected_status = da_status_success;
}

template <typename T> void GetLargeBaseData(KMeansParamType<T> &param) {
    param.test_name = "Larger data matrix to test vectorized kernels";

    param.n_samples = 80;
    param.n_features = 2;
    std::vector<double> A{
        2.1,  -1.0, 3.0,  2.0,  -3.0, -2.2, -2.4, 0.9,  1.0,  -2.0, 2.0,  1.0,  -2.0,
        -1.0, -3.0, 2.3,  2.1,  -1.5, 3.0,  2.0,  -3.0, -2.8, -2.8, 1.0,  1.2,  -2.0,
        2.1,  3.7,  -2.0, -1.0, -3.8, 2.0,  2.0,  -3.9, 3.0,  2.0,  -3.9, -2.7, -2.0,
        1.0,  1.0,  -2.4, 2.1,  3.0,  -2.6, -1.5, -3.4, 2.0,  2.0,  -1.0, 3.0,  2.6,
        -3.0, -2.6, -2.0, 1.0,  1.3,  -2.5, 2.8,  3.5,  -2.0, -1.0, -3.0, 2.2,  2.0,
        -1.0, 3.4,  2.0,  -3.8, -2.0, -2.0, 1.0,  1.9,  -2.1, 2.1,  3.0,  -2.0, -1.0,
        -3.4, 2.0,  2.0,  -8.0, 3.7,  2.0,  -3.2, -2.6, -2.2, 1.0,  1.0,  -2.5, 6.0,
        -5.5, -2.0, -1.4, -3.7, 2.0,  2.0,  -1.2, 3.4,  2.0,  -4.0, -2.0, -2.0, 1.0,
        1.0,  -2.0, 2.0,  3.3,  -2.0, -1.4, -3.0, 2.3,  2.0,  -9.0, 3.0,  2.0,  -3.0,
        -2.0, -2.0, 1.7,  1.3,  -2.0, 2.3,  3.3,  -2.0, -6.0, -3.0, 2.7,  2.0,  -1.5,
        3.0,  2.6,  -3.3, -2.0, -2.0, 1.0,  1.7,  -2.0, 2.0,  8.5,  -2.0, -1.7, -3.0,
        2.0,  2.2,  -1.2, 3.6,  2.9,  -6.0, -2.0, -2.1, 1.5,  9.0,  -2.0, 2.0,  3.8,
        -2.0, -1.0, -8.1, 2.9};
    param.A = convert_vector<double, T>(A);
    param.lda = 80;

    std::vector<double> C{1.9, -5.1, 1.0,  -3.0, 1.0,  -3.2, 6.0, -3.7,
                          8.5, -3.0, 1.7,  -3.0, 1.0,  -7.0, 1.0, -3.4,
                          4.8, -3.0, 1.0,  -3.4, 1.9,  -3.0, 1.3, -3.8,
                          5.4, -3.2, -8.0, -3.0, -6.1, -3.3, 1.6, -3.0};
    param.C = convert_vector<double, T>(C);
    param.ldc = 16;

    param.m_samples = 1;
    param.m_features = 2;
    std::vector<double> X{-3.700001, -7.7};
    param.X = convert_vector<double, T>(X);
    param.ldx = 1;
    std::vector<double> X_transform{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    param.X_transform = convert_vector<double, T>(X_transform);
    param.ldx_transform = 1;

    param.k_samples = 1;
    param.k_features = 2;
    std::vector<double> Y{-3.7, -7.7};
    param.Y = convert_vector<double, T>(Y);
    param.ldy = 1;
    std::vector<da_int> Y_labels{0};
    param.Y_labels = Y_labels;

    param.n_clusters = 16;
    param.n_init = 1;
    param.max_iter = 30;
    param.seed = 78;
    param.convergence_tolerance = (T)1.0e-4;
    param.initialization_method = "supplied";
    param.algorithm = "lloyd";
    param.is_random = false;
    param.order = "column-major";

    std::vector<double> expected_rinfo{80.0, 2.0, 16.0, 4.0, 45.78715};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);
    std::vector<double> expected_centres{
        2.4666666667, 0.0,  1.02000,        -3.0, 2.42593,      0.0,   0.0,
        -3.7,         0.0,  -3.0,           -1.0, -1.94815,     -0.25, 0.0,
        1.075,        -3.7, 7.833333333333, 0.0,  1.0,          -3.85, 2.55555556,
        0.0,          0.0,  -7.7,           0.0,  -3.166666667, -8.0,  -1.88148,
        -5.75,        0.0,  1.55,           -3.0};
    param.expected_centres = convert_vector<double, T>(expected_centres);
    std::vector<da_int> expected_labels{
        4,  10, 4,  4, 9,  11, 11, 2, 2,  11, 0,  12, 11, 11, 3,  4, 4,  11, 4,  4,
        3,  11, 11, 2, 2,  11, 4,  4, 11, 11, 15, 4,  4,  7,  4,  4, 15, 11, 11, 14,
        14, 11, 4,  4, 11, 12, 15, 4, 4,  11, 4,  4,  9,  11, 11, 2, 14, 11, 4,  0,
        11, 11, 9,  4, 4,  11, 4,  4, 7,  11, 11, 14, 0,  11, 4,  4, 11, 11, 7,  4};
    param.expected_labels = expected_labels;
    std::vector<double> expected_X_transform{
        16.712636602948745, 8.5428332536694178, 9.8978987669100764, 3.9131189606246322,
        11.945852343094563, 8.5428332536694178, 8.5428332536694178, 0.0,
        8.5428332536694178, 4.5870590917396212, 2.7166155414412292, 6.0765238962074113,
        3.9629534440868719, 8.5428332536694178, 10.409761044327579, 4.7000000000000002};
    param.expected_X_transform = convert_vector<double, T>(expected_X_transform);
    std::vector<da_int> expected_Y_labels{7};
    param.expected_Y_labels = expected_Y_labels;

    param.tol = 10000 * std::sqrt(std::numeric_limits<T>::epsilon());
    param.max_allowed_inertia = (T)46.0;
    param.expected_status = da_status_success;
}

template <typename T> void GetPseudoRandomBaseData(KMeansParamType<T> &param) {
    param.test_name = "Data matrix with pseudorandom values";

    param.is_random = true;

    param.n_samples = 20;
    param.n_features = 2;
    std::vector<double> A{0.31,  0.61,  0.65,  -0.49, -0.7,  -0.35, 0.53, 0.29,
                          0.23,  -0.58, 1.0,   0.23,  -0.04, -0.79, 0.25, 0.51,
                          -0.41, 0.95,  -0.81, -0.61, -0.41, 0.12,  0.75, 0.49,
                          -0.47, -0.85, -0.53, 0.19,  0.25,  -0.82, 0.52, -0.26,
                          -0.01, -0.49, 0.56,  0.51,  -0.61, 0.95,  0.83, -0.76};
    param.A = convert_vector<double, T>(A);
    param.lda = 20;

    param.n_clusters = 3;
    param.n_init = 1;
    param.max_iter = 30;
    param.seed = 593228;
    param.convergence_tolerance = (T)1.0e-4;
    param.initialization_method = "random";
    param.algorithm = "hartigan-wong";

    param.tol = 1000 * std::sqrt(std::numeric_limits<T>::epsilon());
    param.max_allowed_inertia = (T)4.8;
    param.expected_status = da_status_success;
}

template <typename T> void Get1by1Data(std::vector<KMeansParamType<T>> &params) {
    // Tests with a 1 x 1 data matrix
    KMeansParamType<T> param;
    Get1by1BaseData(param);
    params.push_back(param);
    param.initialization_method = "random";
    params.push_back(param);
    param.algorithm = "lloyd";
    params.push_back(param);
    param.initialization_method = "random partitions";
    params.push_back(param);
    param.algorithm = "macqueen";
    params.push_back(param);
    param.initialization_method = "supplied";
    params.push_back(param);
}

template <typename T> void Get3ClustersData(std::vector<KMeansParamType<T>> &params) {
    // Tests with a data matrix in 3 distinct clusters
    KMeansParamType<T> param;
    Get3ClustersBaseData(param);
    // Three cluster tests
    params.push_back(param);
    param.algorithm = "lloyd";
    param.expected_rinfo[3] = (T)1.0;
    params.push_back(param);
    param.algorithm = "macqueen";
    param.expected_rinfo[3] = (T)0.0;
    params.push_back(param);
    param.expected_rinfo[3] = (T)1.0;
    param.algorithm = "elkan";
    params.push_back(param);
    // Tests with some inherent randomness
    param.max_iter = 30;
    param.is_random = true;
    param.initialization_method = "k-means++";
    param.n_init = 10;
    params.push_back(param);
    param.initialization_method = "random";
    params.push_back(param);
    param.initialization_method = "random partitions";
    params.push_back(param);
    // Tests looking for n or 1 clusters
    param.n_init = 1;
    param.n_clusters = 1;
    param.max_allowed_inertia = (T)34.89176;
    params.push_back(param);
    param.initialization_method = "k-means++";
    param.algorithm = "lloyd";
    param.expected_rinfo[3] = (T)1.0;
    params.push_back(param);
    param.initialization_method = "random";
    param.algorithm = "macqueen";
    param.max_iter = 30;
    params.push_back(param);
    param.n_clusters = 10;
    param.initialization_method = "k-means++";
    param.algorithm = "lloyd";
    param.max_allowed_inertia = 0.11;
    params.push_back(param);
    param.initialization_method = "random";
    param.algorithm = "macqueen";
    params.push_back(param);
    param.n_init = 5;
    param.max_allowed_inertia = (T)0.11;
    param.max_iter = 30;
    param.initialization_method = "random partitions";
    param.algorithm = "elkan";
    params.push_back(param);
    // Tests looking for 2 or 5 clusters, comparing against Scikit-learn inertia
    param.n_init = 10;
    param.n_clusters = 2;
    param.max_allowed_inertia = (T)15.7915238095238;
    param.initialization_method = "k-means++";
    param.algorithm = "lloyd";
    params.push_back(param);
    param.initialization_method = "random";
    param.algorithm = "elkan";
    params.push_back(param);
    param.algorithm = "hartigan-wong";
    params.push_back(param);
    param.initialization_method = "random partitions";
    param.algorithm = "macqueen";
    params.push_back(param);
    param.n_clusters = 5;
    param.max_allowed_inertia = (T)0.076;
    param.initialization_method = "k-means++";
    param.algorithm = "lloyd";
    params.push_back(param);
    param.initialization_method = "random";
    param.max_allowed_inertia = (T)0.11;
    param.algorithm = "elkan";
    params.push_back(param);
    param.algorithm = "hartigan-wong";
    params.push_back(param);
    param.n_init = 100;
    param.max_allowed_inertia = (T)0.12;
    param.initialization_method = "random partitions";
    param.algorithm = "macqueen";
    param.max_iter = 30;
    params.push_back(param);
}

template <typename T> void GetZeroData(std::vector<KMeansParamType<T>> &params) {
    // Tests with a data matrix full of zeros
    KMeansParamType<T> param;
    GetZeroBaseData(param);
    params.push_back(param);
    param.algorithm = "elkan";
    param.initialization_method = "k-means++";
    param.n_init = 3;
    std::vector<double> expected_rinfo{5.0, 3.0, 2.0, 2.0, 0.0};
    param.expected_rinfo = convert_vector<double, T>(expected_rinfo);
    param.expected_status = da_status_success;
    params.push_back(param);
    param.algorithm = "lloyd";
    param.initialization_method = "random";
    params.push_back(param);
    param.algorithm = "macqueen";
    param.initialization_method = "random partitions";
    params.push_back(param);
}

template <typename T> void GetPseudoRandomData(std::vector<KMeansParamType<T>> &params) {
    // Tests with a data matrix with fairly random points to properly exercise the Elkan algorithm for code coverage purposes
    KMeansParamType<T> param;
    GetPseudoRandomBaseData(param);
    // Start with Hartigan-Wong for comparison and to get 'correct' answer
    params.push_back(param);
    param.algorithm = "elkan";
    params.push_back(param);
}

template <typename T> void GetLargeData(std::vector<KMeansParamType<T>> &params) {
    // Tests with a larger data matrix to exercise vectorized kernels
    KMeansParamType<T> param;
    GetLargeBaseData(param);
    // Start with Hartigan-Wong for comparison and to get 'correct' answer
    params.push_back(param);
    param.algorithm = "elkan";
    params.push_back(param);
}

template <typename T> void GetSubarrayData(std::vector<KMeansParamType<T>> &params) {
    // Tests with a data matrix in 3 distinct clusters
    KMeansParamType<T> param;
    GetSubarrayBaseData(param);
    // Three cluster tests, data stored in a subarray
    params.push_back(param);
    param.expected_rinfo[3] = (T)1.0;
    param.algorithm = "lloyd";
    params.push_back(param);
    param.algorithm = "macqueen";
    param.expected_rinfo[3] = (T)0.0;
    params.push_back(param);
    param.algorithm = "elkan";
    param.expected_rinfo[3] = (T)1.0;
    params.push_back(param);
    param.max_iter = 30;
    // Tests with some inherent randomness
    param.is_random = true;
    param.initialization_method = "k-means++";
    param.n_init = 10;
    params.push_back(param);
    param.initialization_method = "random";
    params.push_back(param);
    param.initialization_method = "random partitions";
    params.push_back(param);
}

template <typename T> void GetRowMajorData(std::vector<KMeansParamType<T>> &params) {
    // Tests with a data matrix in 3 distinct clusters
    KMeansParamType<T> param;
    GetRowMajorBaseData(param);
    params.push_back(param);
}

template <typename T> void GetKMeansData(std::vector<KMeansParamType<T>> &params) {
    Get1by1Data(params);
    Get3ClustersData(params);
    GetSubarrayData(params);
    GetZeroData(params);
    GetPseudoRandomData(params);
    GetRowMajorData(params);
    GetLargeData(params);
}
