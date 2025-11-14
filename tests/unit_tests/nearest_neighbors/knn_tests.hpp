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
#include <string>

#include "../datests_cblas.hh"
#include "../utest_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

static std::list<std::string> MetricType = {
    "euclidean", "sqeuclidean", "l2", "minkowski", "euclidean_gemm", "sqeuclidean_gemm"};
static std::list<std::string> AlgoType = {"brute", "kd tree", "ball tree", "auto"};
static std::list<std::string> WeightsType = {"uniform", "distance"};
static std::list<da_int> NumNeighConstructor = {3, 5};
static std::list<da_int> NumNeighKNeighAPI = {3, 4};
static std::list<da_int> leaf_size = {1, 3, 10, 20};

template <typename T> struct KNNParamType {

    std::string name;

    da_int n_samples = 1;
    da_int n_features = 1;
    std::vector<T> X_train;
    da_int ldx_train = 1;
    std::vector<da_int> y_train_class;
    std::vector<T> y_train_regression;

    da_int n_queries = 1;
    std::vector<T> X_test;
    da_int ldx_test = 1;

    da_int n_neigh_knn = 1;
    da_int n_neigh_kneighbors = 1;
    std::string metric = "euclidean";
    std::string algorithm = "brute";
    std::string weights = "uniform";
    std::string order = "column-major";

    da_int leaf_size = 20;
    da_status expected_status = da_status_success;
    T tol = 40 * std::numeric_limits<T>::epsilon();

    // Data of expected solution
    std::vector<T> expected_kdist;
    std::vector<da_int> expected_kind;
    std::vector<T> expected_proba;
    std::vector<da_int> expected_labels;
    std::vector<T> expected_targets;

    // Set constructor to initialize data in test bodies as simply as possible.
    KNNParamType(){};
    KNNParamType(da_int n_neigh_knn, da_int n_neigh_kneighbors, std::string metric,
                 std::string algorithm, std::string weights, da_int leaf_size = 20)
        : n_neigh_knn(n_neigh_knn), n_neigh_kneighbors(n_neigh_kneighbors),
          metric(metric), algorithm(algorithm), weights(weights), leaf_size(leaf_size){};
};

template <typename T> void get_expected_kind_k_dist(KNNParamType<T> &param) {
    if ((param.metric == "euclidean") || (param.metric == "sqeuclidean") ||
        (param.metric == "l2") || (param.metric == "minkowski") ||
        (param.metric == "euclidean_gemm") || (param.metric == "sqeuclidean_gemm")) {
        if (param.n_neigh_kneighbors == 3) {
            std::vector<da_int> kind{1, 2, 3, 0, 0, 5, 3, 1, 4};
            param.expected_kind = convert_vector<da_int, da_int>(kind);
            std::vector<T> kdist{3.,
                                 2.,
                                 4.58257569495584,
                                 3.3166247903554,
                                 3.1622776601683795,
                                 5.477225575051661,
                                 3.7416573867739413,
                                 4.242640687119285,
                                 5.656854249492381};
            if ((param.metric == "sqeuclidean") || (param.metric == "sqeuclidean_gemm")) {
                for (auto &kd : kdist)
                    kd = kd * kd;
            }
            param.expected_kdist = convert_vector<T, T>(kdist);
        } else if (param.n_neigh_kneighbors == 4) {
            std::vector<da_int> kind{1, 2, 3, 0, 0, 5, 3, 1, 4, 4, 5, 2};
            param.expected_kind = convert_vector<da_int, da_int>(kind);
            std::vector<T> kdist{3.,
                                 2.,
                                 4.58257569495584,
                                 3.3166247903554,
                                 3.1622776601683795,
                                 5.477225575051661,
                                 3.7416573867739413,
                                 4.242640687119285,
                                 5.656854249492381,
                                 5.385164807134504,
                                 5.0990195135927845,
                                 6.164414002968976};
            if ((param.metric == "sqeuclidean") || (param.metric == "sqeuclidean_gemm")) {
                for (auto &kd : kdist)
                    kd = kd * kd;
            }
            param.expected_kdist = convert_vector<T, T>(kdist);
        } else {
            throw std::runtime_error("n_neigh_kneigh must be 3 or 4");
        }
    } else {
        throw std::runtime_error("metric must be euclidean or sqeuclidean");
    }
}

template <typename T> void get_proba(KNNParamType<T> &param) {
    if (param.n_neigh_knn == 5) {
        if (param.weights == "uniform") {
            std::vector<T> proba{0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4};
            param.expected_proba = convert_vector<T, T>(proba);
        } else if (param.weights == "distance") {
            if ((param.metric == "euclidean") || (param.metric == "l2") ||
                (param.metric == "minkowski") || (param.metric == "euclidean_gemm")) {
                std::vector<T> proba{
                    0.1379511568268668, 0.3515868265794006, 0.1798440493222374,
                    0.4507346784224799, 0.3447698547319956, 0.4217676420329797,
                    0.4113141647506533, 0.3036433186886039, 0.3983883086447829};
                param.expected_proba = convert_vector<T, T>(proba);
            } else if ((param.metric == "sqeuclidean") ||
                       (param.metric == "sqeuclidean_gemm")) {
                std::vector<T> proba{
                    0.0895917616770872, 0.5270701941190925, 0.1596502898413,
                    0.4799558661272527, 0.2747153739044967, 0.4485412905065095,
                    0.4304523721956602, 0.1982144319764108, 0.3918084196521904};
                param.expected_proba = convert_vector<T, T>(proba);
            }
        } else
            throw std::runtime_error("Weights must be uniform or distance");
    } else if (param.n_neigh_knn == 3) {
        if (param.weights == "uniform") {
            std::vector<T> proba{0.,
                                 0.3333333333333333,
                                 0.,
                                 0.6666666666666666,
                                 0.3333333333333333,
                                 0.3333333333333333,
                                 0.3333333333333333,
                                 0.3333333333333333,
                                 0.6666666666666666};
            param.expected_proba = convert_vector<T, T>(proba);
        } else if (param.weights == "distance") {
            if ((param.metric == "euclidean") || (param.metric == "l2") ||
                (param.metric == "minkowski") || (param.metric == "euclidean_gemm")) {
                std::vector<T> proba{0.,
                                     0.47531678671182,
                                     0.,
                                     0.6304942401901729,
                                     0.3006167312243614,
                                     0.3778214838715254,
                                     0.3695057598098272,
                                     0.2240664820638185,
                                     0.6221785161284746};
                param.expected_proba = convert_vector<T, T>(proba);
            } else if ((param.metric == "sqeuclidean") ||
                       (param.metric == "sqeuclidean_gemm")) {
                std::vector<T> proba{0.,
                                     0.6164383561643836,
                                     0.,
                                     0.5936675461741425,
                                     0.2465753424657534,
                                     0.4244031830238727,
                                     0.4063324538258575,
                                     0.136986301369863,
                                     0.5755968169761273};
                param.expected_proba = convert_vector<T, T>(proba);
            }
        } else
            throw std::runtime_error("Weights must be uniform or distance");
    } else {
        throw std::runtime_error("n_neigh_knn must be 5 or 2");
    }
}

template <typename T> void get_labels(KNNParamType<T> &param) {
    if (param.n_neigh_knn == 5) {
        if (param.weights == "uniform") {
            std::vector<da_int> expected_labels{1, 1, 1};
            param.expected_labels = convert_vector<da_int, da_int>(expected_labels);
        } else if (param.weights == "distance") {
            std::vector<da_int> expected_labels{1, 0, 1};
            param.expected_labels = convert_vector<da_int, da_int>(expected_labels);
        } else
            throw std::runtime_error("Weights must be uniform or distance");
    } else if (param.n_neigh_knn == 3) {
        if (param.weights == "uniform") {
            std::vector<da_int> expected_labels{1, 0, 2};
            param.expected_labels = convert_vector<da_int, da_int>(expected_labels);
        } else if (param.weights == "distance") {
            std::vector<da_int> expected_labels{1, 0, 2};
            param.expected_labels = convert_vector<da_int, da_int>(expected_labels);
        } else
            throw std::runtime_error("Weights must be uniform or distance");
    } else {
        throw std::runtime_error("n_neigh_knn must be 5 or 2");
    }
}

template <typename T> void get_targets(KNNParamType<T> &param) {
    if ((param.metric == "sqeuclidean") || (param.metric == "sqeuclidean_gemm")) {
        if (param.n_neigh_knn == 5) {
            if (param.weights == "uniform") {
                std::vector<T> expected_targets{1.6, 2.08, 2.2};
                param.expected_targets = convert_vector<T, T>(expected_targets);
            } else if (param.weights == "distance") {
                std::vector<T> expected_targets{1.6673041559396837, 1.3229994266524698,
                                                2.190592036491495};
                param.expected_targets = convert_vector<T, T>(expected_targets);
            } else
                throw std::runtime_error("Weights must be uniform or distance");
        } else if (param.n_neigh_knn == 3) {
            if (param.weights == "uniform") {
                std::vector<T> expected_targets{1.5666666666666667, 1.4000000000000001,
                                                3.0};
                param.expected_targets = convert_vector<T, T>(expected_targets);
            } else if (param.weights == "distance") {
                std::vector<T> expected_targets{1.6538258575197888, 0.9794520547945206,
                                                2.749071618037135};
                param.expected_targets = convert_vector<T, T>(expected_targets);
            } else
                throw std::runtime_error("Weights must be uniform or distance");
        } else {
            throw std::runtime_error("n_neigh_knn must be 5 or 3");
        }
    } else if ((param.metric == "euclidean") || (param.metric == "l2") ||
               (param.metric == "minkowski") || (param.metric == "euclidean_gemm")) {
        if (param.n_neigh_knn == 5) {
            if (param.weights == "uniform") {
                std::vector<T> expected_targets{1.6, 2.08, 2.2};
                param.expected_targets = convert_vector<T, T>(expected_targets);
            } else if (param.weights == "distance") {
                std::vector<T> expected_targets{1.6323653119203947, 1.7134711834322838,
                                                2.202877674352422};
                param.expected_targets = convert_vector<T, T>(expected_targets);
            } else
                throw std::runtime_error("Weights must be uniform or distance");
        } else if (param.n_neigh_knn == 3) {
            if (param.weights == "uniform") {
                std::vector<T> expected_targets{1.5666666666666667, 1.4000000000000001,
                                                3.0};
                param.expected_targets = convert_vector<T, T>(expected_targets);
            } else if (param.weights == "distance") {
                std::vector<T> expected_targets{1.6105221794760263, 1.1815297507328528,
                                                2.8785808475618806};
                param.expected_targets = convert_vector<T, T>(expected_targets);
            } else
                throw std::runtime_error("Weights must be uniform or distance");
        } else {
            throw std::runtime_error("n_neigh_knn must be 5 or 3");
        }
    } else {
        throw std::runtime_error("metric must be euclidean or sqeuclidean");
    }
}

template <typename T> void GetExampleData(std::vector<KNNParamType<T>> &params) {
    // Test with example data for all different parameter combinations.
    // Iterate through metrics, algorithms, and weights.
    for (auto const &m : MetricType) {
        for (auto const &a : AlgoType) {
            for (auto const &w : WeightsType) {
                for (auto const &nc : NumNeighConstructor) {
                    for (auto const &nk : NumNeighKNeighAPI) {
                        if (a == "kd tree" || a == "ball tree") {
                            if ((m != "sqeuclidean") && (m != "sqeuclidean_gemm")) {
                                // If the algorithm is tree, test for multiple leaf sizes.
                                for (auto const &ls : leaf_size) {

                                    KNNParamType<T> test(nc, nk, m, a, w, ls);
                                    test.n_features = 3;
                                    test.n_samples = 6;
                                    test.n_queries = 3;
                                    test.ldx_train = test.n_samples;
                                    test.ldx_test = test.n_queries;
                                    std::vector<T> X_train{-1., -2., -3., 1., 2., 3.,
                                                           -1., -1., -2., 3., 5., -1.,
                                                           2.,  3.,  -1., 1., 1., 2.};
                                    std::vector<da_int> y_train_class{1, 2, 0, 1, 2, 2};
                                    std::vector<T> y_train_regression{1.5, 2.2, 0.5,
                                                                      1,   2.8, 5.2};
                                    test.X_train = convert_vector<T, T>(X_train);
                                    test.y_train_class =
                                        convert_vector<da_int, da_int>(y_train_class);
                                    test.y_train_regression =
                                        convert_vector<T, T>(y_train_regression);

                                    std::vector<T> X_test{-2., -1., 2.,  2., -2.,
                                                          1.,  3.,  -1., -3.};
                                    test.X_test = convert_vector<T, T>(X_test);

                                    get_expected_kind_k_dist(test);
                                    get_proba(test);
                                    get_labels(test);
                                    get_targets(test);
                                    test.name = "metric=" + m + ", algo=" + a +
                                                ", weights=" + w +
                                                ", nc=" + std::to_string(nc) +
                                                ", nk=" + std::to_string(nk) +
                                                ", leaf_size=" + std::to_string(ls);

                                    params.push_back(test);
                                }
                            }
                        } else {
                            KNNParamType<T> test(nc, nk, m, a, w);
                            test.n_features = 3;
                            test.n_samples = 6;
                            test.n_queries = 3;
                            test.ldx_train = test.n_samples;
                            test.ldx_test = test.n_queries;
                            std::vector<T> X_train{-1., -2., -3., 1., 2., 3.,
                                                   -1., -1., -2., 3., 5., -1.,
                                                   2.,  3.,  -1., 1., 1., 2.};
                            std::vector<da_int> y_train_class{1, 2, 0, 1, 2, 2};
                            std::vector<T> y_train_regression{1.5, 2.2, 0.5, 1, 2.8, 5.2};
                            test.X_train = convert_vector<T, T>(X_train);
                            test.y_train_class =
                                convert_vector<da_int, da_int>(y_train_class);
                            test.y_train_regression =
                                convert_vector<T, T>(y_train_regression);

                            std::vector<T> X_test{-2., -1., 2.,  2., -2.,
                                                  1.,  3.,  -1., -3.};
                            test.X_test = convert_vector<T, T>(X_test);

                            get_expected_kind_k_dist(test);
                            get_proba(test);
                            get_labels(test);
                            get_targets(test);
                            test.name = "metric=" + m + ", algo=" + a + ", weights=" + w +
                                        ", nc=" + std::to_string(nc) +
                                        ", nk=" + std::to_string(nk);

                            params.push_back(test);
                        }
                    }
                }
            }
        }
    }
}

template <typename T> void GetRowMajorData(std::vector<KNNParamType<T>> &params) {
    // Test with row major data.

    KNNParamType<T> test(5, 4, "euclidean", "brute", "uniform");
    test.n_features = 3;
    test.n_samples = 6;
    test.n_queries = 3;
    test.ldx_train = test.n_samples;
    test.ldx_test = test.n_queries;
    std::vector<T> X_train{-1., -2., -3., 1., 2., 3.,  -1., -1., -2.,
                           3.,  5.,  -1., 2., 3., -1., 1.,  1.,  2.};
    std::vector<da_int> y_train_class{1, 2, 0, 1, 2, 2};
    std::vector<T> y_train_regression{1.5, 2.2, 0.5, 1, 2.8, 5.2};
    test.X_train = convert_vector<T, T>(X_train);
    test.y_train_class = convert_vector<da_int, da_int>(y_train_class);
    test.y_train_regression = convert_vector<T, T>(y_train_regression);

    std::vector<T> X_test{-2., -1., 2., 2., -2., 1., 3., -1., -3.};
    test.X_test = convert_vector<T, T>(X_test);

    get_expected_kind_k_dist(test);
    get_proba(test);
    get_labels(test);
    get_targets(test);

    // Now convert everything to row major order
    test.order = "row-major";
    datest_blas::imatcopy('T', test.n_samples, test.n_features, 1.0, test.X_train.data(),
                          test.n_samples, test.n_features);
    datest_blas::imatcopy('T', test.n_queries, test.n_features, 1.0, test.X_test.data(),
                          test.n_queries, test.n_features);
    datest_blas::imatcopy('T', test.n_queries, 3, 1.0, test.expected_proba.data(),
                          test.n_queries, 3);
    datest_blas::imatcopy('T', test.n_queries, 4, 1.0, test.expected_kdist.data(),
                          test.n_queries, 4);
#if defined(AOCLDA_ILP64)
    datest_blas::imatcopy('T', test.n_queries, 4, 1.0,
                          reinterpret_cast<double *>(test.expected_kind.data()),
                          test.n_queries, 4);
#else
    datest_blas::imatcopy('T', test.n_queries, 4, 1.0,
                          reinterpret_cast<float *>(test.expected_kind.data()),
                          test.n_queries, 4);
#endif
    test.ldx_train = test.n_features;
    test.ldx_test = test.n_features;
    test.name = "Row major test";

    params.push_back(test);
}

static std::list<std::tuple<da_int, da_int, da_int, da_int, std::string>>
    large_data_sets = {
        // samples, queries, features, n_classes, weights
        std::make_tuple(2063, 2070, 130, 4, "uniform"),
        std::make_tuple(2063, 2070, 130, 4, "distance"),
        std::make_tuple(2125, 2068, 20, 3, "uniform"),
        std::make_tuple(2125, 2068, 20, 3, "distance"),
        std::make_tuple(2092, 2071, 5, 3, "uniform"),
        std::make_tuple(2092, 2071, 5, 3, "distance")};

template <typename T>
void set_test_data_from_csv(std::string datafile, std::string datatype, da_int n_rows,
                            da_int n_cols, T *testdata) {
    da_datastore csv_store = nullptr;
    EXPECT_EQ(da_datastore_init(&csv_store), da_status_success);
    EXPECT_EQ(da_datastore_options_set_string(csv_store, "datatype", datatype.c_str()),
              da_status_success);
    EXPECT_EQ(da_data_load_from_csv(csv_store, datafile.c_str()), da_status_success);
    EXPECT_EQ(da_data_get_n_rows(csv_store, &n_rows), da_status_success);
    EXPECT_EQ(da_data_get_n_cols(csv_store, &n_cols), da_status_success);
    EXPECT_EQ(
        da_data_select_slice(csv_store, datafile.c_str(), 0, n_rows - 1, 0, n_cols - 1),
        da_status_success);
    EXPECT_EQ(da_data_extract_selection(csv_store, datafile.c_str(), column_major,
                                        testdata, n_rows),
              da_status_success);
    da_datastore_destroy(&csv_store);
}

template <typename T> void GetLargeData(std::vector<KNNParamType<T>> &params) {
    for (auto const &ds : large_data_sets) {

        da_int samples = std::get<0>(ds);
        da_int queries = std::get<1>(ds);
        da_int features = std::get<2>(ds);
        da_int n_classes = std::get<3>(ds);
        std::string weights = std::get<4>(ds);

        da_int n_neigh = 5;
        KNNParamType<T> test(n_neigh, n_neigh, "euclidean", "brute", weights, 30);
        test.n_samples = samples;
        test.n_queries = queries;
        test.n_features = features;
        test.ldx_train = samples;
        test.ldx_test = queries;
        test.tol = 500 * std::numeric_limits<T>::epsilon();

        test.name = "Large data set: samples=" + std::to_string(test.n_samples) +
                    ", features=" + std::to_string(test.n_features) +
                    ", queries=" + std::to_string(test.n_queries) +
                    ", weights=" + weights;

        da_int one = 1;
        // Get the training data
        std::string input_data_fname = std::string(DATA_DIR) + "/knn_data/X_train_" +
                                       std::to_string(samples) + "x" +
                                       std::to_string(features) + ".csv";

        test.X_train.resize(test.n_samples * test.n_features);
        set_test_data_from_csv<T>(input_data_fname, type_opt_name<T>(), test.n_samples,
                                  test.n_features, test.X_train.data());

        // Get the test data
        input_data_fname = std::string(DATA_DIR) + "/knn_data/X_test_" +
                           std::to_string(queries) + "x" + std::to_string(features) +
                           ".csv";
        test.X_test.resize(test.n_queries * test.n_features);
        set_test_data_from_csv<T>(input_data_fname, type_opt_name<T>(), test.n_queries,
                                  test.n_features, test.X_test.data());

        // Get the training targets for classification and regression
        input_data_fname = std::string(DATA_DIR) + "/knn_data/y_train_" +
                           std::to_string(samples) + "x" + std::to_string(features) +
                           ".csv";
        test.y_train_regression.resize(test.n_samples);
        set_test_data_from_csv<T>(input_data_fname, type_opt_name<T>(), test.n_samples,
                                  one, test.y_train_regression.data());

        test.y_train_class.resize(test.n_samples);
        set_test_data_from_csv<da_int>(input_data_fname, "integer", test.n_samples, one,
                                       test.y_train_class.data());

        // Extract expected distances
        input_data_fname = std::string(DATA_DIR) + "/knn_data/k_dist_" +
                           std::to_string(samples) + "x" + std::to_string(queries) + "x" +
                           std::to_string(features) + "_5.csv";

        test.expected_kdist.resize(test.n_queries * n_neigh);
        set_test_data_from_csv<T>(input_data_fname, type_opt_name<T>(), test.n_queries,
                                  n_neigh, test.expected_kdist.data());

        // Extract expected indices
        input_data_fname = std::string(DATA_DIR) + "/knn_data/k_ind_" +
                           std::to_string(samples) + "x" + std::to_string(queries) + "x" +
                           std::to_string(features) + "_5.csv";
        test.expected_kind.resize(test.n_queries * n_neigh);
        set_test_data_from_csv<da_int>(input_data_fname, "integer", test.n_queries,
                                       n_neigh, test.expected_kind.data());

        // Extract expected probabilities
        input_data_fname = std::string(DATA_DIR) + "/knn_data/proba_" +
                           std::to_string(queries) + "x" + std::to_string(features) +
                           "_" + weights + ".csv";
        test.expected_proba.resize(test.n_queries * n_classes);
        set_test_data_from_csv<T>(input_data_fname, type_opt_name<T>(), test.n_queries,
                                  n_classes, test.expected_proba.data());

        // Extract expected labels
        input_data_fname = std::string(DATA_DIR) + "/knn_data/predict_class_" +
                           std::to_string(queries) + "x" + std::to_string(features) +
                           "_" + weights + ".csv";
        test.expected_labels.resize(test.n_queries);
        set_test_data_from_csv<da_int>(input_data_fname, "integer", test.n_queries, one,
                                       test.expected_labels.data());

        // Extract expected targets
        input_data_fname = std::string(DATA_DIR) + "/knn_data/predict_reg_" +
                           std::to_string(queries) + "x" + std::to_string(features) +
                           "_" + weights + ".csv";
        test.expected_targets.resize(test.n_queries);
        set_test_data_from_csv<T>(input_data_fname, type_opt_name<T>(), test.n_queries,
                                  one, test.expected_targets.data());

        params.push_back(test);
    }
}

template <typename T> void GetKNNData(std::vector<KNNParamType<T>> &params) {
    GetExampleData(params);
    GetRowMajorData(params);
    GetLargeData(params);
}
