/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../datests_cblas.hh"
#include "../utest_utils.hpp"
#include "aoclda.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cmath>
#include <limits>
#include <list>
#include <random>
#include <string.h>
#include <unordered_set>

template <typename T> struct ANNParamType {

    std::string test_name;

    // csvname and target_recall only for recall tests
    std::string csvname;
    T target_recall = 0.2;

    // training data
    da_int n_samples = 1;
    da_int n_features = 1;
    std::vector<T> X_train;
    da_int ldx_train = 1;

    // testing data
    da_int n_queries = 1;
    std::vector<T> X_test;
    da_int ldx_test = 1;

    // parameters
    da_int nlist = 1;
    da_int nprobe = 1;
    da_int k = 1;
    da_int seed = 0;
    da_int kmeans_iter = 10;
    T train_fraction = 1.0;

    // algorithm specifics
    std::string metric = "sqeuclidean";
    std::string algorithm = "ivfflat";
    std::string order = "column-major";

    std::vector<T> expected_rinfo;
    std::vector<T> expected_centroids;

    // To test adding data to the same trained index twice, in test_functionality
    // I simply add the same data twice. This is convenient as we can anticipate the
    // expected results by using some helpers below
    std::vector<da_int> expected_kind, expected_kind_two_adds;
    std::vector<T> expected_kdist, expected_kdist_two_adds;

    // We check that adding data after clustering doesn't return empty lists.
    // This is unavoidable in cases like zero data, so we can turn the check
    // off if necessary.
    bool allow_empty_lists = false;

    ANNParamType(){};
    ANNParamType(da_int nlist, da_int nprobe, da_int k, std::string metric,
                 std::string algorithm, std::string order)
        : nlist(nlist), nprobe(nprobe), k(k), metric(metric), algorithm(algorithm),
          order(order){};
};

// Helper function - use kmeans handle to compute expected centroids
template <typename T> da_status GetExpectedResults(ANNParamType<T> &test) {
    da_status status;
    test.expected_rinfo.resize(4);
    test.expected_rinfo[0] = test.nlist;
    test.expected_rinfo[1] = test.n_samples;
    test.expected_rinfo[2] = test.n_features;

    da_handle kmeans_handle = nullptr;

    status = da_handle_init<T>(&kmeans_handle, da_handle_kmeans);
    status = da_options_set_string(kmeans_handle, "storage order", test.order.c_str());

    status = da_options_set_int(kmeans_handle, "n_clusters", test.nlist);
    status = da_options_set_int(kmeans_handle, "n_init", 1);
    status = da_options_set_int(kmeans_handle, "max_iter", test.kmeans_iter);
    status = da_options_set_int(kmeans_handle, "seed", test.seed);

    status = da_options_set(kmeans_handle, "convergence tolerance",
                            std::sqrt(std::numeric_limits<T>::epsilon()));

    status = da_kmeans_set_data(kmeans_handle, test.n_samples, test.n_features,
                                test.X_train.data(), test.ldx_train);
    status = da_kmeans_compute<T>(kmeans_handle);

    da_int centres_size = test.nlist * test.n_features;
    test.expected_centroids.resize(centres_size);
    status = da_handle_get_result(kmeans_handle, da_kmeans_cluster_centres, &centres_size,
                                  test.expected_centroids.data());

    da_int temp_size = 5;
    std::vector<T> temp_result(temp_size);
    status =
        da_handle_get_result(kmeans_handle, da_rinfo, &temp_size, temp_result.data());
    test.expected_rinfo[3] = temp_result[3];

    da_handle_destroy(&kmeans_handle);

    return status;
}

// Helper function to convert sqeuclidean tests to euclidean
template <typename T> void ConvertToEuclidean(ANNParamType<T> &test) {
    test.metric = "euclidean";
    test.test_name = test.test_name + " euclidean";
    for (auto &dist : test.expected_kdist) {
        dist = std::sqrt(dist);
    }

    for (auto &dist : test.expected_kdist_two_adds) {
        dist = std::sqrt(dist);
    }
}

// Helper to get what the expected indices are when we add the data in test twice
// and look for twice the amount of neighbors
template <typename T> void GetKindTwoAdds(ANNParamType<T> &test) {
    test.expected_kind_two_adds.resize(test.k * test.n_queries * 2);
    if (test.order == "column-major") {
        for (da_int i = 0; i < test.k; i++) {
            for (da_int q = 0; q < test.n_queries; q++) {
                test.expected_kind_two_adds[q + (2 * i) * test.n_queries] =
                    test.expected_kind[q + i * test.n_queries];
                test.expected_kind_two_adds[q + (2 * i + 1) * test.n_queries] =
                    test.expected_kind[q + i * test.n_queries] + test.n_samples;
            }
        }
    } else {
        for (da_int q = 0; q < test.n_queries; q++) {
            for (da_int i = 0; i < test.k; i++) {
                test.expected_kind_two_adds[(i + q * test.k) * 2] =
                    test.expected_kind[i + q * test.k];
                test.expected_kind_two_adds[(i + q * test.k) * 2 + 1] =
                    test.expected_kind[i + q * test.k] + test.n_samples;
            }
        }
    }
}

// Helper to get what the expected distances are when we add the data in test twice
// and look for twice the amount of neighbors
template <typename T> void GetKdistTwoAdds(ANNParamType<T> &test) {
    test.expected_kdist_two_adds.resize(test.k * test.n_queries * 2);
    if (test.order == "column-major") {
        for (da_int i = 0; i < test.k; i++) {
            for (da_int q = 0; q < test.n_queries; q++) {
                test.expected_kdist_two_adds[q + (2 * i) * test.n_queries] =
                    test.expected_kdist[q + i * test.n_queries];
                test.expected_kdist_two_adds[q + (2 * i + 1) * test.n_queries] =
                    test.expected_kdist[q + i * test.n_queries];
            }
        }
    } else {
        for (da_int q = 0; q < test.n_queries; q++) {
            for (da_int i = 0; i < test.k; i++) {
                test.expected_kdist_two_adds[(i + q * test.k) * 2] =
                    test.expected_kdist[i + q * test.k];
                test.expected_kdist_two_adds[(i + q * test.k) * 2 + 1] =
                    test.expected_kdist[i + q * test.k];
            }
        }
    }
}

// Helper to ensure pairs of indices are in sorted order when testing doubly added data
void SortDuplicateResults(da_order order, da_int *data, da_int m, da_int n) {
    // Transpose if column major
    if (order == column_major) {
#if defined(AOCLDA_ILP64)
        datest_blas::imatcopy('T', m, n, 1.0, reinterpret_cast<double *>(data), m, n);
#else
        datest_blas::imatcopy('T', m, n, 1.0, reinterpret_cast<float *>(data), m, n);
#endif
    }

    // Sort pairs of indices so we can test for equality between indices arrays
    for (int i = 0; i < m * n; i += 2) {
        if (data[i] > data[i + 1]) {
            std::swap(data[i], data[i + 1]);
        }
    }

    // Transpose back if needed
    if (order == column_major) {
#if defined(AOCLDA_ILP64)
        datest_blas::imatcopy('T', n, m, 1.0, reinterpret_cast<double *>(data), n, m);
#else
        datest_blas::imatcopy('T', n, m, 1.0, reinterpret_cast<float *>(data), n, m);
#endif
    }
}

// Some helpers needed to compute exact neighbors
template <typename T>
inline void smaller_values_and_indices(da_int n, T *D, da_int k, da_int *k_ind, T *k_dist,
                                       da_int init_index, bool init = true) {
    // Initialize the first k values of k_ind with init_index, init_index+1, ..., init_index+k-1
    if (init)
        std::iota(k_ind, k_ind + k, init_index);
    // Find the index of the maximum element and the corresponding maximum value.
    //da_int max_index = da_blas::cblas_iamax(k, k_dist, 1);
    da_int max_index = std::max_element(k_dist, k_dist + k) - k_dist;
    T max_val = k_dist[max_index];

    for (da_int i = k; i < n; i++) {
        // Check if an element of D is smaller than the maximum value. If it is,
        // we need to replace it's index in k_ind and replace the corresponding D[i] in k_dist.
        if (D[i] <= max_val) {
            // We know D[i] is smaller than Dmax. So we update k_ind[max_index] and D[max_index]
            // so that they hold the new value.
            k_ind[max_index] = i;
            k_dist[max_index] = D[i];
            // Now we need to find the new maximum so that we compare against that in the next iteration.
            //max_index = da_blas::cblas_iamax(k, k_dist, 1);
            max_index = std::max_element(k_dist, k_dist + k) - k_dist;
            max_val = k_dist[max_index];
        }
    }
}

// For most distances, we can simply use the knn api to get exact neighbors
template <typename T>
da_status compute_neighbors_knn(ANNParamType<T> param, da_int *true_neighbors,
                                const T *X_train, da_int n_samples_train,
                                da_int n_features, da_int ldx_train, const T *X_test,
                                da_int n_samples_test, da_int ldx_test) {
    da_status status;
    da_handle knn_handle = nullptr;

    status = da_handle_init<T>(&knn_handle, da_handle_nn);
    status = da_options_set_string(knn_handle, "storage order", param.order.c_str());

    status = da_options_set_int(knn_handle, "number of neighbors", param.k);
    status = da_options_set_string(knn_handle, "metric", param.metric.c_str());
    status = da_options_set_string(knn_handle, "algorithm", param.algorithm.c_str());
    status = da_nn_set_data(knn_handle, n_samples_train, n_features, X_train, ldx_train);
    if (status == da_status_success) {
        status = da_nn_kneighbors(knn_handle, n_samples_test, n_features, X_test,
                                  ldx_test, true_neighbors, nullptr, param.k, false);
    }

    da_handle_destroy(&knn_handle);
    return status;
}

// For inner product based search we need a separate helper
template <typename T>
da_status compute_neighbors_ip(ANNParamType<T> param, da_int *true_neighbors,
                               const T *X_train, da_int n_samples_train,
                               da_int n_features, da_int ldx_train, const T *X_test,
                               da_int n_samples_test, da_int ldx_test) {
    std::vector<T> distances(n_samples_test * n_samples_train, 0.0);
    std::vector<T> kdist(param.k);
    // Perform the gemm calls so the distances for each query are contiguous in memory
    if (param.order == "row-major") {
        datest_blas::cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_samples_test,
                                n_samples_train, n_features, -(T)1.0, X_test, ldx_test,
                                X_train, ldx_train, (T)0.0, distances.data(),
                                n_samples_train);
    } else {
        datest_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans, n_samples_train,
                                n_samples_test, n_features, -(T)1.0, X_train, ldx_train,
                                X_test, ldx_test, (T)0.0, distances.data(),
                                n_samples_train);
    }

    for (da_int i = 0; i < n_samples_test; i++) {
        std::copy_n(distances.data() + i * n_samples_train, param.k, kdist.data());
        smaller_values_and_indices(n_samples_train,
                                   distances.data() + i * n_samples_train, param.k,
                                   true_neighbors + i * param.k, kdist.data(), 0, true);
    }

    return da_status_success;
}

template <typename T>
T compute_recall(ANNParamType<T> param, da_int *approx_neighbors, const T *X_train,
                 da_int n_samples_train, da_int n_features, da_int ldx_train,
                 const T *X_test, da_int n_samples_test, da_int ldx_test) {

    std::vector<da_int> true_neighbors(n_samples_test * param.k);
    if (param.metric == "inner product") {
        compute_neighbors_ip(param, true_neighbors.data(), X_train, n_samples_train,
                             n_features, ldx_train, X_test, n_samples_test, ldx_test);
    } else {
        compute_neighbors_knn(param, true_neighbors.data(), X_train, n_samples_train,
                              n_features, ldx_train, X_test, n_samples_test, ldx_test);
    }

    // Recall = fraction of true nearest neighbors found by approximate search
    // If data is column major, do some transposes so the indices returned by each query are contiguous in memory
    // Don't need to tranpose true_neighbors for inner product as we handled that with the gemm formulation
    if (param.order == "column-major") {
        // transpose in place
#if defined(AOCLDA_ILP64)
        if (param.metric != "inner product") {
            datest_blas::imatcopy('T', n_samples_test, param.k, 1.0,
                                  reinterpret_cast<double *>(true_neighbors.data()),
                                  n_samples_test, param.k);
        }
        datest_blas::imatcopy('T', n_samples_test, param.k, 1.0,
                              reinterpret_cast<double *>(approx_neighbors),
                              n_samples_test, param.k);
#else
        if (param.metric != "inner product") {
            datest_blas::imatcopy('T', n_samples_test, param.k, 1.0,
                                  reinterpret_cast<float *>(true_neighbors.data()),
                                  n_samples_test, param.k);
        }
        datest_blas::imatcopy('T', n_samples_test, param.k, 1.0,
                              reinterpret_cast<float *>(approx_neighbors), n_samples_test,
                              param.k);
#endif
    }

    // now each row is contiguous and contains the indices for each query
    da_int count = 0;
    for (da_int i = 0; i < n_samples_test; i++) {
        // simple approach: create a set of true_neighbors and check each approx_neighbor (which != -1)
        std::unordered_set<da_int> true_set(true_neighbors.data() + i * param.k,
                                            true_neighbors.data() + (i + 1) * param.k);
        for (da_int j = 0; j < param.k; j++) {
            da_int val = approx_neighbors[j + i * param.k];
            if ((val != -1) && (true_set.count(val) == 1)) {
                count++;
            }
        }
    }
    return count / (T)(param.k * n_samples_test);
}

// For cases where many expected indices are valid (e.g., zero data),
// simply validate that indices are in range and unique per query
template <typename T>
bool validate_indices(const std::vector<da_int> &indices, da_int n_queries, da_int k,
                      da_int max_valid_index, da_order order) {
    // Check all indices are in valid range and unique per query
    for (da_int q = 0; q < n_queries; q++) {
        std::unordered_set<da_int> seen;
        for (da_int i = 0; i < k; i++) {
            da_int idx;
            if (order == column_major) {
                idx = indices[q + i * n_queries];
            } else {
                idx = indices[i + q * k];
            }

            // Check index is in valid range
            if (idx < 0 || idx >= max_valid_index) {
                return false;
            }

            // Check index is unique for this query
            if (seen.count(idx) > 0) {
                return false;
            }
            seen.insert(idx);
        }
    }
    return true;
}

// Functionality tests
template <typename T> void ZeroCol(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(2, 1, 2, "sqeuclidean", "ivfflat", "column-major");
    test.allow_empty_lists = true;
    test.test_name = "col zero";
    test.n_features = 4;
    test.n_samples = 8;
    test.n_queries = 2;
    test.ldx_train = test.n_samples;
    test.ldx_test = test.n_queries;
    std::vector<T> X_train{0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{1., 2., 3., 0.5, 1.5, 2.5, -0.5, -1};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<T> expected_kdist{12.5, 11.5, 12.5, 11.5};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void ZeroRow(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(2, 1, 2, "sqeuclidean", "ivfflat", "row-major");
    test.allow_empty_lists = true;
    test.test_name = "row zero";
    test.n_features = 4;
    test.n_samples = 8;
    test.n_queries = 2;
    test.ldx_train = test.n_features;
    test.ldx_test = test.n_features;
    std::vector<T> X_train{0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{1., 3., 1.5, -0.5, 2., 0.5, 2.5, -1};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<T> expected_kdist{12.5, 12.5, 11.5, 11.5};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void ZeroColIP(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(2, 1, 2, "inner product", "ivfflat", "column-major");
    test.allow_empty_lists = true;
    test.test_name = "col zero ip";
    test.n_features = 3;
    test.n_samples = 10;
    test.n_queries = 3;
    test.ldx_train = test.n_samples;
    test.ldx_test = test.n_queries;
    std::vector<T> X_train{
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    };
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{1., 2., 3., 0.5, 1.5, 2.5, -0.5, -1., 2.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<T> expected_kdist{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void ZeroRowIP(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(2, 1, 2, "inner product", "ivfflat", "row-major");
    test.allow_empty_lists = true;
    test.test_name = "row zero ip";
    test.n_features = 3;
    test.n_samples = 10;
    test.n_queries = 3;
    test.ldx_train = test.n_features;
    test.ldx_test = test.n_features;
    std::vector<T> X_train{
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    };
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{1, 0.5, -0.5, 2., 1.5, -1., 3., 2.5, 2.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<T> expected_kdist{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void OneByOneCol(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(1, 1, 1, "sqeuclidean", "ivfflat", "column-major");
    test.test_name = "col 1x1";
    test.n_features = 1;
    test.n_samples = 1;
    test.n_queries = 3;
    test.ldx_train = test.n_samples;
    test.ldx_test = test.n_queries;
    std::vector<T> X_train{0.5};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{-1., 0., 1.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{0, 0, 0};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{2.25, 0.25, 0.25};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void OneByOneRow(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(1, 1, 1, "sqeuclidean", "ivfflat", "row-major");
    test.test_name = "row 1x1";
    test.n_features = 1;
    test.n_samples = 1;
    test.n_queries = 3;
    test.ldx_train = test.n_features;
    test.ldx_test = test.n_features;
    std::vector<T> X_train{0.5};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{-1., 0., 1.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{0, 0, 0};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{2.25, 0.25, 0.25};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void OneByOneColIP(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(1, 1, 1, "inner product", "ivfflat", "column-major");
    test.test_name = "col 1x1 ip";
    test.n_features = 1;
    test.n_samples = 1;
    test.n_queries = 3;
    test.ldx_train = test.n_samples;
    test.ldx_test = test.n_queries;
    std::vector<T> X_train{0.5};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{-1., 0., 1.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{0, 0, 0};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{-0.5, 0., 0.5};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void OneByOneRowIP(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(1, 1, 1, "inner product", "ivfflat", "row-major");
    test.test_name = "row 1x1 ip";
    test.n_features = 1;
    test.n_samples = 1;
    test.n_queries = 3;
    test.ldx_train = test.n_features;
    test.ldx_test = test.n_features;
    std::vector<T> X_train{0.5};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{-1., 0., 1.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{0, 0, 0};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{-0.5, 0., 0.5};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void OneListColSqEuclidean(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(1, 1, 1, "sqeuclidean", "ivfflat", "column-major");
    test.test_name = "col one list";
    test.n_features = 3;
    test.n_samples = 6;
    test.n_queries = 3;
    test.ldx_train = test.n_samples;
    test.ldx_test = test.n_queries;
    std::vector<T> X_train{-1., -2., -3., 1., 2., 3.,  -1., -1., -2.,
                           3.,  5.,  -1., 2., 3., -1., 1.,  1.,  2.};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{-1.01, -3.01, 2., -1.01, -2., 5.05, 1.99, -1., 1.01};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{0, 2, 4};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{3e-4, 1e-4, 2.6e-3};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void OneListRowSqEuclidean(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(1, 1, 1, "sqeuclidean", "ivfflat", "row-major");
    test.test_name = "row one list";
    test.n_features = 3;
    test.n_samples = 6;
    test.n_queries = 3;
    test.ldx_train = test.n_features;
    test.ldx_test = test.n_features;
    // test non-deterministic path in single list case
    test.seed = -1;

    std::vector<T> X_train{-1., -1., 2., -2., -1., 3., -3., -2., -1.,
                           1.,  3.,  1., 2.,  5.,  1., 3.,  -1., 2.};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{-1.01, -1.01, 1.99, -3.01, -2., -1., 2., 5.05, 1.01};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{0, 2, 4};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{3e-4, 1e-4, 2.6e-3};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void ColSqEuclidean(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(3, 2, 3, "sqeuclidean", "auto", "column-major");
    test.test_name = "col";
    test.n_features = 3;
    test.n_samples = 9;
    test.n_queries = 3;

    test.ldx_train = test.n_samples;
    test.ldx_test = test.n_queries;

    std::vector<T> X_train{10.1, 11.1, 9.1, -20.2, -21.2, -19.2, 5.0,   5.0,   5.0,
                           10.2, 11.2, 9.7, -20.3, -21.3, -19.3, -10.1, -11.1, -9.1,
                           10.4, 11.4, 9.4, -20.3, -21.3, -18.3, 5.2,   11.2,  9.2};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{10.1, -20.2, 6.0, 11.2, -21.3, -10.1, 9.4, -19.3, 9.2};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{0, 3, 8, 2, 4, 7, 1, 5, 6};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{2.0, 2.0, 2.0, 3.25, 5.0, 6.0, 5.0, 6.0, 17.0};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void RowSqEuclidean(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(3, 2, 3, "sqeuclidean", "ivfflat", "row-major");
    test.test_name = "row";
    test.n_features = 3;
    test.n_samples = 9;
    test.n_queries = 3;

    test.ldx_train = test.n_features;
    test.ldx_test = test.n_features;

    std::vector<T> X_train{10.1,  10.2,  10.4,  11.1,  12.2,  11.4,  9.1,   9.2,   9.4,
                           -20.2, -20.3, -20.3, -21.2, -21.3, -21.3, -19.2, -19.3, -18.3,
                           5.0,   -10.1, 5.2,   5.0,   -11.1, 11.2,  5.0,   -9.1,  9.2};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{10.1, 11.2, 9.4, -20.2, -21.3, -19.3, 6.0, -10.1, 9.2};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{0, 2, 1, 3, 4, 5, 8, 7, 6};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{2.0, 5.0, 6.0, 2.0, 5.0, 6.0, 2.0, 6.0, 17.0};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void WideColSqEuclidean(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(2, 1, 1, "sqeuclidean", "ivfflat", "column-major");
    test.test_name = "wide col";
    test.n_features = 8;
    test.n_samples = 6;
    test.n_queries = 2;

    test.ldx_train = test.n_samples;
    test.ldx_test = test.n_queries;

    std::vector<T> X_train{3.0,  2.0,  4.0,  -3.0, -2.0, -4.0, 4.0,  3.0,  2.0,  -4.0,
                           -3.0, -2.0, 2.5,  1.0,  3.0,  -2.5, -1.0, -3.0, 3.0,  0.5,
                           4.0,  -3.0, -0.5, -4.0, 1.0,  1.5,  2.0,  -1.0, -1.5, -2.0,
                           1.5,  1.5,  1.0,  -1.5, -1.5, -1.0, 5.0,  3.5,  3.5,  -5.0,
                           -3.5, -3.5, 2.0,  5.0,  2.0,  -2.0, -5.0, -2.0};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{5.0, -5.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0,
                          3.0, -3.0, 2.0, -2.0, 4.5, -4.5, 3.0, -3.0};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{2, 5};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{8.0, 8.0};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void WideRowSqEuclidean(std::vector<ANNParamType<T>> &params) {
    // data in two distinct +ve and -ve clusters
    ANNParamType<T> test(2, 1, 1, "sqeuclidean", "ivfflat", "row-major");
    test.test_name = "wide row";
    test.n_features = 8;
    test.n_samples = 6;
    test.n_queries = 2;

    test.ldx_train = test.n_features;
    test.ldx_test = test.n_features;

    std::vector<T> X_train{3.0,  4.0,  2.5,  3.0,  1.0,  1.5,  5.0,  2.0,  2.0,  3.0,
                           1.0,  0.5,  1.5,  1.5,  3.5,  5.0,  4.0,  2.0,  3.0,  4.0,
                           2.0,  1.0,  3.5,  2.0,  -3.0, -4.0, -2.5, -3.0, -1.0, -1.5,
                           -5.0, -2.0, -2.0, -3.0, -1.0, -0.5, -1.5, -1.5, -3.5, -5.0,
                           -4.0, -2.0, -3.0, -4.0, -2.0, -1.0, -3.5, -2.0};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{5.0,  3.0,  4.0,  5.0,  3.0,  2.0,  4.5,  3.0,
                          -5.0, -3.0, -4.0, -5.0, -3.0, -2.0, -4.5, -3.0};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{2, 5};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{8.0, 8.0};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void PaddedColSqEuclidean(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(3, 2, 3, "sqeuclidean", "ivfflat", "column-major");
    test.test_name = "col Padded";
    test.n_features = 3;
    test.n_samples = 9;
    test.n_queries = 3;
    test.ldx_train = 11;
    test.ldx_test = 8;
    test.kmeans_iter = 25;

    std::vector<T> X_train{10.1,  11.1, 9.1,  -20.2, -21.2, -19.2, 5.0,   5.0,   5.0,
                           0.0,   0.0,  10.2, 11.2,  9.7,   -20.3, -21.3, -19.3, -10.1,
                           -11.1, -9.1, 0.0,  0.0,   10.4,  11.4,  9.4,   -20.3, -21.3,
                           -18.3, 5.2,  11.2, 9.2,   0.0,   0.0};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{
        10.1, -20.2, 6.0, 0.0, 0.0, 0.0,   0.0, 0.0, 11.2, -21.3, -10.1, 0.0,
        0.0,  0.0,   0.0, 0.0, 9.4, -19.3, 9.2, 0.0, 0.0,  0.0,   0.0,   0.0,
    };
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{0, 3, 8, 2, 4, 7, 1, 5, 6};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{2.0, 2.0, 2.0, 3.25, 5.0, 6.0, 5.0, 6.0, 17.0};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void PaddedRowSqEuclidean(std::vector<ANNParamType<T>> &params) {
    // a slightly larger simple problem, data has 3 well separated clusters
    ANNParamType<T> test(3, 2, 3, "sqeuclidean", "ivfflat", "row-major");
    test.test_name = "row Padded";
    test.n_features = 3;
    test.n_samples = 9;
    test.n_queries = 3;

    test.ldx_train = 5;
    test.ldx_test = 6;

    std::vector<T> X_train{
        10.1,  10.2,  10.4,  0.,    0.,    11.1,  12.2, 11.4,  0.,    0.,    9.1,   9.2,
        9.4,   0.,    0.,    -20.2, -20.3, -20.3, 0.,   0.,    -21.2, -21.3, -21.3, 0.,
        0.,    -19.2, -19.3, -18.3, 0.,    0.,    5.0,  -10.1, 5.2,   0.,    0.,    5.0,
        -11.1, 11.2,  0.,    0.,    5.0,   -9.1,  9.2,  0.,    0.,
    };
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{
        10.1, 11.2, 9.4, 0.,  0.,    0.,  -20.2, -21.3, -19.3,
        0.,   0.,   0.,  6.0, -10.1, 9.2, 0.,    0.,    0.,
    };
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{0, 2, 1, 3, 4, 5, 8, 7, 6};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{2.0, 5.0, 6.0, 2.0, 5.0, 6.0, 2.0, 6.0, 17.0};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void OneListColIP(std::vector<ANNParamType<T>> &params) {
    // Data here is purposefully chosen so the
    // expected_kind for ip and euclidean are different.
    // Add large vectors in upper right quadrant, and small vectors in lower left.
    // We will search with small vectors in upper right.
    ANNParamType<T> test(1, 1, 1, "inner product", "ivfflat", "column-major");
    test.test_name = "col ip one list";
    test.n_features = 2;
    test.n_samples = 8;
    test.n_queries = 3;
    test.ldx_train = test.n_samples;
    test.ldx_test = test.n_queries;
    std::vector<T> X_train{15., 13., 10., 9.,  -1., -2., -3., -4.,
                           10., 11., 12., 13., -1., -1., -1., -3.};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{1., 0.01, -1., 1., 2., -2.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{0, 3, 7};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{25., 26.09, 10.};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void OneListRowIP(std::vector<ANNParamType<T>> &params) {
    // Data here is purposefully chosen so the
    // expected_kind for ip and euclidean are different.
    // Add large vectors in upper right quadrant, and small vectors in lower left.
    // We will search with small vectors in upper right.
    ANNParamType<T> test(1, 1, 1, "inner product", "ivfflat", "row-major");
    test.test_name = "row ip one list";
    test.n_features = 2;
    test.n_samples = 8;
    test.n_queries = 3;
    test.ldx_train = test.n_features;
    test.ldx_test = test.n_features;
    std::vector<T> X_train{15., 10., 13., 11., 10., 12., 9.,  13.,
                           -1., -1., -2., -1., -3., -1., -4., -3.};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{1., 1, 0.01, 2., -1., -2.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{0, 3, 7};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{25., 26.09, 10.};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void ColIP(std::vector<ANNParamType<T>> &params) {
    // Divide train vectors by octant. 3 each in:
    // (+, +, +), (-, -, -), (-, +, +), (+, -, +)
    ANNParamType<T> test(4, 2, 3, "inner product", "ivfflat", "column-major");
    test.test_name = "col ip";
    test.n_features = 3;
    test.n_samples = 12;
    test.n_queries = 3;
    test.ldx_train = test.n_samples;
    test.ldx_test = test.n_queries;
    std::vector<T> X_train{1., 2., 3,   -1.,  -2., -3,   -2., -3., -2., 2.,  3.,  1.,
                           1., 5., 2.,  -1.,  -3., -0.5, 1.,  1.5, 3.,  -1., -2., -4.,
                           1., 4,  2.5, -1.5, -2., -1,   2.,  3.,  1.5, 1.,  1.5, 1.};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{1., -2., 1., 1., -2., -1., 1., -2., 3.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{1, 4, 10, 2, 5, 1, 0, 3, 2};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{11., 14., 9.5, 7.5, 9., 9., 3., 7, 8.5};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void RowIP(std::vector<ANNParamType<T>> &params) {
    // Divide train vectors by octant. 3 each in:
    // (+, +, +), (-, -, -), (-, +, +), (+, -, +)
    ANNParamType<T> test(4, 2, 3, "inner product", "ivfflat", "row-major");
    test.test_name = "row ip";
    test.n_features = 3;
    test.n_samples = 12;
    test.n_queries = 3;
    test.kmeans_iter = 20;
    test.ldx_train = test.n_features;
    test.ldx_test = test.n_features;
    std::vector<T> X_train{1.,  1.,  1.,  2.,  5.,   4.,  3.,  2.,  2.5, -1., -1., -1.5,
                           -2., -3., -2., -3., -0.5, -1., -2., 1.,  2.,  -3., 1.5, 3.,
                           -2., 3.,  1.5, 2.,  -1.,  1.,  3.,  -2., 1.5, 1.,  -4., 1.};
    test.X_train = convert_vector<T, T>(X_train);

    std::vector<T> X_test{1., 1., 1., -2., -2., -2., 1., -1., 3.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{1, 2, 0, 4, 5, 3, 10, 1, 2};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{11., 7.5, 3., 14., 9., 7., 9.5, 9., 8.5};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void WideColIP(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(2, 1, 1, "inner product", "ivfflat", "column-major");
    test.test_name = "wide col ip";
    test.n_features = 8;
    test.n_samples = 6;
    test.n_queries = 2;

    test.ldx_train = test.n_samples;
    test.ldx_test = test.n_queries;

    std::vector<T> X_train{3.0,  2.0,  4.0,  -3.0, -2.0, -4.0, 4.0,  3.0,  2.0,  -4.0,
                           -3.0, -2.0, 2.5,  1.0,  3.0,  -2.5, -1.0, -3.0, 3.0,  0.5,
                           4.0,  -3.0, -0.5, -4.0, 1.0,  1.5,  2.0,  -1.0, -1.5, -2.0,
                           1.5,  1.5,  1.0,  -1.5, -1.5, -1.0, 5.0,  3.5,  3.5,  -5.0,
                           -3.5, -3.5, 2.0,  5.0,  2.0,  -2.0, -5.0, -2.0};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{5.0, -5.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0,
                          3.0, -3.0, 2.0, -2.0, 4.5, -4.5, 3.0, -3.0};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{2, 5};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{87.75, 87.75};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void WideRowIP(std::vector<ANNParamType<T>> &params) {
    // data in two distinct +ve and -ve clusters
    ANNParamType<T> test(2, 1, 1, "inner product", "ivfflat", "row-major");
    test.test_name = "wide row ip";
    test.n_features = 8;
    test.n_samples = 6;
    test.n_queries = 2;

    test.ldx_train = test.n_features;
    test.ldx_test = test.n_features;

    std::vector<T> X_train{3.0,  4.0,  2.5,  3.0,  1.0,  1.5,  5.0,  2.0,  2.0,  3.0,
                           1.0,  0.5,  1.5,  1.5,  3.5,  5.0,  4.0,  2.0,  3.0,  4.0,
                           2.0,  1.0,  3.5,  2.0,  -3.0, -4.0, -2.5, -3.0, -1.0, -1.5,
                           -5.0, -2.0, -2.0, -3.0, -1.0, -0.5, -1.5, -1.5, -3.5, -5.0,
                           -4.0, -2.0, -3.0, -4.0, -2.0, -1.0, -3.5, -2.0};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{5.0,  3.0,  4.0,  5.0,  3.0,  2.0,  4.5,  3.0,
                          -5.0, -3.0, -4.0, -5.0, -3.0, -2.0, -4.5, -3.0};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{2, 5};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{87.75, 87.75};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void PaddedColIP(std::vector<ANNParamType<T>> &params) {
    // Divide train vectors by octant. 3 each in:
    // (+, +, +), (-, -, -), (-, +, +), (+, -, +)
    ANNParamType<T> test(4, 2, 3, "inner product", "ivfflat", "column-major");
    test.test_name = "padded col ip";
    test.n_features = 3;
    test.n_samples = 12;
    test.n_queries = 3;
    test.ldx_train = 16;
    test.ldx_test = 6;

    std::vector<T> X_train{1.,  2.,  3,   -1., -2., -3, -2., -3., -2., 2.,   3.,  1.,
                           0.,  0.,  0.,  0.,  1.,  5., 2.,  -1., -3., -0.5, 1.,  1.5,
                           3.,  -1., -2., -4., 0.,  0., 0.,  0.,  1.,  4,    2.5, -1.5,
                           -2., -1,  2.,  3.,  1.5, 1., 1.5, 1.,  0.,  0.,   0.,  0.};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{1., -2., 1., 0., 0.,  0., 1., -2., -1.,
                          0., 0.,  0., 1., -2., 3., 0., 0.,  0.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{1, 4, 10, 2, 5, 1, 0, 3, 2};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{11., 14., 9.5, 7.5, 9., 9., 3., 7, 8.5};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void PaddedRowIP(std::vector<ANNParamType<T>> &params) {
    // Divide train vectors by octant. 3 each in:
    // (+, +, +), (-, -, -), (-, +, +), (+, -, +)
    ANNParamType<T> test(4, 2, 3, "inner product", "ivfflat", "row-major");
    test.test_name = "padded row ip";
    test.n_features = 3;
    test.n_samples = 12;
    test.n_queries = 3;
    test.ldx_train = 5;
    test.ldx_test = 6;

    std::vector<T> X_train{
        1.,  1.,  1.,   0., 0., 2.,  5.,  4.,  0., 0., 3.,  2.,   2.5, 0., 0.,
        -1., -1., -1.5, 0., 0., -2., -3., -2., 0., 0., -3., -0.5, -1., 0., 0.,
        -2., 1.,  2.,   0., 0., -3., 1.5, 3.,  0., 0., -2., 3.,   1.5, 0., 0.,
        2.,  -1., 1.,   0., 0., 3.,  -2., 1.5, 0., 0., 1.,  -4.,  1.,  0., 0.,
    };
    test.X_train = convert_vector<T, T>(X_train);

    std::vector<T> X_test{1., 1., 1., 0., 0.,  0., -2., -2., -2.,
                          0., 0., 0., 1., -1., 3., 0.,  0.,  0.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{1, 2, 0, 4, 5, 3, 10, 1, 2};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{11., 7.5, 3., 14., 9., 7., 9.5, 9., 8.5};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void SubsampleCol(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(2, 1, 1, "sqeuclidean", "ivfflat", "column-major");
    test.test_name = "subsample col";
    test.n_features = 3;
    test.n_samples = 12;
    test.n_queries = 1;
    test.ldx_train = 12;
    test.ldx_test = 1;

    test.train_fraction = 0.51;

    std::vector<T> X_train{1.0,  1.1,  1.2,  1.3,  1.4,  1.5,  -1.0, -1.1, -1.2,
                           -1.3, -1.4, -1.5, 1.0,  1.1,  1.2,  1.3,  1.4,  1.5,
                           -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, 1.0,  1.1,  1.2,
                           1.3,  1.4,  1.5,  -1.0, -1.1, -1.2, -1.3, -1.4, -1.5};
    test.X_train = convert_vector<T, T>(X_train);

    std::vector<T> X_test{1.5, 1.5, 1.5};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{5};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{0};
    test.expected_kdist = expected_kdist;

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

// Not testing get_result for subsampled data as the kmeans api doesn't
// allow to subsample so results won't match with current setup
template <typename T> void SubsampleRow(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(2, 1, 1, "sqeuclidean", "ivfflat", "row-major");
    test.test_name = "subsample row";
    test.n_features = 3;
    test.n_samples = 12;
    test.n_queries = 1;
    test.ldx_train = 3;
    test.ldx_test = 3;

    test.train_fraction = 0.51;

    std::vector<T> X_train{1.0,  1.0,  1.0,  1.1,  1.1,  1.1,  1.2,  1.2,  1.2,
                           1.3,  1.3,  1.3,  1.4,  1.4,  1.4,  1.5,  1.5,  1.5,
                           -1.0, -1.0, -1.0, -1.1, -1.1, -1.1, -1.2, -1.2, -1.2,
                           -1.3, -1.3, -1.3, -1.4, -1.4, -1.4, -1.5, -1.5, -1.5};
    test.X_train = convert_vector<T, T>(X_train);

    std::vector<T> X_test{1.5, 1.5, 1.5};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{5};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{0};
    test.expected_kdist = expected_kdist;

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void SubsamplePaddedCol(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(2, 1, 1, "sqeuclidean", "ivfflat", "column-major");
    test.test_name = "subsample padded col";
    test.n_features = 3;
    test.n_samples = 12;
    test.n_queries = 1;
    test.ldx_train = 14;
    test.ldx_test = 1;

    test.train_fraction = 0.51;

    std::vector<T> X_train{
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, 0.0, 0.0,
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, 0.0, 0.0,
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, 0.0, 0.0};
    test.X_train = convert_vector<T, T>(X_train);

    std::vector<T> X_test{1.5, 1.5, 1.5};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{5};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{0};
    test.expected_kdist = expected_kdist;

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void SubsamplePaddedRow(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(2, 1, 1, "sqeuclidean", "ivfflat", "row-major");
    test.test_name = "subsample padded row";
    test.n_features = 3;
    test.n_samples = 12;
    test.n_queries = 1;
    test.ldx_train = 5;
    test.ldx_test = 3;

    test.train_fraction = 0.51;

    std::vector<T> X_train{1.0,  1.0,  1.0,  0.0, 0.0, 1.1,  1.1,  1.1,  0.0, 0.0,
                           1.2,  1.2,  1.2,  0.0, 0.0, 1.3,  1.3,  1.3,  0.0, 0.0,
                           1.4,  1.4,  1.4,  0.0, 0.0, 1.5,  1.5,  1.5,  0.0, 0.0,
                           -1.0, -1.0, -1.0, 0.0, 0.0, -1.1, -1.1, -1.1, 0.0, 0.0,
                           -1.2, -1.2, -1.2, 0.0, 0.0, -1.3, -1.3, -1.3, 0.0, 0.0,
                           -1.4, -1.4, -1.4, 0.0, 0.0, -1.5, -1.5, -1.5, 0.0, 0.0};
    test.X_train = convert_vector<T, T>(X_train);

    std::vector<T> X_test{1.5, 1.5, 1.5};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{5};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{0};
    test.expected_kdist = expected_kdist;

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void ZeroColCosine(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(2, 1, 2, "cosine", "ivfflat", "column-major");
    test.allow_empty_lists = true;
    test.test_name = "col zero cos";
    test.n_features = 3;
    test.n_samples = 10;
    test.n_queries = 3;
    test.ldx_train = test.n_samples;
    test.ldx_test = test.n_queries;
    std::vector<T> X_train{
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    };
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{1., 2., 3., 0.5, 1.5, 2.5, -0.5, -1., 2.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<T> expected_kdist{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void ZeroRowCosine(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(2, 1, 2, "cosine", "ivfflat", "row-major");
    test.allow_empty_lists = true;
    test.test_name = "row zero cos";
    test.n_features = 3;
    test.n_samples = 10;
    test.n_queries = 3;
    test.ldx_train = test.n_features;
    test.ldx_test = test.n_features;
    std::vector<T> X_train{
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    };
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{1, 0.5, -0.5, 2., 1.5, -1., 3., 2.5, 2.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<T> expected_kdist{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void OneByOneColCosine(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(1, 1, 1, "cosine", "ivfflat", "column-major");
    test.test_name = "col 1x1 cos";
    test.n_features = 1;
    test.n_samples = 1;
    test.n_queries = 3;
    test.ldx_train = test.n_samples;
    test.ldx_test = test.n_queries;
    std::vector<T> X_train{0.5};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{-1., 0., 1.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{0, 0, 0};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{2., 1., 0.};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void OneByOneRowCosine(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(1, 1, 1, "cosine", "ivfflat", "row-major");
    test.test_name = "row 1x1 cos";
    test.n_features = 1;
    test.n_samples = 1;
    test.n_queries = 3;
    test.ldx_train = test.n_features;
    test.ldx_test = test.n_features;
    std::vector<T> X_train{0.5};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{-1., 0., 1.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{0, 0, 0};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{2., 1., 0.};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void OneListColCosine(std::vector<ANNParamType<T>> &params) {
    // Data here is purposefully chosen so the
    // expected_kind for ip and euclidean are different.
    // Add large vectors in upper right quadrant, and small vectors in lower left.
    // We will search with small vectors in upper right.
    ANNParamType<T> test(1, 1, 1, "cosine", "ivfflat", "column-major");
    test.test_name = "col cos one list";
    test.n_features = 2;
    test.n_samples = 8;
    test.n_queries = 3;
    test.ldx_train = test.n_samples;
    test.ldx_test = test.n_queries;
    std::vector<T> X_train{15., 13., 10., 9.,  -1., -2., -3., -4.,
                           10., 11., 12., 13., -1., -1., -1., -3.};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{1., 0.01, -1., 1., 2., -2.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{1, 3, 4};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{0.003454241755120546, 0.17497207124672476,
                                  0.05131670194948634};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void OneListRowCosine(std::vector<ANNParamType<T>> &params) {
    // Data here is purposefully chosen so the
    // expected_kind for ip and euclidean are different.
    // Add large vectors in upper right quadrant, and small vectors in lower left.
    // We will search with small vectors in upper right.
    ANNParamType<T> test(1, 1, 1, "cosine", "ivfflat", "row-major");
    test.test_name = "row cos one list";
    test.n_features = 2;
    test.n_samples = 8;
    test.n_queries = 3;
    test.ldx_train = test.n_features;
    test.ldx_test = test.n_features;
    std::vector<T> X_train{15., 10., 13., 11., 10., 12., 9.,  13.,
                           -1., -1., -2., -1., -3., -1., -4., -3.};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{1., 1, 0.01, 2., -1., -2.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{1, 3, 4};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{0.003454241755120546, 0.17497207124672476,
                                  0.05131670194948634};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void ColCosine(std::vector<ANNParamType<T>> &params) {
    // Divide train vectors by octant. 3 each in:
    // (+, +, +), (-, -, -), (-, +, +), (+, -, +)
    ANNParamType<T> test(4, 3, 3, "cosine", "ivfflat", "column-major");
    test.test_name = "col cos";
    test.n_features = 3;
    test.n_samples = 12;
    test.n_queries = 3;
    test.seed = 8;
    test.ldx_train = test.n_samples;
    test.ldx_test = test.n_queries;
    std::vector<T> X_train{1., 2., 3,   -1.,  -2., -3,   -2., -3., -2., 2.,  3.,  1.,
                           1., 5., 2.,  -1.,  -4., -0.5, 1.,  1.5, 3.,  -1., -2., -4.,
                           1., 4,  2.5, -1.5, -2., -1,   2.,  3.,  1.5, 1.,  1.5, 1.};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{1., -2., 1., 1., -2., -1., 1., -2., 3.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{0, 3, 9, 2, 4, 10, 1, 5, 2};
    test.expected_kind = expected_kind;
    std::vector<T> expected_kdist{0.0,
                                  0.01980394118039297,
                                  0.2614510541240035,
                                  0.013072457560346473,
                                  0.05719095841793642,
                                  0.2665131353419262,
                                  0.053270737593742346,
                                  0.1884973287993108,
                                  0.4158730871675579};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void RowCosine(std::vector<ANNParamType<T>> &params) {
    // Divide train vectors by octant. 3 each in:
    // (+, +, +), (-, -, -), (-, +, +), (+, -, +)
    ANNParamType<T> test(4, 3, 3, "cosine", "ivfflat", "row-major");
    test.test_name = "row cos";
    test.n_features = 3;
    test.n_samples = 12;
    test.n_queries = 3;
    test.seed = 8;
    test.kmeans_iter = 20;
    test.ldx_train = test.n_features;
    test.ldx_test = test.n_features;
    std::vector<T> X_train{1.,  1.,  1.,  2.,  5.,   4.,  3.,  2.,  2.5, -1., -1., -1.5,
                           -2., -4., -2., -3., -0.5, -1., -2., 1.,  2.,  -3., 1.5, 3.,
                           -2., 3.,  1.5, 2.,  -1.,  1.,  3.,  -2., 1.5, 1.,  -4., 1.};
    test.X_train = convert_vector<T, T>(X_train);

    std::vector<T> X_test{1., 1., 1., -2., -2., -2., 1., -1., 3.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{0, 2, 1, 3, 4, 5, 9, 10, 2};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{
        0.0,
        0.013072457560346473,
        0.053270737593742346,
        0.01980394118039297,
        0.05719095841793642,
        0.1884973287993108,
        0.2614510541240035,
        0.2665131353419262,
        0.4158730871675579,
    };
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void WideColCosine(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(2, 1, 1, "cosine", "ivfflat", "column-major");
    test.test_name = "wide col cos";
    test.n_features = 8;
    test.n_samples = 6;
    test.n_queries = 2;

    test.ldx_train = test.n_samples;
    test.ldx_test = test.n_queries;

    std::vector<T> X_train{3.0,  2.0,  4.0,  -3.0, -2.0, -4.0, 4.0,  3.0,  2.0,  -4.0,
                           -3.0, -2.0, 2.5,  1.0,  3.0,  -2.5, -1.0, -3.0, 3.0,  0.5,
                           4.0,  -3.0, -0.5, -4.0, 1.0,  1.5,  2.0,  -1.0, -1.5, -2.0,
                           1.5,  1.5,  1.0,  -1.5, -1.5, -1.0, 5.0,  3.5,  3.5,  -5.0,
                           -3.5, -3.5, 2.0,  5.0,  2.0,  -2.0, -5.0, -2.0};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{5.0, -5.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0,
                          3.0, -3.0, 2.0, -2.0, 4.5, -4.5, 3.0, -3.0};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{2, 5};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{0.004370495217352333, 0.004370495217352333};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void WideRowCosine(std::vector<ANNParamType<T>> &params) {
    // data in two distinct +ve and -ve clusters
    ANNParamType<T> test(2, 1, 1, "cosine", "ivfflat", "row-major");
    test.test_name = "wide row cosine";
    test.n_features = 8;
    test.n_samples = 6;
    test.n_queries = 2;

    test.ldx_train = test.n_features;
    test.ldx_test = test.n_features;

    std::vector<T> X_train{3.0,  4.0,  2.5,  3.0,  1.0,  1.5,  5.0,  2.0,  2.0,  3.0,
                           1.0,  0.5,  1.5,  1.5,  3.5,  5.0,  4.0,  2.0,  3.0,  4.0,
                           2.0,  1.0,  3.5,  2.0,  -3.0, -4.0, -2.5, -3.0, -1.0, -1.5,
                           -5.0, -2.0, -2.0, -3.0, -1.0, -0.5, -1.5, -1.5, -3.5, -5.0,
                           -4.0, -2.0, -3.0, -4.0, -2.0, -1.0, -3.5, -2.0};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{5.0,  3.0,  4.0,  5.0,  3.0,  2.0,  4.5,  3.0,
                          -5.0, -3.0, -4.0, -5.0, -3.0, -2.0, -4.5, -3.0};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{2, 5};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{0.004370495217352333, 0.004370495217352333};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void PaddedColCosine(std::vector<ANNParamType<T>> &params) {
    // Divide train vectors by octant. 3 each in:
    // (+, +, +), (-, -, -), (-, +, +), (+, -, +)
    ANNParamType<T> test(4, 3, 3, "cosine", "ivfflat", "column-major");
    test.test_name = "padded col cosine";
    test.n_features = 3;
    test.n_samples = 12;
    test.n_queries = 3;
    test.ldx_train = 16;
    test.ldx_test = 6;
    test.seed = 8;
    test.kmeans_iter = 20;
    std::vector<T> X_train{1.,  2.,  3,   -1., -2., -3, -2., -3., -2., 2.,   3.,  1.,
                           0.,  0.,  0.,  0.,  1.,  5., 2.,  -1., -4., -0.5, 1.,  1.5,
                           3.,  -1., -2., -4., 0.,  0., 0.,  0.,  1.,  4,    2.5, -1.5,
                           -2., -1,  2.,  3.,  1.5, 1., 1.5, 1.,  0.,  0.,   0.,  0.};
    test.X_train = convert_vector<T, T>(X_train);
    std::vector<T> X_test{1., -2., 1., 0., 0.,  0., 1., -2., -1.,
                          0., 0.,  0., 1., -2., 3., 0., 0.,  0.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{0, 3, 9, 2, 4, 10, 1, 5, 2};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{0.0,
                                  0.01980394118039297,
                                  0.2614510541240035,
                                  0.013072457560346473,
                                  0.05719095841793642,
                                  0.2665131353419262,
                                  0.053270737593742346,
                                  0.1884973287993108,
                                  0.4158730871675579};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void PaddedRowCosine(std::vector<ANNParamType<T>> &params) {
    // Divide train vectors by octant. 3 each in:
    // (+, +, +), (-, -, -), (-, +, +), (+, -, +)
    ANNParamType<T> test(4, 3, 3, "cosine", "ivfflat", "row-major");
    test.test_name = "padded row cosine";
    test.n_features = 3;
    test.n_samples = 12;
    test.n_queries = 3;
    test.ldx_train = 5;
    test.ldx_test = 6;
    test.seed = 8;
    test.kmeans_iter = 20;
    std::vector<T> X_train{
        1.,  1.,  1.,   0., 0., 2.,  5.,  4.,  0., 0., 3.,  2.,   2.5, 0., 0.,
        -1., -1., -1.5, 0., 0., -2., -4., -2., 0., 0., -3., -0.5, -1., 0., 0.,
        -2., 1.,  2.,   0., 0., -3., 1.5, 3.,  0., 0., -2., 3.,   1.5, 0., 0.,
        2.,  -1., 1.,   0., 0., 3.,  -2., 1.5, 0., 0., 1.,  -4.,  1.,  0., 0.,
    };
    test.X_train = convert_vector<T, T>(X_train);

    std::vector<T> X_test{1., 1., 1., 0., 0.,  0., -2., -2., -2.,
                          0., 0., 0., 1., -1., 3., 0.,  0.,  0.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{0, 2, 1, 3, 4, 5, 9, 10, 2};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{
        0.0,
        0.013072457560346473,
        0.053270737593742346,
        0.01980394118039297,
        0.05719095841793642,
        0.1884973287993108,
        0.2614510541240035,
        0.2665131353419262,
        0.4158730871675579,
    };
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);

    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T>
void SubsamplePaddedColCosine(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(2, 1, 1, "cosine", "ivfflat", "column-major");
    test.test_name = "subsample padded col cos";
    test.n_features = 3;
    test.n_samples = 12;
    test.n_queries = 1;
    test.ldx_train = 14;
    test.ldx_test = 1;

    test.train_fraction = 0.51;

    std::vector<T> X_train{
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, 0.0, 0.0,
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, 0.0, 0.0,
        1.3, 1.3, 1.3, 1.2, 1.2, 1.5, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, 0.0, 0.0};
    test.X_train = convert_vector<T, T>(X_train);

    std::vector<T> X_test{0.6, 0.6, 0.6};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{5};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{0};
    test.expected_kdist = expected_kdist;

    params.push_back(test);
}

template <typename T>
void SubsamplePaddedRowCosine(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(2, 1, 1, "cosine", "ivfflat", "row-major");
    test.test_name = "subsample padded row cos";
    test.n_features = 3;
    test.n_samples = 12;
    test.n_queries = 1;
    test.ldx_train = 5;
    test.ldx_test = 3;

    test.train_fraction = 0.51;

    std::vector<T> X_train{1.0,  1.0,  1.3,  0.0, 0.0, 1.1,  1.1,  1.3,  0.0, 0.0,
                           1.2,  1.2,  1.3,  0.0, 0.0, 1.3,  1.3,  1.4,  0.0, 0.0,
                           1.4,  1.4,  1.3,  0.0, 0.0, 1.5,  1.5,  1.5,  0.0, 0.0,
                           -1.0, -1.0, -1.0, 0.0, 0.0, -1.1, -1.1, -1.1, 0.0, 0.0,
                           -1.2, -1.2, -1.2, 0.0, 0.0, -1.3, -1.3, -1.3, 0.0, 0.0,
                           -1.4, -1.4, -1.4, 0.0, 0.0, -1.5, -1.5, -1.5, 0.0, 0.0};
    test.X_train = convert_vector<T, T>(X_train);

    std::vector<T> X_test{0.5, 0.5, 0.5};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{5};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{0};
    test.expected_kdist = expected_kdist;

    params.push_back(test);
}

// Verify large outliers get handled correctly in the inner product case
template <typename T> void OutliersColIP(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(2, 1, 1, "inner product", "ivfflat", "column-major");
    test.test_name = "outliers col ip";
    test.n_features = 2;
    test.n_samples = 12;
    test.n_queries = 2;
    test.ldx_train = test.n_samples;
    test.ldx_test = test.n_queries;

    std::vector<T> X_train{1., 2., 1.,  3., 2., 1., 2., 3., 1., 2., 80., 90.,
                           8., 7., 10., 6., 8., 7., 9., 8., 7., 9., 85., 88.};
    test.X_train = convert_vector<T, T>(X_train);

    std::vector<T> X_test{1., 50., 10., 55.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{2, 11};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{101, 9340};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);
    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void OutliersRowIP(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(2, 1, 1, "inner product", "ivfflat", "row-major");
    test.test_name = "outliers row ip";
    test.n_features = 2;
    test.n_samples = 12;
    test.n_queries = 2;
    test.ldx_train = test.n_features;
    test.ldx_test = test.n_features;

    std::vector<T> X_train{1., 8., 2., 7., 1., 10., 3., 6., 2.,  8.,  1.,  7.,
                           2., 9., 3., 8., 1., 7.,  2., 9., 80., 85., 90., 88.};
    test.X_train = convert_vector<T, T>(X_train);

    std::vector<T> X_test{1., 10., 50., 55.};
    test.X_test = convert_vector<T, T>(X_test);

    std::vector<da_int> expected_kind{2, 11};
    test.expected_kind = expected_kind;

    std::vector<T> expected_kdist{101, 9340};
    test.expected_kdist = expected_kdist;

    GetExpectedResults(test);
    GetKindTwoAdds(test);
    GetKdistTwoAdds(test);

    params.push_back(test);
}

template <typename T> void OneListColEuclidean(std::vector<ANNParamType<T>> &params) {
    OneListColSqEuclidean(params);
    ConvertToEuclidean(params.back());
}

template <typename T> void OneListRowEuclidean(std::vector<ANNParamType<T>> &params) {
    OneListRowSqEuclidean(params);
    ConvertToEuclidean(params.back());
}

template <typename T> void ColEuclidean(std::vector<ANNParamType<T>> &params) {
    ColSqEuclidean(params);
    ConvertToEuclidean(params.back());
}

template <typename T> void RowEuclidean(std::vector<ANNParamType<T>> &params) {
    RowSqEuclidean(params);
    ConvertToEuclidean(params.back());
}

template <typename T> void WideColEuclidean(std::vector<ANNParamType<T>> &params) {
    WideColSqEuclidean(params);
    ConvertToEuclidean(params.back());
}

template <typename T> void WideRowEuclidean(std::vector<ANNParamType<T>> &params) {
    WideRowSqEuclidean(params);
    ConvertToEuclidean(params.back());
}

template <typename T> void PaddedColEuclidean(std::vector<ANNParamType<T>> &params) {
    PaddedColSqEuclidean(params);
    ConvertToEuclidean(params.back());
}

template <typename T> void PaddedRowEuclidean(std::vector<ANNParamType<T>> &params) {
    PaddedRowSqEuclidean(params);
    ConvertToEuclidean(params.back());
}

template <typename T> void GetANNFunctionalityData(std::vector<ANNParamType<T>> &params) {
    ZeroCol(params);
    ZeroRow(params);
    ZeroColIP(params);
    ZeroRowIP(params);
    OneByOneCol(params);
    OneByOneRow(params);
    OneByOneColIP(params);
    OneByOneRowIP(params);
    OneListColSqEuclidean(params);
    OneListRowSqEuclidean(params);
    ColSqEuclidean(params);
    RowSqEuclidean(params);
    WideColSqEuclidean(params);
    WideRowSqEuclidean(params);
    PaddedColSqEuclidean(params);
    PaddedRowSqEuclidean(params);
    OneListColIP(params);
    OneListRowIP(params);
    ColIP(params);
    RowIP(params);
    WideColIP(params);
    WideRowIP(params);
    PaddedColIP(params);
    PaddedRowIP(params);
    OneListColEuclidean(params);
    OneListRowEuclidean(params);
    ColEuclidean(params);
    RowEuclidean(params);
    WideColEuclidean(params);
    WideRowEuclidean(params);
    PaddedColEuclidean(params);
    PaddedRowEuclidean(params);
    SubsampleCol(params);
    SubsampleRow(params);
    SubsamplePaddedCol(params);
    SubsamplePaddedRow(params);
    ZeroColCosine(params);
    ZeroRowCosine(params);
    OneByOneColCosine(params);
    OneByOneRowCosine(params);
    OneListColCosine(params);
    OneListRowCosine(params);
    ColCosine(params);
    RowCosine(params);
    WideColCosine(params);
    WideRowCosine(params);
    PaddedColCosine(params);
    PaddedRowCosine(params);
    SubsamplePaddedColCosine(params);
    SubsamplePaddedRowCosine(params);
    OutliersColIP(params);
    OutliersRowIP(params);
}

// Recall tests
template <typename T>
void RandomUniformEuclideanCol(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(8, 3, 5, "euclidean", "ivfflat", "column-major");
    test.test_name = "random uniform l2 col";
    test.csvname = "randu";
    test.target_recall = 0.80;
    test.seed = 0;
    params.push_back(test);
}

template <typename T>
void RandomUniformEuclideanRow(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(8, 3, 5, "euclidean", "ivfflat", "row-major");
    test.test_name = "random uniform l2 row";
    test.csvname = "randu";
    test.target_recall = 0.80;
    test.seed = 0;
    params.push_back(test);
}

template <typename T> void RandomUniformIPCol(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(8, 3, 5, "inner product", "ivfflat", "column-major");
    test.test_name = "random uniform ip col";
    test.csvname = "randu";
    test.target_recall = 0.80;
    test.seed = 1;
    params.push_back(test);
}

template <typename T> void RandomUniformIPRow(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(8, 3, 5, "inner product", "ivfflat", "row-major");
    test.test_name = "random uniform ip row";
    test.csvname = "randu";
    test.target_recall = 0.80;
    test.seed = 2;
    params.push_back(test);
}

template <typename T> void UnitSphereEuclideanCol(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(16, 5, 4, "euclidean", "ivfflat", "column-major");
    test.test_name = "unit sphere l2 col";
    test.csvname = "unitsphere";
    test.target_recall = 0.60;
    test.seed = 0;
    test.train_fraction = 0.64;
    params.push_back(test);
}

template <typename T> void UnitSphereEuclideanRow(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(16, 5, 4, "euclidean", "ivfflat", "row-major");
    test.test_name = "unit sphere l2 row";
    test.csvname = "unitsphere";
    test.target_recall = 0.60;
    test.seed = 0;
    test.train_fraction = 0.64;
    params.push_back(test);
}

template <typename T> void UnitSphereIPCol(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(16, 5, 4, "inner product", "ivfflat", "column-major");
    test.test_name = "unit sphere ip col";
    test.csvname = "unitsphere";
    test.target_recall = 0.60;
    test.seed = 0;
    test.train_fraction = 0.64;
    params.push_back(test);
}

template <typename T> void UnitSphereIPRow(std::vector<ANNParamType<T>> &params) {
    ANNParamType<T> test(16, 5, 4, "inner product", "ivfflat", "row-major");
    test.test_name = "unit sphere ip row";
    test.csvname = "unitsphere";
    test.target_recall = 0.60;
    test.seed = 0;
    test.train_fraction = 0.64;
    params.push_back(test);
}

template <typename T> void GetANNRecallData(std::vector<ANNParamType<T>> &params) {
    RandomUniformEuclideanCol(params);
    RandomUniformEuclideanRow(params);
    RandomUniformIPCol(params);
    RandomUniformIPRow(params);
    UnitSphereEuclideanCol(params);
    UnitSphereEuclideanRow(params);
    UnitSphereIPCol(params);
    UnitSphereIPRow(params);
}
