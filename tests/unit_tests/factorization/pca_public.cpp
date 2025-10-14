/*
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <limits>
#include <list>
#include <random>
#include <stdio.h>
#include <string.h>

#include "../datests_cblas.hh"
#include "aoclda.h"
#include "pca_test_data.hpp"

template <typename T> class PCATest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

template <typename T> std::vector<PCAParamType<T>> getParams() {
    std::vector<PCAParamType<T>> params;
    GetPCAData(params);
    return params;
}

template <typename T> void test_functionality(const PCAParamType<T> &param) {
    da_handle handle = nullptr;
    std::cout << "Functionality test: " << param.test_name << std::endl;
    EXPECT_EQ(da_handle_init<T>(&handle, da_handle_pca), da_status_success);

    EXPECT_EQ(da_options_set_string(handle, "storage order", param.order.c_str()),
              da_status_success);

    EXPECT_EQ(da_pca_set_data(handle, param.n, param.p, param.A.data(), param.lda),
              da_status_success);

    if (param.method.size() != 0) {
        EXPECT_EQ(da_options_set_string(handle, "PCA method", param.method.c_str()),
                  da_status_success);
    }
    if (param.degrees_of_freedom.size() != 0) {
        EXPECT_EQ(da_options_set_string(handle, "degrees of freedom",
                                        param.degrees_of_freedom.c_str()),
                  da_status_success);
    }

    EXPECT_EQ(da_options_set_int(handle, "n_components", param.components_required),
              da_status_success);

    EXPECT_EQ(da_options_set_int(handle, "store U", param.store_U), da_status_success);

    EXPECT_EQ(da_options_set_int(handle, "whiten", param.whiten), da_status_success);

    if (param.svd_solver.size() != 0) {
        EXPECT_EQ(da_options_set_string(handle, "svd solver", param.svd_solver.c_str()),
                  da_status_success);
    }

    EXPECT_EQ(da_pca_compute<T>(handle), param.expected_status);

    if (param.X.size() > 0) {
        da_int size_X_transform;
        if (param.order == "column-major")
            size_X_transform = param.ldx_transform * param.expected_n_components;
        else
            size_X_transform = param.ldx_transform * param.m;
        std::vector<T> X_transform(size_X_transform);
        EXPECT_EQ(da_pca_transform(handle, param.m, param.p, param.X.data(), param.ldx,
                                   X_transform.data(), param.ldx_transform),
                  da_status_success);
        EXPECT_ARR_NEAR(size_X_transform, X_transform.data(),
                        param.expected_X_transform.data(), param.epsilon);
    }

    if (param.Xinv.size() > 0) {
        da_int size_Xinv_transform;
        if (param.order == "column-major")
            size_Xinv_transform = param.ldxinv_transform * param.p;
        else
            size_Xinv_transform = param.ldxinv_transform * param.k;
        std::vector<T> Xinv_transform(size_Xinv_transform);
        EXPECT_EQ(da_pca_inverse_transform(handle, param.k, param.expected_n_components,
                                           param.Xinv.data(), param.ldxinv,
                                           Xinv_transform.data(), param.ldxinv_transform),
                  da_status_success);
        EXPECT_ARR_NEAR(size_Xinv_transform, Xinv_transform.data(),
                        param.expected_Xinv_transform.data(), param.epsilon);
    }

    da_int size_one = 1;

    da_int size_scores = param.n * param.expected_n_components;
    da_int size_principal_components = param.p * param.expected_n_components;
    da_int size_variance = param.expected_n_components;
    da_int size_u = param.n * param.expected_n_components;
    da_int size_vt = param.p * param.expected_n_components;
    da_int size_sigma = param.expected_n_components;
    std::vector<T> scores(size_scores);
    std::vector<T> components(size_principal_components);
    std::vector<T> variance(size_variance);
    std::vector<T> u(size_u);
    std::vector<T> vt(size_vt);
    std::vector<T> sigma(size_sigma);
    T total_variance;

    if (param.expected_scores.size() != 0) {
        EXPECT_EQ(
            da_handle_get_result(handle, da_pca_scores, &size_scores, scores.data()),
            da_status_success);
        EXPECT_ARR_NEAR(size_scores, scores.data(), param.expected_scores.data(),
                        param.epsilon);
    }
    if (param.expected_components.size() != 0) {
        EXPECT_EQ(da_handle_get_result(handle, da_pca_principal_components,
                                       &size_principal_components, components.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(size_principal_components, components.data(),
                        param.expected_components.data(), param.epsilon);
    }
    if (param.expected_variance.size() != 0) {
        EXPECT_EQ(da_handle_get_result(handle, da_pca_variance, &size_variance,
                                       variance.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(size_variance, variance.data(), param.expected_variance.data(),
                        param.epsilon);
    }
    if (param.expected_u.size() != 0) {
        EXPECT_EQ(da_handle_get_result(handle, da_pca_u, &size_u, u.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(size_u, u.data(), param.expected_u.data(), param.epsilon);
    }
    if (param.expected_vt.size() != 0) {
        EXPECT_EQ(da_handle_get_result(handle, da_pca_vt, &size_vt, vt.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(size_vt, vt.data(), param.expected_vt.data(), param.epsilon);
    }
    if (param.expected_sigma.size() != 0) {
        EXPECT_EQ(da_handle_get_result(handle, da_pca_sigma, &size_sigma, sigma.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(size_sigma, sigma.data(), param.expected_sigma.data(),
                        param.epsilon);
    }
    EXPECT_EQ(
        da_handle_get_result(handle, da_pca_total_variance, &size_one, &total_variance),
        da_status_success);
    EXPECT_NEAR(total_variance, param.expected_total_variance, param.epsilon);

    if (param.expected_means.size() != 0) {
        da_int size_means = param.p;
        std::vector<T> means(size_means);
        EXPECT_EQ(
            da_handle_get_result(handle, da_pca_column_means, &size_means, means.data()),
            da_status_success);
        EXPECT_ARR_NEAR(size_means, means.data(), param.expected_means.data(),
                        param.epsilon);
    }

    if (param.expected_sdevs.size() != 0) {
        da_int size_sdevs = param.p;
        std::vector<T> sdevs(size_sdevs);
        EXPECT_EQ(
            da_handle_get_result(handle, da_pca_column_sdevs, &size_sdevs, sdevs.data()),
            da_status_success);
        EXPECT_ARR_NEAR(size_sdevs, sdevs.data(), param.expected_sdevs.data(),
                        param.epsilon);
    }

    if (param.expected_rinfo.size() > 0) {
        da_int size_rinfo = 3;
        std::vector<T> rinfo(size_rinfo);
        EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &size_rinfo, rinfo.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(size_rinfo, rinfo.data(), param.expected_rinfo.data(),
                        param.epsilon);
    }

    da_handle_destroy(&handle);
}

class DoubleFunctionalityTest : public testing::TestWithParam<PCAParamType<double>> {};
class FloatFunctionalityTest : public testing::TestWithParam<PCAParamType<float>> {};

template <typename T> void PrintTo(const PCAParamType<T> &param, ::std::ostream *os) {
    *os << param.test_name;
}

TEST_P(DoubleFunctionalityTest, ParameterizedTest) {
    const PCAParamType<double> &p = GetParam();
    test_functionality(p);
}

TEST_P(FloatFunctionalityTest, ParameterizedTest) {
    const PCAParamType<float> &p = GetParam();
    test_functionality(p);
}

INSTANTIATE_TEST_SUITE_P(PCA_Functionality_Tests_Double, DoubleFunctionalityTest,
                         ::testing::ValuesIn(getParams<double>()));
INSTANTIATE_TEST_SUITE_P(PCA_Functionality_Tests_Float, FloatFunctionalityTest,
                         ::testing::ValuesIn(getParams<float>()));

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(PCATest, FloatTypes);

TYPED_TEST(PCATest, TallSkinny) {
    // Check the route through the blocked, tall, skinny QR decomposition
    da_handle handle = nullptr;
    std::vector<std::string> block_sizes = {"9", "15"};

    for (const auto &block_size : block_sizes) {
        EXPECT_EQ(da_debug_set("pca.qr_block_size_override", block_size.c_str()),
                  da_status_success);
        da_int m = 100;
        da_int n = 5;
        da_int size_A = m * n;
        std::vector<TypeParam> A(size_A);
        std::vector<TypeParam> zeros(size_A, 0.0);
        std::random_device rd;
        std::mt19937 gen(rd());
        gen.seed(474);
        auto uniform_real_dist = std::uniform_real_distribution<TypeParam>(-1.0, 1.0);
        std::generate(A.begin(), A.end(), [&]() { return uniform_real_dist(gen); });

        EXPECT_EQ(da_standardize(column_major, da_axis_col, m, n, A.data(), m, 0, 0,
                                 nullptr, nullptr),
                  da_status_success);
        std::vector<TypeParam> A_copy;
        A_copy = A;
        EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_pca), da_status_success);
        EXPECT_EQ(da_pca_set_data(handle, m, n, A.data(), m), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_components", n), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "store U", 1), da_status_success);
        EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_success);
        char answer[100];
        EXPECT_EQ(da_debug_get("pca.block_size", 100, answer), da_status_success);
        std::string expected_block_size = block_size;
        EXPECT_EQ(std::string(answer), expected_block_size);

        // All we want to do here is check that we can reform A via U * Sigma * Vt so we know the QR worked properly
        da_int size_u = m * n;
        da_int size_vt = n * n;
        da_int size_sigma = n;
        std::vector<TypeParam> u(size_u);
        std::vector<TypeParam> vt(size_vt);
        std::vector<TypeParam> sigma(size_sigma);
        EXPECT_EQ(da_handle_get_result(handle, da_pca_u, &size_u, u.data()),
                  da_status_success);
        EXPECT_EQ(da_handle_get_result(handle, da_pca_vt, &size_vt, vt.data()),
                  da_status_success);
        EXPECT_EQ(da_handle_get_result(handle, da_pca_sigma, &size_sigma, sigma.data()),
                  da_status_success);
        // Compute A - U * Sigma * Vt
        for (da_int j = 0; j < n; j++) {
            for (da_int i = 0; i < n; i++) {
                vt[i + j * n] *= sigma[i];
            }
        }
        datest_blas::cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, n,
                                (TypeParam)1.0, u.data(), m, vt.data(), n,
                                (TypeParam)-1.0, A.data(), m);
        TypeParam epsilon = 2000 * std::numeric_limits<TypeParam>::epsilon();
        EXPECT_ARR_NEAR(size_A, A.data(), zeros.data(), epsilon);
        da_handle_destroy(&handle);

        // Now do the same test with 'store U' set to 0 and check we get the same components
        EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_pca), da_status_success);
        EXPECT_EQ(da_pca_set_data(handle, m, n, A_copy.data(), m), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_components", n), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "store U", 0), da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "svd solver", "gesdd"),
                  da_status_success);
        EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_success);
        std::vector<TypeParam> sigma2(size_sigma);
        std::vector<TypeParam> vt2(size_vt);
        EXPECT_EQ(da_handle_get_result(handle, da_pca_vt, &size_vt, vt2.data()),
                  da_status_success);
        EXPECT_EQ(da_handle_get_result(handle, da_pca_sigma, &size_sigma, sigma2.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(size_sigma, sigma.data(), sigma2.data(), epsilon);
        da_handle_destroy(&handle);

        // Finally, the same test with the eigendecomposition method
        EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_pca), da_status_success);
        EXPECT_EQ(da_pca_set_data(handle, m, n, A_copy.data(), m), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_components", n), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "store U", 0), da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "svd solver", "syevd"),
                  da_status_success);
        EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_success);
        std::vector<TypeParam> sigma3(size_sigma);
        EXPECT_EQ(da_handle_get_result(handle, da_pca_sigma, &size_sigma, sigma3.data()),
                  da_status_success);
        EXPECT_ARR_NEAR(size_sigma, sigma.data(), sigma3.data(), epsilon);

        std::vector<TypeParam> vt3(size_vt);
        EXPECT_EQ(da_handle_get_result(handle, da_pca_vt, &size_vt, vt3.data()),
                  da_status_success);
        for (da_int i = 0; i < size_vt; i++) {
            vt2[i] = std::abs(vt2[i]);
            vt3[i] = std::abs(vt3[i]);
        }
        EXPECT_ARR_NEAR(size_vt, vt2.data(), vt3.data(), epsilon);
        da_handle_destroy(&handle);
    }
}

TYPED_TEST(PCATest, UncentredData) {
    // Check that SVD and eigenvalue approaches match when input is not centred
    da_int m = 20;
    da_int n = 10;
    da_int size_A = m * n;
    std::vector<TypeParam> A(size_A);
    std::random_device rd;
    std::mt19937 gen(rd());
    gen.seed(1534);
    auto uniform_real_dist = std::uniform_real_distribution<TypeParam>(-0.5, 1.0);
    std::generate(A.begin(), A.end(), [&]() { return uniform_real_dist(gen); });

    da_handle handle = nullptr;
    // Compute PCA via SVD
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_pca), da_status_success);
    EXPECT_EQ(da_pca_set_data(handle, m, n, A.data(), m), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_components", n), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "svd solver", "gesdd"), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "store U", 0), da_status_success);
    EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_success);

    da_int size_vt = n * n;
    da_int size_sigma = n;
    std::vector<TypeParam> vt(size_vt);
    std::vector<TypeParam> sigma(size_sigma);
    EXPECT_EQ(da_handle_get_result(handle, da_pca_vt, &size_vt, vt.data()),
              da_status_success);
    EXPECT_EQ(da_handle_get_result(handle, da_pca_sigma, &size_sigma, sigma.data()),
              da_status_success);

    da_handle_destroy(&handle);

    // Compute via Eigendecomposition
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_pca), da_status_success);
    EXPECT_EQ(da_pca_set_data(handle, m, n, A.data(), m), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_components", n), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "svd solver", "syevd"), da_status_success);
    EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_success);
    std::vector<TypeParam> sigma2(size_sigma);
    EXPECT_EQ(da_handle_get_result(handle, da_pca_sigma, &size_sigma, sigma2.data()),
              da_status_success);

    // check both sigma and vt match for the two methods
    TypeParam epsilon = 2000 * std::numeric_limits<TypeParam>::epsilon();
    EXPECT_ARR_NEAR(size_sigma, sigma.data(), sigma2.data(), epsilon);

    std::vector<TypeParam> vt2(size_vt);
    EXPECT_EQ(da_handle_get_result(handle, da_pca_vt, &size_vt, vt2.data()),
              da_status_success);
    for (da_int i = 0; i < size_vt; i++) {
        vt[i] = std::abs(vt[i]);
        vt2[i] = std::abs(vt2[i]);
    }
    EXPECT_ARR_NEAR(size_vt, vt.data(), vt2.data(), epsilon);
    da_handle_destroy(&handle);
}

TYPED_TEST(PCATest, MultipleCalls) {
    // Check we can repeatedly call compute etc with the same single handle

    // Get some data to use
    std::vector<PCAParamType<TypeParam>> params;
    GetSquareData1(params);
    GetTallThinData1(params);
    GetShortFatData(params);

    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_pca), da_status_success);

    for (auto &param : params) {
        EXPECT_EQ(da_pca_set_data(handle, param.n, param.p, param.A.data(), param.lda),
                  da_status_success);

        if (param.method.size() != 0) {
            EXPECT_EQ(da_options_set_string(handle, "PCA method", param.method.c_str()),
                      da_status_success);
        }
        if (param.components_required != 0) {
            EXPECT_EQ(
                da_options_set_int(handle, "n_components", param.components_required),
                da_status_success);
        }
        if (param.svd_solver.size() != 0) {
            EXPECT_EQ(
                da_options_set_string(handle, "svd solver", param.svd_solver.c_str()),
                da_status_success);
        }
        EXPECT_EQ(da_options_set_int(handle, "store U", param.store_U),
                  da_status_success);

        EXPECT_EQ(da_pca_compute<TypeParam>(handle), param.expected_status);

        if (param.X.size() > 0) {
            da_int size_X_transform = param.ldx_transform * param.expected_n_components;
            std::vector<TypeParam> X_transform(size_X_transform);
            EXPECT_EQ(da_pca_transform(handle, param.m, param.p, param.X.data(),
                                       param.ldx, X_transform.data(),
                                       param.ldx_transform),
                      da_status_success);
            EXPECT_ARR_NEAR(size_X_transform, X_transform.data(),
                            param.expected_X_transform.data(), param.epsilon);
        }

        if (param.Xinv.size() > 0) {
            da_int size_Xinv_transform = param.ldxinv_transform * param.p;
            std::vector<TypeParam> Xinv_transform(size_Xinv_transform);
            EXPECT_EQ(da_pca_inverse_transform(
                          handle, param.k, param.expected_n_components, param.Xinv.data(),
                          param.ldxinv, Xinv_transform.data(), param.ldxinv_transform),
                      da_status_success);
            EXPECT_ARR_NEAR(size_Xinv_transform, Xinv_transform.data(),
                            param.expected_Xinv_transform.data(), param.epsilon);
        }

        da_int size_scores = param.n * param.expected_n_components;
        da_int size_principal_components = param.p * param.expected_n_components;
        da_int size_variance = param.expected_n_components;
        da_int size_u = param.n * param.expected_n_components;
        da_int size_vt = param.p * param.expected_n_components;
        da_int size_sigma = param.expected_n_components;
        da_int size_one = 1;
        std::vector<TypeParam> scores(size_scores);
        std::vector<TypeParam> components(size_principal_components);
        std::vector<TypeParam> variance(size_variance);
        std::vector<TypeParam> u(size_u);
        std::vector<TypeParam> vt(size_vt);
        std::vector<TypeParam> sigma(size_sigma);
        TypeParam total_variance;

        if (param.expected_scores.size() != 0) {
            EXPECT_EQ(
                da_handle_get_result(handle, da_pca_scores, &size_scores, scores.data()),
                da_status_success);
            EXPECT_ARR_NEAR(size_scores, scores.data(), param.expected_scores.data(),
                            param.epsilon);
        }
        if (param.expected_components.size() != 0) {
            EXPECT_EQ(da_handle_get_result(handle, da_pca_principal_components,
                                           &size_principal_components, components.data()),
                      da_status_success);
            EXPECT_ARR_NEAR(size_principal_components, components.data(),
                            param.expected_components.data(), param.epsilon);
        }
        if (param.expected_variance.size() != 0) {
            EXPECT_EQ(da_handle_get_result(handle, da_pca_variance, &size_variance,
                                           variance.data()),
                      da_status_success);
            EXPECT_ARR_NEAR(size_variance, variance.data(),
                            param.expected_variance.data(), param.epsilon);
        }
        if (param.expected_u.size() != 0) {
            EXPECT_EQ(da_handle_get_result(handle, da_pca_u, &size_u, u.data()),
                      da_status_success);
            EXPECT_ARR_NEAR(size_u, u.data(), param.expected_u.data(), param.epsilon);
        }
        if (param.expected_vt.size() != 0) {
            EXPECT_EQ(da_handle_get_result(handle, da_pca_vt, &size_vt, vt.data()),
                      da_status_success);
            EXPECT_ARR_NEAR(size_vt, vt.data(), param.expected_vt.data(), param.epsilon);
        }
        if (param.expected_sigma.size() != 0) {
            EXPECT_EQ(
                da_handle_get_result(handle, da_pca_sigma, &size_sigma, sigma.data()),
                da_status_success);
            EXPECT_ARR_NEAR(size_sigma, sigma.data(), param.expected_sigma.data(),
                            param.epsilon);
        }
        EXPECT_EQ(da_handle_get_result(handle, da_pca_total_variance, &size_one,
                                       &total_variance),
                  da_status_success);
        EXPECT_NEAR(total_variance, param.expected_total_variance, param.epsilon);

        if (param.expected_rinfo.size() > 0) {
            da_int size_rinfo = 3;
            std::vector<TypeParam> rinfo(size_rinfo);
            EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &size_rinfo, rinfo.data()),
                      da_status_success);
            EXPECT_ARR_NEAR(size_rinfo, rinfo.data(), param.expected_rinfo.data(),
                            param.epsilon);
        }
    }

    da_handle_destroy(&handle);
}

TYPED_TEST(PCATest, ErrorExits) {

    // Get some data to use
    std::vector<PCAParamType<TypeParam>> params;
    GetSquareData1(params);
    TypeParam results_arr[1];
    TypeParam *null_arr = nullptr;
    da_int *null_arr_int = nullptr;
    da_int results_arr_int[1];
    da_int dim = 1;

    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_pca), da_status_success);

    // Check da_pca_set_data error exits
    EXPECT_EQ(da_pca_set_data(handle, params[0].n, params[0].p, params[0].A.data(),
                              params[0].n - 1),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_pca_set_data(handle, 0, params[0].p, params[0].A.data(), params[0].lda),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_pca_set_data(handle, params[0].n, 0, params[0].A.data(), params[0].lda),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_pca_set_data(handle, params[0].n, params[0].p, null_arr, params[0].n),
              da_status_invalid_pointer);

    // Check error exits to catch incorrect order of routine calls
    EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_no_data);
    EXPECT_EQ(da_pca_transform(handle, params[0].m, params[0].p, params[0].X.data(),
                               params[0].ldx, results_arr, 1),
              da_status_no_data);
    EXPECT_EQ(
        da_pca_inverse_transform(handle, params[0].k, params[0].expected_n_components,
                                 params[0].Xinv.data(), params[0].ldxinv, results_arr, 1),
        da_status_no_data);
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, results_arr),
              da_status_no_data);
    EXPECT_EQ(da_handle_get_result_int(handle, da_rinfo, &dim, results_arr_int),
              da_status_unknown_query);

    EXPECT_EQ(da_options_set_int(handle, "n_components", params[0].components_required),
              da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "store U", params[0].store_U),
              da_status_success);
    EXPECT_EQ(da_pca_set_data(handle, params[0].n, params[0].p, params[0].A.data(),
                              params[0].lda),
              da_status_success);
    EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_success);

    // Check da_pca_transform and da_pca_inverse_transform error exits
    EXPECT_EQ(da_pca_transform(handle, params[0].m, params[0].p, params[0].X.data(),
                               params[0].m - 1, results_arr, params[0].ldx_transform),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_pca_transform(handle, 0, params[0].p, params[0].X.data(), params[0].ldx,
                               results_arr, params[0].ldx_transform),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_pca_transform(handle, params[0].m, params[0].p + 1, params[0].X.data(),
                               params[0].ldx, results_arr, params[0].ldx_transform),
              da_status_invalid_input);
    EXPECT_EQ(da_pca_transform(handle, params[0].m, params[0].p, params[0].X.data(),
                               params[0].ldx, results_arr, params[0].m - 1),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_pca_transform(handle, params[0].m, params[0].p, params[0].X.data(),
                               params[0].ldx, null_arr, params[0].m),
              da_status_invalid_pointer);
    EXPECT_EQ(da_pca_inverse_transform(handle, params[0].k,
                                       params[0].expected_n_components,
                                       params[0].Xinv.data(), params[0].k - 1,
                                       results_arr, params[0].ldxinv_transform),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_pca_inverse_transform(handle, 0, params[0].expected_n_components,
                                       params[0].Xinv.data(), params[0].ldxinv,
                                       results_arr, params[0].ldxinv_transform),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_pca_inverse_transform(handle, params[0].k,
                                       params[0].expected_n_components + 1,
                                       params[0].Xinv.data(), params[0].ldxinv,
                                       results_arr, params[0].ldxinv_transform),
              da_status_invalid_input);
    EXPECT_EQ(da_pca_inverse_transform(
                  handle, params[0].k, params[0].expected_n_components + 1,
                  params[0].Xinv.data(), params[0].ldxinv, results_arr, params[0].k - 1),
              da_status_invalid_input);
    EXPECT_EQ(da_pca_inverse_transform(
                  handle, params[0].k, params[0].expected_n_components,
                  params[0].Xinv.data(), params[0].ldxinv, null_arr, params[0].k),
              da_status_invalid_pointer);

    // Check da_handle_get_results error exits for 'standard' results
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, null_arr),
              da_status_invalid_input);
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, null_arr_int, results_arr),
              da_status_invalid_input);
    EXPECT_EQ(da_handle_get_result_int(handle, da_rinfo, &dim, null_arr_int),
              da_status_invalid_input);
    EXPECT_EQ(da_handle_get_result_int(handle, da_rinfo, null_arr_int, null_arr_int),
              da_status_invalid_input);
    EXPECT_EQ(da_handle_get_result_int(handle, da_linmod_coef, &dim, results_arr_int),
              da_status_unknown_query);
    dim = 0;
    EXPECT_EQ(da_handle_get_result_int(handle, da_rinfo, &dim, results_arr_int),
              da_status_unknown_query);
    dim = 0;
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, results_arr),
              da_status_invalid_array_dimension);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_rinfo, &dim, results_arr),
              da_status_invalid_array_dimension);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_u, &dim, results_arr),
              da_status_invalid_array_dimension);
    EXPECT_EQ(dim, params[0].n * params[0].components_required);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_scores, &dim, results_arr),
              da_status_invalid_array_dimension);
    EXPECT_EQ(dim, params[0].n * params[0].components_required);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_variance, &dim, results_arr),
              da_status_invalid_array_dimension);
    EXPECT_EQ(dim, params[0].components_required);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_vt, &dim, results_arr),
              da_status_invalid_array_dimension);
    EXPECT_EQ(dim, params[0].p * params[0].components_required);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_sigma, &dim, results_arr),
              da_status_invalid_array_dimension);
    EXPECT_EQ(dim, params[0].components_required);
    dim = 1;
    EXPECT_EQ(
        da_handle_get_result(handle, da_pca_principal_components, &dim, results_arr),
        da_status_invalid_array_dimension);
    EXPECT_EQ(dim, params[0].n * params[0].components_required);

    // da_handle_results error exits for columns means and column sdevs
    EXPECT_EQ(da_options_set_string(handle, "PCA method", "svd"), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "store U", 1), da_status_success);
    EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_success);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_column_means, &dim, results_arr),
              da_status_unknown_query);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_column_sdevs, &dim, results_arr),
              da_status_unknown_query);
    EXPECT_EQ(da_options_set_string(handle, "PCA method", "covariance"),
              da_status_success);
    EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_success);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_column_means, &dim, results_arr),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_options_set_string(handle, "PCA method", "correlation"),
              da_status_success);
    EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_success);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_column_sdevs, &dim, results_arr),
              da_status_invalid_array_dimension);

    // da_handle_results error exits for store_U false
    EXPECT_EQ(da_options_set_int(handle, "store U", 0), da_status_success);
    EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_success);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_u, &dim, results_arr),
              da_status_invalid_option);
    dim = 1;
    EXPECT_EQ(da_handle_get_result(handle, da_pca_scores, &dim, results_arr),
              da_status_invalid_option);

    da_handle_destroy(&handle);

    // Incompatible options
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_pca), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "store U", 1), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "svd solver", "syevd"), da_status_success);
    EXPECT_EQ(da_pca_set_data(handle, params[0].n, params[0].p, params[0].A.data(),
                              params[0].lda),
              da_status_success);
    EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_incompatible_options);

    da_handle_destroy(&handle);
}

TYPED_TEST(PCATest, WhitenCovariance) {
    // Check that covariance of transformed data is identity matrix
    // for both covariance and correlation
    std::vector<std::string> methods = {"covariance", "correlation"};
    da_handle handle = nullptr;
    for (const auto &method : methods) {
        da_int m = 8;
        da_int n = 5;
        da_int n_components = 3;
        da_int lda = 8;
        da_int size_A = m * n;
        std::vector<TypeParam> A(size_A);
        std::random_device rd;
        std::mt19937 gen(rd());
        gen.seed(26853);
        auto uniform_real_dist = std::uniform_real_distribution<TypeParam>(-0.5, 1.0);
        std::generate(A.begin(), A.end(), [&]() { return uniform_real_dist(gen); });

        // Compute PCA via SVD
        EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_pca), da_status_success);
        EXPECT_EQ(da_pca_set_data(handle, m, n, A.data(), m), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_components", n_components),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "svd solver", "gesdd"),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "PCA method", method.c_str()),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "store U", 0), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "whiten", 1), da_status_success);
        EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_success);

        std::vector<TypeParam> X_transform(m * n_components);
        da_int ldx_transform = m;

        EXPECT_EQ(da_pca_transform(handle, m, n, A.data(), lda, X_transform.data(),
                                   ldx_transform),
                  da_status_success);

        // covariance of X_transform should be identity matrix
        da_int size_cov = n_components * n_components;
        std::vector<TypeParam> eye_matrix(size_cov, 0.0);
        for (da_int i = 0; i < n_components; i++) {
            eye_matrix[i + n_components * i] = (TypeParam)1.0;
        }

        std::vector<TypeParam> cov(size_cov);
        da_int ld_cov = n_components;
        TypeParam epsilon = 10 * std::numeric_limits<TypeParam>::epsilon();
        EXPECT_EQ(da_covariance_matrix(column_major, m, n_components, X_transform.data(),
                                       ldx_transform, 0, cov.data(), ld_cov, 0),
                  da_status_success);
        EXPECT_ARR_NEAR(size_cov, cov.data(), eye_matrix.data(), epsilon);
        da_handle_destroy(&handle);
    }
}

TYPED_TEST(PCATest, BadHandleTests) {

    // handle not initialized
    da_handle handle = nullptr;
    EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_handle_not_initialized);

    TypeParam A = 1;
    EXPECT_EQ(da_pca_set_data(handle, 1, 1, &A, 1), da_status_handle_not_initialized);

    EXPECT_EQ(da_pca_transform(handle, 1, 1, &A, 1, &A, 1),
              da_status_handle_not_initialized);
    EXPECT_EQ(da_pca_inverse_transform(handle, 1, 1, &A, 1, &A, 1),
              da_status_handle_not_initialized);

    // incorrect handle type
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_linmod), da_status_success);

    EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_invalid_handle_type);

    EXPECT_EQ(da_pca_set_data(handle, 1, 1, &A, 1), da_status_invalid_handle_type);

    EXPECT_EQ(da_pca_transform(handle, 1, 1, &A, 1, &A, 1),
              da_status_invalid_handle_type);
    EXPECT_EQ(da_pca_inverse_transform(handle, 1, 1, &A, 1, &A, 1),
              da_status_invalid_handle_type);

    da_handle_destroy(&handle);
}

TYPED_TEST(PCATest, RankDeficientWhiten) {
    // Simple test to check we handle zero singular values correctly when whitening
    da_int m = 3;
    da_int n = 3;
    // rank deficient A
    std::vector<TypeParam> A{1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 2.0, 3.0};

    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_pca), da_status_success);
    EXPECT_EQ(da_pca_set_data(handle, m, n, A.data(), m), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_components", n), da_status_success);
    // syevd should return one singular value to be exactly zero here
    EXPECT_EQ(da_options_set_string(handle, "svd solver", "syevd"), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "whiten", 1), da_status_success);
    EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_success);

    da_int size_A_transform = m * n;
    da_int size_sigma = n;
    std::vector<TypeParam> A_transform(size_A_transform);
    std::vector<TypeParam> sigma(size_sigma);
    // check final sigma is exactly 0
    EXPECT_EQ(da_handle_get_result(handle, da_pca_sigma, &size_sigma, sigma.data()),
              da_status_success);
    EXPECT_EQ((TypeParam)0.0, sigma[size_sigma - 1]);

    // check transformed data is finite
    EXPECT_EQ(da_pca_transform(handle, m, n, A.data(), m, A_transform.data(), m),
              da_status_success);
    for (da_int i = 0; i < size_A_transform; i++) {
        EXPECT_TRUE(std::isfinite(A_transform[i]));
    }

    da_handle_destroy(&handle);
}

TYPED_TEST(PCATest, RecoverOriginalData) {
    // Check whether inverse_transform(transform(A)) is suffciently close to A
    // Check when whitening and not
    std::vector<da_int> whiten = {1, 0};
    da_handle handle = nullptr;
    for (const auto &w : whiten) {
        da_int m = 4;
        da_int n = 4;
        da_int n_components = 3;
        // A is rank 3
        std::vector<TypeParam> A{1.0, -1.0, 1.0, -1.0, 2.0, -2.0, 2.0,  -2.0,
                                 1.0, 3.0,  4.0, 0.0,  7.0, -3.0, -0.5, 10.0};

        EXPECT_EQ(da_handle_init<TypeParam>(&handle, da_handle_pca), da_status_success);
        EXPECT_EQ(da_pca_set_data(handle, m, n, A.data(), m), da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "n_components", n_components),
                  da_status_success);
        EXPECT_EQ(da_options_set_string(handle, "svd solver", "gesdd"),
                  da_status_success);
        EXPECT_EQ(da_options_set_int(handle, "whiten", w), da_status_success);
        EXPECT_EQ(da_pca_compute<TypeParam>(handle), da_status_success);

        // transform A
        da_int size_A_transform = m * n_components;
        std::vector<TypeParam> A_transform(size_A_transform);
        EXPECT_EQ(da_pca_transform(handle, m, n, A.data(), m, A_transform.data(), m),
                  da_status_success);

        // reconstruct A_reconstructed and check it's close to A
        da_int size_A_reconstructed = m * n;
        std::vector<TypeParam> A_reconstructed(size_A_reconstructed);
        EXPECT_EQ(da_pca_inverse_transform(handle, m, n_components, A_transform.data(), m,
                                           A_reconstructed.data(), m),
                  da_status_success);
        TypeParam epsilon = 100 * std::numeric_limits<TypeParam>::epsilon();
        EXPECT_ARR_NEAR(size_A_reconstructed, A.data(), A_reconstructed.data(), epsilon);

        da_handle_destroy(&handle);
    }
}

TEST(PCATest, IncorrectHandlePrecision) {

    da_handle handle_d = nullptr;
    da_handle handle_s = nullptr;

    EXPECT_EQ(da_handle_init_d(&handle_d, da_handle_pca), da_status_success);
    EXPECT_EQ(da_handle_init_s(&handle_s, da_handle_pca), da_status_success);

    double Ad = 1.0;
    float As = 1.0f;

    EXPECT_EQ(da_pca_set_data_d(handle_s, 1, 1, &Ad, 1), da_status_wrong_type);
    EXPECT_EQ(da_pca_set_data_s(handle_d, 1, 1, &As, 1), da_status_wrong_type);

    EXPECT_EQ(da_pca_compute_d(handle_s), da_status_wrong_type);
    EXPECT_EQ(da_pca_compute_s(handle_d), da_status_wrong_type);

    EXPECT_EQ(da_pca_transform_d(handle_s, 1, 1, &Ad, 1, &Ad, 1), da_status_wrong_type);
    EXPECT_EQ(da_pca_transform_s(handle_d, 1, 1, &As, 1, &As, 1), da_status_wrong_type);

    EXPECT_EQ(da_pca_inverse_transform_d(handle_s, 1, 1, &Ad, 1, &Ad, 1),
              da_status_wrong_type);
    EXPECT_EQ(da_pca_inverse_transform_s(handle_d, 1, 1, &As, 1, &As, 1),
              da_status_wrong_type);

    da_handle_destroy(&handle_d);
    da_handle_destroy(&handle_s);
}
