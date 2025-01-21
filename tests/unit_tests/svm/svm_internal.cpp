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
#include "svm.hpp"
#include "svm_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <list>

using namespace TEST_ARCH;

template <typename T> class svm_internal_test : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

template <typename T> void GetLocalSMOData(std::vector<test_local_smo_type<T>> &params) {
    set_rbf_kernel_data(params);
    set_linear_kernel_data(params);
    set_polynomial_kernel_data(params);
    set_sigmoid_kernel_data(params);
}

template <typename T>
void GetIsUpLowData(std::vector<test_is_upper_lower_type<T>> &params) {
    set_lower1_data(params);
    set_lower2_data(params);
    set_upper1_data(params);
    set_upper2_data(params);
    set_both1_data(params);
    set_both2_data(params);
}

template <typename T>
void GetWSSData(std::vector<test_working_set_selection_type<T>> &params) {
    set_wss1_data(params);
    set_wss2_data(params);
    set_wss3_data(params);
    set_wss4_data(params);
    set_wss5_data(params);
    set_wss6_data(params);
}

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(svm_internal_test, FloatTypes);

TYPED_TEST(svm_internal_test, local_smo) {
    // Test correctness of SMO procedure with results from LibSVM
    // We are checking optimal alphas and bias
    std::vector<test_local_smo_type<TypeParam>> params;
    GetLocalSMOData(params);
    for (auto &data : params) {
        TypeParam tolerance = 5.0e-3;
        TypeParam bias;
        std::vector<TypeParam> alpha(data.n), gradient(data.n), response(data.n),
            alpha_diff(data.n);
        std::vector<da_int> index_aux(data.n);
        std::vector<bool> I_up_p(data.n), I_low_p(data.n), I_up_n(data.n),
            I_low_n(data.n);

        /////////////////////////////// SVC test
        da_svm::svc<TypeParam> svc_obj;

        svc_obj.ws_size = data.n;
        svc_obj.kernel_matrix = data.kernel_data;
        svc_obj.n = data.n;
        svc_obj.ws_indexes = data.idx;
        svc_obj.C = data.C;
        svc_obj.y = data.y.data();

        svc_obj.initialisation(data.n, gradient, response, alpha);

        svc_obj.local_smo(data.n, data.idx, data.local_kernel_data, alpha,
                          data.local_alpha, gradient, data.local_gradient, response,
                          data.local_response, I_low_p, I_up_p, I_low_n, I_up_n,
                          data.first_diff, alpha_diff, data.tol);

        svc_obj.set_bias(alpha, data.local_gradient, response, data.n, bias);

        EXPECT_NEAR(bias, data.svc_expected_bias, tolerance);

        for (da_int i = 0; i < data.n; i++)
            alpha[i] *= response[i];

        EXPECT_ARR_NEAR(data.n, alpha, data.svc_alpha_expected, tolerance);

        /////////////////////////////// nu-SVC test
        da_svm::nusvc<TypeParam> nusvc_obj;
        std::fill(alpha.begin(), alpha.end(), 0.0);
        std::fill(gradient.begin(), gradient.end(), 0.0);

        nusvc_obj.ws_size = data.n;
        nusvc_obj.kernel_matrix = data.kernel_data;
        nusvc_obj.n = data.n;
        nusvc_obj.ws_indexes = data.idx;
        nusvc_obj.nu = data.nu;
        nusvc_obj.y = data.y.data();
        nusvc_obj.tau = 1e-4;

        ////////
        // This part of the code is initialisation of response, alpha and gradient which
        // is doing similar computation to initialisation() method but with consideration of
        // precomputed kernel matrix.
        for (da_int i = 0; i < data.n; i++) {
            response[i] = nusvc_obj.y[i] == 0 ? -1.0 : nusvc_obj.y[i];
        }
        TypeParam sum_pos = nusvc_obj.nu * nusvc_obj.n / 2;
        TypeParam sum_neg = sum_pos;
        for (da_int i = 0; i < data.n; i++) {
            if (response[i] > 0) {
                alpha[i] = std::min((TypeParam)1.0, sum_pos);
                sum_pos -= alpha[i];
            } else {
                alpha[i] = std::min((TypeParam)1.0, sum_neg);
                sum_neg -= alpha[i];
            }
        }
        // Compute gradient based on alpha
        da_int counter = 0;
        for (da_int i = 0; i < data.n; i++) {
            if (alpha[i] != (TypeParam)0.0) {
                index_aux[counter] = i;
                alpha_diff[counter] = alpha[i] * response[i];
                counter++;
            }
        }
        std::vector<TypeParam> kernel_matrix_nusvc(counter * data.n);
        for (da_int i = 0; i < counter; i++) {
            for (da_int j = 0; j < data.n; j++) {
                kernel_matrix_nusvc[i * data.n + j] =
                    data.kernel_data[index_aux[i] * data.n + j];
            }
        }

        nusvc_obj.update_gradient(gradient, alpha_diff, nusvc_obj.n, counter,
                                  kernel_matrix_nusvc);
        ////////

        nusvc_obj.local_smo(data.n, data.idx, data.local_kernel_data, alpha,
                            data.local_alpha, gradient, data.local_gradient, response,
                            data.local_response, I_low_p, I_up_p, I_low_n, I_up_n,
                            data.first_diff, alpha_diff, data.tol);

        nusvc_obj.set_bias(alpha, data.local_gradient, response, data.n, bias);

        EXPECT_NEAR(bias, data.nusvc_expected_bias, tolerance);

        for (da_int i = 0; i < data.n; i++)
            alpha[i] *= response[i];

        EXPECT_ARR_NEAR(data.n, alpha, data.nusvc_alpha_expected, tolerance);
    }
}

TYPED_TEST(svm_internal_test, isUpperLower) {
    // Check correctness of auxiliary function for checking belonging in upper or lower set
    std::vector<test_is_upper_lower_type<TypeParam>> params;
    GetIsUpLowData(params);
    for (auto &data : params) {
        EXPECT_EQ(is_lower(data.alpha, data.y, data.C), data.is_low);
        EXPECT_EQ(is_upper(data.alpha, data.y, data.C), data.is_up);
        if (data.y > 0) {
            EXPECT_EQ(is_lower_pos(data.alpha, data.y), data.is_low && data.y > 0);
            EXPECT_EQ(is_upper_pos(data.alpha, data.y, data.C), data.is_up && data.y > 0);
        } else if (data.y < 0) {
            EXPECT_EQ(is_lower_neg(data.alpha, data.y, data.C),
                      data.is_low && data.y < 0);
            EXPECT_EQ(is_upper_neg(data.alpha, data.y), data.is_up && data.y < 0);
        }
    }
}

TYPED_TEST(svm_internal_test, WSS) {
    // Check correctness of auxiliary function for checking belonging in upper or lower set
    std::vector<test_working_set_selection_type<TypeParam>> params;
    GetWSSData(params);
    for (auto &data : params) {
        TypeParam tolerance = 1.0e-7;
        da_svm::svc<TypeParam> svc_obj;
        svc_obj.ws_size = data.size;
        svc_obj.n = data.size;
        svc_obj.ws_indexes = data.idx;
        svc_obj.C = data.C;
        svc_obj.tau = data.tau;
        for (da_int i = 0; i < data.size; i++) {
            data.I_up[i] = is_upper(data.alpha[i], data.response[i], data.C);
            data.I_low[i] = is_lower(data.alpha[i], data.response[i], data.C);
        }
        svc_obj.wssi(data.I_up, data.gradient, data.i, data.min_gradient);
        EXPECT_EQ(data.i, data.i_expected);
        EXPECT_NEAR(data.min_gradient, data.min_gradient_expected, tolerance);
        svc_obj.wssj(data.I_low, data.gradient, data.i, data.min_gradient, data.j,
                     data.max_gradient, data.kernel_matrix, data.delta,
                     data.max_function);
        EXPECT_EQ(data.j, data.j_expected);
        EXPECT_NEAR(data.max_gradient, data.max_gradient_expected, tolerance);
        EXPECT_NEAR(data.delta, data.delta_expected,
                    da_numeric::tolerance<TypeParam>::safe_tol());
    }
}