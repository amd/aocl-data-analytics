/*
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "context.hpp"
#include "da_handle.hpp"
#include "linear_model.hpp"
#include "macros.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <type_traits>

// Test for solver-changeable settings
// Uses internal APIs directly to avoid dynamic_cast namespace mismatch
// that occurs in dynamic dispatch builds (TEST_ARCH != runtime arch).
TEST(linmod, saveOptions) {
    using namespace da_linmod_types;
    using T = double;
    // problem data
    da_int m = 5, n = 2;
    double Ad[10] = {1, 2, 3, 4, 5, 1, 3, 5, 1, 1};
    double bd[5] = {1, 1, 1, 1, 1};
    da_handle h = nullptr, h2 = nullptr;
    [[maybe_unused]] auto const expect_scaling{scaling_t::automatic};
    [[maybe_unused]] scaling_t user_scaling{scaling_t::standardize};
    da_int len{100};
    char arch[100], ns[100];

    EXPECT_EQ(da_get_arch_info(&len, arch, ns), da_status_success);
    std::cout << "Version: " << da_get_version() << '\n';
    std::cout << "ARCH: " << arch << "   NS: " << ns << '\n';

    // clang-format off
    auto check_user_option = [&]([[maybe_unused]] da_handle &handle,[[maybe_unused]] bool eq = true) -> da_status {
        DISPATCHER(handle->err,
            da_linmod::linear_model<T> * h_lm{nullptr};
            h_lm = dynamic_cast<da_linmod::linear_model<T> *>(handle->get_alg_handle<T>());
            if (!h_lm)
                return da_error_bypass(handle->err, da_status_internal_error, "dynamic_cast<handle> failed?");
            h_lm->get_user_options(user_scaling);
            std::cout << "SCALINGINT expect: " << expect_scaling << "    user: " << user_scaling<< '\n';
            if (eq) {
                EXPECT_EQ(expect_scaling, user_scaling);
            } else {
                EXPECT_NE(expect_scaling, user_scaling);
            }
            return da_status_success;
        )
        // clang-format on
    };

    ASSERT_EQ(da_handle_init<T>(&h, da_handle_linmod), da_status_success);
    ASSERT_EQ(da_handle_init<T>(&h2, da_handle_linmod), da_status_success);

    EXPECT_EQ(da_linmod_define_features_d(h, m, n, Ad, m, bd), da_status_success);
    EXPECT_EQ(da_linmod_select_model_d(h, linmod_model_mse), da_status_success);

    EXPECT_EQ(check_user_option(h), da_status_success); // [user] Options same as expected

    EXPECT_EQ(da_options_set_string(h, "scaling", "auto"), da_status_success);
    EXPECT_EQ(da_options_set_string(h, "print options", "no"), da_status_success);

    da_int expect_key_scaling, key;
    char value[100];
    len = 100;
    EXPECT_EQ(da_options_get_string_key(h, "scaling", value, &len, &expect_key_scaling),
              da_status_success);

    EXPECT_EQ(check_user_option(h), da_status_success); // [user] Options same as expected

    // stores the actual user settings
    EXPECT_EQ(da_linmod_fit_d(h), da_status_success);
    EXPECT_EQ(check_user_option(h), da_status_success); // [user] Options same as expected

    // Check that solver did change the "auto" settings
    len = 100;
    EXPECT_EQ(da_options_get_string_key(h, "scaling", value, &len, &key),
              da_status_success);
    // [solver] changed the options
    EXPECT_NE(expect_key_scaling, key);
    // check that the [user] options didn't change
    EXPECT_EQ(check_user_option(h), da_status_success);

    // Overwrites new user settings that were set by the solver (and user didn't set again)
    // Force to refit (set any option)
    EXPECT_EQ(da_options_set_string(h, "print options", "no"), da_status_success);
    // set [user] options to the user-passed option values
    EXPECT_EQ(da_linmod_fit_d(h), da_status_success);
    EXPECT_EQ(check_user_option(h, false),
              da_status_success); // [user] options not the same as the defaults

    // Check that new handle in same thread does not change the settings
    EXPECT_EQ(da_linmod_define_features_d(h2, m, n, Ad, m, bd), da_status_success);
    EXPECT_EQ(da_linmod_select_model_d(h2, linmod_model_mse), da_status_success);
    EXPECT_EQ(check_user_option(h, false),
              da_status_success); // h2 should not interfere with the values in h

    EXPECT_EQ(da_linmod_fit_d(h2), da_status_success);
    // [user] options for h2 are the defaults so matches with expected*
    EXPECT_EQ(check_user_option(h2), da_status_success);
    // h2 should not interfere with h1 [user] values, they remain different from defaults
    EXPECT_EQ(check_user_option(h, false), da_status_success);

    da_handle_destroy(&h);
    da_handle_destroy(&h2);
}

TEST(linmod_internal, methodType) {
    using namespace TEST_ARCH;
    using namespace da_linmod;

    EXPECT_FALSE(linmod_method_type::is_iterative((linmod_method)0));
    EXPECT_FALSE(linmod_method_type::is_iterative(linmod_method::undefined));
    EXPECT_FALSE(linmod_method_type::is_iterative(linmod_method::cholesky));
    da_int mid{linmod_method::cholesky};
    EXPECT_FALSE(linmod_method_type::is_iterative(linmod_method(mid)));
    linmod_method id{linmod_method::svd};
    EXPECT_FALSE(linmod_method_type::is_iterative(id));

    EXPECT_TRUE(linmod_method_type::is_iterative(linmod_method::lbfgsb));
    mid = linmod_method::coord;
    EXPECT_TRUE(linmod_method_type::is_iterative(linmod_method(mid)));
    id = linmod_method::cg;
    EXPECT_TRUE(linmod_method_type::is_iterative(id));
}

TEST(linmod_internal, eval_feature_matrix) {
    using namespace TEST_ARCH;
    using namespace da_linmod;

    const double NA = std::numeric_limits<double>::quiet_NaN();
    const da_int n = 2;
    const da_int m = 5;
    std::vector<double> x({3, 5});
    std::vector<double> xt({2, 1, 8, 3, 4});
    std::vector<double> v(m);
    std::vector<double> vt(n);
    std::vector<double> v_exp(m);
    std::vector<double> vt_exp(n);
    double alpha = 2.0;
    double beta = -3.0;

    // Store X in both formats column, then row-major
    std::vector<std::vector<double>> XX(
        {{9, 1, 7, 4, 11, NA, NA, 5, 17, 9, 21, 3, NA, NA},
         {9, 5, NA, 1, 17, NA, 7, 9, NA, 4, 21, NA, 11, 3, NA}});
    std::vector<da_order> orders({column_major, row_major});
    std::vector<da_int> lds({5 + 2, 2 + 1});

    for (auto order : orders) {
        auto X = XX[order == column_major ? 0 : 1];
        auto ldX = lds[order == column_major ? 0 : 1];

        // v = alpha * [X] * x + beta * v
        bool intercept = false;
        bool trans = false;
        v.assign({1, 3, 2, 7, 4});
        v_exp.assign({101, 167, 126, 213, 84});
        eval_feature_matrix(order, n, x.data(), m, X.data(), ldX, v.data(), intercept,
                            trans, alpha, beta);
        EXPECT_ARR_EQ(m, v, v_exp, 1, 1, 0, 0);

        // v = alpha * [X]^T x + beta * v
        intercept = false;
        trans = true;
        vt.assign({1, 3});
        vt_exp.assign({259, 339});
        eval_feature_matrix(order, n, xt.data(), m, X.data(), ldX, vt.data(), intercept,
                            trans, alpha, beta);
        EXPECT_ARR_EQ(n, vt, vt_exp, 1, 1, 0, 0);

        // v = alpha * [X, 1^T] * x + beta * v
        intercept = true;
        trans = false;
        x.assign({3, 5, 2}); // add itercept coefficient
        v.assign({1, 3, 2, 7, 4});
        v_exp.assign({105, 171, 130, 217, 88});
        eval_feature_matrix(order, n + 1, x.data(), m, X.data(), ldX, v.data(), intercept,
                            trans, alpha, beta);
        EXPECT_ARR_EQ(m, v, v_exp, 1, 1, 0, 0);

        // v = alpha * [X, 1^T]^T x + beta * v
        intercept = true;
        trans = true;
        vt.assign({7, 3, 11});
        vt_exp.assign({241, 339, 3});
        eval_feature_matrix(order, n + 1, xt.data(), m, X.data(), ldX, vt.data(),
                            intercept, trans, alpha, beta);
        EXPECT_ARR_EQ(n + 1, vt, vt_exp, 1, 1, 0, 0);
    }
}
