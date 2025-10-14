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

#include "../utest_utils.hpp"
#include "linear_model.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace TEST_ARCH;

TEST(linmod_internal, methodType) {
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