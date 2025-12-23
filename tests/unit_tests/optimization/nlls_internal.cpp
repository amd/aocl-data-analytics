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
#include "da_handle.hpp"
#include "nlls_functions.hpp"
#include "gtest/gtest.h"

#ifndef NO_FORTRAN

using namespace TEST_ARCH;

TEST(nlls, tamperNllsHandle) {
    using namespace template_nlls_example_box_c;
    da_handle handle_s{nullptr};
    da_handle handle_d{nullptr};
    EXPECT_EQ(da_handle_init<float>(&handle_s, da_handle_type::da_handle_nlls),
              da_status_success);
    EXPECT_EQ(da_handle_init<double>(&handle_d, da_handle_type::da_handle_nlls),
              da_status_success);

    // save...
    da_nlls::nlls<float> *nlls_s =
        dynamic_cast<da_nlls::nlls<float> *>(handle_s->alg_handle_s);
    da_nlls::nlls<double> *nlls_d =
        dynamic_cast<da_nlls::nlls<double> *>(handle_d->alg_handle_d);
    // tamper...
    handle_s->alg_handle_s = nullptr;
    handle_d->alg_handle_d = nullptr;

    da_int n{2}, m{5};
    EXPECT_EQ(da_nlls_define_residuals(handle_s, n, m, eval_r<float>, eval_J<float>,
                                       nullptr, nullptr),
              da_status_invalid_handle_type);
    EXPECT_EQ(da_nlls_define_residuals(handle_d, n, m, eval_r<double>, eval_J<double>,
                                       nullptr, nullptr),
              da_status_invalid_handle_type);
    float lower_s[2]{0};
    double lower_d[2]{0};
    EXPECT_EQ(da_nlls_define_bounds(handle_s, n, lower_s, nullptr),
              da_status_invalid_handle_type);
    EXPECT_EQ(da_nlls_define_bounds(handle_d, n, lower_d, nullptr),
              da_status_invalid_handle_type);
    float w_s[5]{0};
    double w_d[5]{0};
    EXPECT_EQ(da_nlls_define_weights(handle_s, m, w_s), da_status_invalid_handle_type);
    EXPECT_EQ(da_nlls_define_weights(handle_d, m, w_d), da_status_invalid_handle_type);
    float x_s[2]{0};
    double x_d[2]{0};
    EXPECT_EQ(da_nlls_fit(handle_s, n, x_s, nullptr), da_status_invalid_handle_type);
    EXPECT_EQ(da_nlls_fit(handle_d, n, x_d, nullptr), da_status_invalid_handle_type);
    // restore...
    handle_s->alg_handle_s = nlls_s;
    handle_d->alg_handle_d = nlls_d;
    da_handle_destroy(&handle_s);
    da_handle_destroy(&handle_d);
}

#endif