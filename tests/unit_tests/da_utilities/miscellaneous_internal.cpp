/*
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "context.hpp"
#include "da_handle.hpp"
#include "linear_model.hpp"
#include "linmod_public.hpp"
#include "macros.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <list>
#include <type_traits>

TEST(miscellaneous, aocl_da_version_string) {
    const char *version_string = da_get_version();
    std::cout << "version_string = " << version_string << std::endl;
    ASSERT_THAT(version_string, ::testing::StartsWith(AOCLDA_VERSION_STRING));
}

TEST(miscellaneous, aocl_da_git_commit) {
    const char *git_commit = da_get_git_commit();
    std::cout << "git_commit = " << git_commit << std::endl;
    ASSERT_THAT(git_commit, ::testing::StartsWith(AOCLDA_GIT_COMMIT));
}

template <typename T> class misc_test_suite : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};
using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(misc_test_suite, FloatTypes);

TYPED_TEST(misc_test_suite, refresh) {
    da_int nsamples = 5, nfeat = 2;
    TypeParam Ad[10] = {1, 2, 3, 4, 5, 1, 3, 5, 1, 1};
    TypeParam bd[5] = {1, 1, 1, 1, 1};
    da_handle handle = nullptr;

    ASSERT_EQ(da_handle_init<TypeParam>(&handle, da_handle_linmod), da_status_success);
    EXPECT_EQ(da_linmod_define_features(handle, nsamples, nfeat, Ad, nsamples, bd),
              da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "optim method", "QR"), da_status_success);
    EXPECT_EQ(da_linmod_select_model<TypeParam>(handle, linmod_model_mse),
              da_status_success);
    EXPECT_EQ(da_linmod_fit<TypeParam>(handle), da_status_success);

    // clang-format off
    auto check_refresh = [&]() -> da_status {
        DISPATCHER(handle->err, 
            da_linmod::linear_model<TypeParam> * linmod{nullptr};
            if constexpr (std::is_same_v<TypeParam, float>) {
                linmod = dynamic_cast<da_linmod::linear_model<TypeParam> *>(handle->alg_handle_s);
            }
            if constexpr (std::is_same_v<TypeParam, double>) {
                linmod = dynamic_cast<da_linmod::linear_model<TypeParam> *>(handle->alg_handle_d);
            }
            if (!linmod)
                return da_error_bypass(handle->err, da_status_internal_error, "dynamic_cast<handle> failed?");
            EXPECT_TRUE(linmod->get_model_trained());
            da_handle_refresh(handle);
            EXPECT_FALSE(linmod->get_model_trained());
            return da_status_success;
        )
    };
    // clang-format on

    EXPECT_EQ(check_refresh(), da_status_success);

    da_handle_destroy(&handle);
}