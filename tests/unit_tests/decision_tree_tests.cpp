/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "aoclda.h"

#include <random>

#include "gtest/gtest.h"

TEST(decision_tree, cpp_api_sample_features) {
    da_status status;

    std::vector<float> x = {
        0.0,
    };
    std::vector<uint8_t> y = {
        0,
    };
    da_int n_obs = 0, d = 0;

    // Initialize the decision forest class and fit model
    da_handle df_handle = nullptr;
    status = da_handle_init_s(&df_handle, da_handle_decision_tree);
    EXPECT_EQ(status, da_status_success);

    // call set_training_data with invalid value
    status =
        da_df_tree_set_training_data_s(df_handle, n_obs, d, x.data(), n_obs, y.data());
    EXPECT_EQ(status, da_status_invalid_input);

    da_handle_destroy(&df_handle);
}
