/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

// Testing the da_error stack framework

#include "aoclda.h"
// #include "da_datastore.hpp"
// #include "da_error.hpp"
// #include "da_handle.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <string>

namespace {

TEST(ErrorStack, PublicChecks) {
    // Check for handle and datastore
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_print_error_message(handle), da_status_invalid_input);

    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_print_error_message(store), da_status_invalid_input);

    EXPECT_EQ(da_handle_init_d(&handle, da_handle_type::da_handle_linmod),
              da_status_success);
    EXPECT_EQ(da_handle_print_error_message(handle), da_status_success);

    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    EXPECT_EQ(da_datastore_print_error_message(store), da_status_success);

    da_handle_destroy(&handle);
    da_datastore_destroy(&store);
}

} // namespace