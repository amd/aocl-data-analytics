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

// Testing for the Options and Registry framework

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "aoclda.h"
#include "options.hpp"
#include "da_handle.hpp"

namespace {

using namespace da_options;

OptionString opt_string("string option", "Preloaded String Option", {{"yes", 1}, {"no", 0}, {"maybe", 2}}, "yes");
std::shared_ptr<OptionString> oS;

da_status preload(OptionRegistry &r){
    oS = std::make_shared<OptionString>(opt_string);
    da_status status;
    status = r.register_opt(oS);
    if (status != da_status_success)
        return status;
    //status = r.regsiter_opt(opt_int);
    //if (status != da_status_success)
    //  return status;
    return status;
}

TEST(OpRegistryWrappers, get_string) {
    da_handle handle;
    OptionRegistry *opts;
    ASSERT_EQ(da_handle_init_d(&handle, da_handle_linmod), da_status_success);
    ASSERT_EQ(handle->get_current_opts(&opts), da_status_success);
    ASSERT_EQ(preload(*opts), da_status_success);
    char value[16];
    ASSERT_EQ(da_options_get_string(handle, "string option", value, 16), da_status_success);
    ASSERT_EQ(string("yes"), string(value));
    // target char * is too small
    ASSERT_EQ(da_options_get_string(handle, "string option", value, 1), da_status_invalid_input);
    // Try to get wrong option
    ASSERT_EQ(da_options_get_string(handle, "nonexistent option", value, 1), da_status_option_not_found);
}

} // namespace