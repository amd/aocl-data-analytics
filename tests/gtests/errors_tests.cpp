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

// Testing the da_error stack framework

#include "aoclda.h"
#include "da_error.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <string>

namespace {

using namespace da_errors;

da_status auxiliary(da_error_t *e, da_status status = da_status_file_not_found,
                    bool trace = false, bool warn = false) {
    if (warn) {
        if (trace) {
            return da_warn_trace(e, status, "a string describing the issue...");
        } else {
            return da_warn(e, status, "a string describing the issue...");
        }
    } else {
        if (trace) {
            return da_error_trace(e, status, "a string describing the issue...");
        } else {
            return da_error(e, status, "a string describing the issue...");
        }
    }
}

TEST(ErrorStack, SingleCall) {
    da_error_t *err = new da_error_t(action_t::DA_RECORD);
    da_error(err, da_status_file_not_found, "file not found!");
    err->print();
    delete err;
};

TEST(ErrorStack, TraceCall) {
    da_error_t *err = new da_error_t(action_t::DA_RECORD);
    auxiliary(err);
    da_error_trace(err, da_status_file_not_found, "file not found!");
    err->print();
    delete err;
};

TEST(ErrorStack, TraceCall3) {
    da_error_t *err = new da_error_t(action_t::DA_RECORD);
    auxiliary(err, da_status_invalid_input);
    auxiliary(err, da_status_file_reading_error);
    auxiliary(err, da_status_option_invalid_value);
    da_error_trace(err, da_status_file_not_found, "file not found!");
    err->print();
    delete err;
};

TEST(ErrorStack, TraceMulti) {
    da_error_t *err = new da_error_t(action_t::DA_RECORD);
    auxiliary(err, da_status_invalid_input, true, true);
    da_error_trace(err, da_status_file_not_found, "file not found!");
    // this gets recorded [2]
    auxiliary(err, da_status_file_reading_error, true);
    // this gets recorded [3]
    da_error_trace(err, da_status_file_not_found, "file not found!");
    err->print();
    delete err;
};

TEST(ErrorStack, TraceMulti2) {
    da_error_t *err = new da_error_t(action_t::DA_RECORD);
    // this gets recorded [0]
    da_warn_trace(err, da_status_file_not_found, "file not found!");
    // this gets recorded [1]
    da_error_trace(err, da_status_file_not_found, "file not found!");
    err->print();
    delete err;
};

} // namespace