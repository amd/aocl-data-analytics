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
#include "da_datastore.hpp"
#include "da_error.hpp"
#include "da_handle.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>
#include <string>

namespace {

using namespace da_errors;

da_status auxiliary(da_error_t *e, da_status status = da_status_file_reading_error,
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
    da_error(err, da_status_file_reading_error, "file not found!");
    err->print();
    delete err;
};

TEST(ErrorStack, TraceCall) {
    da_error_t *err = new da_error_t(action_t::DA_RECORD);
    auxiliary(err);
    da_error_trace(err, da_status_file_reading_error, "file not found!");
    err->print();
    delete err;
};

TEST(ErrorStack, TraceCall3) {
    da_error_t *err = new da_error_t(action_t::DA_RECORD);
    auxiliary(err, da_status_invalid_input);
    auxiliary(err, da_status_file_reading_error);
    auxiliary(err, da_status_option_invalid_value);
    da_error_trace(err, da_status_file_reading_error, "file not found!");
    err->print();
    delete err;
};

TEST(ErrorStack, TraceMulti) {
    da_error_t *err = new da_error_t(action_t::DA_RECORD);
    auxiliary(err, da_status_invalid_input, true, true);
    da_error_trace(err, da_status_file_reading_error, "file not found!");
    // this gets recorded [2]
    auxiliary(err, da_status_file_reading_error, true);
    // this gets recorded [3]
    da_error_trace(err, da_status_file_reading_error, "file not found!");
    err->print();
    delete err;
};

TEST(ErrorStack, TraceMulti2) {
    da_error_t *err = new da_error_t(action_t::DA_RECORD);
    // this gets recorded [0]
    da_warn_trace(err, da_status_file_reading_error, "file not found!");
    // this gets recorded [1]
    da_error_trace(err, da_status_file_reading_error, "file not found!");
    err->print();
    delete err;
};

TEST(ErrorStack, TraceStackMax) {
    da_error_t *err = new da_error_t(action_t::DA_RECORD);
    da_warn_trace(err, da_status_file_reading_error, "Stack [0] - file not found!");
    da_error_trace(err, da_status_file_reading_error, "Stack [1] - file not found!");
    da_warn_trace(err, da_status_file_reading_error, "Stack [2] - file not found!");
    da_error_trace(err, da_status_file_reading_error, "Stack [3] - file not found!");
    da_warn_trace(err, da_status_file_reading_error, "Stack [4] - file not found!");
    da_error_trace(err, da_status_file_reading_error, "Stack [5] - file not found!");
    da_warn_trace(err, da_status_file_reading_error, "Stack [6] - file not found!");
    da_error_trace(err, da_status_file_reading_error, "Stack [7] - file not found!");
    da_warn_trace(err, da_status_file_reading_error, "Stack [8] - file not found!");
    da_status status =
        da_error_trace(err, da_status_parsing_error, "Stack [9] - no digits!");
    EXPECT_EQ(status, da_status_parsing_error);
    status =
        da_error_trace(err, da_status_parsing_error, "Stack [10] - invalid boolean!");
    EXPECT_EQ(status, da_status_parsing_error);
    status =
        da_error_trace(err, da_status_invalid_pointer, "Stack [11] - invalid pointer!");
    EXPECT_EQ(status, da_status_invalid_pointer);
    err->print();
    delete err;
};

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

TEST(ErrorStack, HandleReset) {
    // Check for handle reset at public API entry point
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_type::da_handle_linmod),
              da_status_success);
    // Register an error
    EXPECT_EQ(da_options_set_int(handle, "Invalid Option", 0),
              da_status_option_not_found);
    da_handle_print_error_message(handle);
    EXPECT_NE(handle->err->get_status(), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "Print Level", 1), da_status_success);
    EXPECT_EQ(handle->err->get_status(), da_status_success);
    da_handle_destroy(&handle);
}

TEST(ErrorStack, StoreReset) {
    // Check for store reset at public API entry point
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);
    // Register an error
    EXPECT_EQ(da_datastore_options_set_int(store, "Invalid Option", 0),
              da_status_option_not_found);
    EXPECT_NE(store->err->get_status(), da_status_success);
    EXPECT_EQ(da_datastore_options_set_int(store, "CSV use header row", 1),
              da_status_success);
    EXPECT_EQ(store->err->get_status(), da_status_success);
    da_datastore_destroy(&store);
}

} // namespace