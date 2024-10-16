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

#ifndef BASIC_HANDLE_OPTIONS_HPP
#define BASIC_HANDLE_OPTIONS_HPP

#include "da_error.hpp"
#include "options.hpp"

template <typename T>
inline da_status register_common_options(da_options::OptionRegistry &opts,
                                         da_errors::da_error_t &err) {
    using namespace da_options;

    try {

        // String options
        std::shared_ptr<OptionString> os;

        os = std::make_shared<OptionString>(OptionString(
            "storage order",
            "Whether data is supplied and returned in row- or column-major order.",
            {{"row-major", row_major},
             {"column-major", column_major},
             {"fortran", column_major},
             {"f", column_major},
             {"c", row_major}},
            "column-major"));
        opts.register_opt(os);

        os = std::make_shared<OptionString>(OptionString(
            "check data", "Check input data for NaNs prior to performing computation.",
            {{"yes", 1}, {"no", 0}}, "no"));
        opts.register_opt(os);

    } catch (std::bad_alloc &) {
        return da_error(&err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    } catch (...) {
        // Invalid use of the constructor, shouldn't happen (invalid_argument)
        return da_error(&err, da_status_internal_error, // LCOV_EXCL_LINE
                        "Unexpected error while registering options");
    }

    return da_status_success;
};

#endif
