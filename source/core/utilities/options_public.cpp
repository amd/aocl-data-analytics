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
#include "da_handle.hpp"
#include "options.hpp"
#include <string>

// Public (C) handlers

da_status da_options_set_int(da_handle handle, const char *option, da_int value) {
    da_status status;

    if (!handle)
        return da_status_invalid_pointer;

    da_options::OptionRegistry *opts;
    status = handle->get_current_opts(&opts);
    if (status != da_status_success)
        // invalid pointer or uninitialized handle
        return status;

    status = opts->set(option, value, da_options::user);

    return status;
}

da_status da_options_set_string(da_handle handle, const char *option, const char *value) {
    da_status status;

    if (!handle)
        return da_status_invalid_pointer;

    da_options::OptionRegistry *opts;
    status = handle->get_current_opts(&opts);
    if (status != da_status_success)
        // invalid pointer or uninitialized handle
        return status;

    status = opts->set(option, value, da_options::user);

    return status;
}

da_status da_options_set_s_real(da_handle handle, const char *option, float value) {
    da_status status;

    if (!handle)
        return da_status_invalid_pointer;
    if (handle->precision != da_single)
        return da_status_wrong_type;

    da_options::OptionRegistry *opts;
    status = handle->get_current_opts(&opts);
    if (status != da_status_success)
        // invalid pointer or uninitialized handle
        return status;

    status = opts->set(option, value, da_options::user);
    return status;
}

da_status da_options_set_d_real(da_handle handle, const char *option, double value) {
    da_status status;

    if (!handle)
        return da_status_invalid_pointer;
    if (handle->precision != da_double)
        return da_status_wrong_type;

    da_options::OptionRegistry *opts;
    status = handle->get_current_opts(&opts);
    if (status != da_status_success)
        // invalid pointer or uninitialized handle
        return status;

    status = opts->set(option, value, da_options::user);
    return status;
}

da_status da_options_get_int(da_handle handle, const char *option, da_int *value) {
    da_status status;

    if (!handle)
        return da_status_invalid_pointer;
    da_options::OptionRegistry *opts;
    status = handle->get_current_opts(&opts);
    if (status != da_status_success)
        // invalid pointer or uninitialized handle
        return status;

    status = opts->get(option, *value);
    return status;
}

da_status da_options_get_string(da_handle handle, const char *option, char *value,
                                size_t lvalue) {
    da_status status;

    if (!handle)
        return da_status_invalid_pointer;
    da_options::OptionRegistry *opts;
    status = handle->get_current_opts(&opts);
    if (status != da_status_success)
        // invalid pointer or uninitialized handle
        return status;
    std::string svalue;
    status = opts->get(option, svalue);
    // Need to make sure *value is big enough...
    // String options should be guaranteed to be no longer than 256 chars? <--- FIXME
    if (status == da_status_success) {
        size_t n = svalue.size();
        if (n >= lvalue) {
            // FIXME errmsg should be string handle->error_message = "target storage where to store option string value is too small, make it at least " + std::string(n+1) + "characters long.";
            return da_status_invalid_input;
        }
        svalue.copy(value, n);
        value[n] = '\0';
    }
    return status;
}

da_status da_options_get_d_real(da_handle handle, const char *option, double *value) {
    da_status status;

    if (!handle)
        return da_status_invalid_pointer;
    if (handle->precision != da_double)
        return da_status_wrong_type;

    da_options::OptionRegistry *opts;
    status = handle->get_current_opts(&opts);
    if (status != da_status_success)
        // invalid pointer or uninitialized handle
        return status;

    status = opts->get(option, *value);
    return status;
}
da_status da_options_get_s_real(da_handle handle, const char *option, float *value) {
    da_status status;

    if (!handle)
        return da_status_invalid_pointer;
    if (handle->precision != da_single)
        return da_status_wrong_type;

    da_options::OptionRegistry *opts;
    status = handle->get_current_opts(&opts);
    if (status != da_status_success)
        // invalid pointer or uninitialized handle
        return status;

    status = opts->get(option, *value);
    return status;
}