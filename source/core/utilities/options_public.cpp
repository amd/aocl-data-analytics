/* ************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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
#include "da_datastore.hpp"
#include "da_handle.hpp"
#include "options.hpp"
#include <string>

// Public (C) handlers

da_status da_options_set_int(da_handle handle, const char *option, da_int value) {
    da_status status;

    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    da_options::OptionRegistry *opts;
    status = handle->get_current_opts(&opts, true);
    if (status != da_status_success)
        return status; // Error message already loaded

    status = opts->set(option, value, da_options::user);
    if (status != da_status_success) {
        // Construct error based on status & opts->errmsg string
        return da_error(handle->err, status, opts->errmsg);
    }
    return da_status_success;
}

da_status da_options_set_string(da_handle handle, const char *option, const char *value) {
    da_status status;

    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    da_options::OptionRegistry *opts;
    status = handle->get_current_opts(&opts, true);
    if (status != da_status_success)
        return status; // Error message already loaded

    status = opts->set(option, value, da_options::user);
    if (status != da_status_success) {
        // Construct error based on status & opts->errmsg string
        return da_error(handle->err, status, opts->errmsg);
    }
    return da_status_success;
}

da_status da_options_set_real_s(da_handle handle, const char *option, float value) {
    da_status status;

    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "handle was not initialized with single precision floating point type.");

    da_options::OptionRegistry *opts;
    status = handle->get_current_opts(&opts, true);
    if (status != da_status_success)
        return status; // Error message already loaded

    status = opts->set(option, value, da_options::user);
    if (status != da_status_success) {
        // Construct error based on status & opts->errmsg string
        return da_error(handle->err, status, opts->errmsg);
    }
    return da_status_success;
}

da_status da_options_set_real_d(da_handle handle, const char *option, double value) {
    da_status status;

    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "handle was not initialized with double precision floating point type.");

    da_options::OptionRegistry *opts;
    status = handle->get_current_opts(&opts, true);
    if (status != da_status_success)
        return status; // Error message already loaded

    status = opts->set(option, value, da_options::user);
    if (status != da_status_success) {
        // Construct error based on status & opts->errmsg string
        return da_error(handle->err, status, opts->errmsg);
    }
    return da_status_success;
}

da_status da_options_get_int(da_handle handle, const char *option, da_int *value) {
    da_status status;

    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    da_options::OptionRegistry *opts;
    status = handle->get_current_opts(&opts);
    if (status != da_status_success)
        return status; // Error message already loaded

    status = opts->get(option, *value);
    if (status != da_status_success) {
        // Construct error based on status & opts->errmsg string
        return da_error(handle->err, status, opts->errmsg);
    }
    return da_status_success;
}

da_status da_options_get_string(da_handle handle, const char *option, char *value,
                                da_int *lvalue) {
    da_status status;

    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    da_options::OptionRegistry *opts;
    status = handle->get_current_opts(&opts);
    if (status != da_status_success)
        return status; // Error message already loaded

    std::string svalue;
    status = opts->get(option, svalue);
    // Need to make sure *value is big enough...
    if (status == da_status_success) {
        size_t n = svalue.size();
        if (n >= (size_t)(*lvalue)) {
            *lvalue = (da_int)(n + 1); // inform the user of the correct size
            std::string buf = "target storage where to store option string value is too "
                              "small, make it at least " +
                              std::to_string(n + 1);
            buf += " characters long";
            return da_error(handle->err, da_status_invalid_input, buf);
        }
        svalue.copy(value, n);
        value[n] = '\0';
        return da_status_success;
    } else {
        // Construct error based on status & opts->errmsg string
        return da_error(handle->err, status, opts->errmsg);
    }
}

da_status da_options_get_real_s(da_handle handle, const char *option, float *value) {
    da_status status;

    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    if (handle->precision != da_single)
        return da_error(
            handle->err, da_status_wrong_type,
            "handle was not initialized with single precision floating point type.");

    da_options::OptionRegistry *opts;
    status = handle->get_current_opts(&opts);
    if (status != da_status_success)
        return status; // Error message already loaded

    status = opts->get(option, *value);
    if (status != da_status_success) {
        // Construct error based on status & opts->errmsg string
        return da_error(handle->err, status, opts->errmsg);
    }
    return da_status_success;
}

da_status da_options_get_real_d(da_handle handle, const char *option, double *value) {
    da_status status;

    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // clean up handle logs

    if (handle->precision != da_double)
        return da_error(
            handle->err, da_status_wrong_type,
            "handle was not initialized with double precision floating point type.");

    da_options::OptionRegistry *opts;
    status = handle->get_current_opts(&opts);
    if (status != da_status_success)
        return status; // Error message already loaded

    status = opts->get(option, *value);
    if (status != da_status_success) {
        // Construct error based on status & opts->errmsg string
        return da_error(handle->err, status, opts->errmsg);
    }
    return da_status_success;
}

da_status da_options_print(da_handle handle) {
    if (!handle)
        return da_status_handle_not_initialized;

    da_options::OptionRegistry *opts;
    da_status status = handle->get_current_opts(&opts);
    if (status != da_status_success)
        return status; // Error message already loaded

    opts->print_details();
    handle->clear(); // clean up handle logs
    return da_status_success;
}

da_status da_datastore_options_set_int(da_datastore store, const char *option,
                                       da_int value) {
    da_status status;

    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs

    status = store->opts->set(option, value, da_options::user);
    if (status != da_status_success) {
        // Construct error based on status & opts->errmsg string
        return da_error(store->err, status, store->opts->errmsg);
    }
    return da_status_success;
}

da_status da_datastore_options_set_string(da_datastore store, const char *option,
                                          const char *value) {
    da_status status;

    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs

    status = store->opts->set(option, value, da_options::user);
    if (status != da_status_success) {
        // Construct error based on status & opts->errmsg string
        return da_error(store->err, status, store->opts->errmsg);
    }
    return da_status_success;
}

da_status da_datastore_options_set_real_s(da_datastore store, const char *option,
                                          float value) {
    da_status status;

    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs

    status = store->opts->set(option, value, da_options::user);
    if (status != da_status_success) {
        // Construct error based on status & opts->errmsg string
        return da_error(store->err, status, store->opts->errmsg);
    }
    return da_status_success;
}

da_status da_datastore_options_set_real_d(da_datastore store, const char *option,
                                          double value) {
    da_status status;

    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs

    status = store->opts->set(option, value, da_options::user);
    if (status != da_status_success) {
        // Construct error based on status & opts->errmsg string
        return da_error(store->err, status, store->opts->errmsg);
    }
    return da_status_success;
}

da_status da_datastore_options_get_int(da_datastore store, const char *option,
                                       da_int *value) {
    da_status status;

    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs

    status = store->opts->get(option, *value);
    if (status != da_status_success) {
        // Construct error based on status & opts->errmsg string
        return da_error(store->err, status, store->opts->errmsg);
    }
    return da_status_success;
}

da_status da_datastore_options_get_string(da_datastore store, const char *option,
                                          char *value, da_int *lvalue) {
    da_status status;

    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs

    std::string svalue;
    status = store->opts->get(option, svalue);
    // Need to make sure *value is big enough...
    if (status == da_status_success) {
        size_t n = svalue.size();
        if (n >= (size_t)(*lvalue)) {
            *lvalue = (da_int)(n + 1); // inform the user of the correct size
            std::string buf = "target storage where to store option string value is too "
                              "small, make it at least " +
                              std::to_string(n + 1);
            buf += " characters long";
            return da_error(store->err, da_status_invalid_input, buf);
        }
        svalue.copy(value, n);
        value[n] = '\0';
        return da_status_success;
    } else {
        // Construct error based on status & opts->errmsg string
        return da_error(store->err, status, store->opts->errmsg);
    }
}

da_status da_datastore_options_get_real_s(da_datastore store, const char *option,
                                          float *value) {
    da_status status;

    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs

    status = store->opts->get(option, *value);
    if (status != da_status_success) {
        // Construct error based on status & opts->errmsg string
        return da_error(store->err, status, store->opts->errmsg);
    }
    return da_status_success;
}

da_status da_datastore_options_get_real_d(da_datastore store, const char *option,
                                          double *value) {
    da_status status;

    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs

    status = store->opts->get(option, *value);
    if (status != da_status_success) {
        // Construct error based on status & opts->errmsg string
        return da_error(store->err, status, store->opts->errmsg);
    }
    return da_status_success;
}

da_status da_datastore_options_print(da_datastore store) {
    if (!store)
        return da_status_store_not_initialized;
    if (store->opts) {
        store->opts->print_details();
        store->clear(); // clean up store logs
        return da_status_success;
    } else {
        return da_error(store->err, da_status_internal_error, "store is invalid?");
    }
}
