/* ************************************************************************
 * Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#include "aoclda.h"
#include "da_datastore.hpp"
#include "da_handle.hpp"
#include "options.hpp"
#include <string>

// Public (C) handlers

template <typename T>
da_status da_options_set(da_handle handle, const char *option, T value) {
    da_status status;

    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    if constexpr (!std::is_same_v<T, da_int> && !std::is_same_v<T, const char *>) {
        da_status status = handle->check_precision<T>();
        if (status != da_status_success)
            return da_error_trace(handle->err, status, "Wrong precision type.");
    }

    da_options::OptionRegistry *opts;
    status = handle->get_current_opts(&opts);
    if (status != da_status_success)
        return status; // Error message already loaded

    bool destructive = true;
    status = opts->set(option, value, da_options::user, &destructive);
    if (status != da_status_success) {
        // Construct error based on status & opts->errmsg string
        return da_error(handle->err, status, opts->errmsg);
    }
    if (destructive)
        /* call the appropriate sub-handle's refresh() member to indicate that substantial
         * changes have occurred in the handle.
         * E.g. options changes that alter the model requiring re-training, etc...  
         */
        handle->handle_refresh();
    return da_status_success;
}

da_status da_options_get(da_handle handle, const char *option, char *value,
                         da_int *lvalue, da_int *key) {
    da_status status;

    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    da_options::OptionRegistry *opts;
    status = handle->get_current_opts(&opts);
    if (status != da_status_success)
        return status; // Error message already loaded

    std::string svalue;
    if (key == nullptr) {
        status = opts->get(option, svalue);
    } else {
        status = opts->get(option, svalue, *key);
    }
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

template <typename T>
da_status da_options_get(da_handle handle, const char *option, T *value) {
    da_status status;

    if (!handle)
        return da_status_handle_not_initialized;
    handle->clear(); // Clean up handle logs

    if constexpr (!std::is_same_v<T, da_int>) {
        da_status status = handle->check_precision<T>();
        if (status != da_status_success)
            return da_error_trace(handle->err, status, "Wrong precision type.");
    }

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
    handle->clear(); // Clean up handle logs
    return da_status_success;
}

template <typename T>
da_status da_datastore_options_set(da_datastore store, const char *option, T value) {
    da_status status;

    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // Clean up store logs

    bool destructive = true;
    status = store->opts->set(option, value, da_options::user, &destructive);
    if (status != da_status_success) {
        // Construct error based on status & opts->errmsg string
        return da_error(store->err, status, store->opts->errmsg);
    }
    return da_status_success;
}

da_status da_datastore_options_get(da_datastore store, const char *option, char *value,
                                   da_int *lvalue) {
    da_status status;

    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // Clean up store logs

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

template <typename T>
da_status da_datastore_options_get(da_datastore store, const char *option, T *value) {
    da_status status;

    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // Clean up store logs

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
        store->clear(); // Clean up store logs
        return da_status_success;
    } else {
        return da_error(store->err, da_status_internal_error, "store is invalid?");
    }
}

template da_status da_options_set<float>(da_handle, const char *, float);
template da_status da_options_set<double>(da_handle, const char *, double);
template da_status da_options_set<da_int>(da_handle, const char *, da_int);
template da_status da_options_set<const char *>(da_handle, const char *, const char *);
template da_status da_options_get<float>(da_handle, const char *, float *);
template da_status da_options_get<double>(da_handle, const char *, double *);
template da_status da_options_get<da_int>(da_handle, const char *, da_int *);
template da_status da_datastore_options_set<float>(da_datastore, const char *, float);
template da_status da_datastore_options_set<double>(da_datastore, const char *, double);
template da_status da_datastore_options_set<da_int>(da_datastore, const char *, da_int);
template da_status da_datastore_options_set<const char *>(da_datastore, const char *,
                                                          const char *);
template da_status da_datastore_options_get<float>(da_datastore, const char *, float *);
template da_status da_datastore_options_get<double>(da_datastore, const char *, double *);
template da_status da_datastore_options_get<da_int>(da_datastore, const char *, da_int *);
