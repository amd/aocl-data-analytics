/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclda.h"
#include "da_datastore.hpp"
#include <vector>

da_status da_datastore_init(da_datastore *store) {

    // Exclude memory checks from coverage
    // LCOV_EXCL_START
    try {
        *store = new _da_datastore;
        (*store)->err = new da_errors::da_error_t(da_errors::action_t::DA_RECORD);
    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    }
    try {
        (*store)->store = new da_data::data_store(*(*store)->err);
        (*store)->opts = new da_options::OptionRegistry();
        (*store)->csv_parser = new da_csv::csv_reader(*(*store)->opts, *(*store)->err);
    } catch (std::bad_alloc &) {
        return da_error((*store)->err, da_status_memory_error, "Memory allocation error");
    }
    // LCOV_EXCL_STOP

    da_status status = da_csv::register_csv_options(*(*store)->opts);
    return status; // Error message already loaded
}

da_status da_datastore_print_error_message(da_datastore store) {
    // check to see if we have a valid store
    if (store) {
        if (store->err) {
            store->err->print();
            return da_status_success;
        } else {
            return da_status_internal_error; // LCOV_EXCL_LINE
        }
    }
    return da_status_invalid_input;
}

void da_datastore_destroy(da_datastore *store) {

    if (store) {
        if (*store) {
            if ((*store)->store)
                delete (*store)->store;
            if ((*store)->csv_parser)
                delete (*store)->csv_parser;
            if ((*store)->err)
                delete (*store)->err;
            if ((*store)->opts)
                delete (*store)->opts;
        }
        delete (*store);
        *store = nullptr;
    }
}

da_status da_data_print_options(da_datastore store) {
    if (!store)
        return da_status_store_not_initialized;

    store->opts->print_options();

    return da_status_success;
}

da_status da_data_hconcat(da_datastore *store1, da_datastore *store2) {
    da_status status;
    if (!store1 || !store2 || !(*store1) || !(*store2))
        return da_status_store_not_initialized;
    if ((*store1)->store == nullptr || (*store2)->store == nullptr) {
        // LCOV_EXCL_START
        da_error((*store1)->err, da_status_internal_error,
                 "store1 or store2 seems to be invalid?");
        status = da_error((*store2)->err, da_status_internal_error,
                          "store1 or store2 seems to be invalid?");
        return status; // Error message already loaded
        // LCOV_EXCL_STOP
    }

    status = (*store1)->store->horizontal_concat(*(*store2)->store);
    if (status == da_status_success) {
        da_datastore_destroy(store2);
    }
    return status; // Error message already loaded
}

/* ********************************** Load routines ********************************** */
/* *********************************************************************************** */
da_status da_data_load_col_int(da_datastore store, da_int n_rows, da_int n_cols,
                               da_int *block, da_ordering order, da_int copy_data) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE
    if (!block)
        return da_error(store->err, da_status_invalid_input, "block has to be defined");

    bool cpy = copy_data != 0;
    return store->store->concatenate_columns(n_rows, n_cols, block, order, cpy);
}
da_status da_data_load_row_int(da_datastore store, da_int n_rows, da_int n_cols,
                               da_int *block, da_ordering order, da_int copy_data) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE
    if (!block)
        return da_error(store->err, da_status_invalid_input, "block has to be defined");

    bool cpy = copy_data != 0;
    return store->store->concatenate_rows(n_rows, n_cols, block, order, cpy);
}

da_status da_data_load_col_str(da_datastore store, da_int n_rows, da_int n_cols,
                               const char **block, da_ordering order) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (!block)
        return da_error(store->err, da_status_invalid_input, "block has to be defined");
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE

    // FIXME
    // The current version  is copying the data TWICE.
    // We need a better way to convert C style character arrays to C++ strings
    std::vector<std::string> vecstr(block, block + n_rows * n_cols);
    return store->store->concatenate_columns(n_rows, n_cols, vecstr.data(), order, true);
}
da_status da_data_load_row_str(da_datastore store, da_int n_rows, da_int n_cols,
                               const char **block, da_ordering order) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE
    if (!block)
        return da_error(store->err, da_status_invalid_input, "block has to be defined");

    // FIXME
    // The current version  is copying the data TWICE.
    // We need a better way to convert C style character arrays to C++ strings
    std::vector<std::string> vecstr(block, block + n_rows * n_cols);
    return store->store->concatenate_rows(n_rows, n_cols, vecstr.data(), order, true);
}

da_status da_data_load_col_real_d(da_datastore store, da_int n_rows, da_int n_cols,
                                  double *block, da_ordering order, da_int copy_data) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (!block)
        return da_error(store->err, da_status_invalid_input, "block has to be defined");
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE

    bool cpy = copy_data != 0;
    return store->store->concatenate_columns(n_rows, n_cols, block, order, cpy);
}
da_status da_data_load_row_real_d(da_datastore store, da_int n_rows, da_int n_cols,
                                  double *block, da_ordering order, da_int copy_data) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (!block)
        return da_error(store->err, da_status_invalid_input, "block has to be defined");
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE

    bool cpy = copy_data != 0;
    return store->store->concatenate_rows(n_rows, n_cols, block, order, cpy);
}

da_status da_data_load_col_real_s(da_datastore store, da_int n_rows, da_int n_cols,
                                  float *block, da_ordering order, da_int copy_data) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (!block)
        return da_error(store->err, da_status_invalid_input, "block has to be defined");
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE

    bool cpy = copy_data != 0;
    return store->store->concatenate_columns(n_rows, n_cols, block, order, cpy);
}
da_status da_data_load_row_real_s(da_datastore store, da_int n_rows, da_int n_cols,
                                  float *block, da_ordering order, da_int copy_data) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (!block)
        return da_error(store->err, da_status_invalid_input, "block has to be defined");
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE

    bool cpy = copy_data != 0;
    return store->store->concatenate_rows(n_rows, n_cols, block, order, cpy);
}

da_status da_data_load_col_uint8(da_datastore store, da_int n_rows, da_int n_cols,
                                 uint8_t *block, da_ordering order, da_int copy_data) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (!block)
        return da_error(store->err, da_status_invalid_input, "block has to be defined");
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE

    bool cpy = copy_data != 0;
    return store->store->concatenate_columns(n_rows, n_cols, block, order, cpy);
}
da_status da_data_load_row_uint8(da_datastore store, da_int n_rows, da_int n_cols,
                                 uint8_t *block, da_ordering order, da_int copy_data) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (!block)
        return da_error(store->err, da_status_invalid_input, "block has to be defined");
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE

    bool cpy = copy_data != 0;
    return store->store->concatenate_rows(n_rows, n_cols, block, order, cpy);
}

/* ************************************* selection *********************************** */
/* *********************************************************************************** */
da_status da_data_select_columns(da_datastore store, const char *key, da_int lbound,
                                 da_int ubound) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE
    if (!key)
        return da_error(store->err, da_status_invalid_input, "key has to be defined");

    std::string key_str(key);
    if (!da_data::check_internal_string(key_str)) {
        std::string errmsg = "key cannot contain the prefix: ";
        errmsg += DA_STRINTERNAL;
        return da_error(store->err, da_status_invalid_input, errmsg);
    }

    return store->store->select_columns(key_str, {lbound, ubound});
}
da_status da_data_select_rows(da_datastore store, const char *key, da_int lbound,
                              da_int ubound) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE
    if (!key)
        return da_error(store->err, da_status_invalid_input, "key has to be defined");

    std::string key_str(key);
    if (!da_data::check_internal_string(key_str)) {
        std::string errmsg = "key cannot contain the prefix: ";
        errmsg += DA_STRINTERNAL;
        return da_error(store->err, da_status_invalid_input, errmsg);
    }

    return store->store->select_rows(key_str, {lbound, ubound});
}
da_status da_data_select_slice(da_datastore store, const char *key, da_int row_lbound,
                               da_int row_ubound, da_int col_lbound, da_int col_ubound) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE
    if (!key)
        return da_error(store->err, da_status_invalid_input, "key has to be defined");

    std::string key_str(key);
    if (!da_data::check_internal_string(key_str)) {
        std::string errmsg = "key cannot contain the prefix: ";
        errmsg += DA_STRINTERNAL;
        return da_error(store->err, da_status_invalid_input, errmsg);
    }

    return store->store->select_slice(key_str, {row_lbound, row_ubound},
                                      {col_lbound, col_ubound});
}
da_status da_data_select_non_missing(da_datastore store, const char *key,
                                     uint8_t full_rows) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE
    if (!key)
        return da_error(store->err, da_status_invalid_input, "key has to be defined");

    std::string key_str(key);
    bool fr = full_rows > 0;
    if (!da_data::check_internal_string(key_str)) {
        std::string errmsg = "key cannot contain the prefix: ";
        errmsg += DA_STRINTERNAL;
        return da_error(store->err, da_status_invalid_input, errmsg);
    }

    return store->store->select_non_missing(key_str, fr);
}
da_status da_data_select_remove_columns(da_datastore store, const char *key,
                                        da_int lbound, da_int ubound) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs

    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE
    if (!key)
        return da_warn(store->err, da_status_invalid_input, "key has to be defined");

    std::string key_str(key);
    return store->store->remove_columns_from_selection(key_str, {lbound, ubound});
}
da_status da_data_select_remove_rows(da_datastore store, const char *key, da_int lbound,
                                     da_int ubound) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs

    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");
    if (!key)
        return da_warn(store->err, da_status_invalid_input, "key has to be defined");

    std::string key_str(key);
    return store->store->remove_rows_from_selection(key_str, {lbound, ubound});
}

/* ********************************** extract columns ******************************** */
/* *********************************************************************************** */
da_status da_data_extract_column_int(da_datastore store, da_int idx, da_int dim,
                                     da_int *col) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE
    if (col == nullptr)
        return da_error(store->err, da_status_invalid_input, "col has to be defined");

    return store->store->extract_column(idx, dim, col);
}
da_status da_data_extract_column_real_s(da_datastore store, da_int idx, da_int dim,
                                        float *col) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE
    if (col == nullptr)
        return da_error(store->err, da_status_invalid_input, "col has to be defined");

    return store->store->extract_column(idx, dim, col);
}

da_status da_data_extract_column_real_d(da_datastore store, da_int idx, da_int dim,
                                        double *col) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE
    if (col == nullptr)
        return da_error(store->err, da_status_invalid_input, "col has to be defined");

    return store->store->extract_column(idx, dim, col);
}
da_status da_data_extract_column_uint8(da_datastore store, da_int idx, da_int dim,
                                       uint8_t *col) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE
    if (col == nullptr)
        return da_error(store->err, da_status_invalid_input, "col has to be defined");

    return store->store->extract_column(idx, dim, col);
}
da_status da_data_extract_column_str(da_datastore store, da_int idx, da_int dim,
                                     char **col) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE
    if (col == nullptr)
        return da_error(store->err, da_status_invalid_input, "col has to be defined");

    return store->store->extract_column(idx, dim, col);
}

/* ********************************* extract selections ****************************** */
/* *********************************************************************************** */
da_status da_data_extract_selection_int(da_datastore store, const char *key, da_int *data,
                                        da_int lddata) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (!key)
        return da_error(store->err, da_status_invalid_input, "key has to be defined");
    if (!data)
        return da_error(store->err, da_status_invalid_input, "data has to be defined");

    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE

    std::string key_str(key);
    return store->store->extract_selection(key, lddata, data);
}
da_status da_data_extract_selection_real_d(da_datastore store, const char *key,
                                           double *data, da_int lddata) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (!key)
        return da_error(store->err, da_status_invalid_input, "key has to be defined");
    if (!data)
        return da_error(store->err, da_status_invalid_input, "data has to be defined");

    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE

    std::string key_str(key);
    return store->store->extract_selection(key, lddata, data);
}
da_status da_data_extract_selection_real_s(da_datastore store, const char *key,
                                           float *data, da_int lddata) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (!key)
        return da_error(store->err, da_status_invalid_input, "key has to be defined");
    if (!data)
        return da_error(store->err, da_status_invalid_input, "data has to be defined");

    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE

    std::string key_str(key);
    return store->store->extract_selection(key, lddata, data);
}
da_status da_data_extract_selection_uint8(da_datastore store, const char *key,
                                          uint8_t *data, da_int lddata) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (!key)
        return da_error(store->err, da_status_invalid_input, "key has to be defined");
    if (!data)
        return da_error(store->err, da_status_invalid_input, "data has to be defined");

    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE

    std::string key_str(key);
    return store->store->extract_selection(key, lddata, data);
}

/* ************************************* headings ************************************ */
/* *********************************************************************************** */
da_status da_data_label_column(da_datastore store, const char *label, da_int col_idx) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (!label)
        return da_error(store->err, da_status_invalid_input, "label has to be defined");

    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE

    std::string label_str(label);
    return store->store->label_column(label_str, col_idx);
}
da_status da_data_get_col_idx(da_datastore store, const char *label, da_int *col_idx) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (!label)
        return da_error(store->err, da_status_invalid_input, "label has to be defined");
    if (!col_idx)
        return da_error(store->err, da_status_invalid_input, "col_idx has to be defined");

    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE

    std::string label_str(label);
    return store->store->get_idx_from_label(label_str, *col_idx);
}
da_status da_data_get_col_label(da_datastore store, da_int col_idx, da_int *label_sz,
                                char *label) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (!label)
        return da_error(store->err, da_status_invalid_input, "label has to be defined");
    if (!label_sz)
        return da_error(store->err, da_status_invalid_input,
                        "label_sz has to be defined");
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE

    std::string label_str;
    da_status status;
    status = store->store->get_col_label(col_idx, label_str);
    if (status != da_status_success) {
        return status; // Error message already loaded
    }
    if (*label_sz < label_str.size() + 1) {
        *label_sz = (da_int)label_str.size() + 1;
        std::string buff = "label_sz was set to " + std::to_string(*label_sz) +
                           " but the output label is of size " +
                           std::to_string(label_str.size() + 1);
        return da_error(store->err, da_status_invalid_input, buff);
    }
    for (da_int i = 0; i < label_str.size(); i++)
        label[i] = label_str[i];
    label[label_str.size()] = '\0';

    return da_status_success;
}
/* **************************************** csv ************************************** */
/* *********************************************************************************** */
da_status da_data_load_from_csv(da_datastore store, const char *filename) {

    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE

    if (filename == nullptr)
        return da_error(store->err, da_status_invalid_input,
                        "filename has to be defined");

    return store->store->load_from_csv(store->csv_parser, filename);
}

/* ********************************** setters/getters ******************************** */
/* *********************************************************************************** */
da_status da_data_get_n_rows(da_datastore store, da_int *n_rows) {

    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE
    if (n_rows == nullptr)
        return da_error(store->err, da_status_invalid_input, "n_rows has to be defined");

    *n_rows = store->store->get_num_rows();
    return da_status_success;
}
da_status da_data_get_n_cols(da_datastore store, da_int *n_cols) {

    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE
    if (n_cols == nullptr)
        return da_error(store->err, da_status_invalid_input, "n_cols has to be defined");

    *n_cols = store->store->get_num_cols();
    return da_status_success;
}
da_status da_data_get_element_int(da_datastore store, da_int i, da_int j, da_int *elem) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE
    if (elem == nullptr)
        return da_error(store->err, da_status_invalid_input, "elem has to be defined");

    return store->store->get_element(i, j, *elem);
}
da_status da_data_get_element_real_d(da_datastore store, da_int i, da_int j,
                                     double *elem) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE
    if (elem == nullptr)
        return da_error(store->err, da_status_invalid_input, "elem has to be defined");

    return store->store->get_element(i, j, *elem);
}
da_status da_data_get_element_real_s(da_datastore store, da_int i, da_int j,
                                     float *elem) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE
    if (elem == nullptr)
        return da_error(store->err, da_status_invalid_input, "elem has to be defined");

    return store->store->get_element(i, j, *elem);
}
da_status da_data_get_element_uint8(da_datastore store, da_int i, da_int j,
                                    uint8_t *elem) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE
    if (elem == nullptr)
        return da_error(store->err, da_status_invalid_input, "elem has to be defined");

    return store->store->get_element(i, j, *elem);
}
da_status da_data_set_element_int(da_datastore store, da_int i, da_int j, da_int elem) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE

    return store->store->set_element(i, j, elem);
}
da_status da_data_set_element_real_d(da_datastore store, da_int i, da_int j,
                                     double elem) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE

    return store->store->set_element(i, j, elem);
}
da_status da_data_set_element_real_s(da_datastore store, da_int i, da_int j, float elem) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE

    return store->store->set_element(i, j, elem);
}
da_status da_data_set_element_uint8(da_datastore store, da_int i, da_int j,
                                    uint8_t elem) {
    if (!store)
        return da_status_store_not_initialized;
    store->clear(); // clean up store logs
    if (store->store == nullptr)
        return da_error(store->err, da_status_internal_error, // LCOV_EXCL_LINE
                        "store seems to be invalid?");        // LCOV_EXCL_LINE

    return store->store->set_element(i, j, elem);
}
