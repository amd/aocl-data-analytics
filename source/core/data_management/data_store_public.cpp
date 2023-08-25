#include "aoclda.h"
#include "da_datastore.hpp"
#include <vector>

da_status da_datastore_init(da_datastore *store) {
    da_status exit_status = da_status_success;

    try {
        *store = new _da_datastore;
    } catch (std::bad_alloc &) {
        exit_status = da_status_memory_error;
    }
    try {
        (*store)->err = new da_errors::da_error_t(da_errors::action_t::DA_RECORD);
    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    }
    try {
        (*store)->store = new da_data::data_store(*(*store)->err);
    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    }
    try {
        (*store)->opts = new da_options::OptionRegistry();
    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    }
    try {
        (*store)->csv_parser = new da_csv::csv_reader(*(*store)->opts, *(*store)->err);
    } catch (std::bad_alloc &) {
        return da_status_memory_error;
    }

    exit_status = da_csv::register_csv_options(*(*store)->opts);

    return exit_status;
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
        return da_status_invalid_input;

    store->opts->print_options();

    return da_status_success;
}

da_status da_data_hconcat(da_datastore *store1, da_datastore *store2) {
    if (!store1 || !store2 || !(*store1) || !(*store2))
        return da_status_invalid_input;
    if ((*store1)->store == nullptr || (*store2)->store == nullptr)
        return da_status_invalid_pointer;

    da_status exit_status;
    exit_status = (*store1)->store->horizontal_concat(*(*store2)->store);
    if (exit_status == da_status_success) {
        da_datastore_destroy(store2);
    }
    return exit_status;
}

/* ********************************** Load routines ********************************** */
/* *********************************************************************************** */
da_status da_data_load_col_int(da_datastore store, da_int m, da_int n, da_int *int_block,
                               da_ordering order, da_int copy_data) {
    if (!store)
        return da_status_invalid_input;
    if (!int_block)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    bool cpy = copy_data != 0;
    return store->store->concatenate_columns(m, n, int_block, order, cpy);
}
da_status da_data_load_row_int(da_datastore store, da_int m, da_int n, da_int *int_block,
                               da_ordering order, da_int copy_data) {
    if (!store)
        return da_status_invalid_input;
    if (!int_block)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    bool cpy = copy_data != 0;
    return store->store->concatenate_rows(m, n, int_block, order, cpy);
}

da_status da_data_load_col_str(da_datastore store, da_int m, da_int n,
                               const char **str_block, da_ordering order) {
    if (!store)
        return da_status_invalid_input;
    if (!str_block)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    // FIXME
    // The current version  is copying the data TWICE.
    // We need a better way to convert C style character arrays to C++ strings
    std::vector<std::string> vecstr(str_block, str_block + m * n);
    return store->store->concatenate_columns(m, n, vecstr.data(), order, true);
}
da_status da_data_load_row_str(da_datastore store, da_int m, da_int n,
                               const char **str_block, da_ordering order) {
    if (!store)
        return da_status_invalid_input;
    if (!str_block)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    // FIXME
    // The current version  is copying the data TWICE.
    // We need a better way to convert C style character arrays to C++ strings
    std::vector<std::string> vecstr(str_block, str_block + m * n);
    return store->store->concatenate_rows(m, n, vecstr.data(), order, true);
}

da_status da_data_load_col_real_d(da_datastore store, da_int m, da_int n,
                                  double *real_block, da_ordering order,
                                  da_int copy_data) {
    if (!store)
        return da_status_invalid_input;
    if (!real_block)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    bool cpy = copy_data != 0;
    return store->store->concatenate_columns(m, n, real_block, order, cpy);
}
da_status da_data_load_row_real_d(da_datastore store, da_int m, da_int n,
                                  double *real_block, da_ordering order,
                                  da_int copy_data) {
    if (!store)
        return da_status_invalid_input;
    if (!real_block)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    bool cpy = copy_data != 0;
    return store->store->concatenate_rows(m, n, real_block, order, cpy);
}

da_status da_data_load_col_real_s(da_datastore store, da_int m, da_int n,
                                  float *real_block, da_ordering order,
                                  da_int copy_data) {
    if (!store)
        return da_status_invalid_input;
    if (!real_block)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    bool cpy = copy_data != 0;
    return store->store->concatenate_columns(m, n, real_block, order, cpy);
}
da_status da_data_load_row_real_s(da_datastore store, da_int m, da_int n,
                                  float *real_block, da_ordering order,
                                  da_int copy_data) {
    if (!store)
        return da_status_invalid_input;
    if (!real_block)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    bool cpy = copy_data != 0;
    return store->store->concatenate_rows(m, n, real_block, order, cpy);
}

da_status da_data_load_col_uint8(da_datastore store, da_int m, da_int n,
                                 uint8_t *uint_block, da_ordering order,
                                 da_int copy_data) {
    if (!store)
        return da_status_invalid_input;
    if (!uint_block)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    bool cpy = copy_data != 0;
    return store->store->concatenate_columns(m, n, uint_block, order, cpy);
}
da_status da_data_load_row_uint8(da_datastore store, da_int m, da_int n,
                                 uint8_t *uint_block, da_ordering order,
                                 da_int copy_data) {
    if (!store)
        return da_status_invalid_input;
    if (!uint_block)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    bool cpy = copy_data != 0;
    return store->store->concatenate_rows(m, n, uint_block, order, cpy);
}

/* ************************************* selection *********************************** */
/* *********************************************************************************** */
da_status da_data_select_columns(da_datastore store, const char *key, da_int lbound,
                                 da_int ubound) {
    if (!store)
        return da_status_invalid_input;
    if (!key)
        return da_status_invalid_input;

    if (store->store == nullptr)
        return da_status_invalid_pointer;

    std::string key_str(key);
    return store->store->select_columns(key_str, {lbound, ubound});
}
da_status da_data_select_rows(da_datastore store, const char *key, da_int lbound,
                              da_int ubound) {
    if (!store)
        return da_status_invalid_input;
    if (!key)
        return da_status_invalid_input;

    if (store->store == nullptr)
        return da_status_invalid_pointer;

    std::string key_str(key);
    return store->store->select_rows(key_str, {lbound, ubound});
}
da_status da_data_select_slice(da_datastore store, const char *key, da_int row_lbound,
                               da_int row_ubound, da_int col_lbound, da_int col_ubound) {
    if (!store)
        return da_status_invalid_input;
    if (!key)
        return da_status_invalid_input;

    if (store->store == nullptr)
        return da_status_invalid_pointer;

    std::string key_str(key);
    return store->store->select_slice(key_str, {row_lbound, row_ubound},
                                      {col_lbound, col_ubound});
}
da_status da_data_select_non_missing(da_datastore store, const char *key,
                                     uint8_t full_rows) {
    if (!store)
        return da_status_invalid_input;
    if (!key)
        return da_error(store->err, da_status_invalid_input, "Key has to be defined");
    std::string key_str(key);
    bool fr = full_rows > 0;
    return store->store->select_non_missing(key_str, fr);
}

/* ********************************** extract columns ******************************** */
/* *********************************************************************************** */
da_status da_data_extract_column_int(da_datastore store, da_int idx, da_int m,
                                     da_int *col) {
    if (!store)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;
    if (col == nullptr)
        return da_status_invalid_input;

    return store->store->extract_column(idx, m, col);
}
da_status da_data_extract_column_real_s(da_datastore store, da_int idx, da_int m,
                                        float *col) {
    if (!store)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;
    if (col == nullptr)
        return da_status_invalid_input;

    return store->store->extract_column(idx, m, col);
}

da_status da_data_extract_column_real_d(da_datastore store, da_int idx, da_int m,
                                        double *col) {
    if (!store)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;
    if (col == nullptr)
        return da_status_invalid_input;

    return store->store->extract_column(idx, m, col);
}
da_status da_data_extract_column_uint8(da_datastore store, da_int idx, da_int m,
                                       uint8_t *col) {
    if (!store)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;
    if (col == nullptr)
        return da_status_invalid_input;

    return store->store->extract_column(idx, m, col);
}
da_status da_data_extract_column_str(da_datastore store, da_int idx, da_int m,
                                     char **col) {
    if (!store)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;
    if (col == nullptr)
        return da_status_invalid_input;

    return store->store->extract_column(idx, m, col);
}

/* ********************************* extract selections ****************************** */
/* *********************************************************************************** */
da_status da_data_extract_selection_int(da_datastore store, const char *key, da_int ld,
                                        da_int *data) {
    if (!store || !data)
        return da_status_invalid_input;
    if (!key)
        return da_status_invalid_input;

    if (store->store == nullptr)
        return da_status_invalid_pointer;

    std::string key_str(key);
    return store->store->extract_selection(key, ld, data);
}
da_status da_data_extract_selection_real_d(da_datastore store, const char *key, da_int ld,
                                           double *data) {
    if (!store || !data)
        return da_status_invalid_input;
    if (!key)
        return da_status_invalid_input;

    if (store->store == nullptr)
        return da_status_invalid_pointer;

    std::string key_str(key);
    return store->store->extract_selection(key, ld, data);
}
da_status da_data_extract_selection_real_s(da_datastore store, const char *key, da_int ld,
                                           float *data) {
    if (!store || !data)
        return da_status_invalid_input;
    if (!key)
        return da_status_invalid_input;

    if (store->store == nullptr)
        return da_status_invalid_pointer;

    std::string key_str(key);
    return store->store->extract_selection(key, ld, data);
}
da_status da_data_extract_selection_uint8(da_datastore store, const char *key, da_int ld,
                                          uint8_t *data) {
    if (!store || !data)
        return da_status_invalid_input;
    if (!key)
        return da_status_invalid_input;

    if (store->store == nullptr)
        return da_status_invalid_pointer;

    std::string key_str(key);
    return store->store->extract_selection(key, ld, data);
}

/* ************************************* headings ************************************ */
/* *********************************************************************************** */
da_status da_data_label_column(da_datastore store, const char *label, da_int col_idx) {
    if (!store || !label)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    std::string label_str(label);
    return store->store->label_column(label_str, col_idx);
}
da_status da_data_get_col_idx(da_datastore store, const char *label, da_int *col_idx) {
    if (!store || !label || !col_idx)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    std::string label_str(label);
    return store->store->get_idx_from_label(label_str, *col_idx);
}
da_status da_data_get_col_label(da_datastore store, da_int col_idx, da_int *label_sz, char *label) {
    if (!store || !label || !label_sz)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    std::string label_str;
    da_status status;
    status = store->store->get_col_label(col_idx, label_str);
    if (*label_sz < label_str.size() + 1 ){
        *label_sz = (da_int)label_str.size() + 1;
        return da_status_invalid_input;
    }
    for (da_int i=0; i < label_str.size(); i++)
        label[i] = label_str[i];
    label[label_str.size()] = '\0';

    return status;
}
/* **************************************** csv ************************************** */
/* *********************************************************************************** */
da_status da_data_load_from_csv(da_datastore store, const char *filename) {

    if (!store)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    return store->store->load_from_csv(store->csv_parser, filename);
}

/* ********************************** setters/getters ******************************** */
/* *********************************************************************************** */
da_status da_data_get_num_rows(da_datastore store, da_int *num_rows) {

    if (!store)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    *num_rows = store->store->get_num_rows();
    return da_status_success;
}
da_status da_data_get_num_cols(da_datastore store, da_int *num_cols) {

    if (!store)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    *num_cols = store->store->get_num_cols();
    return da_status_success;
}
da_status da_data_get_element_int(da_datastore store, da_int i, da_int j, da_int *elem) {
    if (!store)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    return store->store->get_element(i, j, elem);
}
da_status da_data_get_element_real_d(da_datastore store, da_int i, da_int j,
                                     double *elem) {
    if (!store)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    return store->store->get_element(i, j, elem);
}
da_status da_data_get_element_real_s(da_datastore store, da_int i, da_int j,
                                     float *elem) {
    if (!store)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    return store->store->get_element(i, j, elem);
}
da_status da_data_get_element_uint8(da_datastore store, da_int i, da_int j,
                                    uint8_t *elem) {
    if (!store)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    return store->store->get_element(i, j, elem);
}
da_status da_data_set_element_int(da_datastore store, da_int i, da_int j, da_int elem) {
    if (!store)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    return store->store->set_element(i, j, elem);
}
da_status da_data_set_element_real_d(da_datastore store, da_int i, da_int j,
                                     double elem) {
    if (!store)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    return store->store->set_element(i, j, elem);
}
da_status da_data_set_element_real_s(da_datastore store, da_int i, da_int j, float elem) {
    if (!store)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    return store->store->set_element(i, j, elem);
}
da_status da_data_set_element_uint8(da_datastore store, da_int i, da_int j,
                                    uint8_t elem) {
    if (!store)
        return da_status_invalid_input;
    if (store->store == nullptr)
        return da_status_invalid_pointer;

    return store->store->set_element(i, j, elem);
}