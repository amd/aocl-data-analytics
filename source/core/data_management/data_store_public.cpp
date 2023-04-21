#include "aoclda.h"
#include "data_store.hpp"
#include <vector>

da_status da_datastore_init(da_datastore *store) {
    da_status exit_status = da_status_success;

    try {
        *store = new _da_datastore;
    } catch (std::bad_alloc &) {
        exit_status = da_status_memory_error;
    }

    try {
        (*store)->store = new da_data::data_store();
    } catch (std::bad_alloc &) {
        exit_status = da_status_memory_error;
    }

    return exit_status;
}

void da_datastore_destroy(da_datastore *store) {

    if (store) {
        if (*store) {
            if ((*store)->store)
                delete (*store)->store;
        }
        delete (*store);
        *store = nullptr;
    }
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
//da_status da_data_extract_column_str(da_datastore store, da_int idx, da_int m,
//                                     char **col) {
//    if (!store)
//        return da_status_invalid_input;
//    if (store->store == nullptr)
//        return da_status_invalid_pointer;
//    if (col == nullptr)
//        return da_status_invalid_input;
//
//    return store->store->extract_column(idx, m, col);
//}
