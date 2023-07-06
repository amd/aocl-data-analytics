#ifndef AOCLDA_DATASTORE_H
#define AOCLDA_DATASTORE_H

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _da_datastore *da_datastore;

da_status da_datastore_init(da_datastore *store);
void da_datastore_destroy(da_datastore *store);

da_status da_data_print_options(da_datastore store);

da_status da_data_hconcat(da_datastore *store1, da_datastore *store2);

/* ********************************** Load routines ********************************** */
/* *********************************************************************************** */
da_status da_data_load_col_int(da_datastore store, da_int m, da_int n, da_int *int_block,
                               da_ordering order, da_int copy_data);
da_status da_data_load_row_int(da_datastore store, da_int m, da_int n, da_int *int_block,
                               da_ordering order, da_int copy_data);
da_status da_data_load_col_str(da_datastore store, da_int m, da_int n,
                               const char **str_block, da_ordering order);
da_status da_data_load_row_str(da_datastore store, da_int m, da_int n,
                               const char **str_block, da_ordering order);
da_status da_data_load_col_real_d(da_datastore store, da_int m, da_int n,
                                  double *real_block, da_ordering order,
                                  da_int copy_data);
da_status da_data_load_row_real_d(da_datastore store, da_int m, da_int n,
                                  double *real_block, da_ordering order,
                                  da_int copy_data);
da_status da_data_load_col_real_s(da_datastore store, da_int m, da_int n,
                                  float *real_block, da_ordering order, da_int copy_data);
da_status da_data_load_row_real_s(da_datastore store, da_int m, da_int n,
                                  float *real_block, da_ordering order, da_int copy_data);

da_status da_data_load_col_uint8(da_datastore store, da_int m, da_int n,
                                 uint8_t *uint_block, da_ordering order,
                                 da_int copy_data);
da_status da_data_load_row_uint8(da_datastore store, da_int m, da_int n,
                                 uint8_t *uint_block, da_ordering order,
                                 da_int copy_data);

da_status da_data_load_from_csv(da_datastore store, const char *filename);

/* ************************************* selection *********************************** */
/* *********************************************************************************** */
da_status da_data_select_columns(da_datastore store, const char *key, da_int lbound,
                                 da_int ubound);
da_status da_data_select_rows(da_datastore store, const char *key, da_int lbound,
                              da_int ubound);
da_status da_data_select_slice(da_datastore store, const char *key, da_int row_lbound,
                               da_int row_ubound, da_int col_lbound, da_int col_ubound);
da_status da_data_select_non_missing(da_datastore store, const char *key,
                                     uint8_t full_rows);

/* ********************************** extract columns ******************************** */
/* *********************************************************************************** */
da_status da_data_extract_column_int(da_datastore store, da_int idx, da_int m,
                                     da_int *col);
da_status da_data_extract_column_real_s(da_datastore store, da_int idx, da_int m,
                                        float *col);
da_status da_data_extract_column_real_d(da_datastore store, da_int idx, da_int m,
                                        double *col);
da_status da_data_extract_column_uint8(da_datastore store, da_int idx, da_int m,
                                       uint8_t *col); // For boolean data

da_status da_data_extract_column_str(da_datastore store, da_int idx, da_int m,
                                     char **col);

/* ********************************* extract selections ****************************** */
/* *********************************************************************************** */
da_status da_data_extract_selection_int(da_datastore store, const char *key, da_int ld,
                                        da_int *data);
da_status da_data_extract_selection_real_d(da_datastore store, const char *key, da_int ld,
                                           double *data);
da_status da_data_extract_selection_real_s(da_datastore store, const char *key, da_int ld,
                                           float *data);
da_status da_data_extract_selection_uint8(da_datastore store, const char *key, da_int ld,
                                          uint8_t *data);

/* ************************************* headings ************************************ */
/* *********************************************************************************** */
da_status da_data_extract_headings(da_datastore store, da_int n, char **headings);
da_status da_data_label_column(da_datastore store, const char *label, da_int col_idx);
da_status da_data_get_col_idx(da_datastore store, const char *label, da_int *col_idx);
da_status da_data_get_col_label(da_datastore store, da_int col_idx, da_int *label_sz, char *label);

/* ********************************** setters/getters ******************************** */
/* *********************************************************************************** */
da_status da_data_get_num_rows(da_datastore store, da_int *num_rows);
da_status da_data_get_num_cols(da_datastore store, da_int *num_cols);
da_status da_data_get_element_int(da_datastore store, da_int i, da_int j, da_int *elem);
da_status da_data_get_element_real_d(da_datastore store, da_int i, da_int j,
                                     double *elem);
da_status da_data_get_element_real_s(da_datastore store, da_int i, da_int j, float *elem);
da_status da_data_get_element_uint8(da_datastore store, da_int i, da_int j,
                                    uint8_t *elem);
da_status da_data_set_element_int(da_datastore store, da_int i, da_int j, da_int elem);
da_status da_data_set_element_real_d(da_datastore store, da_int i, da_int j, double elem);
da_status da_data_set_element_real_s(da_datastore store, da_int i, da_int j, float elem);
da_status da_data_set_element_uint8(da_datastore store, da_int i, da_int j, uint8_t elem);

#ifdef __cplusplus
}
#endif
#endif