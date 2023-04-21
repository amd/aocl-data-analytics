#ifndef AOCLDA_DATASTORE_H
#define AOCLDA_DATASTORE_H

#include "aoclda_error.h"
#include "aoclda_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _da_datastore *da_datastore;

da_status da_datastore_init(da_datastore *store);
void da_datastore_destroy(da_datastore *store);

da_status da_data_hconcat(da_datastore *store1, da_datastore *store2);

/* Questionable if all functions below should be public... */
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

/* Data extractions routines */
/* TODO Columns extraction needed ???*/
da_status da_data_extract_column_int(da_datastore store, da_int idx, da_int m,
                                     da_int *col);
da_status da_data_extract_column_real_s(da_datastore store, da_int idx, da_int m,
                                        float *col);
da_status da_data_extract_column_real_d(da_datastore store, da_int idx, da_int m,
                                        double *col);
/*TODO study how to extract string columns into C interface*/
//da_status da_data_extract_column_str(da_datastore store, da_int idx, da_int m,
//                                     char **col);

#ifdef __cplusplus
}
#endif
#endif