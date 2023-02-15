#ifndef AOCLDA_CSV
#define AOCLDA_CSV

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Routines for reading in a csv and storing an array of data, with or without a headings
 * row */
da_status da_read_csv_d(da_handle handle, const char *filename, double **a, da_int *nrows,
                        da_int *ncols);

da_status da_read_csv_s(da_handle handle, const char *filename, float **a, da_int *nrows,
                        da_int *ncols);

da_status da_read_csv_int64(da_handle handle, const char *filename, int64_t **a,
                            da_int *nrows, da_int *ncols);

da_status da_read_csv_uint64(da_handle handle, const char *filename, uint64_t **a,
                             da_int *nrows, da_int *ncols);

da_status da_read_csv_uint8(da_handle handle, const char *filename, uint8_t **a,
                            da_int *nrows, da_int *ncols);

da_status da_read_csv_d_h(da_handle handle, const char *filename, double **a,
                          da_int *nrows, da_int *ncols, char ***headings);

da_status da_read_csv_s_h(da_handle handle, const char *filename, float **a,
                          da_int *nrows, da_int *ncols, char ***headings);

da_status da_read_csv_int64_h(da_handle handle, const char *filename, int64_t **a,
                              da_int *nrows, da_int *ncols, char ***headings);

da_status da_read_csv_uint64_h(da_handle handle, const char *filename, uint64_t **a,
                               da_int *nrows, da_int *ncols, char ***headings);

da_status da_read_csv_uint8_h(da_handle handle, const char *filename, uint8_t **a,
                              da_int *nrows, da_int *ncols, char ***headings);

#ifdef __cplusplus
}
#endif

#endif