#ifndef AOCLDA_CSV
#define AOCLDA_CSV

/**
 * \file
 * \anchor chapter_b
 * \brief Chapter B - Interfacing with CSV files.
 *
 * \todo Routines in this header have the purpose to read in a CSV (comma separated
 * values)  and storing an array of data, with or without a headings row.
 */

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** \{
 * \brief Routine to read a CSV file.
 *
 * This is a member of a group of functions to read CSV files without a header, they share
 * a similar API interface.
 * \param[in,out] handle a handle object, initialized using \ref da_handle_init with type \ref da_handle_csv_opts.
 * \param[in] filename the relative or absolute path to a file or stream that can be opened for reading.
 * \param[out] a data pointer object FIXME.
 * \param[out] nrows number of rows loaded.
 * \param[out] ncols number of columns loaded.
 */
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
/** \} */

/** \{
 * \brief Routine to read a CSV file.
 *
 * This is a member of a group of functions to read CSV files with a header, they share
 * a similar API interface.
 * \param[in,out] handle a handle object, initialized using \ref da_handle_init with type \ref da_handle_csv_opts.
 * \param[in] filename the relative or absolute path to a file or stream that can be opened for reading.
 * \param[out] a data pointer object FIXME.
 * \param[out] nrows number of rows loaded.
 * \param[out] ncols number of columns loaded.
 * \param[out] headings FIXME.
 */

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
/** \} */
#ifdef __cplusplus
}
#endif

#endif
