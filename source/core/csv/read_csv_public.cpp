/* disable some MSVC warnings about fopen and strcpy */
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include "char_to_num.hpp"
#include "da_datastore.hpp"
#include "read_csv.hpp"

/* Public facing routines for reading in a csv and storing an array of data */

da_status da_read_csv_d(da_datastore store, const char *filename, double **a,
                          da_int *nrows, da_int *ncols, char ***headings) {
    if (store == nullptr)
        return da_status_store_not_initialized;
    return da_csv::read_csv(store->csv_parser, filename, a, nrows, ncols, headings);
}

da_status da_read_csv_s(da_datastore store, const char *filename, float **a,
                          da_int *nrows, da_int *ncols, char ***headings) {
    if (store == nullptr)
        return da_status_store_not_initialized;
    return da_csv::read_csv(store->csv_parser, filename, a, nrows, ncols, headings);
}

da_status da_read_csv_int(da_datastore store, const char *filename, da_int **a,
                              da_int *nrows, da_int *ncols, char ***headings) {
    if (store == nullptr)
        return da_status_store_not_initialized;
    return da_csv::read_csv(store->csv_parser, filename, a, nrows, ncols, headings);
}

da_status da_read_csv_uint8(da_datastore store, const char *filename, uint8_t **a,
                              da_int *nrows, da_int *ncols, char ***headings) {
    if (store == nullptr)
        return da_status_store_not_initialized;
    return da_csv::read_csv(store->csv_parser, filename, a, nrows, ncols, headings);
}

da_status da_read_csv_char(da_datastore store, const char *filename, char ***a,
                             da_int *nrows, da_int *ncols, char ***headings) {
    if (store == nullptr)
        return da_status_store_not_initialized;
    return da_csv::read_csv(store->csv_parser, filename, a, nrows, ncols, headings);
}