/*
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "char_to_num.hpp"
#include "da_datastore.hpp"
#include "read_csv.hpp"

/* Public facing routines for reading in a csv and storing an array of data */

da_status da_read_csv_d(da_datastore store, const char *filename, double **a,
                        da_int *n_rows, da_int *n_cols, char ***headings) {
    if (store == nullptr)
        return da_status_store_not_initialized;
    return da_csv::read_csv(store->csv_parser, filename, a, n_rows, n_cols, headings);
}

da_status da_read_csv_s(da_datastore store, const char *filename, float **a,
                        da_int *n_rows, da_int *n_cols, char ***headings) {
    if (store == nullptr)
        return da_status_store_not_initialized;
    return da_csv::read_csv(store->csv_parser, filename, a, n_rows, n_cols, headings);
}

da_status da_read_csv_int(da_datastore store, const char *filename, da_int **a,
                          da_int *n_rows, da_int *n_cols, char ***headings) {
    if (store == nullptr)
        return da_status_store_not_initialized;
    return da_csv::read_csv(store->csv_parser, filename, a, n_rows, n_cols, headings);
}

da_status da_read_csv_uint8(da_datastore store, const char *filename, uint8_t **a,
                            da_int *n_rows, da_int *n_cols, char ***headings) {
    if (store == nullptr)
        return da_status_store_not_initialized;
    return da_csv::read_csv(store->csv_parser, filename, a, n_rows, n_cols, headings);
}

da_status da_read_csv_char(da_datastore store, const char *filename, char ***a,
                           da_int *n_rows, da_int *n_cols, char ***headings) {
    if (store == nullptr)
        return da_status_store_not_initialized;
    return da_csv::read_csv(store->csv_parser, filename, a, n_rows, n_cols, headings);
}