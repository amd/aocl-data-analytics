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

#ifndef AOCLDA_CSV
#define AOCLDA_CSV

#include "aoclda_datastore.h"
#include "aoclda_error.h"
#include "aoclda_types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** \{
 * \brief Read data of a single type from a CSV file, optionally with a header row containing the column labels.
 * 
 * \param[in,out] store a \ref _da_datastore object, initialized using \ref da_datastore_init.
 * \param[in] filename the relative or absolute path to a file or stream that can be opened for reading.
 * \param[out] a a pointer to the size nrows @f$\times@f$ ncols array of data read from the CSV file. Data is stored in row major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>i</i> - 1) &times; \a ncols + <i>j</i> - 1]th entry of the array. Note that this routine allocates memory for \a a internally. It is your responsibility to deallocate this memory.
 * \param[out] nrows the number of rows loaded.
 * \param[out] ncols the number of columns loaded.
 * \param[out] headings a pointer to the size ncols array of strings containing the column headings. If the option <em>CSV use header row</em> is set to 0 (the default) then this argument is not used. Otherwise, note that this routine allocates memory for \a headings internally. It is your responsibility to deallocate this memory.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success
 * - \ref da_status_file_reading_error
 * - \ref da_status_parsing_error
 * - \ref da_status_missing_data
 * */
da_status da_read_csv_d(da_datastore store, const char *filename, double **a,
                          da_int *nrows, da_int *ncols, char ***headings);

da_status da_read_csv_s(da_datastore store, const char *filename, float **a,
                          da_int *nrows, da_int *ncols, char ***headings);

da_status da_read_csv_int(da_datastore store, const char *filename, da_int **a,
                              da_int *nrows, da_int *ncols, char ***headings);

/** \} */

/** \{
 * \brief Read boolean data from a CSV file, optionally with a header row containing the column labels.
 *
 * This routine reads files consisting of the (case-insensitive) words \a True and \a False and stores them in an array of type uint8_t containing the values 1 or 0 respectively.
 * 
 * \param[in,out] store a \ref _da_datastore object, initialized using \ref da_datastore_init.
 * \param[in] filename the relative or absolute path to a file or stream that can be opened for reading.
 * \param[out] a a pointer to the size nrows @f$\times@f$ ncols array of data read from the CSV file. Data is stored in row major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>i</i> - 1) &times; \a ncols + <i>j</i> - 1]th entry of the array. Note that this routine allocates memory for \a a internally. It is your responsibility to deallocate this memory.
 * \param[out] nrows the number of rows loaded.
 * \param[out] ncols the number of columns loaded.
 * \param[out] headings a pointer to the size ncols array of strings containing the column headings. If the option <em>CSV use header row</em> is set to 0 (the default) then this argument is not used. Otherwise, note that this routine allocates memory for \a headings internally. It is your responsibility to deallocate this memory.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success
 * - \ref da_status_file_reading_error
 * - \ref da_status_parsing_error
 * - \ref da_status_missing_data
 */
da_status da_read_csv_uint8(da_datastore store, const char *filename, uint8_t **a,
                              da_int *nrows, da_int *ncols, char ***headings);
/** \} */

/** \{
 * \brief Read char string data from a CSV file, optionally with a header row containing the column labels.
 * 
 * This routine reads files consisting of the character data and stores them in an array of type char* character strings.
 * 
 * \param[in,out] store a \ref _da_datastore object, initialized using \ref da_datastore_init.
 * \param[in] filename the relative or absolute path to a file or stream that can be opened for reading.
 * \param[out] a a pointer to the size nrows @f$\times@f$ ncols array of data read from the CSV file. Data is stored in row major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>i</i> - 1) &times; \a ncols + <i>j</i> - 1]th entry of the array. Note that this routine allocates memory for \a a internally. It is your responsibility to deallocate this memory.
 * \param[out] nrows the number of rows loaded.
 * \param[out] ncols the number of columns loaded.
 * \param[out] headings a pointer to the size ncols array of strings containing the column headings. If the option <em>CSV use header row</em> is set to 0 (the default) then this argument is not used. Otherwise, note that this routine allocates memory for \a headings internally. It is your responsibility to deallocate this memory.
 * \return \ref da_status. The function returns:
 * - \ref da_status_success
 * - \ref da_status_file_reading_error
 * - \ref da_status_parsing_error
 * - \ref da_status_missing_data
 */
da_status da_read_csv_char(da_datastore store, const char *filename, char ***a,
                             da_int *nrows, da_int *ncols, char ***headings);
/** \} */

#ifdef __cplusplus
}
#endif

#endif
