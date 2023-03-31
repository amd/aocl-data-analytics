#ifndef AOCLDA_CSV
#define AOCLDA_CSV

/**
 * \file
 * \anchor chapter_b
 * \brief Chapter B - Interfacing with CSV files.
 *
 * These routines read data of a single type from a CSV (comma separated
 * values) file into an array. For each data type, two routines are provided: one to read the data alone, and one which, in addition to the data, returns a character array of column headings.
 *
 * The routines all take a \ref da_handle object as their first argument, which must be initialized prior to the routine call using \ref da_handle_init with type \ref da_handle_csv_opts.
 * 
 * \subsection option_setting Option setting
 * Prior to reading the CSV file, various options can be set by passing the \ref da_handle to \ref da_options_set_int or \ref da_options_set_string for integer or string options respectively.
 * 
 * The following string options can be set:
 * - CSV delimiter - specify the delimiter the routines should use when reading CSV files.
 * - CSV thousands - specifcy the character used to separate thousands when reading numeric values in CSV files.
 * - CSV decimal - specify which character denotes a decimal point in CSV files.
 * - CSV comment - specify which character is used to denote comments in CSV files.
 * - CSV quote character - specify which character is used to denote quotations in CSV files.
 * - CSV escape character - specify the escape character in CSV files.
 * - CSV line terminator - specify which character is used to denote line termination in CSV files.
 * - CSV scientific notation character - specify which character is used to denote powers of 10 in floating point values in CSV files.
 * - CSV skip rows - comma- or space-separated list of rows to ignore in CSV files.
 * Note that, with the exception of the skip rows option, only single characters can be used in the options above.
 * 
 * The following \ref da_int options can be set:
 * - CSV double quote - specify whether or not to interpret two consecutive quote characters within a field as a single quote character. This option can only take the values 0 or 1.
 * - CSV whitespace delimiter - specify whether or not to use whitespace as the delimiter when reading CSV files. This option can only take the values 0 or 1.
 * - CSV skip first rows - ignore the specified number of rows from the top of the CSV file.
 * - CSV skip empty lines - specify Whether or not to ignore empty lines in CSV files. This option can only take the values 0 or 1.
 * - CSV skip initial space - specify whether or not to ignore initial spaces in CSV file lines. This option can only take the values 0 or 1.
 * - CSV skip footer - specify whether or not to ignore the last line when reading a CSV file. This option can only take the values 0 or 1.
 * - CSV warn for missing data - if set to 0 then return an error if missing data is encountered; if set to 1, issue a warning and store missing data as either a NaN (for floating point data) or the maximum value of the integer type being used.
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
 * \brief Read data of a single type from a CSV file without a header.
 * 
 * \param[in,out] handle a \ref da_handle object, initialized using \ref da_handle_init with type \ref da_handle_csv_opts.
 * \param[in] filename the relative or absolute path to a file or stream that can be opened for reading.
 * \param[out] a a pointer to the nrows &times; ncols array of data read from the CSV file. Data is stored in row major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>i</i> - 1) &times; \a ncols + <i>j</i> - 1]th entry of the array. Note that this routine allocates memory for \a a internally. It is your responsibility to deallocate this memory.
 * \param[out] nrows the number of rows loaded.
 * \param[out] ncols the number of columns loaded.
 * \return \ref da_status_. The function returns:
 * - \ref da_status_success - the operation was successfully completed
 * - \ref da_status_file_not_found = 100 - the CSV file could not be found
 * - \ref da_status_range_error - a number was beyond the range of the data type
 * - \ref da_status_no_digits - no valid digits were found in a field
 * - \ref da_status_invalid_chars - invalid characters were found when reading an integer
 * - \ref da_status_invalid_boolean - invalid data was found when reading a boolean value
 * - \ref da_status_sign_error - a negative sign was found when reading in an unsigned integer
 * - \ref da_status_file_reading_error - an error occured when reading the file
 * - \ref da_status_parsing_error - an error occured when parsing the file
 * - \ref da_status_ragged_csv - not all rows of the CSV have the same number of entries
 * - \ref da_status_warn_bad_lines - the CSV file contained one or more lines that could not be read
 * - \ref da_status_warn_missing_data - the returned array contains missing data
 * - \ref da_status_warn_no_data - no data was found

 */
da_status da_read_csv_d(da_handle handle, const char *filename, double **a, da_int *nrows,
                        da_int *ncols);

da_status da_read_csv_s(da_handle handle, const char *filename, float **a, da_int *nrows,
                        da_int *ncols);

da_status da_read_csv_int64(da_handle handle, const char *filename, int64_t **a,
                            da_int *nrows, da_int *ncols);

da_status da_read_csv_uint64(da_handle handle, const char *filename, uint64_t **a,
                             da_int *nrows, da_int *ncols);
/** \} */

/** \{
 * \brief Read boolean data from a CSV file without a header.
 * 
 * This routine reads files consisting of the (case-insensitive) words \a True and \a False and stores them in an array of type uint8_t containing the values 1 or 0 respectively.
 * 
 * \param[in,out] handle a \ref da_handle object, initialized using \ref da_handle_init with type \ref da_handle_csv_opts.
 * \param[in] filename the relative or absolute path to a file or stream that can be opened for reading.
 * \param[out] a a pointer to the nrows &times; ncols array of data read from the CSV file. Data is stored in row major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>i</i> - 1) &times; \a ncols + <i>j</i> - 1]th entry of the array. Note that this routine allocates memory for \a a internally. It is your responsibility to deallocate this memory.
 * \param[out] nrows the number of rows loaded.
 * \param[out] ncols the number of columns loaded.
 * \return \ref da_status_. The function returns:
 * - \ref da_status_success - the operation was successfully completed
 * - \ref da_status_file_not_found = 100 - the CSV file could not be found
 * - \ref da_status_range_error - a number was beyond the range of the data type
 * - \ref da_status_no_digits - no valid digits were found in a field
 * - \ref da_status_invalid_chars - invalid characters were found when reading an integer
 * - \ref da_status_invalid_boolean - invalid data was found when reading a boolean value
 * - \ref da_status_sign_error - a negative sign was found when reading in an unsigned integer
 * - \ref da_status_file_reading_error - an error occured when reading the file
 * - \ref da_status_parsing_error - an error occured when parsing the file
 * - \ref da_status_ragged_csv - not all rows of the CSV have the same number of entries
 * - \ref da_status_warn_bad_lines - the CSV file contained one or more lines that could not be read
 * - \ref da_status_warn_missing_data - the returned array contains missing data
 * - \ref da_status_warn_no_data - no data was found

 */
da_status da_read_csv_uint8(da_handle handle, const char *filename, uint8_t **a,
                            da_int *nrows, da_int *ncols);
/** \} */

/** \{
 * \brief Read data of a single type from a CSV file, together with a header row containing the column labels.
 * 
 * \param[in,out] handle a \ref da_handle object, initialized using \ref da_handle_init with type \ref da_handle_csv_opts.
 * \param[in] filename the relative or absolute path to a file or stream that can be opened for reading.
 * \param[out] a a pointer to the nrows &times; ncols array of data read from the CSV file. Data is stored in row major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>i</i> - 1) &times; \a ncols + <i>j</i> - 1]th entry of the array. Note that this routine allocates memory for \a a internally. It is your responsibility to deallocate this memory.
 * \param[out] nrows the number of rows loaded.
 * \param[out] ncols the number of columns loaded.
 * \param[out] headings a pointer to the ncols array of strings containing the column headings. Note that this routine allocates memory for \a headings internally. It is your responsibility to deallocate this memory.
 * \return \ref da_status_. The function returns:
 * - \ref da_status_success - the operation was successfully completed
 * - \ref da_status_file_not_found = 100 - the CSV file could not be found
 * - \ref da_status_range_error - a number was beyond the range of the data type
 * - \ref da_status_no_digits - no valid digits were found in a field
 * - \ref da_status_invalid_chars - invalid characters were found when reading an integer
 * - \ref da_status_invalid_boolean - invalid data was found when reading a boolean value
 * - \ref da_status_sign_error - a negative sign was found when reading in an unsigned integer
 * - \ref da_status_file_reading_error - an error occured when reading the file
 * - \ref da_status_parsing_error - an error occured when parsing the file
 * - \ref da_status_ragged_csv - not all rows of the CSV have the same number of entries
 * - \ref da_status_warn_bad_lines - the CSV file contained one or more lines that could not be read
 * - \ref da_status_warn_missing_data - the returned array contains missing data
 * - \ref da_status_warn_no_data - no data was found
 */

da_status da_read_csv_d_h(da_handle handle, const char *filename, double **a,
                          da_int *nrows, da_int *ncols, char ***headings);

da_status da_read_csv_s_h(da_handle handle, const char *filename, float **a,
                          da_int *nrows, da_int *ncols, char ***headings);

da_status da_read_csv_int64_h(da_handle handle, const char *filename, int64_t **a,
                              da_int *nrows, da_int *ncols, char ***headings);

da_status da_read_csv_uint64_h(da_handle handle, const char *filename, uint64_t **a,
                               da_int *nrows, da_int *ncols, char ***headings);
/** \} */

/** \{
 * \brief Read boolean data from a CSV file, together with a header row containing the column labels.
 *
 * This routine reads files consisting of the (case-insensitive) words \a True and \a False and stores them in an array of type uint8_t containing the values 1 or 0 respectively.
 * 
 * \param[in,out] handle a \ref da_handle object, initialized using \ref da_handle_init with type \ref da_handle_csv_opts.
 * \param[in] filename the relative or absolute path to a file or stream that can be opened for reading.
 * \param[out] a a pointer to the nrows &times; ncols array of data read from the CSV file. Data is stored in row major order, so that the element in the <i>i</i>th row and <i>j</i>th column is stored in the [(<i>i</i> - 1) &times; \a ncols + <i>j</i> - 1]th entry of the array. Note that this routine allocates memory for \a a internally. It is your responsibility to deallocate this memory.
 * \param[out] nrows the number of rows loaded.
 * \param[out] ncols the number of columns loaded.
 * \param[out] headings a pointer to the ncols array of strings containing the column headings. Note that this routine allocates memory for \a headings internally. It is your responsibility to deallocate this memory.
 * \return \ref da_status_. The function returns:
 * - \ref da_status_success - the operation was successfully completed
 * - \ref da_status_file_not_found = 100 - the CSV file could not be found
 * - \ref da_status_range_error - a number was beyond the range of the data type
 * - \ref da_status_no_digits - no valid digits were found in a field
 * - \ref da_status_invalid_chars - invalid characters were found when reading an integer
 * - \ref da_status_invalid_boolean - invalid data was found when reading a boolean value
 * - \ref da_status_sign_error - a negative sign was found when reading in an unsigned integer
 * - \ref da_status_file_reading_error - an error occured when reading the file
 * - \ref da_status_parsing_error - an error occured when parsing the file
 * - \ref da_status_ragged_csv - not all rows of the CSV have the same number of entries
 * - \ref da_status_warn_bad_lines - the CSV file contained one or more lines that could not be read
 * - \ref da_status_warn_missing_data - the returned array contains missing data
 * - \ref da_status_warn_no_data - no data was found

 */
da_status da_read_csv_uint8_h(da_handle handle, const char *filename, uint8_t **a,
                              da_int *nrows, da_int *ncols, char ***headings);
/** \} */
#ifdef __cplusplus
}
#endif

#endif
