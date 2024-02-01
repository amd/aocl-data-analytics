/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef AOCLDA_DATASTORE_H
#define AOCLDA_DATASTORE_H

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \file
 */

/**
 * @brief The main structure used by AOCL-DA to store data.
 * All functions of this chapter operate on this internal data structure.
 */
typedef struct _da_datastore *da_datastore;

/**
 * @brief Initialize an empty @ref da_datastore.
 *
 * @param store the @ref da_datastore to initialize
 */
da_status da_datastore_init(da_datastore *store);

/**
 * @brief  Print error information stored in the store handle.
 *
 * Print (trace of) error message(s) stored in the handle.
 * Some functions store extra information about errors and
 * this function prints (to standard output) the stored error message(s).
 *
 * @param[in,out] store the da_datastore structure.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successful.
 * - @ref da_status_invalid_input - the handle pointer is invalid.
 */
da_status da_datastore_print_error_message(da_datastore store);
void da_datastore_destroy(da_datastore *store);

/**
 * @anchor da_datastore_destroy
 * @brief Free all memory linked to a @ref da_datastore
 *
 * @note Memory leaks can occur if @ref da_datastore structures are not destroyed after use.
 *
 * @param store the @ref da_datastore to destroy
 */
void da_datastore_destroy(da_datastore *store);

/**
 * @brief print The list and the values of all the optional parameter of a given @ref da_datastore.
 *
 * @param store the main @ref da_datastore
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successful.
 * - @ref da_status_invalid_input - the store passed as argument was not correctly initialized.
 */
da_status da_data_print_options(da_datastore store);

/**
 * @brief Concatenate horizontally two @ref da_datastore objects.
 *
 * The two data stores must have a matching number of rows to successfully perform this operation.
 *
 * If successful, on output, @p store1 will contain the concatenation of the 2 stores and store 2 will be empty.
 * No copy of the data is performed when this function is called.
 *
 * @param store1 the @ref da_datastore that will contain the concatenation on output.
 * @param store2 the @ref da_datastore to concatenate with @p store1. Will be empty on output.
 * @return da_status. The function returns:
 * - @ref da_status_success The operation was successful.
 * - @ref da_status_invalid_input - some of the input data was not correct.
 *        Use @ref da_handle_print_error_message to get more details.
 * - @ref da_status_invalid_pointer - the store was not correctly initialized.
 */
da_status da_data_hconcat(da_datastore *store1, da_datastore *store2);

/* ********************************** Load routines ********************************** */
/* *********************************************************************************** */
/** \{
 * @brief Load new columns into a @ref da_datastore from a dense matrix.
 * The last suffix of the function name marks the type of the data to add.
 *
 * The *da_data_load_col_X* routines will try to add columns to the right of the @ref da_datastore passed in argument.
 * If data was already loaded in the store, the number of rows of the new block must match
 * with the number of rows already present.
 *
 * The new data is expected to be provided as an @p n_rows @f$\times @f$ @p n_cols dense block and can be passed in row
 * major or column major ordering.
 *
 * The data provided can be optionally copied into the store (for non C-string data blocks)
 * if @p copy_data was set to true.
 * Warning: if @p copy_data is set to false, the pointer will be copied as it is provided.
 * Modifying or deallocating the memory before calling @ref da_datastore_destroy can create
 * unintended behaviour.
 *
 * @param[inout] store the main structure.
 * @param[in] n_rows number of rows of the new block.
 * @param[in] n_cols number of columns of the new block.
 * @param[in] block pointer to the raw data to add to the store.
 * @param[in] order a @ref da_ordering enumerated type.
  *                 Specifies if the data block was stored in a column or row major ordering.
 * @param[in] copy_data specifies if the data needs to be copied in the store.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successful.
 * - @ref da_status_invalid_input - some of the input data was not correct.
 *        Use @ref da_handle_print_error_message to get more details.
 * - @ref da_status_invalid_pointer - the store was not correctly initialized.
 * - @ref da_status_memory_error - internal memory allocation encountered a problem.
 */
da_status da_data_load_col_int(da_datastore store, da_int n_rows, da_int n_cols,
                               da_int *block, da_ordering order, da_int copy_data);
da_status da_data_load_col_str(da_datastore store, da_int n_rows, da_int n_cols,
                               const char **block, da_ordering order);
da_status da_data_load_col_real_d(da_datastore store, da_int n_rows, da_int n_cols,
                                  double *block, da_ordering order, da_int copy_data);
da_status da_data_load_col_real_s(da_datastore store, da_int n_rows, da_int n_cols,
                                  float *block, da_ordering order, da_int copy_data);
da_status da_data_load_col_uint8(da_datastore store, da_int n_rows, da_int n_cols,
                                 uint8_t *block, da_ordering order, da_int copy_data);
/** \} */

/** \{
 * @brief Load new rows into a @ref da_datastore from a dense matrix.
 * The last suffix of the function name marks the type of the data to add.
 *
 * @rst
 * The *da_data_load_row_X* routines will try to add rows at the bottom of the @ref da_datastore passed in argument.
 * If data was already loaded in the store, the routines must be called repeatedly until the columns of the new blocks
 * match the structure of the existing store (see :ref:`the introduction section <datastores_intro>` for more details on the stores structure).
 * @endrst
 *
 * The new data is expected to be provided as an @p n_rows @f$\times @f$ @p n_cols dense block and can be passed in row major
 * or column major ordering.
 *
 * The data provided can be optionally copied into the store (for non C-string data blocks)
 * if @p copy_data was set to true.
 * Warning: if @p copy_data is set to false, the pointer will be copied as it is provided.
 * modifying or deallocating the memory before calling @ref da_datastore_destroy can create
 * unintended behaviour.
 *
 * @param[inout] store the main structure.
 * @param[in] n_rows number of rows of the new block.
 * @param[in] n_cols number of columns of the new block.
 * @param[in] block pointer to the raw data to add to the store.
 * @param[in] order a @ref da_ordering enumerated type.
 *              Specifies if the data block was stored in a column or row major ordering.
 * @param[in] copy_data specifies if the data needs to be copied in the store.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successful.
 * - @ref da_status_invalid_input - some of the input data was not correct.
 *        Use @ref da_handle_print_error_message to get more details.
 * - @ref da_status_invalid_pointer - the store was not correctly initialized.
 * - @ref da_status_memory_error - internal memory allocation encountered a problem.
 */
da_status da_data_load_row_int(da_datastore store, da_int n_rows, da_int n_cols,
                               da_int *block, da_ordering order, da_int copy_data);
da_status da_data_load_row_str(da_datastore store, da_int n_rows, da_int n_cols,
                               const char **block, da_ordering order);
da_status da_data_load_row_real_d(da_datastore store, da_int n_rows, da_int n_cols,
                                  double *block, da_ordering order, da_int copy_data);
da_status da_data_load_row_real_s(da_datastore store, da_int n_rows, da_int n_cols,
                                  float *block, da_ordering order, da_int copy_data);
da_status da_data_load_row_uint8(da_datastore store, da_int n_rows, da_int n_cols,
                                 uint8_t *block, da_ordering order, da_int copy_data);
/** \} */

/**
 * @brief Read data from a CSV file into a @ref da_datastore object. The data type of each column will be automatically detected.
 *
 * Prior to calling this function, the standard CSV options can be set using calls to @ref da_options_set_int or @ref da_options_set_string.
 *
 * The following additional options can be set:
 *
 \rst
 .. csv-table:: @ref da_datastore file reading options
   :header: "Option Name", "Type", "Default", "Description", "Constraints"

   "CSV integers as floats", "da_int", ":math:`i = 0`", "Whether or not to interpret integers as floating point numbers when using autodetection", ":math:`0 \le i \le 1`"
   "CSV datastore precision", "string", ":math:`s =` `'double'`", "The precision used when reading floating point numbers using autodetection", ":math:`s =` `'double'`, or `'single'`"
   "CSV datatype", "string", ":math:`s =` `'auto'`", "If a CSV file is known to be of a single datatype, set this option to disable autodetection and make reading the file quicker", ":math:`s =` `'auto'`, `'boolean'`, `'double'`, `'float'`, `'integer'`, or `'string'`"
\endrst
 *
 * @param[in,out] store a @ref _da_datastore object, initialized using @ref da_datastore_init.
 * @param[in] filename the relative or absolute path to a file or stream that can be opened for reading.
 * \return @ref da_status. The function returns:
 * - @ref da_status_success
 * - @ref da_status_file_reading_error
 * - @ref da_status_parsing_error
 * - @ref da_status_missing_data
 */
da_status da_data_load_from_csv(da_datastore store, const char *filename);

/* ************************************* selection *********************************** */
/* *********************************************************************************** */
/**
 * @brief Select all columns indexed between the values @p lbound and @p ubound in the selection labeled by @p key.
 * Column indices are zero-based, meaning the index of the first column is 0 and the index of the last one is n_cols-1.
 *
 * Overlapping columns cannot be selected at this time.
 *
 * @param[inout] store the main data structure.
 * @param[in] key label of the selection.
 * @param[in] lbound lower bound of the column indices to select.
 * @param[in] ubound upper bound of the column indices to select.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successful.
 * - @ref da_status_invalid_input - some of the input data was not correct.
 *        Use @ref da_handle_print_error_message to get more details.
 * - @ref da_status_invalid_pointer - the store was not correctly initialized.
 */
da_status da_data_select_columns(da_datastore store, const char *key, da_int lbound,
                                 da_int ubound);
/**
 * @brief Select all rows indexed between the values @p lbound and @p ubound in the selection labeled by @p key.
 * Row indices are zero-based, meaning the index of the first row is 0 and the index of the last one is n_rows-1.
 *
 * @param[inout] store the main data structure.
 * @param[in] key label of the selection.
 * @param[in] lbound lower bound of the row indices to select.
 * @param[in] ubound upper bound of the row indices to select.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success The operation was successful.
 * - @ref da_status_invalid_input - some of the input data was not correct.
 *        Use @ref da_handle_print_error_message to get more details.
 * - @ref da_status_invalid_pointer - the store was not correctly initialized.
 */
da_status da_data_select_rows(da_datastore store, const char *key, da_int lbound,
                              da_int ubound);
/**
 * @brief Select all rows indexed between the values @p row_lbound and @p row_ubound and
 * all columns indexed between the values @p col_lbound and @p col_ubound in the selection labeled by @p key.
 * Row and Column indices are zero-based.
 *
 * @param[inout] store the main data structure.
 * @param[in] key label of the selection.
 * @param[in] row_lbound lower bound of the row indices to select.
 * @param[in] row_ubound upper bound of the row indices to select.
 * @param[in] col_lbound lower bound of the column indices to select.
 * @param[in] col_ubound upper bound of the column indices to select.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successful.
 * - @ref da_status_invalid_input - some of the input data was not correct.
 *        Use @ref da_handle_print_error_message to get more details.
 * - @ref da_status_invalid_pointer - the store was not correctly initialized.
 */
da_status da_data_select_slice(da_datastore store, const char *key, da_int row_lbound,
                               da_int row_ubound, da_int col_lbound, da_int col_ubound);
/**
 * @brief Remove all the rows containing missing data from the selection labeled by @p key.
 * See @ref da_datastore for more details on selections.
 *
 * If @p key had a previously empty row selection, all rows are checked for missing elements.
 *
 * @param[inout] store the main data structure.
 * @param[in] key the label of the selection
 * @param[in] full_rows serves as a boolean variable indicating if only the columns already
 * in the selection key are to be checked for missing data.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success The operation was successful.
 * - @ref da_status_invalid_input - some of the input data was not correct.
 *        Use @ref da_handle_print_error_message to get more details.
 * - @ref da_status_invalid_pointer - the store was not correctly initialized.
 */
da_status da_data_select_non_missing(da_datastore store, const char *key,
                                     uint8_t full_rows);

/* ********************************** extract columns ******************************** */
/* *********************************************************************************** */
/** \{ */
/**
 * @brief Extract a column from a store into a pre-allocated array.
 * The last suffix of the function name marks the type of the data to add.
 *
 *  @p dim is the size of the output array provided to the function and must
 *  be at least the number or rows in the store.
 *
 * @param[in] store main data structure.
 * @param[in] idx index of the column to extract.
 * @param[in] dim size of the vector provided.
 * @param[out] col array the column will be exported to.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successful.
 * - @ref da_status_invalid_input - some of the input data was not correct.
 *        Use @ref da_handle_print_error_message to get more details.
 * - @ref da_status_invalid_pointer - the store was not correctly initialized.
 * - @ref da_status_missing_block - the store contains incomplete row blocks.
 * - @ref da_status_internal_error - an unexpected error occurred.
 */
da_status da_data_extract_column_int(da_datastore store, da_int idx, da_int dim,
                                     da_int *col);
da_status da_data_extract_column_real_s(da_datastore store, da_int idx, da_int dim,
                                        float *col);
da_status da_data_extract_column_real_d(da_datastore store, da_int idx, da_int dim,
                                        double *col);
da_status da_data_extract_column_uint8(da_datastore store, da_int idx, da_int dim,
                                       uint8_t *col); // For boolean data
da_status da_data_extract_column_str(da_datastore store, da_int idx, da_int dim,
                                     char **col);

/** \} */
/* ********************************* extract selections ****************************** */
/* *********************************************************************************** */
/** \{ */
/**
 * @brief Extract a selection labeled by @p key.
 * The last suffix of the function name marks the type of the data to be extracted.
 *
 * The data marked by the set of columns and rows in the selection @p key is extracted into
 * a dense matrix of the corresponding type. The matrix is returned in column major ordering.
 *
 * @param[in] store main data structure.
 * @param[in] key label of the selection.
 * @param[out] data output matrix
 * @param[in] lddata leading dimension of the output data
 * @return @ref da_status. The function returns:
 * - @ref da_status_success the operation was successful.
 * - @ref da_status_invalid_input - some of the input data was not correct.
 *        Use @ref da_handle_print_error_message to get more details.
 * - @ref da_status_invalid_pointer - the store was not correctly initialized.
 * - @ref da_status_missing_block - the store contains incomplete row blocks.
 * - @ref da_status_internal_error - an unexpected error occurred.
 */
da_status da_data_extract_selection_int(da_datastore store, const char *key, da_int *data,
                                        da_int lddata);
da_status da_data_extract_selection_real_d(da_datastore store, const char *key,
                                           double *data, da_int lddata);
da_status da_data_extract_selection_real_s(da_datastore store, const char *key,
                                           float *data, da_int lddata);
da_status da_data_extract_selection_uint8(da_datastore store, const char *key,
                                          uint8_t *data, da_int lddata);
/** \} */

/* ************************************* headings ************************************ */
/* *********************************************************************************** */
/**
 * @brief Label a column
 *
 * @param[inout] store main data structure.
 * @param[in] label new label for the column @p idx.
 * @param[in] col_idx index of the column to label.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successful.
 * - @ref da_status_invalid_input - some of the input data was not correct.
 *        Use @ref da_handle_print_error_message to get more details.
 * - @ref da_status_invalid_pointer - the store was not correctly initialized.
 * - @ref da_status_internal_error - an unexpected error occurred.
 */
da_status da_data_label_column(da_datastore store, const char *label, da_int col_idx);
/**
 * @brief Get the index of the column labeled by the input @p label.
 *
 * @param[in] store main data store.
 * @param[in] label name of the column to look for.
 * @param[out] col_idx on output contains the index  of the column @p label.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successful.
 * - @ref da_status_invalid_input - some of the input data was not correct.
 *        Use @ref da_handle_print_error_message to get more details.
 * - @ref da_status_invalid_pointer - the store was not correctly initialized.
 * - @ref da_status_internal_error - an unexpected error occurred.
 */
da_status da_data_get_col_idx(da_datastore store, const char *label, da_int *col_idx);
/**
 * @brief Get the label of a column from its index.
 *
 * On output the C string @p label will contain the label of the column @p idx. @p label_sz
 * indicates the size of the C string @p label. If it is smaller than the size
 * of the column label, the function will return @ref da_status_invalid_input and @p label_sz
 * will be set to the minimum size acceptable for @p col_idx.
 *
 * @param[in] store main data structure .
 * @param[in] col_idx index of the column to search for.
 * @param[in] label_sz the size of the C string being provided to the function.
 * @param[out] label if successful, contains the label of the column @p idx on output.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successful.
 * - @ref da_status_invalid_input - some of the input data was not correct.
 *        Use @ref da_handle_print_error_message to get more details.
 * - @ref da_status_invalid_pointer - the store was not correctly initialized.
 */
da_status da_data_get_col_label(da_datastore store, da_int col_idx, da_int *label_sz,
                                char *label);

/* ********************************** setters/getters ******************************** */
/* *********************************************************************************** */
/** \{ */
/**
 * @brief Get the number of rows in the store.
 *
 * @param[in] store main data structure.
 * @param[out] n_rows contains the number of rows in @p store on output.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successful.
 * - @ref da_status_invalid_input - some of the input data was not correct.
 *     Use @ref da_handle_print_error_message to get more details.
 * - @ref da_status_invalid_pointer - the store was not correctly initialized.
 */
da_status da_data_get_n_rows(da_datastore store, da_int *n_rows);
/** \} */

/** \{ */
/**
 * @brief Get the number of rows in the store.
 *
 * @param[in] store main data structure.
 * @param[out] n_cols contains the number of columns in @p store on output.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successful.
 * - @ref da_status_invalid_input - some of the input data was not correct.
 *     Use @ref da_handle_print_error_message to get more details.
 * - @ref da_status_invalid_pointer - the store was not correctly initialized.
 */
da_status da_data_get_n_cols(da_datastore store, da_int *n_cols);
/** \} */

/** \{ */
/**
 * @brief Get an individual element of a data store.
 * The last suffix of the function name marks the type of the data to be extracted.
 *
 * @param[in] store main data structure.
 * @param[in] i index of the row of the element to look for.
 * @param[in] j index of the column of the element to look for.
 * @param[out] elem contains the value of the element at indices @p i, @p j on output.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successful.
 * - @ref da_status_invalid_input - some of the input data was not correct.
 *        Use @ref da_handle_print_error_message to get more details.
 * - @ref da_status_invalid_pointer - the store was not correctly initialized.
 */
da_status da_data_get_element_int(da_datastore store, da_int i, da_int j, da_int *elem);
da_status da_data_get_element_real_d(da_datastore store, da_int i, da_int j,
                                     double *elem);
da_status da_data_get_element_real_s(da_datastore store, da_int i, da_int j, float *elem);
da_status da_data_get_element_uint8(da_datastore store, da_int i, da_int j,
                                    uint8_t *elem);
/** \} */

/** \{ */
/**
 * @brief Set an individual element of a data store to a new value.
 * The last suffix of the function name marks the type of the data to be set.
 *
 * @param[inout] store main data structure.
 * @param[in] i index of the row of the element to modify.
 * @param[in] j index of the column of the element to modify.
 * @param[in] elem the new value for the element at index @p i, @p j.
 * @return @ref da_status. The function returns:
 * - @ref da_status_success - the operation was successful.
 * - @ref da_status_invalid_input - some of the input data was not correct.
 *        Use @ref da_handle_print_error_message to get more details.
 * - @ref da_status_invalid_pointer - the store was not correctly initialized.
 */
da_status da_data_set_element_int(da_datastore store, da_int i, da_int j, da_int elem);
da_status da_data_set_element_real_d(da_datastore store, da_int i, da_int j, double elem);
da_status da_data_set_element_real_s(da_datastore store, da_int i, da_int j, float elem);
da_status da_data_set_element_uint8(da_datastore store, da_int i, da_int j, uint8_t elem);
/** \} */

#ifdef __cplusplus
}
#endif
#endif
