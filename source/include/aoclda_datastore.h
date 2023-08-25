#ifndef AOCLDA_DATASTORE_H
#define AOCLDA_DATASTORE_H

#include "aoclda_error.h"
#include "aoclda_handle.h"
#include "aoclda_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The main structure storing the data. 
 * All functions of this chapter operate on this internal data structure.
 */
typedef struct _da_datastore *da_datastore;

/**
 * \brief Initialize an empty \ref datastore.
 * 
 * @param store The \ref da_datastore to initialize
 */
da_status da_datastore_init(da_datastore *store);
void da_datastore_destroy(da_datastore *store);


/**
 * @anchor da_datastore_destroy
 * @brief Free all memory linked to a \ref da_datastore 
 * 
 * @note Memory leaks can occur if store are not destroyed after use.
 * 
 * @param store The \ref da_datastore to destroy
 */
void da_datastore_destroy(da_datastore *store);


/**
 * @brief print The list and the values of all the optional parameter of a given datastore.
 * 
 * @param store The main datastore
 * @return \ref da_status.
 * - \ref da_status_success The operation was successful.
 * - \ref da_status_invalid_input The store passed as argument was not correctly initialized.
 */
da_status da_data_print_options(da_datastore store);

da_status da_data_hconcat(da_datastore *store1, da_datastore *store2);

/* ********************************** Load routines ********************************** */
/* *********************************************************************************** */
/** \{
 * @brief Load new columns into a datastore from a dense matrix. 
 * The last suffix of the function name marks the type of the data to add. 
 * 
 * The @a da_data_load_col_X routines will try to add columns to the right of the datastore passed in argument.
 * If data was already loaded in the store, the number of rows of the new block must match
 * with the number of rows already present. 
 * 
 * The new data is expected to be provided as an \p m * \p n dense block and can be passed in row
 * major or column major ordering.
 * 
 * The data provided can be optionally copied into the store (for non C-string data blocks) 
 * if \p copy_data was set to true.
 * Warning: if \p copy_data is set to false, the pointer will be copied as it is provided. 
 * modifying or deallocating the memory before calling \ref da_datastore_destroy can create 
 * unintended behaviour. 
 * 
 * @param[inout] store The main structure.
 * @param[in] m Number of rows of the new block.
 * @param[in] n Number of columns of the new block. 
 * @param[in] int_block Pointer to the raw data to add to the store [integer].  
 * @param[in] str_block Pointer to the raw data to add to the store [C string].  
 * @param[in] real_block Pointer to the raw data to add to the store [real].  
 * @param[in] uint_block Pointer to the raw data to add to the store [uint].  
 * @param[in] order a \ref da_ordering enumerated type. 
 *              Specifies if the data block was stored in a column or row major ordering. 
 * @param[in] copy_data Specifies if the data needs to be copied in the store. 
 * @return \ref da_status.
 * - \ref da_status_success The operation was successful.
 * - \ref da_status_invalid_input Some of the input data was not correct. 
 *        Use da_handle_print_error_message to get more details. 
 * - \ref da_status_invalid_pointer The store was not correctly initialized
 * - \ref da_status_memory_error Internal memory allocation encountered a problem. 
 */
da_status da_data_load_col_int(da_datastore store, da_int m, da_int n, da_int *int_block,
                               da_ordering order, da_int copy_data);
da_status da_data_load_col_str(da_datastore store, da_int m, da_int n,
                               const char **str_block, da_ordering order);
da_status da_data_load_col_real_d(da_datastore store, da_int m, da_int n,
                                  double *real_block, da_ordering order,
                                  da_int copy_data);
da_status da_data_load_col_real_s(da_datastore store, da_int m, da_int n,
                                  float *real_block, da_ordering order, da_int copy_data);
da_status da_data_load_col_uint8(da_datastore store, da_int m, da_int n,
                                 uint8_t *uint_block, da_ordering order,
                                 da_int copy_data);
/** \} */

/** \{
 * @brief Load new rows into a datastore from a dense matrix. 
 * The last suffix of the function name marks the type of the data to add. 
 * 
 * @rst
 * The *da_data_load_row_X* routines will try to add rows at the bottom of the datastore passed in argument.
 * If data was already loaded in the store, the routines must be called repeatedly until the columns of the new blocks
 * match the structure of the existing store (see :ref:`the introduction section <datastores_intro>` for more details on the stores structure).
 * @endrst
 *  
 * The new data is expected to be provided as an \p m * \p n dense block and can be passed in row major
 * or column major ordering.
 * 
 * The data provided can be optionally copied into the store (for non C-string data blocks) 
 * if \p copy_data was set to true.
 * Warning: if \p copy_data is set to false, the pointer will be copied as it is provided. 
 * modifying or deallocating the memory before calling \ref da_datastore_destroy can create 
 * unintended behaviour. 
 * 
 * @param[inout] store The main structure.
 * @param[in] m Number of rows of the new block.
 * @param[in] n Number of columns of the new block. 
 * @param[in] int_block Pointer to the raw data to add to the store [integer].  
 * @param[in] str_block Pointer to the raw data to add to the store [C string].  
 * @param[in] real_block Pointer to the raw data to add to the store [real].  
 * @param[in] uint_block Pointer to the raw data to add to the store [uint].  
 * @param[in] order a \ref da_ordering enumerated type. 
 *              Specifies if the data block was stored in a column or row major ordering. 
 * @param[in] copy_data Specifies if the data needs to be copied in the store. 
 * @return \ref da_status.
 * - \ref da_status_success The operation was successful.
 * - \ref da_status_invalid_input Some of the input data was not correct. 
 *        Use da_handle_print_error_message to get more details. 
 * - \ref da_status_invalid_pointer The store was not correctly initialized
 * - \ref da_status_memory_error Internal memory allocation encountered a problem. 
 */
da_status da_data_load_row_int(da_datastore store, da_int m, da_int n, da_int *int_block,
                               da_ordering order, da_int copy_data);
da_status da_data_load_row_str(da_datastore store, da_int m, da_int n,
                               const char **str_block, da_ordering order);
da_status da_data_load_row_real_d(da_datastore store, da_int m, da_int n,
                                  double *real_block, da_ordering order,
                                  da_int copy_data);
da_status da_data_load_row_real_s(da_datastore store, da_int m, da_int n,
                                  float *real_block, da_ordering order, da_int copy_data);
da_status da_data_load_row_uint8(da_datastore store, da_int m, da_int n,
                                 uint8_t *uint_block, da_ordering order,
                                 da_int copy_data);
/** \} */

/**
 * \brief Read data from a CSV file into a \ref da_datastore object. The data type of each column will be automatically detected.
 * 
 * Prior to calling this function, the standard CSV options can be set using calls to \ref da_options_set_int or \ref da_options_set_string.
 * 
 * The following additional options can be set using \ref da_options_set_string.
 * 
 * - <em>CSV datatype</em> - if the CSV file is known to be of a single datatype, set this option to disable autodetection and make reading the file quicker. The allowed values are: \a float, \a double, \a integer, \a string and \a boolean, with a default of \a auto to use autodetection.
 * - <em>CSV datastore precision</em> - select the precision when reading floating point numbers. The allowed values are \a double (default) or \a single.
 *
 * The following additional option can be set using \ref da_options_set_int.
 * 
 * - <em>CSV integers as floats</em> - whether or not to interpret integers as floating point numbers when using autodetection. This option can take the values 0 or 1.
 * 
 * \param[in,out] store a \ref _da_datastore object, initialized using \ref da_datastore_init.
 * \param[in] filename the relative or absolute path to a file or stream that can be opened for reading.
 * \return \ref da_status.
 * - \ref da_status_success
 * - \ref da_status_file_reading_error
 * - \ref da_status_parsing_error
 * - \ref da_status_bad_lines
 * - \ref da_status_missing_data
 */
da_status da_data_load_from_csv(da_datastore store, const char *filename);


/* ************************************* selection *********************************** */
/* *********************************************************************************** */
/**
 * @brief Select all columns indexed between the values \p lbound and \p ubound in the selection labeled by \p key.
 * 
 * Overlapping columns cannot be selected at this time.
 * 
 * @param[inout] store the main data structure.
 * @param[in] key label of the selection.
 * @param[in] lbound lower bound of the column indices to select. 
 * @param[in] ubound upper bound of the column indices to select.
 * @return \ref da_status.
 * - \ref da_status_success The operation was successful.
 * - \ref da_status_invalid_input Some of the input data was not correct. 
 *        Use da_handle_print_error_message to get more details. 
 * - \ref da_status_invalid_pointer The store was not correctly initialized
 */
da_status da_data_select_columns(da_datastore store, const char *key, da_int lbound,
                                 da_int ubound);
/**
 * @brief Select all rows indexed between the values \p lbound and \p ubound in the selection labeled by \p key.
 * 
 * Overlapping rows cannot be selected at this time.
 * 
 * @param[inout] store the main data structure.
 * @param[in] key label of the selection.
 * @param[in] lbound lower bound of the row indices to select. 
 * @param[in] ubound upper bound of the row indices to select. 
 * @return \ref da_status. 
 * - \ref da_status_success The operation was successful.
 * - \ref da_status_invalid_input Some of the input data was not correct. 
 *        Use da_handle_print_error_message to get more details. 
 * - \ref da_status_invalid_pointer The store was not correctly initialized
 */
da_status da_data_select_rows(da_datastore store, const char *key, da_int lbound,
                              da_int ubound);
/**
 * @brief Select all rows indexed between the values \p row_lbound and \p row_ubound and
 * all columns indexed between the values \p col_lbound and \p col_ubound in the selection labeled by \p key.
 * 
 * Overlapping rows and columns cannot be selected at this time. 
 * 
 * @param[inout] store The main data structure.
 * @param[in] key label of the selection.
 * @param[in] row_lbound lower bound of the row indices to select. 
 * @param[in] row_ubound upper bound of the row indices to select. 
 * @param[in] col_lbound lower bound of the column indices to select. 
 * @param[in] col_ubound upper bound of the column indices to select.
 * @return \ref da_status. 
 * - \ref da_status_success The operation was successful.
 * - \ref da_status_invalid_input Some of the input data was not correct. 
 *        Use da_handle_print_error_message to get more details. 
 * - \ref da_status_invalid_pointer The store was not correctly initialized
 */
da_status da_data_select_slice(da_datastore store, const char *key, da_int row_lbound,
                               da_int row_ubound, da_int col_lbound, da_int col_ubound);
/**
 * @brief Remove all the rows containing missing data from the selection labeled by \p key.
 * See \ref da_datastore for more details on selections. 
 * 
 * If no rows were selected, before in selection \p key, all rows are checked.
 * 
 * @param[inout] store The main data structure.
 * @param[in] key The label of the selection
 * @param[in] full_rows serves as a boolean variable indicating if only the columns already
 * in the selection key are to be checked for missing data. 
 * @return \ref da_status. 
 * - \ref da_status_success The operation was successful.
 * - \ref da_status_invalid_input Some of the input data was not correct. 
 *        Use da_handle_print_error_message to get more details. 
 * - \ref da_status_invalid_pointer The store was not correctly initialized
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
 *  \p m is the size of the output array provided to the function and must
 *  be at least the number or rows in the store. 
 * 
 * @param[in] store Main data structure. 
 * @param[in] idx Index of the column to extract. 
 * @param[in] m Size of the vector provided.
 * @param[out] col Array the column will be exported to.
 * @return \ref da_status. 
 * - \ref da_status_success The operation was successful.
 * - \ref da_status_invalid_input Some of the input data was not correct. 
 *        Use da_handle_print_error_message to get more details. 
 * - \ref da_status_invalid_pointer The store was not correctly initialized
 * - \ref da_status_missing_block The store contains incomplete row blocks 
 * - \ref da_status_internal_error An unexpected error occured. 
 */
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

/** \} */
/* ********************************* extract selections ****************************** */
/* *********************************************************************************** */
/** \{ */
/**
 * @brief Extract a selection labeled by \p key. 
 * The last suffix of the function name marks the type of the data to be extracted.
 * 
 * The data marked by the set of columns and rows in the selection \p key is extracted into 
 * a dense matrix of the corresponding type. The matrix is returned in column major ordering. 
 * 
 * @param[in] store Main data structure.
 * @param[in] key Label of the selection.
 * @param[in] ld Leading dimension of the output data
 * @param[out] data Output matrix  
 * @return \ref da_status. 
 * - \ref da_status_success The operation was successful.
 * - \ref da_status_invalid_input Some of the input data was not correct. 
 *        Use da_handle_print_error_message to get more details. 
 * - \ref da_status_invalid_pointer The store was not correctly initialized
 * - \ref da_status_missing_block The store contains incomplete row blocks 
 * - \ref da_status_internal_error An unexpected error occured.  
 */
da_status da_data_extract_selection_int(da_datastore store, const char *key, da_int ld,
                                        da_int *data);
da_status da_data_extract_selection_real_d(da_datastore store, const char *key, da_int ld,
                                           double *data);
da_status da_data_extract_selection_real_s(da_datastore store, const char *key, da_int ld,
                                           float *data);
da_status da_data_extract_selection_uint8(da_datastore store, const char *key, da_int ld,
                                          uint8_t *data);
/** \} */

/* ************************************* headings ************************************ */
/* *********************************************************************************** */
/**
 * @brief label a column
 * 
 * @param[inout] store Main data structure.
 * @param[in] label New label for the column \p idx.
 * @param[in] col_idx Index of the column to label.
 * @return \ref da_status. 
 * - \ref da_status_success The operation was successful.
 * - \ref da_status_invalid_input Some of the input data was not correct. 
 *        Use da_handle_print_error_message to get more details. 
 * - \ref da_status_invalid_pointer The store was not correctly initialized
 * - \ref da_status_internal_error An unexpected error occured. 
 */
da_status da_data_label_column(da_datastore store, const char *label, da_int col_idx);
/**
 * @brief Get the index of the column labeled by the input \p label.
 * 
 * @param[in] store Main data store.
 * @param[in] label Name of the column to look for.
 * @param[out] col_idx On output contains the index  of the column \p label.
 * @return \ref da_status. 
 * - \ref da_status_success The operation was successful.
 * - \ref da_status_invalid_input Some of the input data was not correct. 
 *        Use da_handle_print_error_message to get more details. 
 * - \ref da_status_invalid_pointer The store was not correctly initialized
 * - \ref da_status_internal_error An unexpected error occured. 
 */
da_status da_data_get_col_idx(da_datastore store, const char *label, da_int *col_idx);
/**
 * @brief Get the label of a column from its index.
 * 
 * On output the C string \p label will contain the label of the column \p idx. \p label_sz 
 * Indicates the size of the buffer \p label being provided. If it is smaller than the size 
 * of the column label, the function will return da_status_invalid_input and \p label_sz 
 * will be set to the minimum size acceptable for the column idx. 
 * 
 * @param[in] store Main data structure .
 * @param[in] col_idx index of the column to search for.
 * @param[in] label_sz The size of the C string being provided to the function.
 * @param[out] label if successful, contains the label of the column \p idx on output. 
 * @return \ref da_status. 
 * - \ref da_status_success The operation was successful.
 * - \ref da_status_invalid_input Some of the input data was not correct. 
 *        Use da_handle_print_error_message to get more details. 
 * - \ref da_status_invalid_pointer The store was not correctly initialized
 */
da_status da_data_get_col_label(da_datastore store, da_int col_idx, da_int *label_sz,
                                char *label);

/* ********************************** setters/getters ******************************** */
/* *********************************************************************************** */
/** \{ */
/**
 * @brief Get the number of rows in the store.
 * 
 * @param[in] store Main data structure.  
 * @param[out] num_rows Contains the number of rows in \p store on output.
 * @return \ref da_status. 
 * - \ref da_status_success The operation was successful.
 * - \ref da_status_invalid_input Some of the input data was not correct. 
 *     Use da_handle_print_error_message to get more details. 
 * - \ref da_status_invalid_pointer The store was not correctly initialized
 */
da_status da_data_get_num_rows(da_datastore store, da_int *num_rows);
da_status da_data_get_num_cols(da_datastore store, da_int *num_cols);
/** \} */

/** \{ */
/**
 * @brief Get an individual element of a data store.
 * The last suffix of the function name marks the type of the data to be extracted.
 * 
 * @param[in] store Main data structure.
 * @param[in] i index of the row of the element to look for.
 * @param[in] j index of the column of the element to look for. 
 * @param[out] elem Contains the value of the element at indices \p i, \p j on output.
 * @return \ref da_status. 
 * - \ref da_status_success The operation was successful.
 * - \ref da_status_invalid_input Some of the input data was not correct. 
 *        Use da_handle_print_error_message to get more details. 
 * - \ref da_status_invalid_pointer The store was not correctly initialized
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
 * @param[inout] store Main data structure.
 * @param[in] i index of the row of the element to modify.
 * @param[in] j index of the column of the element to modify. 
 * @param[in] elem The new value for the element at index \p i, \p j.
 * @return \ref da_status. 
 * - \ref da_status_success The operation was successful.
 * - \ref da_status_invalid_input Some of the input data was not correct. 
 *        Use da_handle_print_error_message to get more details. 
 * - \ref da_status_invalid_pointer The store was not correctly initialized
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