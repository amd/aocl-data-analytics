.. _data-management:

Data management Functionalities
*******************************

Reading and loading data
========================

TODO short intro

CSV files
---------

These routines read data of a single type from a CSV (comma separated values) file into an array. In addition, a character array of column headings can optionally be read.

The routines take a :cpp:type:`da_datastore` struct as their first argument, which must be initialized prior to the routine call
using :cpp:func:`da_datastore_init`. This is used to store options (see below) but is not used to store the actual CSV data, which is instead returned in an array. If you wish to load data driectly from the CSV file to the :cpp:type:`da_datastore` struct, then use :cpp:func:`da_data_load_from_csv`.

:ref:`API documentation. <csv_api>`

CSV Option setting
^^^^^^^^^^^^^^^^^^

Prior to reading the CSV file, various options can be set by passing the :cpp:type:`da_datastore` struct to :cpp:func:`da_options_set_int` or :cpp:func:`da_options_set_string` for integer or string options respectively.

The following string options can be set:

- *CSV delimiter* - specify the delimiter used when reading CSV files.

- *CSV thousands* - specify the character used to separate thousands when reading numeric values in CSV files.

- *CSV decimal* - specify which character denotes a decimal point in CSV files.

- *CSV comment* - specify which character is used to denote comments in CSV files (note, if a line in a CSV file is to be interpreted as only containing a comment, the comment character should be the first character on  the line).

- *CSV quote character* - specify which character is used to denote quotations in CSV files.

- *CSV escape character* - specify the escape character in CSV files.

- *CSV line terminator* - specify which character is used to denote line termination in CSV files (leave this empty to use the default).

- *CSV scientific notation character* - specify which character is used to denote powers of 10 in floating point values in CSV files.

- *CSV skip rows* - a comma- or space-separated list of rows to ignore in CSV files.

Note that, with the exception of the *'CSV skip rows'* option, only single characters can be used in the options above.

The following :ref:`da_int` options can be set:

- *CSV double quote* - specify whether or not to interpret two consecutive quote characters within a field as a single quote character. This option can only take the values 0 or 1.

- *CSV whitespace delimiter* - specify whether or not to use whitespace as the delimiter when reading CSV files. This option can only take the values 0 or 1.

- *CSV row start* - ignore the specified number of rows from the top of the CSV file (note that line numbers in CSV files start at 1).

- *CSV skip empty lines* - specify whether or not to ignore empty lines in CSV files. This option can only take the values 0 or 1. Note that caution should be used when using this in conjunction with options such as *CSV skip rows* since line numbers may no longer correspond to the original line numbers in the CSV file.

- *CSV skip initial space* - specify whether or not to ignore initial spaces in CSV file lines. This option can only take the values 0 or 1.

- *CSV skip footer* - specify whether or not to ignore the last line when reading a CSV file. This option can only take the values 0 or 1.

- *CSV warn for missing data* - if set to 0 then return an error if missing data is encountered; if set to 1, issue a warning and store missing data as either a NaN (for floating point data) or the maximum value of the integer type being used.

- *CSV use headings* - specify whether or not to interpret the first line as a headings row. This option can only take the values 0 or 1.

CSV Examples
^^^^^^^^^^^^

Various files in the examples folder of your installation demonstrate the use of these functions (e.g. ``basic_pca.cpp`` and ``linear_model.cpp``).


.. _datastores_intro:

Data handling: Datastores
=========================

TODO. This is placeholder text to enable datastore references to be inserted in the meantime.

:ref:`API documentation. <datastore_api>`

API documentation
=================

.. _csv_api:

CSV functions
-------------

.. doxygenfunction:: da_read_csv_d
.. doxygenfunction:: da_read_csv_s
.. doxygenfunction:: da_read_csv_int
.. doxygenfunction:: da_read_csv_uint8
.. doxygenfunction:: da_read_csv_char

For reading data directly into a :cpp:type:`da_datastore` struct, see :cpp:func:`da_data_load_from_csv`


.. _datastore_api:

Datastores
----------

.. doxygentypedef:: da_datastore

Initialize and destroy datastores
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: da_datastore_init
.. doxygenfunction:: da_datastore_destroy


Load data into a datastore
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: da_data_load_from_csv
.. doxygenfunction:: da_data_load_row_int
.. doxygenfunction:: da_data_load_row_str
.. doxygenfunction:: da_data_load_row_real_d
.. doxygenfunction:: da_data_load_row_real_s
.. doxygenfunction:: da_data_load_row_uint8
.. doxygenfunction:: da_data_load_col_int
.. doxygenfunction:: da_data_load_col_str
.. doxygenfunction:: da_data_load_col_real_d
.. doxygenfunction:: da_data_load_col_real_s
.. doxygenfunction:: da_data_load_col_uint8

Data selection
^^^^^^^^^^^^^^

.. doxygenfunction:: da_data_select_columns
.. doxygenfunction:: da_data_select_rows
.. doxygenfunction:: da_data_select_slice
.. doxygenfunction:: da_data_select_non_missing

Data extraction
^^^^^^^^^^^^^^^

.. doxygenfunction:: da_data_extract_selection_int
.. doxygenfunction:: da_data_extract_selection_real_d
.. doxygenfunction:: da_data_extract_selection_real_s
.. doxygenfunction:: da_data_extract_selection_uint8
.. doxygenfunction:: da_data_extract_column_int
.. doxygenfunction:: da_data_extract_column_real_s
.. doxygenfunction:: da_data_extract_column_real_d
.. doxygenfunction:: da_data_extract_column_uint8
.. doxygenfunction:: da_data_extract_column_str

Column headers
^^^^^^^^^^^^^^

.. doxygenfunction:: da_data_label_column
.. doxygenfunction:: da_data_get_col_idx
.. doxygenfunction:: da_data_get_col_label

Data edition
^^^^^^^^^^^^

.. doxygenfunction:: da_data_get_num_rows
.. doxygenfunction:: da_data_get_num_cols
.. doxygenfunction:: da_data_get_element_int
.. doxygenfunction:: da_data_get_element_real_d
.. doxygenfunction:: da_data_get_element_real_s
.. doxygenfunction:: da_data_get_element_uint8
.. doxygenfunction:: da_data_set_element_int
.. doxygenfunction:: da_data_set_element_real_d
.. doxygenfunction:: da_data_set_element_real_s
.. doxygenfunction:: da_data_set_element_uint8