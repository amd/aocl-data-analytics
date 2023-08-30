.. _data-management:

Data management Functionalities
*******************************

Introduction
============

AOCL-DA provides a suite of functions designed to allow loading and manipulating data in a straigtforward manner before calling the algorithms of the library. All functions in this suite act on a single structure :cpp:type:`da_datastore`.

There are two ways to load data in the library: 

- directly :ref:`into an array <data_array_intro>` of pre-allocated memory of the correct type, or
- into an AOCL-DA :ref:`data store <datastores_intro>`, containing more functionalities to clean and manipulate the data before being passed to other algorithms of the library.

At the current version of AOCL-DA, the only supported format to load data from hard memory is CSV (comma separated values). 


.. _data_array_intro:

Loading data from hard memory to dense arrays
=============================================

CSV files
---------

These routines read data of a single type from a CSV (comma separated values) file into an array. In addition, a character array of column headings can optionally be read.

The routines take a :cpp:type:`da_datastore` struct as their first argument, which must be initialized prior to the routine call
using :cpp:func:`da_datastore_init`. This is used to store options (see below) but is not used to store the actual CSV data, which is instead returned in an array. If you wish to load data driectly from the CSV file to the :cpp:type:`da_datastore` struct, then use :cpp:func:`da_data_load_from_csv`.

For more details on each of the available functions, see the :ref:`API documentation. <csv_api>`

.. _datastores_intro:

Data manipulation : Data stores
===============================

:cpp:type:`da_datastore` can also be used to load and manipulate data. The life cycle of a data store typically follow these steps:

- :ref:`Initialize <api_init>` the data store structures.
- :ref:`Load <api_load_data>` data into the data stores, from pre-allocated memory, CSV files or other data stores.
- :ref:`Edit the data <api_data_editon>`.
- :ref:`Select <api_data_selection>` subsection of the store.
- :ref:`Extract data <api_data_extraction>`, either columns or sepcific selections.
- :ref:`Cleanly destroy <api_init>` the structure.

For more details on each of the available functions, see the :ref:`API documentation. <datastore_api>`


Loading data into a datastore
-----------------------------

Loading data into a data store can be done by adding blocks from different sources. A typical example would be to load data from a file and add columns that were allocated dynamically in your program. This can be achived by calling :cpp:func:`da_data_load_from_csv` and :cpp:func:`da_data_load_col_int` consecucutively for example.

When calling any of the ``da_data_load_*`` functions on a data store that is not empty, certain constraints must be respected:

- While adding columns, the number of rows in the block to be added must match with the current number of rows present in the data store (:cpp:func:`da_data_get_num_rows` can be used to query the dimension).
- New rows can be added in several sub-blocks. However:
    - the data store will be locked until the current number of columns in the store matches with the number of columns of the new block
    - Each sub-block has a minimum column size determined by the number of consecutive columns of the same type in the store. For example, if a given store already has 2 integer columns and a float column, new rows can be added in 2 sub-blocks (one with 2 integer columns and one with the remaining float column).

The last way to load data into a a given store is from another data store. Calling :cpp:func:`da_data_hconcat` will concatenate horizontally 2 data stores with matching number of rows. 


Selecting and extracting data
-----------------------------

**Selections**

In data stores, *selections* are defined by a label and a set of column and row indices. If any of the functions in :ref:`this subection <api_data_selection>` are called with a label that does not exist, a new selection is added to the store. Any number of selections can be defined at the same time in a given store.

:cpp:func:`da_data_select_columns`, :cpp:func:`da_data_select_rows` and :cpp:func:`da_data_select_slice` can be used to add respectively a set of column and row indices to a given selection label while :cpp:func:`da_data_select_non_missing` will remove all row indices containing missing data from the selection.

**Extraction**

There are 2 ways to :ref:`extract data<api_data_extraction>` from a data store:

- Extract a specific column with one of the ``da_data_extract_column_*`` functions.
- Extract a selection with a given label by calling one of the ``da_data_extract_selection_*`` functions.

All extracted data will be given in column-major format that will be accepted by the rest of the algorithms in the library.


Options
=======

Various options can be set to customize the behavior of the data loading functions by calling one of these :ref:`functions <api_datastore_options>`.

.. _csv_options:

CSV file
--------

The format the CSV file reader expects can be modified.

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

Examples
========

**Reading CSV**

Various files in the examples folder of your installation demonstrate the use of the CSV reading functions (e.g. ``basic_pca.cpp`` and ``linear_model.cpp``).

**Datastore**

- ``datastore.cpp``, ``linear_model.cpp``: loading, selecting and using data
- (TODO add more involved example program)


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

.. _api_init:

Initialize and destroy datastores
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: da_datastore_init
.. doxygenfunction:: da_datastore_destroy


.. _api_load_data:

Load data into a datastore
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: da_data_load_from_csv
.. doxygenfunction:: da_data_hconcat
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


.. _api_data_selection:

Data selection
^^^^^^^^^^^^^^

.. doxygenfunction:: da_data_select_columns
.. doxygenfunction:: da_data_select_rows
.. doxygenfunction:: da_data_select_slice
.. doxygenfunction:: da_data_select_non_missing

.. _api_data_extraction:

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

.. _api_column_header:

Column headers
^^^^^^^^^^^^^^

.. doxygenfunction:: da_data_label_column
.. doxygenfunction:: da_data_get_col_idx
.. doxygenfunction:: da_data_get_col_label


.. _api_data_editon:

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
