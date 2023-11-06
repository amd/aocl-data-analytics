..
    Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.

    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
    3. Neither the name of the copyright holder nor the names of its contributors
       may be used to endorse or promote products derived from this software without
       specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
    IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
    OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

.. |doublequote| replace:: "

.. _data-management:

Data Management
***************

Introduction
============

AOCL-DA provides a suite of functions designed to allow loading and manipulating data in a straightforward
manner before calling the algorithms of the library. All functions in this suite use the :cpp:type:`da_datastore` structure to store and manipulate data.

There are two ways to load data in the library:

- directly :ref:`into an array <data_array_intro>` of pre-allocated memory of the correct type, or

- into a :ref:`data store <datastores_intro>`, which contains functionality for cleaning and manipulating data.

At the current version of AOCL-DA, the only supported format to load data from hard memory is CSV (comma separated
values).


.. _data_array_intro:

Loading data from hard memory to dense arrays
=============================================

CSV files
---------

AOCL-DA contains routines for reading data of a single type from a CSV (comma separated values) file into an array. In addition, a
character array of column headings can optionally be read.

The routines take a :cpp:type:`da_datastore` structure as their first argument, which must be initialized prior
to the routine call
using :cpp:func:`da_datastore_init`. This is used to store :ref:`options <csv_options>` but is not used to store the actual
CSV data, which is instead returned in an array. If you wish to load data directly from the CSV file to the
:cpp:type:`da_datastore` struct, then use :cpp:func:`da_data_load_from_csv`.

For more details on each of the available functions, see the :ref:`API documentation. <csv_api>`

.. _datastores_intro:

Data manipulation : Data stores
===============================

The :cpp:type:`da_datastore` structure can also be used to load and manipulate data. The life cycle of a data store typically follows these steps:

- :ref:`Initialize <api_init>` the data store structures.
- :ref:`Load <api_load_data>` data into the data stores, from pre-allocated memory, CSV files or other data stores.
- :ref:`Edit the data <api_data_edition>`.
- :ref:`Select <api_data_selection>` a subsection of the store.
- :ref:`Extract data <api_data_extraction>`, either columns or specific selections.
- :ref:`Cleanly destroy <api_init>` the structure.

For more details on each of the available functions, see the :ref:`API documentation. <datastore_api>`


Loading data into a :cpp:type:`da_datastore`
--------------------------------------------

Loading data into a :cpp:type:`da_datastore` can be done by adding blocks from different sources. A typical example would be to load data from a file and add columns that were allocated dynamically in your program. This can be achieved by calling :cpp:func:`da_data_load_from_csv` and :cpp:func:`da_data_load_col_int` consecutively for example.

When calling any of the ``da_data_load_*`` functions on a :cpp:type:`da_datastore` that is not empty, certain constraints must be
respected:

- While adding columns, the number of rows in the block to be added must match the current number of rows present in the :cpp:type:`da_datastore` (:cpp:func:`da_data_get_num_rows` can be used to query the dimension).

- New rows can be added in several sub-blocks. However:
    - the :cpp:type:`da_datastore` will be locked until the current number of columns in the store matches the number of columns of the new block
    - Each sub-block has a minimum column size determined by the number of consecutive columns of the same type in the store. For example, if a given store already has 2 integer columns and a float column, new rows can be added in 2 sub-blocks (one with 2 integer columns and one with the remaining float column).

The last way to load data into a given store is from another :cpp:type:`da_datastore`. Calling :cpp:func:`da_data_hconcat` will
concatenate horizontally 2 :cpp:type:`da_datastore` with matching number of rows.


Selecting and extracting data
-----------------------------

The :cpp:type:`da_datastore` structure uses *selections* to select and or extract a subset of the data it contains. Note that column and row indices are always zero-based, meaning the first index is 0 and the indices of the last column and row are ``n_cols-1`` and ``n_rows-1`` respectively.

**Selections**

*Selections* are defined by a label and a set of column and row indices. If any of the functions in :ref:`this subsection <api_data_selection>` are called with a label that does not exist, a new selection is added to the store. Any number of selections can be defined at the same time in a given store.

:cpp:func:`da_data_select_columns`, :cpp:func:`da_data_select_rows` and :cpp:func:`da_data_select_slice` can be used to add respectively a set of column and row indices to a given selection label while :cpp:func:`da_data_select_non_missing` will remove all row indices containing missing data from the selection.

**Extraction**

Once the data is fully loaded, one may want to extract it into dense blocks of contiguous memory suitable for the various algorithms of AOCL-DA. There are two ways to :ref:`extract data<api_data_extraction>` from a :cpp:type:`da_datastore`:

- Extract a specific column with one of the ``da_data_extract_column_*`` functions.
- Extract a selection with a given label by calling one of the ``da_data_extract_selection_*`` functions.

All extracted data will be given in column-major format that will be accepted by the rest of the algorithms
in the library.


Options
=======

Various options can be set to customize the behavior of the data loading functions by calling one of these
:ref:`functions <api_datastore_options>`. The following table details the available options

.. _csv_options:

.. csv-table:: CSV file reading options
   :header: "Option Name", "Type", "Default", "Description", "Constraints"

   "csv use header row", "da_int", ":math:`i = 0`", "Whether or not to interpret the first row as a header", ":math:`0 \le i \le 1`"
   "csv warn for missing data", "da_int", ":math:`i = 0`", "If set to 0, return error if missing data is encountered; if set to, 1 issue a warning and store missing data as either a NaN (for floating point data) or the maximum value of the integer type being used", ":math:`0 \le i \le 1`"
   "csv skip footer", "da_int", ":math:`i = 0`", "Whether or not to ignore the last line when reading a CSV file", ":math:`0 \le i \le 1`"
   "csv delimiter", "string", ":math:`s =` `','`", "The delimiter used when reading CSV files",
   "csv whitespace delimiter", "da_int", ":math:`i = 0`", "Whether or not to use whitespace as the delimiter when reading CSV files", ":math:`0 \le i \le 1`"
   "csv decimal", "string", ":math:`s =` `'.'`", "The character used to denote a decimal point in CSV files",
   "csv skip initial space", "da_int", ":math:`i = 0`", "Whether or not to ignore initial spaces in CSV file lines", ":math:`0 \le i \le 1`"
   "csv line terminator", "string", ":math:`s =` `''`", "The character used to denote line termination in CSV files (leave this empty to use the default)",
   "csv row start", "da_int", ":math:`i = 0`", "Ignore the specified number of lines from the top of the CSV file (note that line numbers in CSV files start at 1)", ":math:`0 \le i \le` INT_MAX"
   "csv comment", "string", ":math:`s =` `'#'`", "The character used to denote comments in CSV files (note, if a line in a CSV file is to be interpreted as only containing a comment, the comment character should be the first character on the line)",
   "csv quote character", "string", ":math:`s =` '|doublequote|'", "The character used to denote quotations in CSV files",
   "csv scientific notation character", "string", ":math:`s =` `'e'`", "The character used to denote powers of 10 in floating point values in CSV files",
   "csv escape character", "string", ":math:`s =` `'l'`", "The escape character in CSV files",
   "csv thousands", "string", ":math:`s =` `''`", "The character used to separate thousands when reading numeric values in CSV files",
   "csv skip rows", "string", ":math:`s =` `''`", "A comma- or space-separated list of rows to ignore in CSV files",
   "csv skip empty lines", "da_int", ":math:`i = 0`", "Whether or not to ignore empty lines in CSV files (note that caution should be used when using this in conjunction with options such as CSV skip rows since line numbers may no longer correspond to the original line numbers in the CSV file)", ":math:`0 \le i \le 1`"
   "csv double quote", "da_int", ":math:`i = 0`", "Whether or not to interpret two consecutive quotechar characters within a field as a single quotechar character", ":math:`0 \le i \le 1`"

Note that, with the exception of the ``CSV skip rows`` option, only single characters can be used in the string options above.

Examples
========

**Reading CSV**

Various files in the examples folder of your installation demonstrate the use of the CSV reading functions (e.g. ``linear_model.cpp``).

**:cpp:type:`da_datastore`**

The source files ``datastore.cpp`` and ``linmod_diabetes.cpp`` showcase loading, selecting and using data with a :cpp:type:`da_datastore` handle.


..
    Link to the APIs

.. toctree::
    :maxdepth: 1
    :hidden:

    csv_api
    datastores_api
