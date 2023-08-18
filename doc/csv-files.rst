
Reading data from CSV files
===========================

These routines read data of a single type from a CSV (comma separated values) file into an array. In addition, a character array of column headings can optionally be read.

The routines take a :cpp:struct:`_da_datastore` struct as their first argument, which must be initialized prior to the routine call
using :cpp:func:`da_datastore_init`. This is used to store options (see below) but is not used to store the actual CSV data, which is instead returned in an array. If you wish to load data driectly from the CSV file to the :cpp:struct:`_da_datastore` struct, then use :cpp:func:`da_data_load_from_csv`.

Option setting
--------------

Prior to reading the CSV file, various options can be set by passing the :cpp:struct:`_da_datastore` struct to :cpp:func:`da_options_set_int` or :cpp:func:`da_options_set_string` for integer or string options respectively.

The following string options can be set:

- *CSV delimiter* - specify the delimiter used when reading CSV files.

- *CSV thousands* - specify the character used to separate thousands when reading numeric values in CSV files.

- *CSV decimal* - specify which character denotes a decimal point in CSV files.

- *CSV comment* - specify which character is used to denote comments in CSV files (note, if a line in a CSV file is to be interpreted as only containing a comment, the comment character should appear in column 0).

- *CSV quote character* - specify which character is used to denote quotations in CSV files.

- *CSV escape character* - specify the escape character in CSV files.

- *CSV line terminator* - specify which character is used to denote line termination in CSV files (leave this empty to use the default).

- *CSV scientific notation character* - specify which character is used to denote powers of 10 in floating point values in CSV files.

- *CSV skip rows* - a comma- or space-separated list of rows to ignore in CSV files.

Note that, with the exception of the *'CSV skip rows'* option, only single characters can be used in the options above.

The following :ref:`da_int` options can be set:

- *CSV double quote* - specify whether or not to interpret two consecutive quote characters within a field as a single quote character. This option can only take the values 0 or 1.

- *CSV whitespace delimiter* - specify whether or not to use whitespace as the delimiter when reading CSV files. This option can only take the values 0 or 1.

- *CSV row start* - ignore the specified number of rows from the top of the CSV file.

- *CSV skip empty lines* - specify whether or not to ignore empty lines in CSV files. This option can only take the values 0 or 1.

- *CSV skip initial space* - specify whether or not to ignore initial spaces in CSV file lines. This option can only take the values 0 or 1.

- *CSV skip footer* - specify whether or not to ignore the last line when reading a CSV file. This option can only take the values 0 or 1.

- *CSV warn for missing data* - if set to 0 then return an error if missing data is encountered; if set to 1, issue a warning and store missing data as either a NaN (for floating point data) or the maximum value of the integer type being used.

- *CSV use headings* - specify whether or not to interpret the first line as a headings row. This option can only take the values 0 or 1.

Examples
--------

Various files in the examples folder of your installation demonstrate the use of these functions (e.g. :math:`\texttt{basic_pca.cpp}` and :math:`\linear_model.cpp``).

API Reference
-------------

.. doxygenfunction:: da_read_csv_d

.. doxygenfunction:: da_read_csv_s

.. doxygenfunction:: da_read_csv_int

.. doxygenfunction:: da_read_csv_uint8

.. doxygenfunction:: da_read_csv_char

For reading data directly into a :cpp:struct:`_da_datastore` struct, see :cpp:func:`da_data_load_from_csv`