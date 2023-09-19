Data Stores APIs
****************

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


.. _api_data_edition:

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
