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

.. doxygenfunction:: da_data_get_n_rows
.. doxygenfunction:: da_data_get_n_cols
.. doxygenfunction:: da_data_get_element_int
.. doxygenfunction:: da_data_get_element_real_d
.. doxygenfunction:: da_data_get_element_real_s
.. doxygenfunction:: da_data_get_element_uint8
.. doxygenfunction:: da_data_set_element_int
.. doxygenfunction:: da_data_set_element_real_d
.. doxygenfunction:: da_data_set_element_real_s
.. doxygenfunction:: da_data_set_element_uint8
