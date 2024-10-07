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



.. _intro_handle:

Building and Solving Models
***************************

Most of the computational functions in the AOCL-DA C API use a :cpp:type:`da_handle` as their first argument. This serves several purposes.

- It is used to store internal data which can be passed between the different APIs (for example, :ref:`da_pca_compute_? <da_pca_compute>` computes the principal components of a data matrix, which can be passed via the handle to :ref:`da_pca_transform_? <da_pca_transform>`).
- It stores :ref:`error information <error_api>` which may be useful if a function call does not complete as expected.
- It can be used to set additional :ref:`options <option_setting>` for customizing computations.

Using the AOCL-DA computational functions involves the following steps.

1. Initialize a :cpp:type:`da_handle` using :ref:`da_handle_init_? <da_handle_init>` and specify the :cpp:type:`da_handle_type`.
2. Pass data into the :cpp:type:`da_handle`. The exact API for doing this will depend on the computation you wish to perform and will be detailed in the relevant chapter of this documentation.
3. Set any additional options for the computation. All algorithms in AOCL-DA use the same :ref:`option setting APIs <api_handle_options>`, but the exact choice of options will depend on the algorithm.
4. Perform the computation.
5. Extract :ref:`results <extracting-results>` from the handle.
6. Destroy the handle using :cpp:func:`da_handle_destroy`. Destroying the handle will free up the internally allocated memory and avoid potential memory leaks.

Throughout this workflow, you may wish to query the handle for any additional :ref:`error <error-handling>` information by calling :cpp:func:`da_handle_print_error_message` or :cpp:func:`da_datastore_print_error_message`.

Further details on the APIs for creating/destroying handles, setting options, error handling and extracting results are provided in the subsequent sections of this documentation.

.. toctree::
    :maxdepth: 1
    :hidden:

    handle_api
