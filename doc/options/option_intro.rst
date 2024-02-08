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

.. _option_setting:

Setting Options
***************

The C APIs in AOCL-DA often rely on a large number of parameters, which can all influence the internal behaviour of the functions.
For example, when reading in a CSV file the specific characters used to denote comments, decimal points, scientific notation, quotes and delimiters can all be specified.

Rather than forcing users to supply all such parameters through the API signature, the defaults are automatically stored.
Any parameters which you wish to change from their default values can then be changed using the option setting APIs.

The :cpp:type:`da_handle` and :cpp:type:`da_datastore` structs can both be used to set options.
This should be done after the :cpp:type:`da_handle` or :cpp:type:`da_datastore` has been initialized, but prior to calling any computational routines.

The available options depend on the particular computation being performed, and are detailed in the specific APIs you are using, but all options have a string name and are set or queried using the functions described in :ref:`Options APIs for da_handle <api_handle_options>` and :ref:`Options APIs for da_datastore <api_datastore_options>`.
The APIs are all very similar and differ only in the type of the final parameter (which contains the value of the option itself), which may be a string, integer, double or float.

.. toctree::
    :maxdepth: 1
    :hidden:

    option_handle_api
    option_datastore_api

