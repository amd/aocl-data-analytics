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
    


.. _extracting-results:

Extracting Results
******************

In order to simplify APIs, the results of computations are stored within the :cpp:type:`da_handle`.
Results can be extracted from handles using :cpp:func:`da_handle_get_result_d`, :cpp:func:`da_handle_get_result_s` and :cpp:func:`da_handle_get_result_int`, depending on the datatype of the required result.
Each of these routines takes four arguments:

1. the :cpp:type:`da_handle` used in the computation,
2. a :cpp:type:`da_result` enumeration, specifying which result is required,
3. a pointer to the size of the array which the result will be written to,
4. an array in which to write the result.

The specific results available (i.e. the possible values of :cpp:type:`da_result`) are described in the documentation for the particular computational API you are using.

If the array supplied is too small, then the error :cpp:enum:`da_status_invalid_array_dimension` will be returned and the pointer to the size of the array will be overwritten with the minimum size required to hold the result.


.. toctree::
    :maxdepth: 1
    :hidden:
    
    results_api