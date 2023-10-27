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
    



Basic Statistics
****************


This chapter contains functions to compute basic statistical quantities such as the mean, 
variance or quantiles of a data matrix. Utility routines for standardizing the data are 
also available, and correlation and covariance matrices can also be computed.
These functions operate on an :math:`n \times p` data array (:math:`n` observations and 
:math:`p` variables) stored in column major format.

Choosing an `axis`
------------------

Most statistical quantities can be computed by column, by row or for the data matrix overall. 
This is specified using the :cpp:enum:`da_axis_` enum.

The :cpp:enum:`da_axis_` enum can take the following values:

- :cpp:enumerator:`da_axis_col` - statistical quantities will be computed for each column of the data matrix

- :cpp:enumerator:`da_axis_row` - statistical quantities will be computed for each row of the data matrix

- :cpp:enumerator:`da_axis_all` - statistical quantities will be computed for the whole data matrix

For example, if the routine :cpp:func:`da_mean_s` is called with the :cpp:enum:`da_axis_` argument set 
to :cpp:enumerator:`da_axis_col`, then :math:`p` means will be computed, one for each column. If the 
:cpp:enum:`da_axis_` argument is set to :cpp:enumerator:`da_axis_all`, then a single mean will be computed.

Examples
--------

See ``basic_statistics.cpp`` in the ``examples`` folder of your installation for examples of how to use these functions.

.. toctree::
    :maxdepth: 1
    :hidden:
    
    basic_stats_api
