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
    


.. AOCL-DA documentation master file

AOCL-DA Documentation
*********************

The AOCL Data Analytics Library is a software library providing optimized building blocks for data analysis.
It provides users with a complete workflow, including data handling, preprocessing, modeling and validation.
It is designed with a focus on usability, reliability and performance.

he AOCL Data Analytics Library (AOCL-DA) is a data analytics library providing 
optimized building blocks for data analysis. It is written with a C-compatible 
interface to make it as seamless as possible to integrate with the library from 
whichever programming language you are using. This documentation is also available 
online in the form of web pages.

The intended workflow for using the library is as follows:

 - load data from memory by reading CSV files or using the in-built da_datastore object
 
 - preprocess the data by removing missing values, standardizing, and selecting certain subsets of the data, before extracting contiguous arrays of data from the da_datastore objects
  
 -  data processing (e.g. principal component analysis, linear model fitting, etc.)


C++ example programs can be found in the ``examples`` folder of your installation.


AOCL-DA is developed and maintained by AMD (https://www.amd.com/). For support or queries, you can e-mail us on 
toolchainsupport@amd.com.


.. toctree::
   :maxdepth: 1
   :caption: AOCL Data Analytics Library

   general_intro
   data_management/data_intro
   da_handle/handle_intro
   options/option_intro
   errors/error_intro
   results/results_intro

.. toctree::
   :maxdepth: 1
   :caption: Algorithms

   basic_statistics/basic_stats_intro
   factorization/factorization_intro
   linear_models/linmod_intro
   trees_forests/df_intro

.. only:: internal

   .. toctree::
      :maxdepth: 1
      :caption: Internal help:

      doc_utilities/utils.rst

Search the Documentation
========================

* :ref:`genindex`
* :ref:`search`

References
==========

.. bibliography::
