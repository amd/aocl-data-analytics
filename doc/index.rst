..
    Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.

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

.. _aocl-da-index:

AOCL-Data Analytics
********************

The AOCL Data Analytics Library (AOCL-DA) is a software library providing optimized building blocks for data analysis and classical machine learning applications.
It provides users with a complete workflow, including data handling, preprocessing, modeling and validation.
It is designed with a focus on usability, reliability and performance.

AOCL-DA is written with both C-compatible and Python interfaces to make it as seamless as possible to integrate with the library from
whichever programming language you are using. Both sets of interfaces call the same underlying optimized code base.

This documentation is available online in the form of web pages. It consists of three main sections:

* **AOCL-DA for Python**: contains instructions for calling and using the Python APIs.
* **AOCL-DA for C**: contains instructions for calling and using the C APIs.
* **Algorithms**: details the specific computational algorithms available in AOCL-DA, which have both C and Python APIs.

Your installation is also packaged with C++ and Python example programs.

The source code for AOCL-DA can be obtained from https://github.com/amd/aocl-data-analytics/.
You can also install AOCL-DA using the AOCL spack recipe: https://www.amd.com/en/developer/zen-software-studio/applications/spack/spack-aocl.html.

.. AOCL-DA is developed and maintained by AMD (https://www.amd.com/). For support or queries, you can e-mail us on
   toolchainsupport@amd.com.

AOCL-DA for Python
------------------

.. toctree::
   :maxdepth: 1

   python_intro
   sklearn

AOCL-DA for C
-------------

.. toctree::
   :maxdepth: 1

   C_intro
   data_management/data
   da_handle/handle
   options/option
   errors/error
   results/results

Algorithms
----------

.. toctree::
   :maxdepth: 1

   basic_statistics/basic_stats
   clustering/clustering
   factorization/factorization
   linear_models/linear_models
   nonlinear_models/nldf
   trees_forests/trees_forest
   metrics/metrics
   nearest_neighbors/knn
   kernel_functions/kernel_functions
   svm/svm

Appendices
----------

.. toctree::
    :maxdepth: 1

    options/all_table
    bibliography
    genindex
    search

.. only:: internal

   .. toctree::
      :maxdepth: 1

      doc_utilities/utils
