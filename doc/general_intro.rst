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



.. _chapter_gen_intro:

Introduction
************

This section contains instructions for writing code that calls AOCL-DA and for building
applications that link to the library.
Numerous example programs are also provided in the ``examples`` folder within your
AOCL-DA installation directory.

AOCL-DA has a C-compatible interface, which has been chosen to make it as seamless as
possible to integrate with the library from whichever programming language you are using.
Future releases of the library will introduce further language-specific interfaces,
such as a Python interface.

Library Workflow
================

The intended workflow for using the AOCL-DA library is as follows:

1. **Load data from memory.** Data can be obtained from various sources:

  * pass arrays of data directly from another part of your application;
  * read directly from CSV files into floating-point or integer arrays;
  * read mixed data from CSV files into a :cpp:type:`da_datastore`.

2. **Data preprocessing.** Functions are available for:

  * standardizing, scaling and shifting arrays of floating point data;
  * removing missing data from a :cpp:type:`da_datastore`;
  * selecting certain subsets of the data in a :cpp:type:`da_datastore`;
  * extracting contiguous arrays of data from a :cpp:type:`da_datastore`.

3. **Data processing.** This is often split into several function calls:

  * initialize a :cpp:type:`da_handle` struct, which is used internally to store
    algorithmic information;
  * pass arrays of data, or data extracted from a :cpp:type:`da_datastore` to the
    handle (for best possible
    performance, algorithmic functions typically operate on two-dimension arrays
    stored in column major format);
  * computation (e.g. clustering, linear model, principal component analysis);
  * extract results from the :cpp:type:`da_handle`.


Linking your Application to AOCL-DA
===================================

Linking on Linux
------------------
These instructions assume your application is written in C++, but AOCL-DA has been
designed to make calling from other languages as straightforward as possible.
In the example compilation commands below, ``INT_LIB`` is either ``LP64`` or
``ILP64`` for 32 and 64 bit integers respectively.

To compile and link to static AOCL libraries using ``g++``:

.. code-block::

    g++ <your_source_code>.cpp -I /<path to aocl-da headers>/include_<INT_LIB>
        /<path to aocl-da>/lib_<INT_LIB>/libaocl-da.a /<path to libflame>/lib_<INT_LIB>/libflame.a
        /<path to blis>/lib_<INT_LIB>/libblis-mt.a -lgfortran -lgomp

To compile and link to static AOCL libraries using ``clang++``:

.. code-block::

    clang++ <your_source_code>.cpp -I /<path to aocl-da headers>/include_<INT_LIB>
            /<path to aocl-da>/lib_<INT_LIB>/libaocl-da.a /<path to libflame>/lib_<INT_LIB>/libflame.a
            /<path to blis>/lib_<INT_LIB>/libblis-mt.a -lflang -lomp

To compile and link to dynamic AOCL libraries using ``g++``:

.. code-block::

    g++ <your_source_code>.cpp -I /<path to aocl-da headers>/include_<INT_LIB>
        -L /<path to aocl-da>/lib_<INT_LIB> -L /<path to libflame>/lib_<INT_LIB>
        -L /<path to blis>/lib_<INT_LIB> -laocl-da -lflame -lblis-mt
        -lgfortran -lgomp

To compile and link to dynamic AOCL libraries using ``clang++``:

.. code-block::

    clang++ <your_source_code>.cpp -I /<path to aocl-da headers>/include_<INT_LIB>
            -L /<path to aocl-da>/lib_<INT_LIB> -L /<path to libflame>/lib_<INT_LIB>
            -L /<path to blis>/lib_<INT_LIB> -laocl-da -lflame -lblis-mt
            -lflang -lomp

Note that for dynamic linking you will need to update our ``LD_LIBRARY_PATH`` environment
variable e.g. export ``LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<AOCL_ROOT>/lib_<INT_LIB>``.

If you wish to call AOCL-DA from a C code, then you should compile using your C compiler
(e.g. ``gcc``), but link separately, using a C++ linker (e.g. ``g++``).

Linking on Windows
------------------

In the example compilation command below, ``INT_LIB`` is either ``LP64`` or ``ILP64``
for 32 and 64 bit integers respectively.

.. code-block::

    cl <example_name>.cpp /I \<path to aocl-da headers>\include_<INT_LIB> /EHsc /MD
       \<path to aocl-da>\lib_<INT_LIB>\aocl-da.lib
       \<path to libflame>\lib_<INT_LIB>\AOCL-LibFlame-Win-MT-dll.lib
       \<path to blis>\lib_<INT_LIB>\AOCL-LibBlis-Win-MT-dll.lib

The same command should work with ``cl`` replaced by ``clang-cl`` and linking statically using ``/MT``.

**Note** that you should ensure the folders containing the libraries to be linked are on your
Windows ``PATH`` environment variable e.g. using ``set PATH=%PATH%;C:\<path_to_BLAS_and_LAPACK>``.
Depending on how your system is set up, and which functions you are using, you may also need to
link to some Fortran runtime libraries such as ``libfifcore-mt.lib``, ``ifconsol.lib``,
``libifportmd.lib``, ``libmmd.lib``, ``libirc.lib`` and ``svml_dispmd.lib``.


Miscellaneous Topics
====================

Data Types in AOCL-DA
---------------------

.. _da_int:

AOCL-DA uses the ``da_int`` integer type throughout the library.
For the 32-bit integer library (``LP64``) this is defined to be a 32-bit signed integer.
For the 64-bit integer library (``ILP64``, compiled with ``-DUSE_ILP64``) this is defined to be a 64-bit signed integer.

.. _da_real_prec:

Algorithmic routines operating on floating-point data are typically available in both
single and double precision.
Routines expecting single precision data have ``_s`` appended onto their names.
Routines expecting double precision data have ``_d`` appended onto their names.

Array Storage
-------------

Algorithmic routines in the library expect two-dimensional arrays to be in column major format.

Interpreting Missing Data
-------------------------

When using a :cpp:type:`da_datastore` for data management, the special convention for floating point *not a number* (``NaN``) can be used to denote missing
data. See :ref:`Data Management Functionalities<data-management>` for further details.

In order to provide the best possible performance, the algorithmic functions will not check for
``NaN`` data. If a ``NaN`` is passed into an algorithmic function, its behaviour is undefined.
It is therefore the user's responsibility to ensure data is sanitized (for example, by using
:cpp:func:`da_data_select_non_missing`) before passing it to one of the algorithms.

Error Handling
--------------

Functions in AOCL-DA return :cpp:type:`da_status`, which provides basic information about whether
the function call was successful.
Further information about errors can be obtained by querying :cpp:type:`da_datastore` and
:cpp:type:`da_handle` types. For further details see the :ref:`error handling<error-handling>` pages.