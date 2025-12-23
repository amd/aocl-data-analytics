..
    Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.

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



.. _C_intro:

Introduction to the C APIs
*****************************

This section contains instructions for writing code that calls AOCL-DA using the C APIs and for building
applications that link to the library.
Numerous example programs are also provided in the ``examples`` folder within your
AOCL-DA installation directory.

The C interface has been designed to make it as seamless as
possible to integrate with the library from whatever programming language you are using.
In addition, a header file containing :ref:`C++ overloads<cpp_overloads>` is available for C++ users who wish to abstract away the floating-point data type.

Library workflow
================

The intended workflow for using the AOCL-DA C APIs is as follows:

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
  * pass arrays of data, or data extracted from a :cpp:type:`da_datastore`, to the
    handle (for best possible
    performance, algorithmic functions typically operate on two-dimensional arrays
    stored in column major format);
  * computation (e.g. clustering, linear model, principal component analysis);
  * extract results from the :cpp:type:`da_handle`.


Linking your application to AOCL-DA
===================================

Linking on Linux
------------------
These instructions assume your application is written in C++, but the AOCL-DA C APIs have been
designed to make calling from other languages as straightforward as possible.
In the example compilation commands below, ``INT_LIB`` is either ``LP64`` or
``ILP64`` for 32 and 64 bit integers respectively.

To compile and link to static AOCL libraries using ``g++``:

.. code-block::

    g++ <your_source_code>.cpp -I /<path to aocl-da headers>/include_<INT_LIB>
        /<path to aocl-da>/lib_<INT_LIB>/libaocl-da.a
        /<path to amd-sparse>/lib_<INT_LIB>/libaoclsparse.a
        /<path to amd-libflame>/lib_<INT_LIB>/libflame.a
        /<path to amd-blis>/lib_<INT_LIB>/libblis-mt.a
        /<path to libaoclutils>/lib_<INT_LIB>/libaoclutils.a -lgfortran -lgomp

To compile and link to static AOCL libraries using ``clang++``:

.. code-block::

    clang++ <your_source_code>.cpp -I /<path to aocl-da headers>/include_<INT_LIB>
            /<path to aocl-da>/lib_<INT_LIB>/libaocl-da.a
            /<path to amd-sparse>/lib_<INT_LIB>/libaoclsparse.a
            /<path to amd-libflame>/lib_<INT_LIB>/libflame.a
            /<path to amd-blis>/lib_<INT_LIB>/libblis-mt.a
            /<path to libaoclutils>/lib_<INT_LIB>/libaoclutils.a -lflang -lomp -lpgmath

To compile and link to dynamic AOCL libraries using ``g++``:

.. code-block::

    g++ <your_source_code>.cpp -I /<path to aocl-da headers>/include_<INT_LIB>
        -L /<path to aocl-da>/lib_<INT_LIB> -L /<path to amd-sparse>/lib_<INT_LIB>
        -L /<path to amd-libflame>/lib_<INT_LIB> -L /<path to amd-blis>/lib_<INT_LIB>
        -L /<path to amd-utils>/lib -laocl-da -laoclsparse -lflame -lblis-mt -laoclutils
        -lgfortran -lgomp

To compile and link to dynamic AOCL libraries using ``clang++``:

.. code-block::

    clang++ <your_source_code>.cpp -I /<path to aocl-da headers>/include_<INT_LIB>
            -L /<path to aocl-da>/lib_<INT_LIB> -L /<path to amd-sparse>/lib_<INT_LIB>
            -L /<path to amd-libflame>/lib_<INT_LIB> -L /<path to amd-blis>/lib_<INT_LIB>
            -L /<path to amd-utils>/lib -laocl-da -laoclsparse -lflame -lblis-mt -laoclutils
            -lflang -lomp -lpgmath

Note that for dynamic linking you will need to update your ``LD_LIBRARY_PATH`` environment
variable, e.g. ``export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<AOCL_ROOT>/lib_<INT_LIB>``.

If you wish to call AOCL-DA from a C code, then you should compile using your C compiler
(e.g. ``gcc``), but link separately, using a C++ linker (e.g. ``g++``).

Linking on Windows
------------------

In the example compilation command below, ``INT_LIB`` is either ``LP64`` or ``ILP64``
for 32 and 64 bit integers respectively.

.. code-block::

    cl <example_name>.cpp /I \<path to aocl-da headers>\include\<INT_LIB> /EHsc /MD
       \<path to aocl-da>\lib\<INT_LIB>\aocl-da.lib
       \<path to amd-sparse>\lib\<INT_LIB>\shared\aoclsparse.lib
       \<path to amd-libflame>\lib\<INT_LIB>\AOCL-LibFlame-Win-MT-dll.lib
       \<path to amd-blis>\lib\<INT_LIB>\AOCL-LibBlis-Win-MT-dll.lib
       \<path to amd-utils>\lib\libaoclutils.lib /openmp:llvm

The same command will work with ``cl`` replaced by ``clang-cl`` (in which case, simply use ``/openmp``) and linking statically using ``/MT``.

.. note::
   You should ensure the folders containing the libraries to be linked are on your
   Windows ``PATH`` environment variable e.g. using ``set PATH=%PATH%;C:\<path_to_BLAS_and_LAPACK>``.
   Depending on how your system is set up, and which functions you are using, you may also need to
   link to some Fortran runtime libraries such as ``libfifcore-mt.lib``.
   The easiest way to do this is to source the ifort compiler using e.g. ``C:\Program Files (x86)\Intel\oneAPI\setvars.bat``.

Compiling using CMake
---------------------

A CMakeLists.txt file is supplied in the examples folder of your installation. You will need to set ``AOCL_ROOT`` to point to your AOCL installation.
On Windows, you may also need to source the ifort compiler using e.g. ``C:\Program Files (x86)\Intel\oneAPI\setvars.bat``.
To configure cmake and compile, you can then use commands such as:

.. code-block::

    mkdir build
    cd build
    cmake -G Ninja -DCMAKE_CXX_COMPILER=clang-cl ..
    cmake --build .

Miscellaneous topics
====================

Data types in AOCL-DA
---------------------

.. _da_int:

AOCL-DA uses the ``da_int`` integer type throughout the library.
For the 32 bit integer library (``LP64``), this is defined to be a 32 bit signed integer.
For the 64 bit integer library (``ILP64``, compiled with the CMake flag ``-DBUILD_ILP64=On``), this is defined to be a 64 bit signed integer.
When compiling your own code to use the 64 bit integer library, you must ensure that the ``AOCLDA_ILP64`` build variable is set.

.. _da_real_prec:

Algorithmic routines operating on floating-point data are typically available in both
single and double precision.
Routines expecting single precision data have ``_s`` appended onto their names.
Routines expecting double precision data have ``_d`` appended onto their names.
Some routines (such as option setting routines) may expect other data types, and will have ``_int`` or ``_string`` appended onto their names accordingly.
In this documentation we frequently use ``_?`` at the end of routine names to indicate a suite of routines that differ only in the expected data type.

Array storage
-------------

Algorithmic routines in the library can handle two-dimensional arrays in either column-major or row-major order.
However, for best possible performance, it is recommended that you store your data in column-major format, since row-major arrays may be copied and transposed internally.

Interpreting missing data
-------------------------

When using a :cpp:type:`da_datastore` for data management, the special convention for floating point *not a number* (``NaN``) can be used to denote missing
data. See :ref:`Data Management Functionalities<data-management>` for further details.

In order to provide the best possible performance, the algorithmic functions will not automatically check for
``NaN`` data. If a ``NaN`` is passed into an algorithmic function, its behavior is undefined.
It is therefore the user's responsibility to ensure data is sanitized before passing it to one of the algorithms (for example, by using
:cpp:func:`da_data_select_non_missing`, by calling :cpp:func:`da_check_data_s`, or by setting the ``check data`` option in the algorithmic APIs that use handles).

Error handling
--------------

Functions in AOCL-DA return :cpp:type:`da_status`, which provides basic information about whether
the function call was successful.
Further information about errors can be obtained by querying :cpp:type:`da_datastore` and
:cpp:type:`da_handle` types. For further details, see the :ref:`error handling<error-handling>` pages.

Version string
--------------

To get the version string of AOCL-DA, call the function ``const char* da_get_version()``.


.. _cpp_overloads:

C++ overloads
--------------

To facilitate calling AOCL-DA from C++, a set of overloaded functions has been made available.
These are identical to the C interface, except that none of the functions have data type indicators such as ``_s`` or ``_d`` appended onto their names.
Your C++ compiler will instead call the correct function based on the floating point precision you are using.

For some functions, overloading is not possible (for example, functions such as :cpp:func:`da_handle_init_s` and :cpp:func:`da_handle_init_d` do not use ``double`` or ``float`` arguments).
In these cases, templated functions are available (e.g. ``da_handle_init<T>``, where ``T`` can be ``double`` or ``float``).

The complete list of available C++ functions is found in ``aoclda_cpp_overloads.hpp`` in the include folder of your installation (and reproduced below).

.. collapse:: AOCL-DA C++ overloads

    .. literalinclude:: ../source/include/aoclda_cpp_overloads.hpp
      :language: C++
      :linenos:
