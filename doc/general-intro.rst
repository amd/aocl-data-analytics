.. _chapter_gen_intro:

Introduction to AOCL-DA
************************

This section contains instructions for writing code that calls AOCL-DA and for building applications that link to the library.
Numerous example programs are also provided in the ``examples`` folder within your AOCL-DA installation directory.

AOCL-DA has a C-compatible interface, which has been chosen to make it as seemless as possible to integrate with the library from whichever programming language you are using.

Library workflow
================

The intended workflow for using the AOCL-DA library is as follows:

1. **Load data from memory.** Data can be obtained from various sources (see :ref:`Data management functionalities<data-management>` for full details):

  * pass arrays of data directly from another part of your application;
  * read directly from CSV files into floating-point or integer arrays;
  * read mixed data from CSV files into a :cpp:type:`da_datastore`.

2. **Data preprocessing.** Functions are available for:

  * standardizing, scaling and shifting arrays of floating point data;
  * removing missing data from a :cpp:type:`da_datastore`;
  * selecting certain subsets of the data in a :cpp:type:`da_datastore`;
  * extracting contiguous arrays of data from a :cpp:type:`da_datastore`.

3. **Data processing.** This is often split into several function calls:

  * initialize a :cpp:type:`da_handle` struct, which is used internally to store algorithmic information;
  * pass arrays of data, or data extracted from a :cpp:type:`da_datastore` to the handle (for best possible performance, algorithmic functions tyically operate on two-dimension arrays stored in column major format);
  * computation (e.g. clustering, linear model, principal component analysis);
  * extract results from the :cpp:type:`da_handle`.


Linking your application to AOCL-DA
===================================

Linking on Linux
------------------
These instructions assume your application is written in C++, but AOCL-DA has been designed to make calling from other languages as straightforward as possible.
``AOCL_ROOT`` is taken to be the path to your AOCL library installation, for example ``/opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc``.
``INT_LIB`` is either ``LP64`` or ``ILP64`` for 32 and 64 bit integers.

To compile and link to static AOCL libraries using ``g++``:

.. code-block::

    g++ <your_source_code>.cpp -I <AOCL_ROOT>/include_<INT_LIB>
        <AOCL_ROOT>/lib_<INT_LIB>/libaocl-da.a <AOCL_ROOT>/lib_<INT_LIB>/libflame.a
        <AOCL_ROOT>/lib_<INT_LIB>/libblis-mt.a -lgfortran -lgomp

To compile and link to static AOCL libraries using ``aocc``:

.. code-block::

    clang++ <your_source_code>.cpp -I <AOCL_ROOT>/include_<INT_LIB>
            <AOCL_ROOT>/lib_<INT_LIB>/libaocl-da.a <AOCL_ROOT>/lib_<INT_LIB>/libflame.a
            <AOCL_ROOT>/lib_<INT_LIB>/libblis-mt.a -lflang -lomp

To compile and link to dynamic AOCL libraries using ``g++``:

.. code-block::

    g++ <your_source_code>.cpp -I <AOCL_ROOT>/include_<INT_LIB>
        -L<AOCL_ROOT>/lib_<INT_LIB> -laocl-da -lflame -lblis-mt
        -lgfortran -lgomp

To compile and link to dynamic AOCL libraries using ``clang++``:

.. code-block::

    clang++ <your_source_code>.cpp -I <AOCL_ROOT>/include_<INT_LIB>
            -L<AOCL_ROOT>/lib_<INT_LIB> -laocl-da -lflame -lblis-mt
            -lflang -lomp

Note that for dynamic linking you will need to update our ``LD_LIBRARY_PATH`` environment variable e.g. export ``LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<AOCL_ROOT>/lib_<INT_LIB>``.

If you wish to call AOCL-DA from a C code, then you should compile using your C compiler (e.g. ``gcc``), but link separately, using a C++ linker (e.g. ``g++``).

Linking on Windows
------------------

``AOCL_ROOT`` is taken to be the path to your AOCL library installation, for example ``C:\Users\<your_name>\AMD\AOCL``.
``INT_LIB`` is either ``LP64`` or ``ILP64`` for 32 and 64 bit integers.

.. code-block::

    cl <example_name>.cpp /I <AOCL_ROOT>\include\<INT_LIB> /EHsc /MD
       <AOCL_ROOT>\aocl-da\lib\<INT_LIB>\aocl-da.lib
       <AOCL_ROOT>\amd-libflame\lib\<INT_LIB>\AOCL-LibFlame-Win-MT-dll.lib
       <AOCL_ROOT>\amd-blis\lib\<INT_LIB>\AOCL-LibBlis-Win-MT-dll.lib

The same command should work with cl replaced by clang-cl and linking statically using /MT.

Note that you should ensure the folders containing the libraries to be linked are on your Windows ``PATH`` environment variable e.g. using ``set PATH=%PATH%;C:\<path_to_BLAS_and_LAPACK>``. You may also need to link to a Fortran runtime library such as ``libfifcore-mt.lib``.

Miscellaneous topics
====================

Datatypes used by AOCL-DA
-------------------------

.. _da_int:

AOCL-DA uses the ``da_int`` integer type throughout the library.
For the 32-bit integer library (``LP64``) this is defined to be a 32-bit signed integer.
For the 64-bit integer library (``ILP64``) this is defined to be a 64-bit signed integer.

Algorithmic routines operating on floating-point data are typically available in both single and double precision.
Routines expecting single precision data have ``_s`` appended onto their names.
Routines expecting double precision data have ``_d`` appended onto their names.

Array storage
-------------

Algorithmic routines in the library expect two-dimensional arrays to be in column major format.

NaN data
--------

When using a :cpp:type:`da_datastore` for data management, ``NaN`` can be used to denote missing data. See :ref:`Data management functionalities<data-management>` for further details.

In order to provide the best possible performance, the data processing functions will not check for ``NaN`` data. If a ``NaN`` is passed into an algorithmic function, its behaviour is undefined.
It is therefore your responsibility to ensure your data is sanitized (for example, by using :cpp:func:`da_data_select_non_missing`) before passing it to one of the algorithms.

Error handling
--------------

Functions in AOCL-DA return :cpp:type:`da_status`, which provides basic information about whether the function call was successful.
Further information about errors can be obtained by querying :cpp:type:`da_datastore` and :cpp:type:`da_handle` types. For further details see :ref:`Error handling<error-handling>`.