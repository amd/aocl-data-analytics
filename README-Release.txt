AOCL Data Analytics Library
===========================

The AOCL Data Analytics Library (AOCL-DA) is a data analytics library providing
optimized building blocks for data analysis. It is written with a `C`-compatible
interface to make it as seamless as possible to integrate with the library from
whichever programming language you are using. For further details on the library
contents, please refer to the online help or PDF user guide.

The intended workflow for using the library is as follows:

 - load data from memory by reading CSV files or using the in-built da_datastore object

 - preprocess the data by removing missing values, standardizing, and selecting certain subsets of the data, before extracting contiguous arrays of data from the da_datastore objects

 -  data processing (e.g. principal component analysis, linear model fitting, etc.)

C++ example programs can be found in the `examples` folder of your installation.

AOCL-DA is developed and maintained by [AMD](https://www.amd.com/). For support or queries, you can email us on
[toolchainsupport@amd.com](toolchainsupport@amd.com).

Building the Library
====================

AOCL-DA is built with CMake, with supported compilers GNU and AOCC on Linux and MSVC on MS Windows.

AOCL-DA is dependent on BLAS and LAPACK.

Building on Linux
-----------------

1. You will need to have BLAS and LAPACK installed.

2. Make sure you have set the environment variable `$AOCL_ROOT` to where the AOCL libraries are
   installed e.g. `/home/username/amd/aocl/4.0`.

3. Configure cmake with any of the following options:

   * `-DMEMSAN=On` for memory sanitization

   * `-DASAN=On` for address sanitization

   * `-DVALGRIND=On` for valgrind use

   * `-DBUILD_ILP64=On` for 64-bit integer build

   * `-DCMAKE_BUILD_TYPE=Debug` or `Release`

   * `-DCOVERAGE=On` to build code coverage report

   * `-DBUILD_EXAMPLES=On` and `–DBUILD_GTEST=On` both of which are `On` by default

   * `-DBUILD_SHARED_LIBS=On` for a shared library build (`Off` by default)

   * `-DBUILD_DOC=On` to build the documentation. Use `cmake --build . --target doc` to build all documentation formats (or `doc_pdf`, `doc_html` to build only PDF or only HTML formats)

   * `-DINTERNAL_DOC=On` to build the internal documentation alongside the main one

   * `-DBUILD_SMP=On` to build using OpenMP and threaded BLAS (`On` by default)

   * `-DVECTORIZATION_REPORTS=On` to build with vectorization reports enabled

   * `-DDA_LOGGING=On` to enable debug printing

   * `-DBUILD_PYTHON=On` to build the Python interfaces

   * `-DCMAKE_AOCL_ROOT=<path to AOCL>` to specify a location for AOCL libraries. This has precedence over the environment variable `$AOCL_ROOT`

   * `-DCMAKE_INSTALL_PREFIX=<install path>` to specify the install path for the library

   * Any combination of `-DLAPACK_LIB`, `-DBLAS_LIB`, `-DLAPACK_INCLUDE_DIR` and `-DBLAS_INCLUDE_DIR` if you wish to override the use of `AOCL_ROOT` with specific choices of BLAS and LAPACK libraries and include directories. Care should be taken if you do this as there will be no checks for the correctness of the linked libraries.

   **Note** that not all the options available in `Release` build mode.

5. Type `cmake --build . --target all` (or `--target doc`, to build the documentation).

6. Run the tests or examples using ctest e.g. `ctest -V –R` followed by a string to find a particular set of tests.

Building on MS Windows
----------------------

1. You will need either:
   * a Visual Studio installation and compatible Fortran compiler (this will allow you to build with cl or with the MSVC compatibility layer for clang (clang-cl)).
   * GCC and gfortran compilers, which are available via MinGW and MSYS2.

2. Make sure you have set the `AOCL_ROOT` environment variable to your AOCL installation directory (e.g. `C:\Users\username\AOCL-4.0`), and update your `PATH` to take in the relevant BLAS and LAPACK libraries e.g.
`set PATH=C:\path\to\AOCL\amd-blis\lib\LP64;C:\path\to\AOCL\amd-libflame\lib\LP64;%PATH%`.

3. Configure cmake with any of the following options:

   * `-DBUILD_ILP64=On` for 64-bit integer build

   * `-DBUILD_EXAMPLES=On` and `–DBUILD_GTEST=On` both of which are on by default

   * `-DBUILD_SMP=On` to build using OpenMP and threaded BLAS (`On` by default)

   * `-DBUILD_SHARED_LIBS=On` for a shared library build (`Off` by default)

   * `-DCMAKE_AOCL_ROOT=<path to AOCL>` if you wish to specify a location for AOCL libraries without using environment variables

   * Any combination of `-DLAPACK_LIB`, `-DBLAS_LIB`, `-DLAPACK_INCLUDE_DIR` and `-DBLAS_INCLUDE_DIR` if you wish to override the use of `AOCL_ROOT` with specific choices of BLAS and LAPACK libraries and include directories. Care should be taken if you do this as there will be no checks for the correctness of the linked libraries.

    **Note** that not all the options available in Linux are available in Windows

4. Either:

   * Open Visual Studio and load the `AOCL-DA.sln` file then build Debug or Release builds using the GUI, or

   * In a powershell type `devenv .\AOCL-DA.sln /build "Debug"` to build the solution (change to Release as appropriate)

   * If using the GNU toolchain, build using cmake and Ninja.

Building the Python interfaces
------------------------------

To build the Python interfaces, use `-DBUILD_PYTHON=On` (note that this will only work with shared library builds).
You will need PyBind11, which can be installed using `pip install pybind11`.
On Windows you may also need to set the `CMAKE_PREFIX_PATH` to point to the location of you pybind11 installation, e.g.  `C:\path\to\your\python-environment\site-packages\pybind11\share\cmake\pybind11`
By default, cmake will compile the bindings but will not install them.
If you set `-DCMAKE_INSTALL_PREFIX=<install path>` in your configure step and run `cmake --build . --target install`, then cmake will also create a Python wheel, `aoclda-*.whl`, where `*` depends on your system. This wheel can be installed using `pip install aoclda-*.whl`.
When using the bindings on Windows, the Intel Fortran runtime must be available. This can be done by setting the environment variable `INTEL_FCOMPILER`.