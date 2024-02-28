# AOCL Data Analytics Library

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


# Building the library

AOCL-DA is built with CMake, with supported compilers GNU and AOCC on Linux and MSVC (with ifort) on MS Windows.

AOCL-DA is dependent on BLAS and LAPACK currently, and ultimately may be dependent on other AOCL libraries (such as Sparse).

## Building on Linux

1. You will need to have AOCL-BLAS and AOCL-LAPACK installed somewhere

2. Make sure you have set the environment variable `$AOCL_ROOT` to where the AOCL libraries are
   installed e.g. `/home/username/amd/aocl/4.0`

3. In your checkout create a directory called build and navigate to it

4. Type `cmake ..`` along with any (or none) of the following options depending on the build that is desired:

   * `-DMEMSAN=On` for memory sanitization

   * `-DASAN=On` for address sanitization

   * `-DVALGRIND=On` for valgrind use

   * `-DBUILD_ILP64=On` for 64-bit integer build

   * `-DCMAKE_BUILD_TYPE=Debug` or `Release`

   * `-DCOVERAGE=On` to build code coverage report

   * `-DBUILD_EXAMPLES=On` and `–DBUILD_GTEST=On` both of which are `On` by default

   * `-DBUILD_SHARED_LIBS=On` for a shared library build (`Off` by default)

   * `-DBUILD_DOC=On` to build the documentation. Use `cmake --build . --target doc` to build all documentation formats (or `doc_pdf`, `doc_html` to build only PDF or only HTML formats). Note that to build the Python documentation the PYTHONPATH environment variable must be set to aocl-da/python_interface/python_package

   * `-DINTERNAL_DOC=On` to build the internal documentation alongside the main one

   * `-DBUILD_SMP=On` to build using OpenMP and threaded BLAS (`On` by default)

   * `-DVECTORIZATION_REPORTS=On` to build with vectorization reports enabled

   * `-DDA_LOGGING=On` to enable debug printing

   * `-DBUILD_PYTHON=On` to build the Python interfaces

   * `-DCMAKE_AOCL_ROOT=<path to AOCL>` to specify a location for AOCL libraries. It is set to the environment variable `$AOCL_ROOT` by default

   * `-DCMAKE_INSTALL_PREFIX=<install path>`. Path where to install the library (using the build target `install` of step 5)

   * Any combination of `-DLAPACK_LIB`, `-DBLAS_LIB`, `-DLAPACK_INCLUDE_DIR` and `-DBLAS_INCLUDE_DIR` if you wish to override the use of `AOCL_ROOT` with specific choices of BLAS and LAPACK libraries and include directories. Care should be taken if you do this as there will be no checks for the correctness of the linked libraries.

   **Note** that not all the options are available in `Release` build mode

5. Type `cmake --build . --target all` (or `--target doc`, to build the documentation)

6. Run the tests or examples using ctest e.g. `ctest -V –R` followed by a string to find a particular set of tests

## Building on MS Windows using the MSVC toolchain

1. Install Visual Studio 2022 (the free version, which also comes with Ninja, CMake and clang) and the Intel Fortran compiler for MS Windows

2. Make sure you have set the `AOCL_ROOT` environment variable to your AOCL installation directory (e.g. `C:\Users\username\AOCL-4.0`) and `INTEL_FCOMPILER` to the location of the Intel fortran compiler (e.g. `C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows`). You can also update your `PATH` to take in the relevant BLAS and LAPACK libraries e.g.
`set PATH=C:\path\to\AOCL\amd-blis\lib\LP64;C:\path\to\AOCL\amd-libflame\lib\LP64;%PATH%`
It is most likely to work if BLAS and LAPACK are installed within your user directory rather than e.g. `Program Files`.

1. In your checkout create a directory called build

2. Open a Developer command prompt for VS2022 and navigate to the build directory

3. Type `"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"` to load the Intel compiler environment variables (if your compiler is installed elsewhere then you will need to edit this command accordingly).

4. Type `cmake .. -DCMAKE_Fortran_COMPILER=ifort` along with any (or none) of the following options depending on the build that is desired:

   * `-DBUILD_ILP64=On` for 64-bit integer build

   * `-DBUILD_EXAMPLES=On` and `–DBUILD_GTEST=On` both of which are on by default

   * `-DBUILD_SMP=On` to build using OpenMP and threaded BLAS (`On` by default)

   * `-DBUILD_SHARED_LIBS=On` for a shared library build (`Off` by default)

   * `-DCMAKE_AOCL_ROOT=<path to AOCL>` if you wish to specify a location for AOCL libraries without using environment variables

   * Any combination of `-DLAPACK_LIB`, `-DBLAS_LIB`, `-DLAPACK_INCLUDE_DIR` and `-DBLAS_INCLUDE_DIR` if you wish to override the use of `AOCL_ROOT` with specific choices of BLAS and LAPACK libraries and include directories. Care should be taken if you do this as there will be no checks for the correctness of the linked libraries.

   * `-DOpenMP_libomp_LIBRARY=<path to preferred OpenMP library>` to link a specific OpenMP library.

    **Note** that not all the options available in Linux are available in Windows.

    **Note** if you don't specify the Fortran compiler, Windows may default to ifx, which can cause linking issues.

5. Either:

* Open Visual Studio and load the `AOCL-DA.sln` file then build Debug or Release builds using the GUI, or

* In your powershell type `devenv .\AOCL-DA.sln /build "Debug"` to build the solution (change to Release as appropriate or use `cmake --build .`)

8. Depending on whether BLAS/LAPACK libraries are on your `PATH`, the compiled executables may only work if the BLAS and LAPACK dlls are in the same directory so you might need to copy `AOCL-LibBlis-Win-MT-dll.dll` and `AOCL-LibFlame-Win-MT-dll.dll` into, for example, `C:\path\to\aocl-da\build\tests\gtests\Debug`

9. Use ctest in your powershell/command prompt window to run the tests

10. It is possible that you will have issues with your MSVC installation, in particular numerous users have reported that installation of the Windows SDK is partly broken. You may need to fix this by correcting the INCLUDE paths that are set by the batch scripts when loading the command prompt e.g.:
    ```
    set INCLUDE=%INCLUDE%;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\winrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\cppwinrt
    ```
For further troubleshooting, you can also try updating LIB:

   ```
   set LIB=%LIB%;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64
   ```

11. Note that by default, the Windows build uses the MSVC compiler and the cmake supplied with Visual Studio generates Visual Studio makefiles. If you wish to use Clang with ifort, use the following commands:
    ```
    cmake -T ClangCL -DCMAKE_Fortran_COMPILER=ifort
    ```
to use Visual Studio's build system, or
    ```
    cmake -G Ninja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_Fortran_COMPILER=ifort ..
    (optionally with e.g. -DCMAKE_BUILD_TYPE=Debug) then build using ninja. You can also specify an install directory, using -DCMAKE_INSTALL_PREFIX, and build with ninja install. Depending on your system, you may also need to contact IT to enable registry editing on your machine.
    ```

12.  Note also that it is possible to build with shared libraries on Windows by adding `-DBUILD_SHARED_LIBS=On` to the cmake command. However, gtest is unable to discover tests and add them to the ctests due to an issue with gtest and shared library dependencies on Windows, so not all tests will run.

## Building on Windows using the GCC toolchain

1. Install GCC and gfortran compilers. The only way this has been tested so far is using MinGW, via MSYS. The following commands may be needed, to install the relevant compilers (within the MSYS command window):
    ```
    pacman -S mingw-w64-x86_64-gcc-fortran
    pacman -S --needed base-devel mingw-w64-x86_64-toolchain
    pacman -S mingw-w64-x86_64-cmake
    pacman -S mingw-w64-x86_64-ninja
    pacman -S mingw-w64-clang-x86_64-openmp
    pacman -S mingw-w64-x86_64-openmp
    ```

2. Make sure you have set the `AOCL_ROOT` environment variable to your AOCL installation directory (e.g. `C:\Users\username\AOCL-4.0`), gcc, g++ and gfortran (and clang too ,if you wish to use it) on your Windows `PATH` and `MINGW_BIN` to the location of the library `libgfortran` (e.g. `C:\msys64\mingw64\bin`)

3. In your checkout create a directory called build.

4. Open a standard Windows command prompt, navigate to build and type `cmake -G Ninja -DCMAKE_C_COMPILER=gcc` (or `clang`) `-DCMAKE_CXX_COMPILER=g++` (or `clang++`) `..` along with any (or none) of the following options depending on the build that is desired:

    * `-DBUILD_ILP64=On` for 64-bit integer build

    * `-DBUILD_EXAMPLES=On` and `–DBUILD_GTEST=On` both of which are on by default

    * `-DBUILD_SMP=On` to build using OpenMP and threaded BLAS (on by default)

    * `-DCMAKE_AOCL_ROOT=<path to AOCL>` if you wish to specify a location for AOCL libraries without using environment variables

    * `-DCMAKE_INSTALL_PREFIX=<install path>`. If you wish to install the library (using the command ninja install in step 5) you can specify an install path here.

    **Note** that not all the options available in Linux are available in Windows

5. Type `ninja` to build

6. Depending on your particular environment variables, the compiled executables may only work if the BLAS and LAPACK dlls are in the same directory so you might need to copy `AOCL-LibBlis-Win-MT-dll.dll` and `AOCL-LibFlame-Win-MT-dll.dll` into, for example, `C:\path\to\aocl-da\build\tests\gtests\Debug`

7. Use `ctest` in your command prompt window to run the tests

## Building the Python interfaces

To build the Python interfaces, use `-DBUILD_PYTHON=On` (note that this will only work with shared library builds). You will need PyBind11, which can be installed using `pip install pybind11`. On Windows you may also need to set the `CMAKE_PREFIX_PATH` to point to the location of you pybind11 installation, e.g.  `C:\path\to\your\python-environment\site-packages\pybind11\share\cmake\pybind11`.

By default, cmake will compile the bindings but will not install them. If you set `-DCMAKE_INSTALL_PREFIX=<install path>` in your configure step and run `cmake --build . --target install`, then cmake will also create a Python wheel, `aoclda-*.whl`, where `*` depends on your system. This wheel can be installed using `pip install aoclda-*.whl`.

When using the bindings on Windows with MSVC/Clang and ifort, the Intel Fortran runtime must also be available. This can be done by setting the environment variable `FORTRAN_RUNTIME` to the location of the DLL. Alternatively,. you can also build using AOCC on Windows. This is somewhat involved though; see the corresponding Jenkins build for full details.