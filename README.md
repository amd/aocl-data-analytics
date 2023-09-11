# Data-analytics_exp

AOCL-da is built with cmake, with supported compilers GNU and AOCC on linux and MSVC (with ifort) on Windows. 

AOCL-DA is dependent on BLAS and LAPACK currently, and ultimately may be dependent on other AOCL libraries (such as sparse). 

## Building on Linux 

1. You will need to have AOCL Blis and LibFlame installed somewhere 

2. Make sure you have set the environment variable AOCL_ROOT to where AOCL libraries are installed eg. /home/username/amd/aocl/4.0

3. In your checkout create a directory called build and navigate to it 

4. Type cmake .. along with any (or none) of the following options depending on the build that is desired: 

* -DMEMSAN=On for memory sanitization

* -DASAN=On for address sanitization

* -DVALGRIND=On for valgrind use

* -DBUILD_ILP64=On for 64-bit integer build 

* -DCMAKE_BUILD_TYPE=Debug or Release 

* -DCOVERAGE=On to build code coverage report 

* -DBUILD_EXAMPLES and –DBUILD_GTEST both of which are on by default

* -DBUILD_DOC=On to build the documentation

* -DINTERNAL_DOC=ON to build the internal documentation alongside the main one

* -DBUILD_SMP=On to build using OpenMP and threaded BLIS (on by default)

* -DVECTORIZATION_REPORTS=On to build with vectorizastion reports enabled

* -DDA_LOGGING=On to enable debug printing

* -DCMAKE_AOCL_ROOT=<path to AOCL> if you wish to specify a location for AOCL libraries without using environment variables

* -DCMAKE_INSTALL_PREFIX=<install path>. If you wish to install the library (using the command make install in step 5) you can specify an install path here.

1. Type cmake --build . --target all (or --target doc, to build the documentation)

2. Run the tests or examples using ctest e.g. ctest -V –R followed by a string to find a particular set of tests 

## Building on Windows using the MSVC toolchain

1. Install Visual Studio 2022 (the free version, which also comes with Ninja, CMake and clang) and the Intel Fortran compiler for Windows

2. Make sure you have set the AOCL_ROOT environment variable to your AOCL installation directory (e.g. C:\Users\username\AOCL-4.0) and INTEL_FCOMPILER to the location of the Intel fortran compiler (e.g. C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows)

3. In your checkout create a directory called build

4. Open a Developer Powershell for VS2022 window and navigate to the build directory 

5. Type cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell' to load the Intel compiler environment variables (if your compiler is installed elsewhere then you will need to edit this command accordingly). Note that you can also do this in a Developer Command Prompt, in which case simply use the command "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

6. Type cmake .. along with any (or none) of the following options depending on the build that is desired: 

* -DBUILD_ILP64=On for 64-bit integer build 

* -DBUILD_EXAMPLES and –DBUILD_GTEST both of which are on by default

* -DBUILD_SMP=On to build using OpenMP and threaded BLIS (on by default)

* -DCMAKE_AOCL_ROOT= <path to AOCL> if you wish to specify a location for AOCL libraries without using environment variables

Note that not all the options available in Linux are available in Windows

7. Either: 

* Open Visual Studio and load the AOCL-DA.sln file then build Debug or Release builds using the GUI, or 

* In your powershell type devenv .\AOCL-DA.sln /build "Debug" to build the solution (change to Release as appropriate) 

8. Depending on your particular environment variables, the compiled executables may only work if the Blis and Flame dlls are in the same directory so you might need to copy AOCL-LibBlis-Win-MT-dll.dll and AOCL-LibFlame-Win-MT-dll.dll into, for example, C:\path\to\aocl-da\build\tests\gtests\Debug

9. Use ctest in your powershell/command prompt window to run the tests

10. Note that by default, the Windows build uses the MSVC compiler and the cmake supplied with Visual Studio generates Visual Studio makefiles. If you wish to use Clang with ifort, use the following command:
cmake -G Ninja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl ..
(optionally with e.g. -DCMAKE_BUILD_TYPE=Debug) then build using ninja. You can also specify an install directory, using -DCMAKE_INSTALL_PREFIX, and build with ninja install.

11. Note also that it is possible to build with shared libraries on Windows by adding -DBUILD_SHARED_LIBS=ON to the cmake command. However, you must ensure that your Windows PATH environemnt variable contains all the folders where cmake will put dlls (gtest, da_core, lbfgsb, etc.). Furthermore, gtest is unable to discover tests and add them to the ctests due to an issue with gtest and shared library dependencies on Windows.

## Building on Windows using the GCC toolchain

1. Install GCC and gfortran compilers. The only way this has been tested so far is using MinGW, via MSYS. The following commands may be needed, to install the relevant compilers (within the MSYS command window):

* pacman -S mingw-w64-x86_64-gcc-fortran
* pacman -S --needed base-devel mingw-w64-x86_64-toolchain
* pacman -S mingw-w64-x86_64-cmake
* pacman -S mingw-w64-x86_64-ninja
* pacman -S mingw-w64-clang-x86_64-openmp
* pacman -S mingw-w64-x86_64-openmp

2. Make sure you have set the AOCL_ROOT environment variable to your AOCL installation directory (e.g. C:\Users\username\AOCL-4.0), gcc, g++ and gfortran (and clang too ,if you wish to use it) on your Windows PATH and MINGW_BIN to the location of the library libgfortran (e.g. C:\msys64\mingw64\bin)

3. In your checkout create a directory called build.

4. Open a standard Windows command prompt, navigate to build and type cmake -G Ninja -DCMAKE_C_COMPILER=gcc (or clang) -DCMAKE_CXX_COMPILER=g++ (or clang++) .. along with any (or none) of the following options depending on the build that is desired: 

* -DBUILD_ILP64=On for 64-bit integer build 

* -DBUILD_EXAMPLES and –DBUILD_GTEST both of which are on by default

* -DBUILD_SMP=On to build using OpenMP and threaded BLIS (on by default)

* -DCMAKE_AOCL_ROOT= <path to AOCL> if you wish to specify a location for AOCL libraries without using environment variables

* -DCMAKE_INSTALL_PREFIX=<install path>. If you wish to install the library (using the command ninja install in step 5) you can sepcify an install path here.

Note that not all the options available in Linux are available in Windows

5. Type ninja to build

6. Depending on your particular environment variables, the compiled executables may only work if the Blis and Flame dlls are in the same directory so you might need to copy AOCL-LibBlis-Win-MT-dll.dll and AOCL-LibFlame-Win-MT-dll.dll into, for example, C:\path\to\aocl-da\build\tests\gtests\Debug

7. Use ctest in your powershell/command prompt window to run the tests