# Data-analytics_exp

AOCL-da is built with cmake, with supported compilers GNU and AOCC on linux and MSVC (with ifort) on Windows. 

AOCL-DA is dependent on BLAS and LAPACK currently, and ultimately may be dependent on other AOCL libraries (such as sparse). 

## Building on Linux 

1. You will need to have AOCL Blis and LibFlame installed somewhere 

2. In the CMakeLists.txt file find the following lines: 

  if($ENV{USER} MATCHES "bmarteau") 

      set(ENV{AOCL_ROOT} /home/bmarteau/amd/aocl/4.0) 

  endif() 

and add similar lines to point cmake to where your AOCL build is installed 

3. In your checkout create a directory called build and navigate to it 

4. Type cmake .. along with any (or none) of the following options depending on the build that is desired: 

* -DCHECK_MEM=On for address sanitization 

* -DBUILD_ILP64=On for 64-bit integer build 

* -DCMAKE_BUILD_TYPE=Debug or Release 

* -DCOVERAGE=On to build code coverage report 

* -DBUILD_EXAMPLES and –DBUILD_GTEST both of which are on by default

* -DBUILD_DOC=On to build the documentation

* -DBUILD_SMP=On to build using OpenMP and threaded BLIS (on by default)

5. Type make (or make doc, to make the documentation)

6. Run the tests or examples using ctest e.g. ctest -V –R followed by a string to find a particular set of tests 

## Building on Windows 

1. Install Visual Studio 2022 (the free version), the Intel Fortran compiler for Windows and cmake 

2. In the CMakeLists.txt file find the following lines: 

  if($ENV{USERNAME} MATCHES "ehopkins") 

    set(ENV{AOCL_INSTALL} "C:/Users/ehopkins/AOCL-4.0") 

    set(ENV{IFORT_LIBS} "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/compiler/lib/intel64_win") 

  endif() 

and add similar lines to point cmake to where your AOCL build and Fortran compiler are installed 

3. In your checkout create a directory called build 

4. Open a Developer Powershell for VS2022 window and navigate to the build directory 

5. Type cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell' to load the Intel compiler environment variables (if your compiler is installed elsewhere then you will need to edit this command accordingly) 

6. Type cmake .. along with any (or none) of the following options depending on the build that is desired: 

* -DBUILD_ILP64=On for 64-bit integer build 

* -DBUILD_EXAMPLES and –DBUILD_GTEST both of which are on by default

* -DBUILD_SMP=On to build using OpenMP and threaded BLIS (on by default)

Note that not all the options available in Linux are available in Windows 

7. Either: 

* Open Visual Studio and load the AOCL-DA.sln file then build Debug or Release builds using the GUI, or 

* In your powershell type devenv .\AOCL-DA.sln /build "Debug" to build the solution (change to Release as appropriate) 

8. The compiled executables only work if the Blis and Flame dlls are in the same directory so you need to copy AOCL-LibBlis-Win-MT-dll.dll and AOCL-LibFlame-Win-MT-dll.dll into, for example, C:\path\to\aocl-da\build\tests\gtests\Debug (if you can figure out a better way which avoids this issue, then let Edvin know!) 

9. Use ctest in your powershell window to run the tests 