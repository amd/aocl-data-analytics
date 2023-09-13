This file contains instructions for compiling the AOCL-DA C++ example programs.
The precise paths to your AOCL library installation will depend on your system and will need to be amended accordingly.

Compiling examples on Linux
===========================
AOCL_ROOT is taken to be the path to your AOCL library installation, for example /opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc.
INT_LIB is either LP64 or ILP64 for 32 and 64 bit integers.

To compile and link to static AOCL libraries using g++:

g++ <example name>.cpp -I <AOCL_ROOT>/include_<INT_LIB>
    <AOCL_ROOT>/lib_<INT_LIB>/libaocl-da.a <AOCL_ROOT>/lib_<INT_LIB>/libflame.a
    <AOCL_ROOT>/lib_<INT_LIB>/libblis-mt.a -lgfortran -lgomp

To compile and link to static AOCL libraries using clang++:

clang++ <example name>.cpp -I <AOCL_ROOT>/include_<INT_LIB>
    <AOCL_ROOT>/lib_<INT_LIB>/libaocl-da.a <AOCL_ROOT>/lib_<INT_LIB>/libflame.a
    <AOCL_ROOT>/lib_<INT_LIB>/libblis-mt.a -lflang -lomp

To compile and link to dynamic AOCL libraries using g++:

g++ <example name>.cpp -I <AOCL_ROOT>/include_<INT_LIB>
    -L<AOCL_ROOT>/lib_<INT_LIB> -laocl-da -lflame -lblis-mt
    -lgfortran -lgomp

To compile and link to dynamic AOCL libraries using clang++:

clang++ <example name>.cpp -I <AOCL_ROOT>/include_<INT_LIB>
    -L<AOCL_ROOT>/lib_<INT_LIB> -laocl-da -lflame -lblis-mt
    -lflang -lomp

(note that for dynamic linking you will need to update LD_LIBRARY_PATH e.g. export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<AOCL_ROOT>/lib_<INT_LIB>)

Compiling examples on Windows
=============================

AOCL_ROOT is taken to be the path to your AOCL library installation, for example C:\Users\<your_name>\AMD\AOCL
INT_LIB is either LP64 or ILP64 for 32 and 64 bit integers.

cl <example_name>.cpp /I <AOCL_ROOT>\include\<INT_LIB> /EHsc /MD
   <AOCL_ROOT>\aocl-da\lib\<INT_LIB>\aocl-da.lib
   <AOCL_ROOT>\amd-libflame\lib\<INT_LIB>\AOCL-LibFlame-Win-MT-dll.lib
   <AOCL_ROOT>\amd-blis\lib\<INT_LIB>\AOCL-LibBlis-Win-MT-dll.lib

The same command should work with cl replaced by clang-cl and linking statically using /MT

(note that you should ensure the folders containing the libraries to be linked are on your Windows PATH e.g. using set PATH=%PATH%;C:\<path_to_BLAS_and_LAPACK>)