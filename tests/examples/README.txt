This file contains instructions for compiling the AOCL-DA C++ example programs.
The precise paths to your AOCL library installation will depend on your system and will need to be amended accordingly.

Compiling examples on Linux
===========================
INT_LIB is either LP64 or ILP64 for 32 and 64 bit integers.

To compile and link to static AOCL libraries using g++:

g++ <example name>.cpp -I /<path to aocl-da headers>/include_<INT_LIB>
    /<path to aocl-da>/lib_<INT_LIB>/libaocl-da.a /<path to libflame>/lib_<INT_LIB>/libflame.a
    /<path to blis>/lib_<INT_LIB>/libblis-mt.a -lgfortran -lgomp

To compile and link to static AOCL libraries using clang++:

clang++ <example name>.cpp -I /<path to aocl-da headers>/include_<INT_LIB>
    /<path to aocl-da>/lib_<INT_LIB>/libaocl-da.a /<path to libflame>/lib_<INT_LIB>/libflame.a
    /<path to blis>/lib_<INT_LIB>/libblis-mt.a -lflang -lomp

To compile and link to dynamic AOCL libraries using g++:

g++ <example name>.cpp -I /<path to aocl-da headers>/include_<INT_LIB>
    -L/<path to aocl-da and similarly for blis and flame>/lib_<INT_LIB> -laocl-da -lflame -lblis-mt
    -lgfortran -lgomp

To compile and link to dynamic AOCL libraries using clang++:

clang++ <example name>.cpp -I /<path to aocl-da headers>/include_<INT_LIB>
    -L/<path to aocl-da and similarly for blis and flame>/lib_<INT_LIB> -laocl-da -lflame -lblis-mt
    -lflang -lomp

(note that for dynamic linking you will need to update LD_LIBRARY_PATH e.g. export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<AOCL_ROOT>/lib_<INT_LIB>)

Compiling examples on Windows
=============================

INT_LIB is either LP64 or ILP64 for 32 and 64 bit integers.

cl <example_name>.cpp /I \<path to aocl-da headers>\include\<INT_LIB> /EHsc /MD
   \<path to>\aocl-da\lib\<INT_LIB>\aocl-da.lib
   \<path to>\amd-libflame\lib\<INT_LIB>\AOCL-LibFlame-Win-MT-dll.lib
   \<path to>\amd-blis\lib\<INT_LIB>\AOCL-LibBlis-Win-MT-dll.lib

The same command should work with cl replaced by clang-cl and linking statically using /MT

(note that you should ensure the folders containing the libraries to be linked are on your Windows PATH e.g. using set PATH=%PATH%;C:\<path_to_BLAS_and_LAPACK>)