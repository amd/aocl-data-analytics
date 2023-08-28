enable_language(Fortran)

if(NOT WIN32)
    set(CMAKE_Fortran_FLAGS -cpp -Wno-unused-parameter)
else()
    set(CMAKE_Fortran_PREPROCESS ON)
endif()

if(BUILD_ILP64)
    if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
        list(APPEND CMAKE_Fortran_FLAGS -finteger-4-integer-8)
    elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
        list(APPEND CMAKE_Fortran_FLAGS -fdefault-integer-8)
    elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
        list(APPEND CMAKE_Fortran_FLAGS -i8)
    else()
        message(FATAL_ERROR "Unknown Fortran compiler!")
    endif()
endif()

if(BUILD_SMP)
   if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
     list(APPEND CMAKE_Fortran_FLAGS -qopenmp)
   elseif(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
     list(APPEND CMAKE_Fortran_FLAGS -fopenmp)
   endif()
endif()

if(CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
    link_libraries("-lflang")
endif()

if(WIN32 AND NOT CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    find_library(LIBGFORTRAN NAMES libgfortran libgfortran-5 HINTS ${MINGW_PATH} DOC "gfortran library")
endif()

list(JOIN CMAKE_Fortran_FLAGS " " CMAKE_Fortran_FLAGS)

set(LBFGSB_SRC  ../lbfgsb_mod.F90 ../linpack.F90 ../timer.F90 ../lbfgsb.F90)

# Build the double version
add_subdirectory(double_cmake)

# build the single version
add_subdirectory(single_cmake)

# Suppress warnings
if(NOT WIN32)
    include(${CMAKE_SOURCE_DIR}/cmake/Suppressions_external.cmake)
endif()

message(
    NOTICE
    "Fortran Compiler ID     : ${CMAKE_Fortran_COMPILER_ID}"
)

message(
    NOTICE
    "Fortran Compiler flags  : ${CMAKE_Fortran_FLAGS}"
)