# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: 1.
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer. 2. Redistributions in binary
# form must reproduce the above copyright notice, this list of conditions and
# the following disclaimer in the documentation and/or other materials provided
# with the distribution. 3. Neither the name of the copyright holder nor the
# names of its contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# Flags that are common for GNU and Clang (C, C++) and selectively tested for Flang / gfortran
set(FLAGS_COMMON "
    -Wall
    -Wextra
    -Wno-sign-conversion
    -Wpedantic
    -Wunused
    -Wunused-but-set-variable
    -Wunused-function
    -Wunused-parameter
    -Wunused-variable
    -fno-fast-math"
)
set(FLAGS_COMMON_TO_TEST "-Wno-macro-redefined -Wno-nan-infinity-disabled -Wno-variadic-macro-arguments-omitted -Wno-gnu-zero-variadic-macro-arguments")
set(FLAGS_RELEASE "-falign-loops=32 -fmacro-prefix-map=${PROJECT_SOURCE_DIR}=. -fopenmp-simd")
set(FLAGS_DEBUG "-O0 -gdwarf-5 -g3")

# Flags that are common for Fortran but are selectively tested for Flang / gfortran
set(FLAGS_COMMON_FORTRAN "-fno-fast-math")
set(FLAGS_RELEASE_FORTRAN "-fopenmp-simd -falign-loops=32 -DNDEBUG")
set(FLAGS_DEBUG_FORTRAN "${FLAGS_DEBUG}")

AppendCFlags(C_FLAGS_COMMON "${FLAGS_COMMON_TO_TEST}")
AppendCXXFlags(CXX_FLAGS_COMMON "${FLAGS_COMMON_TO_TEST}")

# Fortran flags
if(COMPILER_FAMILY STREQUAL "LLVM")
# LLVM -------------------------------------------------------------------------
   # Check flang, flang-new, ... and add supported flags only
   # Tunning flags are added by the external packages
        set(CXX_FLAGS_RELEASE "${FLAGS_COMMON} ${CXX_FLAGS_COMMON} ${FLAGS_RELEASE}")
        set(C_FLAGS_RELEASE "${FLAGS_COMMON} ${C_FLAGS_COMMON} ${FLAGS_RELEASE}")
        AppendFortranFlags(Fortran_FLAGS_RELEASE "${FLAGS_COMMON_FORTRAN} ${FLAGS_RELEASE_FORTRAN}")
        set(CXX_FLAGS_DEBUG "${FLAGS_COMMON} ${CXX_FLAGS_COMMON} ${FLAGS_DEBUG}")
        set(C_FLAGS_DEBUG "${FLAGS_COMMON} ${C_FLAGS_COMMON} ${FLAGS_DEBUG}")
        AppendFortranFlags(Fortran_FLAGS_DEBUG "${FLAGS_COMMON_FORTRAN} -O0 -gdwarf-5 -g")
        set(CXX_FLAGS_RELWITHDEBINFO "${FLAGS_COMMON} ${CXX_FLAGS_COMMON} ${FLAGS_RELEASE}")
        set(C_FLAGS_RELWITHDEBINFO "${FLAGS_COMMON} ${C_FLAGS_COMMON} ${FLAGS_RELEASE}")
        AppendFortranFlags(Fortran_FLAGS_RELWITHDEBINFO "${FLAGS_COMMON_FORTRAN} -fopenmp-simd")
elseif(COMPILER_FAMILY STREQUAL "GNU")
# GNU --------------------------------------------------------------------------
        set(CXX_FLAGS_RELEASE "${FLAGS_COMMON} ${CXX_FLAGS_COMMON} ${FLAGS_RELEASE}")
        set(C_FLAGS_RELEASE "${FLAGS_COMMON} ${C_FLAGS_COMMON} ${FLAGS_RELEASE}")
        AppendFortranFlags(Fortran_FLAGS_RELEASE "${FLAGS_COMMON_FORTRAN} ${FLAGS_RELEASE_FORTRAN}")
        set(CXX_FLAGS_DEBUG "${FLAGS_COMMON} ${CXX_FLAGS_COMMON} ${FLAGS_DEBUG}")
        set(C_FLAGS_DEBUG "${FLAGS_COMMON} ${C_FLAGS_COMMON} ${FLAGS_DEBUG}")
        AppendFortranFlags(Fortran_FLAGS_DEBUG "${FLAGS_COMMON_FORTRAN} ${FLAGS_DEBUG_FORTRAN}")
        set(CXX_FLAGS_RELWITHDEBINFO "${FLAGS_COMMON} ${CXX_FLAGS_COMMON} ${FLAGS_RELEASE}")
        set(C_FLAGS_RELWITHDEBINFO "${FLAGS_COMMON} ${C_FLAGS_COMMON} ${FLAGS_RELEASE}")
        AppendFortranFlags(Fortran_FLAGS_RELWITHDEBINFO "${FLAGS_COMMON_FORTRAN} -fopenmp-simd")
elseif(COMPILER_FAMILY STREQUAL "LLVM-CL")
# Clang-cl (windows) -----------------------------------------------------------
        set(C_FLAGS_RELEASE "/EHsc /O2") # /O2
        set(CXX_FLAGS_RELEASE "/std:c++17 /EHsc /O2")
        set(Fortran_FLAGS_RELEASE "") # -O2 -DNDEBUG
        set(C_FLAGS_DEBUG "/W3 /Od -gdwarf") # /Od
        set(CXX_FLAGS_DEBUG "/std:c++17 /W3 /Od -gdwarf")
        set(Fortran_FLAGS_DEBUG "") # -O0 -g
        set(C_FLAGS_RELWITHDEBINFO "/EHsc") # /O2 /Zi
        set(CXX_FLAGS_RELWITHDEBINFO "/std:c++17 /EHsc")
        set(Fortran_FLAGS_RELWITHDEBINFO "") # -O2 -g
else()
    message(FATAL_ERROR "Unknown compiler family ${COMPILER_FAMILY}!")
endif()

if(VECTORIZATION_REPORTS)
  if(COMPILER_FAMILY STREQUAL "GNU")
      set( FLAG_VECTORIZATION "-fopt-info-vec-all=vectorization.txt")
  elseif (COMPILER_FAMILY STREQUAL "LLVM")
      set( FLAG_VECTORIZATION "
        -Rpass=loop-vectorize
        -Rpass-analysis=loop-vectorize
        -Rpass-missed=loop-vectorize
        -gline-tables-only
        -gcolumn-info"
      )
  elseif(COMPILER_FAMILY STREQUAL "LLVM-CL")
      set( FLAG_VECTORIZATION "/Qvec-report:2")
  endif()
  foreach(_flag_var FLAG_VECTORIZATION)
    separate_arguments(${_flag_var} UNIX_COMMAND "${${_flag_var}}")
  endforeach()
  add_compile_options(
    "$<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:${FLAG_VECTORIZATION}>"
  )
  # Try to add it to Fortran
  AppendFortranFlags(Fortran_VECTORIZATION "${FLAG_VECTORIZATION}")
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:Fortran>:${Fortran_VECTORIZATION}>"
  )
endif()

if(ASAN)
  set(ASAN_FLAGS "-fsanitize=address")
  add_compile_options( "$<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:${ASAN_FLAGS}>")
  # Keep fortran flag optional to accomodate flang
  AppendFortranFlags(Fortran_ASAN_FLAGS "${ASAN_FLAGS}")
  add_compile_options( "$<$<COMPILE_LANGUAGE:Fortran>:${Fortran_ASAN_FLAGS}>")
  # The ASan runtime must still be linked into final executables/shared libs
  link_libraries("-fsanitize=address")
endif()

if(COVERAGE)
  # Only enabled for GCC
  set(COVERAGE_FLAGS "-fprofile-arcs -ftest-coverage")
  foreach(_flag_var COVERAGE_FLAGS)
    separate_arguments(${_flag_var} UNIX_COMMAND "${${_flag_var}}")
  endforeach()
  add_compile_options(${COVERAGE_FLAGS})
  link_libraries(gcov)
endif()

# Convert the space-separated flag strings into CMake lists so that each flag is
# emitted as a separate compiler argument. Otherwise the whole string is passed
# as a single quoted token (e.g. "-Wall -Wextra ...") in compile_commands.json.
foreach(_flag_var
    C_FLAGS_DEBUG CXX_FLAGS_DEBUG
    C_FLAGS_RELEASE CXX_FLAGS_RELEASE
    C_FLAGS_RELWITHDEBINFO CXX_FLAGS_RELWITHDEBINFO)
  separate_arguments(${_flag_var} UNIX_COMMAND "${${_flag_var}}")
endforeach()

# Add the flags
add_compile_options(
  "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:C>>:${C_FLAGS_DEBUG}>"
  "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:${CXX_FLAGS_DEBUG}>"
  "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:Fortran>>:${Fortran_FLAGS_DEBUG}>"
  "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C>>:${C_FLAGS_RELEASE}>"
  "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:${CXX_FLAGS_RELEASE}>"
  "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:Fortran>>:${Fortran_FLAGS_RELEASE}>"
  "$<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:C>>:${C_FLAGS_RELWITHDEBINFO}>"
  "$<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CXX>>:${CXX_FLAGS_RELWITHDEBINFO}>"
  "$<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:Fortran>>:${Fortran_FLAGS_RELWITHDEBINFO}>"
)

# Pretty print
set(_deb_cxx_flags "${CXX_FLAGS_DEBUG} ${FLAG_VECTORIZATION} ${ASAN_FLAGS} ${COVERAGE_FLAGS}")
set(_deb_c_flags "${C_FLAGS_DEBUG} ${FLAG_VECTORIZATION} ${ASAN_FLAGS} ${COVERAGE_FLAGS}")
set(_deb_fortran_flags "${Fortran_FLAGS_DEBUG} ${Fortran_VECTORIZATION} ${Fortran_ASAN_FLAGS} ${Fortran_COVERAGE_FLAGS}")
set(_rel_cxx_flags "${CXX_FLAGS_RELEASE} ${FLAG_VECTORIZATION} ${ASAN_FLAGS} ${COVERAGE_FLAGS}")
set(_rel_c_flags "${C_FLAGS_RELEASE} ${FLAG_VECTORIZATION} ${ASAN_FLAGS} ${COVERAGE_FLAGS}")
set(_rel_fortran_flags "${Fortran_FLAGS_RELEASE} ${Fortran_VECTORIZATION} ${Fortran_ASAN_FLAGS} ${Fortran_COVERAGE_FLAGS}")
set(_rde_cxx_flags "${CXX_FLAGS_RELWITHDEBINFO} ${FLAG_VECTORIZATION} ${ASAN_FLAGS} ${COVERAGE_FLAGS}")
set(_rde_c_flags "${C_FLAGS_RELWITHDEBINFO} ${FLAG_VECTORIZATION} ${ASAN_FLAGS} ${COVERAGE_FLAGS}")
set(_rde_fortran_flags "${Fortran_FLAGS_RELWITHDEBINFO} ${Fortran_VECTORIZATION} ${Fortran_ASAN_FLAGS} ${Fortran_COVERAGE_FLAGS}")

if(NOT CMAKE_REQUIRED_QUIET)
    if (NOT WIN32)
        message( NOTICE "Explicit compiler flags: [${CMAKE_BUILD_TYPE}]")
        if(CMAKE_BUILD_TYPE MATCHES "Debug")
            PrettyPrintFlags("CXX" "${_deb_cxx_flags}")
            PrettyPrintFlags("C" "${_deb_c_flags}")
            if (BUILD_FORTRAN)
              PrettyPrintFlags("Fortran" "${_deb_fortran_flags}")
            endif()
        elseif(CMAKE_BUILD_TYPE MATCHES "Release")
            PrettyPrintFlags("CXX" "${_rel_cxx_flags}")
            PrettyPrintFlags("C" "${_rel_c_flags}")
            if (BUILD_FORTRAN)
              PrettyPrintFlags("Fortran" "${_rel_fortran_flags}")
            endif()
        else() # RelWithDebInfo
            PrettyPrintFlags("CXX" "${_rde_cxx_flags}")
            PrettyPrintFlags("C" "${_rde_c_flags}")
            if (BUILD_FORTRAN)
              PrettyPrintFlags("Fortran" "${_rde_fortran_flags}")
            endif()
        endif()
    else()
        message( NOTICE "Explicit compiler flags: [Debug]")
            PrettyPrintFlags("CXX" "${_deb_cxx_flags}")
            PrettyPrintFlags("C" "${_deb_c_flags}")
            if(BUILD_FORTRAN)
              PrettyPrintFlags("Fortran" "${_deb_fortran_flags}")
            endif()
        message( NOTICE "Explicit compiler flags: [Release]")
            PrettyPrintFlags("CXX" "${_rel_cxx_flags}")
            PrettyPrintFlags("C" "${_rel_c_flags}")
            if(BUILD_FORTRAN)
              PrettyPrintFlags("Fortran" "${_rel_fortran_flags}")
            endif()
        message( NOTICE "Explicit compiler flags: [RelWithDebInfo]")
            PrettyPrintFlags("CXX" "${_rde_cxx_flags}")
            PrettyPrintFlags("C" "${_rde_c_flags}")
            if(BUILD_FORTRAN)
              PrettyPrintFlags("Fortran" "${_rde_fortran_flags}")
            endif()
    endif()
endif()
