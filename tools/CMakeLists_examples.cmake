# Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(aocl-da_examples LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ##############################################################################
# options
option(BUILD_ILP64 "ILP64 support" OFF)
option(BUILD_SMP "Enable Shared Memory parallelism" ON)

# Set paths to AOCL-Utils, BLAS, LAPACK and AOCL-Sparse installations.
set(CMAKE_AOCL_ROOT
  $ENV{AOCL_ROOT}
  CACHE
  STRING
  "AOCL_ROOT directory to be used to find AOCL BLAS/LAPACK/SPARSE/UTILS libraries"
)
if(CMAKE_AOCL_ROOT STREQUAL "")
  message(
    FATAL_ERROR
    "CMAKE_AOCL_ROOT is empty. Either set environment variable AOCL_ROOT or set -DCMAKE_AOCL_ROOT=<path_to_AOCL_libs>."
  )
endif()

find_package(OpenMP REQUIRED)

if(BUILD_ILP64)
  set(INT_LIB "ILP64")
  set(AOCLDA_ILP64 -DAOCLDA_ILP64)
else()
  set(INT_LIB "LP64")
endif()

# ##############################################################################
# Location of the AOCL-DA installation. Either set AOCLDA_ROOT specifically or
# inherit AOCL_ROOT where AOCL-DA artifacts would be as part of AOCL
# installation
set(CMAKE_AOCLDA_ROOT
  $ENV{AOCLDA_ROOT}
  CACHE STRING "AOCLDA_ROOT directory to be used to find DA artifacts")
if(CMAKE_AOCLDA_ROOT STREQUAL "")
  message(
    WARNING
    "AOCLDA_ROOT was not set. Will search for it in main CMAKE_AOCL_ROOT directory."
  )
  set(CMAKE_AOCLDA_ROOT ${CMAKE_AOCL_ROOT})
endif()

find_library(
  AOCL_DA
  HINTS ${CMAKE_AOCLDA_ROOT} ${CMAKE_AOCLDA_ROOT}/da ${CMAKE_AOCLDA_ROOT}/amd-da
  NAMES aocl-da
  PATH_SUFFIXES "lib/${INT_LIB}" "lib_${INT_LIB}" "lib" REQUIRED)
find_path(
  DA_INCLUDE_DIR
  NAMES "aoclda.h"
  HINTS ${CMAKE_AOCLDA_ROOT} ${CMAKE_AOCLDA_ROOT}/da ${CMAKE_AOCLDA_ROOT}/amd-da
  PATH_SUFFIXES "include/${INT_LIB}" "include_${INT_LIB}" "include" REQUIRED)

# ##############################################################################
# Other AOCL dependencies
if(WIN32)
  set(CMAKE_FIND_LIBRARY_PREFIXES "")
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
  set(BLAS_NAME "AOCL-LibBlis-Win-MT-dll")
  set(LAPACK_NAME "AOCL-LibFlame-Win-MT-dll")
  set(UTILS_NAME "libaoclutils")
  set(SPARSE_NAME "aoclsparse")

  set(BLAS_PATH "${CMAKE_AOCL_ROOT}/amd-blis/lib/${INT_LIB}")
  set(LAPACK_PATH "${CMAKE_AOCL_ROOT}/amd-libflame/lib/${INT_LIB}")
  set(UTILS_PATH "${CMAKE_AOCL_ROOT}/amd-utils/lib")
  set(SPARSE_PATH "${CMAKE_AOCL_ROOT}/amd-sparse/lib/${INT_LIB}/shared")
else() # Linux
  if(BUILD_SMP)
    set(BLAS_NAME "blis-mt")
  else()
    set(BLAS_NAME "blis")
  endif()
  set(LAPACK_NAME "flame")
  set(UTILS_NAME "aoclutils")
  set(SPARSE_NAME "aoclsparse")
  set(DLP_NAME "aocl-dlp")

  set(BLAS_PATH ${CMAKE_AOCL_ROOT}/lib_${INT_LIB})
  set(LAPACK_PATH ${CMAKE_AOCL_ROOT}/lib_${INT_LIB})
  set(SPARSE_PATH ${CMAKE_AOCL_ROOT}/lib_${INT_LIB})
  set(UTILS_PATH ${CMAKE_AOCL_ROOT}/lib_${INT_LIB})
  set(DLP_PATH ${CMAKE_AOCL_ROOT}/lib_${INT_LIB})
endif()

find_library(
  BLAS name ${BLAS_NAME}
  PATHS ${BLAS_PATH} REQUIRED
  NO_DEFAULT_PATH)

find_library(
  LAPACK name ${LAPACK_NAME}
  PATHS ${LAPACK_PATH} REQUIRED
  NO_DEFAULT_PATH)

find_library(
  SPARSE name ${SPARSE_NAME}
  PATHS ${SPARSE_PATH} REQUIRED
  NO_DEFAULT_PATH)

find_library(UTILS name ${UTILS_NAME} PATHS ${UTILS_PATH})

if(NOT WIN32)
  find_library(
    DLP name ${DLP_NAME}
    PATHS ${DLP_PATH} REQUIRED
    NO_DEFAULT_PATH)

  # AOCL-DLP static library must be linked with --whole-archive
  if(DLP MATCHES "\\.a$")
    set(DLP_LINK "$<LINK_LIBRARY:WHOLE_ARCHIVE,${DLP}>")
  else()
    set(DLP_LINK "${DLP}")
  endif()
endif()

enable_language(Fortran)
# Aux function to distinguish between AOCC Flang and upstream LLVM Flang-new
# Sets OUT_VAR to TRUE if the current Fortran compiler is AMD's AOCC flang,
function(IsAOCCFlang OUT_VAR)
  execute_process(
    COMMAND "${CMAKE_Fortran_COMPILER}" -v
    OUTPUT_VARIABLE _ver_stdout
    ERROR_VARIABLE _ver_stderr
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  # flang-new -v writes to stderr; capture both to be safe
  if("${_ver_stdout}\n${_ver_stderr}" MATCHES "AOCC|AMD AOCC aof")
    set(${OUT_VAR} TRUE PARENT_SCOPE)
  else()
    set(${OUT_VAR} FALSE PARENT_SCOPE)
  endif()
endfunction()

if(NOT WIN32)
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "Flang") # AOCC 5.x
    set(FORTRAN_RUNTIME "flang")
  elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(FORTRAN_RUNTIME "gfortran")
  elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "LLVMFlang")
    IsAOCCFlang(IS_AOCC_FLANG)
    if(IS_AOCC_FLANG)
      set(FORTRAN_RUNTIME "flang") # AOCC AOF 6+
    else()
      set(FORTRAN_RUNTIME "FortranRuntime") # upstream LLVM 19+ Flang-new
    endif()
  else()
    message(FATAL_ERROR "Unsupported Fortran compiler on GNU/Linux: ${CMAKE_Fortran_COMPILER_ID}")
  endif()
endif()

# ##############################################################################
# Create example targets
file(GLOB_RECURSE DA_EX *.cpp)
message("Targets")
foreach(ex_source ${DA_EX})
  string(REPLACE ".cpp" "" ex_name ${ex_source})
  get_filename_component(ex_target ${ex_name} NAME)
  # Exclude cmake produced source files
  if(${ex_target} MATCHES ".*CMake.*")
    continue()
  endif()
  add_executable(${ex_target} ${ex_source})
  target_include_directories(${ex_target} PRIVATE ${DA_INCLUDE_DIR})
  target_link_libraries(
    ${ex_target}
    PRIVATE ${AOCL_DA}
    ${SPARSE}
    ${LAPACK}
    ${BLAS}
    ${UTILS}
    ${FORTRAN_RUNTIME}
    OpenMP::OpenMP_CXX)
  if(NOT WIN32)
    target_link_libraries(${ex_target} PRIVATE ${DLP_LINK})
  endif()
  target_compile_definitions(${ex_target} PRIVATE ${AOCLDA_ILP64})

  message(NOTICE "   ${ex_target}")
endforeach()
message("")

message(NOTICE "Dependent libraries")
message(NOTICE "   AOCL-DA               : ${AOCL_DA}")
message(NOTICE "   AOCL-BLAS             : ${BLAS}")
message(NOTICE "   AOCL-LAPACK           : ${LAPACK}")
message(NOTICE "   AOCL-Sparse           : ${SPARSE}")
message(NOTICE "   AOCL-Utils            : ${UTILS}")
if(NOT WIN32)
  message(NOTICE "   AOCL-DLP              : ${DLP}")
endif()
message(NOTICE "\nOptions")
message(NOTICE "   Building for ILP64    : ${BUILD_ILP64}")
message("")
