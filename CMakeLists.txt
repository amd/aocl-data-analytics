# Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(AOCL-DA VERSION 4.2.1)

set(AOCLDA_VERSION
    "${CMAKE_PROJECT_VERSION_MAJOR}.${CMAKE_PROJECT_VERSION_MINOR}.${CMAKE_PROJECT_VERSION_PATCH}"
)

# Get timestamp.
string(TIMESTAMP BUILD_DATE "%Y%m%d")
# Set version string.
set(AOCLDA_VERSION_STR "AOCL-DA ${AOCLDA_VERSION} Build ${BUILD_DATE}")
# Define a C preprocessor macro to communicate the current version so that it
# can be embedded into the library and queried later.
set(VERS_DEF -DAOCLDA_VERSION_STRING="${AOCLDA_VERSION_STR}")

# Add version string in the variable that holds all definitions required to
# build da_core.
list(APPEND AOCLDA_DEFS ${VERS_DEF})

# Export Release Information automatically
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Required for Windows shared library build
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified")
  set(CMAKE_BUILD_TYPE
      "Debug"
      CACHE STRING "Choose the type of build" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release"
                                               "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_EXAMPLES "Build the example programs" ON)
option(BUILD_GTEST "Build Gtest unit-tests" ON)
option(COVERAGE "Builds coverage of Gtests and examples (Debug/GCC)" OFF)
option(BUILD_ILP64 "ILP64 Support" OFF)
# Windows options only
if(WIN32)
  # Setting path to OpenMP runtime.
  set(OpenMP_libomp_LIBRARY
      "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/lib/libomp.lib"
      CACHE STRING "openmp library path")
else() # Linux options only
  option(ASAN "Enable Address SANitizer tool (GNU/Linux)" OFF)
  option(MEMSAN "Enable MEMory SANitizer tool (GNU/Linux)" OFF)
  option(VALGRIND "Enable memory leak detection tool (GNU/Linux)" OFF)
endif()
option(BUILD_DOC "Build HTML documentation" OFF)
option(BUILD_SMP "Build using OpenMP and link to threaded BLAS and LAPACK" ON)
option(VECTORIZATION_REPORTS "Generate vectorization reports" OFF)
option(DA_LOGGING "Compile with debug printing" OFF)
option(BUILD_SHARED_LIBS "Shared library build" OFF)
option(BUILD_PYTHON "Python bindings" OFF)
# Set paths to AOCLUTILS, BLAS and LAPACK installations.
set(CMAKE_AOCL_ROOT
    $ENV{AOCL_ROOT}
    CACHE
      STRING
      "AOCL_ROOT directory to be used to find BLAS/LAPACK/AOCLUTILS libraries")
# Override the above path using the variables below.
set(BLAS_INCLUDE_DIR
    ""
    CACHE STRING "Path to BLAS include directory")
set(LAPACK_INCLUDE_DIR
    ""
    CACHE STRING "Path to LAPACK include directory")
set(BLAS_LIB
    ""
    CACHE STRING "Path to BLAS library")
set(LAPACK_LIB
    ""
    CACHE STRING "Path to LAPACK library")
set(UTILS_LIB
    ""
    CACHE STRING "Path to AOCL-UTILS library")

if(BUILD_SMP OR UNIX)
  # On Linux, even serial builds need an OpenMP runtime to enable correct
  # LibFlame linking
  find_package(OpenMP REQUIRED)
  if((NOT WIN32) OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

# Quick checks to disable requested features Disable *SAN & VALGRIND if running
# from Windows
if(ASAN AND NOT UNIX)
  message(WARNING "Address SANitizer is only supported in GNU/Linux")
  set(ASAN OFF)
endif()

if(MEMSAN AND NOT UNIX)
  message(WARNING "MEMory SANitizer is only supported in GNU/Linux")
  set(MEMSAN OFF)
endif()

if(VALGRIND AND NOT UNIX)
  message(WARNING "Memory leak detector tool is only supported in GNU/Linux")
  set(VALGRIND OFF)
endif()

# Coverage is only valid with GNU/Linux GCC Debug builds
if(COVERAGE
   AND (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "GNU"
        OR NOT ${CMAKE_BUILD_TYPE} MATCHES "Debug"
        OR NOT UNIX))
  message(WARNING "Coverage is only supported for  GNU/Linux GCC Debug build")
  set(COVERAGE OFF)
endif()

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  # Be verbose about da_error trace on Debug builds
  list(APPEND AOCLDA_DEFS -DVERBOSE_ERROR)
endif()

if(DA_LOGGING)
  add_compile_definitions(DA_LOGGING)
endif()

# modules directory for easy include
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

if(BUILD_ILP64)
  set(INT_LIB "ILP64")
  set(AOCLDA_ILP64 -DAOCLDA_ILP64)
else()
  set(INT_LIB "LP64")
endif()

# Check for CL compatibility layer of clang
cmake_path(GET CMAKE_CXX_COMPILER FILENAME COMPILER_BASENAME)
if(COMPILER_BASENAME MATCHES
   "^(.+-)?(clang-cl)(-[0-9]+(\\.[0-9]+)*)?(-[^.]+)?(\\.exe)?$")
  set(CLANG_OPTIONS_STYLE "MSVC")
else()
  set(CLANG_OPTIONS_STYLE "GNU")
endif()

# Check we can find AOCL installation
if(CMAKE_AOCL_ROOT STREQUAL ""
   AND (BLAS_INCLUDE_DIR STREQUAL ""
        OR LAPACK_INCLUDE_DIR STREQUAL ""
        OR BLAS_LIB STREQUAL ""
        OR LAPACK_LIB STREQUAL ""
        OR UTILS_LIB STREQUAL ""
       ))
  message(
    FATAL_ERROR
      "Environment variable \$AOCL_ROOT not found.\n - Perhaps source AOCL config script, or\n - Define \$AOCL_ROOT to point to AOCL install dir, and try again, or set -DCMAKE_AOCL_ROOT.\n - Alternative, specify BLAS/LAPACK_LIB/UTILS_LIB and BLAS/LAPACK_INCLUDE_DIR."
  )
endif()

enable_language(Fortran)
if(WIN32)
  # MSVC has warnings about deprecated CRT functions; these definitions overload
  # them automatically with _s versions
  add_compile_definitions(
    _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES=1
    _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES_COUNT=1
    _SILENCE_CXX20_CISO646_REMOVED_WARNING=1)

  # Clang-cl on Windows: ensure we are linking against the correct MSVC runtime
  # libraries
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CLANG_OPTIONS_STYLE STREQUAL
                                                "MSVC")
    if(BUILD_SHARED_LIBS)
      set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    else()
      set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
  endif()

  # Clang-cl or MSVC require ifort for the fortran compilation
  if(CLANG_OPTIONS_STYLE STREQUAL "MSVC" OR CMAKE_CXX_COMPILER_ID STREQUAL
                                            "MSVC")
    if(CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
      if(NOT DEFINED ENV{FLANG_COMPILER})
        message(FATAL_ERROR "Environment variable \$FLANG_COMPILER not found.")
      endif()

      file(TO_CMAKE_PATH $ENV{FLANG_COMPILER} FLANG_PATH)
      set(CMAKE_Fortran_COMPILER "${FLANG_PATH}/bin/flang.exe")
      set(FLANG_LIBS "${FLANG_PATH}/lib")
      set(FORTRAN_RUNTIME "${FLANG_LIBS}/flang.lib")
    else()
      set(IFORT_LIBS
          "$ENV{LIB}"
          CACHE STRING
                "Setting path to Intel Fortran runtime libraries on Windows")
      set(FORTRAN_RUNTIME
          "$<IF:$<CONFIG:Debug>,libifcoremdd.lib,libifcoremd.lib>")
      add_compile_definitions(WIN_IFORT_NAME_MANGLING)
    endif()
  else()
    if(NOT DEFINED ENV{MINGW_BIN})
      message(FATAL_ERROR "Environment variable \$MINGW_BIN not found.")
    endif()

    file(TO_CMAKE_PATH $ENV{MINGW_BIN} MINGW_PATH)
    set(CMAKE_Fortran_COMPILER gfortran)
  endif()

  set(CMAKE_FIND_LIBRARY_PREFIXES "")
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
else()
  if(NOT WIN32)
    if(CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
      set(FORTRAN_RUNTIME "flang")
    else() # Gnu
      set(FORTRAN_RUNTIME "gfortran")
    endif()
  endif()
endif()

# look for linear algebra libraries dependencies
include(AOCLDependencies)

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC"
   OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CLANG_OPTIONS_STYLE STREQUAL
                                                  "MSVC"))
  # MSVC and clang-cl compiler flags
  if(CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
    set(COMPILER_FLAGS_RELEASE "${COMPILER_FLAGS_RELEASE};-DNDEBUG;-O2")
    set(COMPILER_FLAGS_DEBUG "${COMPILER_FLAGS_DEBUG};-Od")
    set(COMPILER_FLAGS_CXX_DEBUG "${COMPILER_FLAGS_CXX_DEBUG};-W3;-DEBUG")
  else()
    set(COMPILER_FLAGS_COMMON "${COMPILER_FLAGS_COMMON};/arch:AVX2")
    set(COMPILER_FLAGS_RELEASE
        "${COMPILER_FLAGS_RELEASE};/DNDEBUG;/O2;/fp:fast")
    set(COMPILER_FLAGS_DEBUG "${COMPILER_FLAGS_DEBUG};/Od")
    set(COMPILER_FLAGS_CXX_DEBUG "${COMPILER_FLAGS_CXX_DEBUG};/W3;/DEBUG")
  endif()
  set(COMPILER_FLAGS_CXX "${COMPILER_FLAGS_CXX};/std:c++17;/EHsc")

  if(USE_AVX512 AND (NOT CMAKE_Fortran_COMPILER_ID MATCHES "Flang"))
    set(COMPILER_FLAGS_COMMON "${COMPILER_FLAGS_COMMON};/arch:AVX512")
  endif()

  if(VECTORIZATION_REPORTS)
    set(COMPILER_FLAGS_CXX "${COMPILER_FLAGS_CXX};/Qvec-report:2")
  endif()

  if($ENV{USERNAME} MATCHES "aoclwinuser")
    set(CMAKE_VS_MSBUILD_COMMAND
        "${CMAKE_VS_MSBUILD_COMMAND} /nodereuse:false /maxcpucount:1")
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU"
       OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CLANG_OPTIONS_STYLE
                                                      STREQUAL "GNU"))
  # GNU style compiler flags
  set(COMPILER_FLAGS_COMMON
      "${COMPILER_FLAS_COMMON};-Wall;-Wextra;$<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-conversion>;"
  )
  if(NOT CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
    set(COMPILER_FLAGS_COMMON
        "${COMPILER_FLAS_COMMON};-mtune=native;-mavx2;-mfma;")
  endif()

  # Release flags
  set(COMPILER_FLAGS_RELEASE "${COMPILER_FLAGS_RELEASE};-DNDEBUG;-O2;")

  if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(COMPILER_FLAGS_DEBUG "${COMPILER_FLAGS_DEBUG};-O0")
  else()
    set(COMPILER_FLAGS_DEBUG "${COMPILER_FLAGS_DEBUG};-O0;-gdwarf-5;-g3;")
  endif()

  if(USE_AVX512)
    set(COMPILER_FLAGS_COMMON
        "${COMPILER_FLAGS_COMMON};-mavx512f;-mavx512dq;-mavx512vl")
  endif()

  if(VECTORIZATION_REPORTS)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      set(COMPILER_FLAGS_COMMON
          "${COMPILER_FLAGS_COMMON};-Rpass=loop-vectorize;-Rpass-analysis=loop-vectorize"
      )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
      set(COMPILER_FLAGS_COMMON
          "${COMPILER_FLAGS_COMMON};-fopt-info-vec-all=vectorization.txt")
    endif()
  endif()
endif()

# Add convenience cleaning add "add_dependencies(cleanall ...) to all custom
# cleans, e.g. clean-doc and clean-coverage is a dependency of cleanall Note:
# this needs to be before any `add_dependencies`
add_custom_target(cleanall COMMAND ${CMAKE_MAKE_PROGRAM} clean)

if(ASAN)
  list(APPEND COMPILER_FLAGS_COMMON "-fsanitize=address")
  link_libraries("-fsanitize=address")
endif()

if(COVERAGE)
  # include coverage flags and targets
  include(Coverage)
endif()

add_compile_options(
  "${COMPILER_FLAGS_COMMON};$<$<COMPILE_LANGUAGE:CXX>:${COMPILER_FLAGS_CXX}>")
add_compile_options("$<$<CONFIG:Release>:${COMPILER_FLAGS_RELEASE}>")
add_compile_options(
  "$<$<CONFIG:Debug>:${COMPILER_FLAGS_DEBUG};$<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:${COMPILER_FLAGS_CXX_DEBUG}>>"
)

if(BUILD_DOC)
  add_subdirectory(doc)
endif()

if(NOT BUILD_DOC AND INTERNAL_DOC)
  message(
    WARNING
      "Turning off internal documentation since docs are not being built, set BUILD_DOC=On to enable also internal documentation"
  )
  set(INTERNAL_DOC Off)
endif()

# compile external packages
add_subdirectory(external/Lbfgsb.3.0)

# compile data analytics core library
add_subdirectory(source)

# Put all custom configuration BEFORE including "CTest"
if(VALGRIND)
  set(MEMORYCHECK_COMMAND_OPTIONS
      "--track-origins=yes --leak-check=full -s --show-leak-kinds=all --gen-suppressions=all"
  )
  set(MEMORYCHECK_TYPE "valgrind")
  set(MEMORYCHECK_SUPPRESSIONS_FILE
      "${PROJECT_SOURCE_DIR}/tools/valgrind_suppress.txt")
endif()

# build example and unit test programs
include(CTest)

# Suppression list file needs to be defined AFTER including ctest
if(VALGRIND)
  set(MEMORYCHECK_SUPPRESSIONS_FILE
      "${PROJECT_SOURCE_DIR}/tools/valgrind_suppress.txt")
endif()

add_subdirectory(tests)

if(BUILD_PYTHON)
  add_subdirectory(python_interface)
endif()

# Install examples and README files
install(
  FILES README-Release.txt
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  RENAME README.txt)
install(FILES License DESTINATION ${CMAKE_INSTALL_PREFIX})
install(
  DIRECTORY tests/examples/
  DESTINATION examples
  FILES_MATCHING
  PATTERN "*.cpp"
  PATTERN "*tree*" EXCLUDE
  PATTERN "*.csv")
install(
  FILES tools/CMakeLists_examples.cmake
  DESTINATION examples
  RENAME CMakeLists.txt)
install(FILES tests/examples/README.txt DESTINATION examples)

# Convenience Banner, print relevent info
message(STATUS "Building AOCL-DA ${AOCLDA_VERSION} (${CMAKE_BUILD_TYPE})")
message(
  NOTICE
  "CXX Compiler ID         : ${CMAKE_CXX_COMPILER_ID} (${COMPILER_BASENAME}) platform id: ${CMAKE_CXX_PLATFORM_ID}"
)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  message(
    NOTICE
    "COMPILER_FLAGS_COMMON   : ${COMPILER_FLAGS_COMMON}${COMPILER_FLAGS_DEBUG}")
endif()

if(CMAKE_BUILD_TYPE MATCHES "Release")
  message(
    NOTICE
    "COMPILER_FLAGS_COMMON   : ${COMPILER_FLAGS_COMMON}${COMPILER_FLAGS_RELEASE}"
  )
endif()

message(NOTICE "\nDEPENDENT LIBRARIES")
message(NOTICE "blas                    : ${BLAS}")
message(NOTICE "lapack                  : ${LAPACK}")
message(NOTICE "utils                   : ${UTILS}")

message(NOTICE "\nOPTIONS")
message(NOTICE "Building for ILP64      : ${BUILD_ILP64}")
message(NOTICE "Building with ASan      : ${ASAN}")
message(NOTICE "Building with MemSan    : ${MEMSAN}")
message(NOTICE "Building with Valgrind  : ${VALGRIND}")
message(NOTICE "Building with OpenMP    : ${BUILD_SMP}")
message(NOTICE "Shared library build    : ${BUILD_SHARED_LIBS}")
message(NOTICE "Vectorization reports   : ${VECTORIZATION_REPORTS}")
message(NOTICE "Debug printing          : ${DA_LOGGING}")

if(${VALGRIND})
  message(
    NOTICE
    "         Valgrind flags : ${MEMORYCHECK_COMMAND_OPTIONS}. Use with `ctest -T MemCheck`"
  )
  message(NOTICE
          "         Valgrind suppression: ${MEMORYCHECK_SUPPRESSIONS_FILE}")
endif()

message(NOTICE "Building with Coverage  : ${COVERAGE}")
message(NOTICE "Building docs           : ${BUILD_DOC}")

if(INTERNAL_DOC)
  message(NOTICE "Build internal docs     : ${INTERNAL_DOC}")
endif()

message(NOTICE "Building Python bindings:  ${BUILD_PYTHON}")

message(NOTICE "\nTARGETS")
message(NOTICE "Build the EXAMPLE programs   : ${BUILD_EXAMPLES}")
message(NOTICE "Build GTEST unit-tests       : ${BUILD_GTEST}")
