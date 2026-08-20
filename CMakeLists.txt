# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD) # FindBoost
endif()

project(AOCL-DA VERSION 5.3.2)

set(AOCLDA_VERSION
  "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
)

# Capture git commit date and tag if available, otherwise fallback to commit hash, if unavailable then "unknown"
set(GIT_COMMIT "unknown")
set(GIT_COMMIT_DATE "")
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  # Try to get the current tag
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --exact-match
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_TAG
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(GIT_COMMIT_TAG STREQUAL "")
    # No tag found, get commit hash
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      OUTPUT_VARIABLE GIT_COMMIT_HASH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(GIT_COMMIT "${GIT_COMMIT_HASH}")
  else()
    set(GIT_COMMIT "${GIT_COMMIT_TAG}")
  endif()
  # Try to get the commit date in YYYY-MM-DD hh:mm:ss format
  execute_process(
    COMMAND ${GIT_EXECUTABLE} show -s "--date=format:%Y-%m-%d %H:%M:%S" --format=%cd HEAD
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_DATE_OUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(NOT GIT_COMMIT_DATE_OUT STREQUAL "")
    set(GIT_COMMIT_DATE "${GIT_COMMIT_DATE_OUT}")
  endif()
endif()
set(GIT_DEF -DAOCLDA_GIT_COMMIT="${GIT_COMMIT} ${GIT_COMMIT_DATE}")
list(APPEND AOCLDA_DEFS ${GIT_DEF})

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
option(BUILD_FORTRAN
  "Builds aocl-da with the Fortran dependencies, RALFit and lbfgsb" ON)
option(BUILD_ILP64 "ILP64 Support" OFF)
if(NOT WIN32) # Linux options only
  option(ASAN "Enable Address SANitizer tool (GNU/Linux)" OFF)
  option(VALGRIND "Enable memory leak detection tool (GNU/Linux)" OFF)
endif()
option(BUILD_DOC "Build HTML documentation" OFF)
option(INTERNAL_DOC "Build interal HTML documentation" OFF)
option(UPDATE_OPTIONS_DOC "Update the option tables in the documentation (triggers UT)" OFF)
# Do not let INTERNAL_DOC and UPDATE_OPTIONS_DOC appear on cmake-gui
mark_as_advanced(INTERNAL_DOC)
mark_as_advanced(UPDATE_OPTIONS_DOC)
option(BUILD_SMP "Build using OpenMP and link to threaded BLAS and LAPACK" ON)
option(VECTORIZATION_REPORTS "Generate vectorization reports" OFF)
option(DA_LOGGING "Compile with debug printing" OFF)
option(BUILD_SHARED_LIBS "Shared library build" OFF)
# ARCH can be set as shown below, with default "native". If "dynamic" is set,
# then a list of target architectures will be built and returned in BUILD_ARCHS.
# If "native" is set, then the native architecture will be built, and the
# internal namespace will be set to NATIVE_NAMESPACE to ensure appropriate
# performant kernels are called.
set(ARCH
  "native"
  CACHE
  STRING
  "Architecture(s) to build: generic, generic_avx512, dynamic, native, znver2, znver3, ... (default: native)"
)
set(ARCH_LIST
  "generic"
  "generic_avx512"
  "dynamic"
  "native"
  "znver2"
  "znver3"
  "znver4"
  "znver5"
  "znver6")
if(NOT ARCH IN_LIST ARCH_LIST)
  message(FATAL_ERROR "Invalid ARCH value: \"${ARCH}\"\n"
    "Valid values are: ${ARCH_LIST}")
endif()
option(BUILD_PYTHON "Python bindings" OFF)
# Set paths to AOCL-Utils, BLAS, LAPACK and AOCL-Sparse installations.
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
set(SPARSE_INCLUDE_DIR
  ""
  CACHE STRING "Path to SPARSE include directory")
set(UTILS_INCLUDE_DIR
  ""
  CACHE STRING "Path to UTILS include directory")
if(NOT WIN32)
  set(DLP_INCLUDE_DIR
    ""
    CACHE STRING "Path to DLP include directory")
endif()
set(KT_PATH
  ""
  CACHE PATH "Path to kernel templates (defaults to SPARSE_INCLUDE_DIR if unset)")
set(BLAS_LIB
  ""
  CACHE STRING "Path to BLAS library")
set(LAPACK_LIB
  ""
  CACHE STRING "Path to LAPACK library")
set(SPARSE_LIB
  ""
  CACHE STRING "Path to SPARSE library")
set(UTILS_LIB
  ""
  CACHE STRING "Path to AOCL-UTILS library")
if(NOT WIN32)
  set(DLP_LIB
    ""
    CACHE STRING "Path to DLP library")
endif()
set(UTILS_CPUID_LIB
  ""
  CACHE STRING "Path to AU-CPUID library")
set(UTILS_CORE_LIB
  ""
  CACHE STRING "Path to AU_CORE library")
if(USE_EXISTING_DA)
  set(DA_LIB
    ""
    CACHE STRING "Path to existing DA library")
endif()

# Even serial builds need an OpenMP runtime to enable correct LAPACK linking
if(BUILD_SMP)
  find_package(OpenMP REQUIRED)
  message(STATUS "OpenMP C++ libs: ${OpenMP_CXX_LIB_NAMES}")
  get_target_property(_omp_libs OpenMP::OpenMP_CXX INTERFACE_LINK_LIBRARIES)
  get_target_property(_omp_opts OpenMP::OpenMP_CXX INTERFACE_COMPILE_OPTIONS)
  message(STATUS "OpenMP imported target libs: ${_omp_libs}")
  message(STATUS "OpenMP imported target compile options: ${_omp_opts}")

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
  message(
    WARNING
    "Building without OpenMP may cause linking issues with threaded BLAS/LAPACK/Sparse libraries"
  )
endif()

# Quick checks to disable requested features Disable *SAN & VALGRIND if running
# from Windows
if(ASAN AND NOT UNIX)
  message(WARNING "Address SANitizer is only supported in GNU/Linux")
  set(ASAN OFF)
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
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if(BUILD_ILP64)
  set(INT_LIB "ILP64")
  set(AOCLDA_ILP64 -DAOCLDA_ILP64)
else()
  set(INT_LIB "LP64")
endif()

# Use -fPIC always
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Check we can find AOCL installation
if(CMAKE_AOCL_ROOT STREQUAL ""
  AND (BLAS_INCLUDE_DIR STREQUAL ""
     OR LAPACK_INCLUDE_DIR STREQUAL ""
     OR SPARSE_INCLUDE_DIR STREQUAL ""
     OR BLAS_LIB STREQUAL ""
     OR LAPACK_LIB STREQUAL ""
     OR SPARSE_LIB STREQUAL ""
     OR UTILS_LIB STREQUAL ""
  ))
  message(
    FATAL_ERROR
    "Environment variable \$AOCL_ROOT not found.\n - Perhaps source AOCL config script, or\n - Define \$AOCL_ROOT to point to AOCL install dir, and try again, or set -DCMAKE_AOCL_ROOT.\n - Alternatively, specify BLAS/LAPACK/SPARSE/UTILS_LIB and BLAS/LAPACK/SPARSE_INCLUDE_DIR."
  )
endif()

if(BUILD_FORTRAN)
  enable_language(Fortran)
else()
  add_compile_definitions(NO_FORTRAN)
endif()

include(CompilerFlagsSupport)
GetCompilerFamily(COMPILER_FAMILY)


if(WIN32)
  # MSVC has warnings about deprecated CRT functions; these definitions overload
  # them automatically with _s versions
  add_compile_definitions(
    _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES=1
    _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES_COUNT=1 _CRT_SECURE_NO_WARNINGS=1
    _SILENCE_CXX20_CISO646_REMOVED_WARNING=1)

  # Deal with issues concerning min/max macros in Windows STL headers
  add_compile_definitions(NOMINMAX)

  # Clang-cl on Windows: ensure we are linking against the correct MSVC runtime
  # libraries
  if(COMPILER_FAMILY MATCHES "LLVM-CL")
    if(BUILD_SHARED_LIBS)
      set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    else()
      set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
    if(BUILD_FORTRAN AND CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
      # Blank out the Fortran expansions so they aren't passed to AOCC flang.
      foreach(_rt MultiThreaded MultiThreadedDLL
        MultiThreadedDebug MultiThreadedDebugDLL)
        set(CMAKE_Fortran_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_${_rt} "")
      endforeach()
    endif()
  endif()

  set(CMAKE_FIND_LIBRARY_PREFIXES "")
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")

  if(BUILD_FORTRAN)
    if(COMPILER_FAMILY MATCHES "LLVM-CL")
      if(CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
        if(NOT DEFINED ENV{FLANG_COMPILER})
          message(
            FATAL_ERROR "Environment variable \$FLANG_COMPILER not found.")
        endif()

        file(TO_CMAKE_PATH $ENV{FLANG_COMPILER} FLANG_PATH)
        set(CMAKE_Fortran_COMPILER "${FLANG_PATH}/bin/flang.exe")
        set(FLANG_LIBS "${FLANG_PATH}/lib")
        set(FORTRAN_RUNTIME "${FLANG_LIBS}/flang.lib"
          "${FLANG_LIBS}/libpgmath.lib")
      elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
        set(IFORT_LIBS
          "$ENV{LIB}"
          CACHE STRING
          "Setting path to Intel Fortran runtime libraries on Windows")
        set(FORTRAN_RUNTIME
          "$<IF:$<CONFIG:Debug>,libifcoremdd.lib,libifcoremd.lib>")
        add_compile_definitions(WIN_IFORT_NAME_MANGLING)
      else()
        message(FATAL_ERROR
          "Unsupported Fortran compiler on Windows: ${CMAKE_Fortran_COMPILER_ID}"
        )
      endif()
    else()
      message(FATAL_ERROR
        "Unsupported compiler on Windows: ${COMPILER_FAMILY}"
      )
    endif()
  endif()

else()
  if(BUILD_FORTRAN)
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
endif()
if(BUILD_FORTRAN)
  message(NOTICE "Fortran (${CMAKE_Fortran_COMPILER_ID}) runtime: ${FORTRAN_RUNTIME}")
endif()

# Look for linear algebra libraries dependencies
include(AOCLDependencies)
include(CompilerFlags)

# If the user did not explicitly provide KT_PATH, default it to SPARSE_INCLUDE_DIR
if(KT_PATH STREQUAL "")
  set(KT_PATH "${SPARSE_INCLUDE_DIR}")
endif()

# Add convenience cleaning add "add_dependencies(cleanall ...) to all custom
# cleans, e.g. clean-doc and clean-coverage is a dependency of cleanall Note:
# this needs to be before any `add_dependencies`
if(NOT TARGET cleanall)
  add_custom_target(cleanall COMMAND ${CMAKE_MAKE_PROGRAM} clean)
endif()

if(COVERAGE)
  # Include coverage flags and targets
  include(Coverage)
endif()

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

if(NOT BUILD_DOC AND UPDATE_OPTIONS_DOC)
  message(
    WARNING
    "Turning off update options in documentation since docs are not being built, set BUILD_DOC=On to enable option table updates"
  )
  set(UPDATE_OPTIONS_DOC Off)
endif()

if(UPDATE_OPTIONS_DOC)
  set(BUILD_GTEST On)
endif()

if(NOT USE_EXISTING_DA)
  # Compile external packages
  if(BUILD_FORTRAN)
    add_subdirectory(external/Lbfgsb.3.0)
    add_subdirectory(external/RALFit EXCLUDE_FROM_ALL)
  endif()

  # add Boost for Boost.Sort functionality (assume Boost is installed on system)
  set(BOOST_MIN_VERSION "1.66.0")
  find_package(Boost ${BOOST_MIN_VERSION} REQUIRED)

  # Compile data analytics core library
  add_subdirectory(source)

  # Put all custom configuration BEFORE including "CTest"
  if(VALGRIND)
    set(MEMORYCHECK_COMMAND_OPTIONS
      "--track-origins=yes --leak-check=full -s --show-leak-kinds=all" # --gen-suppressions=all
    )
    set(MEMORYCHECK_TYPE "valgrind")
    set(MEMORYCHECK_SUPPRESSIONS_FILE
      "${PROJECT_SOURCE_DIR}/tools/valgrind_suppress.txt")
  endif()

  if(BUILD_EXAMPLES OR BUILD_GTEST)
    # Build example and unit test programs
    include(CTest)
  endif()

  # Suppression list file needs to be defined AFTER including ctest
  if(VALGRIND)
    set(MEMORYCHECK_SUPPRESSIONS_FILE
      "${PROJECT_SOURCE_DIR}/tools/valgrind_suppress.txt")
  endif()

  if(BUILD_EXAMPLES OR BUILD_GTEST)
    add_subdirectory(tests)
  endif()
endif()

if(BUILD_PYTHON)
  add_subdirectory(python_interface)
endif()

if(NOT USE_EXISTING_DA)
  # Install examples and README files
  install(FILES README.md DESTINATION ${CMAKE_INSTALL_PREFIX})
  install(FILES License Notices DESTINATION ${CMAKE_INSTALL_PREFIX})
  install(
    DIRECTORY tests/examples/
    DESTINATION examples
    FILES_MATCHING
    PATTERN "*.cpp"
    PATTERN "*.csv")
  install(
    FILES tools/CMakeLists_examples.cmake
    DESTINATION examples
    RENAME CMakeLists.txt)
  install(FILES tests/examples/README.txt DESTINATION examples)
endif()

# Convenience banner, print relevant info
message(STATUS "Building AOCL-DA ${AOCLDA_VERSION} (${CMAKE_BUILD_TYPE})")
message(NOTICE "CXX Compiler ID         : ${CMAKE_CXX_COMPILER_ID} platform id: ${CMAKE_CXX_PLATFORM_ID}")
message(NOTICE "C Compiler ID           : ${CMAKE_C_COMPILER_ID}")
message(NOTICE "Fortran Compiler ID     : ${CMAKE_Fortran_COMPILER_ID}")
message(NOTICE "OpenMP flags            : ${OpenMP_CXX_FLAGS}")
message(NOTICE "OpenMP linker flags     : ${OpenMP_EXE_LINKER_FLAGS}")
message(NOTICE "\nOPTIONS")
message(
  NOTICE
  "Requested architecture(s) ARCH.................: ${ARCH} (${BUILD_ARCHS})")
message(NOTICE "Building with ASan        ASAN.................: ${ASAN}")
message(NOTICE "Building docs             BUILD_DOC............: ${BUILD_DOC}")
if(INTERNAL_DOC OR CMAKE_BUILD_TYPE MATCHES "Debug")
  message(NOTICE "Build internal docs       INTERNAL_DOC.........: ${INTERNAL_DOC}")
endif()
if(UPDATE_OPTIONS_DOC OR CMAKE_BUILD_TYPE MATCHES "Debug")
  message(NOTICE "Update options in docs    UPDATE_OPTIONS_DOC...: ${UPDATE_OPTIONS_DOC}")
endif()
message(NOTICE
  "Building for ILP64        BUILD_ILP64..........: ${BUILD_ILP64}")
message(NOTICE
  "Building Python bindings  BUILD_PYTHON.........: ${BUILD_PYTHON}")
message(NOTICE
  "Building Fortran code     BUILD_FORTRAN........: ${BUILD_FORTRAN}")
message(NOTICE "Building with OpenMP      BUILD_SMP............: ${BUILD_SMP}")
message(NOTICE
  "Shared library build      BUILD_SHARED_LIBS....: ${BUILD_SHARED_LIBS}")
message(NOTICE "Building with Coverage    COVERAGE.............: ${COVERAGE}")
message(NOTICE "Debug printing            DA_LOGGING...........: ${DA_LOGGING}")
message(NOTICE "Building with Valgrind    VALGRIND.............: ${VALGRIND}")
if(${VALGRIND})
  message(
    NOTICE
    "         Valgrind flags : ${MEMORYCHECK_COMMAND_OPTIONS}. Use with `ctest -T MemCheck`"
  )
  message(NOTICE
    "         Valgrind suppression: ${MEMORYCHECK_SUPPRESSIONS_FILE}")
endif()
message(
  NOTICE
  "Vectorization reports     VECTORIZATION_REPORTS: ${VECTORIZATION_REPORTS}")

message(NOTICE "\nTARGETS")
message(NOTICE
  "Build the EXAMPLE programs  BUILD_EXAMPLES.....: ${BUILD_EXAMPLES}")
message(NOTICE
  "Build GTEST unit-tests      BUILD_GTEST........: ${BUILD_GTEST}")

message(NOTICE "\nDEPENDENT LIBRARIES")
message(NOTICE "BLAS             BLAS=${BLAS}")
message(STATUS "BLAS_INCLUDE_DIR      ${BLAS_INCLUDE_DIR}")
message(NOTICE "LAPACK           LAPACK=${LAPACK}")
message(STATUS "LAPACK_INCLUDE_DIR    ${LAPACK_INCLUDE_DIR}")
message(NOTICE "AOCL-Sparse      SPARSE=${SPARSE}")
message(NOTICE "Kernel Templates KT_PATH=${KT_PATH}")
message(NOTICE "AOCL-Utils       UTILS=${UTILS}")
message(NOTICE "AOCL-Utils/CPUID UTILS_CPUID=${UTILS_CPUID}")
message(STATUS "UTILS_INCLUDE_DIR     ${UTILS_INCLUDE_DIR}")

message(NOTICE "Boost include path    Boost_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}")
if(NOT WIN32)
  message(NOTICE "AOCL-DLP         DLP=${DLP}")
  message(STATUS "DLP_INCLUDE_DIR       ${DLP_INCLUDE_DIR}")
endif()
if(USE_EXISTING_DA)
  message(NOTICE "Existing AOCL-DA DA=${DA}")
endif()
