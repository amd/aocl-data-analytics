cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(data_analytics VERSION 0.1.0)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified.")
  set(CMAKE_BUILD_TYPE
      "Debug"
      CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_EXAMPLES "Build the example programs" ON)
option(BUILD_GTEST "Build gtest unit tests" ON)
option(COVERAGE "Builds coverage of all the gtests and example programs" OFF)
option(CHECK_MEM "Use google's asan tool on all C/C++ code." OFF)

# modules directory for easy include
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Set environment variables for users, add your own paths here or directly on
# the environment
if($ENV{USER} MATCHES "bmarteau")
  set(ENV{AOCL_INSTALL} /home/bmarteau/amd/aocl/4.0)
endif()

if($ENV{USER} MATCHES "ehopkins")
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(ENV{AOCL_INSTALL} /opt/AMD/aocl/aocl-linux-gcc-4.0)
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(ENV{AOCL_INSTALL} /opt/AMD/aocl/aocl-linux-aocc-4.0)
  else()
    message(FATAL_ERROR "GCC or CLANG compiler must be used")
  endif()
endif()

# find dependencies. all aocl libraries are allowed it is assumed that aocl is
# installed either on the system's paths or in a folder designated by the
# environment variable AOCL_INSTALL
find_library(BLIS name blis PATHS $ENV{AOCL_INSTALL}/lib REQUIRED)

find_library(FLAME name flame PATHS $ENV{AOCL_INSTALL}/lib REQUIRED)

include_directories($ENV{AOCL_INSTALL}/include_LP64)

cmake_path(GET CMAKE_CXX_COMPILER FILENAME COMPILER_BASENAME)

# Coverage is only valid with GNU Debug builds
if(COVERAGE AND (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "GNU"
                 OR NOT ${CMAKE_BUILD_TYPE} MATCHES "Debug"))
  message(
    FATAL_ERROR "Coverage builds are only compatible with GNU debug builds")
endif()

# compiler flags
message(
  STATUS
    "CXX Compiler ID= ${CMAKE_CXX_COMPILER_ID} (${COMPILER_BASENAME}) platform id: ${CMAKE_CXX_PLATFORM_ID}"
)
# GCC compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  # Common flags
  set(COMPILER_FLAGS_COMMON "-mtune=native;-mavx2;-mfma")
  # Release flags
  set(COMPILER_FLAGS_RELEASE
      "-DNDEBUG;-O2;-ffast-math;-ftree-vectorize;-funroll-loops;-ffp-contract=fast"
  )
  set(COMPILER_FLAGS_DEBUG "-O0;-gdwarf-5;-g3;-Wall;-Wextra")
  if(USE_AVX512)
    set(COMPILER_FLAGS_COMMON
        "${COMPILER_FLAGS_COMMON};-mavx512f;-mavx512dq;-mavx512vl")
  endif()
endif()

if(CHECK_MEM)
  list(APPEND COMPILER_FLAGS_COMMON "-fsanitize=address")
  link_libraries("-fsanitize=address")
endif()

if(COVERAGE)
  # Include coverage flags and targets
  include(Coverage)
  set(COV_EXECUTABLES "")
endif()

add_compile_options(
  "$<$<CONFIG:Release>:${COMPILER_FLAGS_COMMON};${COMPILER_FLAGS_RELEASE}>")
add_compile_options(
  "$<$<CONFIG:Debug>:${COMPILER_FLAGS_COMMON};${COMPILER_FLAGS_DEBUG}>")

# compile external packages
add_subdirectory(external/Lbfgsb.3.0)

# compile data analytics core library
add_subdirectory(source)

# build example and unit test programs
include(CTest)
add_subdirectory(tests)