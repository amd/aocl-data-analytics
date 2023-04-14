cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(AOCL-DA VERSION 0.1.0)

set(AOCLDA_VERSION
    "${CMAKE_PROJECT_VERSION_MAJOR}.${CMAKE_PROJECT_VERSION_MINOR}.${CMAKE_PROJECT_VERSION_PATCH}"
)

# Export Release Information automatically
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Debug' as none was specified")
  set(CMAKE_BUILD_TYPE
      "Debug"
      CACHE STRING "Choose the type of build" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_EXAMPLES "Build the example programs" ON)
option(BUILD_GTEST "Build Gtest unit-tests" ON)
option(COVERAGE "Builds coverage of Gtests and examples (Debug/GCC)" OFF)
option(BUILD_ILP64 "ILP64 Support" OFF)
option(ASAN "Enable Address SANitizer tool (GNU/Linux)" OFF)
option(MEMSAN "Enable MEMory SANitizer tool (GNU/Linux)" OFF)
option(VALGRIND "Enable memory leak detection tool (GNU/Linux)" OFF)
option(BUILD_DOC "Build PDF and HTML documentation" OFF)

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

# modules directory for easy include
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

if(WIN32)
  set(CMAKE_Fortran_COMPILER "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/bin/intel64/ifort.exe")
  set(CMAKE_FIND_LIBRARY_PREFIXES "")
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
  # Set environment variables for users, add your own paths here or directly on
  # the environment
  set(ENV{AOCL_INSTALL} "C:/Users/aoclwinuser/AMD")
  set(ENV{IFORT_LIBS}
      "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/compiler/lib/intel64_win"
  )
  if($ENV{USERNAME} MATCHES "ehopkins")
    set(ENV{AOCL_INSTALL} "C:/Users/ehopkins/AOCL-4.0")
    set(ENV{IFORT_LIBS}
        "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/compiler/lib/intel64_win"
    )
  endif()

  message(AOCL_INSTALL=$ENV{AOCL_INSTALL})
  if(NOT BUILD_ILP64)
    find_library(BLIS name AOCL-LibBlis-Win-dll
                 PATHS "$ENV{AOCL_INSTALL}/amd-blis/lib/LP64" REQUIRED)
    find_library(FLAME name AOCL-LibFlame-Win-dll
                 PATHS "$ENV{AOCL_INSTALL}/amd-libflame/lib/LP64" REQUIRED)
    include_directories("$ENV{AOCL_INSTALL}/amd-blis/include/LP64")
    include_directories("$ENV{AOCL_INSTALL}/amd-libflame/include/LP64")
  else()
    find_library(BLIS name AOCL-LibBlis-Win-dll
                 PATHS "$ENV{AOCL_INSTALL}/amd-blis/lib/ILP64" REQUIRED)
    find_library(FLAME name AOCL-LibFlame-Win-dll
                 PATHS "$ENV{AOCL_INSTALL}/amd-libflame/lib/ILP64" REQUIRED)
    include_directories("$ENV{AOCL_INSTALL}/amd-blis/include/ILP64")
    include_directories("$ENV{AOCL_INSTALL}/amd-libflame/include/ILP64")
  endif()

  set(ENV{LIBIFCORE}
      "$<IF:$<CONFIG:Debug>,$ENV{IFORT_LIBS}/libifcoremdd.lib,$ENV{IFORT_LIBS}/libifcoremd.lib>"
  )

else()

  if($ENV{USER} MATCHES "bmarteau")
    set(ENV{AOCL_ROOT} /home/bmarteau/amd/aocl/4.0)
  endif()
  if($ENV{USER} MATCHES "andrsajo")
    set(ENV{AOCL_ROOT} /home/andrsajo/amd/aocl/4.0)
  endif()
  if($ENV{USER} MATCHES "ehopkins")
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      set(ENV{AOCL_ROOT} /opt/AMD/aocl/aocl-linux-gcc-4.0)
    elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
      set(ENV{AOCL_ROOT} /opt/AMD/aocl/aocl-linux-aocc-4.0)
    else()
      message(FATAL_ERROR "GCC or CLANG compiler must be used")
    endif()
  endif()

  if(NOT DEFINED ENV{AOCL_ROOT})
    message(
      FATAL_ERROR
        "Environmental variable \$AOCL_ROOT not found.\n - Perhaps source AOCL config script, or\n - Define \$AOCL_ROOT to point to AOCL install dir, and try again."
    )
  endif()

  # find dependencies. all aocl libraries are allowed it is assumed that aocl is
  # installed either on the system's paths or in a folder designated by the
  # environment variable AOCL_ROOT
  find_library(
    BLIS name blis
    PATHS $ENV{AOCL_ROOT}/lib REQUIRED
    NO_DEFAULT_PATH)

  find_library(
    FLAME name flame
    PATHS $ENV{AOCL_ROOT}/lib REQUIRED
    NO_DEFAULT_PATH)

  if(NOT BUILD_ILP64)
    include_directories($ENV{AOCL_ROOT}/include_LP64)
  else()
    include_directories($ENV{AOCL_ROOT}/include_ILP64)
  endif()

endif()

cmake_path(GET CMAKE_CXX_COMPILER FILENAME COMPILER_BASENAME)

if(BUILD_ILP64)
  message(STATUS "BUILD_ILP64 = ON")
  add_definitions(-DAOCLDA_ILP64)
endif(BUILD_ILP64)

# Convenience Banner, print relevent info
message(
  STATUS
    "CXX Compiler ID: ${CMAKE_CXX_COMPILER_ID} (${COMPILER_BASENAME}) platform id: ${CMAKE_CXX_PLATFORM_ID}"
)
message(STATUS "Building AOCL-DA version: ${AOCLDA_VERSION}")

# Check for CL compatibility layer of clang
if(COMPILER_BASENAME MATCHES
   "^(.+-)?(clang-cl)(-[0-9]+(\\.[0-9]+)*)?(-[^.]+)?(\\.exe)?$")
  set(CLANG_OPTIONS_STYLE "MSVC")
else()
  set(CLANG_OPTIONS_STYLE "GNU")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  # MSVC compiler flags
  set(COMPILER_FLAGS_COMMON "/openmp;/arch:AVX2")
  set(COMPILER_FLAGS_CXX "/std:c++17;/EHsc")
  if(USE_AVX512)
    set(COMPILER_FLAGS_COMMON "${COMPILER_FLAGS_COMMON}; /arch:AVX512")
  endif()
  set(COMPILER_FLAGS_RELEASE "/DNDEBUG;/O2;/fp:fast")
  set(COMPILER_FLAGS_DEBUG "/Od")
  set(COMPILER_FLAGS_CXX_DEBUG "/W3")
  if($ENV{USERNAME} MATCHES "aoclwinuser")
    set(CMAKE_VS_MSBUILD_COMMAND "${CMAKE_VS_MSBUILD_COMMAND} /nodereuse:false /maxcpucount:1")
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU"
       OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CLANG_OPTIONS_STYLE
                                                      STREQUAL "GNU"))
  # GNU style compiler flags Common flags
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

# compile external packages
add_subdirectory(external/Lbfgsb.3.0)

# compile data analytics core library
add_subdirectory(source)

# build example and unit test programs
include(CTest)
add_subdirectory(tests)
