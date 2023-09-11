cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(AOCL-DA VERSION 0.1.0)

set(AOCLDA_VERSION
    "${CMAKE_PROJECT_VERSION_MAJOR}.${CMAKE_PROJECT_VERSION_MINOR}.${CMAKE_PROJECT_VERSION_PATCH}"
)

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
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release")
endif()

set(CMAKE_CXX_STANDARD 17)
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
option(BUILD_DOC "Build HTML documentation" OFF)
option(BUILD_SMP "Build using OpenMP and link to threaded BLIS and Flame" ON)
option(VECTORIZATION_REPORTS "Generate vectorization reports" OFF)
option(DA_LOGGING "Compile with debug printing" OFF)

if(BUILD_SMP)
  find_package(OpenMP REQUIRED)
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
  add_compile_definitions(VERBOSE_ERROR)
endif()

if (DA_LOGGING)
  add_compile_definitions(DA_LOGGING)
endif()

# modules directory for easy include
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

if(NOT BUILD_ILP64)
  set(INT_LIB "LP64")
else()
  set(INT_LIB "ILP64")
  add_definitions(-DAOCLDA_ILP64)
endif()

# Check for CL compatibility layer of clang
if(COMPILER_BASENAME MATCHES
   "^(.+-)?(clang-cl)(-[0-9]+(\\.[0-9]+)*)?(-[^.]+)?(\\.exe)?$")
  set(CLANG_OPTIONS_STYLE "MSVC")
else()
  set(CLANG_OPTIONS_STYLE "GNU")
endif()

# Check we can find AOCL installation
if(NOT DEFINED ENV{AOCL_ROOT})
  if((DEFINED CMAKE_AOCL_ROOT))
    set(ENV{AOCL_ROOT} ${CMAKE_AOCL_ROOT}) 
  else()
    message(
      FATAL_ERROR
      "Environment variable \$AOCL_ROOT not found.\n - Perhaps source AOCL config script, or\n - Define \$AOCL_ROOT to point to AOCL install dir, and try again, or set -DCMAKE_AOCL_ROOT."
    )
  endif()
endif()

if(WIN32)

  #MSVC has warnings about deprecated CRT functions; these definitions overload them automatically with _s versions
  add_compile_definitions(_CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES=1 _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES_COUNT=1)

  #Clang-cl on Windows: ensure we are linking against the correct MSVC runtime libraries
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CLANG_OPTIONS_STYLE STREQUAL "MSVC")
    if(BUILD_SHARED_LIBS)
      set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    else()
      set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
  endif()

  #Clang-cl or MSVC require ifort for the fortran compilation
  if(CLANG_OPTIONS_STYLE STREQUAL "MSVC" OR CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if(NOT DEFINED ENV{INTEL_FCOMPILER})
      message(
        FATAL_ERROR
          "Environment variable \$INTEL_FCOMPILER not found."
      )
    endif()

    file(TO_CMAKE_PATH $ENV{INTEL_FCOMPILER} IFORT_PATH)
    set(CMAKE_Fortran_COMPILER "${IFORT_PATH}/bin/intel64/ifort.exe")
    set(ENV{IFORT_LIBS} "${IFORT_PATH}/compiler/lib/intel64_win")
	set(ENV{LIBIFCORE}
      "$<IF:$<CONFIG:Debug>,$ENV{IFORT_LIBS}/libifcoremdd.lib,$ENV{IFORT_LIBS}/libifcoremd.lib>"
    )
	add_compile_definitions(WIN_IFORT_NAME_MANGLING)
  else()
    if(NOT DEFINED ENV{MINGW_BIN})
      message(
        FATAL_ERROR
          "Environment variable \$MINGW_BIN not found."
      )
    endif()
	file(TO_CMAKE_PATH $ENV{MINGW_BIN} MINGW_PATH)
	set(CMAKE_Fortran_COMPILER gfortran)
  endif()

  set(CMAKE_FIND_LIBRARY_PREFIXES "")
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")

  if(BUILD_SMP)
    set(BLIS_NAME "AOCL-LibBlis-Win-MT-dll")
    set(FLAME_NAME "AOCL-LibFlame-Win-MT-dll")
  else()
    set(BLIS_NAME "AOCL-LibBlis-Win-dll")
    set(FLAME_NAME "AOCL-LibFlame-Win-dll")
  endif()
  set(BLIS_PATH "$ENV{AOCL_ROOT}/amd-blis/lib/${INT_LIB}")
  set(FLAME_PATH "$ENV{AOCL_ROOT}/amd-libflame/lib/${INT_LIB}")

  include_directories("$ENV{AOCL_ROOT}/amd-blis/include/${INT_LIB}")
  include_directories("$ENV{AOCL_ROOT}/amd-libflame/include/${INT_LIB}")

else()

  include_directories($ENV{AOCL_ROOT}/include_${INT_LIB})
  set(BLIS_PATH $ENV{AOCL_ROOT}/lib_${INT_LIB})
  set(FLAME_PATH $ENV{AOCL_ROOT}/lib_${INT_LIB})

  if(BUILD_SMP)
    set(BLIS_NAME "blis-mt")
  else()
    set(BLIS_NAME "blis")
  endif()
  set(FLAME_NAME "flame")
endif()

find_library(BLIS name ${BLIS_NAME} PATHS ${BLIS_PATH} REQUIRED NO_DEFAULT_PATH)
find_library(FLAME name ${FLAME_NAME} PATHS ${FLAME_PATH} REQUIRED NO_DEFAULT_PATH)
cmake_path(GET CMAKE_CXX_COMPILER FILENAME COMPILER_BASENAME)

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC"
   OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CLANG_OPTIONS_STYLE
                                                      STREQUAL "MSVC"))
  # MSVC and clang-cl compiler flags
  set(COMPILER_FLAGS_COMMON "/arch:AVX2")
  set(COMPILER_FLAGS_CXX "/std:c++17;/EHsc")
  if(USE_AVX512)
    set(COMPILER_FLAGS_COMMON "${COMPILER_FLAGS_COMMON}; /arch:AVX512")
  endif()
  if(VECTORIZATION_REPORTS)
    set(COMPILER_FLAGS_CXX "${COMPILER_FLAGS_CXX}; /Qvec-report:2")
  endif()
  set(COMPILER_FLAGS_RELEASE "/DNDEBUG;/O2;/fp:fast")
  set(COMPILER_FLAGS_DEBUG "/Od")
  set(COMPILER_FLAGS_CXX_DEBUG "/W3;/DEBUG")
  if($ENV{USERNAME} MATCHES "aoclwinuser")
    set(CMAKE_VS_MSBUILD_COMMAND
        "${CMAKE_VS_MSBUILD_COMMAND} /nodereuse:false /maxcpucount:1")
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU"
       OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CLANG_OPTIONS_STYLE
                                                      STREQUAL "GNU"))
  # GNU style compiler flags
  set(COMPILER_FLAGS_COMMON
      "-mtune=native;-mavx2;-mfma;-Wall;-Wextra;$<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-conversion>;"
  )
  # Release flags
  set(COMPILER_FLAGS_RELEASE "-DNDEBUG;-O2;")
  if (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(COMPILER_FLAGS_DEBUG "-O0")
  else()
    set(COMPILER_FLAGS_DEBUG "-O0;-gdwarf-5;-g3;")
  endif()
  if(USE_AVX512)
    set(COMPILER_FLAGS_COMMON
        "${COMPILER_FLAGS_COMMON};-mavx512f;-mavx512dq;-mavx512vl")
  endif()
  if (BUILD_SHARED_LIBS)
    set(COMPILER_FLAGS_COMMON "${COMPILER_FLAGS_COMMON};-fPIC")
  endif()
    if(VECTORIZATION_REPORTS)
      if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(COMPILER_FLAGS_COMMON "${COMPILER_FLAGS_COMMON};-Rpass=loop-vectorize;-Rpass-analysis=loop-vectorize")
      elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set(COMPILER_FLAGS_COMMON "${COMPILER_FLAGS_COMMON};-fopt-info-vec-all=vectorization.txt")
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
  message(WARNING "Turning off internal documentation since docs are not being built, set BUILD_DOC=On to enable also internal documentation")
  set(INTERNAL_DOC Off)
endif()


# compile external packages
add_subdirectory(external/Lbfgsb.3.0)

# compile data analytics core library
add_subdirectory(source)

# Put all custom configuration BEFORE including "CTest"
if(VALGRIND)
  set(MEMORYCHECK_COMMAND_OPTIONS
      "--track-origins=yes --leak-check=full -s --show-leak-kinds=all --gen-suppressions=all")
  set(MEMORYCHECK_TYPE "valgrind")
  set(MEMORYCHECK_SUPPRESSIONS_FILE "${PROJECT_SOURCE_DIR}/tools/valgrind_suppress.txt")
endif()

# build example and unit test programs
include(CTest)

# Suppression list file needs to be defined AFTER including ctest
if(VALGRIND)
  set(MEMORYCHECK_SUPPRESSIONS_FILE "${PROJECT_SOURCE_DIR}/tools/valgrind_suppress.txt")
endif()

add_subdirectory(tests)

# Install examples and README files
install(FILES README-Release.txt
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    RENAME README.txt
)
install(FILES License
    DESTINATION ${CMAKE_INSTALL_PREFIX}
)
install(DIRECTORY tests/examples/
    DESTINATION examples
    FILES_MATCHING PATTERN "*.cpp"
)
install(FILES tests/examples/README.txt
    DESTINATION examples
)

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
message(NOTICE "blis                    : ${BLIS}")
message(NOTICE "flame                   : ${FLAME}")

message(NOTICE "\nOPTIONS")
message(NOTICE "Building for ILP64      : ${BUILD_ILP64}")
message(NOTICE "Building with ASan      : ${ASAN}")
message(NOTICE "Building with MemSan    : ${MEMSAN}")
message(NOTICE "Building with Valgrind  : ${VALGRIND}")
message(NOTICE "Building with OpenMP    : ${BUILD_SMP}")
message(NOTICE "Vectorization reports   : ${VECTORIZATION_REPORTS}")
message(NOTICE "Debug printing          : ${DA_LOGGING}")
if(${VALGRIND})
  message(
    NOTICE
    "         Valgrind flags : ${MEMORYCHECK_COMMAND_OPTIONS}. Use with `ctest -T MemCheck`"
  )
  message(
    NOTICE
    "         Valgrind suppression: ${MEMORYCHECK_SUPPRESSIONS_FILE}"
  )

endif()
message(NOTICE "Building with Coverage  : ${COVERAGE}")
message(NOTICE "Building docs           : ${BUILD_DOC}")
if(INTERNAL_DOC)
    message(NOTICE "Build internal docs     : ${INTERNAL_DOC}")
endif()
message(NOTICE "\nTARGETS")
message(NOTICE "Build the EXAMPLE programs   : ${BUILD_EXAMPLES}")
message(NOTICE "Build GTEST unit-tests       : ${BUILD_GTEST}")
