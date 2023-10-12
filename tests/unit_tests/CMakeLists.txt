# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# 


# Exclude gtest and gmock from installation
set(INSTALL_GTEST OFF)
set(INSTALL_GMOCK OFF)

# get gtest from repo
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.13.x
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)
include_directories(
  ${CMAKE_SOURCE_DIR}/source/include
  ${CMAKE_SOURCE_DIR}/source/core/csv
  ${CMAKE_SOURCE_DIR}/source/core/basic_statistics
  ${CMAKE_SOURCE_DIR}/source/core/linear_model
  ${CMAKE_SOURCE_DIR}/source/core/factorization
  ${CMAKE_SOURCE_DIR}/source/core/optimization
  ${CMAKE_SOURCE_DIR}/source/core/drivers
  ${CMAKE_SOURCE_DIR}/source/core/utilities
  ${CMAKE_SOURCE_DIR}/source/core/data_management
  ${CMAKE_SOURCE_DIR}/source/core/decision_forest
  ${CMAKE_SOURCE_DIR}/external/Lbfgsb.3.0/)

add_executable(basic_stats basic_statistics_tests.cpp)
target_link_libraries(basic_stats aocl-da gtest_main gmock_main)
if(WIN32)
  target_link_libraries(basic_stats "$ENV{LIBIFCORE}")
endif()

add_executable(csv csv_tests.cpp)
target_link_libraries(csv aocl-da gtest_main gmock_main)
if(WIN32)
  target_link_libraries(csv "$ENV{LIBIFCORE}")
endif()
target_compile_definitions(
  csv PUBLIC DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/")

add_executable(linmod linmod_tests.cpp)
target_link_libraries(linmod aocl-da gtest_main gmock_main)
if(WIN32)
  target_link_libraries(linmod "$ENV{LIBIFCORE}")
endif()
target_compile_definitions(
  linmod PUBLIC DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/linmod_data")

add_executable(decision_tree decision_tree_tests.cpp)
target_link_libraries(decision_tree aocl-da gtest_main)
if(WIN32)
  target_link_libraries(decision_tree "$ENV{LIBIFCORE}")
endif()
target_compile_definitions(
  decision_tree PUBLIC DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/")

add_executable(decision_forest decision_forest_tests.cpp)
target_link_libraries(decision_forest aocl-da gtest_main)
if(WIN32)
  target_link_libraries(decision_forest "$ENV{LIBIFCORE}")
endif()
target_compile_definitions(
  decision_forest PUBLIC DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/")

add_executable(option_registry option_registry_tests.cpp)
target_link_libraries(option_registry aocl-da gtest_main gmock_main)

add_executable(errors_tests errors_tests.cpp)
target_link_libraries(errors_tests aocl-da gtest_main gmock_main)

add_executable(pca pca_tests.cpp)
target_link_libraries(pca aocl-da gtest_main gmock_main)
if(WIN32)
  target_link_libraries(pca "$ENV{LIBIFCORE}")
endif()
target_compile_definitions(
  pca PUBLIC DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/factorization_data/")

add_executable(data data_tests.cpp)
target_link_libraries(data aocl-da gtest_main gmock_main)
if(WIN32)
  target_link_libraries(data "$ENV{LIBIFCORE}")
endif()
target_compile_definitions(
  data PUBLIC DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/")

add_executable(interval_map interval_map_tests.cpp)
target_link_libraries(interval_map aocl-da gtest_main gmock_main)
if(WIN32)
  target_link_libraries(interval_map "$ENV{LIBIFCORE}")
endif()

add_executable(printf_debug printf_debug.cpp)
target_link_libraries(printf_debug aocl-da)

set(GTEST_EXES
    basic_stats
	csv
	linmod
	decision_forest
	option_registry
	data
	interval_map
	errors_tests
	pca
)

# For Windows shared library builds copying some relevant DLLs into the unit_tests bin folder enables ctest to run without needing to update PATH
if (WIN32 AND BUILD_SHARED_LIBS)
  foreach(GTEST_EXE ${GTEST_EXES})
    add_custom_command(
        TARGET ${GTEST_EXE} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${GTEST_EXE}> $<TARGET_FILE_DIR:${GTEST_EXE}>
        COMMAND_EXPAND_LISTS
	)
	# gtest_discover_tests does not work on Windows shared library builds - this is the best we can do for now
	add_test(${GTEST_EXE} ${GTEST_EXE})
  endforeach()
endif ()

foreach(GTEST_EXE ${GTEST_EXES})
    gtest_discover_tests(${GTEST_EXE})
endforeach()

# Add unit test programs that are not gtest programs to CTEST
string(CONCAT TNAME printf_debug "_ex")
add_test(${TNAME} printf_debug)

# C compatibility test - compile with C compiler and link with C++ compiler
set_source_files_properties(c_compatibility.c PROPERTIES LANGUAGE C)
add_executable(c_compatibility c_compatibility.c)
set_target_properties(c_compatibility PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(c_compatibility aocl-da)
string(CONCAT TNAME c_compatibility "_test")
add_test(${TNAME} c_compatibility)

# Add all executables to the coverage test
if(COVERAGE)
  list(APPEND COV_EXECUTABLES "${CMAKE_CURRENT_BINARY_DIR}/printf_debug")
  set(COV_EXECUTABLES
      ${COV_EXECUTABLES}
      PARENT_SCOPE)
endif()
