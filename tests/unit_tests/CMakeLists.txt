# Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
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
  GIT_TAG v1.13.x)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
  ON
  CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)
include_directories(
  ${CMAKE_SOURCE_DIR}/source/include
  ${CMAKE_SOURCE_DIR}/source/core/basic_statistics
  ${CMAKE_SOURCE_DIR}/source/core/clustering
  ${CMAKE_SOURCE_DIR}/source/core/csv
  ${CMAKE_SOURCE_DIR}/source/core/data_management
  ${CMAKE_SOURCE_DIR}/source/core/decision_forest
  ${CMAKE_SOURCE_DIR}/source/core/drivers
  ${CMAKE_SOURCE_DIR}/source/core/factorization
  ${CMAKE_SOURCE_DIR}/source/core/linear_model
  ${CMAKE_SOURCE_DIR}/source/core/optimization
  ${CMAKE_SOURCE_DIR}/source/core/utilities
  ${CMAKE_SOURCE_DIR}/external/Lbfgsb.3.0/)

add_executable(basic_stats basic_statistics_tests.cpp)

add_executable(csv csv_tests.cpp)
target_compile_definitions(
  csv PRIVATE DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/")

add_executable(linmod linmod_tests.cpp)
target_compile_definitions(
  linmod PRIVATE DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/linmod_data")

add_executable(logreg linmod_logreg_tests.cpp)
target_compile_definitions(
  logreg
  PRIVATE DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/linmod_data/logistic")

add_executable(linreg linmod_linreg_tests.cpp)
target_compile_definitions(
  linreg
  PRIVATE DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/linmod_data/linear_reg")

add_executable(decision_tree_tests decision_tree_tests.cpp)
target_compile_definitions(
  decision_tree_tests PRIVATE DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/")

add_executable(option_registry option_registry_tests.cpp)

add_executable(errors_tests errors_tests.cpp)

add_executable(parallel_test parallel_test.cpp)

add_executable(kmeans_tests kmeans_tests.cpp)

add_executable(pca_tests pca_tests.cpp)
target_compile_definitions(
  pca PRIVATE DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/factorization_data/")

add_executable(data data_tests.cpp)
target_compile_definitions(
  data PRIVATE DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/")

add_executable(interval_map interval_map_tests.cpp)

add_executable(interval_set interval_set_tests.cpp)

add_executable(doc_tests doc_tests.cpp)

add_executable(printf_debug printf_debug.cpp)
target_link_libraries(printf_debug PRIVATE aocl-da)

add_executable(lbfgsb_rc lbfgsb_rc.cpp)
target_link_libraries(lbfgsb_rc PRIVATE aocl-da)

add_executable(lbfgsb_fcomm lbfgsb_fcomm.cpp)
target_link_libraries(lbfgsb_fcomm PRIVATE aocl-da)


add_executable(coord coord_tests.cpp)
target_compile_definitions(
  coord
  PUBLIC DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/linmod_data/linear_reg")

# C compatibility test - compile with C compiler and link with C++ compiler
set_source_files_properties(c_compatibility.c PROPERTIES LANGUAGE C)
add_executable(c_compatibility_test c_compatibility.c)
set_target_properties(c_compatibility_test PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(c_compatibility_test PRIVATE aocl-da)
# add_test(c_compatibility_test c_compatibility_test)

add_executable(miscellaneous_tests miscellaneous_tests.cpp)
target_compile_definitions(miscellaneous_tests PRIVATE ${VERS_DEF})

set(GTEST_EXES
  basic_stats
  csv
  linmod
  logreg
  linreg
  decision_tree_tests
  option_registry
  data
  interval_map
  interval_set
  errors_tests
  doc_tests
  pca_tests
  kmeans_tests
  parallel_test
  miscellaneous_tests
  coord
)

foreach(target ${GTEST_EXES})
  target_link_libraries(${target} PRIVATE aocl-da gtest_main gmock_main)
  if(BUILD_SMP OR UNIX)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  target_link_directories(${target} PRIVATE ${IFORT_LIBS})
  target_link_directories(${target} PRIVATE ${FLANG_LIBS})
  target_link_libraries(${target} PRIVATE ${FORTRAN_RUNTIME})
  # Define -DAOCL_ILP64 when ILP64 libraries are built. For LP64, this is an empty string.
  # This is what AOCL-DA users will need to do explicitly to get the correct symbols, otherwise
  # they will get linking errors or aborts.
  target_compile_definitions(${target} PRIVATE ${AOCLDA_ILP64})
endforeach()

if(WIN32 AND BUILD_SHARED_LIBS)
  foreach(GTEST_EXE ${GTEST_EXES})
    add_custom_command(
      TARGET ${GTEST_EXE}
      POST_BUILD
      COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_RUNTIME_DLLS:${GTEST_EXE}> $<TARGET_FILE_DIR:${GTEST_EXE}>
      COMMAND_EXPAND_LISTS)

    # gtest_discover_tests does not work on Windows shared library builds - this
    # is the best we can do for now
    add_test(${GTEST_EXE} ${GTEST_EXE})
  endforeach()
endif()

foreach(GTEST_EXE ${GTEST_EXES})
  gtest_discover_tests(${GTEST_EXE})
endforeach()

# Add unit test programs that are not gtest programs to CTEST
set(INDEP_TESTS c_compatibility_test printf_debug lbfgsb_rc lbfgsb_fcomm)

foreach(ITEST ${INDEP_TESTS})
  add_test(${ITEST} ${ITEST})
  if(BUILD_SMP OR UNIX)
    target_link_libraries(${ITEST} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  # Define -DAOCL_ILP64 when ILP64 libraries are built. For LP64, this is an empty string.
  # This is what AOCL-DA users will need to do explicitly to get the correct symbols, otherwise
  # they will get linking errors or aborts.
  target_compile_definitions(${ITEST} PRIVATE ${AOCLDA_ILP64})
  target_link_directories(${ITEST} PRIVATE ${IFORT_LIBS})
  target_link_libraries(${ITEST} PRIVATE ${FORTRAN_RUNTIME})
endforeach()

# Add all executables to the coverage test
if(COVERAGE)
  list(APPEND COV_EXECUTABLES "${CMAKE_CURRENT_BINARY_DIR}/printf_debug")
  set(COV_EXECUTABLES
    ${COV_EXECUTABLES}
    PARENT_SCOPE)
endif()
