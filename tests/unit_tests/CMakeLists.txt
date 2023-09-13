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

add_executable(option_registry option_registry_tests.cpp)
target_link_libraries(option_registry aocl-da gtest_main gmock_main)

add_executable(errors_tests errors_tests.cpp)
target_link_libraries(errors_tests aocl-da gtest_main gmock_main)

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

gtest_discover_tests(basic_stats)
gtest_discover_tests(csv)
gtest_discover_tests(linmod)
gtest_discover_tests(option_registry)
gtest_discover_tests(data)
gtest_discover_tests(interval_map)
gtest_discover_tests(errors_tests)

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
