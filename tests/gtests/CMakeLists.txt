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
  ${CMAKE_SOURCE_DIR}/external/Lbfgsb.3.0/)

add_executable(basic_statistics basic_statistics.cpp)
target_link_libraries(basic_statistics da_core gtest_main)

add_executable(csv csv_tests.cpp)
target_link_libraries(csv da_core gtest_main gmock_main)
if(WIN32)
  target_link_libraries(csv "$ENV{LIBIFCORE}")
endif()
target_compile_definitions(
  csv PUBLIC DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/")

add_executable(linmod linmod_tests.cpp)
target_link_libraries(linmod da_core gtest_main gmock_main)
if(WIN32)
  target_link_libraries(linmod "$ENV{LIBIFCORE}")
endif()
target_compile_definitions(
  linmod PUBLIC DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/linmod_data")

add_executable(option_registry option_registry_tests.cpp)
target_link_libraries(option_registry da_core gtest_main gmock_main)

add_executable(errors_tests errors_tests.cpp)
target_link_libraries(errors_tests da_core gtest_main gmock_main)

target_include_directories(basic_statistics PRIVATE)
target_include_directories(csv PRIVATE)

add_executable(data data_tests.cpp)
target_link_libraries(data da_core gtest_main gmock_main)
if(WIN32)
  target_link_libraries(data "$ENV{LIBIFCORE}")
endif()

add_executable(printf_debug printf_debug.cpp)
target_link_libraries(printf_debug da_core)

gtest_discover_tests(basic_statistics)
gtest_discover_tests(csv)
gtest_discover_tests(linmod)
gtest_discover_tests(option_registry)
gtest_discover_tests(data)
gtest_discover_tests(errors_tests)


# add unit test programs that are not gtest programs to CTEST
string(CONCAT TNAME printf_debug "_ex")
add_test(${TNAME} printf_debug)

# Add all executables to the coverage test
if(COVERAGE)
  list(APPEND COV_EXECUTABLES "${CMAKE_CURRENT_BINARY_DIR}/printf_debug")
  set(COV_EXECUTABLES
      ${COV_EXECUTABLES}
      PARENT_SCOPE)
endif()
