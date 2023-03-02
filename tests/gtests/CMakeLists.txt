# gtest instructions, harvested from
# https://github.com/google/googletest/blob/main/googletest/README.md
include(FetchContent)
FetchContent_Declare(
  googletest
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)
include_directories(${CMAKE_SOURCE_DIR}/source/include
                    ${CMAKE_SOURCE_DIR}/source/core/csv
                    ${CMAKE_SOURCE_DIR}/source/core/basic_statistics
                    ${CMAKE_SOURCE_DIR}/source/core/linear_model
                    ${CMAKE_SOURCE_DIR}/source/core/optimization
                    ${CMAKE_SOURCE_DIR}/source/core/drivers
                    ${CMAKE_SOURCE_DIR}/source/core/utilities)

add_executable(basic_statistics basic_statistics.cpp)
target_link_libraries(basic_statistics da_core gtest_main)

add_executable(csv csv.cpp)
target_link_libraries(csv da_core gtest_main gmock_main)
target_compile_definitions(
  csv PUBLIC DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/")

add_executable(linmod linmod_tests.cc)
target_link_libraries(linmod da_core gtest_main gmock_main)
target_compile_definitions(
  linmod PUBLIC DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/linmod_data")

add_executable(option_registry option_registry_tests.cpp)
target_link_libraries(option_registry da_core gtest_main gmock_main)

target_include_directories(
  basic_statistics PRIVATE )
target_include_directories(csv PRIVATE )


gtest_discover_tests(basic_statistics)
gtest_discover_tests(csv)
gtest_discover_tests(linmod)
gtest_discover_tests(option_registry)
