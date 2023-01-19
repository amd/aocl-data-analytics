include(CTest)
include(GoogleTest)

add_executable(basic_statistics basic_statistics.cpp)
add_executable(csv csv.cpp)

target_compile_definitions(csv PUBLIC DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/")

target_include_directories(basic_statistics PRIVATE ${CMAKE_SOURCE_DIR}/source/include ${CMAKE_SOURCE_DIR}/source/core/basic_statistics)
target_include_directories(csv PRIVATE ${CMAKE_SOURCE_DIR}/source/include ${CMAKE_SOURCE_DIR}/source/core/csv)

target_link_libraries(basic_statistics da_core gtest_main)
target_link_libraries(csv da_core gtest_main)

gtest_discover_tests(basic_statistics)
gtest_discover_tests(csv)