add_executable(lbfgsb_rc_example lbfgsb_rc_example.cpp)
target_include_directories(lbfgsb_rc_example PRIVATE ${CMAKE_SOURCE_DIR}/source/core/drivers ${CMAKE_SOURCE_DIR}/source/core/optimization)
target_link_libraries(lbfgsb_rc_example lbfgsb)

add_executable(basic_statistics_test basic_statistics_test.cpp)
target_include_directories(basic_statistics_test PRIVATE ${CMAKE_SOURCE_DIR}/source/include)
target_link_libraries(basic_statistics_test da_core)

add_executable(linear_model linear_model.cpp)
set(LINMOD)
target_include_directories(linear_model PRIVATE ${CMAKE_SOURCE_DIR}/source/include)
target_link_libraries(linear_model da_core)

# Add all executables to the coverage test
# TODO Gather all executables automatically
if (COVERAGE)
    list(APPEND COV_EXECUTABLES "${CMAKE_CURRENT_BINARY_DIR}/linear_model")
    list(APPEND COV_EXECUTABLES "${CMAKE_CURRENT_BINARY_DIR}/lbfgsb_rc_example")
    list(APPEND COV_EXECUTABLES "${CMAKE_CURRENT_BINARY_DIR}/basic_statistics_test")
    set(COV_EXECUTABLES ${COV_EXECUTABLES} PARENT_SCOPE)
endif()