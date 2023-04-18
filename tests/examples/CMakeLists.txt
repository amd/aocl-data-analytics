include_directories(${CMAKE_SOURCE_DIR}/source/include)

add_executable(lbfgsb_rc lbfgsb_rc.cpp)
target_include_directories(
  lbfgsb_rc PRIVATE ${CMAKE_SOURCE_DIR}/source/core/drivers
                    ${CMAKE_SOURCE_DIR}/source/core/optimization
                    ${CMAKE_SOURCE_DIR}/source/core/utilities
                    ${CMAKE_SOURCE_DIR}/external/Lbfgsb.3.0/
                    )
target_link_libraries(lbfgsb_rc lbfgsb lbfgsb_s)

add_executable(lbfgsb_fcomm lbfgsb_fcomm.cpp)
target_include_directories(
  lbfgsb_fcomm PRIVATE ${CMAKE_SOURCE_DIR}/source/core/optimization
                       ${CMAKE_SOURCE_DIR}/source/core/utilities
                       ${CMAKE_SOURCE_DIR}/source/core/drivers
                       ${CMAKE_SOURCE_DIR}/external/Lbfgsb.3.0/
                       )
target_link_libraries(lbfgsb_fcomm da_core)

add_executable(basic_statistics_test basic_statistics_test.cpp)
target_link_libraries(basic_statistics_test da_core)

add_executable(linear_model linear_model.cpp)
target_link_libraries(linear_model da_core)

add_executable(reglinear_model reglinear_model.cpp)
target_link_libraries(reglinear_model da_core)

set(EXAMPLE_EXES reglinear_model linear_model basic_statistics_test lbfgsb_rc lbfgsb_fcomm)

add_executable(basic_pca_test basic_pca_test.cpp)
target_link_libraries(basic_pca_test da_core)

# add all example programs to CTEST
foreach(EX_NAME ${EXAMPLE_EXES})
  string(CONCAT TNAME ${EX_NAME} "_ex")
  add_test(${TNAME} ${EX_NAME})
endforeach()

# Add all executables to the coverage test
if(COVERAGE)
  foreach(EX_NAME ${EXAMPLE_EXES})
    list(APPEND COV_EXECUTABLES "${CMAKE_CURRENT_BINARY_DIR}/${EX_NAME}")
  endforeach()
  set(COV_EXECUTABLES
      ${COV_EXECUTABLES}
      PARENT_SCOPE)
endif()
