include_directories(${CMAKE_SOURCE_DIR}/source/include)

# FIXME: data files need also to be copied at installation of DA

add_executable(lbfgsb_rc lbfgsb_rc.cpp)
target_include_directories(
  lbfgsb_rc
  PRIVATE ${CMAKE_SOURCE_DIR}/source/core/drivers
          ${CMAKE_SOURCE_DIR}/source/core/optimization
          ${CMAKE_SOURCE_DIR}/source/core/utilities
          ${CMAKE_SOURCE_DIR}/external/Lbfgsb.3.0/)
target_link_libraries(lbfgsb_rc aocl-da)

add_executable(lbfgsb_fcomm lbfgsb_fcomm.cpp)
target_include_directories(
  lbfgsb_fcomm
  PRIVATE ${CMAKE_SOURCE_DIR}/source/core/optimization
          ${CMAKE_SOURCE_DIR}/source/core/utilities
          ${CMAKE_SOURCE_DIR}/source/core/drivers
          ${CMAKE_SOURCE_DIR}/external/Lbfgsb.3.0/)
target_link_libraries(lbfgsb_fcomm aocl-da)

add_executable(basic_statistics basic_statistics.cpp)
target_link_libraries(basic_statistics aocl-da)

add_executable(linear_model linear_model.cpp)
target_link_libraries(linear_model aocl-da)

add_executable(reglinear_model reglinear_model.cpp)
target_link_libraries(reglinear_model aocl-da)

add_executable(linmod_diabetes linmod_diabetes.cpp)
target_link_libraries(linmod_diabetes aocl-da)
target_compile_definitions(
  linmod_diabetes PUBLIC DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/linmod_data")

add_executable(datastore datastore.cpp)
target_link_libraries(datastore aocl-da)
target_compile_definitions(
  datastore PUBLIC DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/linmod_data")

add_executable(decision_tree decision_tree.cpp)
target_link_libraries(decision_tree aocl-da)
target_compile_definitions(
  decision_tree PUBLIC DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/")

add_executable(basic_pca basic_pca.cpp)
target_link_libraries(basic_pca aocl-da)

set(EXAMPLE_EXES
    reglinear_model
    linear_model
    linmod_diabetes
    basic_statistics
    lbfgsb_rc
    lbfgsb_fcomm
    datastore
    decision_tree)

# For Windows shared library builds copying some relevant DLLs into the examples bin folder enables ctest to run without needing to update PATH
if (WIN32 AND BUILD_SHARED_LIBS)
  foreach(EXAMPLE_EXE ${EXAMPLE_EXES})
    add_custom_command(
        TARGET ${EXAMPLE_EXE} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${EXAMPLE_EXE}> $<TARGET_FILE_DIR:${EXAMPLE_EXE}>
        COMMAND_EXPAND_LISTS
	)
  endforeach()
endif ()

# Add all example programs to CTEST
foreach(EX_NAME ${EXAMPLE_EXES})
  string(CONCAT TNAME ${EX_NAME} "_ex")
  add_test(${TNAME} ${EX_NAME})
endforeach()

# Add all executables to the coverage test
if(COVERAGE)
  foreach(EX_NAME ${EXAMPLE_EXES})
    list(APPEND COV_EXECUTABLES "${CMAKE_CURRENT_BINARY_DIR}/${EX_NAME}")
  endforeach()
endif()
