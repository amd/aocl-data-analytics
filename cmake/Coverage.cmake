find_program(GCOV_PATH gcov REQUIRED)
find_program(GCOVR_PATH gcovr REQUIRED)

set(COMPILER_FLAGS_DEBUG
    "${COMPILER_FLAGS_DEBUG};-fprofile-arcs;-ftest-coverage")
link_libraries(gcov)

# create a coverage_reports directory in the main build dir
set(COV_DIR ${CMAKE_BINARY_DIR}/coverage_reports)
add_custom_target(create-cov-dir COMMAND ${CMAKE_COMMAND} -E make_directory
                                         ${COV_DIR})
add_custom_target( clean-coverage
  COMMAND ${CMAKE_COMMAND} -E  rm -rf ${CMAKE_BINARY_DIR}/coverage_reports/*
)

# create coverage target for ctest make coverage will run ctest and build a
# report in the build directory
add_custom_target(
  coverage
  COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_CURRENT_BINARY_DIR} create-cov-dir
  COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_CURRENT_BINARY_DIR} all
  COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_CURRENT_BINARY_DIR} clean-coverage
  COMMAND CLICOLOR=0 ctest --timeout 20 --output-junit Testing/Temporary/LastTest_JUnit.xml || true
  COMMAND ${GCOVR_PATH} --html ${COV_DIR}/index.html --html-details -r
          ${CMAKE_SOURCE_DIR} -e .*_deps*.)

# function to build a coverage report from a set of executable names argument:
# string containing a list of all executables to run - runs all the executables
# in ARGN - generate an htm gcovr report in ${COV_DIR
function(exec_cov_target)
  set(REPORT_DEP "")
  foreach(EXEC ${ARGN})
    cmake_path(GET EXEC FILENAME EXE_NAME)
    list(APPEND REPORT_DEP ${EXE_NAME})
    set(EXEC_TARGET_name "run_exe_${EXE_NAME}")
    add_custom_target(
      ${EXEC_TARGET_name} ALL
      DEPENDS ${EXE_NAME}
      COMMAND "${EXEC}" > /dev/null)
  endforeach()
  add_custom_target(
    cov_report ALL
    DEPENDS "${REPORT_DEP}"
    COMMAND ${GCOVR_PATH} --html ${COV_DIR}/index.html --html-details -r
            ${CMAKE_SOURCE_DIR} -e .*_deps*. .*/build/.*)
endfunction()
