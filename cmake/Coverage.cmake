#find_program(GCOV_PATH gcov REQUIRED)
#find_program(GCOVR_PATH gcovr REQUIRED)
find_program(LCOV lcov REQUIRED)
find_program(GENHTML genhtml REQUIRED)
find_program(GCOV_PATH NAMES $ENV{GCOV_NAME} gcov HINTS "/usr" PATH_SUFFIXES "bin" DOC "GNU gcov binary" REQUIRED)
message(STATUS "GNU gcov binary: ${GCOV_PATH}")

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
add_dependencies(cleanall clean-coverage)

# create coverage target for ctest make coverage will run ctest and build a
# report in the build directory
add_custom_target(
  coverage
  COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_CURRENT_BINARY_DIR} create-cov-dir
  COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_CURRENT_BINARY_DIR} all
  COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_CURRENT_BINARY_DIR} clean-coverage
  COMMAND CLICOLOR=0 ctest --timeout 20 --output-junit Testing/Temporary/LastTest_JUnit.xml || true
  # Use LCOV to circumvent Jenkins error
  COMMAND ${LCOV} --rc lcov_branch_coverage=1 -d . -c -o ${COV_DIR}/coverage.info --gcov-tool ${GCOV_PATH}
  COMMAND ${LCOV} --rc lcov_branch_coverage=1 --gcov-tool ${GCOV_PATH} --remove ${COV_DIR}/coverage.info -o ${COV_DIR}/coverage_filtered.info '/usr/*' '/*/_deps/*'
  COMMAND ${GENHTML} --branch-coverage  ${COV_DIR}/coverage_filtered.info --output ${COV_DIR}/html --title "AOCL-DA Code Coverage Report" --legend
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  # COMMAND ${GCOVR_PATH} -v --html ${COV_DIR}/index.html --html-details -r
  #         ${CMAKE_SOURCE_DIR} -e .*_deps*.)
)

# function to build a coverage report from a set of executable names argument:
# string containing a list of all executables to run - runs all the executables
# in ARGN - generate an htm gcovr report in ${COV_DIR
#function(exec_cov_target)
#  set(REPORT_DEP "")
#  foreach(EXEC ${ARGN})
#    cmake_path(GET EXEC FILENAME EXE_NAME)
#    list(APPEND REPORT_DEP ${EXE_NAME})
#    set(EXEC_TARGET_name "run_exe_${EXE_NAME}")
#    add_custom_target(
#      ${EXEC_TARGET_name} ALL
#      DEPENDS ${EXE_NAME}
#      COMMAND "${EXEC}" > /dev/null)
#  endforeach()
#  add_custom_target(
#    cov_report ALL
#    DEPENDS "${REPORT_DEP}"
#    COMMAND ${GCOVR_PATH} --html ${COV_DIR}/index.html --html-details -r
#            ${CMAKE_SOURCE_DIR} -e .*_deps*. .*/build/.*)
#endfunction()
