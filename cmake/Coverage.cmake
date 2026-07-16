# Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
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


find_program(LCOV lcov REQUIRED)
find_program(GENHTML genhtml REQUIRED)
find_program(GCOV_PATH NAMES $ENV{GCOV_NAME} gcov HINTS "$ENV{PATH}" PATH_SUFFIXES "bin" DOC "GNU gcov binary" REQUIRED)
message(STATUS "GNU gcov binary: ${GCOV_PATH}")

# Cap OMP_NUM_THREADS during coverage test runs to avoid oversubscription
# when many tests run concurrently. Defaults to 1 if not set in the environment.
if(DEFINED ENV{OMP_NUM_THREADS})
  set(COV_OMP_NUM_THREADS "$ENV{OMP_NUM_THREADS}")
else()
  set(COV_OMP_NUM_THREADS 1)
endif()
message(STATUS "Coverage: OMP_NUM_THREADS=${COV_OMP_NUM_THREADS}")

# Compute default parallel test level: min(16, NPROC/2).
# Can be overridden at configure time with -DCOV_PARALLEL_LEVEL=<N>.
include(ProcessorCount)
ProcessorCount(NPROC)
if(NPROC EQUAL 0)
  set(NPROC 1)
endif()
math(EXPR HALF_NPROC "${NPROC} / 2")
if(HALF_NPROC LESS 1)
  set(HALF_NPROC 1)
endif()
if(HALF_NPROC GREATER 16)
  set(HALF_NPROC 16)
endif()
set(COV_PARALLEL_LEVEL "${HALF_NPROC}" CACHE STRING
    "Number of tests to run concurrently during coverage (default: min(16, NPROC/2))")

# Create a coverage_reports directory in the main build directory
set(COV_DIR ${CMAKE_BINARY_DIR}/coverage_reports)
add_custom_target(create-cov-dir COMMAND ${CMAKE_COMMAND} -E make_directory
                                         ${COV_DIR})
add_custom_target( clean-coverage
  COMMAND ${CMAKE_COMMAND} -E  rm -rf ${CMAKE_BINARY_DIR}/coverage_reports/*
)
add_dependencies(cleanall clean-coverage)

# Create coverage target for ctest. 'make coverage' will run ctest and build a
# report in the build directory.
# Parallelism is controlled via COV_PARALLEL_LEVEL (configure-time default)
# and OMP_NUM_THREADS (defaults to 2 if unset).
# Runtime CTEST_PARALLEL_LEVEL takes precedence over the configure-time default.
add_custom_target(
  coverage
  COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_CURRENT_BINARY_DIR} create-cov-dir
  COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_CURRENT_BINARY_DIR} all
  COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_CURRENT_BINARY_DIR} clean-coverage
  COMMAND CLICOLOR=0 sh -c "OMP_NUM_THREADS=\$\${OMP_NUM_THREADS:-${COV_OMP_NUM_THREADS}} CTEST_PARALLEL_LEVEL=\$\${CTEST_PARALLEL_LEVEL:-${COV_PARALLEL_LEVEL}} ctest --timeout 600 --output-junit Testing/Temporary/LastTest_JUnit.xml || true"
  # Use LCOV to circumvent Jenkins error
  COMMAND ${LCOV} --rc lcov_branch_coverage=1 -d . -c -o ${COV_DIR}/coverage.info --gcov-tool ${GCOV_PATH}
  COMMAND ${LCOV} --rc lcov_branch_coverage=1 --gcov-tool ${GCOV_PATH} -o ${COV_DIR}/coverage_filtered.info --remove ${COV_DIR}/coverage.info '/amd*' '/usr/*' '/*/_deps/*' '/*/external/*/*.F90' '/*/external/*/*.f90'
  COMMAND ${GENHTML} --branch-coverage  ${COV_DIR}/coverage_filtered.info --output ${COV_DIR}/html --title "AOCL-DA Code Coverage Report" --legend
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)
