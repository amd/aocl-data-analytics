# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# rpackage.cmake
#
# Build script for the AOCL-DA R binary package.
# Invoked at build time via `cmake -P` by the `rpackage` custom target.
#
# Required variables (passed with -D from CMakeLists.txt):
#   R_EXECUTABLE       - path to the R interpreter
#   PATCHELF           - path to the patchelf utility
#   DA_LIB_DIR         - directory containing libaocl-da.so
#   DA_INCLUDE_DIR     - directory containing aoclda.h
#   PKG_CXXFLAGS       - compiler flags to embed in src/Makevars
#   PACKAGE_SOURCE_DIR - absolute path to the r_interface/ directory
#   BUILD_DIR          - absolute path to the CMake binary directory (build/)

message(STATUS "=== AOCL-DA R Package Build ===")

# ── Platform check ────────────────────────────────────────────────────────────
if(WIN32)
  message(FATAL_ERROR
    "The R interface is currently supported on Linux only.")
endif()

# ── Validate required inputs ──────────────────────────────────────────────────

set(DA_LIB "${DA_LIB_DIR}/libaocl-da.so")

if(NOT EXISTS "${DA_LIB}")
  message(FATAL_ERROR
    "libaocl-da.so not found at '${DA_LIB}'.\n"
    "Build and install AOCL-DA first, then re-run this target, "
    "or override the path with -DDA_LIB_DIR=<path>.")
endif()

if(NOT IS_DIRECTORY "${DA_INCLUDE_DIR}" OR NOT EXISTS "${DA_INCLUDE_DIR}/aoclda.h")
  message(FATAL_ERROR
    "aoclda.h not found under '${DA_INCLUDE_DIR}'.\n"
    "Ensure AOCL-DA is installed with headers, "
    "or override with -DR_DA_INCLUDE=<path>.")
endif()

find_program(_ldd ldd REQUIRED)

# ── Stage inst/ directory ─────────────────────────────────────────────────────

set(INST_DIR     "${PACKAGE_SOURCE_DIR}/inst")
set(INST_INCLUDE "${INST_DIR}/include")
set(INST_LIBS    "${INST_DIR}/libs")

file(MAKE_DIRECTORY "${INST_INCLUDE}")
file(MAKE_DIRECTORY "${INST_LIBS}")

# Copy public headers
message(STATUS "Staging headers from: ${DA_INCLUDE_DIR}")
file(GLOB _headers "${DA_INCLUDE_DIR}/*.h" "${DA_INCLUDE_DIR}/*.hpp")
foreach(_hdr IN LISTS _headers)
  file(COPY "${_hdr}" DESTINATION "${INST_INCLUDE}")
endforeach()

# Copy libaocl-da.so, following the symlink chain so both the soname link and
# the versioned real file land in inst/libs/.
message(STATUS "Staging libaocl-da.so from: ${DA_LIB}")
file(COPY "${DA_LIB}" DESTINATION "${INST_LIBS}" FOLLOW_SYMLINK_CHAIN)

# ── Collect and bundle transitive AOCL deps via ldd ──────────────────────────
#
# ldd output is already fully transitive.  We exclude standard system libraries
# that are guaranteed to be present on any compatible Linux machine.

set(_system_exclude_patterns
  "^libm\\.so"
  "^libc\\.so"
  "^libpthread\\.so"
  "^libdl\\.so"
  "^librt\\.so"
  "^libgcc_s\\.so"
  "^libstdc\\+\\+\\.so"
  "^libgomp\\.so"
  "^linux-vdso\\.so"
  "^ld-linux"
)

message(STATUS "Collecting runtime dependencies via ldd ...")
execute_process(
  COMMAND "${_ldd}" "${DA_LIB}"
  OUTPUT_VARIABLE _ldd_output
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(REPLACE "\n" ";" _ldd_lines "${_ldd_output}")
foreach(_line IN LISTS _ldd_lines)
  # ldd format:  "  libfoo.so.N => /resolved/path/libfoo.so.N (0x...)"
  if(_line MATCHES "=> ([^ ]+) ")
    set(_dep_path "${CMAKE_MATCH_1}")
    if(_dep_path STREQUAL "not")
      message(WARNING "Unresolved dependency (not bundled): ${_line}")
      continue()
    endif()
    get_filename_component(_dep_name "${_dep_path}" NAME)
    set(_excluded FALSE)
    foreach(_pat IN LISTS _system_exclude_patterns)
      if(_dep_name MATCHES "${_pat}")
        set(_excluded TRUE)
        break()
      endif()
    endforeach()
    if(NOT _excluded)
      message(STATUS "  Bundling: ${_dep_path}")
      file(COPY "${_dep_path}" DESTINATION "${INST_LIBS}" FOLLOW_SYMLINK_CHAIN)
    endif()
  endif()
endforeach()

# ── Patch RPATHs to $ORIGIN on all non-symlink .so files ─────────────────────
#
# After installation on the target machine, both the compiled Rcpp wrapper and
# the bundled AOCL libs land under <pkgroot>/libs/.  RPATH=$ORIGIN lets them
# find each other without any environment variable setup.

message(STATUS "Patching RPATHs to \$ORIGIN ...")
file(GLOB _bundled "${INST_LIBS}/*.so*")
foreach(_lib IN LISTS _bundled)
  if(NOT IS_SYMLINK "${_lib}")
    execute_process(
      COMMAND "${PATCHELF}" --set-rpath "$ORIGIN" "${_lib}"
      RESULT_VARIABLE _rc
    )
    if(NOT _rc EQUAL 0)
      message(WARNING "patchelf failed on: ${_lib}")
    endif()
  endif()
endforeach()

# ── Generate src/Makevars ─────────────────────────────────────────────────────

message(STATUS "Generating src/Makevars ...")
if(R_CXX)
  set(_cxx_line "CXX = ${R_CXX}\nCXX17 = ${R_CXX}\n")
else()
  set(_cxx_line "")
endif()
file(WRITE "${PACKAGE_SOURCE_DIR}/src/Makevars"
"# Makevars for AOCL-DA R package
# Generated by cmake/rpackage.cmake -- do not edit manually.
${_cxx_line}CXX_STD = CXX17
PKG_CPPFLAGS = -I../inst/include
PKG_LIBS = -L../inst/libs -laocl-da -Wl,-rpath,'\$\$ORIGIN'
PKG_CXXFLAGS = ${PKG_CXXFLAGS} \$(SHLIB_OPENMP_CXXFLAGS)
")

# ── Generate RcppExports ──────────────────────────────────────────────────────

message(STATUS "Generating RcppExports ...")
execute_process(
  COMMAND "${R_EXECUTABLE}" --slave
          -e "Rcpp::compileAttributes('${PACKAGE_SOURCE_DIR}')"
  RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
  file(REMOVE_RECURSE "${INST_DIR}")
  file(REMOVE "${PACKAGE_SOURCE_DIR}/src/Makevars")
  message(FATAL_ERROR "Rcpp::compileAttributes() failed.")
endif()

# ── Clean previous build artefacts ───────────────────────────────────────────

file(GLOB _old_tarballs "${BUILD_DIR}/aoclda_*.tar.gz")
foreach(_f IN LISTS _old_tarballs)
  file(REMOVE "${_f}")
endforeach()
file(REMOVE_RECURSE "${PACKAGE_SOURCE_DIR}/.Rcheck")

# ── R CMD build ──────────────────────────────────────────────────────────────

message(STATUS "Running R CMD build ...")
execute_process(
  COMMAND "${R_EXECUTABLE}" CMD build "${PACKAGE_SOURCE_DIR}"
  WORKING_DIRECTORY "${BUILD_DIR}"
  RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
  file(REMOVE_RECURSE "${INST_DIR}")
  file(REMOVE "${PACKAGE_SOURCE_DIR}/src/Makevars")
  message(FATAL_ERROR "R CMD build failed.")
endif()

file(GLOB _src_tarballs "${BUILD_DIR}/aoclda_*.tar.gz")
if(NOT _src_tarballs)
  file(REMOVE_RECURSE "${INST_DIR}")
  file(REMOVE "${PACKAGE_SOURCE_DIR}/src/Makevars")
  message(FATAL_ERROR "R CMD build produced no aoclda_*.tar.gz tarball.")
endif()
list(GET _src_tarballs 0 _src_tarball)
message(STATUS "Built source package: ${_src_tarball}")

# ── R CMD INSTALL --build ─────────────────────────────────────────────────────
#
# --build compiles the Rcpp wrapper and emits a pre-compiled binary tarball in
# WORKING_DIRECTORY (the build dir).  We install into a temporary library that
# is discarded afterwards so the user's R library is not touched automatically.
# To install: R CMD INSTALL build/aoclda_<ver>_R_x86_64-pc-linux-gnu.tar.gz

set(_tmp_lib "${BUILD_DIR}/_R_tmplib")
file(MAKE_DIRECTORY "${_tmp_lib}")

message(STATUS "Running R CMD INSTALL --build ...")
execute_process(
  COMMAND "${R_EXECUTABLE}" CMD INSTALL --build -l "${_tmp_lib}" "${_src_tarball}"
  WORKING_DIRECTORY "${BUILD_DIR}"
  RESULT_VARIABLE _rc
)

# Clean staging area, generated Makevars, and temporary library regardless of outcome
file(REMOVE_RECURSE "${INST_DIR}")
file(REMOVE "${PACKAGE_SOURCE_DIR}/src/Makevars")
file(REMOVE_RECURSE "${_tmp_lib}")
# Remove the source tarball -- only the binary tarball needs to be kept
file(REMOVE "${_src_tarball}")

if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "R CMD INSTALL --build failed.")
endif()

# ── Report ────────────────────────────────────────────────────────────────────

file(GLOB _binary_tarballs "${BUILD_DIR}/aoclda_*linux*.tar.gz")
message(STATUS "")
message(STATUS "=== R package build complete ===")
if(_binary_tarballs)
  list(GET _binary_tarballs 0 _binary_tarball)
  message(STATUS "Binary package : ${_binary_tarball}")
  message(STATUS "")
  message(STATUS "To install into your R library:")
  message(STATUS "  R CMD INSTALL ${_binary_tarball}")
else()
  message(STATUS "(Binary tarball not detected in ${BUILD_DIR})")
endif()
