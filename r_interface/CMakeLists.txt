# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: 1.
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer. 2. Redistributions in binary
# form must reproduce the above copyright notice, this list of conditions and
# the following disclaimer in the documentation and/or other materials provided
# with the distribution. 3. Neither the name of the copyright holder nor the
# names of its contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Standalone CMake project for the AOCL-DA R binary package.
#
# This project is completely independent from the main AOCL-DA CMake build.
# Configure and build it separately, pointing it at an existing AOCL-DA
# shared-library install:
#
#   cd r_interface
#   cmake -B build [-DR_DA_LIB=<path>] [-DR_DA_INCLUDE=<path>]
#   cmake --build build --target rpackage
#
# The `rpackage` target (not in ALL) bundles libaocl-da.so and all AOCL
# runtime dependencies, patches RPATHs to $ORIGIN, compiles the Rcpp wrapper,
# and produces a pre-compiled binary tarball inside the build directory.
# The package is NOT installed automatically.  To install:
#   R CMD INSTALL build/aoclda_<ver>_R_x86_64-pc-linux-gnu.tar.gz
#
# Cache variables (override at configure time with -D):
#   R_DA_LIB     - path to libaocl-da.so
#                  (default: ../build/install/lib/LP64/libaocl-da.so)
#   R_DA_INCLUDE - directory containing aoclda.h
#                  (default: ../build/install/include/LP64)

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(aoclda-rpackage LANGUAGES NONE)

# ── Find R ────────────────────────────────────────────────────────────────────

find_program(R_EXECUTABLE R)
if(NOT R_EXECUTABLE)
  message(FATAL_ERROR "R not found. Install R and ensure it is on PATH.")
endif()
message(STATUS "Found R: ${R_EXECUTABLE}")

# ── Find patchelf (Linux only; required for RPATH rewriting) ──────────────────

if(NOT UNIX)
  message(FATAL_ERROR "rpackage: binary patching is only supported on Linux.")
endif()

find_program(PATCHELF patchelf)
if(NOT PATCHELF)
  message(FATAL_ERROR
    "patchelf not found. Install it (e.g. `apt install patchelf` or "
    "`yum install patchelf`) before configuring this project.")
endif()
message(STATUS "Found patchelf: ${PATCHELF}")

# ── Locate libaocl-da.so and its headers ─────────────────────────────────────
#
# Defaults point at the standard install location produced by the sibling
# AOCL-DA build (LP64).  Override with -DR_DA_LIB / -DR_DA_INCLUDE if using
# a different install path.  Note: ILP64 builds are not supported by the R
# interface (R integers are 32-bit).

set(DA_LIB_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../build/install/lib/LP64"
    CACHE STRING "Directory containing libaocl-da.so")

set(DA_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../build/install/include/LP64"
    CACHE STRING "Directory containing aoclda.h")

# ── Compiler flags to embed in src/Makevars ───────────────────────────────────

if(CMAKE_BUILD_TYPE MATCHES "[Dd]ebug")
  set(_r_cxxflags "-O0 -g -DDEBUG")
else()
  set(_r_cxxflags "-O3 -DNDEBUG")
endif()

# Optional: override the C++ compiler used by R CMD INSTALL.
# When empty (default), R uses whatever compiler it was configured with.
# Example: cmake -B build -DR_CXX=clang++
set(R_CXX "" CACHE STRING
    "C++ compiler for R package compilation (empty = use R's default)")

# ── rpackage custom target ────────────────────────────────────────────────────

add_custom_target(rpackage
  COMMAND ${CMAKE_COMMAND}
    -DR_EXECUTABLE=${R_EXECUTABLE}
    -DPATCHELF=${PATCHELF}
    -DDA_LIB_DIR=${DA_LIB_DIR}
    -DDA_INCLUDE_DIR=${DA_INCLUDE_DIR}
    -DPKG_CXXFLAGS=${_r_cxxflags}
    -DR_CXX=${R_CXX}
    -DPACKAGE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    -DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rpackage.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Building AOCL-DA R binary package"
  VERBATIM
)

# ── rdocs: regenerate man/ pages from roxygen2 comments ───────────────────────
#
# Usage: cmake --build build --target rdocs
# Requires: roxygen2 R package

add_custom_target(rdocs
  COMMAND ${CMAKE_COMMAND}
    -DR_EXECUTABLE=${R_EXECUTABLE}
    -DDA_LIB_DIR=${DA_LIB_DIR}
    -DDA_INCLUDE_DIR=${DA_INCLUDE_DIR}
    -DPKG_CXXFLAGS=${_r_cxxflags}
    -DR_CXX=${R_CXX}
    -DPACKAGE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    -DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -DPATCHELF=${PATCHELF}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rdocs.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Regenerating R package documentation (man/ pages)"
  VERBATIM
)
