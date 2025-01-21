# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#

# Check we can find AOCL or custom blas/lapack installation
function(linalg_libs)
  # Set paths to BLAS/LAPACK/SPARSE/UTILS if CMAKE_AOCL_ROOT was set but
  # *_INCLUDE_DIR and *_LIB were not set. This means that if they were set, they
  # override the location of CMAKE_AOCL_ROOT.
  if(WIN32)
    if(BLAS_INCLUDE_DIR STREQUAL "")
      set(BLAS_INCLUDE_DIR ${CMAKE_AOCL_ROOT}/amd-blis/include/${INT_LIB})
    endif()
    if(LAPACK_INCLUDE_DIR STREQUAL "")
      set(LAPACK_INCLUDE_DIR ${CMAKE_AOCL_ROOT}/amd-libflame/include/${INT_LIB})
    endif()
    if(SPARSE_INCLUDE_DIR STREQUAL "")
      set(SPARSE_INCLUDE_DIR ${CMAKE_AOCL_ROOT}/amd-sparse/include/)
    endif()
    if(UTILS_INCLUDE_DIR STREQUAL "")
      set(UTILS_INCLUDE_DIR ${CMAKE_AOCL_ROOT}/amd-utils/include/)
    endif()
    if(BLAS_LIB STREQUAL "")
      set(BLAS_LIB_DIR ${CMAKE_AOCL_ROOT}/amd-blis/lib/${INT_LIB})
    endif()
    if(LAPACK_LIB STREQUAL "")
      set(LAPACK_LIB_DIR ${CMAKE_AOCL_ROOT}/amd-libflame/lib/${INT_LIB})
    endif()
    if(SPARSE_LIB STREQUAL "")
      set(SHARED_DIR_NAME shared)
      set(SPARSE_LIB_DIR
          ${CMAKE_AOCL_ROOT}/amd-sparse/lib/${INT_LIB}/${SHARED_DIR_NAME})
    endif()
    if(UTILS_LIB STREQUAL "")
      set(UTILS_LIB_DIR ${CMAKE_AOCL_ROOT}/amd-utils/lib)
    endif()
    if(DA_LIB STREQUAL "")
      # Only used if we are building Python with an existing DA build
      set(DA_LIB_DIR ${CMAKE_AOCL_ROOT}/amd-da/lib/${INT_LIB})
    endif()
  else()
    if(BLAS_INCLUDE_DIR STREQUAL "")
      set(BLAS_INCLUDE_DIR ${CMAKE_AOCL_ROOT}/include_${INT_LIB})
    endif()
    if(LAPACK_INCLUDE_DIR STREQUAL "")
      set(LAPACK_INCLUDE_DIR ${CMAKE_AOCL_ROOT}/include_${INT_LIB})
    endif()
    if(SPARSE_INCLUDE_DIR STREQUAL "")
      set(SPARSE_INCLUDE_DIR ${CMAKE_AOCL_ROOT}/include_${INT_LIB})
    endif()
    if(UTILS_INCLUDE_DIR STREQUAL "")
      set(UTILS_INCLUDE_DIR ${CMAKE_AOCL_ROOT}/include_${INT_LIB})
    endif()
    if(BLAS_LIB STREQUAL "")
      set(BLAS_LIB_DIR ${CMAKE_AOCL_ROOT}/lib_${INT_LIB})
    endif()
    if(LAPACK_LIB STREQUAL "")
      set(LAPACK_LIB_DIR ${CMAKE_AOCL_ROOT}/lib_${INT_LIB})
    endif()
    if(SPARSE_LIB STREQUAL "")
      set(SPARSE_LIB_DIR ${CMAKE_AOCL_ROOT}/lib_${INT_LIB})
    endif()
    if(UTILS_LIB STREQUAL "")
      set(UTILS_LIB_DIR ${CMAKE_AOCL_ROOT}/lib_${INT_LIB})
    endif()
    if(DA_LIB STREQUAL "")
      # Only used if we are building Python with an existing DA build
      set(DA_LIB_DIR ${CMAKE_AOCL_ROOT}/lib_${INT_LIB})
    endif()
    if(LIBMEM_LIB STREQUAL "")
      set(LIBMEM_LIB_DIR ${CMAKE_AOCL_ROOT}/lib_${INT_LIB})
    endif()
  endif()

  # Set names of the libraries we search for
  if(WIN32)
    set(CMAKE_FIND_LIBRARY_PREFIXES "")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
    # Always link to multi-threaded BLAS because aoclsparse also depends on it
    if(BUILD_SMP)
      set(BLAS_NAME "AOCL-LibBlis-Win-MT-dll")
      set(LAPACK_NAME "AOCL-LibFlame-Win-MT-dll")
      if(NOT CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
        # On Windows, certain SMP builds need both serial and threaded
        # BLAS/LAPACK in order to build the Python wheel, since aoclsparse
        # requires serial versions
        set(BLAS_NAME_SERIAL "AOCL-LibBlis-Win-dll")
        set(LAPACK_NAME_SERIAL "AOCL-LibFlame-Win-dll")
        find_library(
          BLAS_SERIAL name ${BLAS_NAME_SERIAL}
          PATHS ${BLAS_LIB_DIR} REQUIRED
          NO_DEFAULT_PATH)
        find_library(
          LAPACK_SERIAL name ${LAPACK_NAME_SERIAL}
          PATHS ${LAPACK_LIB_DIR} REQUIRED
          NO_DEFAULT_PATH)
      endif()
    else()
      set(BLAS_NAME "AOCL-LibBlis-Win-dll")
      set(LAPACK_NAME "AOCL-LibFlame-Win-dll")
    endif()
    set(SPARSE_NAME "aoclsparse")
    set(UTILS_NAME "libaoclutils")
    set(UTILS_CPUID_NAME "au_cpuid")
    set(DA_NAME "aocl-da") # Only used if we are building Python with an
                           # existing DA build
  else(WIN32) # linux
    set(CMAKE_FIND_LIBRARY_PREFIXES "lib")
    if(NOT BUILD_SHARED_LIBS)
      set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
    else()
      set(CMAKE_FIND_LIBRARY_SUFFIXES .so)
    endif()
    # Always link to multi-thread BLAS because aoclsparse depends on it
    set(BLAS_NAME "blis-mt")
    set(LAPACK_NAME "flame")
    set(SPARSE_NAME "aoclsparse")
    set(UTILS_NAME "aoclutils")
    set(UTILS_CPUID_NAME "au_cpuid")
    set(DA_NAME "aocl-da") # Only used if we are building Python with an
                           # existing DA build
    set(LIBMEM_NAME "aocl-libmem")
  endif()

  if(BLAS_LIB STREQUAL "")
    find_library(
      BLAS name ${BLAS_NAME}
      PATHS ${BLAS_LIB_DIR} REQUIRED
      NO_DEFAULT_PATH)
  else()
    set(BLAS
        ${BLAS_LIB}
        PARENT_SCOPE)
  endif()

  if(LAPACK_LIB STREQUAL "")
    find_library(
      LAPACK name ${LAPACK_NAME}
      PATHS ${LAPACK_LIB_DIR} REQUIRED
      NO_DEFAULT_PATH)
  else()
    set(LAPACK
        ${LAPACK_LIB}
        PARENT_SCOPE)
  endif()

  if(SPARSE_LIB STREQUAL "")
    find_library(
      SPARSE name ${SPARSE_NAME}
      PATHS ${SPARSE_LIB_DIR} REQUIRED
      NO_DEFAULT_PATH)
  else()
    set(SPARSE
        ${SPARSE_LIB}
        PARENT_SCOPE)
  endif()

  if(UTILS_LIB STREQUAL "")
    find_library(
      UTILS name ${UTILS_NAME}
      PATHS ${UTILS_LIB_DIR} REQUIRED
      NO_DEFAULT_PATH)
  else()
    set(UTILS
        ${UTILS_LIB}
        PARENT_SCOPE)
  endif()

  if(UTILS_CPUID_LIB STREQUAL "")
    find_library(
      UTILS_CPUID name ${UTILS_CPUID_NAME}
      PATHS ${UTILS_LIB_DIR} REQUIRED
      NO_DEFAULT_PATH)
  else()
    set(UTILS_CPUID
        ${UTILS_CPUID_LIB}
        PARENT_SCOPE)
  endif()

  if(USE_EXISTING_DA)
    if(DA_LIB STREQUAL "")
      find_library(
        DA name ${DA_NAME}
        PATHS ${DA_LIB_DIR} REQUIRED
        NO_DEFAULT_PATH)
    else()
      set(DA
          ${DA_LIB}
          PARENT_SCOPE)
    endif()
  endif()

  if(USE_LIBMEM)
    if(LIBMEM_LIB STREQUAL "")
      find_library(
        LIBMEM name ${LIBMEM_NAME}
        PATHS ${LIBMEM_LIB_DIR} REQUIRED
        NO_DEFAULT_PATH)
    else()
      set(LIBMEM
          ${LIBMEM_LIB}
          PARENT_SCOPE)
    endif()
  endif()

  include_directories(${LAPACK_INCLUDE_DIR})
  include_directories(${BLAS_INCLUDE_DIR})
  include_directories(${SPARSE_INCLUDE_DIR})
  include_directories(${UTILS_INCLUDE_DIR})

  set(BLAS_INCLUDE_DIR
      ${BLAS_INCLUDE_DIR}
      PARENT_SCOPE)
  set(LAPACK_INCLUDE_DIR
      ${LAPACK_INCLUDE_DIR}
      PARENT_SCOPE)
  set(SPARSE_INCLUDE_DIR
      ${SPARSE_INCLUDE_DIR}
      PARENT_SCOPE)
  set(UTILS_INCLUDE_DIR
      ${UTILS_INCLUDE_DIR}
      PARENT_SCOPE)

endfunction(linalg_libs)

# Reset all libraries
set(BLAS)
set(LAPACK)
set(SPARSE)
set(UTILS)
set(UTILS_CPUID)
set(DA)
set(LIBMEM)

linalg_libs()
