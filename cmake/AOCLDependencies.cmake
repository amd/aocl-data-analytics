# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
  if(NOT DEFINED ENV{AOCL_ROOT})
    if((DEFINED CMAKE_AOCL_ROOT))
      set(ENV{AOCL_ROOT} ${CMAKE_AOCL_ROOT})
    else()
      if((NOT DEFINED BLAS_LIB)
         OR (NOT DEFINED LAPACK_LIB)
         OR (NOT DEFINED UTILS_LIB)
         OR (NOT DEFINED BLAS_INCLUDE_DIR)
         OR (NOT DEFINED LAPACK_INCLUDE_DIR))
        message(
          FATAL_ERROR
            "Environment variable \$AOCL_ROOT not found.\n - Perhaps source AOCL config script, or\n - Define \$AOCL_ROOT to point to AOCL install dir, and try again, or set -DCMAKE_AOCL_ROOT.\n - Alternative, specify BLAS_LIB/LAPACK_LIB/UTILS_LIB and BLAS/LAPACK_INCLUDE_DIR."
        )
      endif()
      set(BLAS ${BLAS_LIB} PARENT_SCOPE)
      set(LAPACK ${LAPACK_LIB} PARENT_SCOPE)
      set(UTILS ${UTILS_LIB} PARENT_SCOPE)
      include_directories(${LAPACK_INCLUDE_DIR})
      include_directories(${BLAS_INCLUDE_DIR})
      return()
    endif()
  endif()

  if(WIN32)
    set(CMAKE_FIND_LIBRARY_PREFIXES "")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
    if(BUILD_SMP)
      set(BLAS_NAME "AOCL-LibBlis-Win-MT-dll")
      set(LAPACK_NAME "AOCL-LibFlame-Win-MT-dll")
    else()
      set(BLAS_NAME "AOCL-LibBlis-Win-dll")
      set(LAPACK_NAME "AOCL-LibFlame-Win-dll")
    endif()
    set(UTILS_NAME "libaoclutils")

    set(BLAS_PATH "$ENV{AOCL_ROOT}/amd-blis/lib/${INT_LIB}")
    set(LAPACK_PATH "$ENV{AOCL_ROOT}/amd-libflame/lib/${INT_LIB}")
    set(UTILS_PATH "$ENV{AOCL_ROOT}/amd-utils/lib")

    include_directories("$ENV{AOCL_ROOT}/amd-blis/include/${INT_LIB}")
    include_directories("$ENV{AOCL_ROOT}/amd-libflame/include/${INT_LIB}")

  else(WIN32) # linux

    include_directories($ENV{AOCL_ROOT}/include_${INT_LIB})

    set(BLAS_PATH $ENV{AOCL_ROOT}/lib_${INT_LIB})
    set(LAPACK_PATH $ENV{AOCL_ROOT}/lib_${INT_LIB})
    set(UTILS_PATH $ENV{AOCL_ROOT}/lib_${INT_LIB})

    if(NOT BUILD_SHARED_LIBS)
      set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
    endif()

    if(BUILD_SMP)
      set(BLAS_NAME "blis-mt")
    else()
      set(BLAS_NAME "blis")
    endif()

    set(LAPACK_NAME "flame")
    set(UTILS_NAME "aoclutils")
  endif()

  find_library(
    BLAS name ${BLAS_NAME}
    PATHS ${BLAS_PATH} REQUIRED
    NO_DEFAULT_PATH)

  find_library(
    LAPACK name ${LAPACK_NAME}
    PATHS ${LAPACK_PATH} REQUIRED
    NO_DEFAULT_PATH)

  find_library(UTILS name ${UTILS_NAME} PATHS ${UTILS_PATH})
endfunction(linalg_libs)

# reset all libs
set(BLAS)
set(LAPACK)
set(UTILS)

linalg_libs()
