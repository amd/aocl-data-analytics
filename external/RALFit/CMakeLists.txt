# Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

# If OpenMP is used in the future, please uncomment the following line
# to ensure that the Fortran OpenMP runtime library is found.
# find_package(OpenMP REQUIRED)

set(CMAKE_Fortran_PREPROCESS ON)

if(BUILD_ILP64)
  if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    list(APPEND CMAKE_Fortran_FLAGS -finteger-4-integer-8)
  elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
    list(APPEND CMAKE_Fortran_FLAGS -fdefault-integer-8)
  elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
      list(APPEND CMAKE_Fortran_FLAGS -i8)
  else()
    message(FATAL_ERROR "Unknown Fortran compiler!")
  endif()

  add_compile_definitions(BUILD_ILP64)
  message(NOTICE "RALFit: Requesting ILP64")
endif()

list(JOIN CMAKE_Fortran_FLAGS " " CMAKE_Fortran_FLAGS)

# Sets -fPIC flag
if(BUILD_SHARED_LIBS)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON) # SET fPIC
endif()

if(WIN32 AND NOT CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  find_library(
    LIBGFORTRAN
    NAMES libgfortran libgfortran-5
    HINTS ${MINGW_PATH}
    DOC "gfortran library")
endif()

add_subdirectory(libRALFit/src EXCLUDE_FROM_ALL)

message(NOTICE "RALFit: Fortran Compiler ID       : ${CMAKE_Fortran_COMPILER_ID}")

message(NOTICE "RALFit: Fortran Compiler flags    : ${CMAKE_Fortran_FLAGS}")

set(nlls_obj_list ${nlls_obj_list} PARENT_SCOPE)
