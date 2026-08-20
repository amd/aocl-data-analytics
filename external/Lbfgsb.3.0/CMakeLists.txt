# Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

# If OpenMP is used in the future, please uncomment the following line to ensure
# that the Fortran OpenMP runtime library is found. find_package(OpenMP
# REQUIRED)

set(CMAKE_Fortran_PREPROCESS ON)

if(NOT WIN32)
  # AOCC flang rejects -march= in Ninja's two-phase preprocessing step (-E);
  # GCC, flang-new and Make+AOCC (one-shot) all accept it.
  IsAOCCFlang(IS_AOCC_FLANG)
  if(NOT (CMAKE_GENERATOR MATCHES "Ninja" AND IS_AOCC_FLANG))
    AppendFortranFlags(_Fortran_FLAGS REQUIRED -march=x86-64-v3)
  endif()
  AppendFortranFlags(_Fortran_FLAGS 
    -Qunused-arguments
    -Wno-compare-reals
    -Wno-maybe-uninitialized
    -Wno-unused-dummy-argument
    -Wno-unused-parameter
    -Wno-unused-variable
  )
else()
  if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    set(_Fortran_FLAGS /QxCORE-AVX2)
  endif()
endif()

if(BUILD_ILP64)
  if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    list(APPEND _Fortran_FLAGS -finteger-4-integer-8)
  elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
    list(APPEND _Fortran_FLAGS -fdefault-integer-8)
  elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    list(APPEND _Fortran_FLAGS -i8)
  else()
    message(FATAL_ERROR "Unknown Fortran compiler!")
  endif()
endif()

list(JOIN _Fortran_FLAGS " " _Fortran_FLAGS)

set(LBFGSB_SRC lbfgsb_mod.F90 linpack.F90 timer.F90 lbfgsb.F90)

message(NOTICE
        "L-BFGS-B: Fortran Compiler ID     : ${CMAKE_Fortran_COMPILER_ID}")

message(NOTICE "L-BFGS-B: Fortran Compiler flags  : ${_Fortran_FLAGS}")

# Writing a function that will be used to generate the required object libraries
# for the required precisions.
function(generate_lbfgsb_targets precision)
  # Construct the name of the object library
  set(lbfgsb_prec "lbfgsb")
  if(${precision} STREQUAL "s")
    set(lbfgsb_prec "${lbfgsb_prec}_s")
    set(local_defs "-DSINGLE_PREC")
  endif()
  add_library(${lbfgsb_prec} OBJECT ${LBFGSB_SRC}
                                    ${precision}working_precision.F90)
  # Add macro SINGLE_PREC for single precision, it's empty string for double
  target_compile_definitions(${lbfgsb_prec} PRIVATE ${local_defs})
  separate_arguments(_compiler_flags UNIX_COMMAND "${_Fortran_FLAGS}")
  target_compile_options(${lbfgsb_prec} PRIVATE ${_compiler_flags})
  if(WIN32)
    if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
      target_link_directories(${lbfgsb_prec} PRIVATE "${IFORT_LIBS}")
    elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
      # Specific Windows AOCC requirement
      target_link_libraries(${lbfgsb_prec} PRIVATE ${FORTRAN_RUNTIME})
      include_directories("-I${CMAKE_CURRENT_SOURCE_DIR}")
      link_libraries("flang")
    else()
      set_target_properties(${lbfgsb_prec} PROPERTIES PREFIX "")
    endif()
  endif()
  target_link_libraries(${lbfgsb_prec} PRIVATE ${BLAS})
  target_include_directories(${lbfgsb_prec} PRIVATE ${BLAS_INCLUDE_DIR})
  # Sets -fPIC flag
  if(BUILD_SHARED_LIBS)
    set_target_properties(${lbfgsb_prec} PROPERTIES POSITION_INDEPENDENT_CODE
                                                    ON)
  endif()
endfunction()

# Generate single-precision lbfgsb_s library
generate_lbfgsb_targets("s")
# Generate double-precision lbfgsb library
generate_lbfgsb_targets("d")
