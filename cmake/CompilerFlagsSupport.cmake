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
#

if (BUILD_FORTRAN)
  include(CheckFortranCompilerFlag)
endif()
include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)
# Non-documented option
option(DEBUG_FLAGS_LOGIC "Show per-flag compiler-support test output from the Append*Flags helpers" OFF)

function(AppendFortranFlags OUT_VAR)
  if (NOT BUILD_FORTRAN)
    # Fortran disabled: don't probe the (possibly absent) Fortran compiler,
    # just return an empty result.
    set(${OUT_VAR} "" PARENT_SCOPE)
    return()
  endif()
  if(NOT DEBUG_FLAGS_LOGIC)
    set(CMAKE_REQUIRED_QUIET TRUE)
  endif()
  set(options REQUIRED)
  cmake_parse_arguments(ASF "${options}" "" "" ${ARGN})

  set(_flags ${${OUT_VAR}})
  set(_flag_list "")
  foreach(arg IN LISTS ASF_UNPARSED_ARGUMENTS)
    separate_arguments(_split UNIX_COMMAND "${arg}")
    list(APPEND _flag_list ${_split})
  endforeach()
  foreach(flag IN LISTS _flag_list)
    string(MAKE_C_IDENTIFIER "FORTRAN_FLAG_SUPPORTED_${flag}" _cache_var)
    # GCC validates a negative warning option (-Wno-foo) lazily: an unknown
    # -Wno-foo is silently accepted until another diagnostic fires. Probe the
    # positive form (-Wfoo), which is validated eagerly, but still add the
    # original -Wno-foo flag on success.
    if(flag MATCHES "^-Wno-(.+)$")
      set(_probe "-W${CMAKE_MATCH_1}")
    else()
      set(_probe "${flag}")
    endif()
    set(_saved_link_options "${CMAKE_REQUIRED_LINK_OPTIONS}")
    list(APPEND CMAKE_REQUIRED_LINK_OPTIONS "${flag}")
    check_fortran_compiler_flag("${_probe}" ${_cache_var})
    set(CMAKE_REQUIRED_LINK_OPTIONS "${_saved_link_options}")
    if(${_cache_var})
      list(APPEND _flags "${flag}")
    elseif(ASF_REQUIRED)
      message(
        FATAL_ERROR
          "Fortran compiler '${CMAKE_Fortran_COMPILER_ID}' does not support required flag: ${flag}"
      )
    endif()
  endforeach()

  set(${OUT_VAR}
      ${_flags}
      PARENT_SCOPE)
endfunction()

function(AppendCXXFlags OUT_VAR)
  if(NOT DEBUG_FLAGS_LOGIC)
    set(CMAKE_REQUIRED_QUIET TRUE)
  endif()
  set(options REQUIRED)
  cmake_parse_arguments(ACXX "${options}" "" "" ${ARGN})

  set(_flags ${${OUT_VAR}})
  set(_flag_list "")
  foreach(arg IN LISTS ACXX_UNPARSED_ARGUMENTS)
    separate_arguments(_split UNIX_COMMAND "${arg}")
    list(APPEND _flag_list ${_split})
  endforeach()
  foreach(flag IN LISTS _flag_list)
    string(MAKE_C_IDENTIFIER "CXX_FLAG_SUPPORTED_${flag}" _cache_var)
    # check_cxx_compiler_flag links a test executable. Some flags (e.g.
    # -fsanitize=address) must also be present on the link line, otherwise the
    # test link fails with undefined symbols and the flag is wrongly reported
    # as unsupported. Pass the flag as a link option for the duration of the
    # check so such flags are detected correctly.
    # GCC validates a negative warning option (-Wno-foo) lazily: an unknown
    # -Wno-foo is silently accepted until another diagnostic fires. Probe the
    # positive form (-Wfoo), which is validated eagerly, but still add the
    # original -Wno-foo flag on success.
    if(flag MATCHES "^-Wno-(.+)$")
      set(_probe "-W${CMAKE_MATCH_1}")
    else()
      set(_probe "${flag}")
    endif()
    set(_saved_link_options "${CMAKE_REQUIRED_LINK_OPTIONS}")
    list(APPEND CMAKE_REQUIRED_LINK_OPTIONS "${flag}")
    check_cxx_compiler_flag("${_probe}" ${_cache_var})
    set(CMAKE_REQUIRED_LINK_OPTIONS "${_saved_link_options}")
    if(${_cache_var})
      list(APPEND _flags "${flag}")
    elseif(ACXX_REQUIRED)
      message(
        FATAL_ERROR
          "C++ compiler '${CMAKE_CXX_COMPILER_ID}' does not support required flag: ${flag}"
      )
    endif()
  endforeach()

  # Return a space-separated string rather than a CMake list. Embedding a
  # semicolon-separated list into a flag string would otherwise survive as a
  # single quoted token (e.g. "-Wfoo;-Wbar") on the compiler command line.
  list(JOIN _flags " " _flags)
  set(${OUT_VAR}
      "${_flags}"
      PARENT_SCOPE)
endfunction()

function(AppendCFlags OUT_VAR)
  if(NOT DEBUG_FLAGS_LOGIC)
    set(CMAKE_REQUIRED_QUIET TRUE)
  endif()
  set(options REQUIRED)
  cmake_parse_arguments(AC "${options}" "" "" ${ARGN})

  set(_flags ${${OUT_VAR}})
  set(_flag_list "")
  foreach(arg IN LISTS AC_UNPARSED_ARGUMENTS)
    separate_arguments(_split UNIX_COMMAND "${arg}")
    list(APPEND _flag_list ${_split})
  endforeach()
  foreach(flag IN LISTS _flag_list)
    string(MAKE_C_IDENTIFIER "C_FLAG_SUPPORTED_${flag}" _cache_var)
    # check_c_compiler_flag links a test executable. Some flags (e.g.
    # -fsanitize=address) must also be present on the link line, otherwise the
    # test link fails with undefined symbols and the flag is wrongly reported
    # as unsupported. Pass the flag as a link option for the duration of the
    # check so such flags are detected correctly.
    # GCC validates a negative warning option (-Wno-foo) lazily: an unknown
    # -Wno-foo is silently accepted until another diagnostic fires. Probe the
    # positive form (-Wfoo), which is validated eagerly, but still add the
    # original -Wno-foo flag on success.
    if(flag MATCHES "^-Wno-(.+)$")
      set(_probe "-W${CMAKE_MATCH_1}")
    else()
      set(_probe "${flag}")
    endif()
    set(_saved_link_options "${CMAKE_REQUIRED_LINK_OPTIONS}")
    list(APPEND CMAKE_REQUIRED_LINK_OPTIONS "${flag}")
    check_c_compiler_flag("${_probe}" ${_cache_var})
    set(CMAKE_REQUIRED_LINK_OPTIONS "${_saved_link_options}")
    if(${_cache_var})
      list(APPEND _flags "${flag}")
    elseif(AC_REQUIRED)
      message(
        FATAL_ERROR
          "C compiler '${CMAKE_C_COMPILER_ID}' does not support required flag: ${flag}"
      )
    endif()
  endforeach()

  # Return a space-separated string rather than a CMake list. Embedding a
  # semicolon-separated list into a flag string would otherwise survive as a
  # single quoted token (e.g. "-Wfoo;-Wbar") on the compiler command line.
  list(JOIN _flags " " _flags)
  set(${OUT_VAR}
      "${_flags}"
      PARENT_SCOPE)
endfunction()

function(GetCompilerFamily OUT_VAR)
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    cmake_path(GET CMAKE_CXX_COMPILER FILENAME COMPILER_BASENAME)
    if(COMPILER_BASENAME MATCHES "^(.+-)?(clang-cl)(-[0-9]+(\\.[0-9]+)*)?(-[^.]+)?(\\.exe)?$")
      set(COMPILER_FAMILY "LLVM-CL")
    else()
      set(COMPILER_FAMILY "LLVM")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(COMPILER_FAMILY "GNU")
else()
    message(FATAL_ERROR "Unknown compiler ${CMAKE_CXX_COMPILER_ID}!")
endif()
set(${OUT_VAR} ${COMPILER_FAMILY} PARENT_SCOPE)
endfunction()

# Sets OUT_VAR to TRUE if the current CXX compiler is AMD's AOCC clang,
# Detection relies on the presence of "AOCC" or "AMD clang" in `clang -v`
function(IsAOCC OUT_VAR)
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -v
    OUTPUT_VARIABLE _ver_stdout
    ERROR_VARIABLE  _ver_stderr
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  # clang -v writes to stderr; capture both to be safe
  if("${_ver_stdout}\n${_ver_stderr}" MATCHES "AOCC|AMD clang")
    set(${OUT_VAR} TRUE  PARENT_SCOPE)
  else()
    set(${OUT_VAR} FALSE PARENT_SCOPE)
  endif()
endfunction()

# Sets OUT_VAR to TRUE if the current Fortran compiler is AMD's AOCC flang,
function(IsAOCCFlang OUT_VAR)
  execute_process(
    COMMAND "${CMAKE_Fortran_COMPILER}" -v
    OUTPUT_VARIABLE _ver_stdout
    ERROR_VARIABLE  _ver_stderr
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  # flang-new -v writes to stderr; capture both to be safe
  if("${_ver_stdout}\n${_ver_stderr}" MATCHES "AOCC|AMD AOCC aof")
    set(${OUT_VAR} TRUE  PARENT_SCOPE)
  else()
    set(${OUT_VAR} FALSE PARENT_SCOPE)
  endif()
endfunction()

function (PrettyPrintFlags _ccomp _flags)
  foreach(_var _flags)
    string(REGEX REPLACE "[; \t\r\n]+" " " ${_var} "${${_var}}")
  endforeach()
  message( NOTICE " + ${_ccomp} Compiler flags: ${_flags}")
endfunction()