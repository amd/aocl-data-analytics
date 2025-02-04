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

include(CheckCXXCompilerFlag)

function(extract_znver_generation ARCHITECTURE ZNVER)
  # Given a -march=znverX type flag, extract the zen generation number
  if(${ARCHITECTURE} MATCHES "^znver([0-9]+)$")
    set(ZNVER_TEMP ${CMAKE_MATCH_1})
  endif()
  set(${ZNVER}
      ${ZNVER_TEMP}
      PARENT_SCOPE)
endfunction()

function(supported_architectures ARCHITECTURES DEFINITIONS)
  # Return a list of architectures associated compile definitions for
  # compilation with dynamic dispatch

  set(ARCH_TEMP generic)
  set(DEF_TEMP generic_AVAILABLE)

  set(CANDIDATE_ARCHS znver2 znver3 znver4 znver5)

  foreach(arch IN LISTS CANDIDATE_ARCHS)
    check_cxx_compiler_flag("-march=${arch}" SUPPORTED_${arch})
    if(SUPPORTED_${arch})
      list(APPEND ARCH_TEMP ${arch})
      # extract_znver_generation(${arch} ZNVER_MAX)
      if(${arch} MATCHES "^znver([0-9]+)$")
        set(number ${CMAKE_MATCH_1})
        set(ZNVER_MAX ${number})
      endif()
      list(APPEND DEF_TEMP ${arch}_AVAILABLE=${number})
    endif()
  endforeach()

  if(ZNVER_MAX)
    list(APPEND DEF_TEMP ZNVER_MAX=zen${ZNVER_MAX})
  else()
    message(FATAL_ERROR "Unable to compile for any Zen architectures.")
  endif()

  message(NOTICE "\nSupported architectures: ${ARCH_TEMP}\n")
  set(ARCHITECTURES
      ${ARCH_TEMP}
      PARENT_SCOPE)
  set(DEFINITIONS
      ${DEF_TEMP}
      PARENT_SCOPE)

endfunction(supported_architectures)
