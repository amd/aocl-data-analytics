# Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

# Get a list of unrepaired wheel files using GLOB
file(GLOB wheel_files
  "${CMAKE_INSTALL_PREFIX}/python_package/unrepaired-wheels/*.whl")

foreach(file ${wheel_files})
  if(WIN32)
    execute_process(COMMAND delvewheel show ${file} --add-path
      ${CMAKE_INSTALL_PREFIX}/tmp)
    execute_process(
      COMMAND
      delvewheel repair ${file} --wheel-dir
      ${CMAKE_INSTALL_PREFIX}/python_package --add-path
      ${CMAKE_INSTALL_PREFIX}/tmp)
  else()

    get_filename_component(SPARSE_DIR ${SPARSE} DIRECTORY)
    get_filename_component(BLAS_DIR ${BLAS} DIRECTORY)
    get_filename_component(LAPACK_DIR ${LAPACK} DIRECTORY)
    get_filename_component(UTILS_DIR ${UTILS} DIRECTORY)
    get_filename_component(DLP_DIR ${DLP} DIRECTORY)

    set(ENV{LD_LIBRARY_PATH}
      "${SPARSE_DIR}:${LAPACK_DIR}:${BLAS_DIR}:${UTILS_DIR}:${DLP_DIR}:${CMAKE_INSTALL_PREFIX}/tmp:$ENV{LD_LIBRARY_PATH}"
    )
    message(NOTICE "LD_LIBRARY_PATH             $ENV{LD_LIBRARY_PATH}")

    # The input wheel is deliberately tagged linux_<arch>. Auditwheel determines
    # the compatible manylinux policy from the binaries unless explicitly set.
    message(STATUS "Python wheel policy: ${WHEEL_POLICY}")
    set(repaired_wheel_dir
      "${CMAKE_INSTALL_PREFIX}/python_package")
    set(auditwheel_platform_args "")
    if(WHEEL_POLICY STREQUAL "auto")
      list(APPEND auditwheel_platform_args --plat auto)
    elseif(WHEEL_POLICY STREQUAL "local")
      list(APPEND auditwheel_platform_args --no-update-tags)
    else()
      list(APPEND auditwheel_platform_args --plat ${WHEEL_POLICY} --only-plat)
    endif()
    execute_process(
      COMMAND
      "${Python3_EXECUTABLE}" -m auditwheel repair ${file}
      ${auditwheel_platform_args}
      --wheel-dir ${repaired_wheel_dir} --exclude "libc.so"
      --exclude "libc.so.*" --exclude "libgcc_s.so.*" --exclude
      "libstdc++.so.*" --exclude "libc.so.*" --exclude "librt.so.*" --exclude
      "libdl.so.*" --exclude "libpthread.so.*" --exclude "libm.so.*" --exclude "libmvec.so.*"
      RESULT_VARIABLE result)

    if(result EQUAL 0)
      file(GLOB repaired_wheels "${repaired_wheel_dir}/*.whl")
      foreach(repaired_wheel ${repaired_wheels})
        message(STATUS "Python wheel created: ${repaired_wheel}")
      endforeach()
    else()
      if(WHEEL_POLICY STREQUAL "auto" OR WHEEL_POLICY STREQUAL "local")
        message(
          WARNING
          "Failed to repair the Python wheel. The unrepaired linux wheel has been retained at ${file}."
        )
      else()
        message(FATAL_ERROR
          "Failed to repair the Python wheel for the requested ${WHEEL_POLICY} policy.")
      endif()
    endif()

  endif()
endforeach()
