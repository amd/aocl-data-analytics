# Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

    set(ENV{LD_LIBRARY_PATH}
        "${SPARSE_DIR}:${LAPACK_DIR}:${BLAS_DIR}:${UTILS_DIR}:${CMAKE_INSTALL_PREFIX}/tmp:$ENV{LD_LIBRARY_PATH}"
    )
    message(NOTICE "LD_LIBRARY_PATH             $ENV{LD_LIBRARY_PATH}")

    # Repair the wheel using auditwheel

    execute_process(COMMAND auditwheel show ${file})

    # Use string(REGEX MATCH ...) to extract the manylinux tag
    string(REGEX MATCH "manylinux[^.]*" manylinux_platform ${file})

    # Display the extracted platform - we will repair the wheel with this target
    message(STATUS "Targeting the manylinux platform: ${manylinux_platform}")

    # Try to run auditwheel repair with the same target platform as the original
    # wheel - this is preferable particularly for container builds as it will
    # not change the platform tag
    execute_process(
      COMMAND
        auditwheel repair ${file} --plat ${manylinux_platform} --only-plat
        --wheel-dir ${CMAKE_INSTALL_PREFIX}/python_package --exclude "libc.so"
        --exclude "libc.so.*" --exclude "libgcc_s.so.*" --exclude
        "libstdc++.so.*" --exclude "libc.so.*" --exclude "librt.so.*" --exclude
        "libdl.so.*" --exclude "libpthread.so.*" --exclude "libm.so.*"
      RESULT_VARIABLE result)

    if(NOT result EQUAL 0)
      # If the above command fails, we will try to repair the wheel without
      # specifying the platform tag, which will allow auditwheel to determine
      # the appropriate platform tag based on the contents of the wheel - most
      # useful for dev builds without a container.
      message(WARNING "Audtwheel needs to change the platform tag.")
      execute_process(
        COMMAND
          auditwheel repair ${file} --wheel-dir
          ${CMAKE_INSTALL_PREFIX}/python_package --exclude "libc.so" --exclude
          "libc.so.*" --exclude "libgcc_s.so.*" --exclude "libstdc++.so.*"
          --exclude "libc.so.*" --exclude "librt.so.*" --exclude "libdl.so.*"
          --exclude "libpthread.so.*" --exclude "libm.so.*"
        RESULT_VARIABLE fallback_result)
      if(NOT fallback_result EQUAL 0)
        message(
          STATUS
            "Failed to repair the Python wheel. For local use, add ${CMAKE_INSTALL_PREFIX}/python_package to PYTHONPATH."
        )
      endif()
      # Remove the original unrepaired wheel
      file(REMOVE ${file})
    endif()

  endif()
endforeach()
