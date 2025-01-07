# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

# Get a list of wheel files using GLOB
file(GLOB wheel_files "${CMAKE_INSTALL_PREFIX}/python_package/*.whl")

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
    execute_process(
      # Future auditwheel versions are likely to permit wildcards - for now we
      # also explicitly list the excluded libraries
      COMMAND
        auditwheel repair ${file} --plat linux_x86_64 --wheel-dir
        ${CMAKE_INSTALL_PREFIX}/python_package --exclude "libc.so" --exclude
        "libc.so.6" --exclude "libgcc_s.so.1" --exclude "libstdc++.so.6"
        --exclude "libstdc++.so.8" --exclude "libgcc_s.so.*" --exclude
        "libc.so.*" --exclude "librt.so.*" --exclude "librt.so.1" --exclude
        "libdl.so.*" --exclude "libdl.so.2" --exclude
        "libpthread.so.*" --exclude "libpthread.so.0" --exclude
        "libm.so.*" --exclude "libm.so.6")

  endif()
endforeach()
