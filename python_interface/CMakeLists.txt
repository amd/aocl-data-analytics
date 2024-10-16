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

# Build Python bindings using pybind11

if(NOT BUILD_SHARED_LIBS)
  message(
    FATAL_ERROR "Shared libraries are required to build python interfaces.")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Note on Windows we need something like: set
# CMAKE_PREFIX_PATH=C:\path\to\python\site-packages\pybind11\share\cmake\pybind11;%CMAKE_PREFIX_PATH%

set(CMAKE_INSTALL_MODE COPY)
install(DIRECTORY python_package/aoclda DESTINATION python_package)
install(FILES python_package/setup.py DESTINATION python_package)

if(NOT USE_EXISTING_DA)
  # Install aocl-da libraries in a temporary folder
  install(
    TARGETS aocl-da
    ARCHIVE DESTINATION tmp
    LIBRARY DESTINATION tmp
    RUNTIME DESTINATION tmp)
endif()

if(NOT WIN32)
  find_program(PATCHELF name patchelf REQUIRED)

  if(USE_EXISTING_DA)
    # Copy libaocl-da.so into a temporary folder so we can remove the libompstub
    # dependency
    install(FILES ${DA} DESTINATION tmp)
  endif()

  install(
    CODE "execute_process(COMMAND ${PATCHELF} --set-rpath \$ORIGIN ${CMAKE_INSTALL_PREFIX}/tmp/libaocl-da.so)"
  )

  # Remove libompstub from dependencies
  install(
    CODE "execute_process(COMMAND ${PATCHELF} --remove-needed libompstub.so ${CMAKE_INSTALL_PREFIX}/tmp/libaocl-da.so)"
  )
endif()

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "from wheel.bdist_wheel import bdist_wheel"
  RESULT_VARIABLE EXIT_CODE
  OUTPUT_QUIET)
if(NOT ${EXIT_CODE} EQUAL 0)
  message(
    FATAL_ERROR
      "Python3 package wheel is required to build the Python bindings\n   Use: ${Python3_EXECUTABLE} -m pip install \"wheel\""
  )
endif()

# Build the Python bindings
add_subdirectory(python_package)

# Create the wheel after installation
install(
  CODE "execute_process(COMMAND ${Python3_EXECUTABLE} -m pip wheel ${CMAKE_INSTALL_PREFIX}/python_package/ --no-deps --wheel-dir ${CMAKE_INSTALL_PREFIX}/python_package)"
)

install(CODE "set(BLAS \"${BLAS}\")")
install(CODE "set(LAPACK \"${LAPACK}\")")
install(CODE "set(SPARSE \"${SPARSE}\")")
install(CODE "set(UTILS \"${UTILS}\")")

install(CODE "include(\"${CMAKE_CURRENT_SOURCE_DIR}/repairwheel.cmake\")")
