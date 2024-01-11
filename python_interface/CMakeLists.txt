# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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


# Build Python bindings using pybind11

if(NOT BUILD_SHARED_LIBS)
  message(
    FATAL_ERROR "Shared libraries are required to build python interfaces.")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Note on Windows we need something like:
# set CMAKE_PREFIX_PATH=C:\Users\ehopkins\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\pybind11\share\cmake\pybind11;%CMAKE_PREFIX_PATH%

# Install the full python package as well as all the dependent libraries in python_package/aoclda
set(BASE_DEP_LIBS ${BLAS} ${LAPACK} ${UTILS})
set(DEP_LIBS "")

if (WIN32)
  # create the list of all the libraries and their dll versions
  foreach(LIB ${BASE_DEP_LIBS})
    set(DLL ${LIB})
    cmake_path(REPLACE_EXTENSION DLL ".dll")
    list(APPEND DEP_LIBS ${LIB} ${DLL})
  endforeach()
else()
  # create the list of all the libraries and their symlink versions
  foreach(LIB ${BASE_DEP_LIBS})
    file(GLOB LIB_ALL_SYM "${LIB}*")
    list(APPEND DEP_LIBS ${LIB_ALL_SYM})
  endforeach()
endif()

set(CMAKE_INSTALL_MODE COPY)
install(DIRECTORY python_package/aoclda DESTINATION python_package)
install(DIRECTORY python_package/py_examples DESTINATION python_package)
install(FILES python_package/setup.py DESTINATION python_package)
install(FILES ${DEP_LIBS}
        DESTINATION python_package/aoclda)
install(
  TARGETS aocl-da
  ARCHIVE DESTINATION python_package/aoclda
  LIBRARY DESTINATION python_package/aoclda
  RUNTIME DESTINATION python_package/aoclda)
# TODO Add fortran runtimes?

if (NOT WIN32)
  # Change the runpath of aocl-da to $ORIGIN, making it look for its dependencies in its own directory
  # $ORIGIN is evaluated at runtime by the interpreter
  install(CODE "execute_process(COMMAND patchelf --set-rpath \$ORIGIN install/python_package/aoclda/libaocl-da.so)")
endif()

# Build the Python bindings
add_subdirectory(python_package)

# create the wheel after instalation
install(CODE "execute_process(COMMAND ${Python3_EXECUTABLE} -m pip wheel install/python_package/ --no-deps --wheel-dir install/python_package)")