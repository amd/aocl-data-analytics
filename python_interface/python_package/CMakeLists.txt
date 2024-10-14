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

# Python bindings
cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(py_aoclda VERSION 5.0.1)

set(BUILD_SHARED_LIBS ON)
option(BUILD_ILP64 "ILP64 Support" OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
# On linux, pybind11_DIR needs to be defined. pybind11-config gives the correct
# value for the system install of pybind11. Also serves as a check that pybind11
# is correctly installed
if(NOT WIN32)
  execute_process(
    COMMAND pybind11-config --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    RESULT_VARIABLE EXIT_CODE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT ${EXIT_CODE} EQUAL 0)
    message(
      FATAL_ERROR
        "pybind11 could not be configured (pybind11-config --cmakedir did not finish correctly)\n"
        "Use: ${Python3_EXECUTABLE} -m pip install pybind11")
  endif()
endif()
find_package(pybind11 CONFIG REQUIRED)
execute_process(COMMAND "${Python3_EXECUTABLE}-config --extension-suffix")

if(NOT BUILD_ILP64)
  set(INT_LIB "LP64")
else()
  set(INT_LIB "ILP64")
  add_definitions(-DAOCLDA_ILP64)
endif()
# Create the pybind11 modules
pybind11_add_module(_aoclda bindings/aoclda_pywrappers.cpp
                    bindings/utilities_py.cpp)
# Do not add any new directories here. This is to keep the python
# wrapper separate from the main library
target_include_directories(_aoclda PRIVATE ${CMAKE_SOURCE_DIR}/source/include)
target_link_libraries(_aoclda PRIVATE aocl-da)
if(BUILD_SMP OR UNIX)
  target_link_libraries(_aoclda PRIVATE OpenMP::OpenMP_CXX)
endif()
install(
  TARGETS _aoclda
  ARCHIVE DESTINATION python_package/aoclda
  LIBRARY DESTINATION python_package/aoclda
  RUNTIME DESTINATION python_package/aoclda)

if(NOT WIN32)
  install(
    CODE "execute_process(COMMAND patchelf --set-rpath \$ORIGIN ${CMAKE_INSTALL_PREFIX}/python_package/aoclda/$<TARGET_FILE_NAME:_aoclda>)"
  )
endif()
