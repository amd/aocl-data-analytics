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

# Install the full python package as well as all the dependent libraries in
# python_package/aoclda. The BLAS_SERIAL and LAPACK_SERIAL variables are only
# set in Windows SMP builds to allow aoclsparse to have its dependent libraries
set(BASE_DEP_LIBS
    ${BLAS}
    ${LAPACK}
    ${BLAS_SERIAL}
    ${LAPACK_SERIAL}
    ${SPARSE}
    ${UTILS}
    ${UTILS_CPUID}
    ${DA}) # Only used if we are building with a pre-existing build of aocl-da
set(DEP_LIBS "")

if(WIN32)
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
install(FILES python_package/setup.py DESTINATION python_package)
install(
  FILES ${DEP_LIBS}
  DESTINATION python_package/aoclda
  OPTIONAL)
# Optional accounts for lack of a dll in case that static utils was linked)

if(NOT USE_EXISTING_DA)
  install(
    TARGETS aocl-da
    ARCHIVE DESTINATION python_package/aoclda
    LIBRARY DESTINATION python_package/aoclda
    RUNTIME DESTINATION python_package/aoclda)
endif()

if (WIN32 AND NOT CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
    install(FILES "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin/libomp.dll" DESTINATION python_package/aoclda)
endif()

if(CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
  if(WIN32)
    # Install Fortran runtimes
    set(RUNTIME_LIBS flang.lib flangrti.lib libomp.lib ompstub.lib
                     libpgmath.lib)
    set(RUNTIME_DLLS flang.dll flangrti.dll libomp.dll ompstub.dll pgmath.dll)
    foreach(LIB ${RUNTIME_LIBS})
      install(FILES "$ENV{FLANG_COMPILER}/lib/${LIB}"
              DESTINATION python_package/aoclda)
    endforeach()
    foreach(DLL ${RUNTIME_DLLS})
      install(FILES "$ENV{FLANG_COMPILER}/bin/${DLL}"
              DESTINATION python_package/aoclda)
    endforeach()
  else()
    # Find where the Flang runtime folder is
    string(REPLACE ":" ";" LD_LIBRARY_PATH_LIST $ENV{LD_LIBRARY_PATH})
    find_path(
      FLANG_RUNTIME_DIR
      NAMES libflang.so
      HINTS ${LD_LIBRARY_PATH_LIST})
    if(NOT FLANG_RUNTIME_DIR)
      message(FATAL_ERROR "Flang runtimes not found")
    endif()
    # Lists of required runtimes and symlinks
    set(RUNTIME_LIBS libflang.so libflangrti.so libpgmath.so libomp.so)
    set(SYMLINK_LIBS libamdlibm.so libamdalloc.so)
    # Install the runtimes
    foreach(LIB ${RUNTIME_LIBS})
      install(FILES "${FLANG_RUNTIME_DIR}/${LIB}"
              DESTINATION python_package/aoclda)
    endforeach()
    # Recreate the symlinks
    foreach(LIB ${SYMLINK_LIBS})
      get_filename_component(LIB_TARGET "${FLANG_RUNTIME_DIR}/${LIB}" REALPATH)
      get_filename_component(LIB_TARGET_NAME ${LIB_TARGET} NAME)
      install(FILES ${LIB_TARGET} DESTINATION python_package/aoclda)
      install(
        CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_INSTALL_PREFIX}/python_package/aoclda/${LIB_TARGET_NAME} ${CMAKE_INSTALL_PREFIX}/python_package/aoclda/${LIB})"
      )
    endforeach()

  endif()
endif()

if(NOT WIN32)
  find_program(PATCHELF name patchelf REQUIRED)
  # Change the runpath of aocl-da to $ORIGIN, making it look for its
  # dependencies in its own directory $ORIGIN is evaluated at runtime by the
  # interpreter
  install(
    CODE "execute_process(COMMAND ${PATCHELF} --set-rpath \$ORIGIN ${CMAKE_INSTALL_PREFIX}/python_package/aoclda/libaocl-da.so)"
  )
  # Remove libompstub from dependencies
  install(
    CODE "execute_process(COMMAND ${PATCHELF} --remove-needed libompstub.so ${CMAKE_INSTALL_PREFIX}/python_package/aoclda/libaocl-da.so)"
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

install(CODE "include(\"${CMAKE_CURRENT_SOURCE_DIR}/repairwheel.cmake\")")
