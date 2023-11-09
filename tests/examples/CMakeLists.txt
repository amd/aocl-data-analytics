# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

include_directories(${CMAKE_SOURCE_DIR}/source/include)

add_executable(basic_statistics basic_statistics.cpp)
target_link_libraries(basic_statistics aocl-da)

add_executable(linear_model linear_model.cpp)
target_link_libraries(linear_model aocl-da)

add_executable(reglinear_model reglinear_model.cpp)
target_link_libraries(reglinear_model aocl-da)

add_executable(linmod_diabetes linmod_diabetes.cpp)
target_link_libraries(linmod_diabetes aocl-da)
target_compile_definitions(linmod_diabetes
                           PUBLIC DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

add_executable(datastore datastore.cpp)
target_link_libraries(datastore aocl-da)
target_compile_definitions(datastore
                           PUBLIC DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

add_executable(decision_tree_ex decision_tree_ex.cpp)
target_link_libraries(decision_tree_ex aocl-da)
target_compile_definitions(decision_tree_ex
                           PUBLIC DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/")

add_executable(pca pca.cpp)
target_link_libraries(pca aocl-da)

set(EXAMPLE_EXES
    reglinear_model
    linear_model
    linmod_diabetes
    basic_statistics
    pca
    datastore
    decision_tree_ex)

# For Windows shared library builds copying some relevant DLLs into the examples
# bin folder enables ctest to run without needing to update PATH
if(WIN32 AND BUILD_SHARED_LIBS)
  foreach(EXAMPLE_EXE ${EXAMPLE_EXES})
    add_custom_command(
      TARGET ${EXAMPLE_EXE}
      POST_BUILD
      COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_RUNTIME_DLLS:${EXAMPLE_EXE}> $<TARGET_FILE_DIR:${EXAMPLE_EXE}>
      COMMAND_EXPAND_LISTS)
  endforeach()
endif()

# Add all example programs to CTEST
foreach(EX_NAME ${EXAMPLE_EXES})
  string(CONCAT TNAME ${EX_NAME} "_ex")
  add_test(${TNAME} ${EX_NAME})
endforeach()

# Add all executables to the coverage test
if(COVERAGE)
  foreach(EX_NAME ${EXAMPLE_EXES})
    list(APPEND COV_EXECUTABLES "${CMAKE_CURRENT_BINARY_DIR}/${EX_NAME}")
  endforeach()
endif()
