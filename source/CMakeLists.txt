# Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

set(LINMOD_SRC core/linear_model/linmod_public.cpp)
set(BASIC_STATISTICS_SRC core/basic_statistics/basic_statistics_public.cpp)
set(DA_FACTORIZATION core/factorization/pca_public.cpp)
set(DA_DECISION_FOREST core/decision_forest/decision_forest_public_d.cpp
                       core/decision_forest/decision_forest_public_s.cpp)

# set(DRIVERS core/drivers/lbfgsb_driver.cpp)
set(CSV core/csv/tokenizer.c core/csv/read_csv_public.cpp)
set(DA_HANDLE core/utilities/da_handle.cpp core/utilities/da_handle_public.cpp)
set(DA_OPTIONS core/utilities/options_public.cpp)
set(DA_DATA
    core/data_management/interval_map.cpp
    core/data_management/data_store_public.cpp
    core/data_management/data_store.cpp)
set(DA_MISC core/utilities/miscellaneous.cpp)

set(DA_CORE_SRC
    ${LINMOD_SRC}
    ${BASIC_STATISTICS_SRC}
    ${CSV}
    ${DA_HANDLE}
    ${DA_OPTIONS}
    ${DA_DATA}
    ${DA_FACTORIZATION}
    ${DA_DECISION_FOREST}
    ${DA_MISC})

add_library(da_core OBJECT ${DA_CORE_SRC})
target_include_directories(da_core PUBLIC include)
target_include_directories(
  da_core
  PUBLIC core/drivers/
         core/optimization/
         core/utilities
         core/csv
         core/linear_model
         core/data_management
         core/factorization
         core/decision_forest)
target_include_directories(da_core
  PRIVATE ${CMAKE_SOURCE_DIR}/external/Lbfgsb.3.0/)
# Define the macro that sets the version string
target_compile_definitions(da_core PRIVATE ${VERS_DEF})

if(WIN32)
  target_link_directories(
    da_core PUBLIC
    "${CMAKE_BINARY_DIR}/external/Lbfgsb.3.0/double_cmake/${CMAKE_BUILD_TYPE}"
    "${CMAKE_BINARY_DIR}/external/Lbfgsb.3.0/single_cmake/${CMAKE_BUILD_TYPE}")
endif()

target_link_libraries(da_core lbfgsb lbfgsb_s ${LAPACK} ${BLAS})

if(NOT WIN32)
  include(Suppressions_core)
endif()

# Create combined library from da_core and Fortran libraries
add_library(aocl-da $<TARGET_OBJECTS:da_core> $<TARGET_OBJECTS:lbfgsb>
                    $<TARGET_OBJECTS:lbfgsb_s>)
target_link_libraries(aocl-da PUBLIC da_core lbfgsb lbfgsb_s ${LAPACK} ${BLAS}
                                     ${UTILS})

# Installation commands
install(
  TARGETS aocl-da
  ARCHIVE DESTINATION lib/${INT_LIB}
  LIBRARY DESTINATION lib/${INT_LIB}
  RUNTIME DESTINATION bin/${INT_LIB})

install(
  DIRECTORY include/
  DESTINATION include/${INT_LIB}
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp")
