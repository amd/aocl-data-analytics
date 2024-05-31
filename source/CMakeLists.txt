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
set(DA_CLUSTERING core/clustering/kmeans_public.cpp)
set(DA_DECISION_FOREST core/decision_forest/decision_tree_public.cpp
                       core/decision_forest/random_forest_public.cpp)
set(NLLS_SRC core/nlls/nlls_public.cpp)

set(CSV core/csv/tokenizer.c core/csv/read_csv_public.cpp)
set(DA_HANDLE core/utilities/da_handle.cpp core/utilities/da_handle_public.cpp)
set(DA_OPTIONS core/utilities/options_public.cpp)
set(DA_DATA
    core/data_management/interval_set.cpp
    core/data_management/data_store_public.cpp
    core/data_management/data_store.cpp)
set(DA_MISC core/utilities/miscellaneous.cpp)
set(DA_METRICS core/metrics/pairwise_distances_public.cpp)

set(DA_CORE_SRC
    ${LINMOD_SRC}
    ${BASIC_STATISTICS_SRC}
    ${CSV}
    ${DA_HANDLE}
    ${DA_OPTIONS}
    ${DA_DATA}
    ${DA_FACTORIZATION}
    ${DA_CLUSTERING}
    ${DA_DECISION_FOREST}
    ${NLLS_SRC}
    ${DA_MISC}
    ${DA_METRICS})

add_library(da_core OBJECT ${DA_CORE_SRC})
target_include_directories(
  da_core
  PRIVATE include
          core/drivers
          core/basic_statistics
          core/optimization
          core/utilities
          core/csv
          core/metrics
          core/linear_model
          core/data_management
          core/factorization
          core/clustering
          core/decision_forest
          core/nlls
          ${CMAKE_SOURCE_DIR}/external/Lbfgsb.3.0
          ${CMAKE_SOURCE_DIR}/external/RALFit
          ${CMAKE_SOURCE_DIR}/external/RALFit/libRALFit/include
          )

# Add definitions that depend on the CMake configuration.
target_compile_definitions(da_core PRIVATE ${AOCLDA_DEFS})
# Define -DAOCL_ILP64 when ILP64 libraries are built. For LP64, this is an empty
# string.
target_compile_definitions(da_core PRIVATE ${AOCLDA_ILP64})

# List of targets that should use -O2 for release/benchmark builds
set(USE_O2 "")

# By default release builds have -O3, here we add -O2 which will override it for
# the source files in the USE_O2 list, since it will appear after the -O3
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND CMAKE_CXX_COMPILER_ID STREQUAL
                                           "Clang")
  foreach(SOURCE ${DA_CORE_SRC})
    if(SOURCE IN_LIST USE_O2)
      set_source_files_properties(${SOURCE} PROPERTIES COMPILE_FLAGS "-O2")
    endif()
  endforeach()
endif()

# Sets -fPIC flag
if(BUILD_SHARED_LIBS)
  set_target_properties(da_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(WIN32 AND CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
  link_libraries("flang")
endif()

if(NOT WIN32)
  include(Suppressions_core)
endif()

# Create combined library from da_core and Fortran libraries
add_library(aocl-da $<TARGET_OBJECTS:da_core> $<TARGET_OBJECTS:lbfgsb>
                    $<TARGET_OBJECTS:lbfgsb_s> $<TARGET_OBJECTS:ral_nlls_object>)
target_link_libraries(aocl-da PRIVATE ${SPARSE} ${LAPACK} ${BLAS} ${UTILS}
                                      ${FORTRAN_RUNTIME})

target_link_directories(aocl-da PRIVATE ${IFORT_LIBS})
target_link_directories(aocl-da PRIVATE ${FLANG_LIBS})
target_include_directories(
  aocl-da PRIVATE ${BLAS_INCLUDE_DIR} ${SPARSE_INCLUDE_DIR}
                  ${LAPACK_INCLUDE_DIR})
# Sets -fPIC flag
if(BUILD_SHARED_LIBS)
  set_target_properties(aocl-da PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

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
