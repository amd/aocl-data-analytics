# Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

include(supported_architectures)

function(aoclda_add_dependencies OBJ_NAME)
  target_include_directories(
    ${OBJ_NAME}
    PRIVATE include
            core/drivers
            core/basic_statistics
            core/optimization
            core/utilities
            core/dynamic_dispatch
            core/csv
            core/metrics
            core/linear_model
            core/data_management
            core/factorization
            core/clustering
            core/decision_forest
            core/nlls
            core/nearest_neighbors
            core/svm
            core/kernel_functions
            ${CMAKE_SOURCE_DIR}/external/Lbfgsb.3.0
            ${CMAKE_SOURCE_DIR}/external/RALFit
            ${CMAKE_SOURCE_DIR}/external/RALFit/libRALFit/include
            ${Boost_INCLUDE_DIRS}
            ${UTILS_INCLUDE_DIR})

  # Add definitions that depend on the CMake configuration.
  target_compile_definitions(${OBJ_NAME} PRIVATE ${AOCLDA_DEFS})
  # Define -DAOCL_ILP64 when ILP64 libraries are built. For LP64, this is an
  # empty string.
  target_compile_definitions(${OBJ_NAME} PRIVATE ${AOCLDA_ILP64})
  # Sets -fPIC flag
  if(BUILD_SHARED_LIBS)
    set_target_properties(${OBJ_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endif()
endfunction()

# Public APIs requiring generic compilation
set(DA_LINMOD_PUBLIC core/linear_model/linmod_public.cpp)
set(DA_BASIC_STATISTICS_PUBLIC
    core/basic_statistics/basic_statistics_public.cpp)
set(DA_FACTORIZATION_PUBLIC core/factorization/pca_public.cpp)
set(DA_UTILS_PUBLIC core/utilities/utils_public.cpp
                    core/utilities/aoclda_cpp_overloads.cpp)
set(DA_OPTIONS_PUBLIC core/utilities/options_public.cpp)
set(DA_DATA
    core/data_management/interval_set.cpp
    core/data_management/data_store_public.cpp
    core/data_management/data_store.cpp)
set(DA_MISC core/utilities/miscellaneous.cpp)
set(DA_CONTEXT core/dynamic_dispatch/context.cpp)
set(DA_METRICS_PUBLIC core/metrics/pairwise_distances_public.cpp)
set(DA_NEAREST_NEIGHBORS_PUBLIC core/nearest_neighbors/knn_public.cpp)
set(DA_CLUSTERING_PUBLIC core/clustering/kmeans_public.cpp
                         core/clustering/dbscan_public.cpp)
set(DA_DECISION_FOREST_PUBLIC core/decision_forest/decision_tree_public.cpp
                              core/decision_forest/random_forest_public.cpp)
set(DA_NLLS_PUBLIC core/nlls/nlls_public.cpp)
set(DA_NLLS_INTERNAL core/nlls/nlls.cpp)
set(DA_HANDLE_PUBLIC core/utilities/da_handle.cpp
                     core/utilities/da_handle_public.cpp)
set(DA_CSV_PUBLIC core/csv/tokenizer.c core/csv/read_csv_public.cpp)
set(DA_SVM_PUBLIC core/svm/svm_public.cpp)
set(DA_KERNEL_FUNCTIONS_PUBLIC core/kernel_functions/kernel_functions_public.cpp)

# Internal APIs require multi-compilation for different zen architectures
set(DA_BASIC_STATISTICS_INTERNAL
    core/basic_statistics/statistical_utilities.cpp
    core/basic_statistics/moment_statistics.cpp
    core/basic_statistics/correlation_and_covariance.cpp
    core/basic_statistics/order_statistics.cpp
    core/basic_statistics/row_to_col_major.cpp)
set(DA_FACTORIZATION_INTERNAL core/factorization/pca.cpp)
set(DA_DECISION_FOREST_INTERNAL core/decision_forest/decision_tree.cpp
                                core/decision_forest/random_forest.cpp)
set(DA_NEAREST_NEIGHBORS_INTERNAL core/nearest_neighbors/knn.cpp)
set(DA_CLUSTERING_INTERNAL
    core/clustering/kmeans.cpp core/clustering/dbscan.cpp
    core/clustering/radius_neighbors.cpp)
set(DA_UTILS_INTERNAL core/utilities/da_utils.cpp)
set(DA_OPTIMIZATION_INTERNAL core/optimization/optimization.cpp)
set(DA_SVM_INTERNAL core/svm/svm.cpp core/svm/base_svm.cpp core/svm/c_svm.cpp
                    core/svm/nu_svm.cpp)
set(DA_LINMOD_INTERNAL
    core/linear_model/linear_model.cpp core/linear_model/linmod_cg.cpp
    core/linear_model/linmod_cholesky.cpp core/linear_model/linmod_qr.cpp
    core/linear_model/linmod_svd.cpp core/linear_model/linmod_nln_optim.cpp)
set(DA_BASIC_HANDLE_INTERNAL core/utilities/basic_handle.cpp)
set(DA_KERNEL_FUNCTIONS_INTERNAL core/kernel_functions/kernel_functions.cpp)
set(DA_METRICS_INTERNAL core/metrics/pairwise_distances.cpp
                        core/metrics/euclidean_distance.cpp
                        core/metrics/cosine_distance.cpp
                        core/metrics/manhattan_distance.cpp
                        core/metrics/minkowski_distance.cpp)

set(DA_CORE_PUBLIC
    ${DA_LINMOD_PUBLIC}
    ${DA_BASIC_STATISTICS_PUBLIC}
    ${DA_CSV_PUBLIC}
    ${DA_HANDLE_PUBLIC}
    ${DA_OPTIONS_PUBLIC}
    ${DA_DATA}
    ${DA_FACTORIZATION_PUBLIC}
    ${DA_CLUSTERING_PUBLIC}
    ${DA_DECISION_FOREST_PUBLIC}
    ${DA_NLLS_PUBLIC}
    ${DA_MISC}
    ${DA_CONTEXT}
    ${DA_METRICS_PUBLIC}
    ${DA_UTILS_PUBLIC}
    ${DA_NEAREST_NEIGHBORS_PUBLIC}
    ${DA_SVM_PUBLIC}
    ${DA_KERNEL_FUNCTIONS_PUBLIC})

set(DA_CORE_DYNAMIC_DISPATCH
    ${DA_FACTORIZATION_INTERNAL}
    ${DA_LINMOD_INTERNAL}
    ${DA_UTILS_INTERNAL}
    ${DA_BASIC_HANDLE_INTERNAL}
    ${DA_CLUSTERING_INTERNAL}
    ${DA_NEAREST_NEIGHBORS_INTERNAL}
    ${DA_DECISION_FOREST_INTERNAL}
    ${DA_SVM_INTERNAL}
    ${DA_NLLS_INTERNAL}
    ${DA_OPTIMIZATION_INTERNAL}
    ${DA_BASIC_STATISTICS_INTERNAL}
    ${DA_KERNEL_FUNCTIONS_INTERNAL}
    ${DA_METRICS_INTERNAL})

if(ARCH STREQUAL "dynamic")
  # Build archs list
  supported_architectures(ARCHITECTURES DEFINITIONS)
elseif(ARCH STREQUAL "native")
  set(ARCHITECTURES "native")
else()
  set(ARCHITECTURES ${ARCH})
endif()

# Extend version string with built architecture(s)
string(REPLACE ";" " " ARCHITECTURES_LABELS "${ARCHITECTURES}")
string(REGEX REPLACE "(-DAOCLDA_VERSION_STRING=\".*)(\")"
                "\\1 ${CMAKE_BUILD_TYPE} [${ARCHITECTURES_LABELS}]\\2"
                AOCLDA_DEFS "${AOCLDA_DEFS}")


add_library(da_core_public OBJECT ${DA_CORE_PUBLIC})
aoclda_add_dependencies(da_core_public)

# Generic compilation of objects containing public APIs
add_library(da_core OBJECT $<TARGET_OBJECTS:da_core_public>)

# Create combined library from da_core and Fortran libraries
add_library(
  aocl-da $<TARGET_OBJECTS:da_core> $<TARGET_OBJECTS:da_core_public>
          $<TARGET_OBJECTS:lbfgsb> $<TARGET_OBJECTS:lbfgsb_s> ${nlls_obj_list})

# Set -mavx2 so generic code will still perform acceptably
set(GENERIC_FLAGS "-march=x86-64;-mavx2")

# Add compile definitions
if(ARCH STREQUAL "dynamic")
  # Dynamic dispatch: get a list of compilation architectures and associated
  # compile definitions
  target_compile_options(da_core_public PRIVATE ${GENERIC_FLAGS})
  target_compile_definitions(da_core_public PRIVATE ${DEFINITIONS})
elseif(ARCH STREQUAL "native")
  # For native builds reuse the generic compile definitions to give access to
  # the generic namespaces internally
  target_compile_definitions(da_core_public
                             PRIVATE generic_AVAILABLE;ZNVER_MAX=generic)
else()
  # For a specific target build, use the target architecture and set the
  # appropriate compile options
  target_compile_options(da_core_public PRIVATE -march=${ARCH})
  extract_znver_generation(ARCH MAX_ZNVER)
  target_compile_definitions(
    da_core_public PRIVATE ${ARCH}_AVAILABLE;ZNVER_MAX=zen${MAX_ZNVER})
endif()

# Store for display purposes
set(BUILD_ARCHS
    ${ARCHITECTURES}
    CACHE STRING "Architecture(s) to compile for")

foreach(ARCHITECTURE IN LISTS ARCHITECTURES)

  set(ARCH_OBJECTS ${ARCHITECTURE}_OBJECTS)
  add_library(${ARCH_OBJECTS} OBJECT ${DA_CORE_DYNAMIC_DISPATCH})

  # Set the compile definitions for the architecture
  if(ARCHITECTURE STREQUAL "native" OR ARCHITECTURE STREQUAL "generic")
    # For a native build we will reuse the generic namespaces internally but
    # compile with -march=native
    target_compile_definitions(${ARCH_OBJECTS} PRIVATE ARCH_generic)
  else()
    extract_znver_generation(${ARCHITECTURE} znver)
    target_compile_definitions(${ARCH_OBJECTS} PRIVATE ARCH_znver=${znver})
  endif()

  # Set the compile options for the architecture
  if(ARCHITECTURE STREQUAL "generic")
    target_compile_options(${ARCH_OBJECTS} PRIVATE ${GENERIC_FLAGS})
  else()
    target_compile_options(${ARCH_OBJECTS} PRIVATE -march=${ARCHITECTURE})
  endif()

  # Add the dependencies and sources to the target
  aoclda_add_dependencies(${ARCH_OBJECTS})
  target_sources(da_core PRIVATE $<TARGET_OBJECTS:${ARCH_OBJECTS}>)
  target_link_libraries(aocl-da PRIVATE ${ARCH_OBJECTS})
  add_dependencies(da_core_public ${ARCH_OBJECTS})

endforeach()

aoclda_add_dependencies(da_core)

# List of targets that should use -O2 for release/benchmark builds
set(USE_O2 "")

# List of targets that should use vectorised math for release/benchmark builds
set(USE_VEC_MATH ${DA_SVM})

# By default release builds have -O3, here we add -O2 which will override it for
# the source files in the USE_O2 list, since it will appear after the -O3
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND CMAKE_CXX_COMPILER_ID STREQUAL
                                           "Clang")
  foreach(SOURCE ${DA_CORE_GENERIC} ${DA_CORE_DYNAMIC_DISPATCH})
    if(SOURCE IN_LIST USE_O2)
      set_source_files_properties(${SOURCE} PROPERTIES COMPILE_FLAGS "-O2")
    endif()
    # Add flag to enable vectorised math given that amdlibm has been linked
    if(UNIX AND SOURCE IN_LIST USE_VEC_MATH)
      set_source_files_properties(${SOURCE} PROPERTIES COMPILE_FLAGS "-fveclib=AMDLIBM")
    endif()
  endforeach()
endif()

if(WIN32 AND CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
  link_libraries("flang")
endif()

if(NOT WIN32)
  include(Suppressions_core)
endif()

target_link_libraries(aocl-da PRIVATE ${SPARSE} ${LAPACK} ${BLAS} ${UTILS}
                                      ${FORTRAN_RUNTIME})

# If we use AOCC then link libm library
if(UNIX AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_link_libraries(aocl-da PRIVATE amdlibm)
endif()

target_link_directories(aocl-da PRIVATE ${IFORT_LIBS})
target_link_directories(aocl-da PRIVATE ${FLANG_LIBS})
target_include_directories(
  aocl-da PRIVATE ${BLAS_INCLUDE_DIR} ${SPARSE_INCLUDE_DIR}
                  ${LAPACK_INCLUDE_DIR} ${UTILS_INCLUDE_DIR})
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
