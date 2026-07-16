# Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

if(BUILD_FORTRAN)

  add_executable(nlls nlls.cpp)

  add_executable(nlls_single nlls_single.cpp)

  add_executable(nlls_lanczos_fd nlls_lanczos_fd.cpp)

  add_executable(reglinear_model reglinear_model.cpp)

  add_executable(logit_classifier logit_classifier.cpp)

endif()

add_executable(basic_statistics basic_statistics.cpp)

add_executable(train_test_split train_test_split.cpp)

add_executable(linear_model linear_model.cpp)

add_executable(linmod_diabetes linmod_diabetes.cpp)
target_compile_definitions(linmod_diabetes
  PRIVATE DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

add_executable(datastore datastore.cpp)
target_compile_definitions(datastore
  PRIVATE DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

add_executable(decision_tree decision_tree.cpp)
target_compile_definitions(
  decision_tree PRIVATE DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/df_data")

add_executable(model_persistence_decision_tree model_persistence_decision_tree.cpp)
target_link_libraries(model_persistence_decision_tree PRIVATE aocl-da)

add_executable(decision_forest decision_forest.cpp)

target_compile_definitions(
  decision_forest PRIVATE DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/df_data")

add_executable(pca pca.cpp)
target_link_libraries(pca PRIVATE aocl-da)

add_executable(model_persistence_pca model_persistence_pca.cpp)
target_link_libraries(model_persistence_pca PRIVATE aocl-da)

add_executable(pca_cancer pca_cancer.cpp)
target_link_libraries(pca_cancer PRIVATE aocl-da)
target_compile_definitions(pca_cancer
  PRIVATE DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

add_executable(kmeans kmeans.cpp)
target_link_libraries(kmeans PRIVATE aocl-da)

add_executable(tsne tsne.cpp)
target_link_libraries(tsne PRIVATE aocl-da)

add_executable(dbscan dbscan.cpp)
target_link_libraries(dbscan PRIVATE aocl-da)

add_executable(metrics metrics.cpp)

add_executable(knn_classification knn_classification.cpp)

add_executable(knn_regression knn_regression.cpp)

add_executable(radius_neighbors radius_neighbors.cpp)

add_executable(radius_neighbors_classification radius_neighbors_classification.cpp)

add_executable(radius_neighbors_regression radius_neighbors_regression.cpp)

add_executable(kernel_functions kernel_functions.cpp)

add_executable(kernel_pca kernel_pca.cpp)
target_link_libraries(kernel_pca PRIVATE aocl-da)

add_executable(svc svc.cpp)

add_executable(nusvr nusvr.cpp)

add_executable(ann ann.cpp)

add_executable(cubic_spline cubic_spline.cpp)

set(EXAMPLE_EXES
  linear_model
  linmod_diabetes
  basic_statistics
  train_test_split
  pca
  pca_cancer
  kmeans
  tsne
  dbscan
  datastore
  decision_tree
  decision_forest
  metrics
  model_persistence_decision_tree
  model_persistence_pca
  knn_classification
  knn_regression
  radius_neighbors
  radius_neighbors_classification
  radius_neighbors_regression
  svc
  nusvr
  kernel_functions
  ann
  cubic_spline
  kernel_pca)

if(BUILD_FORTRAN)
  list(APPEND EXAMPLE_EXES nlls nlls_single nlls_lanczos_fd reglinear_model logit_classifier)
endif()

foreach(target ${EXAMPLE_EXES})
  target_compile_options(${target} PRIVATE -march=native)
  target_link_libraries(${target} PRIVATE aocl-da)
  if(BUILD_SMP)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  # Define -DAOCL_ILP64 when ILP64 libraries are built. For LP64, this is an
  # empty string. This is what AOCL-DA users will need to do explicitly to get
  # the correct symbols, otherwise they will get linking errors or aborts.
  target_compile_definitions(${target} PRIVATE ${AOCLDA_ILP64})
  target_link_directories(${target} PRIVATE ${IFORT_LIBS})
  target_link_directories(${target} PRIVATE ${FLANG_LIBS})
  target_link_libraries(${target} PRIVATE ${FORTRAN_RUNTIME})
endforeach()

# For Windows shared library builds, copying the relevant DLLs into the examples
# bin folder enables ctest to run without needing to update PATH.
#
# We do this via a single custom target rather than a per-executable POST_BUILD
# command, because every example writes the same DLLs to the same destination
# directory; running N parallel `cmake -E copy_if_different` invocations on
# Windows causes sporadic sharing-violation failures ("Error copying file (if
# different)"). Using one ALL target gated on all example exes guarantees the
# copy runs exactly once after all examples have linked.
if(WIN32 AND BUILD_SHARED_LIBS)
  list(GET EXAMPLE_EXES 0 EXAMPLE_EXE)

  add_custom_target(copy_example_runtime_dlls ALL
    COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_RUNTIME_DLLS:${EXAMPLE_EXE}>
    $<TARGET_FILE_DIR:${EXAMPLE_EXE}>
    COMMAND_EXPAND_LISTS
    VERBATIM)
  add_dependencies(copy_example_runtime_dlls ${EXAMPLE_EXES})
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
