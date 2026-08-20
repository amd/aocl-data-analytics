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

# Exclude gtest and gmock from installation
set(INSTALL_GTEST OFF)
set(INSTALL_GMOCK OFF)

# get gtest from repo
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.x)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
  ON
  CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

if(NOT WIN32)
  include(Suppressions_tests)
endif()

# Some tests need *_AVAILABLE arch defs but not the whole ARCH_DEFS, so
# define a separate variable for just the ARCH_*_AVAILABLE defs
set(AVAILABLE_ARCH_DEFS ${ARCH_DEFS})
list(FILTER AVAILABLE_ARCH_DEFS INCLUDE REGEX "_AVAILABLE")
message(DEBUG "Available arch defs: ${AVAILABLE_ARCH_DEFS}")

include(GoogleTest)
include_directories(
  ${CMAKE_SOURCE_DIR}/source/include
  ${CMAKE_SOURCE_DIR}/external/Lbfgsb.3.0
  ${CMAKE_SOURCE_DIR}/external/kernel-templates
  ${CMAKE_SOURCE_DIR}/external/RALFit
  ${CMAKE_SOURCE_DIR}/external/RALFit/libRALFit/include
  ${Boost_INCLUDE_DIRS})

set(INTERNAL_INC_DIRS
  ${CMAKE_SOURCE_DIR}/source/core/csv
  ${CMAKE_SOURCE_DIR}/source/core/clustering
  ${CMAKE_SOURCE_DIR}/source/core/optimization
  ${CMAKE_SOURCE_DIR}/source/core/data_management
  ${CMAKE_SOURCE_DIR}/source/core/drivers
  ${CMAKE_SOURCE_DIR}/source/core/factorization
  ${CMAKE_SOURCE_DIR}/source/core/nlls
  ${CMAKE_SOURCE_DIR}/source/core/nearest_neighbors
  ${CMAKE_SOURCE_DIR}/source/core/utilities
  ${CMAKE_SOURCE_DIR}/source/core/decision_forest
  ${CMAKE_SOURCE_DIR}/source/core/decision_forest/tree
  ${CMAKE_SOURCE_DIR}/source/core/metrics
  ${CMAKE_SOURCE_DIR}/source/core/linear_model
  ${CMAKE_SOURCE_DIR}/source/core/basic_statistics
  ${CMAKE_SOURCE_DIR}/source/core/dynamic_dispatch
  ${CMAKE_SOURCE_DIR}/source/core/kernel_functions
  ${CMAKE_SOURCE_DIR}/source/core/svm
  ${CMAKE_SOURCE_DIR}/source/core/interpolation
  ${CMAKE_SOURCE_DIR}/source/core/train_test_split
  ${CMAKE_SOURCE_DIR}/source/core/approximate_neighbors
  ${CMAKE_SOURCE_DIR}/source/core/dimension_reduction
)

set(DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../data/")

# ##############################################################################
# ########## Basic stats #############
# ##############################################################################
add_executable(basic_stats_public basic_statistics/basic_stats_public.cpp)
target_link_libraries(basic_stats_public PRIVATE ${BLAS})

# ##############################################################################
# ########## Train Test Split #############
# ##############################################################################
add_executable(train_test_split_internal train_test_split/train_test_split_internal.cpp)
target_link_libraries(train_test_split_internal PRIVATE ${BLAS})

add_executable(train_test_split_public
  train_test_split/train_test_split_public.cpp)
target_compile_definitions(train_test_split_public PRIVATE DATA_DIR="${DATA_PATH}")
target_link_libraries(train_test_split_public PRIVATE ${BLAS})

add_executable(shuffle_indices_public
  train_test_split/shuffle_indices_public.cpp)
target_link_libraries(shuffle_indices_public PRIVATE ${BLAS})

# ##############################################################################
# ############ Metrics ###############
# ##############################################################################
add_executable(metrics_public metrics/metric_public.cpp)
target_link_libraries(metrics_public PRIVATE ${BLAS})
add_executable(euclidean_distance_internal
  metrics/euclidean_distance_internal.cpp)
target_link_libraries(euclidean_distance_internal PRIVATE ${BLAS})

# ##############################################################################
# ############## CSV #################
# ##############################################################################
add_executable(csv_public csv/csv_utils.cpp csv/csv_public.cpp)
target_compile_definitions(csv_public PRIVATE DATA_DIR="${DATA_PATH}")

add_executable(csv_internal csv/csv_utils.cpp csv/csv_internal.cpp)

# ##############################################################################
# ######### Linear models ############
# ##############################################################################
add_executable(linmod_public linear_models/linmod_public.cpp)
add_executable(linmod_internal linear_models/linmod_internal.cpp)
target_compile_definitions(linmod_internal PRIVATE ${AVAILABLE_ARCH_DEFS})

add_executable(logreg_public linear_models/logreg_public.cpp)
target_compile_definitions(logreg_public PRIVATE DATA_DIR="${DATA_PATH}")

add_executable(linreg_noreg_public linear_models/linreg_noreg_public.cpp)
target_compile_definitions(linreg_noreg_public PRIVATE DATA_DIR="${DATA_PATH}")

add_executable(linreg_ridge_public linear_models/linreg_ridge_public.cpp)
target_compile_definitions(linreg_ridge_public PRIVATE DATA_DIR="${DATA_PATH}")

add_executable(linreg_lasso_public linear_models/linreg_lasso_public.cpp)
target_compile_definitions(linreg_lasso_public PRIVATE DATA_DIR="${DATA_PATH}")

add_executable(linreg_elnet_public linear_models/linreg_elnet_public.cpp)
target_compile_definitions(linreg_elnet_public PRIVATE DATA_DIR="${DATA_PATH}")

# ##############################################################################
# Decision trees/forests ########
# ##############################################################################
add_executable(decision_tree_public decision_trees/decision_tree_public.cpp)
target_compile_definitions(decision_tree_public PRIVATE DATA_DIR="${DATA_PATH}")
target_link_libraries(decision_tree_public PRIVATE ${BLAS})

add_executable(decision_tree_internal decision_trees/decision_tree_internal.cpp)
target_compile_definitions(decision_tree_internal PRIVATE DATA_DIR="${DATA_PATH}")
target_link_libraries(decision_tree_internal PRIVATE ${BLAS})

add_executable(decision_forest_public decision_forests/decision_forest_public.cpp)
target_compile_definitions(decision_forest_public PRIVATE DATA_DIR="${DATA_PATH}")
target_link_libraries(decision_forest_public PRIVATE ${BLAS})
add_executable(decision_forest_internal decision_forests/decision_forest_internal.cpp)
target_compile_definitions(decision_forest_internal PRIVATE DATA_DIR="${DATA_PATH}")



# ##############################################################################
# DA Utilities (opts, errors, ...) ########
# ##############################################################################
add_executable(option_registry_internal
  da_utilities/option_registry_internal.cpp)

add_executable(errors_public da_utilities/errors_public.cpp)
add_executable(errors_internal da_utilities/errors_internal.cpp)

add_executable(parallel_public da_utilities/parallel_public.cpp)

add_executable(doc_internal da_utilities/doc_internal.cpp)

add_executable(utilities_public da_utilities/utilities_public.cpp)
target_link_libraries(utilities_public PRIVATE aocl-da)

add_executable(utilities_internal da_utilities/utilities_internal.cpp)
target_compile_definitions(utilities_internal PRIVATE INT_LIB="${INT_LIB}")
target_link_libraries(utilities_internal PRIVATE aocl-da)

add_executable(printf_debug_internal da_utilities/printf_debug_internal.cpp)
target_link_libraries(printf_debug_internal PRIVATE aocl-da)

add_executable(da_vector_internal da_utilities/da_vector_internal.cpp)

add_executable(da_cache_internal da_utilities/da_cache_internal.cpp)
target_link_libraries(da_cache_internal PRIVATE aocl-da)

add_executable(da_syrk_internal da_utilities/da_syrk_internal.cpp)
target_link_libraries(da_syrk_internal PRIVATE aocl-da ${BLAS})

# ##############################################################################
# ############ FP16 ##################
# ##############################################################################
add_executable(fp16_syrk_internal fp16/fp16_syrk_internal.cpp)
target_link_libraries(fp16_syrk_internal PRIVATE aocl-da ${BLAS})
# Static aocl-da propagates its DLP dependency (whole-archived) transitively,
# so DLP must NOT be repeated here. A shared aocl-da does not propagate its PRIVATE DLP
# dependency, so link DLP explicitly in that case.
if(NOT WIN32 AND BUILD_SHARED_LIBS)
  target_link_libraries(fp16_syrk_internal PRIVATE ${DLP})
endif()

# C compatibility test - compile with C compiler and link with C++ compiler
set_source_files_properties(da_utilities/c_compatibility_nog_public.c
  PROPERTIES LANGUAGE C)
add_executable(c_compatibility_nog_public
  da_utilities/c_compatibility_nog_public.c)
set_target_properties(c_compatibility_nog_public PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(c_compatibility_nog_public PRIVATE aocl-da)

add_executable(miscellaneous_internal da_utilities/miscellaneous_internal.cpp)
target_compile_definitions(miscellaneous_internal PRIVATE ${AOCLDA_DEFS} ${AVAILABLE_ARCH_DEFS})

# ##############################################################################
# ############ kmeans ################
# ##############################################################################
add_executable(kmeans_public kmeans/kmeans_public.cpp)

# ##############################################################################
# ############ t-SNE #################
# ##############################################################################
add_executable(tsne_public tsne/tsne_public.cpp)
target_compile_definitions(tsne_public PRIVATE DATA_DIR="${DATA_PATH}")
add_executable(tsne_internal tsne/tsne_internal.cpp)

# ##############################################################################
# ############ DBSCAN ################
# ##############################################################################
add_executable(dbscan_public dbscan/dbscan_public.cpp)

# ##############################################################################
# ######### Optimization #############
# ##############################################################################
add_executable(nlls_public optimization/nlls_public.cpp)
add_executable(nlls_internal optimization/nlls_internal.cpp)

add_executable(lbfgsb_rc_internal optimization/lbfgsb_rc_internal.cpp)
target_link_libraries(lbfgsb_rc_internal PRIVATE aocl-da)

add_executable(lbfgsb_fcomm_internal optimization/lbfgsb_fcomm_internal.cpp)
target_link_libraries(lbfgsb_fcomm_internal PRIVATE aocl-da)

add_executable(coord_internal optimization/coord_internal.cpp)

# ##############################################################################
# ############# PCA ##################
# ##############################################################################
add_executable(pca_public factorization/pca_public.cpp)
target_link_libraries(pca_public PRIVATE ${BLAS})

# ##############################################################################
# ######## Kernel PCA #############
# ##############################################################################
add_executable(kernel_pca_public factorization/kernel_pca_public.cpp)
target_link_libraries(kernel_pca_public PRIVATE ${BLAS})

# ##############################################################################
# Data management ###########
# ##############################################################################
add_executable(data_public data_management/data_public.cpp)
target_compile_definitions(data_public PRIVATE DATA_DIR="${DATA_PATH}")
add_executable(data_internal data_management/data_internal.cpp)
target_compile_definitions(data_internal PRIVATE DATA_DIR="${DATA_PATH}")

add_executable(interval_map_internal data_management/interval_map_internal.cpp)
add_executable(interval_set_internal data_management/interval_set_internal.cpp)

# ##############################################################################
# ############# nearest neighbors ##################
# ##############################################################################
add_executable(nearest_neighbors_public nearest_neighbors/nearest_neighbors_public.cpp)
target_link_libraries(nearest_neighbors_public PRIVATE aocl-da ${BLAS})
target_compile_definitions(nearest_neighbors_public PRIVATE DATA_DIR="${DATA_PATH}")
add_executable(trees_internal nearest_neighbors/trees_internal.cpp)
add_executable(radius_neighbors_internal
  nearest_neighbors/radius_neighbors_internal.cpp)
add_executable(k_neighbors_internal nearest_neighbors/k_neighbors_internal.cpp)
target_link_libraries(radius_neighbors_internal PRIVATE ${BLAS})

# ##############################################################################
# ######### Kernel function ###########
# ##############################################################################
add_executable(kernel_functions_public
  kernel_functions/kernel_functions_public.cpp)

add_executable(kernel_functions_internal
  kernel_functions/kernel_functions_internal.cpp)

# ##############################################################################
# ############ KT L2 micro kernels used in Kernel functions #################
# ##############################################################################
add_executable(ktl2 kernel_functions/ktl2.cpp)
add_executable(ktl2_fallback kernel_functions/ktl2.cpp)
target_compile_definitions(ktl2_fallback PRIVATE DA_KT_TEST_FALLBACK)

# ktl2 instantiates the AOCC vectorised kt_exp_p() (kt_l2_clang.hpp) directly in
# the test object, so it calls amd_vr* symbols itself and must link amdlibm.
if(COMPILER_FAMILY STREQUAL "LLVM")
  IsAOCC(IS_AOCC)
  if(IS_AOCC)
    target_link_libraries(ktl2 PRIVATE amdlibm)
  endif()
endif()

# ##############################################################################
# ############## SVM #################
# ##############################################################################
add_executable(svm_internal svm/svm_internal.cpp)
target_link_libraries(svm_internal PRIVATE ${BLAS})

add_executable(svm_public svm/svm_public.cpp)
target_compile_definitions(svm_public PRIVATE DATA_DIR="${DATA_PATH}")
target_link_libraries(svm_public PRIVATE ${BLAS})

# ##############################################################################
# ############## Interpolation #################
# ##############################################################################
add_executable(cubic_spline_public interpolation/cubic_spline_public.cpp)
target_compile_definitions(cubic_spline_public PRIVATE DATA_DIR="${DATA_PATH}")
target_link_libraries(cubic_spline_public PRIVATE aocl-da)

# ############## ANN #################
# ##############################################################################
add_executable(ann_public approximate_neighbors/ann_public.cpp)
target_link_libraries(ann_public PRIVATE aocl-da ${BLAS})
target_compile_definitions(ann_public PRIVATE DATA_DIR="${DATA_PATH}")

# ##############################################################################
# ############## Model Persistence #################
# ##############################################################################
set(TEST_OUTPUT_DIR "${CMAKE_BINARY_DIR}/tests/unit_tests/tmp_test_files")
file(MAKE_DIRECTORY ${TEST_OUTPUT_DIR})

add_executable(serialization_kernel_internal model_persistence/kernel_tests_internal.cpp)
target_compile_definitions(serialization_kernel_internal PRIVATE TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR}")

add_executable(serialization_kernel_errors_internal model_persistence/kernel_error_tests_internal.cpp)
target_compile_definitions(serialization_kernel_errors_internal PRIVATE TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR}")

add_executable(ann_persistence_public model_persistence/algorithms/approximate_neighbors_persistence_public.cpp)
target_compile_definitions(ann_persistence_public PRIVATE TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR}")
target_link_libraries(ann_persistence_public PRIVATE aocl-da ${BLAS})

add_executable(dforest_persistence_public model_persistence/algorithms/decision_forest_persistence_public.cpp)
target_compile_definitions(dforest_persistence_public PRIVATE TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR}")
target_link_libraries(dforest_persistence_public PRIVATE aocl-da ${BLAS})

add_executable(dtree_persistence_public model_persistence/algorithms/decision_tree_persistence_public.cpp)
target_compile_definitions(dtree_persistence_public PRIVATE TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR}")
target_link_libraries(dtree_persistence_public PRIVATE aocl-da ${BLAS})

add_executable(kmeans_persistence_public model_persistence/algorithms/kmeans_persistence_public.cpp)
target_compile_definitions(kmeans_persistence_public PRIVATE TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR}")
target_link_libraries(kmeans_persistence_public PRIVATE aocl-da ${BLAS})

add_executable(linmod_persistence_public model_persistence/algorithms/lin_mod_persistence_public.cpp)
target_compile_definitions(linmod_persistence_public PRIVATE TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR}")
target_link_libraries(linmod_persistence_public PRIVATE aocl-da ${BLAS})

add_executable(pca_persistence_public model_persistence/algorithms/pca_persistence_public.cpp)
target_compile_definitions(pca_persistence_public PRIVATE TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR}")
target_link_libraries(pca_persistence_public PRIVATE aocl-da ${BLAS})

add_executable(kernel_pca_persistence_public model_persistence/algorithms/kernel_pca_persistence_public.cpp)
target_compile_definitions(kernel_pca_persistence_public PRIVATE TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR}")
target_link_libraries(kernel_pca_persistence_public PRIVATE aocl-da ${BLAS})

add_executable(knn_class_persistence_public model_persistence/algorithms/knn_class_persistence_public.cpp)
target_compile_definitions(knn_class_persistence_public PRIVATE TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR}")
target_link_libraries(knn_class_persistence_public PRIVATE aocl-da ${BLAS})

add_executable(knn_reg_persistence_public model_persistence/algorithms/knn_reg_persistence_public.cpp)
target_compile_definitions(knn_reg_persistence_public PRIVATE TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR}")
target_link_libraries(knn_reg_persistence_public PRIVATE aocl-da ${BLAS})

add_executable(radius_nn_persistence_public model_persistence/algorithms/radius_neighbors_persistence_public.cpp)
target_compile_definitions(radius_nn_persistence_public PRIVATE TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR}")
target_link_libraries(radius_nn_persistence_public PRIVATE aocl-da ${BLAS})

add_executable(svm_persistence_public model_persistence/algorithms/svm_persistence_public.cpp)
target_compile_definitions(svm_persistence_public PRIVATE TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR}")
target_link_libraries(svm_persistence_public PRIVATE aocl-da ${BLAS})

add_executable(handle_persistence_public model_persistence/handle_errors_public.cpp)
target_compile_definitions(handle_persistence_public PRIVATE TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR}")
target_link_libraries(handle_persistence_public PRIVATE aocl-da ${BLAS})

# ##############################################################################
# ############## C API tests #################
# ##############################################################################
add_executable(ann_capi_public c_tests/ann_capi_public.cpp)
target_link_libraries(ann_capi_public PRIVATE aocl-da ${BLAS})

add_executable(basic_stats_capi_public c_tests/basic_stats_capi_public.cpp)
target_link_libraries(basic_stats_capi_public PRIVATE aocl-da ${BLAS})

add_executable(csv_capi_public c_tests/csv_capi_public.cpp)
target_compile_definitions(csv_capi_public PRIVATE DATA_DIR="${DATA_PATH}")
target_link_libraries(csv_capi_public PRIVATE aocl-da ${BLAS})

add_executable(datastore_capi_public c_tests/datastore_capi_public.cpp)
target_compile_definitions(datastore_capi_public PRIVATE DATA_DIR="${DATA_PATH}")
target_link_libraries(datastore_capi_public PRIVATE aocl-da ${BLAS})

add_executable(dbscan_capi_public c_tests/dbscan_capi_public.cpp)
target_link_libraries(dbscan_capi_public PRIVATE aocl-da ${BLAS})

add_executable(decision_forest_capi_public c_tests/decision_forest_capi_public.cpp)
target_link_libraries(decision_forest_capi_public PRIVATE aocl-da ${BLAS})

add_executable(decision_tree_capi_public c_tests/decision_tree_capi_public.cpp)
target_link_libraries(decision_tree_capi_public PRIVATE aocl-da ${BLAS})

add_executable(handle_capi_public c_tests/handle_capi_public.cpp)
target_compile_definitions(handle_capi_public PRIVATE TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR}")
target_link_libraries(handle_capi_public PRIVATE aocl-da ${BLAS})

add_executable(interpolation_capi_public c_tests/interpolation_capi_public.cpp)
target_link_libraries(interpolation_capi_public PRIVATE aocl-da ${BLAS})

add_executable(kernel_functions_capi_public c_tests/kernel_functions_capi_public.cpp)
target_link_libraries(kernel_functions_capi_public PRIVATE aocl-da ${BLAS})

add_executable(kmeans_capi_public c_tests/kmeans_capi_public.cpp)
target_link_libraries(kmeans_capi_public PRIVATE aocl-da ${BLAS})

add_executable(linmod_capi_public c_tests/linmod_capi_public.cpp)
target_link_libraries(linmod_capi_public PRIVATE aocl-da ${BLAS})

add_executable(metrics_capi_public c_tests/metrics_capi_public.cpp)
target_link_libraries(metrics_capi_public PRIVATE aocl-da ${BLAS})

add_executable(miscellaneous_capi_public c_tests/miscellaneous_capi_public.cpp)
target_link_libraries(miscellaneous_capi_public PRIVATE aocl-da ${BLAS})

add_executable(nlls_capi_public c_tests/nlls_capi_public.cpp)
target_link_libraries(nlls_capi_public PRIVATE aocl-da ${BLAS})

add_executable(nn_capi_public c_tests/nn_capi_public.cpp)
target_link_libraries(nn_capi_public PRIVATE aocl-da ${BLAS})

add_executable(options_capi_public c_tests/options_capi_public.cpp)
target_link_libraries(options_capi_public PRIVATE aocl-da ${BLAS})

add_executable(pca_capi_public c_tests/pca_capi_public.cpp)
target_link_libraries(pca_capi_public PRIVATE aocl-da ${BLAS})

add_executable(result_capi_public c_tests/result_capi_public.cpp)
target_link_libraries(result_capi_public PRIVATE aocl-da ${BLAS})

add_executable(svm_capi_public c_tests/svm_capi_public.cpp)
target_link_libraries(svm_capi_public PRIVATE aocl-da ${BLAS})

add_executable(tsne_capi_public c_tests/tsne_capi_public.cpp)
target_link_libraries(tsne_capi_public PRIVATE aocl-da ${BLAS})

add_executable(utils_capi_public c_tests/utils_capi_public.cpp)
target_link_libraries(utils_capi_public PRIVATE aocl-da ${BLAS})

set(GTEST_INTERNAL
  csv_internal
  linmod_internal
  decision_tree_internal
  decision_forest_internal
  option_registry_internal
  data_internal
  interval_map_internal
  euclidean_distance_internal
  interval_set_internal
  errors_internal
  doc_internal
  coord_internal
  nlls_internal
  svm_internal
  tsne_internal
  train_test_split_internal
  kernel_functions_internal
  ktl2
  ktl2_fallback
  miscellaneous_internal
  da_vector_internal
  da_cache_internal
  radius_neighbors_internal
  k_neighbors_internal
  trees_internal
  utilities_internal
  da_syrk_internal
  fp16_syrk_internal
  serialization_kernel_internal
  serialization_kernel_errors_internal)
set(INDEP_INTERNAL printf_debug_internal lbfgsb_rc_internal
  lbfgsb_fcomm_internal)
set(INTERNAL_UTESTS ${GTEST_INTERNAL} ${INDEP_INTERNAL})

set(GTEST_PUBLIC
  basic_stats_public
  train_test_split_public
  shuffle_indices_public
  metrics_public
  csv_public
  linmod_public
  logreg_public
  linreg_noreg_public
  linreg_ridge_public
  linreg_lasso_public
  linreg_elnet_public
  decision_forest_public
  decision_tree_public
  data_public
  errors_public
  pca_public
  kernel_pca_public
  kmeans_public
  tsne_public
  dbscan_public
  parallel_public
  utilities_public
  nlls_public
  nearest_neighbors_public
  kernel_functions_public
  svm_public
  cubic_spline_public
  ann_public
  ann_persistence_public
  dforest_persistence_public
  dtree_persistence_public
  kmeans_persistence_public
  linmod_persistence_public
  pca_persistence_public
  kernel_pca_persistence_public
  knn_class_persistence_public
  knn_reg_persistence_public
  radius_nn_persistence_public
  svm_persistence_public
  handle_persistence_public
  ann_capi_public
  basic_stats_capi_public
  csv_capi_public
  datastore_capi_public
  dbscan_capi_public
  decision_forest_capi_public
  decision_tree_capi_public
  handle_capi_public
  interpolation_capi_public
  kernel_functions_capi_public
  kmeans_capi_public
  linmod_capi_public
  metrics_capi_public
  miscellaneous_capi_public
  nlls_capi_public
  nn_capi_public
  options_capi_public
  pca_capi_public
  result_capi_public
  svm_capi_public
  tsne_capi_public
  utils_capi_public)
set(INDEP_PUBLIC c_compatibility_nog_public)
set(PUBLIC_UTESTS ${GTEST_PUBLIC} ${INDEP_PUBLIC})

set(INDEP_EXES ${INDEP_INTERNAL} ${INDEP_PUBLIC})
set(GTEST_EXES ${GTEST_INTERNAL} ${GTEST_PUBLIC})
set(UTESTS ${INTERNAL_UTESTS} ${PUBLIC_UTESTS})

# Include source/core directories for internal tests
foreach(internal_test ${INTERNAL_UTESTS})
  target_include_directories(${internal_test} PRIVATE ${INTERNAL_INC_DIRS})
  if(ARCH STREQUAL "dynamic" OR ARCH STREQUAL "generic")
    # Internal tests will just use the generic codepath
    target_compile_definitions(${internal_test}
      PRIVATE ARCH_generic;ZNVER_MAX=generic)
  elseif(ARCH STREQUAL "generic_avx512")
    target_compile_definitions(
      ${internal_test} PRIVATE ARCH_generic_avx512;ZNVER_MAX=generic_avx512)
  elseif(ARCH STREQUAL "native")
    # Extract the target Zen generation, as this namespace is used in the code
    extract_native_namespace(NATIVE_NAMESPACE)
    if(NATIVE_NAMESPACE STREQUAL "generic")
      target_compile_definitions(${internal_test}
        PRIVATE ARCH_generic;ZNVER_MAX=generic)
    elseif(NATIVE_NAMESPACE STREQUAL "generic_avx512")
      target_compile_definitions(
        ${internal_test}
        PRIVATE ARCH_generic_avx512;ZNVER_MAX=generic_avx512)
    else()
      target_compile_definitions(
        ${internal_test}
        PRIVATE ARCH_znver=${NATIVE_NAMESPACE};ZNVER_MAX=zen${NATIVE_NAMESPACE})
    endif()
  else()
    if(${ARCH} MATCHES "^znver([0-9]+)$")
      set(znver ${CMAKE_MATCH_1})
    endif()
    target_compile_definitions(
      ${internal_test} PRIVATE ARCH_znver=${znver};ZNVER_MAX=zen${znver})
  endif()
endforeach()

# Some compilers complain about gtest variadic macros, try to suppress
AppendCXXFlags(SUPPRESSIONS -Wno-gnu-zero-variadic-macro-arguments -Wno-c++20-extensions)
string(REGEX REPLACE "\ " ";" SUPPRESSIONS "${SUPPRESSIONS}")

foreach(target ${UTESTS})
  target_compile_options(${target} PRIVATE -march=native)
  foreach(flag_ ${SUPPRESSIONS})
    target_compile_options(${target} PRIVATE ${flag_})
  endforeach()
  target_link_libraries(${target} PRIVATE aocl-da gtest_main gmock_main)
  if(BUILD_SMP)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  target_link_directories(${target} PRIVATE ${IFORT_LIBS})
  target_link_directories(${target} PRIVATE ${FLANG_LIBS})
  target_link_libraries(${target} PRIVATE ${FORTRAN_RUNTIME})
  # Define -DAOCL_ILP64 when ILP64 libraries are built. For LP64, this is an
  # empty string. This is what AOCL-DA users will need to do explicitly to get
  # the correct symbols, otherwise they will get linking errors or aborts.
  target_compile_definitions(${target} PRIVATE ${AOCLDA_ILP64})
endforeach()

if(WIN32 AND BUILD_SHARED_LIBS)
  # Copy runtime DLLs once, via a single custom target gated on all unit-test
  # exes.
  # We do this via a single custom target rather than a per-executable POST_BUILD
  # command, because every example writes the same DLLs to the same destination
  # directory; running N parallel `cmake -E copy_if_different` invocations on
  # Windows causes sporadic sharing-violation failures ("Error copying file (if
  # different)"). Using one ALL target gated on all example exes guarantees the
  # copy runs exactly once after all examples have linked.
  list(GET UTESTS 0 UTEST_EXE)

  add_custom_target(copy_utest_runtime_dlls ALL
    COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_RUNTIME_DLLS:${UTEST_EXE}>
    $<TARGET_FILE_DIR:${UTEST_EXE}>
    COMMAND_EXPAND_LISTS
    VERBATIM)
  add_dependencies(copy_utest_runtime_dlls ${UTESTS})

  # gtest_discover_tests does not work with Windows shared library builds
  foreach(UTEST ${UTESTS})
    add_test(${UTEST} ${UTEST})
  endforeach()
endif()

# Defer gtest enumeration to ctest time on Windows shared builds (otherwise the
# build-time exe launch hits STATUS_DLL_NOT_FOUND before copy_utest_runtime_dlls
# has run). Also defer when cross-targeting a specific ISA that may be newer
# than the build host (e.g. building znver6 on a Zen 3 box): POST_BUILD would
# launch the freshly-built exe at build time and SIGILL ("Illegal instruction")
# on the unsupported opcodes. PRE_TEST lets the build finish; the tests are
# still only runnable on matching hardware at ctest time. Everywhere else, keep
# the default POST_BUILD enumeration.
if(WIN32 AND BUILD_SHARED_LIBS)
  set(GTEST_DISCOVERY_MODE PRE_TEST)
elseif(NOT (ARCH STREQUAL "native" OR ARCH STREQUAL "dynamic" OR ARCH STREQUAL
     "generic"))
  set(GTEST_DISCOVERY_MODE PRE_TEST)
else()
  set(GTEST_DISCOVERY_MODE POST_BUILD)
endif()

foreach(GTEST_EXE ${GTEST_EXES})
  gtest_discover_tests(${GTEST_EXE} DISCOVERY_TIMEOUT 120
    DISCOVERY_MODE ${GTEST_DISCOVERY_MODE})
endforeach()
foreach(INDEP_EXE ${INDEP_EXES})
  add_test(${INDEP_EXE} ${INDEP_EXE})
endforeach()
