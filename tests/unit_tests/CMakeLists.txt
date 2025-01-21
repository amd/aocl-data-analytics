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

include(GoogleTest)
include_directories(
  ${CMAKE_SOURCE_DIR}/source/include ${CMAKE_SOURCE_DIR}/external/Lbfgsb.3.0
  ${CMAKE_SOURCE_DIR}/external/RALFit
  ${CMAKE_SOURCE_DIR}/external/RALFit/libRALFit/include ${Boost_INCLUDE_DIRS})

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
    ${CMAKE_SOURCE_DIR}/source/core/metrics
    ${CMAKE_SOURCE_DIR}/source/core/linear_model
    ${CMAKE_SOURCE_DIR}/source/core/basic_statistics
    ${CMAKE_SOURCE_DIR}/source/core/dynamic_dispatch
    ${CMAKE_SOURCE_DIR}/source/core/kernel_function
    ${CMAKE_SOURCE_DIR}/source/core/svm)

set(DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../data/")

# ##############################################################################
# ########### Basic stats #############
# ##############################################################################
add_executable(basic_stats_public basic_statistics/basic_stats_public.cpp)
target_link_libraries(basic_stats_public PRIVATE ${BLAS})

# ##############################################################################
# ############# Metrics ###############
# ##############################################################################
add_executable(metrics_public metrics/metric_public.cpp)
target_link_libraries(metrics_public PRIVATE ${BLAS})
add_executable(euclidean_distance_internal
               metrics/euclidean_distance_internal.cpp)
target_link_libraries(euclidean_distance_internal PRIVATE ${BLAS})

# ##############################################################################
# ############### CSV #################
# ##############################################################################
add_executable(csv_public csv/csv_utils.cpp csv/csv_public.cpp)
target_compile_definitions(csv_public PRIVATE DATA_DIR="${DATA_PATH}")

add_executable(csv_internal csv/csv_utils.cpp csv/csv_internal.cpp)

# ##############################################################################
# ########## Linear models ############
# ##############################################################################
add_executable(linmod_public linear_models/linmod_public.cpp)
add_executable(linmod_internal linear_models/linmod_internal.cpp)

add_executable(logreg_public linear_models/logreg_public.cpp)
target_compile_definitions(logreg_public PRIVATE DATA_DIR="${DATA_PATH}")

add_executable(linreg_public linear_models/linreg_public.cpp)
target_compile_definitions(linreg_public PRIVATE DATA_DIR="${DATA_PATH}")

# ##############################################################################
# Decision trees/forests ########
# ##############################################################################
add_executable(dectree_public decision_trees/dectree_public.cpp)
target_compile_definitions(dectree_public PRIVATE DATA_DIR="${DATA_PATH}")
target_link_libraries(dectree_public PRIVATE ${BLAS})

add_executable(dectree_internal decision_trees/dectree_internal.cpp)

add_executable(decforest_public decision_forests/decforest_public.cpp)
target_compile_definitions(decforest_public PRIVATE DATA_DIR="${DATA_PATH}")
target_link_libraries(decforest_public PRIVATE ${BLAS})

# ##############################################################################
# DA Utilities (opts, errors, ...) ##
# ##############################################################################
add_executable(option_registry_internal
               da_utilities/option_registry_internal.cpp)

add_executable(errors_public da_utilities/errors_public.cpp)
add_executable(errors_internal da_utilities/errors_internal.cpp)

add_executable(parallel_public da_utilities/parallel_public.cpp)

add_executable(doc_internal da_utilities/doc_internal.cpp)

add_executable(utilities_public da_utilities/utilities_public.cpp)
target_link_libraries(utilities_public PRIVATE aocl-da)

add_executable(printf_debug_internal da_utilities/printf_debug_internal.cpp)
target_link_libraries(printf_debug_internal PRIVATE aocl-da)

add_executable(da_vector_internal da_utilities/da_vector_internal.cpp)

# C compatibility test - compile with C compiler and link with C++ compiler
set_source_files_properties(da_utilities/c_compatibility_nog_public.c
                            PROPERTIES LANGUAGE C)
add_executable(c_compatibility_nog_public
               da_utilities/c_compatibility_nog_public.c)
set_target_properties(c_compatibility_nog_public PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(c_compatibility_nog_public PRIVATE aocl-da)

add_executable(miscellaneous_internal da_utilities/miscellaneous_internal.cpp)
target_compile_definitions(miscellaneous_internal PRIVATE ${VERS_DEF})

# ##############################################################################
# ############# kmeans ################
# ##############################################################################
add_executable(kmeans_public kmeans/kmeans_public.cpp)

# ##############################################################################
# ############# DBSCAN ################
# ##############################################################################
add_executable(dbscan_public dbscan/dbscan_public.cpp)
add_executable(dbscan_internal dbscan/dbscan_internal.cpp)
target_link_libraries(dbscan_internal PRIVATE ${BLAS})

# ##############################################################################
# ########## Optimization #############
# ##############################################################################
add_executable(nlls_public optimization/nlls_public.cpp)
add_executable(nlls_internal optimization/nlls_internal.cpp)

add_executable(lbfgsb_rc_internal optimization/lbfgsb_rc_internal.cpp)
target_link_libraries(lbfgsb_rc_internal PRIVATE aocl-da)

add_executable(lbfgsb_fcomm_internal optimization/lbfgsb_fcomm_internal.cpp)
target_link_libraries(lbfgsb_fcomm_internal PRIVATE aocl-da)

add_executable(coord_internal optimization/coord_internal.cpp)

# ##############################################################################
# ############## PCA ##################
# ##############################################################################
add_executable(pca_public factorization/pca_public.cpp)
target_link_libraries(pca_public PRIVATE ${BLAS})

# ##############################################################################
# ######### Data management ###########
# ##############################################################################
add_executable(data_public data_management/data_public.cpp)
target_compile_definitions(data_public PRIVATE DATA_DIR="${DATA_PATH}")
add_executable(data_internal data_management/data_internal.cpp)
target_compile_definitions(data_internal PRIVATE DATA_DIR="${DATA_PATH}")

add_executable(interval_map_internal data_management/interval_map_internal.cpp)
add_executable(interval_set_internal data_management/interval_set_internal.cpp)

# ##############################################################################
# ############## knn ##################
# ##############################################################################
add_executable(knn_public knn/knn_public.cpp)
target_link_libraries(knn_public PRIVATE aocl-da ${BLAS})

# ##############################################################################
# ######### Kernel function ###########
# ##############################################################################
add_executable(kernel_function_public
               kernel_function/kernel_function_public.cpp)

# ##############################################################################
# ############### SVM #################
# ##############################################################################
add_executable(svm_internal svm/svm_internal.cpp)
target_link_libraries(svm_internal PRIVATE ${BLAS})

add_executable(svm_public svm/svm_public.cpp)
target_compile_definitions(svm_public PRIVATE DATA_DIR="${DATA_PATH}")
target_link_libraries(svm_public PRIVATE ${BLAS})

set(GTEST_INTERNAL
    csv_internal
    linmod_internal
    dectree_internal
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
    miscellaneous_internal
    da_vector_internal
    dbscan_internal)
set(INDEP_INTERNAL printf_debug_internal lbfgsb_rc_internal
                   lbfgsb_fcomm_internal)
set(INTERNAL_UTESTS ${GTEST_INTERNAL} ${INDEP_INTERNAL})

set(GTEST_PUBLIC
    basic_stats_public
    metrics_public
    csv_public
    linmod_public
    logreg_public
    linreg_public
    decforest_public
    dectree_public
    data_public
    errors_public
    pca_public
    kmeans_public
    dbscan_public
    parallel_public
    utilities_public
    nlls_public
    knn_public
    kernel_function_public
    svm_public)
set(INDEP_PUBLIC c_compatibility_nog_public)
set(PUBLIC_UTESTS ${GTEST_PUBLIC} ${INDEP_PUBLIC})

set(INDEP_EXES ${INDEP_INTERNAL} ${INDEP_PUBLIC})
set(GTEST_EXES ${GTEST_INTERNAL} ${GTEST_PUBLIC})
set(UTESTS ${INTERNAL_UTESTS} ${PUBLIC_UTESTS})

# Include source/core directories for internal tests
foreach(internal_test ${INTERNAL_UTESTS})
  target_include_directories(${internal_test} PRIVATE ${INTERNAL_INC_DIRS})
  if(ARCH STREQUAL "native" OR ARCH STREQUAL "dynamic")
    # Internal tests will just use the generic codepath
    target_compile_definitions(${internal_test}
                               PRIVATE ARCH_generic;ZNVER_MAX=generic)
  else()
    if(${ARCH} MATCHES "^znver([0-9]+)$")
      set(znver ${CMAKE_MATCH_1})
    endif()
    target_compile_definitions(
      ${internal_test} PRIVATE ARCH_znver=${znver};ZNVER_MAX=zen${znver})
  endif()
endforeach()

foreach(target ${UTESTS})
  target_link_libraries(${target} PRIVATE aocl-da gtest_main gmock_main)
  if(BUILD_SMP OR UNIX)
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
  foreach(UTEST ${UTESTS})
    add_custom_command(
      TARGET ${UTEST}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              $<TARGET_RUNTIME_DLLS:${UTEST}> $<TARGET_FILE_DIR:${UTEST}>
      COMMAND_EXPAND_LISTS)

    # gtest_discover_tests does not work with Windows shared library builds
    add_test(${UTEST} ${UTEST})
  endforeach()
endif()

foreach(GTEST_EXE ${GTEST_EXES})
  gtest_discover_tests(${GTEST_EXE})
endforeach()
foreach(INDEP_EXE ${INDEP_EXES})
  add_test(${INDEP_EXE} ${INDEP_EXE})
endforeach()
