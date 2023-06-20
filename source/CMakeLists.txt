set(LINMOD_SRC core/linear_model/linmod_public.cpp)
set(BASIC_STATISTICS_SRC core/basic_statistics/basic_statistics_public.cpp)
set(DA_FACTORIZATION core/factorization/pca_public.cpp)
set(CSV core/csv/tokenizer.c core/csv/read_csv_public.cpp)
set(DA_HANDLE core/utilities/da_handle.cpp core/utilities/da_handle_public.cpp)
set(DA_OPTIONS core/utilities/options_public.cpp)
set(DA_DATA core/data_management/data_store_public.cpp)

set(DA_CORE_SRC
    ${LINMOD_SRC}
    ${BASIC_STATISTICS_SRC}
    ${CSV}
    ${DA_HANDLE}
    ${DA_OPTIONS}
    ${DA_DATA}
    ${DA_FACTORIZATION})

add_library(da_core ${DA_CORE_SRC})
target_include_directories(da_core PUBLIC include)
target_include_directories(
  da_core
  PUBLIC core/drivers/
         core/optimization/
         core/utilities
         core/csv
         core/linear_model
         core/data_management
         core/factorization)
target_include_directories(da_core
                           PRIVATE ${CMAKE_SOURCE_DIR}/external/Lbfgsb.3.0/)

if(WIN32)
  target_link_directories(
    da_core PUBLIC
    "${CMAKE_BINARY_DIR}/external/Lbfgsb.3.0/double_cmake/${CMAKE_BUILD_TYPE}"
    "${CMAKE_BINARY_DIR}/external/Lbfgsb.3.0/single_cmake/${CMAKE_BUILD_TYPE}")
endif()

target_link_libraries(da_core lbfgsb lbfgsb_s ${FLAME} ${BLIS})
if(NOT WIN32)
  target_link_libraries(da_core -lgomp)
endif()

if(NOT WIN32)
  include(Suppressions_core)
endif()
