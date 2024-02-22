

set( source_files 
  ral_nlls_double.f90
  ral_nlls_internal.f90
  ral_nlls_dtrs_double.f90
  ral_nlls_symbols.f90
  ral_nlls_ciface.f90
  ral_nlls_workspaces.f90
  ral_nlls_printing.f90
  ral_nlls_bounds.f90
  nag_export_mod.f90
  ral_nlls_types.F90
  ) 

add_library(ral_nlls_object OBJECT ${source_files})
add_library(ral_nlls-static STATIC $<TARGET_OBJECTS:ral_nlls_object>)
add_library(ral_nlls SHARED $<TARGET_OBJECTS:ral_nlls_object>)

target_link_libraries(ral_nlls ${LIBS} m ${CMAKE_DL_LIBS})
target_include_directories(ral_nlls PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/src)
