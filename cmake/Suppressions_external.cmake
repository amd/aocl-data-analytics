# lbfgsb
set_source_files_properties(
  lbfgsb.F90 linpack.F90 DIRECTORY double_cmake single_cmake
  PROPERTIES
    COMPILE_FLAGS
    "-Wno-unused-variable -Wno-compare-reals -Wno-unused-dummy-argument -Wno-maybe-uninitialized")
