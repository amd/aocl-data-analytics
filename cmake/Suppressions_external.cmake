# lbfgsb
set_source_files_properties(
  lbfgsb.f90 linpack.f90 DIRECTORY double_cmake single_cmake
  PROPERTIES
    COMPILE_FLAGS
    "-Wno-unused-variable -Wno-compare-reals -Wno-unused-dummy-argument")
