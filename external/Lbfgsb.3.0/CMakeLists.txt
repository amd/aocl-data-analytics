enable_language(Fortran)
if (NOT WIN32)
  set(CMAKE_Fortran_FLAGS -cpp)
else()
  set(CMAKE_Fortran_PREPROCESS ON)
endif()

set(LBFGSB_SRC ../working_precision.f90 ../linpack.f90 ../timer.f90 ../lbfgsb.f90)

# Build the double version
add_subdirectory(double_cmake)

# build the single version
add_subdirectory(single_cmake)

# Suppress warnings
if (NOT WIN32)
  include(${CMAKE_SOURCE_DIR}/cmake/Suppressions_external.cmake)
endif()