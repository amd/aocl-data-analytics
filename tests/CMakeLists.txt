# build gtests programs
if (BUILD_GTEST)
  add_subdirectory(gtests)
endif()

# build example programs
if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# transfer definition of the coverage executables to the main cmake file
set(COV_EXECUTABLES ${COV_EXECUTABLES} PARENT_SCOPE)