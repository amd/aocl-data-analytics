# build example programs
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# build gtests programs
if(BUILD_GTEST)
  add_subdirectory(gtests)
endif()
