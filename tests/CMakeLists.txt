# Build example programs
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# Build gtests programs
if(BUILD_GTEST)
  add_subdirectory(unit_tests)
endif()
