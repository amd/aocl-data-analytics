# Build the double version
add_compile_definitions(SINGLE_PREC)
add_library(lbfgsb_s ${LBFGSB_SRC})
target_link_directories(lbfgsb_s PUBLIC "$ENV{IFORT_LIBS}")
target_link_libraries(lbfgsb_s ${BLIS})
if (NOT WIN32)
target_link_libraries(lbfgsb_s -lgfortran)
endif()