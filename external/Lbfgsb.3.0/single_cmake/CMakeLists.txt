# Build the double version
add_compile_definitions(SINGLE_PREC)
add_library(lbfgsb_s ${LBFGSB_SRC})
target_link_libraries(lbfgsb_s -lgfortran ${BLIS})