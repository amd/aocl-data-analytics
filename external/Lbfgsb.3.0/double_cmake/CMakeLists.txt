# Build the double version
remove_definitions(-DSINGLE_PREC)
add_library(lbfgsb ${LBFGSB_SRC})
target_link_libraries(lbfgsb -lgfortran ${BLIS})