# Build the double version
remove_definitions(-DSINGLE_PREC)
add_library(lbfgsb ${LBFGSB_SRC})
target_link_directories(lbfgsb PUBLIC "$ENV{IFORT_LIBS}")
target_link_libraries(lbfgsb ${BLIS})
if (NOT WIN32)
target_link_libraries(lbfgsb -lgfortran)
endif()