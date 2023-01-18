find_program(GCOV_PATH gcov REQUIRED)
find_program(GCOVR_PATH gcovr REQUIRED)

set(COMPILER_FLAGS_DEBUG "${COMPILER_FLAGS_DEBUG};-fprofile-arcs;-ftest-coverage")
link_libraries(gcov)

# create a coverage_reports directory in the main build dir
set(COV_DIR ${CMAKE_BINARY_DIR}/coverage_reports)
add_custom_target(create_cov_dir ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${COV_DIR})

function(exec_cov_target)
    set(REPORT_DEP "")
    foreach(EXEC ${ARGN})
        cmake_path(GET EXEC FILENAME EXE_NAME)
        list(APPEND REPORT_DEP ${EXE_NAME})
        set(EXEC_TARGET_name "run_exe_${EXE_NAME}")
        add_custom_target(
            ${EXEC_TARGET_name} ALL
            DEPENDS ${EXE_NAME}
            COMMAND "${EXEC}" > /dev/null
        )
    endforeach()
    add_custom_target(cov_report ALL
        DEPENDS "${REPORT_DEP}"
        COMMAND ${GCOVR_PATH} --html ${COV_DIR}/index.html --html-details -r ${CMAKE_SOURCE_DIR} -e .*_deps*. .*/build/.*
    )
    
endfunction()