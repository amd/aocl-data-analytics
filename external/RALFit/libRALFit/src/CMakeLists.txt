# Copyright (c) 2016, The Science and Technology Facilities Council (STFC)
# All rights reserved.
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.

# Adapted to build dual precision objects for AOCL-DA

set(source_files
    ral_nlls_types.F90
    ral_nlls_symbols.F90
    ral_nlls_dtrs.F90
    ral_nlls_workspaces.F90
    ral_nlls_printing.F90
    ral_nlls_bounds.F90
    nag_export_mod.F90
    ral_nlls_fd.F90
    ral_nlls_internal.F90
    ral_nlls.F90
    ral_nlls_ciface.F90 CACHE INTERNAL "source_files" FORCE
)

# Suppress warnings
if(NOT WIN32)
    include(${CMAKE_SOURCE_DIR}/cmake/Suppressions_external.cmake)
endif()

# Generate target name based on filename
# (remove file extension and convert to uppercase)
function(gen_name filename precision result)
    get_filename_component(name_without_extension ${filename} NAME_WE)
    string(TOUPPER ${name_without_extension} uppercase_name)
    set(${result} "${uppercase_name}_${precision}" PARENT_SCOPE)
endfunction()


function(generate_ral_nlls_modules precision obj_list)
    set(in_obj_list "")
    foreach(file ${source_files})
        gen_name(${file} ${precision} target)

        add_library(${target} OBJECT ${file})

        if(${precision} STREQUAL "single")
            set(local_defs "-DSINGLE_PRECISION")
            target_compile_definitions(${target} PRIVATE ${local_defs})
        endif()
        list(APPEND in_obj_list $<TARGET_OBJECTS:${target}>)
    endforeach()
    set(obj_list ${in_obj_list} PARENT_SCOPE)

    # Build explicit dependency tree
    add_dependencies("RAL_NLLS_DTRS_${precision}" "RAL_NLLS_TYPES_${precision}" "RAL_NLLS_SYMBOLS_${precision}")
    add_dependencies("RAL_NLLS_WORKSPACES_${precision}" "RAL_NLLS_DTRS_${precision}")
    add_dependencies("NAG_EXPORT_MOD_${precision}" "RAL_NLLS_WORKSPACES_${precision}")
    add_dependencies("RAL_NLLS_PRINTING_${precision}" "RAL_NLLS_WORKSPACES_${precision}")
    add_dependencies("RAL_NLLS_BOUNDS_${precision}" "RAL_NLLS_WORKSPACES_${precision}")
    add_dependencies("RAL_NLLS_FD_${precision}" "RAL_NLLS_WORKSPACES_${precision}" "RAL_NLLS_PRINTING_${precision}")
    add_dependencies("RAL_NLLS_INTERNAL_${precision}" "RAL_NLLS_PRINTING_${precision}" "RAL_NLLS_BOUNDS_${precision}"
        "RAL_NLLS_FD_${precision}" "NAG_EXPORT_MOD_${precision}" "RAL_NLLS_WORKSPACES_${precision}")
    add_dependencies("RAL_NLLS_${precision}" "RAL_NLLS_INTERNAL_${precision}")
    add_dependencies("RAL_NLLS_CIFACE_${precision}" "RAL_NLLS_${precision}")
endfunction()

set(nlls_obj_list "")
# Generate single-precision library
generate_ral_nlls_modules("single" obj_list)
list(APPEND nlls_obj_list ${obj_list})
# Generate double-precision library
generate_ral_nlls_modules("double" obj_list)
list(APPEND nlls_obj_list ${obj_list})
set(nlls_obj_list ${nlls_obj_list} PARENT_SCOPE)
