# ##############################################################################
# Copyright (c) 2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ##############################################################################

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(aoclda-doc)

find_package(Doxygen REQUIRED)
find_package(LATEX REQUIRED)
find_program(SED NAMES sed REQUIRED)
find_program(GREP NAMES grep REQUIRED)
find_program(SORT NAMES sort REQUIRED)

# Check for python and the documentation packages: - breathe -
# sphinxcontrib.bibtex
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import breathe"
  RESULT_VARIABLE EXIT_CODE
  OUTPUT_QUIET)
if(NOT ${EXIT_CODE} EQUAL 0)
  message(
    FATAL_ERROR
      "Python3 package breathe is required to build the documentation\n   Use: ${Python3_EXECUTABLE} -m pip install breathe"
  )
endif()
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sphinxcontrib.bibtex"
  RESULT_VARIABLE EXIT_CODE
  OUTPUT_QUIET)
if(NOT ${EXIT_CODE} EQUAL 0)
  message(
    FATAL_ERROR
      "Python3 package sphinxcontrib.bibtex is required to build the documentation\n   Use: ${Python3_EXECUTABLE} -m pip install sphinxcontrib.bibtex"
  )
endif()

message(STATUS "Building AOCL-DA documentation version ${AOCLDA_VERSION}")

string(TIMESTAMP BUILD_DATE "%d %b %Y")

# This target makes sure all latex packaged are available prior to launching the
# full build build a here-document from the main doc which takes all required
# packages NOTE: this here-document needs to be kept in sync with refman.tex
# TODO: Automate the generation of the here-document
add_custom_target(
  latexdeps.pdf
  COMMAND
    bash -c
    [[ echo -e '\\documentclass[twoside]{book}\n\\usepackage{ifthen}\n\\ifx\\requestedLaTeXdate\\undefined\n\\usepackage{array}\n\\else\n\\usepackage{array}[=2016-10-06]\n\\fi\n\\usepackage{fixltx2e}\n%\\usepackage{doxygen}\n\\usepackage{graphicx}\n\\usepackage[utf8]{inputenc}\n\\usepackage{makeidx}\n\\PassOptionsToPackage{warn}{textcomp}\n\\usepackage{textcomp}\n\\usepackage[nointegrals]{wasysym}\n\\usepackage{ifxetex}\n\\usepackage[T1]{fontenc}\n\\usepackage[scaled=.90]{helvet}\n\\usepackage{courier}\n\\usepackage{sectsty}\n\\usepackage{geometry}\n\\usepackage{fancyhdr}\n\\usepackage{natbib}\n\\usepackage[titles]{tocloft}\n\\usepackage{amsmath}\n\\usepackage{newunicodechar}\n\\ifpdf\n\\usepackage[pdftex,pagebackref=true]{hyperref}\n\\else\n\\ifxetex\n\\usepackage[pagebackref=true]{hyperref}\n\\else\n\\usepackage[ps2pdf,pagebackref=true]{hyperref}\n\\fi\n\\fi\n\\usepackage{etoc}\n\\begin{document}\n\\end{document}' > latexdeps.tex ]]
  COMMAND
    bash -c
    [[ pdflatex -halt-on-error -interaction nonstopmode latexdeps.tex ; if [ $? -ne 0 ] ; then echo -e '\n! LaTeX Error: Could not check for the required dependencies. Check latexdeps.log for\nmore details and make sure your LaTeX distribution has the following packages:' ; grep usepackage latexdeps.tex | sed 's/^ *\\usepackage.*{\(.*\)}.*$/\1/' | sort -u  ; echo ; exit 1 ; fi ]]
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  COMMENT "Checking for LaTeX dependencies..."
  VERBATIM
  SOURCES Doxyfile)

# This target does the following 1. Bump version in configuration file: replace
# PORJECT_NAME = .* for the string "version ${AOCLDA_VERSION} (date)" OR (git
# commit hash?) 2. Run doxygen on the sources 3. Run pdflatex 4. Move and rename
# the pdf file
add_custom_target(
  doc
  COMMAND
    ${SED} -i -e
    "s/^\\(PROJECT_NUMBER[[:space:]]*=\\).*$/\\1 \"version ${AOCLDA_VERSION} (${BUILD_DATE})\"/"
    -e "s+^\\(OUTPUT_DIRECTORY[[:space:]]*=\\).*$+\\1 ${PROJECT_BINARY_DIR}+"
    Doxyfile && cd ../source && doxygen ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile &&
    cd ${PROJECT_BINARY_DIR}/latex/ && make && cp -v refman.pdf
    ${CMAKE_CURRENT_BINARY_DIR}/AOCL-DA_Documentation_${AOCLDA_VERSION}.pdf
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Building documentation (Doxygen XML, no PDF) from source"
  VERBATIM
  SOURCES Doxyfile
  DEPENDS latexdeps.pdf)

add_custom_target(
  clean-doc
  COMMAND cd ${PROJECT_BINARY_DIR} && rm -vrf html/ latex/ rtf/ xml/ sphinx/
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Cleaning temporary documentation files"
  VERBATIM)

add_dependencies(cleanall clean-doc)

# Build sphinx Documentation DOC_TYPE controls what document to produce and can
# be set to either pdf or html, default is html
#
# Inspired from Sphinx in CMake section of MSFT blog post
# https://devblogs.microsoft.com/cppblog/clear-functional-c-documentation-with-sphinx-breathe-doxygen-cmake/
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Add Sphinx doc target
find_package(Sphinx REQUIRED)

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)
set(PDF_DOC_NAME AOCL-DA_Documentation_${AOCLDA_VERSION})
set(SPHINX_PDF ${SPHINX_BUILD}/${PDF_DOC_NAME})

if(NOT DOC_TYPE)
  message(STATUS "Setting the default documentation to html")
  set(DOC_TYPE
      "html"
      CACHE STRING "Choose the documentation format" FORCE)
endif()

if(${DOC_TYPE} MATCHES "pdf")
  set(SPHINX_FORMAT "latex")
  set(SPHINX_DEP "${SPHINX_PDF}")
else()
  set(SPHINX_FORMAT "html")
  set(SPHINX_DEP "${SPHINX_INDEX_FILE}")
endif()

# Target building the doxygen XML Also modifies the output directory and the
# version number in the Doxyfile
add_custom_target(
  doxygen_xml
  COMMAND
    ${SED} -i -e
    "s/^\\(PROJECT_NUMBER[[:space:]]*=\\).*$/\\1 \"version ${AOCLDA_VERSION} (${BUILD_DATE})\"/"
    -e "s+^\\(OUTPUT_DIRECTORY[[:space:]]*=\\).*$+\\1 ${PROJECT_BINARY_DIR}+"
    Doxyfile
  COMMAND doxygen ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  VERBATIM
  SOURCES Doxyfile
  DEPENDS latexdeps.pdf
  COMMENT "Building doxygen xml files")

# Build the sphinx documentation html or latex depending on which one is
# required
add_custom_command(
  OUTPUT ${SPHINX_INDEX_FILE}
  COMMAND
    ${SPHINX_EXECUTABLE} -b ${SPHINX_FORMAT}
    # Tell Breathe where to find the Doxygen output
    -Dbreathe_projects.aocl-da=${DOXYGEN_OUTPUT_DIR}/xml ${SPHINX_SOURCE}
    ${SPHINX_BUILD}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/index.rst doxygen_xml
  MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
  COMMENT "Generating (Sphinx HTML) documentation from source")

# run pdflatex on the tex file produced by sphinx if DOC_TYPE was set to pdf
add_custom_command(
  OUTPUT ${SPHINX_PDF}
  COMMAND pdflatex -jobname=${PDF_DOC_NAME} ${SPHINX_BUILD}/aocl-da.tex
  WORKING_DIRECTORY ${SPHINX_BUILD}
  DEPENDS ${SPHINX_INDEX_FILE})

# Nice named target so we can run the job easily
add_custom_target(sphinx DEPENDS ${SPHINX_DEP})

# Courtesy "Release Notes" section in the doc
write_file(${PROJECT_BINARY_DIR}/ReleaseNotes.md "# Release Notes\n\n")
write_file(${PROJECT_BINARY_DIR}/ReleaseNotes.md
           "AOCL-DA version: ${AOCLDA_VERSION}\n" APPEND)
write_file(${PROJECT_BINARY_DIR}/ReleaseNotes.md
           "OS: ${CMAKE_CXX_PLATFORM_ID}\n" APPEND)
write_file(
  ${PROJECT_BINARY_DIR}/ReleaseNotes.md
  "compiler: ${CMAKE_CXX_COMPILER}: ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER_VERSION})\n"
  APPEND)
write_file(
  ${PROJECT_BINARY_DIR}/ReleaseNotes.md
  "compiler: ${CMAKE_Fortran_COMPILER}: ${CMAKE_Fortran_COMPILER_ID} (${CMAKE_Fortran_COMPILER_VERSION})\n"
  APPEND)
write_file(${PROJECT_BINARY_DIR}/ReleaseNotes.md
           "Flags: <see ${CMAKE_BINARY_DIR}/compile_commands.json>\n" APPEND)
