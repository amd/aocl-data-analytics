#!/bin/bash

# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# 

USAGE="$(basename "$0") [options]

--help            - Print this message
--debug           - Set the build type to Debug. Release by default
--asan            - Turn ON ASAN tool. OFF by default
--ilp64           - Use 64 bits integer. 32 bits by default
--coverage        - Enable coverage utilities for AOCL-DA builds. OFF by default
--no_unit_tests   - Disable the unit tests. ON by defaults
--valgrind        - Enable valgrind utilities. OFF by default
--shared          - build shared libraries. OFF by default
--omp             - Build SMP library. OFF by default
--target target   - Secify which target to build. all by default.
--build folder    - Spcify the build folder. build by default
--compiler comp   - Secify the compiler to use, can be gnu or aocc. gcc by default
--gnu_version     - Specify the version of the gcc compiler to load from spack. Must be provided if --spack_path is set
--aocc_path path  - Specify a path where the aocc envirenment script is install. 
                   Either the option or the environment variable AOCC_PATH must be set if using --compiler aocc
--aocl_path       - Specify the path to AOCL libraries. take ACOL_PATH enviorment variable if not set.
--spack_path path - Specify the spack path. 
                    Either the option or SPACK_PATH must be set if using the --gnu_version option
--parallel nthreads - Specify the number of threads to use. Default is 1      
--verbose         - Build with verbose compilation commands   
"

# Clean up the existing path
export LD_LIBRARY_PATH=""

# Workspace and build environment
if [ -z ${WORKSPACE+x} ]
then
    WORKSPACE=$(pwd)
fi
BUILD_DIR="build"
THREADS=1

# default compiler
CC=gcc
CXX=g++
FC=gfortran
COMPILER_VERSION=""

# Default values for the cmake variables
BUILD_TYPE="Release"
ASAN="OFF"
ILP64="OFF"
COVERAGE="OFF"
UNIT_TESTS="ON"
VALGRIND="OFF"
TARGET="all"
SHARED="OFF"
VERBOSE=""
OMP="OFF"

while [ "${1:-}" != "" ]
do 
    case $1 in
    "--help")
        echo "${USAGE}"
        exit 1;;
    "--debug")
        BUILD_TYPE="Debug"
        ;;
    "--asan")
        ASAN=ON
        ;;
    "--ilp64")
        ILP64="ON"
        ;;
    "--coverage")
        COVERAGE="ON"
        ;;
    "--no_unit_tests")
        UNIT_TESTS="OFF"
        ;;
    "--valgrind")
        VALGRIND="ON"
        ;;
    "--shared")
        SHARED="ON"
        ;;
    "--omp")
        OMP="ON"
        ;;
    "--verbose")
        VERBOSE="--verbose"
        ;;
    "--target")
        if [[ "$2" == "" ]] || [[ $2 =~ ^-.* ]]
        then
            echo "ERROR: A valid target must be provided after the option --target"
            exit 1
        fi
        TARGET="$2"
        shift 1
        ;;
    "--build")
        if [[ "$2" == "" ]] || [[ $2 =~ ^-.* ]]
        then
            echo "ERROR: A build directory must be provided after the option --build"
            exit 1
        fi
        BUILD_DIR="$2"
        shift 1
        ;;
    "--compiler")
        if [[ "$2" == "" ]] || [[ $2 =~ ^-.* ]]
        then
            echo "ERROR: compiler name must be provided after the option --build"
            exit 1
        fi
        if [[ $2 =~ ^[gnu|gcc|g++].* ]]
        then
            CC=gcc
            CXX=g++
            FC=gfortran
        elif [[ $2 =~ ^[aocc|clang].* ]]
        then
            CC=clang
            CXX=clang++
            FC=flang
        fi
        shift 1
        ;;
    "--gnu_version" | "gcc_version")
        if [[ "$2" == "" ]] || [[ $2 =~ ^-.* ]]
        then
            echo "ERROR: A gnu compiler version must be provided after the option --gcc_version (e.g., 13.1.0)"
            exit 1
        fi
        COMPILER_VERSION="$2"
        shift 1
        ;;
    "--aocc_path")
        if [[ "$2" == "" ]] || [[ $2 =~ ^-.* ]]
        then
            echo "ERROR: A path must be provided after the option --aocc_path"
            exit 1
        fi
        AOCC_INSTALL_PATH="$2"
        shift 1
        ;;
    "--aocl_path")
        if [[ "$2" == "" ]] || [[ $2 =~ ^-.* ]]
        then
            echo "ERROR: A valid path for AOCL libraries must be provided after the option --aocl_path"
            exit 1
        fi
        AOCL_PATH="$2"
        shift 1
        ;;
    "--spack_path")
        if [[ "$2" == "" ]] || [[ $2 =~ ^-.* ]]
        then
            echo "ERROR: A valid path for spack must be provided after the option --spack_path"
            exit 1
        fi
        SPACK_PATH=$2
        shift 1
        ;;
    "--parallel")
        if [[ "$2" == "" ]] || [[ $2 =~ ^-.* ]]
        then
            echo "ERROR: A valid number must be provided after the option --parallel"
            exit 1
        fi
        THREADS=$2
        shift 1
        ;;
    *)
        echo "Unrecognized build option $1. EXITING"
        exit 1
        ;;
    esac
    shift 1
done

# Load compiler and corresponding libraries
if [ ! -z ${SPACK_PATH+x} ]
then
    echo "Loading Spack:"
    echo "    source ${SPACK_PATH}/setup-env.sh"
    source ${SPACK_PATH}/setup-env.sh
fi
if [[ $CC == "gcc" ]]
then
    if ! command -v spack &> /dev/null
    then
        echo "WARNING spack was not found. Using default system gcc."
        echo ""
    else
        echo "Loading compiler"
        echo "    spack load gcc@${COMPILER_VERSION}"
        spack load gcc@${COMPILER_VERSION}
    fi
fi
if [[ $CC == "clang" ]]
then
    if [[ -z ${AOCC_INSTALL_PATH+x} ]]
    then
        echo "the aocc path must be defined if the compiler chosen is aocc. use the option --aocc_path"
    fi
    echo "Setting AOCC environment"
    echo "    source ${AOCC_INSTALL_PATH}/setenv_AOCC.sh"
    source ${AOCC_INSTALL_PATH}/setenv_AOCC.sh
fi

# Load AOCL libraries
if [[ ! -z ${AOCL_PATH+x} ]]
then
    echo "loading AOCL libraries: "
    echo "    source ${AOCL_PATH}/amd-libs.cfg"
    source ${AOCL_PATH}/amd-libs.cfg
fi
if [[ -z ${AOCL_ROOT} ]]
then
    echo "AOCL_ROOT must be defined either by providing a path to amd-libs.cfg script or by setting it up manually"
    exit 1
fi

CMAKE_OPTIONS="-DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DASAN=${ASAN} -DBUILD_ILP64=${ILP64} -DCOVERAGE=${COVERAGE} -DBUILD_GTEST=${UNIT_TESTS} -DVALGRIND=${VALGRIND} -DBUILD_SHARED_LIBS=${SHARED} -DBUILD_SMP=${OMP}"
CMAKE_COMPILERS="-DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_Fortran_COMPILER=${FC}"

echo "Work space     : ${WORKSPACE}"
echo "Build directory: ${BUILD_DIR}"
echo "compilers      : $(which ${CC})"
echo "                 $(which ${CXX})"
echo "                 $(which ${FC})"
echo "cmake options  : ${CMAKE_OPTIONS}"

# checkout the DA repository
cd ${WORKSPACE}
rm -rf aocl-da
rm -rf ${BUILD_DIR}
git clone -b amd-main ssh://gerritgit/cpulibraries/er/aocl-da
cmake S aocl-da -B ${BUILD_DIR} ${CMAKE_COMPILERS} ${CMAKE_OPTIONS}
if [[ $VALGRIND=="ON" ]]
then
    # run cmake configure a second time for the valgrind suppression list to be taken into account
    cmake S aocl-da -B ${BUILD_DIR} ${CMAKE_COMPILERS} ${CMAKE_OPTIONS}
fi
cd ${BUILD_DIR}
if [[ $COVERAGE=="ON" ]]
then
    cmake --build . --target coverage --parallel ${THREADS} 
fi
cmake --build . --target ${TARGET} --parallel ${THREADS} ${VERBOSE}
ctest -j ${THREADS}