#!/bin/bash

# Clean up the existing path
export LD_LIBRARY_PATH=""

# Workspace and build environment
if [ -z ${WORKSPACE+x} ]
then
    WORKSPACE=$(pwd)
fi
BUILD_DIR="build"
JOB_THREADS=5

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

while [ "${1:-}" != "" ]
do 
    case $1 in
    "--help")
        echo "TODO print help"
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
        BUILD_DIR="$2"
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
        JOB_THREADS=$2
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
fi

CMAKE_OPTIONS="-DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DASAN=${ASAN} -DBUILD_ILP64=${ILP64} -DCOVERAGE=${COVERAGE} -DBUILD_GTEST=${UNIT_TESTS} -DVALGRIND=${VALGRIND}"
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
cd ${BUILD_DIR}
cmake --build . --target all --parallel ${JOB_THREADS}
ctest -j ${JOB_THREADS}

