#!/usr/bin/env bash

set -e

. /usr/share/lmod/lmod/init/bash

# Detect system parameters
CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')
ARCH=$(uname -m)
CORAL=$([[ $(hostname) =~ (sierra|lassen|ray) ]] && echo 1 || echo 0)

################################################################
# Default options
################################################################

BUILD_TYPE=Release

C_FLAGS=
CXX_FLAGS=-DLBANN_SET_EL_RNG
Fortran_FLAGS=
CLEAN_BUILD=0
DATATYPE=float
VERBOSE=0
CMAKE_INSTALL_MESSAGE=LAZY
MAKE_NUM_PROCESSES=$(($(nproc) + 1))
NINJA_NUM_PROCESSES=0 # Let ninja decide
BUILD_TOOL="make"
BUILD_DIR=
INSTALL_DIR=
BUILD_SUFFIX=
WITH_ALUMINUM=
ALUMINUM_WITH_MPI_CUDA=OFF
ALUMINUM_WITH_NCCL=
WITH_CONDUIT=ON
AVOID_CUDA_AWARE_MPI=OFF
WITH_NVSHMEM=OFF
USE_NINJA=0

################################################################
# Parse command-line arguments
################################################################

while :; do
    case ${1} in
        -h|--help)
            # Help message
            help_message
            exit 0
            ;;
         --suffix)
            # Specify suffix for build directory
            if [ -n "${2}" ]; then
                BUILD_SUFFIX=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
         --ninja)
            USE_NINJA=1
            BUILD_TOOL="ninja"
            ;;
         -v|--verbose)
            # Verbose output
            VERBOSE=1
            CMAKE_INSTALL_MESSAGE=ALWAYS
            ;;
         --disable-aluminum)
            WITH_ALUMINUM=OFF
            ;;
        --disable-aluminum-with-nccl)
            ALUMINUM_WITH_NCCL=OFF
            ;;
        --with-nvshmem)
            WITH_NVSHMEM=ON
            ;;
        -?*)
            # Unknown option
            echo "Unknown option (${1})" >&2
            exit 1
            ;;
        *)
            # Break loop if there are no more options
            break
    esac
    shift
done

USE_MODULES=1

C_COMPILER=$(which gcc)
CXX_COMPILER=$(which g++)
Fortran_COMPILER=$(which gfortran)

C_FLAGS="${C_FLAGS} -O3 -fno-omit-frame-pointer"
CXX_FLAGS="${CXX_FLAGS} -O3  -fno-omit-frame-pointer"
Fortran_FLAGS="${Fortran_FLAGS} -O3"
C_FLAGS="${C_FLAGS} -mcpu=power9 -mtune=power9"
CXX_FLAGS="${CXX_FLAGS} -mcpu=power9 -mtune=power9"
Fortran_FLAGS="${Fortran_FLAGS} -mcpu=power9 -mtune=power9"

# Add flag for libldl: may be needed some compilers
CXX_FLAGS="${CXX_FLAGS} -ldl"
C_FLAGS="${CXX_FLAGS}"

# Set environment variables
CC=${C_COMPILER}
CXX=${CXX_COMPILER}

################################################################
# Initialize directories
################################################################

# Get LBANN root directory
ROOT_DIR=$(realpath $(dirname $0)/..)

# Initialize build directory
if [ -z "${BUILD_DIR}" ]; then
    BUILD_DIR=${ROOT_DIR}/build/${BUILD_TYPE}.${CLUSTER}.llnl.gov
fi
if [ -n "${BUILD_SUFFIX}" ]; then
    BUILD_DIR=${BUILD_DIR}.${BUILD_SUFFIX}
fi
mkdir -p ${BUILD_DIR}

# Initialize install directory
if [ -z "${INSTALL_DIR}" ]; then
    INSTALL_DIR=${BUILD_DIR}/install
fi
mkdir -p ${INSTALL_DIR}

SUPERBUILD_DIR="${ROOT_DIR}/superbuild"


# Get MPI compilers
export MPI_C_COMPILER=$(which mpicc)
export MPI_CXX_COMPILER=$(which mpicxx)
export MPI_Fortran_COMPILER=$(which mpifort)
WITH_SPECTRUM=ON

################################################################
# Initialize GPU libraries
################################################################

HAS_GPU=1
WITH_CUDA=ON
WITH_CUDNN=ON
WITH_CUB=${WITH_CUB:-ON}
WITH_ALUMINUM=${WITH_ALUMINUM:-ON}
ALUMINUM_WITH_NCCL=${ALUMINUM_WITH_NCCL:-ON}
if [[ ${CORAL} -eq 1 ]]; then
	module del cuda
	CUDA_TOOLKIT_MODULE=${CUDA_TOOLKIT_MODULE:-cuda/10.1.243}
fi

# Defines CUDA_TOOLKIT_ROOT_DIR
if [ -z "${CUDA_TOOLKIT_ROOT_DIR}" ]; then
	if [ -n "${CUDA_PATH}" ]; then
		CUDA_TOOLKIT_ROOT_DIR=${CUDA_PATH}
	elif [ -n "${CUDA_HOME}" ]; then
		CUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME}
	elif [ -n "${CUDA_TOOLKIT_MODULE}" -o ${USE_MODULES} -ne 0 ]; then
		CUDA_TOOLKIT_MODULE=${CUDA_TOOLKIT_MODULE:-cuda}
		module load ${CUDA_TOOLKIT_MODULE}
		CUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME:-${CUDA_PATH}}
	fi
fi
#export CUDA_TOOLKIT_ROOT_DIR

# CUDNN
CUDNN_DIR=/usr/workspace/wsb/brain/cudnn/cudnn-7.6.4/cuda-10.1_ppc64le
export CUDNN_DIR

# NCCL
NCCL_DIR=/usr/workspace/wsb/brain/nccl2/nccl_2.4.2-1+cuda10.1_ppc64le
export NCCL_DIR

# NVSHMEM
NVSHMEM_VERSION=0.3.3
NVSHMEM_DIR=/usr/workspace/wsb/brain/nvshmem/nvshmem_0.3.3/cuda-10.1_ppc64le

CMAKE_BLAS_OPTIONS=""
OPENBLAS_ARCH=""

################################################################
# Setup Ninja, if using
################################################################
if [ ${USE_NINJA} -ne 0 ]; then
    if ! which ninja ; then
        if [ "${ARCH}" == "x86_64" ]; then
            export PATH=/usr/workspace/wsb/brain/utils/toss3/ninja/bin:$PATH
        elif [ "${ARCH}" == "ppc64le" ]; then
            export PATH=/usr/workspace/wsb/brain/utils/coral/ninja/bin:$PATH
        fi
    fi
    if ! which ninja ; then
        USE_NINJA=0
    fi
fi

################################################################
# Build LBANN
################################################################

# Work in build directory
pushd ${BUILD_DIR}

# Clean up build directory
if [ ${CLEAN_BUILD} -ne 0 ]; then
    CLEAN_COMMAND="rm -rf ${BUILD_DIR}/*"
    if [ ${VERBOSE} -ne 0 ]; then
        echo "${CLEAN_COMMAND}"
    fi
    eval ${CLEAN_COMMAND}
fi

if [[ ((${BUILD_TOOL} == "make" && -f ${BUILD_DIR}/lbann/build/Makefile) ||
       (${BUILD_TOOL} == "ninja" && -f ${BUILD_DIR}/lbann/build/build.ninja))
      && (${RECONFIGURE} != 1) ]]; then
    echo "Building previously configured LBANN"
    cd ${BUILD_DIR}/lbann/build/
    ${BUILD_TOOL} -j${MAKE_NUM_PROCESSES} all
    ${BUILD_TOOL} install -j${MAKE_NUM_PROCESSES} all
    exit $?
fi

# Setup the CMake generator
GENERATOR="\"Unix Makefiles\""
if [ ${USE_NINJA} -ne 0 ]; then
    GENERATOR="Ninja"
fi

# Configure build with CMake
CONFIGURE_COMMAND=$(cat << EOF
cmake \
-G ${GENERATOR} \
-D CMAKE_EXPORT_COMPILE_COMMANDS=ON \
-D CMAKE_BUILD_TYPE=${BUILD_TYPE} \
-D CMAKE_INSTALL_MESSAGE=${CMAKE_INSTALL_MESSAGE} \
-D CMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
-D LBANN_SB_BUILD_CEREAL=ON \
-D CMAKE_CUDA_FLAGS_DEBUG="-G" \
-D LBANN_SB_BUILD_CNPY=ON \
-D LBANN_SB_BUILD_HYDROGEN=ON \
-D LBANN_SB_FWD_HYDROGEN_Hydrogen_ENABLE_CUDA=${WITH_CUDA} \
-D LBANN_SB_BUILD_OPENCV=ON \
-D LBANN_SB_BUILD_JPEG_TURBO=OFF \
-D LBANN_SB_BUILD_PROTOBUF=ON \
-D LBANN_SB_BUILD_CUB=${WITH_CUB} \
-D LBANN_SB_BUILD_ALUMINUM=${WITH_ALUMINUM} \
-D ALUMINUM_ENABLE_MPI_CUDA=${ALUMINUM_WITH_MPI_CUDA} \
-D ALUMINUM_ENABLE_NCCL=${ALUMINUM_WITH_NCCL} \
-D LBANN_SB_BUILD_CONDUIT=${WITH_CONDUIT} \
-D LBANN_SB_BUILD_HDF5=${WITH_CONDUIT} \
-D LBANN_SB_BUILD_LBANN=ON \
-D CMAKE_CXX_FLAGS="${CXX_FLAGS}" \
-D CMAKE_C_FLAGS="${C_FLAGS}" \
-D CMAKE_C_COMPILER=${C_COMPILER} \
-D CMAKE_CXX_COMPILER=${CXX_COMPILER} \
-D CMAKE_Fortran_COMPILER=${Fortran_COMPILER} \
-D LBANN_WITH_CUDA=${WITH_CUDA} \
-D LBANN_WITH_NVPROF=OFF \
-D LBANN_WITH_TBINF=OFF \
-D LBANN_WITH_TOPO_AWARE=OFF \
-D LBANN_DATATYPE=${DATATYPE} \
-D LBANN_WITH_ALUMINUM=${WITH_ALUMINUM} \
-D LBANN_SB_BUILD_CATCH2=ON \
-D LBANN_NO_OMP_FOR_DATA_READERS=${NO_OMP_FOR_DATA_READERS} \
-D LBANN_CONDUIT_DIR=${CONDUIT_DIR} \
-D LBANN_BUILT_WITH_SPECTRUM=${WITH_SPECTRUM} \
-D OPENBLAS_ARCH_COMMAND=${OPENBLAS_ARCH} \
-D LBANN_SB_BUILD_DIHYDROGEN=OFF \
-D LBANN_WITH_DISTCONV=OFF \
-D LBANN_SB_BUILD_P2P=OFF \
-D LBANN_WITH_P2P=OFF \
-D LBANN_WITH_NVSHMEM=${WITH_NVSHMEM} \
-D NVSHMEM_DIR=${NVSHMEM_DIR} \
-D LBANN_SB_FWD_ALUMINUM_ALUMINUM_ENABLE_STREAM_MEM_OPS=OFF \
-D LBANN_SB_FWD_ALUMINUM_ALUMINUM_HT_USE_PASSTHROUGH=OFF \
${CMAKE_BLAS_OPTIONS} \
${SUPERBUILD_DIR}
EOF
)

if [ ${VERBOSE} -ne 0 ]; then
    echo "${CONFIGURE_COMMAND}" |& tee cmake_superbuild_invocation.txt
else
    echo "${CONFIGURE_COMMAND}" > cmake_superbuild_invocation.txt
fi
eval ${CONFIGURE_COMMAND}
if [ $? -ne 0 ]; then
    echo "--------------------"
    echo "CONFIGURE FAILED"
    echo "--------------------"
    exit 1
fi

BUILD_OPTIONS="-j${MAKE_NUM_PROCESSES}"
if [ ${VERBOSE} -ne 0 ]; then
  if [ "${BUILD_TOOL}" == "ninja" ]; then
      BUILD_OPTIONS+=" -v"
  else
      BUILD_OPTIONS+=" VERBOSE=${VERBOSE}"
  fi
fi

# Build LBANN with make
# Note: Ensure Elemental to be built before LBANN. Dependency violation appears to occur only when using cuda_add_library.
BUILD_COMMAND="make -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE}"
if [ ${USE_NINJA} -ne 0 ]; then
    if [ ${NINJA_NUM_PROCESSES} -ne 0 ]; then
        BUILD_COMMAND="ninja -j${NINJA_NUM_PROCESSES}"
    else
        # Usually equivalent to -j<num_cpus+2>
        BUILD_COMMAND="ninja"
    fi
fi
if [ ${VERBOSE} -ne 0 ]; then
    echo "${BUILD_COMMAND}"
fi
eval ${BUILD_COMMAND}
if [ $? -ne 0 ]; then
    echo "--------------------"
    echo "BUILD FAILED"
    echo "--------------------"
    exit 1
fi
