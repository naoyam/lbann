#!/usr/bin/env bash

set -e

BUILD_SUFFIX=

while :; do
    case ${1} in
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

set -u

BUILD_TYPE=Release

# Compilers
C_COMPILER=$(which gcc)
CXX_COMPILER=$(which g++)
Fortran_COMPILER=$(which gfortran)
# Set environment variables
CC=${C_COMPILER}
CXX=${CXX_COMPILER}
# Get MPI compilers
export MPI_C_COMPILER=$(which mpicc)
export MPI_CXX_COMPILER=$(which mpicxx)
export MPI_Fortran_COMPILER=$(which mpifort)

# Aluminum
WITH_ALUMINUM=ON
ALUMINUM_WITH_NCCL=ON
ALUMINUM_WITH_MPI_CUDA=OFF

# CUDNN
CUDNN_DIR=/usr/workspace/wsb/brain/cudnn/cudnn-7.6.4/cuda-10.1_ppc64le
export CUDNN_DIR

# NCCL
NCCL_DIR=/usr/workspace/wsb/brain/nccl2/nccl_2.4.2-1+cuda10.1_ppc64le
export NCCL_DIR

# NVSHMEM
WITH_NVSHMEM=ON
NVSHMEM_VERSION=0.3.3
NVSHMEM_DIR=/usr/workspace/wsb/brain/nvshmem/nvshmem_0.3.3/cuda-10.1_ppc64le

################################################################
# Build LBANN
################################################################

# Get LBANN root directory
ROOT_DIR=$(realpath $(dirname $0)/..)
# Initialize build directory
BUILD_DIR=${ROOT_DIR}/build/${BUILD_TYPE}
if [ -n "${BUILD_SUFFIX}" ]; then
    BUILD_DIR=${BUILD_DIR}.${BUILD_SUFFIX}
fi
mkdir -p ${BUILD_DIR}
INSTALL_DIR=${BUILD_DIR}/install
mkdir -p ${INSTALL_DIR}
SUPERBUILD_DIR="${ROOT_DIR}/superbuild"

# Work in build directory
pushd ${BUILD_DIR}

# Setup the CMake generator
GENERATOR="\"Unix Makefiles\""

# Configure build with CMake
CONFIGURE_COMMAND=$(cat << EOF
cmake \
-G ${GENERATOR} \
-D CMAKE_EXPORT_COMPILE_COMMANDS=ON \
-D CMAKE_BUILD_TYPE=${BUILD_TYPE} \
-D CMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
-D LBANN_SB_BUILD_CEREAL=ON \
-D LBANN_SB_BUILD_CNPY=ON \
-D LBANN_SB_BUILD_HYDROGEN=ON \
-D LBANN_SB_FWD_HYDROGEN_Hydrogen_ENABLE_CUDA=ON \
-D LBANN_SB_BUILD_OPENCV=ON \
-D LBANN_SB_BUILD_JPEG_TURBO=OFF \
-D LBANN_SB_BUILD_PROTOBUF=ON \
-D LBANN_SB_BUILD_CUB=ON \
-D LBANN_SB_BUILD_ALUMINUM=${WITH_ALUMINUM} \
-D ALUMINUM_ENABLE_MPI_CUDA=${ALUMINUM_WITH_MPI_CUDA} \
-D ALUMINUM_ENABLE_NCCL=${ALUMINUM_WITH_NCCL} \
-D LBANN_SB_BUILD_CONDUIT=ON \
-D LBANN_SB_BUILD_HDF5=ON \
-D LBANN_SB_BUILD_LBANN=ON \
-D CMAKE_C_COMPILER=${C_COMPILER} \
-D CMAKE_CXX_COMPILER=${CXX_COMPILER} \
-D CMAKE_Fortran_COMPILER=${Fortran_COMPILER} \
-D LBANN_WITH_CUDA=ON \
-D LBANN_WITH_TBINF=OFF \
-D LBANN_WITH_TOPO_AWARE=OFF \
-D LBANN_DATATYPE=float \
-D LBANN_WITH_ALUMINUM=${WITH_ALUMINUM} \
-D LBANN_SB_BUILD_CATCH2=ON \
-D LBANN_WITH_NVSHMEM=${WITH_NVSHMEM} \
-D LBANN_WITH_UNIT_TESTING=OFF \
-D NVSHMEM_DIR=${NVSHMEM_DIR} \
${SUPERBUILD_DIR}
EOF
)

echo "${CONFIGURE_COMMAND}" > cmake_superbuild_invocation.txt
eval ${CONFIGURE_COMMAND}
if [ $? -ne 0 ]; then
    echo "--------------------"
    echo "CONFIGURE FAILED"
    echo "--------------------"
    exit 1
fi

# Build LBANN with make
BUILD_COMMAND="make -j VERBOSE=1"
eval ${BUILD_COMMAND}
if [ $? -ne 0 ]; then
    echo "--------------------"
    echo "BUILD FAILED"
    echo "--------------------"
    exit 1
fi
