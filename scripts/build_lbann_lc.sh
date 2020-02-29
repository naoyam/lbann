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

BUILD_TYPE=Release
C_FLAGS=
CXX_FLAGS=
Fortran_FLAGS=
DATATYPE=float
VERBOSE=0

C_COMPILER=$(which gcc)
CXX_COMPILER=$(which g++)
Fortran_COMPILER=$(which gfortran)

# C_FLAGS="${C_FLAGS} -O3 -fno-omit-frame-pointer"
# CXX_FLAGS="${CXX_FLAGS} -O3  -fno-omit-frame-pointer"
# Fortran_FLAGS="${Fortran_FLAGS} -O3"
# C_FLAGS="${C_FLAGS} -mcpu=power9 -mtune=power9"
# CXX_FLAGS="${CXX_FLAGS} -mcpu=power9 -mtune=power9"
# Fortran_FLAGS="${Fortran_FLAGS} -mcpu=power9 -mtune=power9"

# Add flag for libldl: may be needed some compilers
#CXX_FLAGS="${CXX_FLAGS} -ldl"
#C_FLAGS="${CXX_FLAGS}"

# Set environment variables
CC=${C_COMPILER}
CXX=${CXX_COMPILER}

################################################################
# Initialize directories
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

# Get MPI compilers
export MPI_C_COMPILER=$(which mpicc)
export MPI_CXX_COMPILER=$(which mpicxx)
export MPI_Fortran_COMPILER=$(which mpifort)

# GPU
WITH_CUDA=ON
WITH_CUB=ON

# Aluminum
WITH_ALUMINUM=ON
ALUMINUM_WITH_NCCL=ON
ALUMINUM_WITH_MPI_CUDA=OFF

WITH_NVSHMEM=ON

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
# Build LBANN
################################################################

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
-D LBANN_SB_BUILD_CONDUIT=ON \
-D LBANN_SB_BUILD_HDF5=ON \
-D LBANN_SB_BUILD_LBANN=ON \
-D CMAKE_CXX_FLAGS="${CXX_FLAGS}" \
-D CMAKE_C_FLAGS="${C_FLAGS}" \
-D CMAKE_C_COMPILER=${C_COMPILER} \
-D CMAKE_CXX_COMPILER=${CXX_COMPILER} \
-D CMAKE_Fortran_COMPILER=${Fortran_COMPILER} \
-D LBANN_WITH_CUDA=${WITH_CUDA} \
-D LBANN_WITH_TBINF=OFF \
-D LBANN_WITH_TOPO_AWARE=OFF \
-D LBANN_DATATYPE=${DATATYPE} \
-D LBANN_WITH_ALUMINUM=${WITH_ALUMINUM} \
-D LBANN_SB_BUILD_CATCH2=ON \
-D LBANN_NO_OMP_FOR_DATA_READERS=${NO_OMP_FOR_DATA_READERS} \
-D OPENBLAS_ARCH_COMMAND=${OPENBLAS_ARCH} \
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

# Build LBANN with make
BUILD_COMMAND="make -j VERBOSE=${VERBOSE}"
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
