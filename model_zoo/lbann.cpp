////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
//
// lbann_proto.cpp - prototext application
////////////////////////////////////////////////////////////////////////////////

//#include "lbann/lbann.hpp"

#include "nvshmem.h"
#include "nvshmemx.h"

#include <mpi.h>
#include <cuda.h>

#include <iostream>

#include <cstdlib>

#define CHECK_NVSHMEM(nvshmem_call)                             \
  do {                                                          \
    int status = (nvshmem_call);                                \
    if (status != 0) {                                          \
      std::cerr << "NVSHMEM error (" << status << ")";          \
      std::abort();                                             \
    }                                                           \
  } while (0)

#define CHECK_CUDA(cuda_call)                                   \
  do {                                                          \
    cudaError_t status_CHECK_CUDA = (cuda_call);                \
    if (status_CHECK_CUDA != cudaSuccess) {                     \
      std::cerr << "CUDA error ("                               \
                << cudaGetErrorString(status_CHECK_CUDA)        \
                << ")";                                         \
      std::abort();                                             \
    }                                                           \
  } while (0)

int get_number_of_gpus() {
  int num_gpus = 0;
  CHECK_CUDA(cudaGetDeviceCount(&num_gpus));
  return num_gpus;
}

int get_local_rank() {
  char *env = std::getenv("MV2_COMM_WORLD_LOCAL_RANK");
  if (!env) env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (!env) env = std::getenv("SLURM_LOCALID");
  if (!env) {
    std::cerr << "Can't determine local rank\n";
    abort();
  }
  return std::atoi(env);
}

int get_local_size() {
  char *env = std::getenv("MV2_COMM_WORLD_LOCAL_SIZE");
  if (!env) env = std::getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
  if (!env) env = std::getenv("SLURM_TASKS_PER_NODE");  
  if (!env) {
    std::cerr << "Can't determine local size\n";
    abort();
  }
  return std::atoi(env);
}

int choose_gpu() {
  int num_gpus = get_number_of_gpus();
  int local_rank = get_local_rank();
  int local_size = get_local_size();
  if (num_gpus < local_size) {
    std::cerr << "Warning: Number of GPUs, " << num_gpus
              << " is smaller than the number of local MPI ranks, "
              << local_size << std::endl;
  }
  int gpu = local_rank % num_gpus;
  return gpu;
}

int main(int argc, char *argv[]) {
  std::cerr << "Starting LBANN" << std::endl;

  auto dev_id = choose_gpu();
  CHECK_CUDA(cudaSetDevice(dev_id));
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    std::cerr << "MPI_THREAD_MULTIPLE not provided" << std::endl;
    std::abort();
  }
  std::cerr << "Using device " << dev_id << std::endl;
  std::cerr << "Initializing NVSHMEM" << std::endl;

  MPI_Comm comm = MPI_COMM_WORLD;

  nvshmemx_init_attr_t attr;
  attr.mpi_comm = &comm;
  CHECK_NVSHMEM(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr));

  std::cerr << "NVSHMEM initialized" << std::endl;
  return EXIT_SUCCESS;
}
