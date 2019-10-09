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
////////////////////////////////////////////////////////////////////////////////

#define LBANN_ELU_LAYER_INSTANTIATE
#include "lbann/layers/activations/elu.hpp"

namespace lbann {

namespace {

/** CUDA kernel for forward prop computation. */
__global__ void fp_kernel(DataType alpha,
                          El::Int height,
                          El::Int width,
                          const DataType* __restrict__ input,
                          El::Int input_ldim,
                          DataType* __restrict__ output,
                          El::Int output_ldim) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int size = height * width;
  const El::Int num_threads = blockDim.x * gridDim.x;
  for (El::Int pos = gid; pos < size; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& x = input[row + col * input_ldim];
    auto& y = output[row + col * output_ldim];
    y = (x > DataType(0)) ? x : alpha * cuda::expm1(x);
  }
}

/** CUDA kernel for backprop computation. */
__global__ void bp_kernel(DataType alpha,
                          El::Int height,
                          El::Int width,
                          const DataType* __restrict__ input,
                          El::Int input_ldim,
                          const DataType* __restrict__ gradient_wrt_output,
                          El::Int gradient_wrt_output_ldim,
                          DataType* __restrict__ gradient_wrt_input,
                          El::Int gradient_wrt_input_ldim) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int size = height * width;
  const El::Int num_threads = blockDim.x * gridDim.x;
  for (El::Int pos = gid; pos < size; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& x = input[row + col * input_ldim];
    const auto& dy = gradient_wrt_output[row + col * gradient_wrt_output_ldim];
    auto& dx = gradient_wrt_input[row + col * gradient_wrt_input_ldim];
    dx = (x > DataType(0)) ? dy : dy * alpha * cuda::exp(x);
  }
}

/** Local forward prop computation. */
void local_fp(DataType alpha,
              const AbsMat& input,
              AbsMat& output) {

  // Get CUDA grid dimensions
  // Note: Maximum CUDA grid dimension is 2^32-1
  // (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications).
  const El::Int height = input.Height();
  const El::Int width = input.Width();
  const El::Int block_dim = 256;
  El::Int grid_dim = (height * width + block_dim - 1) / block_dim;
  if (sizeof(El::Int) > sizeof(unsigned int)
      && grid_dim > std::numeric_limits<uint32_t>::max()) {
    grid_dim = std::numeric_limits<uint32_t>::max();
  }

  // Launch CUDA kernel
  if (grid_dim > 0) {
    fp_kernel<<<grid_dim, block_dim, 0, El::GPUManager::Stream()>>>(
      alpha, height, width,
      input.LockedBuffer(), input.LDim(),
      output.Buffer(), output.LDim());
  }

}

/** Local backprop computation. */
void local_bp(DataType alpha,
              const AbsMat& input,
              const AbsMat& gradient_wrt_output,
              AbsMat& gradient_wrt_input) {

  // Get CUDA grid dimensions
  // Note: Maximum CUDA grid dimension is 2^32-1
  // (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications).
  const El::Int height = input.Height();
  const El::Int width = input.Width();
  const El::Int block_dim = 256;
  El::Int grid_dim = (height * width + block_dim - 1) / block_dim;
  if (sizeof(El::Int) > sizeof(unsigned int)
      && grid_dim > std::numeric_limits<uint32_t>::max()) {
    grid_dim = std::numeric_limits<uint32_t>::max();
  }

  // Launch CUDA kernel
  if (grid_dim > 0) {
    bp_kernel<<<grid_dim, block_dim, 0, El::GPUManager::Stream()>>>(
      alpha, height, width,
      input.LockedBuffer(), input.LDim(),
      gradient_wrt_output.LockedBuffer(), gradient_wrt_output.LDim(),
      gradient_wrt_input.Buffer(), gradient_wrt_input.LDim());
  }

}

} // namespace

template <>
void elu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
       ::fp_compute() {
  local_fp(m_alpha,
           get_local_prev_activations(),
           get_local_activations());
}
template <>
void elu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
     ::bp_compute() {
  local_bp(m_alpha,
           get_local_prev_activations(),
           get_local_prev_error_signals(),
           get_local_error_signals());
}
template <>
void elu_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>
       ::fp_compute() {
  local_fp(m_alpha,
           get_local_prev_activations(),
           get_local_activations());
}
template <>
void elu_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>
     ::bp_compute() {
  local_bp(m_alpha,
           get_local_prev_activations(),
           get_local_prev_error_signals(),
           get_local_error_signals());
}

template class elu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>;
template class elu_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>;

} // namespace lbann
