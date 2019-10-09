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

#define LBANN_ONE_HOT_LAYER_INSTANTIATE
#include "lbann/layers/misc/one_hot.hpp"

namespace lbann {

namespace {

/**
 *  On input, output is assumed to be filled with zeros.
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimensions: (width / bsize) x 1 x 1
 */
__global__ void fp_kernel(size_t height,
                          size_t width,
                          const DataType* __restrict__ indices,
                          size_t indices_stride,
                          DataType* __restrict__ output,
                          size_t output_ldim) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t nthreads = blockDim.x * gridDim.x;
  for (size_t col = gid; col < width; col += nthreads) {
    const auto& ind = indices[col*indices_stride];
    if (DataType{0} <= ind && ind < DataType(height)) {
      const size_t row = static_cast<size_t>(ind);
      output[row+col*output_ldim] = DataType{1};
    }
  }
}

} // namespace <anon>

template <>
void one_hot_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
     ::fp_compute() {

  // Local matrices
  const auto& local_input = dynamic_cast<const GPUMat&>(get_local_prev_activations());
  auto& local_output = dynamic_cast<GPUMat&>(get_local_activations());

  // Populate one-hot vectors
  El::Zero(local_output);
  if (!local_output.IsEmpty()) {
    const size_t local_height = local_output.Height();
    const size_t local_width = local_output.Width();
    constexpr size_t block_size = 64;
    const size_t grid_size = (local_width + block_size - 1) / block_size;
    fp_kernel
      <<<grid_size, block_size, 0, El::GPUManager::Stream()>>>(
        local_height,
        local_width,
        local_input.LockedBuffer(),
        local_input.LDim(),
        local_output.Buffer(),
        local_output.LDim());
  }

}

template class one_hot_layer<
  data_layout::DATA_PARALLEL, El::Device::GPU>;

} // namespace lbann
