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

#define LBANN_L2_NORM2_LAYER_INSTANTIATE
#include "lbann/layers/loss/l2_norm2.hpp"

namespace lbann {

namespace {

void local_fp_cpu(const AbsMat& local_input,
                  AbsMat& local_contribution) {
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_input.Width(); ++col) {
    DataType sum = 0;
    for (El::Int row = 0; row < local_input.Height(); ++row) {
      const auto& x = local_input(row, col);
      sum += x * x;
    }
    local_contribution(0, col) = sum;
  }
}

void local_bp_cpu(const AbsMat& local_input,
                  const AbsMat& local_gradient_wrt_output,
                  AbsMat& local_gradient_wrt_input) {
  auto const width = local_input.Width();
  auto const height = local_input.Height();
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int col = 0; col < width; ++col) {
    for (El::Int row = 0; row < height; ++row) {
      const auto& x = local_input(row, col);
      const auto& dy = local_gradient_wrt_output(0, col);
      auto& dx = local_gradient_wrt_input(row, col);
      dx = 2 * x * dy;
    }
  }
}

} // namespace

template <>
void l2_norm2_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>
     ::local_fp_compute(const AbsMat& local_input,
                        AbsMat& local_contribution) {
  local_fp_cpu(local_input, local_contribution);
}
template <>
void l2_norm2_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>
     ::local_bp_compute(const AbsMat& local_input,
                        const AbsMat& local_gradient_wrt_output,
                        AbsMat& local_gradient_wrt_input) {
  local_bp_cpu(local_input,
               local_gradient_wrt_output,
               local_gradient_wrt_input);
}
template <>
void l2_norm2_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
     ::local_fp_compute(const AbsMat& local_input,
                        AbsMat& local_contribution) {
  local_fp_cpu(local_input, local_contribution);
}
template <>
void l2_norm2_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
     ::local_bp_compute(const AbsMat& local_input,
                        const AbsMat& local_gradient_wrt_output,
                        AbsMat& local_gradient_wrt_input) {
  local_bp_cpu(local_input,
               local_gradient_wrt_output,
               local_gradient_wrt_input);
}

template class l2_norm2_layer<
  data_layout::DATA_PARALLEL, El::Device::CPU>;
template class l2_norm2_layer<
  data_layout::MODEL_PARALLEL, El::Device::CPU>;

} // namespace lbann
