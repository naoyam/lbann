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

#define LBANN_SOFTMAX_LAYER_INSTANTIATE
#include "lbann/layers/activations/softmax.hpp"

namespace lbann {

namespace {

// Minimum output value to avoid denormalized floats
#ifdef LBANN_ENABLE_SOFTMAX_CUTOFF
const DataType min_output = std::sqrt(std::numeric_limits<DataType>::min());
#else
const DataType min_output = 0;
#endif // LBANN_ENABLE_SOFTMAX_CUTOFF

void fp(lbann_comm& comm,
        const AbsDistMat& input,
        AbsDistMat& output,
        AbsDistMat& workspace) {

  // Local matrices
  const auto& local_input = input.LockedMatrix();
  auto& local_output = output.Matrix();
  auto& local_workspace = workspace.Matrix();
  const auto& local_height = local_input.Height();
  const auto& local_width = local_input.Width();

  // Find column-wise maximum entries
  El::Fill(workspace, std::numeric_limits<DataType>::lowest());
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    auto& max_entry = local_workspace(0, col);
    for (El::Int row = 0; row < local_height; ++row) {
      max_entry = std::max(max_entry, local_input(row, col));
    }
  }
  comm.allreduce(workspace, workspace.RedundantComm(), El::mpi::MAX);

  // Exponentiate outputs and compute column sums
  // Note: Subtracting by the column max prevents output from blowing
  // up. Large negative values underflow to 0.
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    const auto shift = local_workspace(0, col);
    DataType sum = 0;
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& x = local_input(row, col);
      auto& y = local_output(row, col);
      y = std::exp(x - shift);
      sum += y;
    }
    local_workspace(0, col) = sum;
  }
  comm.allreduce(workspace, workspace.RedundantComm());

  // Divide outputs by column sums
  // Note: Small values can be rounded to minimum output value to
  // avoid denormalized floats.
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    const auto& scale = 1 / local_workspace(0, col);
    for (El::Int row = 0; row < local_height; ++row) {
      auto& y = local_output(row, col);
      y = std::max(scale * y, min_output);
    }
  }

}

void bp(lbann_comm& comm,
        const AbsDistMat& output,
        const AbsDistMat& gradient_wrt_output,
        AbsDistMat& gradient_wrt_input,
        AbsDistMat& workspace) {

  // Local matrices
  const auto& local_output = output.LockedMatrix();
  const auto& local_gradient_wrt_output = gradient_wrt_output.LockedMatrix();
  auto& local_gradient_wrt_input = gradient_wrt_input.Matrix();
  auto& local_workspace = workspace.Matrix();
  const auto& local_height = local_output.Height();
  const auto& local_width = local_output.Width();

  // Compute dot products between output and gradient w.r.t. output
  El::Zero(local_workspace);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    auto& y_dot_dy = local_workspace(0, col);
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& y = local_output(row, col);
      const auto& dy = local_gradient_wrt_output(row, col);
      y_dot_dy += y * dy;
    }
  }
  comm.allreduce(workspace, workspace.RedundantComm());

  // Compute gradient w.r.t. input
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    const auto& y_dot_dy = local_workspace(0, col);
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& y = local_output(row, col);
      const auto& dy = local_gradient_wrt_output(row, col);
      auto& dx = local_gradient_wrt_input(row, col);
      dx = (y > min_output) ? y * (dy - y_dot_dy) : DataType(0);
    }
  }

}

} // namespace

template <>
void softmax_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::fp_compute() {
  fp(*get_comm(),
     get_prev_activations(),
     get_activations(),
     *m_workspace);
}
template <>
void softmax_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::bp_compute() {
  bp(*get_comm(),
     get_activations(),
     get_prev_error_signals(),
     get_error_signals(),
     *m_workspace);
}
template <>
void softmax_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::fp_compute() {
  fp(*get_comm(),
     get_prev_activations(),
     get_activations(),
     *m_workspace);
}
template <>
void softmax_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::bp_compute() {
  bp(*get_comm(),
     get_activations(),
     get_prev_error_signals(),
     get_error_signals(),
     *m_workspace);
}

template class softmax_layer<
  data_layout::DATA_PARALLEL, El::Device::CPU>;
template class softmax_layer<
  data_layout::MODEL_PARALLEL, El::Device::CPU>;

} // namespace lbann
