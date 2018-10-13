////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#include "lbann/layers/regularizers/batch_normalization.hpp"

namespace lbann {

template <>
void batch_normalization_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::fp_compute() {
  constexpr DataType zero = 0;
  constexpr DataType one = 1;
  const bool is_training = this->m_model->get_execution_mode() == execution_mode::training;

  // Matrices
  const auto& input = get_prev_activations();
  const auto& local_input = input.LockedMatrix();
  auto& local_output = get_local_activations();

  // Matrix parameters
  const auto& width = input.Width();
  const auto& local_width = local_input.Width();
  const auto& output_dims = get_output_dims();
  const auto& num_channels = output_dims[0];
  const auto& channel_size = get_output_size() / num_channels;

  // Compute statistics
  if (is_training) {

    // Local matrices
    auto& local_mean = m_mean->Matrix();
    auto& local_var = m_var->Matrix();
    auto& local_running_mean = this->m_weights[2]->get_values().Matrix();
    auto& local_running_var = this->m_weights[3]->get_values().Matrix();

    // Compute sums and sums of squares
#pragma omp parallel for
    for (El::Int channel = 0; channel < num_channels; ++channel) {
      DataType sum = zero;
      DataType sqsum = zero;
      const auto& row_start = channel * channel_size;
      const auto& row_end = (channel+1) * channel_size;
      for (El::Int col = 0; col < local_width; ++col) {
        for (El::Int row = row_start; row < row_end; ++row) {
          const auto& x = local_input(row, col);
          sum += x;
          sqsum += x * x;
        }
      }
      local_mean(channel, 0) = sum;
      local_var(channel, 0) = sqsum;
    }
    El::Int num_per_sum;
    if (m_use_global_stats) {
      m_comm->allreduce(*m_mean, m_mean->RedundantComm(), El::mpi::SUM);
      m_comm->allreduce(*m_var, m_var->RedundantComm(), El::mpi::SUM);
      num_per_sum = channel_size * width;
    } else {
      num_per_sum = channel_size * local_width;
    }

    // Compute minibatch statistics
    if (num_per_sum <= 1) {
      El::Fill(local_var, one);
    } else {
#pragma omp parallel for
      for (El::Int channel = 0; channel < num_channels; ++channel) {
        const auto& mean = local_mean(channel, 0) / num_per_sum;
        const auto& sqmean = local_var(channel, 0) / num_per_sum;
        auto var = num_per_sum * (sqmean - mean * mean) / (num_per_sum - 1);
        var = std::max(var, m_epsilon);
        local_mean(channel, 0) = mean;
        local_var(channel, 0) = var;
        auto& running_mean = local_running_mean(channel, 0);
        auto& running_var = local_running_var(channel, 0);
        running_mean = m_decay * running_mean + (one - m_decay) * mean;
        running_var = m_decay * running_var + (one - m_decay) * var;
      }
    }

  }

  // Get matrices
  const auto& local_scale = this->m_weights[0]->get_values().LockedMatrix();
  const auto& local_bias = this->m_weights[1]->get_values().LockedMatrix();
  const auto& local_mean = (is_training ?
                            m_mean->LockedMatrix() :
                            this->m_weights[2]->get_values().LockedMatrix());
  const auto& local_var = (is_training ?
                           m_var->LockedMatrix() :
                           this->m_weights[3]->get_values().LockedMatrix());

  // Iterate through channels
#pragma omp parallel for
  for (El::Int channel = 0; channel < num_channels; ++channel) {

    // Get channel parameters
    const auto& mean = local_mean(channel, 0);
    const auto& var = local_var(channel, 0);
    const DataType inv_stdev = 1 / std::sqrt(var + m_epsilon);
    const auto& scale = local_scale(channel, 0);
    const auto& bias = local_bias(channel, 0);

    // Apply batch normalization to inputs in channel
    const auto& row_start = channel * channel_size;
    const auto& row_end = (channel+1) * channel_size;
    for (El::Int col = 0; col < local_width; ++col) {
      for (El::Int row = row_start; row < row_end; ++row) {
        const auto& x = local_input(row, col);
        const auto& xhat = (x - mean) * inv_stdev;
        auto& y = local_output(row, col);
        y = scale * xhat + bias;
      }
    }

  }

}

template <>
void batch_normalization_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::bp_compute() {
  constexpr DataType one = 1;
  const bool is_training = this->m_model->get_execution_mode() == execution_mode::training;

  // Matrices
  const auto& local_scale = this->m_weights[0]->get_values().LockedMatrix();
  const auto& local_mean = (is_training ?
                            m_mean->LockedMatrix() :
                            this->m_weights[2]->get_values().LockedMatrix());
  const auto& local_var = (is_training ?
                           m_var->LockedMatrix() :
                           this->m_weights[3]->get_values().LockedMatrix());
  const auto& input = get_prev_activations();
  const auto& local_input = input.LockedMatrix();
  const auto& local_gradient_wrt_output = get_local_prev_error_signals();
  auto& local_gradient_wrt_input = get_local_error_signals();
  auto& local_mean_gradient = m_mean_gradient->Matrix();
  auto& local_var_gradient = m_var_gradient->Matrix();
  auto& local_scale_gradient = m_scale_gradient->Matrix();
  auto& local_bias_gradient = m_bias_gradient->Matrix();

  // Matrix parameters
  const El::Int effective_mini_batch_size = this->m_model->get_effective_mini_batch_size();
  const auto& width = input.Width();
  const auto& local_width = local_input.Width();
  const auto& output_dims = get_output_dims();
  const auto& num_channels = output_dims[0];
  const auto& channel_size = get_output_size() / num_channels;

  // Compute local gradients
#pragma omp parallel for
  for (El::Int channel = 0; channel < num_channels; ++channel) {

    // Initialize channel parameters and gradients
    const auto& mean = local_mean(channel, 0);
    const auto& var = local_var(channel, 0);
    const auto& scale = local_scale(channel, 0);
    const DataType inv_stdev = 1 / std::sqrt(var + m_epsilon);
    const auto& dvar_factor = inv_stdev * inv_stdev * inv_stdev / 2;
    DataType dmean = 0;
    DataType dvar = 0;
    DataType dscale = 0;
    DataType dbias = 0;

    // Compute gradient contributions from local entries
    const auto& row_start = channel * channel_size;
    const auto& row_end = (channel+1) * channel_size;
    for (El::Int col = 0; col < local_width; ++col) {
      for (El::Int row = row_start; row < row_end; ++row) {
        const auto& x = local_input(row, col);
        const auto& xhat = (x - mean) * inv_stdev;
        const auto& dy = local_gradient_wrt_output(row, col);
        dscale += dy * xhat;
        dbias += dy;
        const auto& dxhat = dy * scale;
        dmean += - dxhat * inv_stdev;
        dvar += - dxhat * (x - mean) * dvar_factor;
      }
    }
    local_mean_gradient(channel, 0) = dmean;
    local_var_gradient(channel, 0) = dvar;
    local_scale_gradient(channel, 0) = dscale;
    local_bias_gradient(channel, 0) = dbias;

  }

  // Accumulate gradients
  if (is_training) {
    if (m_use_global_stats) {
      m_comm->allreduce(*m_mean_gradient,
                        m_mean_gradient->RedundantComm(),
                        El::mpi::SUM);
      m_comm->allreduce(*m_var_gradient,
                        m_var_gradient->RedundantComm(),
                        El::mpi::SUM);
    }
  } else {
    El::Zero(*m_mean_gradient);
    El::Zero(*m_var_gradient);
  }
  optimizer* scale_optimizer = m_weights[0]->get_optimizer();
  if (scale_optimizer != nullptr) {
    scale_optimizer->add_to_gradient_staging(*m_scale_gradient,
                                             one / effective_mini_batch_size);
  }
  optimizer* bias_optimizer = m_weights[1]->get_optimizer();
  if (bias_optimizer != nullptr) {
    bias_optimizer->add_to_gradient_staging(*m_bias_gradient,
                                            one / effective_mini_batch_size);
  }

  // Compute error signal
  const auto& num_per_sum = (m_use_global_stats ?
                             width * channel_size :
                             local_width * channel_size);
  if (num_per_sum <= 1) {
    El::Zero(local_gradient_wrt_input);
  } else {
#pragma omp parallel for
    for (El::Int channel = 0; channel < num_channels; ++channel) {

      // Initialize channel parameters and gradients
      const auto& mean = local_mean(channel, 0);
      const auto& var = local_var(channel, 0);
      const auto& scale = local_scale(channel, 0);
      const auto& dmean = local_mean_gradient(channel, 0);
      const auto& dvar = local_var_gradient(channel, 0);

      // Compute useful constants
      const DataType inv_stdev = 1 / std::sqrt(var + m_epsilon);
      const auto& dmean_term = dmean / num_per_sum;
      const auto& dvar_term = dvar * 2 / (num_per_sum - 1);

      // Compute error signal for current channel
      const auto& row_start = channel * channel_size;
      const auto& row_end = (channel+1) * channel_size;
      for (El::Int col = 0; col < local_width; ++col) {
        for (El::Int row = row_start; row < row_end; ++row) {
          const auto& x = local_input(row, col);
          const auto& dy = local_gradient_wrt_output(row, col);
          const auto& dxhat = dy * scale;
          auto& dx = local_gradient_wrt_input(row, col);
          dx = dxhat * inv_stdev + dmean_term + dvar_term * (x - mean);
        }
      }

    }
  }

}

} // namespace lbann
