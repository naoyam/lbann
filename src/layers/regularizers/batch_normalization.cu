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
#include "lbann/utils/cuda.hpp"

namespace lbann {

namespace {

// Atomic add functions
#if __CUDA_ARCH__ >= 530
__device__ inline __half atomic_add(__half* address, __half val) {
  static_cast<void>(static_cast<__half (*)(__half*, __half)>(atomic_add)); // Suppress "unused function" warning
#if 0 // TODO: replace this once Nvidia implements atomicAdd for __half
  return atomicAdd(address, val);
#else
  unsigned int* address_as_uint = (unsigned int*) address;
  unsigned int old = *address_as_uint;
  __half* old_as_half = (__half*) &old;
  unsigned int assumed;
  unsigned int updated;
  __half* updated_as_half = (__half*) &updated;
  do {
    assumed = old;
    updated = old;
    *updated_as_half += value;
    old = atomicCAS(address_as_uint, assumed, updated);
  } while (assumed != old);
  return *old_as_half;
#endif // 0
}
#endif // __CUDA_ARCH__ >= 530
__device__ inline float atomic_add(float* address, float val) {
  static_cast<void>(static_cast<float (*)(float*, float)>(atomic_add)); // Suppress "unused function" warning
  return atomicAdd(address, val);
}
__device__ inline double atomic_add(double* address, double val) {
  static_cast<void>(static_cast<double (*)(double*, double)>(atomic_add)); // Suppress "unused function" warning
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                                         __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif // __CUDA_ARCH__ < 600
}

// Reciprocal square root functions
#if __CUDA_ARCH__ >= 530
__device__ inline float rsqrt_(__half x) {
  static_cast<void>(static_cast<__half (*)(__half)>(rsqrt_)); // Suppress "unused function" warning
  return hrsqrt(x);
}
#endif // __CUDA_ARCH__ >= 530
__device__ inline float rsqrt_(float x) {
  static_cast<void>(static_cast<float (*)(float)>(rsqrt_)); // Suppress "unused function" warning
  return rsqrtf(x);
}
__device__ inline double rsqrt_(double x) {
  static_cast<void>(static_cast<double (*)(double)>(rsqrt_)); // Suppress "unused function" warning
  return rsqrt(x);
}

/** CUDA kernel to compute channel sums.
 *  Sums and squares of sums are used to compute mean and variance.
 */
template <El::Int block_size>
__global__ void channel_sums_kernel(
  El::Int channel_height,
  El::Int width,
  const DataType * __restrict__ data, El::Int data_ldim,
        DataType * __restrict__ sums,
        DataType * __restrict__ sqsums) {

  // Indices
  const El::Int tid = threadIdx.x;
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int bidy = blockIdx.y;

  // Initialize shared memory
  __shared__ DataType shared_sums[block_size];
  __shared__ DataType shared_sqsums[block_size];

  // Compute row sums in shared memory
  DataType private_sum = 0;
  DataType private_sqsum = 0;
  if (gidx < channel_height) {
    const auto& row = gidx + bidy * channel_height;
    for (El::Int col = 0; col < width; ++col) {
      const auto& x = data[row + col * data_ldim];
      private_sum += x;
      private_sqsum += x * x;
    }
  }
  shared_sums[tid] = private_sum;
  shared_sqsums[tid] = private_sqsum;

  // Compute channel sum with shared memory reduction
  /// @todo unroll loops
  for (El::Int stride = block_size / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if(tid < stride) {
      shared_sums[tid] += shared_sums[tid + stride];
      shared_sqsums[tid] += shared_sqsums[tid + stride];
    }
  }

  // Output channel sum to global memory
  if (tid == 0) {
    atomic_add(&sums[bidy], shared_sums[0]);
    atomic_add(&sqsums[bidy], shared_sqsums[0]);
  }

}

/** CUDA kernel to compute statistics.
 *  On input, global_mean and global_var are assumed to contain sums
 *  and squares of sums, respectively.
 */
__global__ void compute_statistics_kernel(
  El::Int num_sums,
  El::Int num_per_sum,
  DataType epsilon,
  DataType decay,
  DataType * __restrict__ global_mean,
  DataType * __restrict__ global_var,
  DataType * __restrict__ global_running_mean,
  DataType * __restrict__ global_running_var) {
  constexpr DataType one = 1;
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int num_threads = blockDim.x * gridDim.x;
  for (El::Int i = gid; i < num_sums; i += num_threads) {

    // Compute mean and variance
    const auto& mean = global_mean[i] / num_per_sum;
    const auto& sqmean = global_var[i] / num_per_sum;
    auto var = num_per_sum * (sqmean - mean * mean) / (num_per_sum - 1);
    var = var > epsilon ? var : epsilon;
    global_mean[gid] = mean;
    global_var[gid] = var;

    // Compute running statistics
    auto& running_mean = global_running_mean[gid];
    auto& running_var = global_running_var[gid];
    running_mean = decay * running_mean + (one - decay) * mean;
    running_var = decay * running_var + (one - decay) * var;

  }

}

/** CUDA kernel to apply batch normalization. */
template <El::Int block_size>
__global__ void batch_normalization_kernel(
  El::Int channel_height,
  El::Int width,
  const DataType * __restrict__ global_input, El::Int input_ldim,
  const DataType * __restrict__ global_mean,
  const DataType * __restrict__ global_var,
  DataType epsilon,
  const DataType * __restrict__ global_scale,
  const DataType * __restrict__ global_bias,
        DataType * __restrict__ global_output, El::Int output_ldim) {

  // Indices
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int bidy = blockIdx.y;

  // Copy batch normalization parameters to private memory
  const auto& mean = global_mean[bidy];
  const auto& var = global_var[bidy];
  const auto& scale = global_scale[bidy];
  const auto& bias = global_bias[bidy];

  // Get reciprocal of standard deviation
  const auto& inv_stdev = rsqrt_(var + epsilon);

  // Apply batch normalization
  if (gidx < channel_height) {
    const auto& row = gidx + bidy * channel_height;
    for (El::Int col = 0; col < width; ++col) {
      const auto& x = global_input[row + col * input_ldim];
      const auto& xhat = (x - mean) * inv_stdev;
      const auto& y = scale * xhat + bias;
      global_output[row + col * output_ldim] = y;
    }
  }

}

/** CUDA kernel to compute gradients w.r.t. batch norm parameters. */
template <El::Int block_size>
__global__ void backprop1_kernel(
  El::Int channel_height,
  El::Int width,
  const DataType * __restrict__ global_input,
  El::Int input_ldim,
  const DataType * __restrict__ global_gradient_wrt_output,
  El::Int gradient_wrt_output_ldim,
  const DataType * __restrict__ global_mean,
  const DataType * __restrict__ global_var,
  DataType epsilon,
  const DataType * __restrict__ global_scale,
        DataType * __restrict__ global_dscale,
        DataType * __restrict__ global_dbias,
        DataType * __restrict__ global_dmean,
        DataType * __restrict__ global_dvar) {

  // Indices
  const El::Int tid = threadIdx.x;
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int bidy = blockIdx.y;

  // Initialize shared memory
  __shared__ DataType shared_dscale[block_size];
  __shared__ DataType shared_dbias[block_size];
  __shared__ DataType shared_dmean[block_size];
  __shared__ DataType shared_dvar[block_size];

  // Copy batch normalization parameters to private memory
  const auto& mean = global_mean[bidy];
  const auto& var = global_var[bidy];
  const auto& scale = global_scale[bidy];

  // Compute useful constants
  constexpr DataType zero = 0;
  const auto& inv_stdev = rsqrt_(var + epsilon);
  const auto& dvar_factor = inv_stdev * inv_stdev * inv_stdev / 2;

  // Compute row-wise gradient contributions in shared memory
  auto dscale = zero;
  auto dbias = zero;
  auto dmean = zero;
  auto dvar = zero;
  if (gidx < channel_height) {
    const auto& row = gidx + bidy * channel_height;
    for(El::Int col = 0; col < width; ++col) {
      const auto& x = global_input[row + col * input_ldim];
      const auto& xhat = (x - mean) * inv_stdev;
      const auto& dy = global_gradient_wrt_output[row + col * gradient_wrt_output_ldim];
      dscale += dy * xhat;
      dbias += dy;
      const auto& dxhat = dy * scale;
      dmean += - dxhat * inv_stdev;
      dvar += - dxhat * (x - mean) * dvar_factor;
    }
  }
  shared_dscale[tid] = dscale;
  shared_dbias[tid] = dbias;
  shared_dmean[tid] = dmean;
  shared_dvar[tid] = dvar;

  // Compute gradients with shared memory reduction
  // @todo unroll loops
  for (El::Int stride = block_size / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
      shared_dscale[tid] += shared_dscale[tid + stride];
      shared_dbias[tid] += shared_dbias[tid + stride];
      shared_dmean[tid] += shared_dmean[tid + stride];
      shared_dvar[tid] += shared_dvar[tid + stride];
    }
  }

  // Output channel sum to global memory
  if (tid == 0) {
    atomic_add(&global_dscale[bidy], shared_dscale[0]);
    atomic_add(&global_dbias[bidy], shared_dbias[0]);
    atomic_add(&global_dmean[bidy], shared_dmean[0]);
    atomic_add(&global_dvar[bidy], shared_dvar[0]);
  }

}

/** CUDA kernel to compute gradients w.r.t. input. */
template <El::Int block_size>
__global__ void backprop2_kernel(
  El::Int channel_height,
  El::Int local_width,
  El::Int num_per_sum,
  const DataType * __restrict__ global_input,
  El::Int input_ldim,
  const DataType * __restrict__ global_gradient_wrt_output,
  El::Int gradient_wrt_output_ldim,
  const DataType * __restrict__ global_mean,
  const DataType * __restrict__ global_var,
  DataType epsilon,
  const DataType * __restrict__ global_scale,
  const DataType * __restrict__ global_dmean,
  const DataType * __restrict__ global_dvar,
        DataType * __restrict__ global_gradient_wrt_input,
  El::Int gradient_wrt_input_ldim) {

  // Indices
  const El::Int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int bidy = blockIdx.y;

  // Copy batch normalization parameters to private memory
  const auto& mean = global_mean[bidy];
  const auto& var = global_var[bidy];
  const auto& scale = global_scale[bidy];
  const auto& dmean = global_dmean[bidy];
  const auto& dvar = global_dvar[bidy];

  // Compute useful constants
  const auto& inv_stdev = rsqrt_(var + epsilon);
  const auto& dmean_term = dmean / num_per_sum;
  const auto& dvar_term = dvar * 2 / (num_per_sum - 1);

  // Apply batch normalization
  if (gidx < channel_height) {
    const auto& row = gidx + bidy * channel_height;
    for (El::Int col = 0; col < local_width; ++col) {
      const auto& x = global_input[row + col * input_ldim];
      const auto& dy = global_gradient_wrt_output[row + col * gradient_wrt_output_ldim];
      const auto& dxhat = dy * scale;
      auto& dx = global_gradient_wrt_input[row + col * gradient_wrt_input_ldim];
      dx = dxhat * inv_stdev + dmean_term + dvar_term * (x - mean);
    }
  }

}

} // namespace


#ifdef LBANN_HAS_DISTCONV

template <>
void batch_normalization_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::fp_compute_distconv() {
  dc::MPIPrintStreamDebug() << get_name() << ": " << __FUNCTION__ << "\n";
  assert_always(distconv_enabled());

  const bool is_training =
      this->m_model->get_execution_mode() == execution_mode::training;

  if (keep_original_input()) {
    assert_always(this->m_model->get_current_mini_batch_size() ==
                  get_prev_activations().Width());
  }

  assert0(dc::tensor::View(
      m_scale_t, get_weights()[0]->get_values().LockedBuffer()));
  assert0(dc::tensor::View(
      m_bias_t, get_weights()[1]->get_values().LockedBuffer()));
  assert0(dc::tensor::View(
      m_running_mean_t, get_weights()[2]->get_values().Buffer()));
  assert0(dc::tensor::View(
      m_running_var_t, get_weights()[3]->get_values().Buffer()));

  m_bn->forward(m_prev_activations_t,
                m_mean_t,
                m_var_t,
                m_running_mean_t,
                m_running_var_t,
                m_scale_t,
                m_bias_t,
                m_activations_t,
                is_training);

  copy_out_activations();
}

template <>
void batch_normalization_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::bp_compute_distconv() {
  dc::MPIPrintStreamDebug() << get_name() << ": " << __FUNCTION__ << "\n";
  assert_always(distconv_enabled());

  // Check execution mode
  const bool is_training = this->m_model->get_execution_mode() == execution_mode::training;

  //assert_always(is_training && m_use_global_stats);
  assert_always(is_training);

  assert0(dc::tensor::View(
      m_scale_t, get_weights()[0]->get_values().LockedBuffer()));

  m_bn->backward_stage1(m_prev_activations_t,
                        m_prev_error_signals_t,
                        m_mean_t, m_var_t, m_scale_t,
                        m_scale_gradient_t, m_bias_gradient_t,
                        m_mean_gradient_t, m_var_gradient_t,
                        false);

  // Verbatim copy from bp_compute_gpu
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
    Zero(*m_mean_gradient);
    Zero(*m_var_gradient);
  }

  const int effective_mini_batch_size = this->m_model->get_effective_mini_batch_size();

  optimizer* scale_optimizer = m_weights[0]->get_optimizer();
  if (scale_optimizer != nullptr) {
    scale_optimizer->add_to_gradient_staging(
        *m_scale_gradient,
        DataType(1) / effective_mini_batch_size);
  }
  optimizer* bias_optimizer = m_weights[1]->get_optimizer();
  if (bias_optimizer != nullptr) {
    bias_optimizer->add_to_gradient_staging(
        *m_bias_gradient,
        DataType(1) / effective_mini_batch_size);
  }

  m_bn->backward_stage2(m_prev_activations_t,
                        m_prev_error_signals_t,
                        m_mean_t, m_var_t, m_scale_t,
                        m_mean_gradient_t, m_var_gradient_t,
                        m_error_signals_t);

  copy_out_error_signals();
}

#endif

template <>
void batch_normalization_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::fp_compute() {
#ifdef LBANN_HAS_DISTCONV
  if (distconv_enabled()) {
    fp_compute_distconv();
    if (!early_terminate_last_iteration()) {
      return;
    }
  }
#endif // LBANN_HAS_DISTCONV
  constexpr DataType one = 1;
  const bool is_training = this->m_model->get_execution_mode() == execution_mode::training;

  // CUDA objects
  CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
  auto&& stream = El::GPUManager::Stream();
  
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
    El::Zero(local_mean);
    El::Zero(local_var);
    if (!local_input.IsEmpty()) {
      const El::Int block_size = 256;
      dim3 block_dims, grid_dims;
      block_dims.x = block_size;
      grid_dims.x = (channel_size + block_size - 1) / block_size;
      grid_dims.y = num_channels;
      channel_sums_kernel<block_size>
        <<<grid_dims, block_dims, 0, stream>>>(
          channel_size, local_width,
          local_input.LockedBuffer(), local_input.LDim(),
          local_mean.Buffer(), local_var.Buffer());
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
    } else if (num_channels > 0) {
      const El::Int block_dim = 256;
      const El::Int grid_dim = (num_channels + block_dim - 1) / block_dim;
      compute_statistics_kernel
        <<<grid_dim, block_dim, 0, stream>>>(
          num_channels, num_per_sum, m_epsilon, m_decay,
          local_mean.Buffer(), local_var.Buffer(),
          local_running_mean.Buffer(), local_running_var.Buffer());
    }

  }

  // Apply batch normalization
  const auto& local_scale = this->m_weights[0]->get_values().LockedMatrix();
  const auto& local_bias = this->m_weights[1]->get_values().LockedMatrix();
  const auto& local_mean = (is_training ?
                            m_mean->LockedMatrix() :
                            this->m_weights[2]->get_values().LockedMatrix());
  const auto& local_var = (is_training ?
                           m_var->LockedMatrix() :
                           this->m_weights[3]->get_values().LockedMatrix());
  if (!local_input.IsEmpty()) {
    const El::Int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_size + block_size - 1) / block_size;
    grid_dims.y = num_channels;
    batch_normalization_kernel<block_size>
      <<<grid_dims, block_dims, 0, stream>>>(
        channel_size, local_width,
        local_input.LockedBuffer(), local_input.LDim(),
        local_mean.LockedBuffer(), local_var.LockedBuffer(), m_epsilon,
        local_scale.LockedBuffer(), local_bias.LockedBuffer(),
        local_output.Buffer(), local_output.LDim());
  }
#ifdef LBANN_HAS_DISTCONV
  dump_reference_activations();
#endif // LBANN_HAS_DISTCONV
}

template <>
void batch_normalization_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::bp_compute() {
#ifdef LBANN_HAS_DISTCONV
  if (distconv_enabled()) {
    bp_compute_distconv();
    if (!early_terminate_last_iteration()) {
      return;
    }
    assert0(dc::tensor::View(
        m_error_signals_copyout,
        get_error_signals().Buffer()));
    m_error_signals_copyout.zero();
  }
#endif // LBANN_HAS_DISTCONV
  constexpr DataType one = 1;
  const bool is_training = this->m_model->get_execution_mode() == execution_mode::training;

  // CUDA objects
  CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
  auto&& stream = El::GPUManager::Stream();

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
  // Compute gradients w.r.t. batch norm parameters
  El::Zero(local_scale_gradient);
  El::Zero(local_bias_gradient);
  El::Zero(local_mean_gradient);
  El::Zero(local_var_gradient);
  if (!local_input.IsEmpty()) {
    const El::Int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_size + block_size - 1) / block_size;
    grid_dims.y = num_channels;
    backprop1_kernel<block_size>
      <<<grid_dims, block_dims, 0, stream>>>(
        channel_size, local_width,
        local_input.LockedBuffer(), local_input.LDim(),
        local_gradient_wrt_output.LockedBuffer(), local_gradient_wrt_output.LDim(),
        local_mean.LockedBuffer(), local_var.LockedBuffer(), m_epsilon,
        local_scale.LockedBuffer(),
        local_scale_gradient.Buffer(), local_bias_gradient.Buffer(),
        local_mean_gradient.Buffer(), local_var_gradient.Buffer());
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
  } else if (!local_input.IsEmpty()) {
    const El::Int block_size = 256;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    grid_dims.x = (channel_size + block_size - 1) / block_size;
    grid_dims.y = num_channels;
    backprop2_kernel<block_size>
      <<<grid_dims, block_dims, 0, stream>>>(
        channel_size, local_width, num_per_sum,
        local_input.LockedBuffer(), local_input.LDim(),
        local_gradient_wrt_output.LockedBuffer(), local_gradient_wrt_output.LDim(),
        local_mean.LockedBuffer(), local_var.LockedBuffer(), m_epsilon,
        local_scale.LockedBuffer(),
        local_mean_gradient.LockedBuffer(), local_var_gradient.LockedBuffer(),
        local_gradient_wrt_input.Buffer(), local_gradient_wrt_input.LDim());
  }
#ifdef LBANN_HAS_DISTCONV
  dump_reference_error_signals();
#endif // LBANN_HAS_DISTCONV
}

} // namespace lbann
