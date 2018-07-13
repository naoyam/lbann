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

#include "lbann/layers/activations/relu.hpp"

namespace lbann {

// Model-parallel CPU forward/backward prop
template <>
void relu_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::fp_compute() {
  entrywise_activation_layer::fp_compute_cpu();
}
template <>
void relu_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::bp_compute() {
  entrywise_activation_layer::bp_compute_cpu();
}

// Data-parallel CPU forward/backward prop
template <>
void relu_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::fp_compute() {
  entrywise_activation_layer::fp_compute_cpu();
}
template <>
void relu_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::bp_compute() {
  entrywise_activation_layer::bp_compute_cpu();
}

#ifdef LBANN_HAS_GPU

// Model-parallel GPU forward/backward prop
template <>
void relu_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::fp_compute() {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else
  const DataType zero = DataType(0);
  const DataType one = DataType(1);
  const auto& local_input = get_local_prev_activations();
  auto& local_output = get_local_activations();
  if (local_input.Height() > 0 && local_input.Width() > 0) {
    CHECK_CUDNN(cudnnActivationForward(cudnn::get_handle(),
                                       m_activation_cudnn_desc,
                                       &one,
                                       m_tensors_cudnn_desc.get_prev_activations(),
                                       local_input.LockedBuffer(),
                                       &zero,
                                       m_tensors_cudnn_desc.get_activations(),
                                       local_output.Buffer()));
  }
#endif // LBANN_HAS_CUDNN
}

template <>
void relu_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::bp_compute() {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else
  const DataType zero = DataType(0);
  const DataType one = DataType(1);
  const auto& local_input = get_local_prev_activations();
  const auto& local_output = get_local_activations();
  const auto& local_gradient_wrt_output = get_local_prev_error_signals();
  auto& local_gradient_wrt_input = get_local_error_signals();
  if (local_input.Height() > 0 && local_input.Width() > 0) {
    CHECK_CUDNN(cudnnActivationBackward(cudnn::get_handle(),
                                        m_activation_cudnn_desc,
                                        &one,
                                        m_tensors_cudnn_desc.get_activations(),
                                        local_output.LockedBuffer(),
                                        m_tensors_cudnn_desc.get_prev_error_signals(),
                                        local_gradient_wrt_output.LockedBuffer(),
                                        m_tensors_cudnn_desc.get_prev_activations(),
                                        local_input.LockedBuffer(),
                                        &zero,
                                        m_tensors_cudnn_desc.get_error_signals(),
                                        local_gradient_wrt_input.Buffer()));
  }
#endif // LBANN_HAS_CUDNN
}

#ifdef LBANN_HAS_DISTCONV
using namespace dc;
template <>
void relu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::
fp_compute_distconv() {
  MPIPrintStreamDebug()
      << get_name() << ": " << __FUNCTION__ << "\n";
  assert_always(distconv_enabled());

  // Useful constants
  const DataType one = 1;
  const DataType zero = 0;

  m_relu->forward(one, m_prev_activations_t, zero, m_activations_t);

  copy_out_activations();
}

template <>
void relu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::
bp_compute_distconv() {
  MPIPrintStreamDebug()
      << get_name() << ": " << __FUNCTION__ << "\n";
  assert_always(distconv_enabled());
  const DataType one = 1;

#ifdef DISTCONV_ZERO_OUT_ERROR_SIGNALS
  m_error_signals_t.zero();
  m_relu->backward(one, m_activations_t, m_prev_error_signals_t,
                   m_prev_activations_t, one, m_error_signals_t);
#else
  m_relu->backward(one, m_activations_t, m_prev_error_signals_t,
                   m_prev_activations_t, DataType(0), m_error_signals_t);
#endif
  copy_out_error_signals();
}
#endif

// Data-parallel GPU forward/backward prop
template <>
void relu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::fp_compute() {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else
#ifdef LBANN_HAS_DISTCONV
  if (distconv_enabled()) {
    fp_compute_distconv();
    if (!early_terminate_last_iteration()) {
      return;
    }
    // fall through the normal code path to obtain reference results
  }
#endif
  const DataType zero = DataType(0);
  const DataType one = DataType(1);
  const auto& local_input = get_local_prev_activations();
  auto& local_output = get_local_activations();
  if (local_input.Height() > 0 && local_input.Width() > 0) {
    CHECK_CUDNN(cudnnActivationForward(cudnn::get_handle(),
                                       m_activation_cudnn_desc,
                                       &one,
                                       m_tensors_cudnn_desc.get_prev_activations(),
                                       local_input.LockedBuffer(),
                                       &zero,
                                       m_tensors_cudnn_desc.get_activations(),
                                       local_output.Buffer()));
  }
#ifdef LBANN_HAS_DISTCONV
    if (distconv_enabled() && early_terminate_last_iteration()) {
      dump_reference_activations();
    }
#endif // LBANN_HAS_DISTCONV
#endif // LBANN_HAS_CUDNN
}
template <>
void relu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::bp_compute() {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else
#ifdef LBANN_HAS_DISTCONV
  if (distconv_enabled()) {
    bp_compute_distconv();
    if (!early_terminate_last_iteration()) {
      return;
    }
  }
#endif // LBANN_HAS_DISTCONV
  const DataType zero = DataType(0);
  const DataType one = DataType(1);
  const auto& local_input = get_local_prev_activations();
  const auto& local_output = get_local_activations();
  const auto& local_gradient_wrt_output = get_local_prev_error_signals();
  auto& local_gradient_wrt_input = get_local_error_signals();
  if (local_input.Height() > 0 && local_input.Width() > 0) {
    CHECK_CUDNN(cudnnActivationBackward(cudnn::get_handle(),
                                        m_activation_cudnn_desc,
                                        &one,
                                        m_tensors_cudnn_desc.get_activations(),
                                        local_output.LockedBuffer(),
                                        m_tensors_cudnn_desc.get_prev_error_signals(),
                                        local_gradient_wrt_output.LockedBuffer(),
                                        m_tensors_cudnn_desc.get_prev_activations(),
                                        local_input.LockedBuffer(),
                                        &zero,
                                        m_tensors_cudnn_desc.get_error_signals(),
                                        local_gradient_wrt_input.Buffer()));
  }
#ifdef LBANN_HAS_DISTCONV
  if (distconv_enabled() && early_terminate_last_iteration()) {
    dump_reference_error_signals();
  }
#endif // LBANN_HAS_DISTCONV
#endif // LBANN_HAS_CUDNN
}

#ifdef LBANN_HAS_DISTCONV
using namespace dc;

template <>
void relu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::setup_tensor_distribution_init(
    std::map<const Layer*, std::array<Dist, 4>> &dists,
    std::map<Dist*, std::set<Dist*>> &invariants,
    std::set<Dist*> &updated,
    std::set<Dist*> &fixed)  {
  Layer::setup_tensor_distribution_init(
      dists, invariants, updated, fixed);
  if (!distconv_enabled()) return;
  auto &layer_dists = dists[this];
#ifdef DISTCONV_USE_SAME_RELU_CALL_AS_LBANN
  // This isn't necessary for cuDNN, but necessary to make it work
  // with the ordering of ReLU parameters used in LBANN
  // x == y
  invariants[&layer_dists[0]].insert(
      &layer_dists[1]);
  invariants[&layer_dists[1]].insert(
      &layer_dists[0]);
#endif
  // x == dx
  invariants[&layer_dists[0]].insert(
      &layer_dists[2]);
  invariants[&layer_dists[2]].insert(
      &layer_dists[0]);
  //y == dy
  invariants[&layer_dists[1]].insert(
      &layer_dists[3]);
  invariants[&layer_dists[3]].insert(
      &layer_dists[1]);
}

template <>
void relu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::
setup_tensors_fwd(const std::array<Dist, 4> &dists) {
  Layer::setup_tensors_fwd(dists);
  if (!distconv_enabled()) return;
  setup_prev_activations_tensor(dists);
  setup_activations_tensor(dists);
  setup_activations_copyout_tensor(dists);
}

template <>
void relu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::
setup_tensors_bwd(const std::array<Dist, 4> &dists)  {
  Layer::setup_tensors_bwd(dists);
  if (!distconv_enabled()) return;
  setup_prev_error_signals_tensor(dists);
  setup_error_signals_tensor(dists);
  setup_error_signals_copyout_tensor(dists);
  // Init the dc::Pooling layer
  m_relu = new ReLU(get_backend(this->get_comm()->get_model_comm().comm));
  m_relu->setup(m_prev_activations_t, m_activations_t,
                m_error_signals_t, m_prev_error_signals_t);
}

#endif // LBANN_HAS_DISTCONV
#endif // LBANN_HAS_GPU

} // namespace lbann
