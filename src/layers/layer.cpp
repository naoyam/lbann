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

#include "lbann/layers/layer.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/io/persist.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {

Layer::Layer(lbann_comm *comm)
  : m_comm(comm),
    m_cudnn(nullptr),
    m_frozen(false) {

  // Initialize layer name
  static int num_layers = 0;
  m_name = "layer" + std::to_string(num_layers);
  num_layers++;

  // Initialize neuron tensor dimensions
  m_neuron_dims = std::vector<int>(1, 0);
  m_num_neurons = 0;
  m_num_neuron_dims = 1;
  m_prev_neuron_dims = std::vector<int>(1, 0);
  m_num_prev_neurons = 0;
  m_num_prev_neuron_dims = 1;

  // Initialize GPU information
  m_using_gpus = false;
#ifdef LBANN_HAS_CUDNN
  m_prev_activations_cudnn_desc = nullptr;
  m_activations_cudnn_desc = nullptr;
  m_prev_error_signals_cudnn_desc = nullptr;
  m_error_signals_cudnn_desc = nullptr;
#endif // LBANN_HAS_CUDNN

  // Reset timing counters
  reset_counters();

}

Layer::Layer(const Layer& other) :
  m_comm(other.m_comm),
  m_neuron_dims(other.m_neuron_dims),
  m_num_neurons(other.m_num_neurons),
  m_num_neuron_dims(other.m_num_neuron_dims),
  m_prev_neuron_dims(other.m_prev_neuron_dims),
  m_num_prev_neurons(other.m_num_prev_neurons),
  m_num_prev_neuron_dims(other.m_num_prev_neuron_dims),
  m_weights(other.m_weights),
  m_parent_layers(other.m_parent_layers),
  m_child_layers(other.m_child_layers),
  m_expected_num_parent_layers(other.m_expected_num_parent_layers),
  m_expected_num_child_layers(other.m_expected_num_child_layers),
  m_model(other.m_model),
  m_cudnn(other.m_cudnn),
  m_frozen(other.m_frozen),
  m_fp_time(other.m_fp_time),
  m_fp_compute_time(other.m_fp_compute_time),
  m_bp_time(other.m_bp_time),
  m_bp_compute_time(other.m_bp_compute_time),
  m_update_time(other.m_update_time),
  m_name(other.m_name),
  m_using_gpus(other.m_using_gpus) {

  // Deep matrix copies
  m_prev_activations   = other.m_prev_activations;
  m_activations        = other.m_activations;
  m_prev_error_signals = other.m_prev_error_signals;
  m_error_signals      = other.m_error_signals;
  for (auto& m : m_prev_activations)   { m = m->Copy(); }
  for (auto& m : m_activations)        { m = m->Copy(); }
  for (auto& m : m_prev_error_signals) { m = m->Copy(); }
  for (auto& m : m_error_signals)      { m = m->Copy(); }

#ifdef LBANN_HAS_CUDNN
  m_prev_activations_cudnn_desc = nullptr;
  m_activations_cudnn_desc = nullptr;
  m_prev_error_signals_cudnn_desc = nullptr;
  m_error_signals_cudnn_desc = nullptr;
  cudnn::copy_tensor_cudnn_desc(other.m_prev_activations_cudnn_desc,
                                m_prev_activations_cudnn_desc);
  cudnn::copy_tensor_cudnn_desc(other.m_activations_cudnn_desc,
                                m_activations_cudnn_desc);
  cudnn::copy_tensor_cudnn_desc(other.m_prev_error_signals_cudnn_desc,
                                m_prev_error_signals_cudnn_desc);
  cudnn::copy_tensor_cudnn_desc(other.m_error_signals_cudnn_desc,
                                m_error_signals_cudnn_desc);
#endif // LBANN_HAS_CUDNN
}

Layer& Layer::operator=(const Layer& other) {

  // Shallow copies
  m_comm = other.m_comm;
  m_neuron_dims = other.m_neuron_dims;
  m_num_neurons = other.m_num_neurons;
  m_num_neuron_dims = other.m_num_neuron_dims;
  m_prev_neuron_dims = other.m_prev_neuron_dims;
  m_num_prev_neurons = other.m_num_prev_neurons;
  m_num_prev_neuron_dims = other.m_num_prev_neuron_dims;
  m_weights = other.m_weights;
  m_parent_layers = other.m_parent_layers;
  m_child_layers = other.m_child_layers;
  m_expected_num_parent_layers = other.m_expected_num_parent_layers;
  m_expected_num_child_layers = other.m_expected_num_child_layers;
  m_model = other.m_model;
  m_using_gpus = other.m_using_gpus;
  m_cudnn = other.m_cudnn;
  m_frozen = other.m_frozen;
  m_fp_time = other.m_fp_time;
  m_fp_compute_time = other.m_fp_compute_time;
  m_bp_time = other.m_bp_time;
  m_bp_compute_time = other.m_bp_compute_time;
  m_update_time = other.m_update_time;
  m_name = other.m_name;

  // Deep matrix copies
  deallocate_matrices();
  m_prev_activations   = other.m_prev_activations;
  m_activations        = other.m_activations;
  m_prev_error_signals = other.m_prev_error_signals;
  m_error_signals      = other.m_error_signals;
  for (auto& m : m_prev_activations)   { m = m->Copy(); }
  for (auto& m : m_activations)        { m = m->Copy(); }
  for (auto& m : m_prev_error_signals) { m = m->Copy(); }
  for (auto& m : m_error_signals)      { m = m->Copy(); }

#ifdef LBANN_HAS_CUDNN
  cudnn::copy_tensor_cudnn_desc(other.m_prev_activations_cudnn_desc,
                                m_prev_activations_cudnn_desc);
  cudnn::copy_tensor_cudnn_desc(other.m_activations_cudnn_desc,
                                m_activations_cudnn_desc);
  cudnn::copy_tensor_cudnn_desc(other.m_prev_error_signals_cudnn_desc,
                                m_prev_error_signals_cudnn_desc);
  cudnn::copy_tensor_cudnn_desc(other.m_error_signals_cudnn_desc,
                                m_error_signals_cudnn_desc);
#endif // LBANN_HAS_CUDNN

  return *this;
}

Layer::~Layer() {
#ifdef LBANN_HAS_CUDNN
  if(m_prev_activations_cudnn_desc != nullptr) {
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_prev_activations_cudnn_desc));
  }
  if(m_activations_cudnn_desc != nullptr) {
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_activations_cudnn_desc));
  }
  if(m_prev_error_signals_cudnn_desc != nullptr) {
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_prev_error_signals_cudnn_desc));
  }
  if(m_error_signals_cudnn_desc != nullptr) {
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_error_signals_cudnn_desc));
  }
#endif // LBANN_HAS_CUDNN
  deallocate_matrices();
}

std::string Layer::get_description() const {
  std::stringstream ss;
  ss << get_name() << " (" << get_type() << "): ";
  return ss.str();
}

std::string Layer::get_topo_description() const {
  std::stringstream ss;
  const size_t num_children = get_num_children();
  for (size_t i = 0; i < num_children; ++i) {
    const auto& dims = get_neuron_dims(i);
    if (i > 0) { ss << ", "; }
    ss << "activations";
    if (num_children > 1) { ss << "[" << i << "]"; }
    ss << " = [";
    switch (dims.size()) {
    case 0:
      ss << "0"; break;
    case 2:
      ss << dims[0] << "c x "
         << dims[1] << "w";
      break;
    case 3:
      ss << dims[0] << "c x "
         << dims[1] << "w x "
         << dims[2] << "h";
      break;
    default:
      ss << dims[0];
      for (size_t j = 1; j < dims.size(); ++j) {
        ss << " x " << dims[j];
      }
    }
    ss << ", " << get_activations(i).Width() << "s]";
  }
  return ss.str();
}

void Layer::forward_prop() {
  const auto fp_start = get_time();

  // Setup matrix data, e.g. input matrices
  fp_setup_data(m_model->get_current_mini_batch_size());

  #if defined(LBANN_HAS_CUDNN) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) { this->m_cudnn->check_error(); }
  #endif // defined(LBANN_HAS_CUDNN) && defined(LBANN_DEBUG)

  // Apply layer's compute function
  const auto fp_compute_start = get_time();
  fp_compute();
  m_fp_compute_time += get_time() - fp_compute_start;

  // Add this layer as a gradient source for weight optimizers
  for (auto&& w : m_weights) {
    optimizer* opt = w->get_optimizer();
    if (opt != nullptr) { opt->add_gradient_source(this); }
  }

  #if defined(LBANN_HAS_CUDNN) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) { this->m_cudnn->check_error(); }
  #endif // defined(LBANN_HAS_CUDNN) && defined(LBANN_DEBUG)

  m_fp_time += get_time() - fp_start;
}

void Layer::back_prop() {
  const auto bp_start = get_time();

  // Setup matrix data, e.g. input matrices
  bp_setup_data(m_model->get_current_mini_batch_size());

  #if defined(LBANN_HAS_CUDNN) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) { this->m_cudnn->check_error(); }
  #endif // defined(LBANN_HAS_CUDNN) && defined(LBANN_DEBUG)

  // Backprop the compute function.
  const auto bp_compute_start = get_time();
  bp_compute();
  m_bp_compute_time += get_time() - bp_compute_start;

  // Remove this layer as a gradient source for weight optimizers
  for (auto&& w : m_weights) {
    auto&& opt = w->get_optimizer();
    if (opt != nullptr) { opt->remove_gradient_source(this); }
  }

  #if defined(LBANN_HAS_CUDNN) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) { this->m_cudnn->check_error(); }
  #endif // defined(LBANN_HAS_CUDNN) && defined(LBANN_DEBUG)

  m_bp_time += get_time() - bp_start;
}

bool Layer::update() {
  if (m_frozen) { return true; }
  // Apply any updates.
  const auto update_compute_start = get_time();
  const auto layer_done = update_compute();
  m_update_time += get_time() - update_compute_start;
  return layer_done;
}

void Layer::reset_counters() {
  m_fp_time         = EvalType(0);
  m_fp_compute_time = EvalType(0);
  m_bp_time         = EvalType(0);
  m_bp_compute_time = EvalType(0);
  m_update_time     = EvalType(0);
}

void Layer::synchronize() const {
  #ifdef LBANN_HAS_CUDNN
  if (this->m_cudnn != nullptr) {
    this->m_cudnn->synchronize();
  }
  #endif // LBANN_HAS_CUDNN
}

void Layer::summarize_stats(lbann_summary& summarizer, int step) {
  std::string prefix = m_name + "/";
  summarizer.reduce_scalar(prefix + "fp_time", m_fp_time, step);
  summarizer.reduce_scalar(prefix + "bp_time", m_bp_time, step);
  summarizer.reduce_scalar(prefix + "update_time", m_update_time, step);
  reset_counters();
  // Combine the optimizer step time from all the weights.
  double step_time = 0.0;
  for (weights *w : get_weights()) {
    optimizer *opt = w->get_optimizer();
    if (opt) {
      step_time += opt->get_step_time();
      opt->reset_counters();
    }
  }
  summarizer.reduce_scalar(prefix + "opt_time", step_time, step);
}

void Layer::summarize_matrices(lbann_summary& summarizer, int step) {

  // Summarize activation matrices
  const int num_children = get_num_children();
  for (int i = 0; i < num_children; ++i) {
    AbsDistMatReadProxy<El::Device::CPU> acts(*m_activations[i]);
    std::string prefix = m_name + "/activations";
    if (num_children > 1) { prefix += std::to_string(i); }
    summarizer.reduce_mean(prefix + "/mean", acts.GetLocked(), step);
    summarizer.reduce_min(prefix + "/min", acts.GetLocked(), step);
    summarizer.reduce_max(prefix + "/max", acts.GetLocked(), step);
    summarizer.reduce_stdev(prefix + "/stdev", acts.GetLocked(), step);
    summarizer.reduce_2norm(prefix + "/2norm2", acts.GetLocked(), step);
  }

  // Summarize error signal matrices
  const int num_parents = get_num_parents();
  for (int i = 0; i < num_parents; ++i) {
    AbsDistMatReadProxy<El::Device::CPU> error_signals(*m_error_signals[i]);
    std::string prefix = m_name + "/error_signals";
    if (num_parents > 1) { prefix += std::to_string(i); }
    summarizer.reduce_mean(prefix + "/mean", error_signals.GetLocked(), step);
    summarizer.reduce_min(prefix + "/min", error_signals.GetLocked(), step);
    summarizer.reduce_max(prefix + "/max", error_signals.GetLocked(), step);
    summarizer.reduce_stdev(prefix + "/stdev", error_signals.GetLocked(), step);
    summarizer.reduce_2norm(prefix + "/2norm2", error_signals.GetLocked(), step);
  }

}

// Data matrix access functions
// Note: Using idiom from Item 3, p. 23 in "Effective C++", 3rd ed.,
// by Scott Meyers.
AbsDistMat& Layer::get_prev_activations(int parent_index) {
  return const_cast<AbsDistMat&>(static_cast<const Layer&>(*this).get_prev_activations(parent_index));
}
AbsDistMat& Layer::get_activations(int child_index) {
  return const_cast<AbsDistMat&>(static_cast<const Layer&>(*this).get_activations(child_index));
}
AbsDistMat& Layer::get_prev_error_signals(int child_index) {
  return const_cast<AbsDistMat&>(static_cast<const Layer&>(*this).get_prev_error_signals(child_index));
}
AbsDistMat& Layer::get_error_signals(int parent_index) {
  return const_cast<AbsDistMat&>(static_cast<const Layer&>(*this).get_error_signals(parent_index));
}
const AbsDistMat& Layer::get_prev_activations(int parent_index) const {
  if (parent_index < 0 || parent_index >= (int) m_prev_activations.size()) {
    std::stringstream err;
    err << "attempted to access invalid previous activation matrix "
        << "from " << m_name << " "
        << "(requested index " << parent_index << ", but there are "
        << m_prev_activations.size() << " previous activation matrices)";
    LBANN_ERROR(err.str());
  }
  return *m_prev_activations[parent_index];
}
const AbsDistMat& Layer::get_activations(int child_index) const {
  if (child_index < 0 || child_index >= (int) m_activations.size()) {
    std::stringstream err;
    err << "attempted to access invalid activation matrix "
        << "from " << m_name << " "
        << "(requested index " << child_index << ", but there are "
        << m_activations.size() << " activation matrices)";
    LBANN_ERROR(err.str());
  }
  return *m_activations[child_index];
}
const AbsDistMat& Layer::get_prev_error_signals(int child_index) const {
  if (child_index < 0 || child_index >= (int) m_prev_error_signals.size()) {
    std::stringstream err;
    err << "attempted to access invalid previous error signal matrix "
        << "from " << m_name << " "
        << "(requested index " << child_index << ", but there are "
        << m_prev_error_signals.size() << " previous error signal matrices)";
    LBANN_ERROR(err.str());
  }
  return *m_prev_error_signals[child_index];
}
const AbsDistMat& Layer::get_error_signals(int parent_index) const {
  if (parent_index < 0 || parent_index >= (int) m_error_signals.size()) {
    std::stringstream err;
    err << "attempted to access invalid error signal matrix "
        << "from " << m_name << " "
        << "(requested index " << parent_index << ", but there are "
        << m_error_signals.size() << " error signal matrices)";
    LBANN_ERROR(err.str());
  }
  return *m_error_signals[parent_index];
}
AbsMat& Layer::get_local_prev_activations(int parent_index) {
  return get_prev_activations(parent_index).Matrix();
}
AbsMat& Layer::get_local_activations(int child_index) {
  return get_activations(child_index).Matrix();
}
AbsMat& Layer::get_local_prev_error_signals(int child_index) {
  return get_prev_error_signals(child_index).Matrix();
}
AbsMat& Layer::get_local_error_signals(int parent_index) {
  return get_error_signals(parent_index).Matrix();
}
const AbsMat& Layer::get_local_prev_activations(int parent_index) const {
  return get_prev_activations(parent_index).LockedMatrix();
}
const AbsMat& Layer::get_local_activations(int child_index) const {
  return get_activations(child_index).LockedMatrix();
}
const AbsMat& Layer::get_local_prev_error_signals(int child_index) const {
  return get_prev_error_signals(child_index).LockedMatrix();
}
const AbsMat& Layer::get_local_error_signals(int parent_index) const {
  return get_error_signals(parent_index).LockedMatrix();
}

void Layer::clear_error_signals(int mini_batch_size) {
  for (int i = 0; i < get_num_parents(); ++i) {
    El::Zeros(get_error_signals(i), get_num_prev_neurons(i), mini_batch_size);
  }
}

void Layer::freeze() {
  m_frozen = true;
  for(auto& w : m_weights) {
    w->freeze();
  }
}

void Layer::unfreeze() {
  m_frozen = false;
  for(auto& w : m_weights) {
    w->unfreeze();
  }
}

bool Layer::is_frozen() const {
  for(auto& w : m_weights) {
    if (w->is_frozen() != m_frozen) {
      LBANN_ERROR("layer and weights of them are inconsistently frozen");
    }
  }
  return m_frozen;
}

void Layer::setup() {
  setup_pointers();
  setup_dims();
  setup_matrices(m_comm->get_model_grid());
  setup_data();
  if (using_gpus()) {
    if(m_cudnn == nullptr) {
      std::stringstream err;
      err << "layer " << m_name << " is trying to use GPUs but has an invalid pointer to the cudnn object";
      LBANN_ERROR(err.str());
    }
    setup_gpu();
  } else {
    m_cudnn = nullptr;
  }
}

void Layer::setup_pointers() {

  // Check if the number of parents/children are valid
  if(m_expected_num_parent_layers >= 0
     && get_num_parents() != m_expected_num_parent_layers) {
    std::stringstream err;
    err << "layer " << m_name << " has an invalid number of parent layers "
        << "(expected " << m_expected_num_parent_layers << ", "
        << "but found " << m_parent_layers.size() << ")";
    LBANN_ERROR(err.str());
  }
  if(m_expected_num_child_layers >= 0
     && get_num_children() != m_expected_num_child_layers) {
    std::stringstream err;
    err << "layer " << m_name << " has an invalid number of child layers "
        << "(expected " << m_expected_num_child_layers << ", "
        << "but found " << m_child_layers.size() << ")";
    LBANN_ERROR(err.str());
  }

}

void Layer::setup_dims() {

  // Get dimensions of previous neuron tensor
  if(m_parent_layers.empty()) {
    m_prev_neuron_dims.assign(1, 0);
  } else {
    m_prev_neuron_dims = m_parent_layers.front()->fp_output_dims(this);
  }
  m_num_prev_neuron_dims = m_prev_neuron_dims.size();
  m_num_prev_neurons = std::accumulate(m_prev_neuron_dims.begin(),
                                       m_prev_neuron_dims.end(),
                                       1,
                                       std::multiplies<int>());

  // Set neuron tensor dimensions equal to previous neuron tensor
  m_num_neurons = m_num_prev_neurons;
  m_num_neuron_dims = m_num_prev_neuron_dims;
  m_neuron_dims = m_prev_neuron_dims;

}

///************************************************************************
/// Instantiate CPU Matrices
///************************************************************************
template <>
void Layer::instantiate_matrices<data_layout::MODEL_PARALLEL, El::Device::CPU>(const El::Grid& grid) {
  for (int i = 0; i < get_num_parents(); ++i) {
    m_prev_activations.push_back(new MCMRMat<El::Device::CPU>(grid));
    m_error_signals.push_back(new MCMRMat<El::Device::CPU>(grid));
  }
  for (int i = 0; i < get_num_children(); ++i) {
    m_activations.push_back(new MCMRMat<El::Device::CPU>(grid));
    m_prev_error_signals.push_back(new MCMRMat<El::Device::CPU>(grid));
  }
//  m_using_gpus = false;
}

template <>
void Layer::instantiate_matrices<data_layout::DATA_PARALLEL, El::Device::CPU>(const El::Grid& grid) {
  for (int i = 0; i < get_num_parents(); ++i) {
    m_prev_activations.push_back(new StarVCMat<El::Device::CPU>(grid));
    m_error_signals.push_back(new StarVCMat<El::Device::CPU>(grid));
  }
  for (int i = 0; i < get_num_children(); ++i) {
    m_activations.push_back(new StarVCMat<El::Device::CPU>(grid));
    m_prev_error_signals.push_back(new StarVCMat<El::Device::CPU>(grid));
  }
//  m_using_gpus = false;
}

#ifdef LBANN_HAS_GPU
///************************************************************************
/// Instantiate GPU Matrices
///************************************************************************
template <>
void Layer::instantiate_matrices<data_layout::MODEL_PARALLEL, El::Device::GPU>(const El::Grid& grid) {
  for (int i = 0; i < get_num_parents(); ++i) {
    m_prev_activations.push_back(new MCMRMat<El::Device::GPU>(grid));
    m_error_signals.push_back(new MCMRMat<El::Device::GPU>(grid));
  }
  for (int i = 0; i < get_num_children(); ++i) {
    m_activations.push_back(new MCMRMat<El::Device::GPU>(grid));
    m_prev_error_signals.push_back(new MCMRMat<El::Device::GPU>(grid));
  }
  m_using_gpus = true;
}

template <>
void Layer::instantiate_matrices<data_layout::DATA_PARALLEL, El::Device::GPU>(const El::Grid& grid) {
  for (int i = 0; i < get_num_parents(); ++i) {
    m_prev_activations.push_back(new StarVCMat<El::Device::GPU>(grid));
    m_error_signals.push_back(new StarVCMat<El::Device::GPU>(grid));
  }
  for (int i = 0; i < get_num_children(); ++i) {
    m_activations.push_back(new StarVCMat<El::Device::GPU>(grid));
    m_prev_error_signals.push_back(new StarVCMat<El::Device::GPU>(grid));
  }
  m_using_gpus = true;
}
#endif // LBANN_HAS_GPU

void Layer::setup_matrices(const El::Grid& grid) {

  // Delete any previously allocated matrices
  deallocate_matrices();

  // Allocate input and output matrices for forward an back prop
  switch (get_data_layout()) {
  case data_layout::MODEL_PARALLEL:
    switch (get_device_allocation()) {
    case El::Device::CPU:
      instantiate_matrices<data_layout::MODEL_PARALLEL, El::Device::CPU>(grid); break;
#ifdef LBANN_HAS_GPU
    case El::Device::GPU:
      instantiate_matrices<data_layout::MODEL_PARALLEL, El::Device::GPU>(grid); break;
#endif // LBANN_HAS_GPU
    default:
      LBANN_ERROR("invalid matrix data allocation");
    }
    break;
  case data_layout::DATA_PARALLEL:
    switch (get_device_allocation()) {
    case El::Device::CPU:
      instantiate_matrices<data_layout::DATA_PARALLEL, El::Device::CPU>(grid); break;
#ifdef LBANN_HAS_GPU
    case El::Device::GPU:
      instantiate_matrices<data_layout::DATA_PARALLEL, El::Device::GPU>(grid); break;
#endif // LBANN_HAS_GPU
    default:
      LBANN_ERROR("invalid matrix data allocation");
    }
    break;
  default:
    LBANN_ERROR("invalid distributed matrix layout");
  }

}

void Layer::setup_data() {
  const int mini_batch_size = m_model->get_max_mini_batch_size();

  // Initialize previous activations
  for (int i = 0; i < get_num_parents(); ++i) {
    auto& fp_input = get_prev_activations(i);
    m_parent_layers[i]->get_fp_output(fp_input, this);
    const int expected_height = get_num_prev_neurons(i);
    if (fp_input.Height() != expected_height
        || fp_input.Width() != mini_batch_size) {
      std::stringstream err;
      err << "layer \"" << get_name() << "\" expected a "
          << expected_height << " x " << mini_batch_size << " matrix "
          << "from layer \"" << m_parent_layers[i]->get_name() << "\" "
          << "as forward prop input, but got a "
          << fp_input.Height() << " x " << fp_input.Width() << " matrix";
      LBANN_ERROR(err.str());
    }
  }

  // Initialize error signals
  for (int i = 0; i < get_num_parents(); ++i) {
    get_error_signals(i).Resize(get_num_prev_neurons(i), mini_batch_size);
  }


  // Initialize activations
  for (int i = 0; i < get_num_children(); ++i) {
    get_activations(i).Resize(get_num_neurons(i), mini_batch_size);
  }

}

void Layer::setup_gpu() {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else

  // Set tensor descriptors
  // Note: If the data layout is data-parallel, then the descriptors
  // describe the corresponding neuron tensors. If the data layout is
  // model-parallel, the descriptors describe the local matrix.
  if (get_num_parents() > 0) {
    const auto& input = get_prev_activations();
    const auto& gradient_wrt_input = get_error_signals();
    switch (get_data_layout()) {
    case data_layout::DATA_PARALLEL:
      cudnn::set_tensor_cudnn_desc(m_prev_activations_cudnn_desc,
                                   input.LocalWidth(),
                                   get_prev_neuron_dims(),
                                   input.LDim());
      cudnn::set_tensor_cudnn_desc(m_error_signals_cudnn_desc,
                                   gradient_wrt_input.LocalWidth(),
                                   get_prev_neuron_dims(),
                                   gradient_wrt_input.LDim());
      break;
    case data_layout::MODEL_PARALLEL:
      cudnn::set_tensor_cudnn_desc(m_prev_activations_cudnn_desc,
                                   input.LocalHeight(),
                                   input.LocalWidth(),
                                   input.LDim());
      cudnn::set_tensor_cudnn_desc(m_error_signals_cudnn_desc,
                                   gradient_wrt_input.LocalHeight(),
                                   gradient_wrt_input.LocalWidth(),
                                   gradient_wrt_input.LDim());
      break;
    default:
      LBANN_ERROR("invalid distributed matrix layout");
    }
  }
  if (get_num_children() > 0) {
    const auto& output = get_activations();
    switch (get_data_layout()) {
    case data_layout::DATA_PARALLEL:
      cudnn::set_tensor_cudnn_desc(m_activations_cudnn_desc,
                                   output.LocalWidth(),
                                   get_neuron_dims(),
                                   output.LDim());
      cudnn::set_tensor_cudnn_desc(m_prev_error_signals_cudnn_desc,
                                   get_activations().LocalWidth(),
                                   get_neuron_dims(),
                                   get_activations().LDim());
      break;
    case data_layout::MODEL_PARALLEL:
      cudnn::set_tensor_cudnn_desc(m_activations_cudnn_desc,
                                   output.LocalHeight(),
                                   output.LocalWidth(),
                                   output.LDim());
      cudnn::set_tensor_cudnn_desc(m_prev_error_signals_cudnn_desc,
                                   get_activations().LocalHeight(),
                                   get_activations().LocalWidth(),
                                   get_activations().LDim());
      break;
    default:
      LBANN_ERROR("invalid distributed matrix layout");
    }
  }

#endif // LBANN_HAS_CUDNN
}

void Layer::check_setup() {
  std::stringstream err;

  // Check that matrices matches number of parent/child layers
  const int num_parents = get_num_parents();
  const int num_children = get_num_children();
  if ((int) m_prev_activations.size() != num_parents
      || (int) m_activations.size() != num_children) {
    err << "layer " << m_name << " has an invalid number of "
        << "forward prop matrices (expected "
        << num_parents << " input and " << num_children << " output, "
        << "but found " << m_prev_activations.size() << " and "
        << m_activations.size() << " respectively) ";
    LBANN_ERROR(err.str());
  }
  if ((int) m_prev_error_signals.size() != num_children
      || (int) m_error_signals.size() != num_parents) {
    err << "layer " << m_name << " has an invalid number of "
        << "backward prop matrices (expected "
        << num_children << " input and " << num_parents << " output. "
        << "but found " << m_prev_error_signals.size() << " and "
        << m_error_signals.size() << " respectively) ";
    LBANN_ERROR(err.str());
  }

  // Check that matrices are initialized
  for (const auto& m : m_prev_activations) {
    if (m == nullptr) {
      err << "layer " << m_name << " has an uninitialized previous activation matrix";
      LBANN_ERROR(err.str());
    }
  }
  for (const auto& m : m_activations) {
    if (m == nullptr) {
      err << "layer " << m_name << " has an uninitialized activation matrix";
      LBANN_ERROR(err.str());
    }
  }
  for (const auto& m : m_prev_error_signals) {
    if (m == nullptr) {
      err << "layer " << m_name << " has an uninitialized previous error signal matrix";
      LBANN_ERROR(err.str());
    }
  }
  for (const auto& m : m_error_signals) {
    if (m == nullptr) {
      err << "layer " << m_name << " has an uninitialized error signal matrix";
      LBANN_ERROR(err.str());
    }
  }

  // Check that number of neurons is greater than zero
  if (m_num_neurons <= 0) {
    err << "layer " << m_name << " has invalid output dimensions "
        << "(" << m_neuron_dims[0];
    for (size_t i = 1; i < m_neuron_dims.size(); ++i) {
      err << "x" << m_neuron_dims[i];
    }
    err << ")";
    LBANN_ERROR(err.str());
  }

}

void Layer::replace_weights(Layer* other_layer) {
  if (other_layer == nullptr) {
    LBANN_ERROR("attempted to add null pointer as a replacement layer");
  }

  const std::vector<weights *> other_layer_weights = other_layer->get_weights();
  for (size_t i = 0; i < m_weights.size(); ++i) {
    m_weights[i]->set_values(other_layer_weights[i]->get_values());
  }

}

void Layer::deallocate_matrices() {

  // Deallocate matrices
  for (const auto& m : m_prev_activations) {
    if (m != nullptr) delete m;
  }
  for (const auto& m : m_activations) {
    if (m != nullptr) delete m;
  }
  for (const auto& m : m_prev_error_signals) {
    if (m != nullptr) delete m;
  }
  for (const auto& m : m_error_signals) {
    if (m != nullptr) delete m;
  }
  m_prev_activations.clear();
  m_activations.clear();
  m_prev_error_signals.clear();
  m_error_signals.clear();

}


bool Layer::save_to_checkpoint_shared(persist& p) const {
  return true;
}

bool Layer::load_from_checkpoint_shared(persist& p) {
  return true;
}

bool Layer::save_to_checkpoint_distributed(persist& p) const {
  return true;
}

bool Layer::load_from_checkpoint_distributed(persist& p) {
  return true;
}

void Layer::write_proto(lbann_data::Layer* proto) const {
  proto->Clear();
  proto->set_name(get_name());
  proto->set_type(get_type());
  if(!m_parent_layers.empty()) proto->set_bottom(m_parent_layers.front()->get_name());
  proto->set_top(get_name());
  //Add weights
  for (weights *w : m_weights) {
    auto weight_proto = proto->add_weights_data();
    w->write_proto(weight_proto);
  }
}

void Layer::fp_setup_data(int mini_batch_size) {

  // Initialize previous activations
  for (int i = 0; i < get_num_parents(); ++i) {

    // Get previous activation from parent layer
    const auto& parent = m_parent_layers[i];
    parent->get_fp_output(get_prev_activations(i), this);

    // Check dimensions of previous activations matrix
    const int expected_height = get_num_prev_neurons(i);
    const auto& input = get_prev_activations(i);
    if (input.Height() != expected_height
        || input.Width() != mini_batch_size) {
      std::stringstream err;
      err << "layer \"" << get_name() << "\" expected a "
          << expected_height << " x " << mini_batch_size
          << " input matrix from layer \"" << parent->get_name() << "\""
          << " during forward prop, but got a "
          << input.Height() << " x " << input.Width() << " matrix";
      LBANN_ERROR(err.str());
    }

  }

  // Initialize activations
  for (int i = 0; i < get_num_children(); ++i) {
    auto& activations = get_activations(i);
    const auto num_neurons = get_num_neurons(i);
    if (activations.Height() != num_neurons
        || activations.Width() != mini_batch_size) {
      activations.Empty(false); // Reset matrix views (without deallocating memory)
      activations.Resize(num_neurons, mini_batch_size);
    }
  }

  #ifdef LBANN_HAS_CUDNN
  // Set cuDNN tensor descriptors if needed
  // Note: If the data layout is data-parallel, then the descriptors
  // describe the corresponding neuron tensors. If the data layout is
  // model-parallel, the descriptors describe the local matrix.
  if (using_gpus()) {
    if (get_num_parents() > 0) {
      const auto& input = get_prev_activations();
      switch (get_data_layout()) {
      case data_layout::DATA_PARALLEL:
        cudnn::set_tensor_cudnn_desc(m_prev_activations_cudnn_desc,
                                     input.LocalWidth(),
                                     get_prev_neuron_dims(),
                                     input.LDim());
        break;
      case data_layout::MODEL_PARALLEL:
        cudnn::set_tensor_cudnn_desc(m_prev_activations_cudnn_desc,
                                     input.LocalHeight(),
                                     input.LocalWidth(),
                                     input.LDim());
        break;
      default:
        LBANN_ERROR("invalid distributed matrix layout");
      }
    }
    if (get_num_children() > 0) {
      const auto& output = get_activations();
      switch (get_data_layout()) {
      case data_layout::DATA_PARALLEL:
        cudnn::set_tensor_cudnn_desc(m_activations_cudnn_desc,
                                     output.LocalWidth(),
                                     get_neuron_dims(),
                                     output.LDim());
        break;
      case data_layout::MODEL_PARALLEL:
        cudnn::set_tensor_cudnn_desc(m_activations_cudnn_desc,
                                     output.LocalHeight(),
                                     output.LocalWidth(),
                                     output.LDim());
        break;
      default:
        LBANN_ERROR("invalid distributed matrix layout");
      }
    }
  }
  #endif // LBANN_HAS_CUDNN

}

void Layer::bp_setup_data(int mini_batch_size) {

  // Initialize previous error signals
  for (int i = 0; i < get_num_children(); ++i) {
    const auto& child = m_child_layers[i];

    // Get previous error signal from child layer
    auto& bp_input = get_prev_error_signals(i);
    child->get_bp_output(bp_input, this);
    const int expected_height = get_num_neurons(i);
    if (bp_input.Height() != expected_height
        || bp_input.Width() != mini_batch_size) {
      std::stringstream err;
      err << "layer \"" << get_name() << "\" expected a "
          << expected_height << " x " << mini_batch_size
          << " input matrix from layer \"" << child->get_name() << "\""
          << " during backward prop, but got a "
          << bp_input.Height() << " x " << bp_input.Width() << " matrix";
      LBANN_ERROR(err.str());
    }

  }

  #ifdef LBANN_HAS_CUDNN
  // Set cuDNN tensor descriptors if needed
  // Note: If the data layout is data-parallel, then the descriptors
  // describe the corresponding neuron tensors. If the data layout is
  // model-parallel, the descriptors describe the local matrix.
  if (using_gpus()) {
    if (get_num_children() > 0) {
      const auto& gradient_wrt_output = get_prev_error_signals();
      switch (get_data_layout()) {
      case data_layout::DATA_PARALLEL:
        cudnn::set_tensor_cudnn_desc(m_prev_error_signals_cudnn_desc,
                                     gradient_wrt_output.LocalWidth(),
                                     get_neuron_dims(),
                                     gradient_wrt_output.LDim());
        break;
      case data_layout::MODEL_PARALLEL:
        cudnn::set_tensor_cudnn_desc(m_prev_error_signals_cudnn_desc,
                                     gradient_wrt_output.LocalHeight(),
                                     gradient_wrt_output.LocalWidth(),
                                     gradient_wrt_output.LDim());
        break;
      default:
        LBANN_ERROR("invalid distributed matrix layout");
      }
    }
    if (get_num_parents() > 0) {
      const auto& gradient_wrt_input = get_error_signals();
      switch (get_data_layout()) {
      case data_layout::DATA_PARALLEL:
        cudnn::set_tensor_cudnn_desc(m_error_signals_cudnn_desc,
                                     gradient_wrt_input.LocalWidth(),
                                     get_prev_neuron_dims(),
                                     gradient_wrt_input.LDim());
        break;
      case data_layout::MODEL_PARALLEL:
        cudnn::set_tensor_cudnn_desc(m_error_signals_cudnn_desc,
                                     gradient_wrt_input.LocalHeight(),
                                     gradient_wrt_input.LocalWidth(),
                                     gradient_wrt_input.LDim());
        break;
      default:
        LBANN_ERROR("invalid distributed matrix layout");
      }
    }
  }
  #endif // LBANN_HAS_CUDNN

}


#ifdef LBANN_HAS_CUDNN
void Layer::pin_data() {
  for (int i = 0; i < get_num_parents(); ++i) {
    const auto& parent = *m_parent_layers[i];
    if (using_gpus() && !parent.using_gpus()) {
      m_cudnn->pin_matrix(get_error_signals(i));
      if (get_prev_activations().DistData()
          != parent.get_activations().DistData()) {
        m_cudnn->pin_matrix(get_prev_activations(i));
      }
    }
  }
  for (int i = 0; i < get_num_children(); ++i) {
    const auto& child = *m_child_layers[i];
    if (using_gpus() && !child.using_gpus()) {
      m_cudnn->pin_matrix(get_activations(i));
      if (get_data_layout() != child.get_data_layout()) {
        m_cudnn->pin_matrix(get_prev_error_signals(i));
      }
    }
  }
}

#endif // LBANN_HAS_CUDNN

void Layer::get_fp_output(AbsDistMat& output, const Layer* child) const {

  // Get activation matrix corresponding to child layer
  // Note: the const_cast is morally dubious, but it should be
  // unnecessary once Hydrogen supports GPU memory copies.
  const size_t child_index = (std::find(m_child_layers.begin(),
                                        m_child_layers.end(),
                                        child)
                              - m_child_layers.begin());
  if (child_index >= m_child_layers.size()) {
    std::stringstream err;
    err << get_name() << " has no forward prop output corresponding to "
        << child->get_name();
    LBANN_ERROR(err.str());
  }
  auto& activation = const_cast<Layer*>(this)->get_activations(child_index);

  // Put view or copy of activation matrix in output matrix
  if(activation.DistData() == output.DistData()) {
    El::LockedView(output, activation);
  }
  else {
    El::Copy(activation, output);
  }

}

void Layer::get_bp_output(AbsDistMat& output, const Layer* parent) const {

  // Get error signal matrix corresponding to parent layer
  // Note: the const_cast is morally dubious, but it should be
  // unnecessary once Hydrogen supports GPU memory copies.
  const size_t parent_index = (std::find(m_parent_layers.begin(),
                                         m_parent_layers.end(),
                                         parent)
                               - m_parent_layers.begin());
  if (parent_index >= m_parent_layers.size()) {
    std::stringstream err;
    err << get_name() << " has no backward prop output corresponding to "
        << parent->get_name();
    LBANN_ERROR(err.str());
  }
  auto& error_signal = const_cast<Layer*>(this)->get_error_signals(parent_index);

  // Put view or copy of error signal matrix in output matrix
  if(error_signal.DistData() == output.DistData()) {
    El::LockedView(output, error_signal);
  }
  else {
    El::Copy(error_signal, output);
  }

}

std::string Layer::get_data_layout_string(data_layout d) const {
  switch(d) {
  case data_layout::DATA_PARALLEL:
    return "data_parallel";
  case data_layout::MODEL_PARALLEL:
    return "model_parallel";
  default:
    LBANN_ERROR("invalid data layout");
  }
}

std::string Layer::get_device_allocation_string(El::Device dev) const {
  switch(dev) {
  case El::Device::CPU:
    return "cpu";
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    return "gpu";
#endif // LBANN_HAS_GPU
  default:
    LBANN_ERROR("invalid device allocation");
  }
}

std::string Layer::get_device_allocation_string_short(El::Device dev) const {
  switch(dev) {
  case El::Device::CPU:
    return "C";
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    return "G";
#endif // LBANN_HAS_GPU
  default:
    LBANN_ERROR("invalid device allocation");
  }
}

std::string Layer::get_layer_names(const std::vector<const Layer*>& list) {
  std::string layer_names = ((list.size()==0u || !list[0])? "" : list[0]->get_name());

  for (size_t i=1u; i < list.size(); ++i) {
    if (list[i]) layer_names += ", " + list[i]->get_name();
  }
  return layer_names;
}

void Layer::add_parent_layer(const Layer* parent) {
  auto parent_pos = std::find(m_parent_layers.begin(),
                              m_parent_layers.end(),
                              parent);
  if(parent != nullptr
     && parent != this
     && parent_pos == m_parent_layers.end()) {
    m_parent_layers.push_back(parent);
  }
}

void Layer::add_child_layer(const Layer* child) {
  auto child_pos = std::find(m_child_layers.begin(),
                             m_child_layers.end(),
                             child);
  if(child != nullptr
     && child != this
     && child_pos == m_child_layers.end()) {
    m_child_layers.push_back(child);
  }
}

std::vector<Layer*> Layer::get_layer_pointers() {
  std::vector<Layer*> layers;
  for(const Layer* parent: m_parent_layers) {
    layers.push_back(const_cast<Layer*>(parent));
  }
  for(const Layer* child: m_child_layers) {
    layers.push_back(const_cast<Layer*>(child));
  }
  return layers;
}

void Layer::set_layer_pointers(std::vector<Layer*> layers) {
  if(layers.size() != m_parent_layers.size() + m_child_layers.size()) {
    LBANN_ERROR("attempted to set layer pointers with an invalid number of pointers");
  }
  size_t pos = 0;
  for(const Layer*& parent: m_parent_layers) {
    parent = (const Layer*) layers[pos];
    pos++;
  }
  for(const Layer*& child: m_child_layers) {
    child = (const Layer*) layers[pos];
    pos++;
  }
}

#ifdef LBANN_HAS_DISTCONV
void Layer::setup_tensor_distribution_init(
    std::map<const Layer*, std::array<Dist, 4>> &dists,
    std::map<Dist*, std::set<Dist*>> &invariants,
    std::set<Dist*> &updated,
    std::set<Dist*> &fixed) {
  Dist dist = using_distconv() ?
      Dist({1, m_comm->get_procs_per_model(), 1, 1}) :
      Dist({1, 1, 1, m_comm->get_procs_per_model()});
  std::array<Dist, 4> layer_dists = {dist, dist, dist, dist};
  dists.insert(std::make_pair(this, layer_dists));
  invariants.insert(std::make_pair(&dists[this][0], std::set<Dist*>()));
  invariants.insert(std::make_pair(&dists[this][1], std::set<Dist*>()));
  invariants.insert(std::make_pair(&dists[this][2], std::set<Dist*>()));
  invariants.insert(std::make_pair(&dists[this][3], std::set<Dist*>()));
}

void Layer::setup_tensor_distribution_add_adjacent_invariants(
    std::map<const Layer*, std::array<Dist, 4>> &dists,
    std::map<Dist*, std::set<Dist*>> &invariants) {
  if (!using_distconv()) return;
  auto &layer_dists = dists[this];      
  if (get_child_layers().size() > 0) {
    auto child = get_child_layers()[0];
    if (child->using_distconv()) {
      invariants[&layer_dists[1]].insert(
          &dists[child][0]);
      invariants[&layer_dists[3]].insert(
          &dists[child][2]);
    }
  }
  if (get_parent_layers().size() > 0) {
    auto parent = get_parent_layers()[0];
    if (parent->using_distconv()) {
      invariants[&layer_dists[0]].insert(
          &dists[parent][1]);
      invariants[&layer_dists[2]].insert(
          &dists[parent][3]);
    }
  }
}

void Layer::setup_tensor_distribution_block() {
  m_input_decomposition_block = Array4(1);  
  m_output_decomposition_block = Array4(1);
  // Disable as we don't need to enforce divisible boundaries
#if 0
  if (using_distconv()) {
    const auto *child = get_child_layers()[0];
    if (child->using_distconv()) {
      m_output_decomposition_block =
          child->get_input_decomposition_block();
    }
    m_input_decomposition_block =
        m_output_decomposition_block * get_strides();
  }
#endif
}

void Layer::setup_tensors_fwd(const std::array<Dist, 4> &dists) {
  m_distconv_enabled = using_distconv();

  if (!m_distconv_enabled) {
    MPIPrintStreamInfo() << get_name() << ": distconv disabled\n";
    return;
  }


  MPIPrintStreamInfo() << get_name() << ": distconv enabled\n";    
  const auto &child_layers = get_child_layers();
  MPIPrintStreamDebug() << ": number of children: "
                        << child_layers.size()
                        << ", child name: " << child_layers[0]->get_name()
                        << "\n";
  if (child_layers.size() == 1 &&
      child_layers[0]->using_distconv()) {
    m_child_copy_required = false;
  }
  MPIPrintStreamDebug() << "m_child_copy_required: "
                        << m_child_copy_required << "\n";
  const auto &parent_layers = get_parent_layers();
  MPIPrintStreamDebug() << ": number of parents: "
                        << parent_layers.size()
                        << ", parent name: " << parent_layers[0]->get_name()
                        << "\n";
  if (parent_layers.size() == 1 &&
      parent_layers[0]->using_distconv()) {
    m_parent_copy_required = false;
  }
  MPIPrintStreamDebug() << "m_parent_copy_required: "
                        << m_parent_copy_required << "\n";
}

void Layer::setup_prev_activations_tensor(const std::array<Dist, 4> &dists) {
  const Array4 input_tensor_shape =
      {m_prev_neuron_dims[2], m_prev_neuron_dims[1],
       m_prev_neuron_dims[0], this->m_model->get_max_mini_batch_size()};
  const LocaleMPI loc(m_comm->get_model_comm().comm, false);
  const Array4 sample_block_size = {1, 1, 1, 1};      
  const Dist sample_dist = Dist({1, 1, 1, m_comm->get_procs_per_model()});
  Array4 input_local_shape = input_tensor_shape;
  // Assuming single GPU per rank
  input_local_shape[3] = m_max_mini_batch_size_per_gpu;
  const Array4 spatial_local_size = {0, 0, 0, 0};
  
  if (m_parent_copy_required) {
    m_prev_activations_const_view = TensorDev(input_tensor_shape, loc,
                                              sample_dist,
                                              input_local_shape,
                                              sample_block_size);
    m_prev_activations_t = TensorDev(input_tensor_shape, loc, dists[0],
                                     spatial_local_size, m_input_decomposition_block);
    assert0(m_prev_activations_t.allocate());
    m_prev_activations_t.zero();
    m_prev_activations_shuffler = new TensorShuffler(
        m_prev_activations_const_view, m_prev_activations_t);
  } else {
    m_prev_activations_t = get_parent_layers()[0]->get_activations_t();
    assert_always(m_prev_activations_t.get_distribution() == dists[0]);
    assert_always(m_prev_activations_t.get_requested_local_block()
                  == m_input_decomposition_block);
  }
}

Array4 Layer::get_activations_tensor_local_shape() const {
  return m_prev_activations_t.get_local_shape();
}

void Layer::setup_activations_tensor(const std::array<Dist, 4> &dists) {
  const LocaleMPI loc(m_comm->get_model_comm().comm, false);
  const Array4 output_tensor_shape =
      {m_neuron_dims[2], m_neuron_dims[1],
       m_neuron_dims[0], this->m_model->get_max_mini_batch_size()};
  const Array4 &activations_local_shape =
      get_activations_tensor_local_shape();
  m_activations_t = TensorDev(output_tensor_shape,
                              loc, dists[1], activations_local_shape,
                              m_output_decomposition_block);
  assert0(m_activations_t.allocate());
  m_activations_t.zero();
}

void Layer::setup_activations_copyout_tensor(const std::array<Dist, 4> &dists) {
  const LocaleMPI loc(m_comm->get_model_comm().comm, false);
  const Array4 sample_block_size = {1, 1, 1, 1};    
  const Dist sample_dist = Dist({1, 1, 1, m_comm->get_procs_per_model()});
  const Array4 output_tensor_shape =
      {m_neuron_dims[2], m_neuron_dims[1],
       m_neuron_dims[0], this->m_model->get_max_mini_batch_size()};
  Array4 output_local_shape = output_tensor_shape;
  output_local_shape[3] = m_max_mini_batch_size_per_gpu;
  m_activations_copyout = TensorDev(output_tensor_shape, loc, sample_dist,
                                    output_local_shape, sample_block_size);
  if (m_child_copy_required) {
    m_activations_shuffler = new TensorShuffler(
        m_activations_t, m_activations_copyout);
  }
}

void Layer::setup_tensors_bwd(const std::array<Dist, 4> &dists) {
}

Array4 Layer::get_prev_activations_overlap() const {
  return Array4(0);  
}

Array4 Layer::get_activations_overlap() const {
#if 0  
  if (using_distconv() &&
      get_child_layers().size() > 0) {
    return get_child_layers()[0]->get_prev_activations_overlap();
  } else {
    return Array4(0);
  }
#endif
    return Array4(0);  
}

Array4 Layer::get_prev_error_signals_overlap() const {
  return Array4(0);
}

Array4 Layer::get_error_signals_overlap() const {
#if 0  
  if (using_distconv() &&
      get_parent_layers().size() > 0) {
    return get_parent_layers()[0]->get_prev_error_signals_overlap();
  } else {
    return Array4(0);
  }
#endif
  return Array4(0);  
}

Array4 Layer::get_input_decomposition_block() const {
  return m_input_decomposition_block;
}

Array4 Layer::get_output_decomposition_block() const {
  return m_output_decomposition_block;
}

const TensorDev &Layer::get_activations_t() const {
  return m_activations_t;
}

const TensorDev &Layer::get_error_signals_t() const {
  return m_error_signals_t;
}

Array4 Layer::get_strides() const {
  return Array4(1);
}

#endif

}  // namespace lbann
