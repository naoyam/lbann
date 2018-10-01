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

#include "lbann/utils/cuda.hpp"

namespace lbann {

Layer::Layer(lbann_comm *comm)
  : m_comm(comm),
    m_frozen(false) {

  // Initialize layer name
  static int num_layers = 0;
  m_name = "layer" + std::to_string(num_layers);
  num_layers++;

  // Reset timing counters
  reset_counters();

}

Layer::Layer(const Layer& other) :
  m_comm(other.m_comm),
  m_weights(other.m_weights),
  m_parent_layers(other.m_parent_layers),
  m_child_layers(other.m_child_layers),
  m_expected_num_parent_layers(other.m_expected_num_parent_layers),
  m_expected_num_child_layers(other.m_expected_num_child_layers),
  m_model(other.m_model),
  m_frozen(other.m_frozen),
  m_fp_time(other.m_fp_time),
  m_fp_compute_time(other.m_fp_compute_time),
  m_bp_time(other.m_bp_time),
  m_bp_compute_time(other.m_bp_compute_time),
  m_update_time(other.m_update_time),
  m_name(other.m_name),
  m_output_dims_list(other.m_output_dims_list) {

  // Deep matrix copies
  m_inputs.reserve(other.m_inputs.size());
  m_outputs.reserve(other.m_outputs.size());
  m_gradient_wrt_outputs.reserve(other.m_gradient_wrt_outputs.size());
  m_gradient_wrt_inputs.reserve(other.m_gradient_wrt_inputs.size());
  for (const auto& ptr : other.m_inputs) {
    m_inputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }
  for (const auto& ptr : other.m_outputs) {
    m_outputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }
  for (const auto& ptr : other.m_gradient_wrt_outputs) {
    m_gradient_wrt_outputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }
  for (const auto& ptr : other.m_gradient_wrt_inputs) {
    m_gradient_wrt_inputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }

}

Layer& Layer::operator=(const Layer& other) {

  // Shallow copies
  m_comm = other.m_comm;
  m_weights = other.m_weights;
  m_parent_layers = other.m_parent_layers;
  m_child_layers = other.m_child_layers;
  m_expected_num_parent_layers = other.m_expected_num_parent_layers;
  m_expected_num_child_layers = other.m_expected_num_child_layers;
  m_model = other.m_model;
  m_frozen = other.m_frozen;
  m_fp_time = other.m_fp_time;
  m_fp_compute_time = other.m_fp_compute_time;
  m_bp_time = other.m_bp_time;
  m_bp_compute_time = other.m_bp_compute_time;
  m_update_time = other.m_update_time;
  m_name = other.m_name;
  m_output_dims_list = other.m_output_dims_list;

  // Deep matrix copies
  m_inputs.clear();
  m_outputs.clear();
  m_gradient_wrt_outputs.clear();
  m_gradient_wrt_inputs.clear();
  m_inputs.reserve(other.m_inputs.size());
  m_outputs.reserve(other.m_outputs.size());
  m_gradient_wrt_outputs.reserve(other.m_gradient_wrt_outputs.size());
  m_gradient_wrt_inputs.reserve(other.m_gradient_wrt_inputs.size());
  for (const auto& ptr : other.m_inputs) {
    m_inputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }
  for (const auto& ptr : other.m_outputs) {
    m_outputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }
  for (const auto& ptr : other.m_gradient_wrt_outputs) {
    m_gradient_wrt_outputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }
  for (const auto& ptr : other.m_gradient_wrt_inputs) {
    m_gradient_wrt_inputs.emplace_back(ptr ? nullptr : ptr->Copy());
  }

  return *this;
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
    const auto& dims = get_output_dims(i);
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

  // Setup tensors
  const auto& mini_batch_size = m_model->get_current_mini_batch_size();
  fp_setup_inputs(mini_batch_size);
  fp_setup_outputs(mini_batch_size);

#if defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) { El::GPUManager::SynchronizeDevice(true); }
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)

#ifdef LBANN_HAS_DISTCONV
  fp_setup_distconv(m_model->get_current_mini_batch_size());
#endif

  // Apply layer's compute function
  const auto fp_compute_start = get_time();
  fp_compute();
  m_fp_compute_time += get_time() - fp_compute_start;

#ifdef LBANN_HAS_DISTCONV
  if (early_terminate_last_iteration()) {
    dump_activations();
  }
#endif

  // Add this layer as a gradient source for weight optimizers
  for (auto&& w : m_weights) {
    optimizer* opt = w->get_optimizer();
    if (opt != nullptr) { opt->add_gradient_source(this); }
  }

#if defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) { El::GPUManager::SynchronizeDevice(true); }
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)

  m_fp_time += get_time() - fp_start;
}

void Layer::back_prop() {
  const auto bp_start = get_time();

  // Setup tensors
  const auto& mini_batch_size = m_model->get_current_mini_batch_size();
  bp_setup_gradient_wrt_outputs(mini_batch_size);
  bp_setup_gradient_wrt_inputs(mini_batch_size);

#if defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) { El::GPUManager::SynchronizeDevice(true); }
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)

#ifdef LBANN_HAS_DISTCONV
  bp_setup_distconv(m_model->get_current_mini_batch_size());
#endif

  // Backprop the compute function.
  const auto bp_compute_start = get_time();
  bp_compute();
  m_bp_compute_time += get_time() - bp_compute_start;

#ifdef LBANN_HAS_DISTCONV
  if (early_terminate_last_iteration()) {
    dump_error_signals();
  }
#endif

  // Remove this layer as a gradient source for weight optimizers
  for (auto&& w : m_weights) {
    auto&& opt = w->get_optimizer();
    if (opt != nullptr) { opt->remove_gradient_source(this); }
  }

#if defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)
  // Synchronize GPUs and check for errors
  if (using_gpus()) { El::GPUManager::SynchronizeDevice(true); }
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_DEBUG)

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

void Layer::summarize_stats(lbann_summary& summarizer, int step) {
  std::string prefix = m_name + "/";
  summarizer.reduce_scalar(prefix + "fp_time", m_fp_time, step);
  summarizer.reduce_scalar(prefix + "bp_time", m_bp_time, step);
  summarizer.reduce_scalar(prefix + "update_time", m_update_time, step);
  summarizer.reduce_scalar_all(prefix + "fp_time", m_fp_time, step);
  summarizer.reduce_scalar_all(prefix + "bp_time", m_bp_time, step);
  summarizer.reduce_scalar_all(prefix + "update_time", m_update_time, step);
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
  summarizer.reduce_scalar_all(prefix + "opt_time", step_time, step);
}

void Layer::summarize_matrices(lbann_summary& summarizer, int step) {

  // Summarize activation matrices
  const int num_children = get_num_children();
  for (int i = 0; i < num_children; ++i) {
    AbsDistMatReadProxy<El::Device::CPU> acts(*m_outputs[i]);
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
    AbsDistMatReadProxy<El::Device::CPU> error_signals(*m_gradient_wrt_inputs[i]);
    std::string prefix = m_name + "/error_signals";
    if (num_parents > 1) { prefix += std::to_string(i); }
    summarizer.reduce_mean(prefix + "/mean", error_signals.GetLocked(), step);
    summarizer.reduce_min(prefix + "/min", error_signals.GetLocked(), step);
    summarizer.reduce_max(prefix + "/max", error_signals.GetLocked(), step);
    summarizer.reduce_stdev(prefix + "/stdev", error_signals.GetLocked(), step);
    summarizer.reduce_2norm(prefix + "/2norm2", error_signals.GetLocked(), step);
  }

}


// ===================================================================
// Tensor dimension access functions
// ===================================================================

std::vector<int> Layer::get_input_dims(int input_index) const {

  // Get parent layer
  const auto& num_inputs = get_num_parents();
  if (input_index < 0 || input_index >= num_inputs) {
    std::stringstream err;
    err << "attempted to access dimensions of invalid input tensor "
        << "in layer \"" << get_name() << "\" "
        << "(requested index " << input_index << ", but there are "
        << num_inputs << " input tensors)";
    LBANN_ERROR(err.str());
  } else if (m_parent_layers[input_index] == nullptr) {
    std::stringstream err;
    err << "layer \"" << get_name() << "\" "
        << "has a null pointer to parent layer "
        << "(index " << input_index << ")";
    LBANN_ERROR(err.str());
  }
  const auto& parent = *m_parent_layers[input_index];

  // Get dimensions of corresponding output tensor in parent layer
  const auto num_parent_outputs = parent.get_num_children();
  const int parent_output_index = (std::find(parent.m_child_layers.begin(),
                                             parent.m_child_layers.end(),
                                             this)
                                   - parent.m_child_layers.begin());
  if (parent_output_index >= num_parent_outputs) {
    std::stringstream err;
    err << "layer \"" << parent.get_name() << "\" is a parent of "
        << "layer \"" << get_name() << "\", but "
        << "\"" << get_name() << "\" is not a child of "
        << "\"" << parent.get_name() << "\"";
    LBANN_ERROR(err.str());
  }
  return parent.get_output_dims(parent_output_index);

}

int Layer::get_input_size(int input_index) const {
  const auto& dims = get_input_dims(input_index);
  if (dims.empty()) {
    return 0;
  } else {
    return std::accumulate(dims.begin(), dims.end(), 1,
                           std::multiplies<int>());
  }
}

std::vector<int> Layer::get_output_dims(int output_index) const {
  const auto num_outputs = get_num_children();
  if ((int) m_output_dims_list.size() != num_outputs) {
    std::stringstream err;
    err << "attempted to access dimensions of output tensor "
        << "in layer \"" << get_name() << "\" "
        << "before they are initialized";
    LBANN_ERROR(err.str());
  } else if (output_index < 0 || output_index >= num_outputs) {
    std::stringstream err;
    err << "attempted to access dimensions of invalid output tensor "
        << "in layer \"" << get_name() << "\" "
        << "(requested index " << output_index << ", but there are "
        << num_outputs << " output tensors)";
    LBANN_ERROR(err.str());
  }
  return m_output_dims_list[output_index];
}

int Layer::get_output_size(int output_index) const {
  const auto& dims = get_output_dims(output_index);
  if (dims.empty()) {
    return 0;
  } else {
    return std::accumulate(dims.begin(), dims.end(), 1,
                           std::multiplies<int>());
  }
}

void Layer::set_output_dims(std::vector<int> dims, int output_index) {
  if ((int) m_output_dims_list.size() != get_num_children()
      || (int) m_output_dims_list.size() <= output_index) {
    // Handles case where dims are set before child layers are set
    m_output_dims_list.resize(std::max(get_num_children(),
                                       output_index + 1));
  }
  m_output_dims_list[output_index] = dims;
}

// ===================================================================
// Tensor access functions
// ===================================================================

// Accessing distributed matrices
const AbsDistMat& Layer::get_prev_activations(int parent_index) const {
  if (parent_index < 0 || parent_index >= (int) m_inputs.size()) {
    std::stringstream err;
    err << "attempted to access invalid previous activation matrix "
        << "from " << m_name << " "
        << "(requested index " << parent_index << ", but there are "
        << m_inputs.size() << " previous activation matrices)";
    LBANN_ERROR(err.str());
  }
  return *m_inputs[parent_index];
}
const AbsDistMat& Layer::get_activations(int child_index) const {
  if (child_index < 0 || child_index >= (int) m_outputs.size()) {
    std::stringstream err;
    err << "attempted to access invalid activation matrix "
        << "from " << m_name << " "
        << "(requested index " << child_index << ", but there are "
        << m_outputs.size() << " activation matrices)";
    LBANN_ERROR(err.str());
  }
  return *m_outputs[child_index];
}
const AbsDistMat& Layer::get_prev_error_signals(int child_index) const {
  if (child_index < 0 || child_index >= (int) m_gradient_wrt_outputs.size()) {
    std::stringstream err;
    err << "attempted to access invalid previous error signal matrix "
        << "from " << m_name << " "
        << "(requested index " << child_index << ", but there are "
        << m_gradient_wrt_outputs.size() << " previous error signal matrices)";
    LBANN_ERROR(err.str());
  }
  return *m_gradient_wrt_outputs[child_index];
}
const AbsDistMat& Layer::get_error_signals(int parent_index) const {
  if (parent_index < 0 || parent_index >= (int) m_gradient_wrt_inputs.size()) {
    std::stringstream err;
    err << "attempted to access invalid error signal matrix "
        << "from " << m_name << " "
        << "(requested index " << parent_index << ", but there are "
        << m_gradient_wrt_inputs.size() << " error signal matrices)";
    LBANN_ERROR(err.str());
  }
  return *m_gradient_wrt_inputs[parent_index];
}

// Accessing non-const distributed matrices
// Note: Using idiom from Item 3, p. 23 in "Effective C++", 3rd ed.,
// by Scott Meyers.
AbsDistMat& Layer::get_activations(int child_index) {
  return const_cast<AbsDistMat&>(static_cast<const Layer&>(*this).get_activations(child_index));
}
AbsDistMat& Layer::get_error_signals(int parent_index) {
  return const_cast<AbsDistMat&>(static_cast<const Layer&>(*this).get_error_signals(parent_index));
}

// Accessing local matrices
AbsMat& Layer::get_local_activations(int child_index) {
  return get_activations(child_index).Matrix();
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

// Accessing matrices corresponding to parent/child layer
const AbsDistMat& Layer::get_activations(const Layer& child) const {
  const int child_index = (std::find(m_child_layers.begin(),
                                     m_child_layers.end(),
                                     &child)
                           - m_child_layers.begin());
  if (child_index >= get_num_children()) {
    std::stringstream err;
    err << "attempted to get activation tensor of "
        << "layer \"" << get_name() << "\" "
        << "corresponding to layer\"" << child.get_name() << "\", "
        << "which is not a child layer";
    LBANN_ERROR(err.str());
  }
  return get_activations(child_index);
}
const AbsDistMat& Layer::get_error_signals(const Layer& parent) const {
  const int parent_index = (std::find(m_parent_layers.begin(),
                                      m_parent_layers.end(),
                                      &parent)
                           - m_parent_layers.begin());
  if (parent_index >= get_num_parents()) {
    std::stringstream err;
    err << "attempted to get error signal tensor of "
        << "layer \"" << get_name() << "\" "
        << "corresponding to layer\"" << parent.get_name() << "\", "
        << "which is not a parent layer";
    LBANN_ERROR(err.str());
  }
  return get_error_signals(parent_index);
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
  if (using_gpus()) { setup_gpu(); }
}

void Layer::setup_pointers() {
  std::stringstream err;

  // Check that the parent pointers are valid
  for (size_t i = 0; i < m_parent_layers.size(); ++i) {
    const auto* parent = m_parent_layers[i];
    if (parent == nullptr) {
      err << "layer \"" << get_name() << "\" "
          << "has a null pointer to parent layer " << i;
      LBANN_ERROR(err.str());
    }
    const auto& parent_children = parent->m_child_layers;
    if (std::find(parent_children.begin(), parent_children.end(), this)
        == parent_children.end()) {
      err << "layer \"" << parent->get_name() << "\" is a parent of "
          << "layer \"" << get_name() << "\", but "
          << "\"" << get_name() << "\" is not a child of "
          << "\"" << parent->get_name() << "\"";
      LBANN_ERROR(err.str());
    }
  }

  // Check that the child pointers are valid
  for (size_t i = 0; i < m_child_layers.size(); ++i) {
    const auto* child = m_child_layers[i];
    if (child == nullptr) {
      err << "layer \"" << get_name() << "\" "
          << "has a null pointer to child layer " << i;
      LBANN_ERROR(err.str());
    }
    const auto& child_parents = child->m_parent_layers;
    if (std::find(child_parents.begin(), child_parents.end(), this)
        == child_parents.end()) {
      err << "layer \"" << child->get_name() << "\" is a child of "
          << "layer \"" << get_name() << "\", but "
          << "\"" << get_name() << "\" is not a parent of "
          << "\"" << child->get_name() << "\"";
      LBANN_ERROR(err.str());
    }
  }
  
  // Check that the number of parents/children are valid
  if(m_expected_num_parent_layers >= 0
     && get_num_parents() != m_expected_num_parent_layers) {
    err << get_type() << " layer \"" << get_name() << "\" "
        << "expects " << m_expected_num_parent_layers << " "
        << "parent layer" << (m_expected_num_parent_layers != 1 ? "s" : "")
        << ", but found " << get_num_parents();
    if (get_num_parents() > 0) {
      err << " (";
      for (int i = 0; i < get_num_parents(); ++i) {
        err << (i > 0 ? ", " : "")
            << "\"" << m_parent_layers[i]->get_name() << "\"";
      }
      err << ")";
    }
    LBANN_ERROR(err.str());
  }
  if(m_expected_num_child_layers >= 0
     && get_num_children() != m_expected_num_child_layers) {
    err << get_type() << " layer \"" << get_name() << "\" "
        << "expects " << m_expected_num_child_layers << " "
        << "child layer" << (m_expected_num_child_layers != 1 ? "s" : "")
        << ", but found " << get_num_children();
    if (get_num_children() > 0) {
      err << " (";
      for (int i = 0; i < get_num_children(); ++i) {
        err << (i > 0 ? ", " : "")
            << "\"" << m_child_layers[i]->get_name() << "\"";
      }
      err << ")";
    }
    LBANN_ERROR(err.str());
  }

}

void Layer::setup_dims() {
  m_output_dims_list.resize(get_num_children());
  if (get_num_parents() > 0) {
    const auto& input_dims = get_input_dims();
    for (auto& output_dims : m_output_dims_list) {
      if (output_dims.empty()) {
        output_dims = input_dims;
      }
    }
  }
}
  
void Layer::setup_matrices(const El::Grid& grid) {

  // Destroy previously setup matrices
  m_inputs.clear();
  m_outputs.clear();
  m_gradient_wrt_outputs.clear();
  m_gradient_wrt_inputs.clear();

  // Construct matrices
  m_inputs.resize(get_num_parents());
  m_outputs.resize(get_num_children());
  m_gradient_wrt_outputs.resize(get_num_children());
  m_gradient_wrt_inputs.resize(get_num_parents());
  for (int i = 0; i < get_num_parents(); ++i) {
    m_inputs[i] = construct_matrix(grid, "input", i);
  }
  for (int i = 0; i < get_num_children(); ++i) {
    m_outputs[i] = construct_matrix(grid, "output", i);
  }
  for (int i = 0; i < get_num_children(); ++i) {
    m_gradient_wrt_outputs[i]
      = construct_matrix(grid, "gradient_wrt_output", i);
  }
  for (int i = 0; i < get_num_parents(); ++i) {
    m_gradient_wrt_inputs[i]
      = construct_matrix(grid, "gradient_wrt_input", i);
  }

}

std::unique_ptr<AbsDistMat> Layer::construct_matrix(const El::Grid& grid,
                                                    std::string type,
                                                    El::Int index) {

  // Choose matrix distribution
  El::Distribution col_dist, row_dist;
  El::DistWrap wrap;
  El::Device device = get_device_allocation();
  switch (get_data_layout()) {
  case data_layout::DATA_PARALLEL:
    col_dist = El::STAR;
    row_dist = El::VC;
    wrap     = El::ELEMENT;
    break;
  case data_layout::MODEL_PARALLEL:
    col_dist = El::MC;
    row_dist = El::MR;
    wrap     = El::ELEMENT;
    break;
  default: LBANN_ERROR("invalid data layout");
  }

  // Construct matrix
  std::unique_ptr<AbsDistMat> mat;
  mat.reset(AbsDistMat::Instantiate(grid, 0,
                                    col_dist, row_dist, wrap, device));
  
#ifdef LBANN_HAS_GPU
  // Allocate GPU memory with the CUDA API
  if (device == El::Device::GPU) { mat->Matrix().SetMemoryMode(0); }
  // Use pinned memory for data on the host.
  if (device == El::Device::CPU) { mat->Matrix().SetMemoryMode(1); }
#endif // LBANN_HAS_GPU

  return mat;
}

void Layer::setup_data() {

  // Get mini-batch size
  const auto& mini_batch_size = m_model->get_max_mini_batch_size();

  // Initialize input and output tensors
  fp_setup_inputs(mini_batch_size);
  fp_setup_outputs(mini_batch_size);

  // Initialize gradient w.r.t. output tensors
  // Note: We guess whether the tensor is a view or needs to allocate
  // memory, but there are some edge cases that are not handled.
  for (int i = 0; i < get_num_children(); ++i) {
    const auto& child = *m_child_layers[i];
    const auto& output = get_activations(i);
    auto& gradient_wrt_output = *m_gradient_wrt_outputs[i];
    gradient_wrt_output.Empty(false);
    gradient_wrt_output.AlignWith(output);
    if (child.get_data_layout() == get_data_layout()
        && child.get_device_allocation() == get_device_allocation()
        && gradient_wrt_output.DistData() == output.DistData()) {
      El::LockedView(gradient_wrt_output, output);
    } else {
      El::Copy(output, gradient_wrt_output);
    }
  }

  // Initialize gradient w.r.t. input tensors
  bp_setup_gradient_wrt_inputs(mini_batch_size);

}

void Layer::bp_compute() {
  for (int i = 0; i < get_num_parents(); ++i) {
    El::Zero(get_error_signals(i));
  }
}

void Layer::check_setup() {
  std::stringstream err;

  // Check tensor dimensions
  for (int i = 0; i < get_num_parents(); ++i) {
    const auto& dims = get_input_dims(i);
    if (dims.empty()) {
      err << "layer \"" << get_name() << "\" has "
          << "uninitialized input tensor dimensions "
          << "(index " << i << ")";
      LBANN_ERROR(err.str());
    }
    if (std::any_of(dims.begin(), dims.end(),
                    [](int d) { return d <= 0; })) {
      err << "layer \"" << get_name() << "\" has invalid "
          << "input tensor dimensions (";
      for (size_t j = 0; j < dims.size(); ++j) {
        err << (j > 0 ? " x " : "") << dims[j];
      }
      err << " at index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }
  for (int i = 0; i < get_num_children(); ++i) {
    const auto& dims = get_output_dims(i);
    if (dims.empty()) {
      err << "layer \"" << get_name() << "\" has "
          << "uninitialized output tensor dimensions "
          << "(index " << i << ")";
      LBANN_ERROR(err.str());
    }
    if (std::any_of(dims.begin(), dims.end(),
                    [](int d) { return d <= 0; })) {
      err << "layer \"" << get_name() << "\" has invalid "
          << "output tensor dimensions (";
      for (size_t j = 0; j < dims.size(); ++j) {
        err << (j > 0 ? " x " : "") << dims[j];
      }
      err << " at index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }

  // Check number of tensors
  const int num_parents = get_num_parents();
  const int num_children = get_num_children();
  if ((int) m_inputs.size() != num_parents
      || (int) m_outputs.size() != num_children
      || (int) m_gradient_wrt_outputs.size() != num_children
      || (int) m_gradient_wrt_inputs.size() != num_parents) {
    err << "layer \"" << get_name() << "\" has an "
        << "invalid number of input and output tensors "
        << "(found " << num_parents << " parent layers, "
        << num_children << " child layers, "
        << m_inputs.size() << " input tensors, "
        << m_outputs.size() << " output tensors, "
        << m_gradient_wrt_outputs.size() << " gradient w.r.t. output tensors, "
        << m_gradient_wrt_inputs.size() << " gradient w.r.t. input tensors)";
    LBANN_ERROR(err.str());
  }

  // Check that tensors are initialized
  for (int i = 0; i < get_num_parents(); ++i) {
    if (m_inputs[i] == nullptr) {
      err << "layer \"" << get_name() << "\" has an "
          << "uninitialized input tensor (index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }
  for (int i = 0; i < get_num_children(); ++i) {
    if (m_outputs[i] == nullptr) {
      err << "layer \"" << get_name() << "\" has an "
          << "uninitialized output tensor (index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }
  for (int i = 0; i < get_num_children(); ++i) {
    if (m_gradient_wrt_outputs[i] == nullptr) {
      err << "layer \"" << get_name() << "\" has an "
          << "uninitialized gradient w.r.t. output tensor "
          << "(index " << i << ")";
      LBANN_ERROR(err.str());
    }
  }
  for (int i = 0; i < get_num_parents(); ++i) {
    if (m_gradient_wrt_inputs[i] == nullptr) {
      err << "layer \"" << get_name() << "\" has an "
          << "uninitialized gradient w.r.t. input tensor "
          << "(index " << i << ")";
      LBANN_ERROR(err.str());
    }
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

void Layer::fp_setup_inputs(El::Int mini_batch_size) {
  if (get_num_parents() < 1) { return; }

  // Determine distributed matrix alignment
  const auto& alignment_dist
    = m_parent_layers.front()->get_activations(*this).DistData();

  // Iterate through input tensors
  for (int i = 0; i < get_num_parents(); ++i) {

    // Initialize input tensor
    const auto& parent = *m_parent_layers[i];
    const auto& parent_output = parent.get_activations(*this);
    auto& input = *m_inputs[i];
    input.Empty(false);
    input.AlignWith(alignment_dist);
    if (parent_output.DistData() == input.DistData()) {
      El::LockedView(input, parent_output);
    } else {
      El::Copy(parent_output, input);
    }

    // Check input matrix dimensions
    const auto& height = get_input_size(i);
    const auto& width = mini_batch_size;
    if (input.Height() != height || input.Width() != width) {
      std::stringstream err;
      err << "layer \"" << get_name() << "\" "
          << "expected an input tensor stored in a "
          << height << " x " << width << " matrix "
          << "from layer \"" << parent.get_name() << "\", but got a "
          << input.Height() << " x " << input.Width() << " matrix";
      LBANN_ERROR(err.str());
    }

  }
  
}

void Layer::fp_setup_outputs(El::Int mini_batch_size) {
  if (get_num_children() < 1) { return; }

  // Determine distributed matrix alignment
  const bool align_outputs = get_num_parents() > 0;
  const auto& alignment_dist = (align_outputs ?
                                get_prev_activations().DistData() :
                                get_activations().DistData());

  // Initialize output tensors
  for (int i = 0; i < get_num_children(); ++i) {
    auto& output = get_activations(i);
    output.Empty(false);
    if (align_outputs) { output.AlignWith(alignment_dist); }
    output.Resize(get_output_size(i), mini_batch_size);
  }

}

void Layer::bp_setup_gradient_wrt_outputs(El::Int mini_batch_size) {
  for (int i = 0; i < get_num_children(); ++i) {

    // Initialize gradient w.r.t. output tensor
    const auto& child = *m_child_layers[i];
    const auto& child_gradient_wrt_input = child.get_error_signals(*this);
    auto& gradient_wrt_output = *m_gradient_wrt_outputs[i];
    gradient_wrt_output.Empty(false);
    gradient_wrt_output.AlignWith(get_activations(i));
    if (child_gradient_wrt_input.DistData()
        == gradient_wrt_output.DistData()) {
      El::LockedView(gradient_wrt_output, child_gradient_wrt_input);
    } else {
      El::Copy(child_gradient_wrt_input, gradient_wrt_output);
    }

    // Check gradient w.r.t. output matrix dimensions
    const auto& height = get_output_size(i);
    const auto& width = mini_batch_size;
    if (gradient_wrt_output.Height() != height
        || gradient_wrt_output.Width() != width) {
      std::stringstream err;
      err << "layer \"" << get_name() << "\" "
          << "expected a gradient w.r.t. output tensor stored in a "
          << height << " x " << width << " matrix "
          << "from layer \"" << child.get_name() << "\", but got a "
          << gradient_wrt_output.Height() << " x "
          << gradient_wrt_output.Width() << " matrix";
      LBANN_ERROR(err.str());
    }

  }
}

void Layer::bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) {
  for (int i = 0; i < get_num_parents(); ++i) {
    auto& gradient_wrt_input = get_error_signals(i);
    gradient_wrt_input.Empty(false);
    gradient_wrt_input.AlignWith(get_prev_activations(i));
    gradient_wrt_input.Resize(get_input_size(i), mini_batch_size);
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
using namespace dc;

void Layer::early_terminate() {
  if (m_exit_count == 0) {
    dc::MPIPrintStreamDebug() << "Early terminate\n";
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    cudaDeviceReset();
    exit(0);
  }
  if (m_exit_count > 0) --m_exit_count;
}

bool Layer::early_terminate_last_iteration() const {
  return m_exit_count == 0;
}

void Layer::setup_distconv() {
  m_distconv_enabled = using_distconv();
  char *count_str = std::getenv("DISTCONV_EARLY_TERMINATE");
  if (count_str) {
    m_exit_count = atoi(count_str);
    dc::MPIRootPrintStreamInfo()
        << "Exiting after " << m_exit_count
        << " iterations\n";
  }
}

void Layer::setup_tensor_distribution_init(
    std::map<const Layer*, std::array<dc::Dist, 4>> &dists,
    std::map<dc::Dist*, std::set<dc::Dist*>> &invariants,
    std::set<dc::Dist*> &updated,
    std::set<dc::Dist*> &fixed) {
  auto &ps = get_parallel_strategy();
  MPIRootPrintStreamInfo() << "Parallel Strategy for layer " << get_name()
                           << ": " << ps << "\n";
  int n = ps.sample_groups;
  int c = ps.channel_groups;
  int f = ps.filter_groups;
  int h = ps.height_groups;
  int w = ps.width_groups;
  int np = m_comm->get_procs_per_model();
  // if only one process is used, do not parallelize
  if (np == 1) {
    n = c = f = h = w = 1;
  }
  if (distconv_enabled()) {
    if (c != f) {
      MPIRootPrintStreamError() << "The numbers of channel and filter decomposition should be the same.\n";
      throw lbann_exception();
    }
    if (n != 1) {
#if 0
      if (get_type() != "convolution" &&
          get_type() != "ReLU" &&
          get_type() != "pooling") {
        MPIRootPrintStreamError() << "Layers except for convolution, ReLU and pooling do not support sample parallelization yet.\n";
        throw lbann_exception();
      }
#endif
    }
    if (c != 1 || f != 1) {
      MPIRootPrintStreamError() << "Distconv does not support channel/filter parallelization yet.\n";
      throw lbann_exception();      
    }
    int nchw = n * c * h * w;
    if (nchw > np) {
      MPIRootPrintStreamError() <<
          "The number of MPI ranks must be at least as large as the number of processes implied by parallel strategy: "
                            << ps << "\n";
      throw lbann_exception();
    }
    // Put the remaining factor into the outer-most process dimension
    float rem = np / (float)nchw;
    n *= rem;
    nchw = n * c * h * w;
    if (nchw != np) {
      MPIRootPrintStreamError() <<
          "Can't determine factorization of the number of MPI ranks for parallel strategy: "
                            << ps << "\n";
      throw lbann_exception();
    }
    MPIRootPrintStreamInfo() << "Process grid of NxCxHxW: "
                             << n << "x" << c << "x" << h << "x" << w << "\n";
  }
  
  assert_always(!distconv_enabled() || (
      h * w * n * c == np && h * w * n * f == np));
  
  ps.sample_groups = n;
  ps.channel_groups = c;
  ps.filter_groups = f;
  ps.height_groups = h;
  ps.width_groups = w;
  
  Dist prev_activations_dist =  Dist({w, h, c, n});
  Dist activations_dist =  Dist({w, h, f, n});
  Dist prev_error_signals_dist =  activations_dist;
  Dist error_signals_dist =  prev_activations_dist;
  std::array<Dist, 4> layer_dists = {prev_activations_dist,
                                     activations_dist,
                                     error_signals_dist,
                                     prev_error_signals_dist};
  dists.insert(std::make_pair(this, layer_dists));
  invariants.insert(std::make_pair(&dists[this][0], std::set<Dist*>()));
  invariants.insert(std::make_pair(&dists[this][1], std::set<Dist*>()));
  invariants.insert(std::make_pair(&dists[this][2], std::set<Dist*>()));
  invariants.insert(std::make_pair(&dists[this][3], std::set<Dist*>()));
}

void Layer::setup_tensor_distribution_add_adjacent_invariants(
    std::map<const Layer*, std::array<Dist, 4>> &dists,
    std::map<Dist*, std::set<Dist*>> &invariants) {
  if (!distconv_enabled()) return;
  auto &layer_dists = dists[this];
  const auto &ps = get_parallel_strategy();
  if (get_child_layers().size() > 0) {
    auto child = get_child_layers()[0];
    if (child->distconv_enabled() &&
        child->get_parallel_strategy() == ps) {
      invariants[&layer_dists[1]].insert(
          &dists[child][0]);
      invariants[&layer_dists[3]].insert(
          &dists[child][2]);
    }
  }
  if (get_parent_layers().size() > 0) {
    auto parent = get_parent_layers()[0];
    if (parent->distconv_enabled() &&
        parent->get_parallel_strategy() == ps) {
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
  if (distconv_enabled()) {
    const auto *child = get_child_layers()[0];
    if (child->distconv_enabled()) {
      m_output_decomposition_block =
          child->get_input_decomposition_block();
    }
    m_input_decomposition_block =
        m_output_decomposition_block * get_strides();
  }
#endif
}

void Layer::setup_tensors_fwd(const std::array<Dist, 4> &dists) {
  m_distconv_enabled = distconv_enabled();

  if (!m_distconv_enabled) {
    //MPIPrintStreamInfo() << get_name() << ": distconv disabled\n";
    return;
  }

  MPIRootPrintStreamInfo() << get_name() << ": setup_tensors_fwd\n";  
  const auto &child_layers = get_child_layers();
  MPIPrintStreamDebug() << ": number of children: "
                            << child_layers.size()
                            << ", child name: " << child_layers[0]->get_name()
                            << "\n";
  const auto &parent_layers = get_parent_layers();
  MPIPrintStreamDebug() << ": number of parents: "
                            << parent_layers.size()
                            << ", parent name: " << parent_layers[0]->get_name()
                            << "\n";
  assert_always(child_layers.size() == 1);
  assert_always(parent_layers.size() == 1);

  const auto &ps = get_parallel_strategy();
  const auto &parent = *parent_layers[0];
  if (parent.distconv_enabled()) {
    m_parent_copy_in_required = false;
    m_parent_shuffle_required = ps != parent.get_parallel_strategy();
  } else {
    m_parent_copy_in_required = true;
  }
  MPIRootPrintStreamInfo() << "m_parent_copy_in_required: "
                           << m_parent_copy_in_required
                           << ", m_parent_shuffle_required: "
                           << m_parent_shuffle_required      
                           << "\n";
  
  const auto &child = *child_layers[0];
  if (child.distconv_enabled()) {
    m_child_copy_out_required = false;
    m_child_shuffle_required = ps != child.get_parallel_strategy();
  } else {
    m_child_copy_out_required = true;
  }
  MPIRootPrintStreamInfo() << "m_child_copy_out_required: "
                           << m_child_copy_out_required
                           << ", m_child_shuffle_required: "
                           << m_child_shuffle_required 
                           << "\n";
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
  //input_local_shape[3] = m_max_mini_batch_size_per_gpu;
  // m_max_mini_batch_size_per_gpu is the maximum among all GPUs, so
  // it's larger than the actual maximum size for some ranks when the
  // mini batch size is not divisible.
  input_local_shape[3] = 0;
  const Array4 spatial_local_size = {0, 0, 0, 0};

  if (m_parent_copy_in_required || m_parent_shuffle_required) {
    if (m_parent_copy_in_required) {
      m_prev_activations_const_view = TensorDev(input_tensor_shape, loc,
                                                sample_dist,
                                                input_local_shape,
                                                sample_block_size);
    } else {
      m_prev_activations_const_view = get_parent_layers()[0]->get_activations_t();
    }
    m_prev_activations_t = TensorDev(input_tensor_shape, loc, dists[0],
                                     spatial_local_size, m_input_decomposition_block);
    assert0(m_prev_activations_t.allocate());
    m_prev_activations_t.zero();
    m_prev_activations_shuffler = new TensorShuffler(
        m_prev_activations_const_view, m_prev_activations_t);

    for (int i = 0; i < 3; ++i) {
      m_prev_activations_shuffler_last_mb[i] = nullptr;
    }
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
  const Array4 activations_local_shape =
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
  //output_local_shape[3] = m_max_mini_batch_size_per_gpu;
  output_local_shape[3] = 0;
  m_activations_copyout = TensorDev(output_tensor_shape, loc, sample_dist,
                                    output_local_shape, sample_block_size);
  if (m_child_copy_out_required) {
    m_activations_shuffler = new TensorShuffler(
        m_activations_t, m_activations_copyout);
    for (int i = 0; i < 3; ++i) {
      m_activations_shuffler_last_mb[i] = nullptr;
    }
  }
}

void Layer::setup_tensors_bwd(const std::array<Dist, 4> &dists) {
}

void Layer::setup_prev_error_signals_tensor(const std::array<Dist, 4> &dists) {
  const LocaleMPI loc(m_comm->get_model_comm().comm, false);
  const Array4 sample_block_size = {1, 1, 1, 1};
  const Dist sample_dist = Dist({1, 1, 1, m_comm->get_procs_per_model()});
  const Array4 output_tensor_shape =
      {m_neuron_dims[2], m_neuron_dims[1],
       m_neuron_dims[0], this->m_model->get_max_mini_batch_size()};
  Array4 output_local_shape = output_tensor_shape;
  //output_local_shape[3] = m_max_mini_batch_size_per_gpu;
  output_local_shape[3] = 0;
  
  if (m_child_copy_out_required || m_child_shuffle_required) {
    if (m_child_copy_out_required) {
      m_prev_error_signals_const_view = TensorDev(output_tensor_shape, loc,
                                                  sample_dist,
                                                  output_local_shape,
                                                  sample_block_size);
    } else {
      m_prev_error_signals_const_view =
          get_child_layers()[0]->get_error_signals_t();
    }
    m_prev_error_signals_t = TensorDev(output_tensor_shape, loc,
                                       dists[3],
                                       m_activations_t.get_local_shape(),
                                       m_output_decomposition_block);
    assert0(m_prev_error_signals_t.allocate());
    m_prev_error_signals_t.zero();
    m_prev_error_signals_shuffler = new TensorShuffler(
        m_prev_error_signals_const_view, m_prev_error_signals_t);
    for (int i = 0; i < 3; ++i) {
      m_prev_error_signals_shuffler_last_mb[i] = nullptr;
    }
  } else {
    m_prev_error_signals_t = get_child_layers()[0]->get_error_signals_t();
    assert_always(m_prev_error_signals_t.get_distribution() ==
                  dists[3]);
    assert_always(m_prev_error_signals_t.get_requested_local_block() ==
                  m_output_decomposition_block);
  }
}

void Layer::setup_error_signals_tensor(const std::array<Dist, 4> &dists) {
  const Array4 input_tensor_shape =
      {m_prev_neuron_dims[2], m_prev_neuron_dims[1],
       m_prev_neuron_dims[0], this->m_model->get_max_mini_batch_size()};
  const LocaleMPI loc(m_comm->get_model_comm().comm, false);
  m_error_signals_t = TensorDev(input_tensor_shape, loc,
                                dists[2],
                                m_prev_activations_t.get_local_shape(),
                                m_input_decomposition_block);
  assert0(m_error_signals_t.allocate());
  m_error_signals_t.zero();
}

void Layer::setup_error_signals_copyout_tensor(const std::array<Dist, 4> &dists) {
  const Array4 input_tensor_shape =
      {m_prev_neuron_dims[2], m_prev_neuron_dims[1],
       m_prev_neuron_dims[0], this->m_model->get_max_mini_batch_size()};
  const LocaleMPI loc(m_comm->get_model_comm().comm, false);
  const Dist sample_dist = Dist({1, 1, 1, m_comm->get_procs_per_model()});
  Array4 input_local_shape = input_tensor_shape;
  // Assuming single GPU per rank
  //input_local_shape[3] = m_max_mini_batch_size_per_gpu;
  input_local_shape[3] = 0;
  const Array4 sample_block_size = {1, 1, 1, 1};
  
  m_error_signals_copyout = TensorDev(input_tensor_shape, loc, sample_dist,
                                      input_local_shape, sample_block_size);
  if (m_parent_copy_in_required) {
    m_error_signals_shuffler = new TensorShuffler(
        m_error_signals_t, m_error_signals_copyout);
    for (int i = 0; i < 3; ++i) {
      m_error_signals_shuffler_last_mb[i] = nullptr;
    }
  }
}

Array4 Layer::get_prev_activations_overlap() const {
  return Array4(0);  
}

Array4 Layer::get_activations_overlap() const {
#if 0  
  if (distconv_enabled() &&
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
  if (distconv_enabled() &&
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

void Layer::fp_setup_distconv(int mini_batch_size) {
  if (!distconv_enabled()) return;

  early_terminate();

  // Reconfigure the sample dimension as the mini batch size may vary
  // at the end of epoch
  m_prev_activations_t.set_outermost_dimension(mini_batch_size);
  assert_always((int)m_prev_activations_t.get_shape()[-1] ==
                mini_batch_size);
  if (m_parent_copy_in_required || m_parent_shuffle_required) {
    m_prev_activations_const_view.set_outermost_dimension(
        mini_batch_size);
    assert_always((int)m_prev_activations_const_view.get_shape()[-1] ==
                  mini_batch_size);
    if (m_parent_copy_in_required) {
      // then, parent is assumed to be data parallel, so the local
      // size of the sample dimension should be equal to
      // the local width of previous activations.
      assert_always(
          (int)m_prev_activations_const_view.get_local_shape()[-1] ==
          get_prev_activations().LocalWidth());
    }
  } 
  m_activations_t.set_outermost_dimension(mini_batch_size);
  assert_always((int)m_activations_t.get_shape()[-1] ==
                mini_batch_size);
  m_activations_copyout.set_outermost_dimension(mini_batch_size);
  assert_always((int)m_activations_copyout.get_shape()[-1] ==
                mini_batch_size);
  assert_always((int)m_activations_copyout.get_local_shape()[-1] ==
                get_activations().LocalWidth());

  ensure_prev_activations();
}

void Layer::bp_setup_distconv(int mini_batch_size) {
  
  if (!distconv_enabled()) return;

  // Reconfigure the sample dimension as the mini batch size may vary
  // at the end of epoch
  m_prev_error_signals_t.set_outermost_dimension(mini_batch_size);
  assert_always((int)m_prev_error_signals_t.get_shape()[-1] ==
                mini_batch_size);
  if (m_child_copy_out_required || m_child_shuffle_required) {
    m_prev_error_signals_const_view.set_outermost_dimension(mini_batch_size);
    assert_always((int)m_prev_error_signals_const_view.get_shape()[-1] ==
                  mini_batch_size);
    if (m_child_copy_out_required) {
      assert_always(
          (int)m_prev_error_signals_const_view.get_local_shape()[-1] ==
          get_prev_error_signals().LocalWidth());
    }
  }
  m_error_signals_t.set_outermost_dimension(mini_batch_size);
  assert_always((int)m_error_signals_t.get_shape()[-1] ==
                mini_batch_size);
  m_error_signals_copyout.set_outermost_dimension(mini_batch_size);
  assert_always((int)m_error_signals_copyout.get_shape()[-1] ==
                mini_batch_size);
  assert_always((int)m_error_signals_copyout.get_local_shape()[-1] ==
                get_error_signals().LocalWidth());

  ensure_prev_error_signals();
}

void Layer::ensure_prev_activations() {
  if (!(m_parent_copy_in_required || m_parent_shuffle_required)) {
    return;
  }

  if (m_parent_copy_in_required) {
    MPIPrintStreamDebug() << "Copying previous activations from sample decomposition\n";
    assert0(dc::tensor::View(
        m_prev_activations_const_view,
        get_prev_activations().LockedBuffer()));
  } else {
    assert_always(m_parent_shuffle_required);
  }
  TensorShuffler *shuffler = nullptr;
  if (this->m_model->get_max_mini_batch_size() ==
      this->m_model->get_current_mini_batch_size()) {
    shuffler = m_prev_activations_shuffler;
  } else {
    int shfl_idx = static_cast<int>(this->m_model->get_execution_mode());
    assert_always(shfl_idx >= 0 && shfl_idx < 3);
    if (m_prev_activations_shuffler_last_mb[shfl_idx] == nullptr) {
      m_prev_activations_shuffler_last_mb[shfl_idx] = new TensorShuffler(
          m_prev_activations_const_view, m_prev_activations_t);
    }
    shuffler = m_prev_activations_shuffler_last_mb[shfl_idx];      
  }
  assert_always(shuffler != nullptr);
  shuffler->shuffle_forward(
      m_prev_activations_const_view.get_const_base_ptr(),
      m_prev_activations_t.get_base_ptr(),
      El::GPUManager::Stream());
  this->m_model->clock_start();
}

void Layer::copy_out_activations() {
  if (!m_child_copy_out_required) return;

  this->m_model->clock_end();
  
  MPIPrintStreamDebug() << "Copying activations back to sample decomposition\n";
  assert0(dc::tensor::View(
      m_activations_copyout, get_activations().Buffer()));
  TensorShuffler *shuffler = nullptr;
  if (this->m_model->get_max_mini_batch_size() ==
      this->m_model->get_current_mini_batch_size()) {
    shuffler = m_activations_shuffler;
  } else {
    int shfl_idx = static_cast<int>(this->m_model->get_execution_mode());    
    assert_always(shfl_idx >= 0 && shfl_idx < 3);
    if (m_activations_shuffler_last_mb[shfl_idx] == nullptr) {
      m_activations_shuffler_last_mb[shfl_idx] = new TensorShuffler(
          m_activations_t, m_activations_copyout);          
    }
    shuffler = m_activations_shuffler_last_mb[shfl_idx];      
  }
  assert_always(shuffler != nullptr);
  shuffler->shuffle_forward(
      m_activations_t.get_const_base_ptr(),
      m_activations_copyout.get_base_ptr(),
      El::GPUManager::Stream());
}

void Layer::ensure_prev_error_signals() {
  if (!(m_child_copy_out_required || m_child_shuffle_required)) {
    return;
  }

  if (m_child_copy_out_required) {  
    MPIPrintStreamDebug() << "Copying previous error signals from sample decomposition\n";
    assert0(dc::tensor::View(
        m_prev_error_signals_const_view,
        get_prev_error_signals().LockedBuffer()));
  } else {
    assert_always(m_child_shuffle_required);
  }
  TensorShuffler *shuffler = nullptr;
  if (this->m_model->get_max_mini_batch_size() ==
      this->m_model->get_current_mini_batch_size()) {
    shuffler = m_prev_error_signals_shuffler;
  } else {
    int shfl_idx = static_cast<int>(this->m_model->get_execution_mode());
    assert_always(shfl_idx >= 0 && shfl_idx < 3);
    if (m_prev_error_signals_shuffler_last_mb[shfl_idx] == nullptr) {
      m_prev_error_signals_shuffler_last_mb[shfl_idx] = new TensorShuffler(
          m_prev_error_signals_const_view, m_prev_error_signals_t);
    }
    shuffler = m_prev_error_signals_shuffler_last_mb[shfl_idx];      
  }
  assert_always(shuffler != nullptr);
  
  shuffler->shuffle_forward(
      m_prev_error_signals_const_view.get_const_base_ptr(),
      m_prev_error_signals_t.get_base_ptr(),
      El::GPUManager::Stream());
}

void Layer::copy_out_error_signals() {
  if (!m_parent_copy_in_required) return;
  
  const auto &parents = get_parent_layers();  
  assert_always(parents.size() == 1);
  const Layer *parent = parents[0];

  if (parent->get_type().find("input:") == 0) {
    // No need to copy back when the parent is an input layer
    MPIPrintStreamDebug() << "Skipping copy back as the parent is an input layer\n";
    return;
  }
  // No need to copy back as the original layer compute function
  // will be called
  if (m_exit_count == 0) return;
    
  MPIPrintStreamDebug() << "Copying error signals back to sample decomposition\n";
  assert0(dc::tensor::View(
      m_error_signals_copyout, get_error_signals().Buffer()));
  TensorShuffler *shuffler = nullptr;
  if (this->m_model->get_max_mini_batch_size() ==
      this->m_model->get_current_mini_batch_size()) {
    shuffler = m_error_signals_shuffler;
  } else {
    int shfl_idx = static_cast<int>(this->m_model->get_execution_mode());            
    assert_always(shfl_idx >= 0 && shfl_idx < 3);
    if (m_error_signals_shuffler_last_mb[shfl_idx] == nullptr) {
      m_error_signals_shuffler_last_mb[shfl_idx] = new TensorShuffler(
          m_error_signals_t, m_error_signals_copyout);
    }
    shuffler = m_error_signals_shuffler_last_mb[shfl_idx];      
  }
  assert_always(shuffler != nullptr);
  
  shuffler->shuffle_forward(
      m_error_signals_t.get_const_base_ptr(),
      m_error_signals_copyout.get_base_ptr(),
      El::GPUManager::Stream());
}

#endif

}  // namespace lbann
