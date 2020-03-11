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

#include "lbann/layers/data_type_distconv_layer.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"

namespace lbann {

template <typename TensorDataType>
const typename data_type_distconv_layer<TensorDataType>::TensorDevType&
data_type_distconv_layer<TensorDataType>::get_activations(const Layer& child) const {
  if (layer().get_num_children() == 0) {
    LBANN_ERROR("This layer has no children");
  }
  const int child_index = layer().find_child_layer_index(&child);
  if (child_index >= layer().get_num_children()) {
    LBANN_ERROR("attempted to get activation tensor of ",
                "layer \"", get_name(), "\" ",
                "corresponding to layer\"", child.get_name(), "\", ",
                "which is not a child layer");
  }
  return get_activations(child_index);
}

template <typename TensorDataType>
const typename data_type_distconv_layer<TensorDataType>::TensorDevType&
data_type_distconv_layer<TensorDataType>::get_activations(int child_index) const {
  if (child_index < 0 || child_index >= (int) m_outputs.size()) {
    LBANN_ERROR("attempted to access invalid distconv activation tensor ",
                "from ", get_name(), " ",
                "(requested index ", child_index, ", but there are ",
                m_outputs.size(), " activation tensors)");
  }
  return *m_outputs[child_index];
}

template <typename TensorDataType>
typename data_type_distconv_layer<TensorDataType>::TensorDevType&
data_type_distconv_layer<TensorDataType>::get_activations(int child_index) {
  return const_cast<TensorDevType&>(
      static_cast<const data_type_distconv_layer<TensorDataType>&>(*this).get_activations(child_index));
}

template <typename TensorDataType>
const typename data_type_distconv_layer<TensorDataType>::TensorDevType&
data_type_distconv_layer<TensorDataType>::get_prev_activations(int parent_index) const {
  if (parent_index < 0 || parent_index >= (int) m_inputs.size()) {
    LBANN_ERROR("attempted to access invalid distconv previous activation tensor ",
                "from ", get_name(), " ",
                "(requested index ", parent_index, ", but there are ",
                m_inputs.size(), " previous activation tensors)");
  }
  return *m_inputs[parent_index];
}

template <typename TensorDataType>
typename data_type_distconv_layer<TensorDataType>::TensorDevType&
data_type_distconv_layer<TensorDataType>::get_prev_activations(int parent_index) {
  return const_cast<TensorDevType&>(
      static_cast<const data_type_distconv_layer<TensorDataType>&>(*this).get_prev_activations(parent_index));
}

template <typename TensorDataType>
const typename data_type_distconv_layer<TensorDataType>::TensorDevType&
data_type_distconv_layer<TensorDataType>::get_original_prev_activations(
    int parent_index) const {
  if (parent_index < 0 || parent_index >= (int) m_original_inputs.size()) {
    LBANN_ERROR("attempted to access invalid original previous activation tensor ",
                "from ", get_name(), " ",
                "(requested index ", parent_index, ", but there are ",
                m_original_inputs.size(), " original previous activation tensors)");
  }
  return *m_original_inputs[parent_index];
}

template <typename TensorDataType>
typename data_type_distconv_layer<TensorDataType>::TensorDevType&
data_type_distconv_layer<TensorDataType>::get_original_prev_activations(
    int parent_index) {
  return const_cast<TensorDevType&>(
      static_cast<const data_type_distconv_layer<TensorDataType>&>(*this).get_original_prev_activations(parent_index));
}

template <typename TensorDataType>
dc::Shape data_type_distconv_layer<TensorDataType>::get_input_tensor_shape(
    int input_index) const {
  const auto input_dims = layer().get_input_dims();
  std::vector<int> input_tensor_shape_v(input_dims.rbegin(), input_dims.rend());
  input_tensor_shape_v.push_back(layer().get_model()->get_max_mini_batch_size());
  return dc::Shape(input_tensor_shape_v);
}

template <typename TensorDataType>
void data_type_distconv_layer<TensorDataType>::setup_original_prev_activations() {
  auto &l = dynamic_cast<data_type_layer<TensorDataType>&>(layer());
  const auto input_tensor_shape = get_input_tensor_shape();
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
  const auto sample_dist = dc::get_hydrogen_data_parallel_distribution(
      l.get_num_dims());
  auto input_local_shape = input_tensor_shape;
  // Set the sample dimension as 0 so that its actual value is
  // calculated by Distconv
  input_local_shape[-1] = 0;

  m_original_inputs.clear();
  m_original_inputs.resize(l.get_num_parents());

  for (int i = 0; i < l.get_num_parents(); ++i) {
    if (l.parent_copy_in_required(i)) {
      m_original_inputs[i] = make_unique<TensorDevType>(
          input_tensor_shape, loc, sample_dist, input_local_shape);
    } else if (l.parent_shuffle_required(i)) {
      // NOTE: previous activations are assumed to be of the same
      // tensor data type.
      // Create a shallow copy of the activations of the prev layer
      const auto &parent_activations =
          dynamic_cast<const TensorDevType&>(
              l.get_parent_layers()[i]->dc().get_activations(l));
      m_original_inputs[i] = make_unique<TensorDevType>(
          parent_activations);
    }
  }
}

template <typename TensorDataType>
void data_type_distconv_layer<TensorDataType>::setup_prev_activations(
    const dc::Dist& dist) {
  auto &l = dynamic_cast<data_type_layer<TensorDataType>&>(layer());
  const auto input_tensor_shape = get_input_tensor_shape();
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);

  for (int i = 0; i < l.get_num_parents(); ++i) {
    if (l.parent_copy_in_required(i) || l.parent_shuffle_required(i)) {
      if (i != 0) LBANN_ERROR("Copyin of non-first tensor not supported yet");
      m_inputs.emplace_back(make_unique<TensorDevType>(
          input_tensor_shape, loc, dist));
      assert0(m_inputs.back()->allocate());
      m_inputs.back()->zero(dc::get_stream());
    } else {
      // Create a shallow copy
      const auto &parent_activations =
          dynamic_cast<const TensorDevType&>(
              l.get_parent_layers()[i]->dc().get_activations(l));
      m_inputs.emplace_back(make_unique<TensorDevType>(parent_activations));
      // Sanity check
      assert_always(parent_activations.get_distribution() == dist);
    }
  }

  dc::MPIPrintStreamDebug() << get_name() << "; "
                            << "prev activations: " << get_prev_activations();
}

template <typename TensorDataType>
dc::Shape data_type_distconv_layer<TensorDataType>::get_output_tensor_shape(
    int output_index) const {
  const auto output_dims = layer().get_output_dims(output_index);
  std::vector<int> output_tensor_shape_v(output_dims.rbegin(), output_dims.rend());
  output_tensor_shape_v.push_back(layer().get_model()->get_max_mini_batch_size());
  return dc::Shape(output_tensor_shape_v);
}

template <typename TensorDataType>
dc::Shape data_type_distconv_layer<TensorDataType>::get_activations_tensor_local_shape() const {
  // TODO: relace the use of layer::get_activations_tensor_local_shape
#if 0
  const auto &parent_activations = layer().get_parent_layers()[0]->dc().get_activations(layer());
  return parent_activations.get_local_shape();
#else
  return dynamic_cast<const data_type_layer<TensorDataType>&>(layer()).get_activations_tensor_local_shape();
#endif
}

template <typename TensorDataType>
void data_type_distconv_layer<TensorDataType>::setup_activations(
    const dc::Dist& dist, bool allocate) {
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
  const dc::Shape output_tensor_shape = get_output_tensor_shape();
  const auto activations_local_shape =
      get_activations_tensor_local_shape();
  m_outputs.emplace_back(make_unique<TensorDevType>(output_tensor_shape,
                                                    loc, dist, activations_local_shape));
  if (allocate) {
    assert0(m_outputs.back()->allocate());
    m_outputs.back()->zero(dc::get_stream());
  }
}

#define PROTO(T)                                \
  template class data_type_distconv_layer<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

}  // namespace lbann
