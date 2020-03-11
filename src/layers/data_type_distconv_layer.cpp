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
