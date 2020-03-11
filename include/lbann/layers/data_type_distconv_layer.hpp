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

#ifndef LBANN_LAYERS_DATA_TYPE_DISTCONV_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_DATA_TYPE_DISTCONV_LAYER_HPP_INCLUDED

#include "lbann/layers/distconv_layer.hpp"
#include "lbann/layers/layer.hpp"

namespace lbann {

template <typename TensorDataType>
class data_type_distconv_layer: public distconv_layer {
public:
  using TensorDevType = dc::TensorDev<TensorDataType>;

  data_type_distconv_layer(Layer& layer): distconv_layer(layer) {}
  virtual ~data_type_distconv_layer() = default;

  /** Get activation tensor corresponding to child layer. */
  const TensorDevType& get_activations(const Layer& child) const override {
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

  /** Get activation tensor. */
  const TensorDevType& get_activations(int child_index = 0) const {
    if (child_index < 0 || child_index >= (int) m_outputs.size()) {
      LBANN_ERROR("attempted to access invalid distconv activation tensor ",
                  "from ", get_name(), " ",
                  "(requested index ", child_index, ", but there are ",
                  m_outputs.size(), " activation tensors)");
    }
    return *m_outputs[child_index];
  }
  TensorDevType& get_activations(int child_index = 0) {
    return const_cast<TensorDevType&>(
        static_cast<const data_type_distconv_layer<TensorDataType>&>(*this).get_activations(child_index));
  }

  void setup_activations(const dc::Dist& dist, bool allocate) override;

 protected:
  std::vector<std::unique_ptr<TensorDevType>> m_outputs;

  dc::Shape get_output_tensor_shape(int output_index=0) const;
  dc::Shape get_activations_tensor_local_shape() const;
};

} // namespace lbann

#endif // LBANN_LAYERS_DATA_TYPE_DISTCONV_LAYER_HPP_INCLUDED
