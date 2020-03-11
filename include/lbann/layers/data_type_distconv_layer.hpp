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
  const TensorDevType& get_activations(const Layer& child) const override;

  /** Get activation tensor. */
  const TensorDevType& get_activations(int child_index = 0) const;
  /** Get activation tensor. */
  TensorDevType& get_activations(int child_index = 0);
  /** Get original activation tensor. */
  const TensorDevType& get_original_activations(int child_index = 0) const;
  /** Get original activation tensor. */
  TensorDevType& get_original_activations(int child_index = 0);

  /** Get previous activation tensor. */
  const TensorDevType& get_prev_activations(int parent_index = 0) const;
  /** Get previous activation tensor. */
  TensorDevType& get_prev_activations(int parent_index = 0);
  /** Get original previous activation tensor. */
  const TensorDevType& get_original_prev_activations(int parent_index = 0) const;
  /** Get original previous activation tensor. */
  TensorDevType& get_original_prev_activations(int parent_index = 0);

  void setup_prev_activations(const dc::Dist& dist) override;
  void setup_original_prev_activations() override;
  void setup_activations(const dc::Dist& dist, bool allocate=true) override;
  void setup_original_activations() override;

 protected:
  std::vector<std::unique_ptr<TensorDevType>> m_inputs;
  std::vector<std::unique_ptr<TensorDevType>> m_original_inputs;
  std::vector<std::unique_ptr<TensorDevType>> m_outputs;
  std::vector<std::unique_ptr<TensorDevType>> m_original_outputs;

  dc::Shape get_input_tensor_shape(int input_index=0) const;
  dc::Shape get_output_tensor_shape(int output_index=0) const;
  dc::Shape get_activations_tensor_local_shape() const;
};

} // namespace lbann

#endif // LBANN_LAYERS_DATA_TYPE_DISTCONV_LAYER_HPP_INCLUDED
