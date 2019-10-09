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

#ifndef LBANN_LAYERS_LOSS_TOP_K_CATEGORICAL_ACCURACY_HPP_INCLUDED
#define LBANN_LAYERS_LOSS_TOP_K_CATEGORICAL_ACCURACY_HPP_INCLUDED

#include "lbann/layers/layer.hpp"

namespace lbann {


/** @brief
 *
 *  Requires two inputs, which are respectively interpreted as
 *  prediction scores and as a one-hot label vector. The output is one
 *  if the corresponding label matches one of the top-k prediction
 *  scores and is otherwise zero. Ties in the top-k prediction scores
 *  are broken in favor of entries with smaller indices.
 *
 *  @todo Gracefully handle case where label is not a one-hot vector.
 */
template <data_layout T_layout, El::Device Dev>
class top_k_categorical_accuracy_layer : public Layer {
public:

  top_k_categorical_accuracy_layer(lbann_comm *comm, El::Int k)
    : Layer(comm), m_k(k) {
    this->m_expected_num_parent_layers = 2;
  }

  top_k_categorical_accuracy_layer* copy() const override {
    return new top_k_categorical_accuracy_layer(*this);
  }
  std::string get_type() const override { return "top-k accuracy"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  description get_description() const override {
    auto desc = Layer::get_description();
    desc.add("k", m_k);
    return desc;
  }

protected:

  void setup_dims() override {
    Layer::setup_dims();
    set_output_dims({1});

    // Check that input dimensions match
    if (get_input_dims(0) != get_input_dims(1)) {
      const auto& parents = get_parent_layers();
      std::stringstream err;
      err << get_type() << " layer \"" << get_name() << "\" "
          << "has input tensors with different dimensions (";
      for (int i = 0; i < get_num_parents(); ++i) {
        const auto& dims = get_input_dims(i);
        err << (i > 0 ? ", " : "")
            << "layer \"" << parents[i]->get_name() << "\" outputs ";
        for (size_t j = 0; j < dims.size(); ++j) {
          err << (j > 0 ? " x " : "") << dims[j];
        }
      }
      err << ")";
      LBANN_ERROR(err.str());
    }

  }

  void fp_compute() override;

private:

  /** Parameter for top-k search. */
  const El::Int m_k;

};

#ifndef LBANN_TOP_K_CATEGORICAL_ACCURACY_LAYER_INSTANTIATE
extern template class top_k_categorical_accuracy_layer<
  data_layout::DATA_PARALLEL, El::Device::CPU>;
extern template class top_k_categorical_accuracy_layer<
  data_layout::MODEL_PARALLEL, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
extern template class top_k_categorical_accuracy_layer<
  data_layout::DATA_PARALLEL, El::Device::GPU>;
extern template class top_k_categorical_accuracy_layer<
  data_layout::MODEL_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU
#endif // LBANN_TOP_K_CATEGORICAL_ACCURACY_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LOSS_TOP_K_CATEGORICAL_ACCURACY_HPP_INCLUDED
