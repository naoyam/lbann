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

#ifndef LBANN_LAYER_SPLIT_HPP_INCLUDED
#define LBANN_LAYER_SPLIT_HPP_INCLUDED

#include <vector>
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/utils/distconv.hpp"
#endif

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType>
class split_distconv_layer: public data_type_distconv_layer<TensorDataType> {
 public:
  using TensorDevType = typename data_type_distconv_layer<TensorDataType>::TensorDevType;

  split_distconv_layer(Layer& layer): data_type_distconv_layer<TensorDataType>(layer) {}
  virtual ~split_distconv_layer() = default;
  void setup_activations(const dc::Dist& dist) override {
    const auto &parent_activations =
        dynamic_cast<const TensorDevType&>(this->layer().get_parent_layers()[0]->dc().get_activations(this->layer()));
    for (int i = 0; i < this->layer().get_num_children(); ++i) {
      this->m_outputs.emplace_back(make_unique<TensorDevType>(parent_activations));
    }
  }
};

#endif // LBANN_HAS_DISTCONV

/** @brief Present input tensor to multiple outputs. */
template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class split_layer : public transform_layer<TensorDataType> {
public:

  split_layer(lbann_comm *comm) : transform_layer<TensorDataType>(comm) {
    this->m_expected_num_child_layers = -1; // No limit on children
  }

  split_layer* copy() const override { return new split_layer(*this); }
  std::string get_type() const override { return "split"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

protected:

  void setup_dims() override {
    data_type_layer<TensorDataType>::setup_dims();
    for (int i = 0; i < this->get_num_children(); ++i) {
      this->set_output_dims(this->get_input_dims(), i);
    }
  }

  void fp_setup_outputs(El::Int mini_batch_size) override {
    const auto& input = this->get_prev_activations();
    for (int i = 0; i < this->get_num_children(); ++i) {
      El::LockedView(this->get_activations(i), input);
    }
  }

  void fp_compute() override {}

  void bp_compute() override;

#ifdef LBANN_HAS_DISTCONV
 protected:
  using TensorDevType = typename data_type_layer<TensorDataType>::TensorDevType;
  std::vector<TensorDevType> m_prev_error_signals_siblings;

  void setup_distconv_layer() override {
    this->get_dc() = make_unique<split_distconv_layer<TensorDataType>>(*this);
  }

  void fp_compute_distconv() {}

 public:

  void init_distribution(
      std::map<const Layer*, std::array<lbann::dc::Dist, dc::num_dists>> &dists,
      std::map<dc::Dist*, std::set<dc::Dist*>> &equivalents,
      std::set<dc::Dist*> &updated,
      std::set<dc::Dist*> &invariants) override {
    data_type_layer<TensorDataType>::init_distribution(
        dists, equivalents, updated, invariants);
    if (!this->distconv_enabled()) return;
    auto &layer_dists = dists[this];
    // x == y
    equivalents[&layer_dists[0]].insert(&layer_dists[1]);
    equivalents[&layer_dists[1]].insert(&layer_dists[0]);
    // dx == dy
    equivalents[&layer_dists[2]].insert(&layer_dists[3]);
    equivalents[&layer_dists[3]].insert(&layer_dists[2]);
  }

  void setup_tensors_bwd(const std::array<dc::Dist, dc::num_dists> &dists) override {
    data_type_layer<TensorDataType>::setup_tensors_bwd(dists);
    if (!this->distconv_enabled()) return;

    this->setup_prev_error_signals_tensor(dists);
    this->setup_error_signals_tensor(dists);
    this->setup_error_signals_copyout_tensor(dists);

    m_prev_error_signals_siblings.reserve(this->get_num_children() - 1);
    for (int i = 1; i < this->get_num_children(); ++i) {
      if (this->child_shuffle_required(i) || this->child_copy_out_required(i)) {
        LBANN_ERROR("Copyout non-first tensor not supported");
      }
      m_prev_error_signals_siblings.emplace_back(
          dynamic_cast<const data_type_layer<TensorDataType>*>(
              this->get_child_layers()[i])->get_error_signals_t(*this));
    }
  }
#endif

};

LBANN_DEFINE_LAYER_BUILDER(split);

#ifndef LBANN_SPLIT_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class split_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class split_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_SPLIT_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_SPLIT_HPP_INCLUDED
