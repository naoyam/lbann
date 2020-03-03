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

#ifndef LBANN_LAYER_SUM_HPP_INCLUDED
#define LBANN_LAYER_SUM_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/utils/distconv.hpp"
#endif

namespace lbann {

template <data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class sum_layer : public transform_layer {
public:

  sum_layer(lbann_comm *comm)
    : transform_layer(comm) {
    this->m_expected_num_parent_layers = -1; // No limit on parents
  }

  sum_layer* copy() const override { return new sum_layer(*this); }
  std::string get_type() const override { return "sum"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

protected:

  void setup_pointers() override {
    transform_layer::setup_pointers();
    if (get_num_parents() < 1) {
      std::stringstream err;
      err << get_type() << " layer \"" << get_name() << "\" "
          << "has no parent layers";
      LBANN_ERROR(err.str());
    }
  }

  void setup_dims() override {
    transform_layer::setup_dims();
    set_output_dims(get_input_dims());

    // Check that input dimensions match
    const auto& output_dims = get_output_dims();
    for (int i = 0; i < get_num_parents(); ++i) {
      if (get_input_dims(i) != output_dims) {
        const auto& parents = get_parent_layers();
        std::stringstream err;
        err << get_type() << " layer \"" << get_name() << "\" "
            << "has input tensors with incompatible dimensions (";
        for (int j = 0; j < get_num_parents(); ++j) {
          const auto& dims = get_input_dims(j);
          err << (j > 0 ? ", " : "")
              << "layer \"" << parents[j]->get_name() << "\" outputs ";
          for (size_t k = 0; k < dims.size(); ++k) {
            err << (k > 0 ? " x " : "") << dims[k];
          }
        }
        err << ")";
        LBANN_ERROR(err.str());
      }
    }

  }

  void fp_compute() override {
    auto& output = get_activations();
    El::Copy(get_prev_activations(0), output);
    for (int i = 1; i < get_num_parents(); ++i) {
      El::Axpy(DataType(1), get_prev_activations(i), output);
    }
  }

  void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) override {
    const auto& gradient_wrt_output = get_prev_error_signals();
    for (int i = 0; i < get_num_parents(); ++i) {
      El::LockedView(get_error_signals(i), gradient_wrt_output);
    }
  }

  void bp_compute() override {}

#ifdef LBANN_HAS_DISTCONV
 protected:
  std::vector<dc::TensorDev> m_prev_activations_siblings;
  std::vector<dc::TensorDev> m_error_signals_siblings;

 public:

  using Layer::get_error_signals_t;

  const dc::TensorDev &get_error_signals_t(const Layer &parent) const {
    const auto parents = get_parent_layers();
    for (int i = 0; i < (int)parents.size(); ++i) {
      if (parents[i] == &parent) {
        if (i == 0) {
          return m_error_signals_t;
        } else {
          return m_error_signals_siblings[i-1];
        }
      }
    }
    LBANN_ERROR("No such parent found");
  }

  void setup_tensors_fwd(const std::array<dc::Dist, dc::num_dists> &dists) override {
    Layer::setup_tensors_fwd(dists);
    if (!this->distconv_enabled()) return;
    this->setup_prev_activations_tensor(dists);
    this->setup_activations_tensor(dists);
    this->setup_activations_copyout_tensor(dists);

    m_prev_activations_siblings.reserve(get_num_parents() - 1);
    for (int i = 1; i < get_num_parents(); ++i) {
      if (m_parent_shuffle_required[i] ||
          m_parent_copy_in_required[i]) {
        LBANN_ERROR("Copyin non-first tensor not supported");
      }
      m_prev_activations_siblings.emplace_back(
          get_parent_layers()[i]->get_activations_t(*this));
    }
  }

  void setup_tensors_bwd(const std::array<dc::Dist, dc::num_dists> &dists) override {
    Layer::setup_tensors_bwd(dists);
    if (!this->distconv_enabled()) return;

    this->setup_prev_error_signals_tensor(dists);
    m_error_signals_t = m_prev_error_signals_t;
    for (int i = 1; i < get_num_parents(); ++i) {
      m_error_signals_siblings.emplace_back(
          m_prev_error_signals_t);
    }
    this->setup_error_signals_copyout_tensor(dists);
  }

#endif // LBANN_HAS_DISTCONV

};

#ifdef LBANN_HAS_DISTCONV
template <>
void sum_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::fp_compute();
#endif // LBANN_HAS_DISTCONV

#ifndef LBANN_SUM_LAYER_INSTANTIATE
extern template class sum_layer<data_layout::DATA_PARALLEL, El::Device::CPU>;
extern template class sum_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
extern template class sum_layer<data_layout::DATA_PARALLEL, El::Device::GPU>;
extern template class sum_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU
#endif // LBANN_SUM_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_SUM_HPP_INCLUDED
