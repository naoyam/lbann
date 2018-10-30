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

#ifndef LBANN_LAYER_SPLIT_HPP_INCLUDED
#define LBANN_LAYER_SPLIT_HPP_INCLUDED

#include <vector>
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/utils/distconv.hpp"
#endif

namespace lbann {

/** Split layer.
 *  This layer can accommodate an arbitrary number of outputs.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class split_layer : public transform_layer {
 private:

 public:

  split_layer(lbann_comm *comm)
    : transform_layer(comm) {

    // Split layer has no limit on children
    m_expected_num_child_layers = -1;

  }

  split_layer* copy() const override { return new split_layer(*this); }
  std::string get_type() const override { return "split"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream s;
    s << " split; children: ";
    for (size_t h=0; h<this->m_child_layers.size(); h++) {
      s << this->m_child_layers[h]->get_name() << " " << this->m_child_layers[h]->get_type() << " ";
    }
    s << " dataLayout: " << this->get_data_layout_string(get_data_layout());
    return s.str();
  }

 protected:

  void fp_setup_outputs(El::Int mini_batch_size) override {
    const auto& input = get_prev_activations();
    for (int i = 0; i < get_num_children(); ++i) {
      El::LockedView(get_activations(i), input);
    }
  }

  void fp_compute() override {
#ifdef LBANN_HAS_DISTCONV
    if (this->distconv_enabled()) {
      fp_compute_distconv();
    }
#endif
  }

  void bp_compute() override {
    auto& gradient_wrt_input = get_error_signals();
    if (get_num_children() > 0) {
      El::Copy(get_prev_error_signals(0), gradient_wrt_input);
    } else {
      El::Zero(gradient_wrt_input);
    }
    for (int i = 1; i < get_num_children(); ++i) {
      El::Axpy(DataType(1), get_prev_error_signals(i),
               gradient_wrt_input);
    }
  }

#ifdef LBANN_HAS_DISTCONV
 public:
  bool using_distconv() const override {
    char *env = getenv("DISTCONV_DISABLE");
    if (env) {
      std::string s(env);
      if (s.find(get_name()) != std::string::npos) {
        return false;
      }
    }

    // Assumes all child layers are also using distconv. This
    // simplifies fp and bp as it assumes no copyin/copyout is
    // required.
    for (int i = 1; i < get_num_children(); ++i) {
      if (!get_child_layers()[i]->using_distconv()) {
        return false;
      }
    }
    return true;
  }

 protected:
  std::vector<dc::TensorDev> m_prev_error_signals_splits;

  void fp_compute_distconv() {
    assert_always(!m_child_shuffle_required);
  }

 public:
  void setup_tensors_fwd(const std::array<dc::Dist, 4> &dists) override {
    Layer::setup_tensors_fwd(dists);
    if (!this->distconv_enabled()) return;
    this->setup_prev_activations_tensor(dists);
    // activation is just a copy of prev activation
    m_activations_t = m_prev_activations_t;
    this->setup_activations_copyout_tensor(dists);
  }

  void setup_tensors_bwd(const std::array<dc::Dist, 4> &dists) override {
    Layer::setup_tensors_bwd(dists);
    if (!this->distconv_enabled()) return;

    this->setup_prev_error_signals_tensor(dists);
    this->setup_error_signals_tensor(dists);
    this->setup_error_signals_copyout_tensor(dists);

    // Setup copy views for other child layers. Assuming the child
    // layers are also distconv-enabled and using the same parallel
    // strategy.
    assert_always(!m_child_shuffle_required);
    m_prev_error_signals_splits.reserve(get_num_children() - 1);
    for (int i = 1; i < get_num_children(); ++i) {
      m_prev_error_signals_splits.emplace_back(
          get_child_layers()[i]->get_error_signals_t());
    }
  }
#endif

};

#ifdef LBANN_HAS_DISTCONV
template <>
void split_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::bp_compute();
#endif // LBANN_HAS_DISTCONV

} // namespace lbann

#endif // LBANN_LAYER_SPLIT_HPP_INCLUDED
