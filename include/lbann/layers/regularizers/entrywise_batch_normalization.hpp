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

#ifndef LBANN_LAYERS_REGULARIZERS_ENTRYWISE_BATCH_NORMALIZATION_HPP_INCLUDED
#define LBANN_LAYERS_REGULARIZERS_ENTRYWISE_BATCH_NORMALIZATION_HPP_INCLUDED

#include "lbann/layers/layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/memory.hpp"

namespace lbann {

/** @brief
 *
 *  Each input entry is normalized across the mini-batch to have zero
 *  mean and unit standard deviation. This uses the standard approach
 *  of maintaining the running mean and standard deviation (with
 *  exponential decay) for use at test time. See:
 *
 *  Sergey Ioffe and Christian Szegedy. "Batch Normalization:
 *  Accelerating Deep Network Training by Reducing Internal Covariate
 *  Shift." In International Conference on Machine Learning,
 *  pp. 448-456. 2015.
 */
template <data_layout Layout, El::Device Device>
class entrywise_batch_normalization_layer : public Layer {
public:

  entrywise_batch_normalization_layer(lbann_comm* comm,
                                      DataType decay=0.9,
                                      DataType epsilon=1e-5)
    : Layer(comm), m_decay(decay), m_epsilon(epsilon) {}

  entrywise_batch_normalization_layer(const entrywise_batch_normalization_layer& other)
    : Layer(other),
      m_decay(other.m_decay),
      m_epsilon(other.m_epsilon),
      m_batch_statistics(other.m_batch_statistics ?
                         other.m_batch_statistics->Copy() :
                         nullptr),
      m_batch_statistics_gradient(other.m_batch_statistics_gradient ?
                                  other.m_batch_statistics_gradient->Copy() :
                                  nullptr) {}

  entrywise_batch_normalization_layer& operator=(const entrywise_batch_normalization_layer& other) {
    Layer::operator=(other);
    m_decay = other.m_decay;
    m_epsilon = other.m_epsilon;
    m_batch_statistics.reset(other.m_batch_statistics ?
                             other.m_batch_statistics->Copy() :
                             nullptr);
    m_batch_statistics_gradient.reset(other.m_batch_statistics_gradient ?
                                      other.m_batch_statistics_gradient->Copy() :
                                      nullptr);
    return *this;
  }

  entrywise_batch_normalization_layer* copy() const override { return new entrywise_batch_normalization_layer(*this); }
  std::string get_type() const override { return "entry-wise batch normalization"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  description get_description() const override {
    auto desc = Layer::get_description();
    desc.add("Decay", m_decay);
    desc.add("Epsilon", m_epsilon);
    return desc;
  }

protected:

  void setup_matrices(const El::Grid& grid) override {
    Layer::setup_matrices(grid);
    auto dist = get_prev_activations().DistData();
    dist.rowDist = El::STAR;
    m_batch_statistics.reset(AbsDistMat::Instantiate(dist));
    m_batch_statistics_gradient.reset(AbsDistMat::Instantiate(dist));
  }

  void setup_data() override {
    Layer::setup_data();

    // Initialize output dimensions
    set_output_dims(get_input_dims());
    const auto output_dims = get_output_dims();
    const auto output_size = get_output_size();

    // Initialize default weights if none are provided
    if (this->m_weights.size() > 2) {
      std::stringstream err;
      err << "attempted to setup layer \"" << m_name << "\" "
          << "with an invalid number of weights "
          << "(found " << this->m_weights.size() << ", expected 2)";
      LBANN_ERROR(err.str());
    }
    this->m_weights.resize(2, nullptr);
    if (this->m_weights[0] == nullptr) {
      auto w = make_unique<weights>(get_comm());
      auto init = make_unique<constant_initializer>(DataType{0});
      w->set_name(get_name() + "_running_mean");
      w->set_initializer(std::move(init));
      this->m_weights[0] = w.get();
      this->m_model->add_weights(std::move(w));
    }
    if (this->m_weights[1] == nullptr) {
      auto w = make_unique<weights>(get_comm());
      auto init = make_unique<constant_initializer>(DataType{1});
      w->set_name(get_name() + "_running_variance");
      w->set_initializer(std::move(init));
      this->m_weights[1] = w.get();
      this->m_model->add_weights(std::move(w));
    }

    // Setup weights
    auto dist = get_prev_activations().DistData();
    dist.rowDist = El::STAR;
    for (auto* w : this->m_weights) {
      w->set_dims(output_dims);
      w->set_matrix_distribution(dist);
    }

    // Initialize matrices
    m_batch_statistics->AlignWith(dist);
    m_batch_statistics->Resize(output_size, 2);
    m_batch_statistics_gradient->AlignWith(dist);
    m_batch_statistics_gradient->Resize(output_size, 2);

  }

  void fp_setup_outputs(El::Int mini_batch_size) override {
    Layer::fp_setup_outputs(mini_batch_size);
    const auto& input = get_prev_activations();
    const auto input_size = get_input_size();

    // Make sure batch statistics tensor is aligned with input tensor
    m_batch_statistics->Empty(false);
    m_batch_statistics->AlignWith(input);
    m_batch_statistics->Resize(input_size, 2);

#if 0 /// @todo See https://github.com/LLNL/lbann/issues/1123

    // Check that weights tensors is aligned with input tensor
    /// @todo Realign tensors if misaligned
    bool aligned = true;
    try {
      const auto& running_mean = m_weights[0]->get_values();
      const auto& running_var = m_weights[1]->get_values();
      aligned = (input.ColAlign() == running_mean.ColAlign()
                 && input.RowAlign() == running_mean.RowAlign()
                 && input.ColAlign() == running_var.ColAlign()
                 && input.RowAlign() == running_var.RowAlign());
    }
    catch (const exception& e) {
      // An exception is thrown if you try accessing weights values
      // before they are initialized. We don't care if this case is
      // aligned, so it's safe to ignore.
    }
    if (!aligned) {
      std::ostringstream err;
      err << this->get_type() << " layer \"" << this->get_name() << "\" "
          << "has misaligned input and weights matrices";
      LBANN_ERROR(err.str());
    }

#endif // 0

  }

  void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) override {
    Layer::bp_setup_gradient_wrt_inputs(mini_batch_size);
    m_batch_statistics_gradient->Empty(false);
    m_batch_statistics_gradient->AlignWith(get_prev_activations());
    m_batch_statistics_gradient->Resize(get_input_size(), 2);
  }

  void fp_compute() override;
  void bp_compute() override;

private:

  /** Decay rate for the running statistics. */
  DataType m_decay;
  /** Small number to avoid division by zero. */
  DataType m_epsilon;

  /** @brief Current mini-batch statistics.
   *
   *  These are fused for performance when doing non-local batchnorm.
   */
  std::unique_ptr<AbsDistMat> m_batch_statistics;
  /** @brief Gradients w.r.t. current mini-batch statistics.
   *
   * These are fused for performance when doing non-local batchnorm.
   */
  std::unique_ptr<AbsDistMat> m_batch_statistics_gradient;

};

#ifndef LBANN_ENTRYWISE_BATCH_NORMALIZATION_LAYER_INSTANTIATE
extern template class entrywise_batch_normalization_layer<
  data_layout::DATA_PARALLEL, El::Device::CPU>;
extern template class entrywise_batch_normalization_layer<
  data_layout::MODEL_PARALLEL, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
extern template class entrywise_batch_normalization_layer<
  data_layout::DATA_PARALLEL, El::Device::GPU>;
extern template class entrywise_batch_normalization_layer<
  data_layout::MODEL_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU
#endif // LBANN_ENTRYWISE_BATCH_NORMALIZATION_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_REGULARIZERS_ENTRYWISE_BATCH_NORMALIZATION_HPP_INCLUDED
