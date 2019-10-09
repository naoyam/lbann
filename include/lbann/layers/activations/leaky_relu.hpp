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

#ifndef LBANN_LAYERS_ACTIVATIONS_LEAKY_RELU_HPP_INCLUDED
#define LBANN_LAYERS_ACTIVATIONS_LEAKY_RELU_HPP_INCLUDED

#include "lbann/layers/layer.hpp"
#ifdef LBANN_HAS_DISTCONV
#include "lbann/utils/distconv.hpp"
#endif

namespace lbann {

/** @brief
 *
 *  @f[
 *    \text{LeakyReLU}(x; \alpha) =
 *      \begin{cases}
 *        x        & x > 0 \\
 *        \alpha x & x \leq 0
 *      \end{cases}
 *  @f]
 *  See:
 *
 *  Andrew L. Maas, Awni Y. Hannun, and Andrew Y. Ng. "Rectifier
 *  nonlinearities improve neural network acoustic models." In
 *  Proc. ICML, vol. 30, no. 1, p. 3. 2013.
 */
template <data_layout Layout, El::Device Device>
class leaky_relu_layer : public Layer {
public:
  leaky_relu_layer(lbann_comm *comm, DataType negative_slope = 0.01)
    : Layer(comm), m_negative_slope(negative_slope) {}
  leaky_relu_layer* copy() const override { return new leaky_relu_layer(*this); }
  std::string get_type() const override { return "leaky ReLU"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  description get_description() const override {
    auto desc = Layer::get_description();
    desc.add("Negative slope", m_negative_slope);
    return desc;
  }

protected:
  void setup_dims() override {
    Layer::setup_dims();
    set_output_dims(get_input_dims());
  }
  void fp_compute() override;
  void bp_compute() override;

private:
  /** Function slope in negative region. */
  DataType m_negative_slope;

#ifdef LBANN_HAS_DISTCONV
 protected:
  dc::LeakyReLU *m_leaky_relu;
  void fp_compute_distconv();
  void bp_compute_distconv();
 public:
  void setup_tensor_distribution_init(
      std::map<const Layer*, std::array<dc::Dist, dc::num_dists>> &dists,
      std::map<dc::Dist*, std::set<dc::Dist*>> &invariants,
      std::set<dc::Dist*> &updated,
      std::set<dc::Dist*> &fixed) override {
    Layer::setup_tensor_distribution_init(dists, invariants, updated, fixed);
  }
  void setup_tensors_fwd(const std::array<dc::Dist, dc::num_dists> &dists) override {
    Layer::setup_tensors_fwd(dists);
  }
  void setup_tensors_bwd(const std::array<dc::Dist, dc::num_dists> &dists) override {
    Layer::setup_tensors_bwd(dists);
  }
#endif // LBANN_HAS_DISTCONV
};

#ifndef LBANN_LEAKY_RELU_LAYER_INSTANTIATE
extern template class leaky_relu_layer<
  data_layout::DATA_PARALLEL, El::Device::CPU>;
extern template class leaky_relu_layer<
  data_layout::MODEL_PARALLEL, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
extern template class leaky_relu_layer<
  data_layout::DATA_PARALLEL, El::Device::GPU>;
extern template class leaky_relu_layer<
  data_layout::MODEL_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU
#endif // LBANN_LEAKY_RELU_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_ACTIVATIONS_LEAKY_RELU_HPP_INCLUDED
