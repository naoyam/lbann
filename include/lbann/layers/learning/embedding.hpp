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

#ifndef LBANN_LAYERS_LEARNING_EMBEDDING_HPP_INCLUDED
#define LBANN_LAYERS_LEARNING_EMBEDDING_HPP_INCLUDED

#include "lbann/layers/layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/memory.hpp"

namespace lbann {

/** @brief Lookup table to vectors of fixed size.
 *
 *  Takes a scalar input, interprets it as an index, and outputs the
 *  corresponding vector. The number of embedding vectors and the size
 *  of vectors are fixed. If the index is out-of-range, then the
 *  output is a vector of zeros.
 *
 *  The embedding vectors are stored in an
 *  @f$ \text{embedding\_dim} \times \text{num\_embeddings} @f$
 *  weights matrix. Note that this is the transpose of the weights in
 *  the PyTorch embedding layer.
 */
template <data_layout Layout, El::Device Device>
class embedding_layer : public Layer {
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "embedding layer only supports data parallel layout");
public:

  /**
   *  @param comm           LBANN communicator.
   *  @param num_embeddings Size of dictionary of embeddings.
   *  @param embedding_dim  Size of embedding vectors.
   *  @param padding_idx    If set, then the corresponding embedding
   *                        vector is initialized with zeros. The
   *                        objective function gradient w.r.t. this
   *                        embedding vector is always zero.
   */
  embedding_layer(lbann_comm* comm,
                  size_t num_embeddings,
                  size_t embedding_dim,
                  El::Int padding_idx=-1);

  embedding_layer(const embedding_layer& other);
  embedding_layer& operator=(const embedding_layer& other);
  ~embedding_layer() = default;

  embedding_layer* copy() const override {
    return new embedding_layer(*this);
  }

  std::string get_type() const override { return "embedding"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  description get_description() const override;

protected:

  void setup_matrices(const El::Grid& grid) override;
  void setup_dims() override;
  void setup_data() override;

  void fp_compute() override;
  void bp_compute() override;

private:

  /** Size of dictionary of embeddings. */
  size_t m_num_embeddings;
  /** Size of embedding vectors. */
  size_t m_embedding_dim;
  /** If the padding index is set, then the corresponding embedding
   *  vector is initialized with zeros. The objective function
   *  gradient w.r.t. this embedding vector is always zero.
   */
  El::Int m_padding_idx;

  /** Gradient w.r.t. embedding weights. */
  std::unique_ptr<AbsDistMat> m_gradient_wrt_embeddings;

};

// =========================================================
// Implementation
// =========================================================

template <data_layout Layout, El::Device Device>
embedding_layer<Layout,Device>::embedding_layer(
  lbann_comm* comm,
  size_t num_embeddings,
  size_t embedding_dim,
  El::Int padding_idx)
  : Layer(comm),
    m_num_embeddings{num_embeddings},
    m_embedding_dim{embedding_dim},
    m_padding_idx{padding_idx} {}

template <data_layout Layout, El::Device Device>
embedding_layer<Layout,Device>::embedding_layer(
  const embedding_layer<Layout,Device>& other)
  : Layer(other),
    m_num_embeddings{other.m_num_embeddings},
    m_embedding_dim{other.m_embedding_dim},
    m_padding_idx{other.m_padding_idx},
    m_gradient_wrt_embeddings(other.m_gradient_wrt_embeddings
                              ? other.m_gradient_wrt_embeddings->Copy()
                              : nullptr) {}

template <data_layout Layout, El::Device Device>
embedding_layer<Layout,Device>& embedding_layer<Layout,Device>::operator=(
  const embedding_layer<Layout,Device>& other) {
  Layer::operator=(other);
  m_num_embeddings = other.m_num_embeddings;
  m_embedding_dim = other.m_embedding_dim;
  m_padding_idx = other.m_padding_idx;
  m_gradient_wrt_embeddings.reset(other.m_gradient_wrt_embeddings
                                  ? other.m_gradient_wrt_embeddings->Copy()
                                  : nullptr);
  return *this;
}

template <data_layout Layout, El::Device Device>
description embedding_layer<Layout,Device>::get_description() const {
  auto desc = Layer::get_description();
  desc.add("Num embeddings", m_num_embeddings);
  desc.add("Embedding dim", m_embedding_dim);
  desc.add("Padding index", m_padding_idx);
  return desc;
}

template <data_layout Layout, El::Device Device>
void embedding_layer<Layout,Device>::setup_dims() {
  Layer::setup_dims();

  // Make sure input dimensions are valid
  if (this->get_input_size() != 1) {
    const auto& dims = this->get_input_dims();
    std::ostringstream dims_ss;
    for (size_t i = 0; i < dims.size(); ++i) {
      dims_ss << (i > 0 ? "x" : "") << dims[i];
    }
    LBANN_ERROR(this->get_type()," layer \"",this->get_name(),"\" ",
                "recieved an input tensor with invalid dimensions "
                "(expected 1, got ",dims_ss.str(),")");
  }

  // Output is size of embedding vector
  this->set_output_dims({static_cast<int>(m_embedding_dim)});

}

template <data_layout Layout, El::Device Device>
void embedding_layer<Layout,Device>::setup_data() {
  Layer::setup_data();

  // Construct default weights if needed
  // Note: Randomly drawn from normal distribution with mean 0 and
  // standard deviation 1.
  if (this->m_weights.empty()) {
    auto w = make_unique<weights>(get_comm());
    auto init = make_unique<normal_initializer>(0,1);
    auto opt = std::unique_ptr<optimizer>(m_model->create_optimizer());
    w->set_name(this->get_name() + "_weights");
    w->set_initializer(std::move(init));
    w->set_optimizer(std::move(opt));
    this->m_weights.push_back(w.get());
    this->m_model->add_weights(std::move(w));
  }
  if (this->m_weights.size() != 1) {
    LBANN_ERROR("attempted to setup ",
                this->get_type()," layer \"",this->get_name(),"\" ",
                "with an invalid number of weights ",
                "(expected 1, found ",this->m_weights.size(),")");
  }

  // Initialize dictionary
  auto& embeddings = *m_weights[0];
  auto matrix_dist = get_prev_activations().DistData();
  matrix_dist.colDist = El::STAR;
  matrix_dist.rowDist = El::STAR;
  embeddings.set_dims({static_cast<int>(m_embedding_dim)},
                      {static_cast<int>(m_num_embeddings)});
  embeddings.set_matrix_distribution(matrix_dist);
  embeddings.setup();

  // Zero out embedding vector for padding index
  if (0 <= m_padding_idx
      && m_padding_idx < static_cast<El::Int>(m_embedding_dim)) {
    auto& embedding_values = embeddings.get_values();
    std::unique_ptr<AbsDistMat> pad_embedding(
      embedding_values.Construct(embedding_values.Grid(),
                                 embedding_values.Root()));
    El::View(*pad_embedding, embedding_values, El::ALL, El::IR(m_padding_idx));
    El::Zero(*pad_embedding);
  }

  // Initialize gradient w.r.t. embeddings
  m_gradient_wrt_embeddings->Resize(m_embedding_dim, m_num_embeddings);

}

#ifndef LBANN_EMBEDDING_LAYER_INSTANTIATE
extern template class embedding_layer<
  data_layout::DATA_PARALLEL, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
extern template class embedding_layer<
  data_layout::DATA_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU
#endif // LBANN_EMBEDDING_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LEARNING_EMBEDDING_HPP_INCLUDED
