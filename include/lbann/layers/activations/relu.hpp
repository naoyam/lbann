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

#ifndef LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED
#define LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann_config.hpp"
#include "lbann/distconv.hpp"

namespace lbann {

/** Rectified linear unit activation function.
 *  See https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
template <data_layout T_layout, El::Device Dev>
class relu_layer : public entrywise_activation_layer {

 private:

#ifdef LBANN_HAS_CUDNN
  /** Activation descriptor. */
  cudnnActivationDescriptor_t m_activation_cudnn_desc;
#endif // LBANN_HAS_CUDNN

 public:
  relu_layer(lbann_comm *comm,
             cudnn::cudnn_manager *cudnn = nullptr)
    : entrywise_activation_layer(comm) {
  #ifdef LBANN_HAS_CUDNN
    m_activation_cudnn_desc = nullptr;
    this->m_cudnn = cudnn;
  #endif // LBANN_HAS_CUDNN
  }

  relu_layer(const relu_layer& other) :
    entrywise_activation_layer(other) {
  #ifdef LBANN_HAS_CUDNN
    m_activation_cudnn_desc = nullptr;
    cudnn::copy_activation_cudnn_desc(other.m_activation_cudnn_desc,
                                      m_activation_cudnn_desc);
  #endif // LBANN_HAS_CUDNN
  }

  relu_layer& operator=(const relu_layer& other) {
    entrywise_activation_layer::operator=(other);
  #ifdef LBANN_HAS_CUDNN
    cudnn::copy_activation_cudnn_desc(other.m_activation_cudnn_desc,
                                      m_activation_cudnn_desc);
  #endif // LBANN_HAS_CUDNN
    return *this;
  }

  ~relu_layer() override {
  #ifdef LBANN_HAS_CUDNN
    if (m_activation_cudnn_desc != nullptr) {
      CHECK_CUDNN(cudnnDestroyActivationDescriptor(m_activation_cudnn_desc));
    }
  #endif // LBANN_HAS_CUDNN
  }

  relu_layer* copy() const override { return new relu_layer(*this); }
  std::string get_type() const override { return "ReLU"; }

  /** Returns description of ctor params */
  std::string get_description() const override {
    return std::string {} +
     " relu" + " dataLayout: " + this->get_data_layout_string(get_data_layout());
  }

  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  void setup_gpu() override {
    entrywise_activation_layer::setup_gpu();
  #ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
  #else
    if (m_activation_cudnn_desc != nullptr) {
      CHECK_CUDNN(cudnnDestroyActivationDescriptor(m_activation_cudnn_desc));
      m_activation_cudnn_desc = nullptr;
    }
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&m_activation_cudnn_desc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(m_activation_cudnn_desc,
                                             CUDNN_ACTIVATION_RELU,
                                             CUDNN_PROPAGATE_NAN,
                                             0.0));

#endif // LBANN_HAS_CUDNN
  }

 protected:

  DataType activation(DataType x) const override {
    return x > DataType(0) ? x : DataType(0);
  }

  DataType activation_derivative(DataType x) const override {
    return x > DataType(0) ? DataType(1) : DataType(0);
  }

  void fp_compute() override;
  void bp_compute() override;

  void fp_compute_gpu() override {
  #ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
  #else
#ifdef LBANN_HAS_DISTCONV
    if (m_distconv_enabled) {
      early_terminate();
      fp_compute_distconv();
      if (m_exit_count == 0) {
        dump_tensor(m_activations_t, get_name() + "_activations");
      } else {
        return;
      }
    }
#endif
    const DataType one = 1;
    const DataType zero = 0;
    CHECK_CUDNN(cudnnActivationForward(this->m_cudnn->get_handle(),
                                       m_activation_cudnn_desc,
                                       &one,
                                       this->m_prev_activations_cudnn_desc,
                                       get_prev_activations().LockedBuffer(),
                                       &zero,
                                       this->m_activations_cudnn_desc,
                                       get_activations().Buffer()));
#ifdef LBANN_HAS_DISTCONV
    if (m_distconv_enabled && m_exit_count == 0) {
      assert0(dc::tensor::View(
          m_activations_copyout, m_activations_d[0].get_data(0)));
      dump_tensor(m_activations_copyout,
                  get_name() + "_activations_original");
    }
#endif // LBANN_HAS_DISTCONV
  #endif // LBANN_HAS_CUDNN
  }
  

  void bp_compute_gpu() override {
  #ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
  #else
#if 1
#ifdef LBANN_HAS_DISTCONV
    if (m_distconv_enabled) {
      bp_compute_distconv();
      if (m_exit_count == 0) {
        dump_tensor(m_error_signals_t,
                    get_name() + "_error_signals");
      } else {
        return;
      }
    }
#endif // LBANN_HAS_DISTCONV
#endif
    const DataType one = 1;
    CHECK_CUDNN(cudnnActivationBackward(this->m_cudnn->get_handle(),
                                        m_activation_cudnn_desc,
                                        &one,
                                        this->m_activations_cudnn_desc,
                                        get_activations().LockedBuffer(),
                                        this->m_prev_error_signals_cudnn_desc,
                                        get_prev_error_signals().LockedBuffer(),
                                        this->m_prev_activations_cudnn_desc,
                                        get_prev_activations().LockedBuffer(),
                                        &one,
                                        this->m_error_signals_cudnn_desc,
                                        get_error_signals().Buffer()));
#ifdef LBANN_HAS_DISTCONV
    if (m_distconv_enabled && m_exit_count == 0) {
      assert0(dc::tensor::View(m_error_signals_copyout,
                               m_error_signals_d[0].get_data(0)));      
      dump_tensor(m_error_signals_copyout,
                  get_name() + "_error_signals_original");
    }
#endif // LBANN_HAS_DISTCONV
  #endif // LBANN_HAS_CUDNN
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
    return true;
  }

  void setup_tensor_distribution_init(
      std::map<const Layer*, std::array<Dist, 4>> &dists,  
      std::map<Dist*, std::set<Dist*>> &invariants,
      std::set<Dist*> &updated,
      std::set<Dist*> &fixed) override {
    Layer::setup_tensor_distribution_init(
        dists, invariants, updated, fixed);
    if (!using_distconv()) return;
    auto &layer_dists = dists[this];
#ifdef DISTCONV_USE_SAME_RELU_CALL_AS_LBANN
    // This isn't necessary for cuDNN, but necessary to make it work
    // with the ordering of ReLU parameters used in LBANN
    // x == y
    invariants[&layer_dists[0]].insert(
        &layer_dists[1]);
    invariants[&layer_dists[1]].insert(
        &layer_dists[0]);
#endif    
    // x == dx
    invariants[&layer_dists[0]].insert(
        &layer_dists[2]);
    invariants[&layer_dists[2]].insert(
        &layer_dists[0]);
    //y == dy
    invariants[&layer_dists[1]].insert(
        &layer_dists[3]);
    invariants[&layer_dists[3]].insert(
        &layer_dists[1]);
  }

  void setup_tensors_fwd(const std::array<Dist, 4> &dists) override {   
    Layer::setup_tensors_fwd(dists);    
    if (!m_distconv_enabled) return;
    
    setup_prev_activations_tensor(dists);
    setup_activations_tensor(dists);
    setup_activations_copyout_tensor(dists);    
  }

  void setup_tensors_bwd(const std::array<Dist, 4> &dists) override {
    Layer::setup_tensors_bwd(dists);    
    if (!m_distconv_enabled) return;
    
    // REFACTORING: duplicated at convolution::setup_tensors
    const Array4 input_tensor_shape =
        {m_prev_neuron_dims[2], m_prev_neuron_dims[1],
         m_prev_neuron_dims[0], this->m_model->get_max_mini_batch_size()};
    const LocaleMPI loc(m_comm->get_model_comm().comm, false);
    const Array4 sample_block_size = {1, 1, 1, 1};
    const Dist sample_dist = Dist({1, 1, 1, m_comm->get_procs_per_model()});    
    Array4 input_local_shape = input_tensor_shape;
    // Assuming single GPU per rank
    input_local_shape[3] = m_max_mini_batch_size_per_gpu;
    //const Array4 spatial_local_size = {0, 0, 0, 0};
    const Array4 output_tensor_shape =
        {m_neuron_dims[2], m_neuron_dims[1],
         m_neuron_dims[0], this->m_model->get_max_mini_batch_size()};
    Array4 output_local_shape = output_tensor_shape;
    output_local_shape[3] = m_max_mini_batch_size_per_gpu;

    // prev_error_signals
    if (m_child_copy_required) {
      m_prev_error_signals_const_view = TensorDev(output_tensor_shape, loc,
                                                  sample_dist,
                                                  output_local_shape,
                                                  sample_block_size);
      m_prev_error_signals_t = TensorDev(output_tensor_shape, loc,
                                         dists[3],
                                         m_activations_t.get_local_shape(),
                                         m_output_decomposition_block);
      assert0(m_prev_error_signals_t.allocate());
      m_prev_error_signals_t.zero();
      m_prev_error_signals_shuffler = new TensorShuffler(
          m_prev_error_signals_const_view, m_prev_error_signals_t);
    } else {
      m_prev_error_signals_t = get_child_layers()[0]->get_error_signals_t();
      assert_always(m_prev_error_signals_t.get_distribution() ==
                    dists[3]);
      assert_always(m_prev_error_signals_t.get_requested_local_block() ==
                    m_output_decomposition_block);
    }

    // error_signals
    m_error_signals_t = TensorDev(input_tensor_shape, loc,
                                  dists[2], m_prev_error_signals_t.get_local_shape(),
                                  m_input_decomposition_block);
    assert0(m_error_signals_t.allocate());
    m_error_signals_t.zero();

    m_error_signals_copyout = TensorDev(input_tensor_shape, loc, sample_dist,
                                        input_local_shape, sample_block_size);
    if (m_parent_copy_required) {
      m_error_signals_shuffler = new TensorShuffler(
          m_error_signals_t, m_error_signals_copyout);
    }

    // Init the dc::Pooling layer
    m_relu = new dc::ReLU<dc::cudnn::BackendCUDNN>(
        *this->m_cudnn->get_distconv_backend());

    m_relu->setup(m_prev_activations_t, m_activations_t,
                  m_error_signals_t, m_prev_error_signals_t);
  }
  
  void fp_compute_distconv() {
    MPIPrintStreamDebug() << get_name() << ": " << __FUNCTION__ << "\n";    
    assert_always(m_distconv_enabled);
    // Useful constants
    const DataType one = 1;
    const DataType zero = 0;
    m_relu->set_num_samples(this->m_model->get_current_mini_batch_size());        
    if (m_parent_copy_required) {
      MPIPrintStreamDebug() << "Copying from sample decomposition\n";
      assert0(dc::tensor::View(
          m_prev_activations_const_view,
          m_prev_activations_d[0].get_locked_data(0)));
      m_prev_activations_shuffler->shuffle_forward(
          m_prev_activations_const_view.get_const_base_ptr(),
          m_prev_activations_t.get_base_ptr(),
          this->m_cudnn->get_stream(0));
    } else {
      MPIPrintStreamDebug()
          << "Directly reading activations of previous layer\n";
    }
    m_relu->forward(one, m_prev_activations_t,
                    zero, m_activations_t);
    if (m_child_copy_required) {
      MPIPrintStreamDebug() << "Copying back to sample decomposition\n";      
      assert0(dc::tensor::View(
          m_activations_copyout, m_activations_d[0].get_data(0)));
      m_activations_shuffler->shuffle_forward(
          m_activations_t.get_const_base_ptr(),
          m_activations_copyout.get_base_ptr(),
          this->m_cudnn->get_stream(0));
    }
  }

  void bp_compute_distconv() {
    MPIPrintStreamDebug() << get_name() << ": " << __FUNCTION__ << "\n";    
    assert_always(m_distconv_enabled);
    const DataType one = 1;
    if (m_child_copy_required) {
      MPIPrintStreamDebug() << "Copying from sample decomposition\n";            
      assert0(dc::tensor::View(
          m_prev_error_signals_const_view,
          m_prev_error_signals_d[0].get_locked_data(0)));
      m_prev_error_signals_shuffler->shuffle_forward(
          m_prev_error_signals_const_view.get_const_base_ptr(),
          m_prev_error_signals_t.get_base_ptr(),
          this->m_cudnn->get_stream(0));
    } else {
      MPIPrintStreamDebug()
          << "Directly reading activations of previous layer\n";
    }
#ifdef DISTCONV_ZERO_OUT_ERROR_SIGNALS
    m_error_signals_t.zero();
    m_relu->backward(one, m_activations_t, m_prev_error_signals_t,
                     m_prev_activations_t, one, m_error_signals_t);
#else    
    m_relu->backward(one, m_activations_t, m_prev_error_signals_t,
                     m_prev_activations_t, DataType(0), m_error_signals_t);
#endif
    if (m_parent_copy_required) {
      if (m_exit_count != 0) {
        MPIPrintStreamDebug() << "Copying back to sample decomposition\n";            
        assert0(dc::tensor::View(m_error_signals_copyout, m_error_signals_d[0].get_data(0)));
        m_error_signals_shuffler->shuffle_forward(
            m_error_signals_t.get_const_base_ptr(),
            m_error_signals_copyout.get_base_ptr(),
            this->m_cudnn->get_stream(0));
      }
    }
  }
  
 protected:
  dc::ReLU<dc::cudnn::BackendCUDNN> *m_relu;
#endif

};

} // namespace lbann

#endif // LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED
