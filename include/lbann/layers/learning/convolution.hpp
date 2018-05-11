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

#ifndef LBANN_LAYER_CONVOLUTION_HPP_INCLUDED
#define LBANN_LAYER_CONVOLUTION_HPP_INCLUDED

#include <vector>
#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann_config.hpp"
#include "lbann/distconv.hpp"

namespace lbann {

/// Convolution layer
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class convolution_layer : public base_convolution_layer<Dev> {
 private:

  friend class lbann_callback_imcomm;

 public:

  /// kernel tensor is output channels, input channels, conv dimension (w x h)
  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream s;
    s << " convolution; conv_dims: ";
    // for (size_t h=0; h<this->m_kernel_dims.size(); h++) {
    //   if (h == 0) { s << " channels (out x in) "; }
    //   if (h == 2) { s << " filters (w x h) "; }
    //   s << this->m_kernel_dims[h] << " ";
    // }
    s << get_topo_description();
    s << " pads: ";
    for (size_t h=0; h<this->m_pads.size(); h++) {
      s << this->m_pads[h] << " ";
    }
    s << " strides: ";
    for (size_t h=0; h<this->m_strides.size(); h++) {
      s << this->m_strides[h] << " ";
    }
    s << " num_output_channels: " << this->m_neuron_dims[0]
      << " has_bias: " << this->m_bias_scaling_factor
      << " dataLayout: " << this->get_data_layout_string(get_data_layout())
      << " device alloc: " + this->get_device_allocation_string(get_device_allocation());
    return s.str();
  }

  std::string get_topo_description() const override {
    std::stringstream s;
    // Get the topo description from any parent class
    std::string str = base_convolution_layer<Dev>::get_topo_description();
    s << str << " - ";

    // Display the topology of the kernel
    for (size_t h=0; h<this->m_kernel_dims.size(); h++) {
      if (h == 0) { s << "C="; }
      s << this->m_kernel_dims[h] ;
      if (h == 0) { s << "o,"; }
      if (h == 1) { s << "i F="; }
      if (this->m_kernel_dims.size() == 3) {
        if (h == 2) { s << "w "; }
      }else if (this->m_kernel_dims.size() == 4) {
        if (h == 2) { s << "w x "; }
        if (h == 3) { s << "h"; }
      }else {
        if (h > 1) {
          s << " ";
        }
      }
    }
    return s.str();;
  }

  convolution_layer(lbann_comm *comm,
                    int num_data_dims,
                    int num_output_channels,
                    int conv_dim,
                    int pad,
                    int stride,
                    bool has_bias = true,
                    cudnn::cudnn_manager *cudnn = nullptr)
      : convolution_layer(comm,
                          num_data_dims,
                          num_output_channels,
                          std::vector<int>(num_data_dims, conv_dim),
                          std::vector<int>(num_data_dims, pad),
                          std::vector<int>(num_data_dims, stride),
                          has_bias,
                          cudnn) {}

  convolution_layer(lbann_comm *comm,
                    int num_data_dims,
                    int num_output_channels,
                    std::vector<int> conv_dims,
                    std::vector<int> pads,
                    std::vector<int> strides,
                    bool has_bias = true,
                    cudnn::cudnn_manager *cudnn = nullptr)
      : base_convolution_layer<Dev>(comm,
                                    num_data_dims,
                                    num_output_channels,
                                    conv_dims,
                                    pads,
                                    strides,
                                    has_bias,
                                    cudnn) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "convolution only supports DATA_PARALLEL");

  }

  convolution_layer* copy() const override { return new convolution_layer(*this); }

  std::string get_type() const override { return "convolution"; }

  data_layout get_data_layout() const override { return T_layout; }

  El::Device get_device_allocation() const override { return Dev; }

  void setup_dims() override {

    // Initialize previous neuron tensor dimensions
    base_convolution_layer<Dev>::setup_dims();

    // Initialize convolution kernel dimensions
    this->m_kernel_dims.insert(this->m_kernel_dims.begin() + 1,
                               this->m_prev_neuron_dims[0]);

    // Check if previous neuron tensor dimensions are valid
#ifdef LBANN_DEBUG
    if(this->m_num_neuron_dims != (int) this->m_kernel_dims.size() - 1) {
      throw lbann_exception("convolution_layer: neuron tensor dimensions are unexpected");
    }
#endif

    // Initialize neuron tensor dimensions
    this->m_neuron_dims[0] = this->m_kernel_dims[0];
    for(int i=0; i<this->m_num_neuron_dims-1; ++i) {
      const int effective_dim = (this->m_prev_neuron_dims[i+1]
                                 + 2 * this->m_pads[i]
                                 - this->m_kernel_dims[i+2] + 1);
      this->m_neuron_dims[i+1]= ((effective_dim + this->m_strides[i] - 1)
                                 / this->m_strides[i]);
    }
    this->m_num_neurons = std::accumulate(this->m_neuron_dims.begin(),
                                          this->m_neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());

    // Get size of convolutional kernel
    this->m_kernel_size = std::accumulate(this->m_kernel_dims.begin(),
                                          this->m_kernel_dims.end(),
                                          1,
                                          std::multiplies<int>());

  }

  void setup_data() override {
    base_convolution_layer<Dev>::setup_data();
    this->m_weights[0]->setup(this->m_kernel_dims, Dev);
    El::Zeros(this->m_kernel_gradient,
              this->m_weights[0]->get_matrix_height(),
              this->m_weights[0]->get_matrix_width());
  }

 protected:

  
  void fp_compute() override {
    if(this->using_gpus()) {
#ifdef LBANN_HAS_DISTCONV
      if (m_distconv_enabled) {
        early_terminate();
        apply_convolution_distconv();
        apply_bias_distconv();
#if 1
        dump_tensor(m_activations_t,
                    get_name() + "_activations");
#endif
        // activations may be updated with bias, so its copy should
        // be done after applying bias
        if (m_child_copy_required) {
          MPIPrintStreamDebug() << "Copying back to sample decomposition\n";
          assert0(dc::tensor::View(
              m_activations_copyout, m_activations_d[0].get_data(0)));
          assert0(dc::tensor::Copy(
              m_activations_copyout, m_activations_t));
        }
        if (m_exit_count == 0) {
          apply_convolution_cudnn(true);
          apply_bias_cudnn();
          assert0(dc::tensor::View(
              m_activations_copyout, m_activations_d[0].get_data(0)));
#if 1
          dump_tensor(m_activations_copyout,
                      get_name() + "_activations_original");
#endif
        }
      } else {
        base_convolution_layer<Dev>::apply_convolution_cudnn(true);
        base_convolution_layer<Dev>::apply_bias_cudnn();
      }
#else
      base_convolution_layer<Dev>::apply_convolution_cudnn(true);
      base_convolution_layer<Dev>::apply_bias_cudnn();
#endif
    } else {
      base_convolution_layer<Dev>::apply_convolution_im2col(true);
      base_convolution_layer<Dev>::apply_bias_cpu();
    }
  }

  void bp_compute() override {
    if(this->using_gpus()) {
#ifdef LBANN_HAS_DISTCONV
      if (m_distconv_enabled) {
        m_prev_error_signals_redistributed = false;
        compute_gradients_distconv();
        apply_transposed_convolution_distconv();
        if (m_exit_count == 0) {
          dump_tensor(m_error_signals_t,
                      get_name() + "_error_signals");
          compute_gradients_cudnn(false);
          apply_transposed_convolution_cudnn(false);
          assert0(dc::tensor::View(
              m_error_signals_copyout, m_error_signals_d[0].get_data(0)));
          dump_tensor(m_error_signals_copyout,
                      get_name() + "_error_signals_original");
        }
      } else {
        base_convolution_layer<Dev>::compute_gradients_cudnn(false);
        base_convolution_layer<Dev>::apply_transposed_convolution_cudnn(false);
      }
#else
      base_convolution_layer<Dev>::compute_gradients_cudnn(false);
      base_convolution_layer<Dev>::apply_transposed_convolution_cudnn(false);
#endif
    } else {
      base_convolution_layer<Dev>::compute_gradients_im2col(false);
      base_convolution_layer<Dev>::apply_transposed_convolution_im2col(false);
    }
  }

  void setup_gpu() override {
    std::cerr << "setup gpu\n";
    base_convolution_layer::setup_gpu();
  }
  
  void apply_convolution_distconv() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception(
        std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
        "Layer: DISTCONV not detected");
#else
    MPIPrintStreamDebug() << get_name() << ": Forward convolution\n";
    // there may only be a smaller number of samples for the last
    // mini-batch iteration
    m_conv->set_num_samples(this->m_model->get_current_mini_batch_size());
    
    if (m_parent_copy_required) {
      MPIPrintStreamDebug() << "Copying from sample decomposition\n";  
      assert0(dc::tensor::View(
          m_prev_activations_const_view,
          m_prev_activations_d[0].get_locked_data(0)));
      assert0(dc::tensor::Copy(
          m_prev_activations_t, m_prev_activations_const_view));
    } else {
      MPIPrintStreamDebug()
          << "Directly reading activations of previous layer\n";
    }

    assert0(dc::tensor::View(
        m_kernel_t, m_weights[0]->get_values_gpu()[0]));

    m_conv->forward(DataType(1.0), m_prev_activations_t, m_kernel_t,
                    DataType(0.0), m_activations_t);

#endif
  }

  void apply_bias_distconv() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception(
        std::string {} + __FILE__ + " " + std::to_string(__LINE__)
        + " :: " + "Layer: DISTCONV not detected");
#else
    if (m_bias_scaling_factor == DataType(0)) return;

    MPIPrintStreamDebug() << "Applying bias\n";
    
    assert0(dc::tensor::View(
        m_bias_t, m_weights[1]->get_values_gpu()[0]));
    m_conv->apply_bias(m_bias_scaling_factor, m_bias_t,
                       DataType(1), m_activations_t);
#endif
  }

  void apply_transposed_convolution_distconv() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception(
        std::string {} + __FILE__ + " " + std::to_string(__LINE__)
        + " :: " + "Layer: DISTCONV not detected");
#else
    MPIPrintStreamDebug() << get_name() << ": Backward convolution\n";

    // input: m_prev_error_signals_d[0]
    // kernel: m_weights[0]->get_values_gpu()
    // output: m_error_signals_d[0]

    assert0(dc::tensor::View(
        m_kernel_t, m_weights[0]->get_values_gpu()[0]));

    if (m_child_copy_required) {
      if (!m_prev_error_signals_redistributed) {
        MPIPrintStreamDebug() << "Copying from sample decomposition\n";
        assert0(dc::tensor::View(
            m_prev_error_signals_const_view,
            m_prev_error_signals_d[0].get_locked_data(0)));
        assert0(dc::tensor::Copy(
            m_prev_error_signals_t, m_prev_error_signals_const_view));
        m_prev_error_signals_redistributed = true;
      }
      //dump_tensor(m_prev_error_signals_const_view,
      //"prev_error_signals_original");
    }

    //dump_tensor(m_prev_error_signals_t,
    //"prev_error_signals_spatial");

#if 0
    // The beta parameter is non-zero, so need to copy the error signals
    if (m_parent_copy_required) {
      assert0(dc::tensor::View(
          m_error_signals_copyout, m_error_signals_d[0].get_data(0)));
      assert0(dc::tensor::Copy(
          m_error_signals_t, m_error_signals_copyout));
    }
#else
    m_error_signals_t.zero();
#endif

    MPIPrintStreamDebug() << "Calling backward_data\n";
    m_conv->backward_data(DataType(1.0), m_kernel_t, m_prev_error_signals_t,
                          DataType(1.0), m_error_signals_t);


    if (m_parent_copy_required) {
      if (m_exit_count != 0) {
        MPIPrintStreamDebug() << "Copying back to sample decomposition\n";
        assert0(dc::tensor::View(
            m_error_signals_copyout, m_error_signals_d[0].get_data(0)));
        assert0(distconv::tensor::Copy(
            m_error_signals_copyout, m_error_signals_t));
      }
    }
#endif    
  }

  void compute_gradients_distconv() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception(
        std::string {} + __FILE__ + " " + std::to_string(__LINE__)
        + " :: " + "Layer: DISTCONV not detected");
#else
    MPIPrintStreamDebug() << get_name() << ": Compute gradients\n";

    if (m_child_copy_required) {
      assert0(dc::tensor::View(
          m_prev_error_signals_const_view,
          m_prev_error_signals_d[0].get_locked_data(0)));
    }

    const int effective_mini_batch_size =
        this->m_model->get_effective_mini_batch_size();    

    optimizer* bias_optimizer = m_weights[1]->get_optimizer();
    if (bias_optimizer != nullptr && m_bias_scaling_factor != DataType(0)) {
      MPIPrintStreamDebug() << "Compute bias gradients\n";      
      // Copy to sample distribution
      if (m_child_copy_required && !m_prev_error_signals_redistributed) {
        assert0(dc::tensor::Copy(
            m_prev_error_signals_t, m_prev_error_signals_const_view));
        m_prev_error_signals_redistributed = true;
      }
      assert0(dc::tensor::View(m_bias_gradient_t,
                               m_bias_gradient_d.get_data(0)));
      m_conv->backward_bias(DataType(1.0), m_prev_error_signals_t,
                            DataType(0.0), m_bias_gradient_t, false);
      const DataType bias_scale = m_bias_scaling_factor / effective_mini_batch_size;
      if (m_exit_count != 0) {
        bias_optimizer->add_to_gradient_staging(m_bias_gradient_d,
                                                bias_scale);
      }
    }

    optimizer* kernel_optimizer = m_weights[0]->get_optimizer();
    if (kernel_optimizer == nullptr) return;

    MPIPrintStreamDebug() << "Compute kernel gradients\n";          

    assert0(dc::tensor::View(
        m_kernel_gradient_e, m_kernel_gradient_d.get_data(0)));
    
    // Copy to sample distribution
    if (m_child_copy_required && !m_prev_error_signals_redistributed) {
      assert0(dc::tensor::Copy(
          m_prev_error_signals_t, m_prev_error_signals_const_view));
      m_prev_error_signals_redistributed = true;
    }

    m_conv->backward_filter(DataType(1.0), m_prev_activations_t,
                            m_prev_error_signals_t, DataType(0),
                            m_kernel_gradient_e, false);

    // Add gradient contribution
    const DataType kernel_scale = DataType(1) / effective_mini_batch_size;
    if (m_exit_count != 0) {
      kernel_optimizer->add_to_gradient_staging(m_kernel_gradient_d,
                                                kernel_scale);
    }
#endif    
  }

#ifdef LBANN_HAS_DISTCONV
 public:
  bool using_distconv() const override {
    if (!(m_kernel_dims[2] == m_kernel_dims[3] &&
          m_kernel_dims[2] == m_pads[0] * 2 + 1 &&
          m_kernel_dims[3] == m_pads[1] * 2 + 1)) {
      MPIPrintStreamDebug() << "Unsupported as padding does not match the kernel size\n";
      return false;
    }
    // This is no longer necessary
#if 0
    if (!(m_prev_neuron_dims[2] % m_strides[1] == 0 &&
          m_prev_neuron_dims[1] % m_strides[0] == 0)) {
      MPIPrintStreamDebug() << "Unsupported as tensor dimensions not devisible by strides\n";
      return false;
    }
#endif
#if 1
    char *env = getenv("DISTCONV_DISABLE");
    if (env) {
      std::string s(env);
      if (s.find(get_name()) != std::string::npos) {
        return false;
      }
    }
    //if (get_name() == "pool2" || get_name() == "conv3") {
    return true;
#endif
    //return true;
  }
  
  Array4 get_prev_activations_overlap() const override {
    if (using_distconv()) {
      int stencil_h = (m_kernel_dims[2] - 1) / 2;
      int stencil_w = (m_kernel_dims[3] - 1) / 2;
      return Array4({stencil_w, stencil_h, 0, 0});
    } else {
      return Array4(0);
    }
  }

  Array4 get_activations_overlap() const override {
    return Array4(0);
  }

  Array4 get_prev_error_signals_overlap() const override {
    if (using_distconv()) {
      return get_prev_activations_overlap();
    } else {
      return Array4(0);
    }
  }

  Array4 get_error_signals_overlap() const override {
    return Array4(0);
  }

  void setup_tensor_distribution_init(
      std::map<const Layer*, std::array<Dist, 4>> &dists,      
      std::map<Dist*, std::set<Dist*>> &invariants,
      std::set<Dist*> &updated,
      std::set<Dist*> &fixed) override {
    Layer::setup_tensor_distribution_init(
        dists, invariants, updated, fixed);
    if (using_distconv()) {
      int stencil_h = (m_kernel_dims[2] - 1) / 2;
      int stencil_w = (m_kernel_dims[3] - 1) / 2;
      Array4 overlap({stencil_w, stencil_h, 0, 0});
      auto &prev_activations_dist = dists[this][0];
      prev_activations_dist.set_overlap(overlap);
      updated.insert(&prev_activations_dist);
      fixed.insert(&prev_activations_dist);
      auto &prev_error_signals_dist = dists[this][3];      
      prev_error_signals_dist.set_overlap(overlap);
      updated.insert(&prev_error_signals_dist);
      fixed.insert(&prev_error_signals_dist);
      // To deal with strides, error signals must have the same size
      // of overlap 
      auto &error_signals_dist = dists[this][2];
      error_signals_dist.set_overlap(overlap);
      updated.insert(&error_signals_dist);
      fixed.insert(&error_signals_dist);
    }
  }

  Array4 get_strides() const override {
    return Array4({m_strides[1], m_strides[0], 1, 1});
  }

  void setup_tensors_fwd(const std::array<Dist, 4> &dists) override {    
    Layer::setup_tensors_fwd(dists);
    if (!m_distconv_enabled) return;

    std::stringstream ss;
    dc::util::print_vector(ss, m_kernel_dims.begin(), m_kernel_dims.end());
    MPIPrintStreamDebug()
        << "m_kernel_dims: " << ss.str() << "\n";

    const Array4 input_tensor_shape =
        {m_prev_neuron_dims[2], m_prev_neuron_dims[1],
         m_prev_neuron_dims[0], this->m_model->get_max_mini_batch_size()};
    const LocaleMPI loc(m_comm->get_model_comm().comm, false);
    const Array4 sample_block_size = {1, 1, 1, 1};    
    const Dist sample_dist = Dist({1, 1, 1, m_comm->get_procs_per_model()});
    Array4 input_local_shape = input_tensor_shape;
    // Assuming single GPU per rank
    input_local_shape[3] = m_max_mini_batch_size_per_gpu;
    const Array4 spatial_local_size = {0, 0, 0, 0};
    const Array4 output_tensor_shape =
        {m_neuron_dims[2], m_neuron_dims[1],
         m_neuron_dims[0], this->m_model->get_max_mini_batch_size()};
    Array4 output_local_shape = output_tensor_shape;
    output_local_shape[3] = m_max_mini_batch_size_per_gpu;
    const int filter_dims[4] = {m_kernel_dims[3], m_kernel_dims[2],
                                m_kernel_dims[1], m_kernel_dims[0]};
    const int strides[2] = {m_strides[1], m_strides[0]};
    
    if (m_parent_copy_required) {
      MPIPrintStreamDebug() << "copying prev activations required\n";      
      m_prev_activations_const_view = ConstTensorDev(input_tensor_shape, loc,
                                                     sample_dist,
                                                     input_local_shape,
                                                     sample_block_size);
      m_prev_activations_t = TensorDev(input_tensor_shape, loc, dists[0],
                                       spatial_local_size, m_input_decomposition_block);
      assert0(m_prev_activations_t.allocate());
      m_prev_activations_t.zero();      
    } else {
      MPIPrintStreamDebug() << "directly using prev activations: "
                            << get_parent_layers()[0]->get_activations_t() << "\n";
      m_prev_activations_t = get_parent_layers()[0]->get_activations_t();      
      assert_always(m_prev_activations_t.get_distribution() == dists[0]);
      assert_always(m_prev_activations_t.get_requested_local_block()
                    == m_input_decomposition_block);
    }

    const Array4 output_spatial_local_shape =
        dc::get_convolution_output_local_tensor_shape(
            m_prev_activations_t,
            filter_dims, strides, true);
    MPIPrintStreamDebug()
        << "Convolution output_spatial_local_shape: " << output_spatial_local_shape << "\n";
    m_activations_t = TensorDev(output_tensor_shape,
                                loc, dists[1], output_spatial_local_shape,
                                m_output_decomposition_block);
    assert0(m_activations_t.allocate());
    m_activations_t.zero();

    //if (m_child_copy_required) {
    m_activations_copyout = TensorDev(output_tensor_shape, loc, sample_dist,
                                      output_local_shape, sample_block_size);

    Array4 kernel_shape = {m_kernel_dims[3], m_kernel_dims[2],
                           m_kernel_dims[1], m_kernel_dims[0]};
    
    m_kernel_t = TensorDev(kernel_shape, loc, Dist());
    assert0(dc::tensor::View(
        m_kernel_t, m_weights[0]->get_values_gpu()[0]));
    m_kernel_gradient_e = TensorDev(kernel_shape, loc, Dist());
    assert0(dc::tensor::View(
        m_kernel_gradient_e, m_kernel_gradient_d.get_data(0)));
    
    m_conv = new dc::Convolution<dc::cudnn::BackendCUDNN>(
        *this->m_cudnn->get_distconv_backend());

    // Bias tensor. Shared by all procs
    MPIPrintStreamDebug()
        << "Bias desc: "
        << dc::util::tostring(m_bias_cudnn_desc)
        << ", bias factor: " << m_bias_scaling_factor
        << "\n";
    if (m_bias_scaling_factor != DataType(0)) {
      Array4 bias_shape = {1, 1, m_neuron_dims[0], 1};
      m_bias_t = TensorDev(bias_shape, loc, Dist());
      assert0(dc::tensor::View(m_bias_t, m_weights[1]->get_values_gpu()[0]));
      MPIPrintStreamDebug()
          << "Bias tensor: " << m_bias_t << "\n";
      m_conv->setup_bias(m_bias_t);

      // Bias backprop
      optimizer* bias_optimizer = m_weights[1]->get_optimizer();      
      if (bias_optimizer != nullptr) {
        m_bias_gradient_t = TensorDev(bias_shape, loc, Dist());
        assert0(dc::tensor::View(m_bias_gradient_t,
                                 m_bias_gradient_d.get_data(0)));
        m_conv->setup_bias_gradient(m_bias_gradient_t);
      }
    }
  }

  void setup_tensors_bwd(const std::array<Dist, 4> &dists) override {
    Layer::setup_tensors_bwd(dists);
    if (!m_distconv_enabled) return;    
    // REFACTORING: this is repeated again
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
      m_prev_error_signals_const_view = ConstTensorDev(output_tensor_shape, loc,
                                                       sample_dist,
                                                       output_local_shape,
                                                       sample_block_size);
      m_prev_error_signals_t = TensorDev(output_tensor_shape, loc,
                                         dists[3],
                                         m_activations_t.get_local_shape(),
                                         m_output_decomposition_block);
      assert0(m_prev_error_signals_t.allocate());
      m_prev_error_signals_t.zero();
    } else {
      m_prev_error_signals_t = get_child_layers()[0]->get_error_signals_t();
      MPIPrintStreamDebug() << get_name() << ": directly using prev error signals\n";
      assert_always(m_prev_error_signals_t.get_distribution() ==
                    dists[3]);
      assert_always(m_prev_error_signals_t.get_requested_local_block() ==
                    m_output_decomposition_block);
    }
    
    // error_signals
    m_error_signals_t = TensorDev(input_tensor_shape, loc,
                                  dists[2], m_prev_activations_t.get_local_shape(),
                                  m_input_decomposition_block);
    assert0(m_error_signals_t.allocate());
    m_error_signals_t.zero();

    //if (m_parent_copy_required) {
    m_error_signals_copyout = TensorDev(input_tensor_shape, loc, sample_dist,
                                        input_local_shape, sample_block_size);

    if (getenv("DISTCONV_DETERMINISTIC")) {
      // Same algorithm as LBANN
      m_fwd_algo = "IMPLICIT_GEMM";
      // Deterministic algorithm
      m_bwd_data_algo = "ALGO1";
      m_bwd_filter_algo = "ALGO1";
    }
    
    m_conv->setup(m_prev_activations_t,
                  m_kernel_t, m_activations_t,
                  m_error_signals_t, m_kernel_gradient_e,
                  m_prev_error_signals_t,
                  m_pads[0], m_pads[1],
                  m_strides[0], m_strides[1],
                  m_fwd_algo, m_bwd_data_algo,
                  m_bwd_filter_algo);
  }
  
 protected:

  dc::Convolution<dc::cudnn::BackendCUDNN> *m_conv;
  TensorDev m_kernel_t;
  TensorDev m_kernel_gradient_e;
  // Bias
  TensorDev m_bias_t;
  TensorDev m_bias_gradient_t;
  // Algorithms
  std::string m_fwd_algo = "DEFAULT";
  std::string m_bwd_data_algo = "DEFAULT";
  std::string m_bwd_filter_algo = "DEFAULT";

  bool m_prev_error_signals_redistributed = false;

  // For debugging

#endif // LBANN_HAS_DISTCONV
  

};

} // namespace lbann

#endif // LBANN_LAYER_CONVOLUTION_HPP_INCLUDED
