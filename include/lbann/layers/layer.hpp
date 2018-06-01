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

#ifndef LBANN_LAYER_HPP_INCLUDED
#define LBANN_LAYER_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/distconv.hpp"
#include "lbann/utils/summary.hpp"
#include "lbann/optimizers/optimizer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/distconv.hpp"
#include <lbann.pb.h>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <array>

namespace lbann {

// Forward declaration
class model;

/** Represents a parallel strategy for a layer. */
struct ParallelStrategy {
  /** Number of groups the sample dimension is split over. */
  int sample_groups = 0;
  /** Number of groups the height dimension is split over. */
  int height_groups = 0;
  /** Number of groups the width dimension is split over. */
  int width_groups = 0;
  /** Number of groups the channel dimension is split over. */
  int channel_groups = 0;
  /** Number of groups the filter dimension is split over. */
  int filter_groups = 0;
  /** Number of times the layer is replicated (for FC layers right now). */
  int replications = 0;
  bool operator==(const ParallelStrategy &ps) const {
    return sample_groups == ps.sample_groups &&
        height_groups == ps.height_groups &&
        width_groups == ps.width_groups &&
        channel_groups == ps.channel_groups &&
        filter_groups == ps.filter_groups &&
        replications == ps.replications;
  }
  bool operator!=(const ParallelStrategy &ps) const {
    return !(*this == ps);
  }
};

inline std::ostream &operator<<(std::ostream &os,
                                const ParallelStrategy &ps) {
  os << "{" << ps.sample_groups
     << ", " << ps.height_groups
     << ", " << ps.width_groups
     << ", " << ps.channel_groups
     << ", " << ps.filter_groups
     << ", " << ps.replications
     << "}";
  return os;
}

/** Abstract base class for neural network layers.
 *  A layer takes input tensors ("previous activations") and applies a
 *  mathematical operation to obtain output tensors
 *  ("activations"). This operation often has trainable parameters
 *  called "weights." The previous activations are recieved from
 *  "parent layers" and the activations are sent to "child layers,"
 *  making each layer a node in a directed graph. The layer graph and
 *  the weights are managed by a neural network model class. A layer
 *  should also be able to take objective function gradients
 *  w.r.t. the activations ("previous error signals") and compute the
 *  objective function gradients w.r.t. the previous activations
 *  ("error signals") and w.r.t. the weights. This allows the model to
 *  perform automatic differentiation and to apply first-order
 *  optimization methods to the weights.
 */
class Layer {
 public:
  Layer(lbann_comm *comm);
  Layer(const Layer& other);
  Layer& operator=(const Layer& other);
  virtual ~Layer();

  /** Copy function.
   *  This function dynamically allocates memory for a layer instance
   *  and instantiates a copy. The caller is responsible for
   *  deallocating the instance.
   */
  virtual Layer* copy() const = 0;

  /** Get the layer type's name.
   *  A layer type name should be brief, human-readable description of
   *  the layer's mathematical operation.
   */
  virtual std::string get_type() const = 0;
  /** Get the layer instance's name.
   *  Each layer in a model should have a unique, preferably
   *  human-readable, name.
   */
  inline std::string get_name() const { return m_name; }
  /** Set the layer instance's name.
   *  Each layer in a model should have a unique, preferably
   *  human-readable, name.
   */
  inline void set_name(const std::string name) { m_name = name; }

  /** Get a human-readable description of the layer parameters. */
  virtual std::string get_description() const;
  /** Get a human-readable description of the activation tensors.
   *  Activation tensors are stored in distributed matrices where each
   *  column corresponds to a mini-batch sample. Within each column,
   *  the data is packed w.r.t. the last tensor dimension, then
   *  w.r.t. the penultimate dimension, and so on. 3D tensors are
   *  assumed to be 2D images in NCHW format.
   */
  virtual std::string get_topo_description() const;

  /** Forward propagation step.
   *  Apply a mathematical operation to the previous activations to
   *  obtain the activations.
   */
  virtual void forward_prop();
  /** Backward propagation step.
   *  Given the objective function gradients w.r.t. the activations
   *  (the previous error signals), compute the gradients w.r.t. the
   *  previous activations (the error signals) and w.r.t. the
   *  weights. This is essentially an application of the chain
   *  rule. Note that the objective function may have terms that are
   *  independent of the activations, so we add to the gradients
   *  rather than overwriting them. This means the error signals and
   *  weight gradients must be cleared before performing backward
   *  propagation (see the clear_error_signals function).
   */
  virtual void back_prop();
  /** Update step.
   *  Update the layer's internal members. Note that the optimization
   *  step for the weights happens elsewhere.
   */
  virtual bool update();

  /** Set the error signal tensors to zero.
   *  The error signals are resized for the current mini-batch size.
   */
  virtual void clear_error_signals(int mini_batch_size);

  virtual void summarize_stats(lbann_summary& summarizer, int step);
  virtual void summarize_matrices(lbann_summary& summarizer, int step);

  /** Setup layer members.
   *  By default, this calls the setup_pointers, setup_dims,
   *  setup_matrices, setup_data, and setup_gpu (if needed)
   *  functions. Unless the setup_pointers function has been replaced
   *  in an inherited class, it is assumed that pointers to
   *  parent/child layers have already been initialized.
   *
   *  If the layer has already been setup, this function should
   *  destroy all layer members and reinitialize them. However, it is
   *  not guaranteed that derived classes will obey this
   *  behavior. Caveat emptor.
   */
  virtual void setup();
  /** Check that the setup is reasonable. */
  virtual void check_setup();

  /** Get data layout of the data tensors.
   *  We assume that the data layouts of the previous activations,
   *  activations, previous error signals, and error signals are the
   *  same. Each concrete layer that is templated on its data layout
   *  should override this function to return its template parameter.
   */
  virtual data_layout get_data_layout() const = 0;
  /** Get the device allocation for the data tensors.
   *  We assume that the decice allocation of the previous activations,
   *  activations, previous error signals, and error signals are the
   *  same. Each concrete layer that is templated on its device allocation
   *  should override this function to return its template parameter.
   */
  virtual El::Device get_device_allocation() const = 0;
  /** Get a human-readable description of the data_layout */
  std::string get_data_layout_string(data_layout d) const;
  /** Get a human-readable description of the device allocation */
  std::string get_device_allocation_string(El::Device dev) const;
  /** Get a short human-readable description of the device allocation */
  std::string get_device_allocation_string_short(El::Device dev) const;

  /** Get the dimensions of a previous activations tensor. */
  virtual std::vector<int> get_prev_neuron_dims(int parent_index = 0) const {
    return m_prev_neuron_dims;
  }
  /** Get the size of a previous activations tensor. */
  virtual int get_num_prev_neurons(int parent_index = 0) const {
    return m_num_prev_neurons;
  }
  /** Get the number of dimensions of a previous activations tensor. */
  virtual int get_num_prev_neuron_dims(int parent_index = 0) const {
    return m_num_prev_neuron_dims;
  }
  /** Get the dimensions of an activations tensor. */
  virtual std::vector<int> get_neuron_dims(int child_index = 0) const {
    return m_neuron_dims;
  }
  /** Get the size of an activations tensor. */
  virtual int get_num_neurons(int child_index = 0) const {
    return m_num_neurons;
  }
  /** Get the number of dimensions of an activations tensor. */
  virtual int get_num_neuron_dims(int child_index = 0) const {
    return m_num_neuron_dims;
  }

  /** Reset layer stat counters. */
  virtual void reset_counters();

  /** Whether the layer is using a GPU implementation. */
  inline bool using_gpus() const { return m_using_gpus; }

  /** To make sure copying between host and devices is complete */
  void synchronize() const;

  /** Get expected number of parent layers.
   *  A negative value indicates no limit.
   */
  inline int get_expected_num_parent_layers() const { return m_expected_num_parent_layers; }
  /** Get expected number of child layers.
   *  A negative value indicates no limit.
   */
  inline int get_expected_num_child_layers() const { return m_expected_num_child_layers; }

  /** Return the model that manages this layer. */
  inline model* get_model() const { return m_model; }
  /** Set the model that manages this layer. */
  inline void set_model(model* m) { m_model = m; }

  virtual El::Matrix<El::Int>* get_sample_indices_per_mb() { return nullptr; };

  virtual bool save_to_checkpoint_shared(persist& p) const;
  virtual bool load_from_checkpoint_shared(persist& p);

  virtual bool save_to_checkpoint_distributed(persist& p) const;
  virtual bool load_from_checkpoint_distributed(persist& p);

  /** Write layer to proto file */
  virtual void write_proto(lbann_data::Layer* proto) const;

  /** Send forward propagation output to a child layer.
   *  On output, fp_output is either a matrix view or copy of the
   *  appropriate activation tensor.
   */
  virtual void get_fp_output(AbsDistMat& fp_output, const Layer* child) const;
  /** Send backward propagation output to a parent layer.
   *  On output, bp_output is either a matrix view or copy of the
   *  appropriate error signal tensor.
   */
  virtual void get_bp_output(AbsDistMat& bp_output, const Layer* parent) const;

  /** Get dimensions of forward propagation output to a child layer.
   *  Returns the dimensions of the appropriate activations tensor.
   */
  virtual std::vector<int> fp_output_dims(const Layer* child = nullptr) const { return m_neuron_dims; }

  /** Add to the layer's error signal. */
  virtual void add_to_error_signal(const AbsDistMat& error_signals,
                                   DataType scale = DataType(1),
                                   int parent_index = 0) {
    El::Axpy(scale, error_signals, *m_error_signals[parent_index]);
  }

  /** Get parent layers. */
  inline std::vector<const Layer*>& get_parent_layers() { return m_parent_layers; }
  /** Get parent layers. (const) */
  inline const std::vector<const Layer*>& get_parent_layers() const { return m_parent_layers; }
  /** Get child layers. */
  inline std::vector<const Layer*>& get_child_layers() { return m_child_layers; }
  /** Get child layers. (const) */
  inline const std::vector<const Layer*>& get_child_layers() const { return m_child_layers; }

  /** Get number of parent layers. */
  inline int get_num_parents() const { return get_parent_layers().size(); }
  /** Get number of child layers. */
  inline int get_num_children() const { return get_child_layers().size(); }

  /** Get names in a particular list of layers */
  static std::string get_layer_names(const std::vector<const Layer*>& list);
  std::string get_child_names() const { return get_layer_names(m_child_layers); }
  std::string get_parent_names() const { return get_layer_names(m_parent_layers); }

  /** Add a parent layer.
   *  Does nothing if parent is a null pointer, the same layer, or
   *  already a parent.
   */
  void add_parent_layer(const Layer* parent);
  /** Add a child layer.
   *  Does nothing if child is a null pointer, the same layer, or
   *  already a child.
   */
  void add_child_layer(const Layer* child);

  /** Remove all parent layers.
   *  Parent layers are not deallocated.
   */
  void clear_parent_layers() { get_parent_layers().clear(); }
  /** Remove all child layers.
   *  Child layers are not deallocated.
   */
  void clear_child_layers() { get_child_layers().clear(); }

  /** Get list of pointers to other layers. */
  virtual std::vector<Layer*> get_layer_pointers();
  /** Set list of pointers to other layers. */
  virtual void set_layer_pointers(std::vector<Layer*> layers);

  /** Get references to weights. */
  inline std::vector<weights*>& get_weights() { return m_weights; }
  /** Get references to weights. (const) */
  inline const std::vector<weights*>& get_weights() const { return m_weights; }
  /** Set list of pointers to weights. */
  inline void set_weights(std::vector<weights*> w) { get_weights() = w; }
  /** Replace weights with another Layer's weights*/
  void replace_weights(Layer* other_layer);

  /** Get previous activation tensor. */
  AbsDistMat& get_prev_activations(int parent_index = 0);
  /** Get activation tensor. */
  AbsDistMat& get_activations(int child_index = 0);
  /** Get previous error signal tensor. */
  AbsDistMat& get_prev_error_signals(int child_index = 0);
  /** Get error signal tensor. */
  AbsDistMat& get_error_signals(int parent_index = 0);
  /** Get previous activation tensor. (const) */
  const AbsDistMat& get_prev_activations(int parent_index = 0) const;
  /** Get activation tensor. (const) */
  const AbsDistMat& get_activations(int child_index = 0) const;
  /** Get previous error signal tensor. (const) */
  const AbsDistMat& get_prev_error_signals(int child_index = 0) const;
  /** Get error signal tensor. (const) */
  const AbsDistMat& get_error_signals(int parent_index = 0) const;
  /** Get local portion of previous activation tensor. */
  AbsMat& get_local_prev_activations(int parent_index = 0);
  /** Get local portion of activation tensor. */
  AbsMat& get_local_activations(int child_index = 0);
  /** Get local portion of previous error signal tensor. */
  AbsMat& get_local_prev_error_signals(int child_index = 0);
  /** Get local portion of error signal tensor. */
  AbsMat& get_local_error_signals(int parent_index = 0);
  /** Get local portion of previous activation tensor. (const) */
  const AbsMat& get_local_prev_activations(int parent_index = 0) const;
  /** Get local portion of activation tensor. (const) */
  const AbsMat& get_local_activations(int child_index = 0) const;
  /** Get local portion of previous error signal tensor. (const) */
  const AbsMat& get_local_prev_error_signals(int child_index = 0) const;
  /** Get local portion of error signal tensor. (const) */
  const AbsMat& get_local_error_signals(int parent_index = 0) const;

  /** Get reference to LBANN communicator. */
  lbann_comm* get_comm() const { return m_comm; }
  /** Get reference to cuDNN manager. */
  cudnn::cudnn_manager* get_cudnn_manager() const { return m_cudnn; }

  void freeze();
  void unfreeze();
  bool is_frozen() const;

  /** Get the parallel strategy for the layer. */
  ParallelStrategy& get_parallel_strategy() { return m_parallel_strategy; }
  const ParallelStrategy& get_parallel_strategy() const { return m_parallel_strategy; }

 protected:

  /** Reference to LBANN communicator. */
  lbann_comm *m_comm;

  /** Dimensions of activation tensor.
   *  If a derived class has more than one activation tensor, it is
   *  responsible for its own interpretation.
   */
  std::vector<int> m_neuron_dims;
  /** Size of activation tensor. */
  int m_num_neurons;
  /** Number of dimensions of activation tensor. */
  int m_num_neuron_dims;
  /** Dimensions of previous activation tensor.
   *  If a derived class has more than one previous activation tensor,
   *  it is responsible for its own interpretation.
   */
  std::vector<int> m_prev_neuron_dims;
  /** Size of previous activation tensor. */
  int m_num_prev_neurons;
  /** Number of dimensions of previous activation tensor. */
  int m_num_prev_neuron_dims;

  /** Previous activation matrices.
   *  Forward propagation inputs from each parent layer. These are
   *  typically matrix views where each column is a flattened tensor
   *  corresponding to a mini-batch sample. The matrices are owned by
   *  the layer.
   */
  std::vector<AbsDistMat*> m_prev_activations;
  /** Activation matrices.
   *  Forward propagation outputs to each child layer. These are
   *  typically matrices where each column is a flattened tensor
   *  corresponding to a mini-batch sample. The matrices are owned by
   *  the layer.
   */
  std::vector<AbsDistMat*> m_activations;
  /** Error signal matrices.
   *  Backward propagation inputs from each child layer. These are
   *  typically matrix views where each column is a flattened tensor
   *  corresponding to a mini-batch sample. The matrices are owned by
   *  the layer.
   */
  std::vector<AbsDistMat*> m_prev_error_signals;
  /** Error signal matrices.
   *  Backward propagation outputs to each parent layer. These are
   *  typically matrices where each column is a flattened tensor
   *  corresponding to a mini-batch sample. The matrices are owned by
   *  the layer.
   */
  std::vector<AbsDistMat*> m_error_signals;

  /** References to layer weights. */
  std::vector<weights*> m_weights;

  /** References to parent layers. */
  std::vector<const Layer*> m_parent_layers;
  /** References to child layers. */
  std::vector<const Layer*> m_child_layers;

  /** Expected number of parent layers.
   *  A negative value indicates no limit.
   */
  int m_expected_num_parent_layers = 1;
  /** Expected number of child layers.
   *  A negative value indicates no limit.
   */
  int m_expected_num_child_layers = 1;

  /** Reference to model managing this layer. */
  model *m_model = nullptr;

  /** Setup data for forward propagation.
   *  Base method gets previous activations from parent layers and
   *  resizes activations for the current mini-batch size.
   */
  virtual void fp_setup_data(int mini_batch_size);
  /** Setup data for forward propagation.
   *  Base method gets previous error signals from child layers. The
   *  error signals are resized for the current mini-batch size in the
   *  clear_error_signals function.
   */
  virtual void bp_setup_data(int mini_batch_size);
#ifdef LBANN_HAS_CUDNN
  /** Pin host memory if needed for GPU memory transfers. */
  virtual void pin_data();
#endif // LBANN_HAS_CUDNN

  /** Setup pointers to parent and child layers.
   *  Called by the setup function. The base method checks that the
   *  number of parents and children are valid. Pointers to the
   *  parent/child layers are assumed to be already initialized.
   */
  virtual void setup_pointers();
  /** Setup tensor dimensions
   *  Called by the setup function. The base method sets the
   *  dimensions of the activation tensors equal to the dimensions of
   *  the first previous activation tensor.
   */
  virtual void setup_dims();
  /** Instantiate distributed matrices.
   *  If the layer has already been setup, this function should
   *  destroy all matrices and reinstantiate them. However, it is not
   *  guaranteed that derived classes will obey this behavior.
   */
  virtual void setup_matrices(const El::Grid& grid);
  /** Setup layer data.
   *  Called by the setup function. The base method sets the previous
   *  activation, activation, previous error signal, and error signal
   *  matrices to zero matrices with the proper dimensions. Matrix
   *  buffers are pinned if needed for GPU transfers.
   */
  virtual void setup_data();
  /** Setup GPU objects.
   *  Called by the setup function if GPUs are enabled. The base
   *  method initializes GPU matrices for the previous activations,
   *  activations, previous error signals, and error signals. It also
   *  initializes cuDNN tensor descriptors.
   */
  virtual void setup_gpu();

  /** Perform the computation for the forward propagation step. */
  virtual void fp_compute() = 0;
  /** Perform the computation for the backward propagation step. */
  virtual void bp_compute() = 0;
  /** Perform the computation for the update step.
   *  Returns false if the layer must reset for a new training epoch.
   */
  virtual bool update_compute() { return true; }

  /** Reference to cuDNN manager. */
  cudnn::cudnn_manager *m_cudnn;

  /** Avoid back prop if frozen */
  bool m_frozen;

#ifdef LBANN_HAS_CUDNN

  /** cuDNN descriptor for first previous activation tensor. */
  cudnnTensorDescriptor_t m_prev_activations_cudnn_desc;
  /** cuDNN descriptor for first activations tensor. */
  cudnnTensorDescriptor_t m_activations_cudnn_desc;
  /** cuDNN descriptor for first previous error signal tensor. */
  cudnnTensorDescriptor_t m_prev_error_signals_cudnn_desc;
  /** cuDNN descriptor for first error signal tensor. */
  cudnnTensorDescriptor_t m_error_signals_cudnn_desc;

#endif // LBANN_HAS_CUDNN

  /** Time spent in forward propagation. */
  EvalType m_fp_time;
  /** Time spent in the forward propagation computation. */
  EvalType m_fp_compute_time;
  /** Time spent in backward propagation. */
  EvalType m_bp_time;
  /** Time spent in the backward propagation computation. */
  EvalType m_bp_compute_time;
  /** Time spent in updates. */
  EvalType m_update_time;

  /** Layer instance's name.
   *  Each layer in a model should have a unique, preferably
   *  human-readable, name.
   */
  std::string m_name;


  /** Parallel strategy for the layer. */
  ParallelStrategy m_parallel_strategy;

#ifdef LBANN_HAS_DISTCONV
 public:
  virtual void setup_distconv();
  virtual void setup_tensor_distribution_init(
      std::map<const Layer*, std::array<Dist, 4>> &dists, 
      std::map<Dist*, std::set<Dist*>> &invariants,
      std::set<Dist*> &updated,
      std::set<Dist*> &fixed);
  virtual void setup_tensor_distribution_add_adjacent_invariants(
      std::map<const Layer*, std::array<Dist, 4>> &dists,
      std::map<Dist*, std::set<Dist*>> &invariants);
  virtual void setup_tensor_distribution_block();
  // TODO: use dists
  virtual void setup_tensors_fwd(const std::array<Dist, 4> &dists);
  virtual void setup_prev_activations_tensor(const std::array<Dist, 4> &dists);
  virtual Array4 get_activations_tensor_local_shape() const;
  virtual void setup_activations_tensor(const std::array<Dist, 4> &dists);
  virtual void setup_activations_copyout_tensor(const std::array<Dist, 4> &dists);  
  virtual void setup_tensors_bwd(const std::array<Dist, 4> &dists);
  virtual void setup_prev_error_signals_tensor(const std::array<Dist, 4> &dists);
  virtual void setup_error_signals_tensor(const std::array<Dist, 4> &dists);
  virtual void setup_error_signals_copyout_tensor(const std::array<Dist, 4> &dists);
  virtual Array4 get_prev_activations_overlap() const;
  virtual Array4 get_activations_overlap() const;  
  virtual Array4 get_prev_error_signals_overlap() const;
  virtual Array4 get_error_signals_overlap() const;
  virtual Array4 get_input_decomposition_block() const;
  virtual Array4 get_output_decomposition_block() const;
#if 0
  virtual const Dist &get_prev_activations_distribution() const {
    return m_prev_activations_dist;
  }
  virtual const Dist &get_activations_distribution() const {
    return m_activations_dist;
  }
  virtual const Dist &get_prev_error_signals_distribution() const {
    return m_prev_error_signals_dist;
  }
  virtual const Dist &get_error_signals_distribution() const {
    return m_error_signals_dist;
  }
#endif
  
  // REFACTORING: returning non-const tensor should be protected
  virtual const TensorDev &get_activations_t() const;
  virtual const TensorDev &get_error_signals_t() const;  
  //virtual ConstTensorDev get_activations_const_view() const;
  //virtual ConstTensorDev get_prev_activations_const_view() const;

  bool distconv_enabled() const {
    return m_distconv_enabled;
  }
  void disable_distconv() {
    m_distconv_enabled = false;
  }
  
 protected:
  virtual bool using_distconv() const { return false; }
  virtual void fp_setup_distconv(int mini_batch_size);
  virtual void bp_setup_distconv(int mini_batch_size);  
  
  virtual Array4 get_strides() const;

  // Copis and converts input or output tensors when necessary
  void ensure_prev_activations();
  void copy_out_activations();
  void ensure_prev_error_signals();
  void copy_out_error_signals();

  template <typename Tensor>
  void dump_tensor(const Tensor &t, const std::string &path) {
    if (getenv("DISTCONV_DUMP")) {
      if (m_exit_count == 0) {
        MPIPrintStreamDebug() << "Dumping tensor to " << path << "\n";
        cudaDeviceSynchronize();
        dc::dump_tensor(t, path, true);
      }
    }
  }

  // negative value disables early termination. DISTCONV_EARLY_TERMINATE
  // environment value will override if set.
  int m_exit_count = -1;
  void early_terminate();
  
  bool m_distconv_enabled = false;
  bool m_parent_copy_in_required = false;
  bool m_parent_shuffle_required = false;
  bool m_child_copy_out_required = false;
  bool m_child_shuffle_required = false;  
  Array4 m_input_decomposition_block;
  Array4 m_output_decomposition_block;  
  /** Previous activation tensor */
  // Created once, view initialized at fp_setup_data
  TensorDev m_prev_activations_t;
  /** View to Elemental matrix of previous activations */
  // Created once, copied from m_prev_activations_t at fp_setup_data
  TensorDev m_prev_activations_const_view;
  /** Activation tensor */
  // Created once, copied back to m_activations_e after fp_compute
  TensorDev m_activations_t;
  /** Elemental-format activation matrix */  
  TensorDev m_activations_copyout;
  TensorShuffler *m_prev_activations_shuffler = nullptr;
  TensorShuffler *m_prev_activations_shuffler_last_mb[3];
  TensorShuffler *m_activations_shuffler = nullptr;
  TensorShuffler *m_activations_shuffler_last_mb[3];
  /** Previous error signal tensor */
  TensorDev m_prev_error_signals_t;
  /** View to Elemental matrix */
  TensorDev m_prev_error_signals_const_view;
  /** Error signal tensor */
  TensorDev m_error_signals_t;
  /** Elemental-format matrix */
  TensorDev m_error_signals_copyout;
  TensorShuffler *m_prev_error_signals_shuffler = nullptr;
  TensorShuffler *m_prev_error_signals_shuffler_last_mb[3];
  TensorShuffler *m_error_signals_shuffler = nullptr;
  TensorShuffler *m_error_signals_shuffler_last_mb[3];
#endif // LBANN_HAS_DISTCONV

 private:

  /** Whether current layer is using a GPU implementation. */
  bool m_using_gpus;

  /** Instantiate distributed matrices. */
  template <data_layout T, El::Device Dev>
  void instantiate_matrices(const El::Grid& grid);

  /** Deallocate distributed matrices. */
  void deallocate_matrices();

};

} // namespace lbann

#endif // LBANN_LAYER_HPP_INCLUDED
