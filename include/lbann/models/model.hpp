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
//
// lbann_model .hpp .cpp - Abstract class for neural network training models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_MODEL_HPP
#define LBANN_MODEL_HPP

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/utils/summary.hpp"
#include "lbann/utils/graph.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/objective_functions/objective_function.hpp"
#include "lbann/metrics/metric.hpp"
#include "lbann/weights/weights.hpp"
#include "lbann/optimizers/optimizer.hpp"
#include "lbann/utils/threads/thread_pool.hpp"
#include <lbann.pb.h>
#include <vector>
#include <string>
#include <unordered_map>

#ifdef LBANN_HAS_DISTCONV
#include "lbann/utils/cuda.hpp"
#endif

namespace lbann {

// Forward-declare this.
class lbann_callback;

/** Base class for LBANN models. */
class model {
public:

  /** Constructor. */
  model(lbann_comm *comm,
        int mini_batch_size,
        objective_function *obj_fn,
        optimizer* default_optimizer = nullptr);

  /** Copy constructor. */
  model(const model& other);
  /** Copy assignment operator. */
  model& operator=(const model& other);
  /** Destructor. */
  virtual ~model();
  /** Copy model. */
  virtual model* copy() const = 0;

  /** Return the model's type. */
  virtual std::string get_type() const = 0;

  /** Set the model's name; this is an arbitrary string
   *  that may be useful in multi-model scenarios, e.g,
   *  LTFB, jag
   */
  void set_name(std::string name);

  /** Return the model's name; this is an arbitrary string
   *  that may be useful in multi-model scenarios, e.g,
   *  LTFB, jag
   */
  std::string get_name() const {
    return m_name;
  }

  /** Human-readable description. */
  virtual description get_description() const;

  /** Set up the model. */
  virtual void setup(std::shared_ptr<thread_pool> io_thread_pool);

  /** Add layer to model. */
  virtual void add_layer(Layer *layer);

  /** Add weights to model. */
  void add_weights(weights *w);

  /** Register a new callback for the model. */
  void add_callback(lbann_callback *cb);

  /** Get the list of callbacks for the model. */
  virtual std::vector<lbann_callback*>& get_callbacks() {
    return m_callbacks;
  }

  /** Register a new metric for the model. */
  void add_metric(metric *m);

  /** Construct an instance of the default optimizer.
   *  If there is no default optimizer, a null pointer is returned.
   */
  optimizer* create_optimizer() const;

  /** Return the model's objective function. */
  objective_function* get_objective_function() const {
    return m_objective_function;
  }

  /** Return the model's metrics. */
  virtual const std::vector<metric *>& get_metrics() const {
    return m_metrics;
  }

  /** Set the model's layers. */
  void set_layers(std::vector<Layer *>& layers);

  /** Return the model's layers. */
  virtual const std::vector<Layer *>& get_layers() const { return m_layers; }

  const std::vector<weights*> get_weights() const;

  std::vector<weights*> get_weights();

  /** Replace the model's weights. */
  void replace_weights(std::vector<weights *>& w);

  /** Copy trained weights from input parameter w.
 *  Only weight values are placed, pointers and layer structure are in place.
 *  Weights to be copied are of the same name */
  void copy_trained_weights_from(std::vector<weights *>& w);

  /** Return the I/O thread pool */
  std::shared_ptr<thread_pool> get_io_thread_pool() { return m_io_thread_pool; }

  /** Get the model's comm. */
  inline lbann_comm *get_comm() const {
    return m_comm;
  }

  /** Get the current epoch for the model. */
  inline int get_cur_epoch() const {
    return m_current_epoch;
  }
  /** Get the current step for the model. */
  inline int get_cur_step() const {
    return m_current_step;  /// @todo This should be renamed to get_cur_training step and replaced with one that returns the current based on execution mode
  }

  /** Get the current validation step for the model. */
  inline int get_cur_validation_step() const {
    return m_current_validation_step;
  }
  /** Get the current testing step for the model. */
  inline int get_cur_testing_step() const {
    return m_current_testing_step;
  }
  /** Set the model (and all layers') execution mode. */
  inline void set_execution_mode(execution_mode mode) {
    m_execution_mode = mode;
  }
  /** Get the model's execution mode. */
  inline execution_mode get_execution_mode() const {
    return m_execution_mode;
  }
  /** Set the model's current mini-batch size. */
  inline void set_current_mini_batch_size(int mini_batch_size) {
    m_current_mini_batch_size = mini_batch_size;
  }
  /** Get the model's current mini-batch size. */
  inline int get_current_mini_batch_size() const {
    return m_current_mini_batch_size;
  }
  /** Get the model's maximum mini-batch size. */
  inline int get_max_mini_batch_size() const {
    return m_max_mini_batch_size;
  }
  /** Get the model's effective mini-batch size. */
  inline int get_effective_mini_batch_size() const {
    return m_effective_mini_batch_size;
  }
  /** Set the model's effective mini-batch size. */
  inline void set_effective_mini_batch_size(int mini_batch_size) {
    m_effective_mini_batch_size = mini_batch_size;
  }
  int get_num_iterations_per_epoch(execution_mode mode) const;

  /** Get the current phase (multiple epochs) in layer-wise model training. */
  inline int get_current_phase() const {
    return m_current_phase;
  }

  /**
   * Summarize statistics (e.g. timers, counters); these should be computable
   * quickly.
   */
  virtual void summarize_stats(lbann_summary& summarizer);
  /**
   * Summarize matrices (e.g. means); these are called less frequently and can
   * be more expensive.
   */
  virtual void summarize_matrices(lbann_summary& summarizer);

  /** Return true if the flag to stop training is set. */
  bool get_terminate_training() const {
    return m_terminate_training;
  }
  /** Set the terminate training flag (on or off). */
  void set_terminate_training(bool f) {
    m_terminate_training = f;
  }

  /** Train model. */
  virtual void train(int num_epochs, int num_batches=0);
  /** Evaluate model. */
  virtual void evaluate(execution_mode mode, int num_batches=0);

  /** Run one epoch using only the input layer; this supports
   *  data_store functionality
   */
  void collect_indices(execution_mode mode);

  /** Complete any background I/O data fetch for the execution
      mode requested */
  virtual void collect_background_data_fetch(execution_mode mode);

  /** Set a flag that can be used to enable / disable the background I/O activities */
  void allow_background_io_activity(bool enable) { m_background_io_allowed = enable; }

  /** Are background I/O activities enabled by the input layers */
  bool background_io_activity_allowed() { return m_background_io_allowed; }

  /** Checkpoint model to given file descriptor, return number of bytes written */
  virtual bool save_to_checkpoint_shared(persist& p);
  /** Restore model by reading checkpoint from given file descriptor, return number of bytes read */
  virtual bool load_from_checkpoint_shared(persist& p);

  virtual bool save_to_checkpoint_distributed(persist& p);
  virtual bool load_from_checkpoint_distributed(persist& p);

  /** Save the model's weight to file */
  virtual bool save_weights(persist& p);

  /** Reload the model's weights from a file */
  virtual bool reload_weights(const std::string latest,
                              const std::vector<std::string>& weight_list);

  /** Saves the model explicitly if the save_model callback is present */
  virtual bool save_model();

  /** Write model to proto file */
  virtual void write_proto(lbann_data::Model* proto);

protected:

  /** The objective function used to train the model. */
  objective_function *m_objective_function;
  /** Give model a name. */
  std::string m_name;
  /** The model's current execution mode. */
  execution_mode m_execution_mode;
  /** Flag telling the model to terminate training. */
  bool m_terminate_training;
  /** Most recent/current epoch for the model. */
  int m_current_epoch;
  /** Most recent/current training step for the model. */
  int m_current_step;
  int m_current_validation_step;
  int m_current_testing_step;
  /**
   * Maximum possible minibatch size supported by layers in this model.
   * Note that this is local to the particular model, not across multiple
   * models.
   */
  int m_max_mini_batch_size;
  /** Size of the current mini-batch in the model. */
  int m_current_mini_batch_size;
  /**
   * The "effective" size of a minibatch.
   * This is the size of the minibatch across all models and used for e.g.
   * correctly averaging gradients from multiple models.
   */
  int m_effective_mini_batch_size;
  /** current phase (multiple of epoch counts) in training a model */
  int m_current_phase;
  /** Communicator for the model. */
  lbann_comm *m_comm;
  /** Current callbacks to process. */
  std::vector<lbann_callback *> m_callbacks;

  /** Default optimizer.
   *  If a layer needs to construct an optimizer during setup, it will
   *  make a copy of the default optimizer.
   */
  optimizer *m_default_optimizer;

  /** List of model metrics.
   *  A metric can be used to evaluate the performance of the model
   *  without affecting the training process.
   */
  std::vector<metric *> m_metrics;

  /** List of layers in model.
   *  The list is in execution order for forward propagation.
   */
  std::vector<Layer *> m_layers;
  /** List of weights in model. */
  std::vector<weights *> m_weights;

  /** Threads available for I/O */
  std::shared_ptr<thread_pool> m_io_thread_pool;

  /** Flag that allows input layers to fetch data in the background */
  bool m_background_io_allowed;

  /** Check if the model execution mode is valid. */
  virtual bool is_execution_mode_valid(execution_mode mode) const;

  /** Reorder layers. */
  virtual void permute_layers(const std::vector<int>& permutation);

  /** Remap pointers.
   *  Layer and weights pointers are remapped using the provided
   *  maps. If a pointer is not a key in the corresponding map, the
   *  pointer is not changed.
   */
  virtual void remap_pointers(const std::unordered_map<Layer *,Layer *>& layer_map,
                              const std::unordered_map<weights *,weights *>& weights_map);

  /** In case that a layer is frozen, also freeze layers that precede it if that
   *  makes senses for the particular model, such as sequential or siamese.
   *  For othe models, users can manually control the behaivor by indicating
   *  whether to freeze each layer in the model description prototext.
   */
  virtual void freeze_layers_under_frozen_surface();

  /** Set up topology of layer graph.
   *  Called in setup function. All layers in connected component of
   *  layer graph are added to the model and all parent/child
   *  relationships between layers are reciprocated.
   */
  virtual void setup_layer_topology();
  /** Set up layer execution order.
   *  Called in setup function.
   */
  virtual void setup_layer_execution_order();
  /** Set up layers.
   *  Called in setup function.
   */
  virtual void setup_layers();
  /** Set up weights.
   *  Called in setup function. All weights being used by layers or
   *  the objective function are added to the model and all unused
   *  weights are deleted.
   */
  virtual void setup_weights();

  /** Reset model pointer and execution mode. */
  virtual void reset_mode_and_model(execution_mode mode);
  /** Reset model statistics for an epoch. */
  virtual void reset_epoch_statistics(execution_mode mode);
  /** Evaluate model on a mini-batch */
  virtual bool evaluate_mini_batch(execution_mode mode);
  /** Train model on a mini-batch. */
  virtual bool train_mini_batch();

  /** Forward propagation step. */
  virtual void forward_prop(execution_mode mode);
  /** Backward propagation step. */
  virtual void backward_prop();
  /** Clear each optimizer's gradient.
   *  This must be called before training forward prop since layers
   *  set an optimizer flag during forward prop.
   */
  virtual void clear_gradients();
  /** Update weights step. */
  virtual void update_weights();
  /** Update layers step. */
  virtual bool update_layers();
  /** Reconcile weight values.
   *  If weight values are duplicated across multiple processes, they
   *  are set to the average across the processes.
   */
  virtual void reconcile_weight_values();

  ////////////////////////////////////////////////////////////
  // Callbacks
  ////////////////////////////////////////////////////////////

  /** Execute callbacks at start of training. */
  virtual void do_train_begin_cbs();
  /** Execute callbacks at end of training. */
  virtual void do_train_end_cbs();
  /** Execute callbacks at start of evaluation. */
  virtual void do_evaluate_begin_cbs(execution_mode mode);
  /** Execute callbacks at end of evaluation. */
  virtual void do_evaluate_end_cbs(execution_mode mode);
  /** Execute callbacks at start of epoch. */
  virtual void do_epoch_begin_cbs();
  /** Execute callbacks at end of epoch. */
  virtual void do_epoch_end_cbs();
  /** Execute callbacks at start of mini-batch. */
  virtual void do_batch_begin_cbs(execution_mode mode);
  /** Execute callbacks at end of mini-batch. */
  virtual void do_batch_end_cbs(execution_mode mode);
  /** Execute callbacks at start of model forward propagation. */
  virtual void do_model_forward_prop_begin_cbs(execution_mode mode);
  /** Execute callbacks at end of model forward propagation. */
  virtual void do_model_forward_prop_end_cbs(execution_mode mode);
  /** Execute callbacks at start of layer forward propagation. */
  virtual void do_layer_forward_prop_begin_cbs(execution_mode mode, Layer *l);
  /** Execute callbacks at end of layer forward propagation. */
  virtual void do_layer_forward_prop_end_cbs(execution_mode mode, Layer *l);
  /** Execute callbacks at start of model backward propagation. */
  virtual void do_model_backward_prop_begin_cbs();
  /** Execute callbacks at end of model backward propagation. */
  virtual void do_model_backward_prop_end_cbs();
  /** Execute callbacks at start of layer backward propagation. */
  virtual void do_layer_backward_prop_begin_cbs(Layer *l);
  /** Execute callbacks at end of layer backward propagation. */
  virtual void do_layer_backward_prop_end_cbs(Layer *l);
  /** Execute callbacks at start of model optimization. */
  virtual void do_model_optimize_begin_cbs();
  /** Execute callbacks at end of model optimization. */
  virtual void do_model_optimize_end_cbs();
  /** Execute callbacks at the start of weight optimization. */
  virtual void do_weight_optimize_begin_cbs(weights *w);
  /** Execute callbacks at the end of weight optimization. */
  virtual void do_weight_optimize_end_cbs(weights *w);

private:

  /** Search layer graph and add all connected layers. */
  void add_connected_layers();
  /** Insert evaluation layers where needed.
   *  If an objective function layer term or a layer metric
   *  corresponds to a layer that is not an evaluation layer, an
   *  evaluation layer is added as a child of the original layer and
   *  set as the corresponding layer to the layer term or layer
   *  metric.
   */
  void add_evaluation_layers();
  /** Insert dummy layers after layers with too few children.
   *  If a layer expects more child layers than it has, add dummy
   *  layers until it has enough children.
   */
  void add_dummy_layers();
  /** Insert split layers after layers with too many children.
   *  If a layer expects one child layer but has multiple, add a split
   *  layer. The split layer will be the original layer's child and
   *  the split layer's children will be the original children.
   */
  void add_split_layers();

#ifdef LBANN_HAS_DISTCONV
 public:
  CUDAClock m_clk_start;
  CUDAClock m_clk_end;
  float m_elapsed = 0;
  int m_num_iterations = 0;
  void clock_start() {
    m_clk_start.record();
  }
  void clock_end() {
    m_clk_end.record();
    m_elapsed += m_clk_end.get_elapsed_time_since(m_clk_start);
    ++m_num_iterations;
  }
  void clock_clear() {
    m_elapsed = 0.0;
    m_num_iterations = 0;
  }
#endif
};

}  // namespace lbann

#endif  // LBANN_MODEL_HPP
