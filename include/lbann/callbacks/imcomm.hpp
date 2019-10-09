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
//
// imcomm .hpp .cpp - Send gradient updates between models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_IMCOMM_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_IMCOMM_HPP_INCLUDED

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "lbann/callbacks/callback.hpp"

namespace lbann {
namespace callback {

/**
 * Support inter-model communication after each mini-batch to synchronize
 * gradient updates.
 */
class imcomm : public callback_base {
 public:
  using callback_base::on_backward_prop_end;

  enum comm_type {
    NONE,  /** Do no gradient updates. */
    NORMAL,  /** Simply sum gradient updates. */
  };

  /**
   * Initialize with ct being used for all weights.
   */
  imcomm(comm_type ct = NORMAL,
                        const std::shared_ptr<lbann_summary>& summarizer = nullptr);
  imcomm(const imcomm&) = default;
  imcomm& operator=(const imcomm&) = default;
  imcomm* copy() const override {
    return new imcomm(*this);
  }
  /**
   * Convenience initialization to do one update type for specific weights.
   * Implies no inter-model updates for other weights.
   */
  imcomm(comm_type ct, std::unordered_set<weights *> weights_list,
                        const std::shared_ptr<lbann_summary>& summarizer = nullptr);

  /** Choose comm type ct for weights. */
  void set_weights_comm(weights *w, comm_type ct);

  /** Do initialization for this model. */
  void setup(model *m) override;
  /** Make sure all models have the same weights. */
  void on_train_begin(model *m) override;
  /** Do inter-model gradient updates. */
  void on_backward_prop_end(model *m) override;

  std::string name() const override { return "imcomm"; }

 private:
  /** Parameters for a given set of weights. */
  struct imcomm_params {
    /** Type of communication done. */
    comm_type ct = NONE;
  };
  /** Default communication type. */
  comm_type m_default_ct;
  /** Per-weights parameters. */
  std::unordered_map<weights *, imcomm_params> m_weights_params;

  /** Summarize relevant statistics. */
  void do_summary(model *m, weights *w, EvalType im_time);

  /** @brief lbann_summary */
  std::shared_ptr<lbann_summary> m_summarizer = nullptr;
};


/** returns a string representation of the weight_initialization */
std::string get_comm_type_name(imcomm::comm_type m);

// Builder function
std::unique_ptr<callback_base>
build_imcomm_callback_from_pbuf(
  const google::protobuf::Message&, std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_IMCOMM_HPP_INCLUDED
