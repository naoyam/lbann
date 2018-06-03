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
// lbann_callback_timer .hpp .cpp - Callback hooks to time training
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_PROFILER_HPP_INCLUDED
#define LBANN_CALLBACKS_PROFILER_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {

/**
 */
class lbann_callback_profiler : public lbann_callback {
 public:
  lbann_callback_profiler(int num_iterations=0) :
      lbann_callback(), m_num_iterations(num_iterations) {}
  lbann_callback_profiler(const lbann_callback_profiler&) = default;
  lbann_callback_profiler& operator=(const lbann_callback_profiler&) = default;
  lbann_callback_profiler* copy() const override {
    return new lbann_callback_profiler(*this);
  }
  void on_epoch_begin(model *m) override;
  void on_epoch_end(model *m) override;
  void on_batch_begin(model *m) override;
  void on_batch_end(model *m) override;
  void on_forward_prop_begin(model *m) override;
  void on_forward_prop_end(model *m) override;
  void on_backward_prop_begin(model *m) override;
  void on_backward_prop_end(model *m) override;
  void on_forward_prop_begin(model *m, Layer *l) override;
  void on_forward_prop_end(model *m, Layer *l) override;
  void on_backward_prop_begin(model *m, Layer *l) override;
  void on_backward_prop_end(model *m, Layer *l) override;
  std::string name() const override { return "profiler"; }
 private:
  static const int num_colors = 20;
  // http://there4.io/2012/05/02/google-chart-color-list/
  int colors[num_colors] = {0x3366CC, 0xDC3912, 0xFF9900, 0x109618, 0x990099, 0x3B3EAC,
                            0x0099C6, 0xDD4477, 0x66AA00, 0xB82E2E, 0x316395, 0x994499,
                            0x22AA99, 0xAAAA11, 0x6633CC, 0xE67300, 0x8B0707, 0x329262,
                            0x5574A6, 0x3B3EAC};
  int get_color(Layer *l);
  int m_num_iterations;
};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_PROFILER_HPP_INCLUDED
