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

#ifndef LBANN_LAYERS_DISTCONV_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_DISTCONV_LAYER_HPP_INCLUDED

#include "lbann/utils/distconv.hpp"

namespace lbann {

class Layer;

class distconv_layer {
  friend class Layer;
public:
  distconv_layer(Layer& layer);
  virtual ~distconv_layer() = default;

  /** Get activation tensor. */
  virtual const dc::AbsTensor& get_activations(const Layer& child) const = 0;

 protected:
  virtual Layer& layer();
  virtual const Layer& layer() const;
  std::string get_name() const;

  virtual void setup_activations(const dc::Dist& dist, bool allocate=true) = 0;

 private:
  Layer& m_layer;
};

} // namespace lbann

#endif // LBANN_LAYERS_DISTCONV_LAYER_HPP_INCLUDED
