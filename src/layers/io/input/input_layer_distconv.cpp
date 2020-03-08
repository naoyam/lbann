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

#define LBANN_INPUT_LAYER_DISTCONV_INSTANTIATE
#include "lbann/layers/io/input/input_layer_distconv.hpp"

namespace lbann {

template class input_layer_distconv<
  DataType, partitioned_io_buffer<DataType>, data_layout::DATA_PARALLEL, El::Device::CPU,
  DataType>;
template class input_layer_distconv<
  DataType, partitioned_io_buffer<DataType>, data_layout::DATA_PARALLEL, El::Device::CPU,
  int16_t>;
#ifdef LBANN_HAS_GPU
template class input_layer_distconv<
  DataType, partitioned_io_buffer<DataType>, data_layout::DATA_PARALLEL, El::Device::GPU,
  DataType>;
template class input_layer_distconv<
  DataType, partitioned_io_buffer<DataType>, data_layout::DATA_PARALLEL, El::Device::GPU,
  int16_t>;
#endif // LBANN_HAS_GPU

}// namespace lbann