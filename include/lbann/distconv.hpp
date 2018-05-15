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

#ifndef LBANN_DISTCONV_HPP_INCLUDED
#define LBANN_DISTCONV_HPP_INCLUDED

#include "lbann_config.hpp"

#ifdef LBANN_HAS_DISTCONV

#ifdef LBANN_DEBUG
#define DISTCONV_DEBUG
#endif

#define DISTCONV_HAS_CUDNN

#define DISTCONV_ZERO_OUT_ERROR_SIGNALS
// temporary workaround
#define DISTCONV_USE_SAME_RELU_CALL_AS_LBANN

#include "distconv/distconv.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_mpi.hpp"
#include "distconv/tensor/tensor_cuda.hpp"
#include "distconv/tensor/tensor_mpi_cuda.hpp"
#include "distconv/tensor/shuffle.hpp"

namespace dc = distconv;

namespace lbann {

using Array4 = dc::tensor::Array<4>;

using TensorHost = dc::tensor::Tensor<4, DataType, dc::tensor::LocaleMPI,
                                      dc::tensor::CUDAAllocator>;
using ConstTensorHost = dc::tensor::Tensor<4, DataType, dc::tensor::LocaleMPI,
                                           dc::tensor::CUDAAllocator, true>;

using TensorDev = dc::tensor::Tensor<4, DataType, dc::tensor::LocaleMPI,
                                     dc::tensor::CUDAAllocator>;

using ConstTensorDev = dc::tensor::Tensor<4, DataType, dc::tensor::LocaleMPI,
                                          dc::tensor::CUDAAllocator, true>;

template <bool CONST>
using TensorShuffler = dc::tensor::TensorMPICUDAShuffler<4, DataType, CONST>;

using Dist = dc::tensor::Distribution<4>;
using LocaleMPI = dc::tensor::LocaleMPI;

using MPIPrintStreamDebug = dc::util::MPIPrintStreamDebug;
using MPIPrintStreamError = dc::util::MPIPrintStreamError;
using MPIPrintStreamInfo = dc::util::MPIPrintStreamInfo;

} // namespace lbann

#endif // LBANN_HAS_DISTCONV

#endif // LBANN_DISTCONV_HPP_INCLUDED
