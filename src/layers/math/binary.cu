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

#define LBANN_BINARY_LAYER_INSTANTIATE
#include "lbann/layers/math/binary.hpp"

namespace lbann {

namespace {

/** CUDA kernel to apply an binary backprop operator. */
template <typename BinaryBackPropOperator>
__global__
void binary_backprop_operator_kernel(El::Int height, El::Int width,
                                     const DataType* __restrict__ x1,
                                     El::Int x1_ldim,
                                     const DataType* __restrict__ x2,
                                     El::Int x2_ldim,
                                     const DataType* __restrict__ dy,
                                     El::Int dy_ldim,
                                     DataType* __restrict__ dx1,
                                     El::Int dx1_ldim,
                                     DataType* __restrict__ dx2,
                                     El::Int dx2_ldim) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int size = height * width;
  const El::Int num_threads = blockDim.x * gridDim.x;
  BinaryBackPropOperator op;
  for (El::Int pos = gid; pos < size; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    op(x1[row + col * x1_ldim],
       x2[row + col * x2_ldim],
       dy[row + col * dy_ldim],
       dx1[row + col * dx1_ldim],
       dx2[row + col * dx2_ldim]);
  }
}


/** Apply a binary backprop operator to CPU data.
 *  The input and output data must be on CPU and must have the same
 *  dimensions. Given a binary function \f$ y = f(x_1,x_2) \f$, the
 *  corresponding BinaryBackPropOperator is a 5-ary function with the
 *  arguments \f$ x_1 \f$, \f$ x_2 \f$, \f$ dL/dy \f$, \f$ dL/dx_1\f$,
 *  \f$ dL/dx_2 \f$. The last two arguments should be overwritten when
 *  the BinaryBackPropOperator is called.
 */
template <typename BinaryBackPropOperator>
void apply_binary_backprop_operator(const AbsMat& x1,
                                    const AbsMat& x2,
                                    const AbsMat& dy,
                                    AbsMat& dx1,
                                    AbsMat& dx2) {

  // Get CUDA grid dimensions
  // Note: Maximum CUDA grid dimension is 2^32-1
  // (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications).
  const El::Int height = x1.Height();
  const El::Int width = x1.Width();
  const El::Int block_dim = 256;
  El::Int grid_dim = (height * width + block_dim - 1) / block_dim;
  if (sizeof(El::Int) > sizeof(unsigned int)
      && grid_dim > std::numeric_limits<uint32_t>::max()) {
    grid_dim = std::numeric_limits<uint32_t>::max();
  }

  // Launch CUDA kernel
  if (grid_dim > 0) {
    CHECK_CUDA(cudaSetDevice(El::GPUManager::Device()));
    binary_backprop_operator_kernel<BinaryBackPropOperator>
      <<<grid_dim, block_dim, 0, El::GPUManager::Stream()>>>(
        height, width,
        x1.LockedBuffer(), x1.LDim(),
        x2.LockedBuffer(), x2.LDim(),
        dy.LockedBuffer(), dy.LDim(),
        dx1.Buffer(), dx1.LDim(),
        dx2.Buffer(), dx2.LDim());
  }

}

// =========================================================
// Operator objects for entry-wise binary layers
// =========================================================
// Note: Binary operator corresponds to forward prop step
// (\f$ y = f(x_1,x_2) \f$) and 5-ary operator corresponds
// to back prop step
// (\f$ \frac{dL}{dx_i} = \frac{dL}{dy} \frac{df}{dx_i}(x_1,x_2) \f$).

/** Add operator. */
struct add_op {
  inline __device__ DataType operator()(const DataType& x1,
                                        const DataType& x2) const {
    return x1 + x2;
  }
  inline __device__ void operator()(const DataType& x1,
                                    const DataType& x2,
                                    const DataType& dy,
                                    DataType& dx1,
                                    DataType& dx2) const {
    dx1 = dy;
    dx2 = dy;
  }
};

/** Subtract operator. */
struct subtract_op {
  inline __device__ DataType operator()(const DataType& x1,
                                        const DataType& x2) const {
    return x1 - x2;
  }
  inline __device__ void operator()(const DataType& x1,
                                    const DataType& x2,
                                    const DataType& dy,
                                    DataType& dx1,
                                    DataType& dx2) const {
    dx1 = dy;
    dx2 = -dy;
  }
};

/** Multiply operator. */
struct multiply_op {
  inline __device__ DataType operator()(const DataType& x1,
                                        const DataType& x2) const {
    return x1 * x2;
  }
  inline __device__ void operator()(const DataType& x1,
                                    const DataType& x2,
                                    const DataType& dy,
                                    DataType& dx1,
                                    DataType& dx2) const {
    dx1 = dy * x2;
    dx2 = dy * x1;
  }
};

/** Divide operator. */
struct divide_op {
  inline __device__ DataType operator()(const DataType& x1,
                                        const DataType& x2) const {
    return x1 / x2;
  }
  inline __device__ void operator()(const DataType& x1,
                                    const DataType& x2,
                                    const DataType& dy,
                                    DataType& dx1,
                                    DataType& dx2) const {
    dx1 = dy / x2;
    dx2 = -dy * x1 / (x2*x2);
  }
};

/** Modulo operator. */
struct mod_op {
  inline __device__ DataType operator()(const DataType& x1,
                                        const DataType& x2) const {
    return cuda::mod(x1, x2);
  }
  inline __device__ void operator()(const DataType& x1,
                                    const DataType& x2,
                                    const DataType& dy,
                                    DataType& dx1,
                                    DataType& dx2) const {
    dx1 = dy;
    dx2 = -dy * cuda::floor(x1 / x2);
  }
};

/** Power operator. */
struct pow_op {
  inline __device__ DataType operator()(const DataType& x1,
                                        const DataType& x2) const {
    return cuda::pow(x1, x2);
  }
  inline __device__ void operator()(const DataType& x1,
                                    const DataType& x2,
                                    const DataType& dy,
                                    DataType& dx1,
                                    DataType& dx2) const {

    dx1 = dy * x2 * cuda::pow(x1, x2 - DataType(1));
    dx2 = dy * cuda::log(x1) * cuda::pow(x1, x2);
  }
};

/** Safe divide operator.
 *  If a standard division produces an infinity or NaN, zero is output
 *  instead.
 */
struct safe_divide_op {
  inline __device__ DataType operator()(const DataType& x1,
                                        const DataType& x2) const {
    const auto& y = x1 / x2;
    if (isfinite(y)) { return y; }
    else             { return DataType(0); }
  }
  inline __device__ void operator()(const DataType& x1,
                                    const DataType& x2,
                                    const DataType& dy,
                                    DataType& dx1,
                                    DataType& dx2) const {
    const auto& y = x1 / x2;
    if (isfinite(y)) {
      dx1 = dy / x2;
      dx2 = -dy * x1 / (x2*x2);
    } else {
      dx1 = DataType(0);
      dx2 = DataType(0);
    }
  }
};

/** Squared difference operator. */
struct squared_difference_op {
  inline __device__ DataType operator()(const DataType& x1,
                                        const DataType& x2) const {
    const auto& diff = x1 - x2;
    return diff * diff;
  }
  inline __device__ void operator()(const DataType& x1,
                                    const DataType& x2,
                                    const DataType& dy,
                                    DataType& dx1,
                                    DataType& dx2) const {
    dx1 = dy * 2*(x1-x2);
    dx2 = dy * 2*(x2-x1);
  }
};

/** Maximum operator. */
struct max_op {
  inline __device__ DataType operator()(const DataType& x1,
                                        const DataType& x2) const {
    return cuda::max(x1, x2);
  }
  inline __device__ void operator()(const DataType& x1,
                                    const DataType& x2,
                                    const DataType& dy,
                                    DataType& dx1,
                                    DataType& dx2) const {
    if (x1 > x2) {
      dx1 = dy;
      dx2 = DataType(0);
    } else if (x2 > x1) {
      dx1 = DataType(0);
      dx2 = dy;
    } else {
      dx1 = dy / 2;
      dx2 = dy / 2;
    }
  }
};

/** Minimum operator. */
struct min_op {
  inline __device__ DataType operator()(const DataType& x1,
                                        const DataType& x2) const {
    return cuda::min(x1, x2);
  }
  inline __device__ void operator()(const DataType& x1,
                                    const DataType& x2,
                                    const DataType& dy,
                                    DataType& dx1,
                                    DataType& dx2) const {
    if (x1 < x2) {
      dx1 = dy;
      dx2 = DataType(0);
    } else if (x2 < x1) {
      dx1 = DataType(0);
      dx2 = dy;
    } else {
      dx1 = dy / 2;
      dx2 = dy / 2;
    }
  }
};

/** Equal operator. */
struct equal_op {
  inline __device__ DataType operator()(const DataType& x1,
                                        const DataType& x2) const {
    return x1 == x2 ? DataType(1) : DataType(0);
  }
  inline __device__ void operator()(const DataType& x1,
                                    const DataType& x2,
                                    const DataType& dy,
                                    DataType& dx1,
                                    DataType& dx2) const {
    dx1 = DataType(0);
    dx2 = DataType(0);
  }
};

/** Not equal operator. */
struct not_equal_op {
  inline __device__ DataType operator()(const DataType& x1,
                                        const DataType& x2) const {
    return x1 == x2 ? DataType(0) : DataType(1);
  }
  inline __device__ void operator()(const DataType& x1,
                                    const DataType& x2,
                                    const DataType& dy,
                                    DataType& dx1,
                                    DataType& dx2) const {
    dx1 = DataType(0);
    dx2 = DataType(0);
  }
};

/** Less than operator. */
struct less_op {
  inline __device__ DataType operator()(const DataType& x1,
                                        const DataType& x2) const {
    return x1 < x2 ? DataType(1) : DataType(0);
  }
  inline __device__ void operator()(const DataType& x1,
                                    const DataType& x2,
                                    const DataType& dy,
                                    DataType& dx1,
                                    DataType& dx2) const {
    dx1 = DataType(0);
    dx2 = DataType(0);
  }
};

/** Less than or equal operator. */
struct less_equal_op {
  inline __device__ DataType operator()(const DataType& x1,
                                        const DataType& x2) const {
    return x1 <= x2 ? DataType(1) : DataType(0);
  }
  inline __device__ void operator()(const DataType& x1,
                                    const DataType& x2,
                                    const DataType& dy,
                                    DataType& dx1,
                                    DataType& dx2) const {
    dx1 = DataType(0);
    dx2 = DataType(0);
  }
};

/** Greater than operator. */
struct greater_op {
  inline __device__ DataType operator()(const DataType& x1,
                                        const DataType& x2) const {
    return x1 > x2 ? DataType(1) : DataType(0);
  }
  inline __device__ void operator()(const DataType& x1,
                                    const DataType& x2,
                                    const DataType& dy,
                                    DataType& dx1,
                                    DataType& dx2) const {
    dx1 = DataType(0);
    dx2 = DataType(0);
  }
};

/** Greater than or equal operator. */
struct greater_equal_op {
  inline __device__ DataType operator()(const DataType& x1,
                                        const DataType& x2) const {
    return x1 >= x2 ? DataType(1) : DataType(0);
  }
  inline __device__ void operator()(const DataType& x1,
                                    const DataType& x2,
                                    const DataType& dy,
                                    DataType& dx1,
                                    DataType& dx2) const {
    dx1 = DataType(0);
    dx2 = DataType(0);
  }
};

/** Logical and operator. */
struct logical_and_op {
  inline __device__ DataType operator()(const DataType& x1,
                                        const DataType& x2) const {
    const auto& b1 = x1 != DataType(0) && !isnan(x1);
    const auto& b2 = x2 != DataType(0) && !isnan(x2);
    return (b1 && b2) ? DataType(1) : DataType(0);
  }
  inline __device__ void operator()(const DataType& x1,
                                    const DataType& x2,
                                    const DataType& dy,
                                    DataType& dx1,
                                    DataType& dx2) const {
    dx1 = DataType(0);
    dx2 = DataType(0);
  }
};

/** Logical or operator. */
struct logical_or_op {
  inline __device__ DataType operator()(const DataType& x1,
                                        const DataType& x2) const {
    const auto& b1 = x1 != DataType(0) && !isnan(x1);
    const auto& b2 = x2 != DataType(0) && !isnan(x2);
    return (b1 || b2) ? DataType(1) : DataType(0);
  }
  inline __device__ void operator()(const DataType& x1,
                                    const DataType& x2,
                                    const DataType& dy,
                                    DataType& dx1,
                                    DataType& dx2) const {
    dx1 = DataType(0);
    dx2 = DataType(0);
  }
};

/** Logical xor operator. */
struct logical_xor_op {
  inline __device__ DataType operator()(const DataType& x1,
                                        const DataType& x2) const {
    const auto& b1 = x1 != DataType(0) && !isnan(x1);
    const auto& b2 = x2 != DataType(0) && !isnan(x2);
    return (b1 || b2) && !(b1 && b2) ? DataType(1) : DataType(0);
  }
  inline __device__ void operator()(const DataType& x1,
                                    const DataType& x2,
                                    const DataType& dy,
                                    DataType& dx1,
                                    DataType& dx2) const {
    dx1 = DataType(0);
    dx2 = DataType(0);
  }
};

} // namespace

// Template instantiation
#define INSTANTIATE(layer, op)                                          \
  template <>                                                           \
  void layer<data_layout::MODEL_PARALLEL, El::Device::GPU>              \
         ::fp_compute() {                                               \
    cuda::apply_entrywise_binary_operator<op>(get_prev_activations(0),  \
                                              get_prev_activations(1),  \
                                              get_activations());       \
  }                                                                     \
  template <>                                                           \
  void layer<data_layout::MODEL_PARALLEL, El::Device::GPU>              \
         ::bp_compute() {                                               \
    apply_binary_backprop_operator<op>(get_local_prev_activations(0),   \
                                       get_local_prev_activations(1),   \
                                       get_local_prev_error_signals(),  \
                                       get_local_error_signals(0),      \
                                       get_local_error_signals(1));     \
  }                                                                     \
  template <>                                                           \
  void layer<data_layout::DATA_PARALLEL, El::Device::GPU>               \
         ::fp_compute() {                                               \
    cuda::apply_entrywise_binary_operator<op>(get_prev_activations(0),  \
                                              get_prev_activations(1),  \
                                              get_activations());       \
  }                                                                     \
  template <>                                                           \
  void layer<data_layout::DATA_PARALLEL, El::Device::GPU>               \
  ::bp_compute() {                                                      \
    apply_binary_backprop_operator<op>(get_local_prev_activations(0),   \
                                       get_local_prev_activations(1),   \
                                       get_local_prev_error_signals(),  \
                                       get_local_error_signals(0),      \
                                       get_local_error_signals(1));     \
  }                                                                     \
  BINARY_ETI_INST_MACRO_DEV(layer, El::Device::GPU)

INSTANTIATE(add_layer, add_op);
INSTANTIATE(subtract_layer, subtract_op);
INSTANTIATE(multiply_layer, multiply_op);
INSTANTIATE(divide_layer, divide_op);
INSTANTIATE(mod_layer, mod_op);
INSTANTIATE(pow_layer, pow_op);
INSTANTIATE(safe_divide_layer, safe_divide_op);
INSTANTIATE(squared_difference_layer, squared_difference_op);
INSTANTIATE(max_layer, max_op);
INSTANTIATE(min_layer, min_op);
INSTANTIATE(equal_layer, equal_op);
INSTANTIATE(not_equal_layer, not_equal_op);
INSTANTIATE(less_layer, less_op);
INSTANTIATE(less_equal_layer, less_equal_op);
INSTANTIATE(greater_layer, greater_op);
INSTANTIATE(greater_equal_layer, greater_equal_op);
INSTANTIATE(logical_and_layer, logical_and_op);
INSTANTIATE(logical_or_layer, logical_or_op);
INSTANTIATE(logical_xor_layer, logical_xor_op);

} // namespace lbann
