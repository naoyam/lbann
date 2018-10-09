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

#include "lbann/layers/loss/top_k_categorical_accuracy.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/cuda.hpp"
#include "lbann/utils/exception.hpp"

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/discard_iterator.h>

#ifdef LBANN_HAS_DISTCONV
#include "lbann/utils/distconv.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {

namespace {

/** Sparse vector entry. */
struct entry {

  /** Vector entry value. */
  DataType value;
  /** Vector entry index. */
  El::Int index;

  /** Minimum possible value. */
  static constexpr DataType min_value = -std::numeric_limits<DataType>::infinity();
  /** Maximum possible index. */
  static constexpr El::Int max_index = std::numeric_limits<El::Int>::max();

};

/** Comparison operation to sort sparse vector entries.
 *  Entries are sorted by value in decreasing order, with ties broken
 *  in favor of entries with smaller indices.
 */
struct entry_compare : thrust::binary_function<entry,entry,bool> {
  __host__ __device__ bool operator()(const entry& a, const entry& b) const {
    return a.value > b.value || (a.value == b.value && a.index < b.index);
  }
};

/** Reduction operation to get largest sparse vector entry.
 *  Ties are broken in favor of entries with smaller indices.
 */
struct entry_reduce : thrust::binary_function<entry,entry,entry> {
  __host__ __device__ entry operator()(const entry& a, const entry& b) const {
    if (a.value > b.value || (a.value == b.value && a.index < b.index)) {
      return a;
    } else {
      return b;
    }
  }
};

/** Convert columns of a dense matrix into sparse vectors.
 *  The matrix and vectors are both distributed, so entry indices in
 *  the sparse vectors correspond to global row indices in the dense
 *  matrix.
 */
__global__ void dense_matrix_to_sparse_vectors(El::Int local_vector_size,
                                               El::Int local_matrix_height,
                                               El::Int local_matrix_width,
                                               El::Int global_matrix_height,
                                               El::Int global_matrix_col_shift,
                                               El::Int global_matrix_col_stride,
                                               const DataType* __restrict__ local_matrix,
                                               El::Int local_matrix_ldim,
                                               entry* __restrict__ local_entries,
                                               El::Int local_entries_ldim) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int num_threads = blockDim.x * gridDim.x;
  const El::Int num_local_entries = local_vector_size * local_matrix_width;
  for (El::Int i = gid; i < num_local_entries; i += num_threads) {
    const auto& local_row = i % local_vector_size;
    const auto& local_col = i / local_vector_size;
    auto& current_entry = local_entries[local_row + local_col * local_entries_ldim];
    if (local_row < local_matrix_height) {
      const auto& global_row = (global_matrix_col_shift
                                + local_row * global_matrix_col_stride);
      current_entry.value = local_matrix[local_row + local_col * local_matrix_ldim];
      current_entry.index = global_row;
    } else {
      current_entry.value = entry::min_value;
      current_entry.index = global_matrix_height;
    }
  }
}

/** Fill an array with a corresponding tensor index.
 *  Consider a d(1) x d(2) x ... x d(n) tensor with entry indices
 *  denoted with (i(1), ..., i(n)). This tensor is contiguous in
 *  memory with d(1) as the most major dimension and d(n) as the most
 *  minor (e.g. d(1) is the width and d(2) is the height for a
 *  column-major matrix). Given some k, this kernel sets each entry in
 *  the tensor to i(k). Using this notation:
 *    tensor_size = d(1) * ... * d(n)
 *    dim         = d(k)
 *    dim_stride  = d(k+1) * ... * d(n)
 */
__global__ void fill_with_tensor_index(El::Int tensor_size,
                                       El::Int dim,
                                       El::Int dim_stride,
                                       El::Int* tensor) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int num_threads = blockDim.x * gridDim.x;
  for (El::Int i = gid; i < tensor_size; i += num_threads) {
    tensor[i] = (i / dim_stride) % dim;
  }
}

/** Get indices corresponding to one-hot matrix.
 *  Each column of the input matrix is interpreted as a one-hot
 *  vector. Note that we may get race conditions if a matrix column is
 *  not a one-hot vector.
 */
__global__ void one_hot_matrix_to_indices(El::Int local_height,
                                          El::Int local_width,
                                          El::Int global_matrix_col_shift,
                                          El::Int global_matrix_col_stride,
                                          const DataType* __restrict__ local_matrix,
                                          El::Int local_matrix_ldim,
                                          El::Int* __restrict__ indices) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int num_threads = blockDim.x * gridDim.x;
  const El::Int local_size = local_height * local_width;
  for (El::Int i = gid; i < local_size; i += num_threads) {
    const auto& local_row = i % local_height;
    const auto& local_col = i / local_height;
    if (local_matrix[local_row + local_col * local_matrix_ldim] > DataType(0)) {
      const auto& global_row = (global_matrix_col_shift
                                + local_row * global_matrix_col_stride);
      indices[local_col] = global_row;
    }
  }
}

/** Compute categorical accuracy for each matrix column.
 *  Loss is one if the label index matches one of the top-k entries
 *  and is otherwise zero.
 */
__global__ void compute_categorical_accuracy(El::Int k,
                                             El::Int width,
                                             El::Int max_entry,
                                             const entry*  __restrict__ top_entries,
                                             El::Int top_entries_ldim,
                                             const El::Int*  __restrict__ label_indices,
                                             DataType* __restrict__ loss,
                                             El::Int loss_stride) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int num_threads = blockDim.x * gridDim.x;
  const El::Int num_entries = width * k;
  for (El::Int i = gid; i < num_entries; i += num_threads) {
    const auto& ind = i % k;
    const auto& col = i / k;
    const auto& label_index = label_indices[col];
    if (top_entries[ind + col * top_entries_ldim].index == label_index
        && label_index <= max_entry) {
      loss[col * loss_stride] = DataType(1);
    }
  }
}

/** GPU implementation of top-k categorical accuracy layer forward prop. */
void fp_gpu(lbann_comm& comm,
            El::Int k,
            const AbsDistMat& predictions,
            const AbsDistMat& labels,
            AbsDistMat& loss,
            execution_mode mode) {
  if (predictions.Wrap() != El::ELEMENT
      || labels.Wrap() != El::ELEMENT
      || loss.Wrap() != El::ELEMENT) {
    LBANN_ERROR("top-k categorical accuracy layer GPU implementation "
                "assumes elemental distributed matrices");
  }
#ifdef LBANN_HAS_DISTCONV
  if (mode == execution_mode::training &&
      dc::skip_metrics_while_training()) {
    El::Zero(loss);
    return;
  }
#endif

  // Local matrices
  const auto& local_predictions = predictions.LockedMatrix();
  const auto& local_labels = labels.LockedMatrix();
  auto& local_loss = loss.Matrix();
  const El::Int height = predictions.Height();
  const El::Int local_height = local_predictions.Height();
  const El::Int local_width = local_predictions.Width();

  // Trivial cases
  if (k < 1) {
    El::Zero(loss);
    return;
  } else if (k >= height) {
    El::Fill(loss, DataType(1));
    return;
  } else if (local_width < 1) {
    return;
  }

  // Column communicator
  auto&& col_comm = predictions.ColComm();
  const auto& col_comm_rank = El::mpi::Rank(col_comm);
  const auto& col_comm_size = El::mpi::Size(col_comm);
  const auto& col_comm_root = loss.RowOwner(0);

  // GPU objects
  auto&& stream = El::GPUManager::Stream();
  auto&& event = El::GPUManager::Event();
  El::SyncInfo<El::Device::GPU> syncInfo{stream, event};

  cuda::thrust::allocator<> alloc(stream);
  using entry_array = thrust::device_vector<entry, cuda::thrust::allocator<entry>>;
  using index_array = thrust::device_vector<El::Int, cuda::thrust::allocator<El::Int>>;

  // Get label indices
  index_array label_indices(local_width);
  {
    const auto& local_size = local_height * local_width;
    const auto& block_dim = 256;
    const auto& grid_dim = (local_size + block_dim - 1) / block_dim;
    thrust::fill(thrust::cuda::par(alloc).on(stream),
                 label_indices.begin(),
                 label_indices.end(),
                 height);
    one_hot_matrix_to_indices<<<grid_dim, block_dim, 0, stream>>>(
      local_height, local_width,
      labels.ColShift(), labels.ColStride(),
      local_labels.LockedBuffer(), local_labels.LDim(),
      label_indices.data().get());
    /// @todo The LBANN Aluminum interface doesn't gracefully handle
    /// GPU data that is not DataType.
    El::mpi::AllReduce(label_indices.data().get(),
                       label_indices.size(),
                       El::mpi::MIN,
                       col_comm, syncInfo);
  }

  // Find top-k entries in each column of local prediction matrix
  entry_array top_entries(local_width * k);
  {
    const auto& num_local_entries_per_col = std::max(local_height, k);
    const auto& num_local_entries = local_width * num_local_entries_per_col;
    const auto& block_dim = 256;
    const auto& grid_dim = (num_local_entries + block_dim - 1) / block_dim;
    entry_array local_entries(num_local_entries);
    index_array local_entries_cols(num_local_entries);
    dense_matrix_to_sparse_vectors<<<grid_dim, block_dim, 0, stream>>>(
      num_local_entries_per_col, local_height, local_width, height,
      predictions.ColShift(), predictions.ColStride(),
      local_predictions.LockedBuffer(), local_predictions.LDim(),
      local_entries.data().get(), num_local_entries_per_col);
    fill_with_tensor_index<<<grid_dim, block_dim, 0, stream>>>(
      num_local_entries, local_width, num_local_entries_per_col,
      local_entries_cols.data().get());
    if (k == 1) {
      thrust::reduce_by_key(thrust::cuda::par(alloc).on(stream),
                            local_entries_cols.begin(),
                            local_entries_cols.end(),
                            local_entries.begin(),
                            thrust::make_discard_iterator(),
                            top_entries.begin(),
                            thrust::equal_to<El::Int>(),
                            entry_reduce());
    } else {
      thrust::sort_by_key(thrust::cuda::par(alloc).on(stream),
                          local_entries.begin(),
                          local_entries.end(),
                          local_entries_cols.begin(),
                          entry_compare());
      thrust::stable_sort_by_key(thrust::cuda::par(alloc).on(stream),
                                 local_entries_cols.begin(),
                                 local_entries_cols.end(),
                                 local_entries.begin());
      CHECK_CUDA(cudaMemcpy2DAsync(top_entries.data().get(),
                                   k * sizeof(entry),
                                   local_entries.data().get(),
                                   num_local_entries_per_col * sizeof(entry),
                                   k * sizeof(entry),
                                   local_width,
                                   cudaMemcpyDeviceToDevice,
                                   stream));
    }
  }

  // Find top-k entries in each column of global prediction matrix
  if (col_comm_size > 1) {
    const auto& num_entries_per_rank = local_width * k;
    const auto& num_entries = col_comm_size * num_entries_per_rank;
    const auto& block_dim = 256;
    const auto& grid_dim = (num_entries + block_dim - 1) / block_dim;
    if (col_comm_rank != col_comm_root) {
      comm.gather(reinterpret_cast<El::byte*>(top_entries.data().get()),
                  top_entries.size() * sizeof(entry),
                  col_comm_root,
                  col_comm, syncInfo);
    } else {
      entry_array global_top_entries(num_entries);
      index_array global_top_entries_cols(num_entries);
      comm.gather(reinterpret_cast<El::byte*>(top_entries.data().get()),
                  top_entries.size() * sizeof(entry),
                  reinterpret_cast<El::byte*>(global_top_entries.data().get()),
                  col_comm, syncInfo);
      fill_with_tensor_index<<<grid_dim, block_dim, 0, stream>>>(
        num_entries, local_width, k, global_top_entries_cols.data().get());
      thrust::sort_by_key(thrust::cuda::par(alloc).on(stream),
                          global_top_entries.begin(),
                          global_top_entries.end(),
                          global_top_entries_cols.begin(),
                          entry_compare());
      thrust::stable_sort_by_key(thrust::cuda::par(alloc).on(stream),
                                 global_top_entries_cols.begin(),
                                 global_top_entries_cols.end(),
                                 global_top_entries.begin());
      CHECK_CUDA(cudaMemcpy2DAsync(top_entries.data().get(),
                                   k * sizeof(entry),
                                   global_top_entries.data().get(),
                                   col_comm_size * k * sizeof(entry),
                                   k * sizeof(entry),
                                   local_width,
                                   cudaMemcpyDeviceToDevice,
                                   stream));
    }
  }

  // Compute categorical accuracy
  El::Zero(loss);
  if (col_comm_rank == col_comm_root) {
    const auto& num_entries = local_width * k;
    const auto& block_dim = 256;
    const auto& grid_dim = (num_entries + block_dim - 1) / block_dim;
    compute_categorical_accuracy<<<grid_dim, block_dim, 0, stream>>>(
      k, local_width, height-1,
      top_entries.data().get(), k,
      label_indices.data().get(),
      local_loss.Buffer(), local_loss.LDim());
  }

}

} // namespace

template <>
void top_k_categorical_accuracy_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>
     ::fp_compute() {
  fp_gpu(*get_comm(),
         m_k,
         get_prev_activations(0),
         get_prev_activations(1),
         get_activations(),
         get_model()->get_execution_mode());
}
template <>
void top_k_categorical_accuracy_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
     ::fp_compute() {
  fp_gpu(*get_comm(),
         m_k,
         get_prev_activations(0),
         get_prev_activations(1),
         get_activations(),
         get_model()->get_execution_mode());
}

} // namespace lbann
