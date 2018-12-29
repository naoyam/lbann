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
// lbann_summary .hpp .cpp - Write summary statistics to Tensorboard
////////////////////////////////////////////////////////////////////////////////

#include "lbann/utils/summary.hpp"

namespace lbann {

#ifdef LBANN_HAS_TBINF

lbann_summary::lbann_summary(std::string logdir, lbann_comm *comm)
  : m_comm(comm) {
  if (m_comm->am_world_master()) {
    m_sw = new TBinf::SummaryWriter(logdir);
  } else {
    m_sw = nullptr;
  }
  m_histogram_buckets = TBinf::SummaryWriter::get_default_histogram_buckets();
}

lbann_summary::~lbann_summary() {
  flush();
  if (m_sw != nullptr) {
    delete m_sw;
  }
}

void lbann_summary::reduce_mean(const std::string tag,
                                const AbsDistMat& mat,
                                int step) {
  // Local sum
  DataType sum = 0.0;

  // Check distributed matrix format
  El::DistData mat_format(mat);
  if(mat_format.colDist == El::STAR && mat_format.rowDist == El::STAR) {
    // Compute local sum on master process if matrix is Star,Star
    if(m_comm->am_model_master()) {
      sum = local_sum(mat.LockedMatrix());
    }
  } else {
    // Compute local sum on all processes if matrix is in MC,MR;
    // Star,VC; or similar format
    // TODO: implement for matrices in Circ,Circ; MC,Star; or similar
    // formats
    sum = local_sum(mat.LockedMatrix());
  }

  // Add local sum to list of pending means
  m_pending_means.emplace_back(tag, step, sum, 0.0f, mat.Height() * mat.Width());
}

void lbann_summary::reduce_min(const std::string tag,
                               const AbsDistMat& mat,
                               int step) {
  DataType mat_local_min = local_min(mat.LockedMatrix());
  m_pending_mins.emplace_back(tag, step, mat_local_min);
}

void lbann_summary::reduce_max(const std::string tag,
                               const AbsDistMat& mat,
                               int step) {
  DataType mat_local_max = local_max(mat.LockedMatrix());
  m_pending_maxes.emplace_back(tag, step, mat_local_max);
}

void lbann_summary::reduce_stdev(const std::string tag,
                                 const AbsDistMat& mat,
                                 int step) {
  // Local sum and squared sum
  DataType sum = 0.0;
  DataType sqsum = 0.0;

  // Check distributed matrix format
  El::DistData mat_format(mat);
  if(mat_format.colDist == El::STAR && mat_format.rowDist == El::STAR) {
    // Compute local sums on master process if matrix is Star,Star
    if(m_comm->am_model_master()) {
      local_sum_sqsum(mat.LockedMatrix(), sum, sqsum);
    }
  } else {
    // Compute local sums on all processes if matrix is in MC,MR;
    // Star,VC; or similar format
    // TODO: implement for matrices in Circ,Circ; MC,Star; or similar
    // formats
    local_sum_sqsum(mat.LockedMatrix(), sum, sqsum);
  }

  // Add local sums to list of pending stdevs.
  m_pending_stdevs.emplace_back(tag, step, sum, sqsum, mat.Height() * mat.Width());
}

void lbann_summary::reduce_scalar(const std::string tag,
                                  DataType s,
                                  int step) {
  if (m_comm->am_model_master()) {
    m_pending_scalars.emplace_back(tag, step, s);
  }
}

void lbann_summary::sum_reduce_scalar(const std::string tag,
                                      DataType s,
                                      int step) {
  m_pending_sum_scalars.emplace_back(tag, step, s);
}

void lbann_summary::reduce_scalar_all(const std::string tag,
                                      DataType s,
                                      int step) {
  m_pending_scalar_alls.emplace_back(tag, step, s);
}

void lbann_summary::reduce_histogram(const std::string tag,
                                     const AbsDistMat& mat,
                                     int step) {
  DataType mat_local_min = local_min(mat.LockedMatrix());
  DataType mat_local_max = local_max(mat.LockedMatrix());
  // Local sum and squared sum
  DataType sum = 0.0f;
  DataType sqsum = 0.0f;
  // Check distributed matrix format
  El::DistData mat_format(mat);
  if(mat_format.colDist == El::STAR && mat_format.rowDist == El::STAR) {
    // Compute local sums on master process if matrix is Star,Star
    if(m_comm->am_model_master()) {
      local_sum_sqsum(mat.LockedMatrix(), sum, sqsum);
    }
  } else {
    // Compute local sums on all processes if matrix is in MC,MR;
    // Star,VC; or similar format
    // TODO: implement for matrices in Circ,Circ; MC,Star; or similar
    // formats
    local_sum_sqsum(mat.LockedMatrix(), sum, sqsum);
  }
  // Compute local buckets.
  std::vector<float> buckets(m_histogram_buckets.size(), 0.0f);
  const int height = mat.LocalHeight();
  const int width = mat.LocalWidth();
  const int ldim = mat.LDim();
  const DataType *__restrict__ mat_buf = mat.LockedMatrix().LockedBuffer();
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      // Note: This could be optimized; upper_bound takes O(logn) time.
      int bucket = std::upper_bound(
                     m_histogram_buckets.begin(), m_histogram_buckets.end(),
                     mat_buf[row + col * ldim]) - m_histogram_buckets.begin();
      buckets[bucket] += 1.0f;
    }
  }
  // Add to list of pending histograms.
  m_pending_histograms.emplace_back(
    tag, step, buckets, mat_local_min, mat_local_max, mat.Height() * mat.Width(),
    sum, sqsum);
  // TODO: Support histograms on multiple models.
}

void lbann_summary::reduce_2norm(const std::string tag, const AbsDistMat& mat,
                                 int step) {
  // Using a squared 2-norm so that we can just sum this.
  DataType local_norm = local_2norm(mat.LockedMatrix());
  sum_reduce_scalar(tag, local_norm * local_norm, step);
}

void lbann_summary::flush() {
  flush_means();
  flush_mins();
  flush_maxes();
  flush_stdevs();
  flush_scalars();
  flush_sum_scalars();
  flush_scalar_alls();
  flush_histograms();
  if (m_sw != nullptr) {
    m_sw->flush();
  }
}

void lbann_summary::flush_means() {
  if (m_pending_means.empty()) {
    return;
  }
  std::vector<DataType> local_sums;
  for (const auto& op : m_pending_means) {
    local_sums.push_back(op.local);
  }
  if (m_comm->am_model_master()) {
    std::vector<DataType> global_sums(local_sums.size());
    m_comm->model_reduce(local_sums.data(), local_sums.size(),
                         global_sums.data());
    // Compute the means in-place.
    for (unsigned i = 0; i < global_sums.size(); ++i) {
      global_sums[i] /= m_pending_means[i].num;
    }
    gather_scalar_summary(m_pending_means, global_sums);
  } else {
    m_comm->model_reduce(local_sums.data(), local_sums.size(),
                         m_comm->get_model_master());
  }
  m_pending_means.clear();
}

void lbann_summary::flush_mins() {
  if (m_pending_mins.empty()) {
    return;
  }
  std::vector<DataType> local_mins;
  for (const auto& op : m_pending_mins) {
    local_mins.push_back(op.local);
  }
  if (m_comm->am_model_master()) {
    std::vector<DataType> global_mins(local_mins.size());
    m_comm->model_reduce(local_mins.data(), local_mins.size(),
                         global_mins.data(), El::mpi::MIN);
    gather_scalar_summary(m_pending_mins, global_mins);
  } else {
    m_comm->model_reduce(local_mins.data(), local_mins.size(),
                         m_comm->get_model_master(), El::mpi::MIN);
  }
  m_pending_mins.clear();
}

void lbann_summary::flush_maxes() {
  if (m_pending_maxes.empty()) {
    return;
  }
  std::vector<DataType> local_maxes;
  for (const auto& op : m_pending_maxes) {
    local_maxes.push_back(op.local);
  }
  if (m_comm->am_model_master()) {
    std::vector<DataType> global_maxes(local_maxes.size());
    m_comm->model_reduce(local_maxes.data(), local_maxes.size(),
                         global_maxes.data(), El::mpi::MAX);
    gather_scalar_summary(m_pending_maxes, global_maxes);
  } else {
    m_comm->model_reduce(local_maxes.data(), local_maxes.size(),
                         m_comm->get_model_master(), El::mpi::MAX);
  }
  m_pending_maxes.clear();
}

void lbann_summary::flush_stdevs() {
  if (m_pending_stdevs.empty()) {
    return;
  }
  std::vector<DataType> local_sums;
  std::vector<DataType> local_sqsums;
  for (const auto& op : m_pending_stdevs) {
    local_sums.push_back(op.local);
    local_sqsums.push_back(op.local2);
  }
  if (m_comm->am_model_master()) {
    // Compute the model sample standard deviation as:
    // sqrt[1/(n-1) (sqsum - (1/n)*sum^2)]
    // The n-1 is to use an unbiased variance estimate.
    // This unrolls the usual formulation of standard deviation some, to avoid
    // global operations when pushing the operation.
    std::vector<DataType> global_sums(local_sums.size());
    std::vector<DataType> global_sqsums(local_sqsums.size());
    m_comm->model_reduce(local_sums.data(), local_sums.size(),
                         global_sums.data());
    m_comm->model_reduce(local_sqsums.data(), local_sqsums.size(),
                         global_sqsums.data());
    // Re-use the global_sums vector for the standard deviation.
    for (unsigned i = 0; i < global_sums.size(); ++i) {
      global_sums[i] = std::sqrt(
                         (global_sqsums[i] -
                          global_sums[i] * global_sums[i] / m_pending_stdevs[i].num) /
                         (m_pending_stdevs[i].num - 1));
    }
    gather_scalar_summary(m_pending_stdevs, global_sums);
  } else {
    m_comm->model_reduce(local_sums.data(), local_sums.size(),
                         m_comm->get_model_master());
    m_comm->model_reduce(local_sqsums.data(), local_sqsums.size(),
                         m_comm->get_model_master());
  }
  m_pending_stdevs.clear();
}

void lbann_summary::flush_scalars() {
  if (m_pending_scalars.empty()) {
    return;
  }
  if (m_comm->am_model_master()) {
    std::vector<DataType> local_scalars;
    for (const auto& op : m_pending_scalars) {
      local_scalars.push_back(op.local);
    }
    gather_scalar_summary(m_pending_scalars, local_scalars);
  }
  m_pending_scalars.clear();
}

void lbann_summary::flush_sum_scalars() {
  if (m_pending_sum_scalars.empty()) {
    return;
  }
  std::vector<DataType> local_sums;
  for (const auto& op : m_pending_sum_scalars) {
    local_sums.push_back(op.local);
  }
  if (m_comm->am_model_master()) {
    std::vector<DataType> global_sums(local_sums.size());
    m_comm->model_reduce(local_sums.data(), local_sums.size(),
                         global_sums.data());
    gather_scalar_summary(m_pending_sum_scalars, global_sums);
  } else {
    m_comm->model_reduce(local_sums.data(), local_sums.size(),
                         m_comm->get_model_master());
  }
  m_pending_sum_scalars.clear();
}

void lbann_summary::flush_scalar_alls() {
  if (m_pending_scalar_alls.empty()) {
    return;
  }
  // Gather from every process to world master.
  std::vector<DataType> local_scalars;
  for (const auto& op : m_pending_scalar_alls) {
    local_scalars.push_back(op.local);
  }
  if (m_comm->am_world_master()) {
    std::vector<DataType> scalars(
      m_comm->get_procs_in_world()*local_scalars.size());
    m_comm->gather(local_scalars.data(), local_scalars.size(),
                   scalars.data(), m_comm->get_world_comm());
    for (size_t i = 0; i < scalars.size(); ++i) {
      int rank = i / local_scalars.size();
      int model = rank / m_comm->get_procs_per_model();
      int pos = i % local_scalars.size();
      m_sw->add_scalar(
        prepend_model("rank" + std::to_string(rank) + "/" +
                      m_pending_scalar_alls[pos].tag, model),
        scalars[i], m_pending_scalar_alls[pos].step);
    }
  } else {
    m_comm->gather(local_scalars.data(), local_scalars.size(),
                   m_comm->get_world_master(), m_comm->get_world_comm());
  }
  m_pending_scalar_alls.clear();
}

void lbann_summary::flush_histograms() {
  if (m_pending_histograms.empty()) {
    return;
  }
  std::vector<DataType> local_mins;
  std::vector<DataType> local_maxes;
  std::vector<DataType> local_sums;
  std::vector<DataType> local_sqsums;
  std::vector<float> buckets;
  for (const auto& op : m_pending_histograms) {
    local_mins.push_back(op.min);
    local_maxes.push_back(op.max);
    local_sums.push_back(op.sum);
    local_sqsums.push_back(op.sqsum);
    buckets.insert(buckets.end(), op.buckets.begin(), op.buckets.end());
  }
  if (m_comm->am_model_master()) {
    std::vector<DataType> model_mins(local_mins.size());
    std::vector<DataType> model_maxes(local_maxes.size());
    std::vector<DataType> model_sums(local_sums.size());
    std::vector<DataType> model_sqsums(local_sqsums.size());
    std::vector<float> model_buckets(buckets.size());
    m_comm->model_reduce(local_mins.data(), local_mins.size(),
                         model_mins.data(), El::mpi::MIN);
    m_comm->model_reduce(local_maxes.data(), local_maxes.size(),
                         model_maxes.data(), El::mpi::MAX);
    m_comm->model_reduce(local_sums.data(), model_sums.size(),
                         model_sums.data());
    m_comm->model_reduce(local_sqsums.data(), local_sqsums.size(),
                         model_sqsums.data());
    m_comm->model_reduce(buckets.data(), buckets.size(),
                         model_buckets.data());
    // Gather to the world master for writing out.
    if (m_comm->am_world_master()) {
      std::vector<DataType> global_mins(
        m_comm->get_num_models() * model_mins.size());
      std::vector<DataType> global_maxes(
        m_comm->get_num_models() * model_maxes.size());
      std::vector<DataType> global_sums(
        m_comm->get_num_models() * model_sums.size());
      std::vector<DataType> global_sqsums(
        m_comm->get_num_models() * model_sqsums.size());
      std::vector<float> global_buckets(
        m_comm->get_num_models() * model_buckets.size());
      m_comm->intermodel_gather(model_mins.data(), model_mins.size(),
                                global_mins.data());
      m_comm->intermodel_gather(model_maxes.data(), model_maxes.size(),
                                global_maxes.data());
      m_comm->intermodel_gather(model_sums.data(), model_sums.size(),
                                global_sums.data());
      m_comm->intermodel_gather(model_sqsums.data(), model_sqsums.size(),
                                global_sqsums.data());
      m_comm->intermodel_gather(model_buckets.data(), model_buckets.size(),
                                global_buckets.data());
      for (unsigned i = 0; i < global_mins.size(); ++i) {
        int model = i / m_pending_histograms.size();
        unsigned ops_pos = i % m_pending_histograms.size();
        std::vector<float> tmp_buckets(
          global_buckets.begin() + i*m_histogram_buckets.size(),
          global_buckets.begin() + (i+1)*m_histogram_buckets.size());
        m_sw->add_histogram(prepend_model(m_pending_histograms[ops_pos].tag, model),
                            tmp_buckets, global_mins[i], global_maxes[i],
                            m_pending_histograms[ops_pos].num,
                            global_sums[i], global_sqsums[i],
                            m_pending_histograms[ops_pos].step);
      }
    } else {
      m_comm->intermodel_gather(model_mins.data(), model_mins.size(),
                                m_comm->get_intermodel_master());
      m_comm->intermodel_gather(model_maxes.data(), model_maxes.size(),
                                m_comm->get_intermodel_master());
      m_comm->intermodel_gather(model_sums.data(), model_sums.size(),
                                m_comm->get_intermodel_master());
      m_comm->intermodel_gather(model_sqsums.data(), model_sqsums.size(),
                                m_comm->get_intermodel_master());
      m_comm->intermodel_gather(model_buckets.data(), model_buckets.size(),
                                m_comm->get_intermodel_master());
    }
  } else {
    m_comm->model_reduce(local_mins.data(), local_mins.size(),
                         m_comm->get_model_master(), El::mpi::MIN);
    m_comm->model_reduce(local_maxes.data(), local_maxes.size(),
                         m_comm->get_model_master(), El::mpi::MAX);
    m_comm->model_reduce(local_sums.data(), local_sums.size(),
                         m_comm->get_model_master());
    m_comm->model_reduce(local_sqsums.data(), local_sqsums.size(),
                         m_comm->get_model_master());
    m_comm->model_reduce(buckets.data(), buckets.size(),
                         m_comm->get_model_master());
  }
  m_pending_histograms.clear();
}

DataType lbann_summary::local_sum(const Mat& mat) const {
  // Note there are more numerically stable ways to compute a sum.
  const El::Int height = mat.Height();
  const El::Int width = mat.Width();
  const El::Int ldim = mat.LDim();
  const DataType * __restrict__ mat_buf = mat.LockedBuffer();
  auto sum = DataType(0);
  if (ldim == height) {
    const El::Int size = height*width;
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(+:sum))
    for (El::Int i = 0; i < size; ++i) {
      //const int tid = omp_get_thread_num();
      sum += mat_buf[i];
    }
  } else {
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(+:sum) collapse(2))
    for (El::Int row = 0; row < height; ++row) {
      for (El::Int col = 0; col < width; ++col) {
        //const int tid = omp_get_thread_num();
        sum += mat_buf[row + col * ldim];
      }
    }
  }
  return sum;
}

void lbann_summary::local_sum_sqsum(
  const Mat& mat, DataType& sum, DataType& sqsum) const {
  // Note there are more numerically stable ways to compute a sum.
  const El::Int height = mat.Height();
  const El::Int width = mat.Width();
  const El::Int ldim = mat.LDim();
  const DataType * __restrict__ mat_buf = mat.LockedBuffer();
  sum = DataType(0);
  sqsum = DataType(0);
  if (ldim == height) {
    const El::Int size = height*width;
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(+:sum,sqsum))
    for (El::Int i = 0; i < size; ++i) {
      const DataType val = mat_buf[i];
      //const int tid = omp_get_thread_num();
      sum += val;
      sqsum += val*val;
    }
  } else {
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(+:sum,sqsum) collapse(2))
    for (El::Int row = 0; row < height; ++row) {
      for (El::Int col = 0; col < width; ++col) {
        const DataType val = mat_buf[row + col*ldim];
        //const int tid = omp_get_thread_num();
        sum += val;
        sqsum += val * val;
      }
    }
  }
}

DataType lbann_summary::local_min(const Mat& mat) const {
  const El::Int height = mat.Height();
  const El::Int width = mat.Width();
  const El::Int ldim = mat.LDim();
  const DataType * __restrict__ mat_buf = mat.LockedBuffer();
  auto min = std::numeric_limits<DataType>::max();
  if (ldim == height) {
    const El::Int size = height*width;
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(min:min))
    for (El::Int i = 0; i < size; ++i) {
      min = std::min(min, mat_buf[i]);
    }
  } else {
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(min:min) collapse(2))
    for (El::Int row = 0; row < height; ++row) {
      for (El::Int col = 0; col < width; ++col) {
        min = std::min(min, mat_buf[row + col*ldim]);
      }
    }
  }
  return min;
}

DataType lbann_summary::local_max(const Mat& mat) const {
  const El::Int height = mat.Height();
  const El::Int width = mat.Width();
  const El::Int ldim = mat.LDim();
  const DataType * __restrict__ mat_buf = mat.LockedBuffer();
  auto max = std::numeric_limits<DataType>::min();
  if (ldim == height) {
    const El::Int size = height*width;
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(max:max))
    for (El::Int i = 0; i < size; ++i) {
      max = std::max(max, mat_buf[i]);
    }
  } else {
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(max:max) collapse(2))
    for (El::Int row = 0; row < height; ++row) {
      for (El::Int col = 0; col < width; ++col) {
        max = std::max(max, mat_buf[row + col*ldim]);
      }
    }
  }
  return max;
}

DataType lbann_summary::local_2norm(const Mat& mat) const {
  // Note there are more numerically stable ways to compute this.
  const El::Int height = mat.Height();
  const El::Int width = mat.Width();
  const El::Int ldim = mat.LDim();
  const DataType * __restrict__ mat_buf = mat.LockedBuffer();
  auto norm = DataType(0);
  if (ldim == height) {
    const El::Int size = height*width;
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(+:norm))
    for (El::Int i = 0; i < size; ++i) {
      norm += mat_buf[i] * mat_buf[i];
    }
  } else {
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(+:norm) collapse(2))
    for (El::Int row = 0; row < height; ++row) {
      for (El::Int col = 0; col < width; ++col) {
        norm += mat_buf[row + col * ldim] * mat_buf[row + col * ldim];
      }
    }
  }
  return std::sqrt(norm);
}

std::string lbann_summary::prepend_model(const std::string tag,
                                         int model) const {
  return "model" + std::to_string(model) + "/" + tag;
}

void lbann_summary::gather_scalar_summary(
  const std::vector<pending_op>& ops, std::vector<DataType>& scalars) {
  if (m_comm->am_world_master()) {
    std::vector<DataType> data(m_comm->get_num_models() * scalars.size());
    m_comm->intermodel_gather(scalars.data(), scalars.size(), data.data());
    for (unsigned i = 0; i < data.size(); ++i) {
      int model = i / ops.size();
      unsigned ops_pos = i % ops.size();
      m_sw->add_scalar(prepend_model(ops[ops_pos].tag, model),
                       data[i], ops[ops_pos].step);
    }
  } else {
    m_comm->intermodel_gather(scalars.data(), scalars.size(),
                              m_comm->get_intermodel_master());
  }
}

void lbann_summary::gather_scalar_summary(const std::string tag,
                                          DataType s,
                                          int step) {
  if (m_comm->am_world_master()) {
    std::vector<DataType> data(m_comm->get_num_models());
    m_comm->intermodel_gather(s, data);
    for (size_t model = 0; model < data.size(); ++model) {
      m_sw->add_scalar(prepend_model(tag, model), data[model], step);
    }
  } else {
    m_comm->intermodel_gather(s, m_comm->get_intermodel_master());
  }
}

#endif  // LBANN_HAS_TBINF

}  // namespace lbann
