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
// data_reader_triplet .hpp .cpp - data reader class for triplet datasets
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_triplet.hpp"
#include "lbann/data_readers/image_utils.hpp"
#include "lbann/data_store/data_store_imagenet.hpp"
#include "lbann/utils/file_utils.hpp"
#include <fstream>
#include <sstream>
#include <omp.h>

namespace lbann {

data_reader_triplet::data_reader_triplet(const std::shared_ptr<cv_process>& pp, bool shuffle)
  : imagenet_reader(pp, shuffle) {
  set_defaults();
}

data_reader_triplet::data_reader_triplet(const data_reader_triplet& rhs)
  : imagenet_reader(rhs),
    m_num_img_srcs(rhs.m_num_img_srcs)
{}

data_reader_triplet& data_reader_triplet::operator=(const data_reader_triplet& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }

  imagenet_reader::operator=(rhs);
  m_num_img_srcs = rhs.m_num_img_srcs;

  return (*this);
}

data_reader_triplet::~data_reader_triplet() {
}

void data_reader_triplet::set_defaults() {
  m_image_width = 110;
  m_image_height = 110;
  m_image_num_channels = 3;
  set_linearized_image_size();
  m_num_labels = 20;
  m_num_img_srcs = 3;
}

void data_reader_triplet::set_input_params(const int width, const int height, const int num_ch, const int num_labels, const int num_img_srcs) {
  imagenet_reader::set_input_params(width, height, num_ch, num_labels);
  if (num_img_srcs > 0) {
    m_num_img_srcs = num_img_srcs;
  } else if (num_img_srcs < 0) {
    std::stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: " << get_type() << " setup error: invalid number of image sources";
    throw lbann_exception(err.str());
  }
}


std::vector<::Mat> data_reader_triplet::create_datum_views(::Mat& X, const int mb_idx) const {
  std::vector<::Mat> X_v(m_num_img_srcs);
  El::Int h = 0;
  for(unsigned int i=0u; i < m_num_img_srcs; ++i) {
    El::View(X_v[i], X, El::IR(h, h + m_image_linearized_size), El::IR(mb_idx, mb_idx + 1));
    h = h + m_image_linearized_size;
  }
  return X_v;
}


bool data_reader_triplet::fetch_datum(Mat& X, int data_id, int mb_idx, int tid) {

  std::vector<::Mat> X_v = create_datum_views(X, mb_idx);

  sample_t sample = m_samples.get_sample(data_id);
  for(size_t i=0u; i < m_num_img_srcs; ++i) {
    int width=0, height=0, img_type=0;
    const std::string imagepath = get_file_dir() + sample.first[i];
    bool ret = true;
    if (m_data_store != nullptr) {
      std::vector<unsigned char> *image_buf;
      m_data_store->get_data_buf(data_id, image_buf, tid);
      ret = lbann::image_utils::load_image(*image_buf, width, height, img_type, *(m_pps[tid]), X_v[i]);
    } else {
      ret = lbann::image_utils::load_image(imagepath, width, height, img_type, *(m_pps[tid]), X_v[i]);
    }
  
    if(!ret) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " "
                            + get_type() + ": image_utils::load_image failed to load - "
                            + imagepath);
    }
    if((width * height * CV_MAT_CN(img_type)) != m_image_linearized_size) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " "
                            + get_type() + ": mismatch data size -- either width, height or channel - "
                            + imagepath + " [w,h,c]=[" + std::to_string(width) + "x" + std::to_string(height)
                            + "x" + std::to_string(CV_MAT_CN(img_type)) + "] != " + std::to_string(m_image_linearized_size));
    }
  }
  return true;
}


bool data_reader_triplet::fetch_label(Mat& Y, int data_id, int mb_idx, int tid) {
  const label_t label = m_samples.get_label(data_id);
  Y.Set(label, mb_idx, 1);
  return true;
}


std::vector<data_reader_triplet::sample_t> data_reader_triplet::get_image_list_of_current_mb() const {
  std::vector<sample_t> ret;
  ret.reserve(m_mini_batch_size);

  for (El::Int i = 0; i < m_indices_fetched_per_mb.Height(); ++i) {
    El::Int index = m_indices_fetched_per_mb.Get(i, 0);
    ret.emplace_back(m_samples.get_sample(index));
  }
  return ret;
}


std::vector<data_reader_triplet::sample_t> data_reader_triplet::get_image_list() const {
  const size_t num_samples = m_samples.get_num_samples();
  std::vector<sample_t> ret;
  ret.reserve(num_samples);

  for (size_t i=0; i < num_samples; ++i) {
    ret.emplace_back(m_samples.get_sample(i));
  }
  return ret;
}


void data_reader_triplet::load() {
  const std::string data_filename = get_data_filename();
  if (!m_samples.load(data_filename)) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " "
                            + get_type() + ": failed to load the file " + data_filename);
  }

  // reset indices
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_samples.get_num_samples());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);

  select_subset_of_data();
}

}  // namespace lbann
