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

#ifndef CUBLAS_WRAPPER_HPP_INCLUDED
#define CUBLAS_WRAPPER_HPP_INCLUDED

#include "lbann/base.hpp"

#ifdef LBANN_HAS_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Error checking macro
#define FORCE_CHECK_CUBLAS(cublas_call)                         \
  do {                                                          \
    const cublasStatus_t cublas_status = cublas_call;           \
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {               \
      cudaDeviceReset();                                        \
      std::stringstream err;                                    \
      err << __FILE__ << " " << __LINE__ << " :: "              \
          << lbann::cublas::get_error_string(cublas_status);    \
      throw lbann::lbann_exception(err.str());                  \
    }                                                           \
  } while (0)
#ifdef LBANN_DEBUG
#define CHECK_CUBLAS(cublas_call) FORCE_CHECK_CUBLAS(cublas_call)
#else
#define CHECK_CUBLAS(cublas_call) cublas_call
#endif // LBANN_DEBUG

namespace lbann {
namespace cublas {

/** Get string for cuBLAS error. */
const std::string get_error_string(cublasStatus_t status);

// BLAS Level-1 functions
void axpy(cublasHandle_t const& handle,
          int n,
          DataType alpha,
          DataType const* x, int incx,
          DataType * y, int incy);
void dot(cublasHandle_t const& handle,
         int n,
         DataType const* x, int incx,
         DataType const* y, int incy,
         DataType * result);
DataType dot(cublasHandle_t const& handle,
             int n,
             DataType const* x, int incx,
             DataType const* y, int incy);
void nrm2(cublasHandle_t const& handle,
          int n,
          DataType const* x, int incx,
          DataType * result);
DataType nrm2(cublasHandle_t const& handle,
              int n,
              DataType const* x, int incx);
void scal(cublasHandle_t const& handle,
          int n,
          DataType alpha,
          DataType * x, int incx);

// BLAS Level-2 functions
void gemv(cublasHandle_t const& handle,
          cublasOperation_t trans,
          int m, int n,
          DataType alpha,
          DataType const * A, int lda,
          DataType const * x, int incx,
          DataType beta,
          DataType * y, int iny);

// BLAS Level-3 functions
void gemm(cublasHandle_t const& handle,
          cublasOperation_t transa, cublasOperation_t transb,
          int m, int n, int k,
          DataType alpha,
          DataType const * A, int lda,
          DataType const * B, int ldb,
          DataType beta,
          DataType * C, int ldc);

// BLAS-like extension
void geam(cublasHandle_t const& handle,
          cublasOperation_t transa, cublasOperation_t transb,
          int m, int n,
          DataType alpha,
          DataType const * A, int lda,
          DataType beta,
          DataType const * B, int ldb,
          DataType * C, int ldc);

} // namespace cublas
} // namespace lbann

#endif // LBANN_HAS_CUDA
#endif // CUBLAS_WRAPPER_HPP_INCLUDED
