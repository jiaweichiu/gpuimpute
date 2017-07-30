/*
Copyright 2017 Jiawei Chiu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>
#include <cusparse.h>

// When BLAS functions are missing, we fallback on Eigen.
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SparseCore>

#include <cublas_v2.h>
// #include <magma_lapack.h>
#include <magma_v2.h>

#include <lapacke.h>

namespace gi {

using std::abs;
using std::cout;
using std::ifstream;
using std::istream;
using std::istringstream;
using std::max;
using std::min;
using std::ostream;
using std::ofstream;
using std::ostringstream;
using std::sqrt;
using std::string;
using std::unique_ptr;
using std::vector;

void MainInit(int argc, char **argv);

class Timer {
public:
  Timer();
  void reset();
  double elapsed() const;

private:
  typedef std::chrono::high_resolution_clock clock_;
  typedef std::chrono::duration<double, std::ratio<1> > second_;
  std::chrono::time_point<clock_> beg_;
};

struct EngineOptions {
  int omp_num_threads = 1;
  int device = 0;      // For GPU device.
  int num_streams = 4; // Number of CUDA streams.
  int rng_seed = 56150941;
};

#define CUDA_CALL(x)                                                           \
  { CHECK_EQ(x, cudaSuccess) << "CUDA error: " << x; }

#define CURAND_CALL(x)                                                         \
  { CHECK_EQ(x, CURAND_STATUS_SUCCESS) << "CURAND error: " << x; }

#define CUBLAS_CALL(x)                                                         \
  { CHECK_EQ(x, CUBLAS_STATUS_SUCCESS) << "CUBLAS error: " << x; }

#define CUSOLVER_CALL(x)                                                       \
  { CHECK_EQ(x, CUSOLVER_STATUS_SUCCESS) << "CUSOLVER error: " << x; }

#define CUSPARSE_CALL(x)                                                       \
  { CHECK_EQ(x, CUSPARSE_STATUS_SUCCESS) << "CUSPARSE error: " << x; }

class Engine {
public:
  Engine(const EngineOptions &opt);
  ~Engine();

  static Engine *instance() { return instance_; }
  static cudaStream_t stream(int i) { return instance()->stream_[i]; }
  static curandGenerator_t curand() { return instance()->curand_; }
  static cublasHandle_t cublas() { return instance()->cublas_; }
  static cusparseHandle_t cusparse() { return instance()->cusparse_; }
  static cusparseMatDescr_t cusparse_desc() {
    return instance()->cusparse_desc_;
  }
  static cusolverDnHandle_t cusolver_dn() { return instance()->cusolver_dn_; }
  static std::mt19937 &rng() { return instance()->rng_; }
  static float *device_s_one() { return instance()->device_s_one_; }
  static float *device_s_zero() { return instance()->device_s_zero_; }

private:
  vector<cudaStream_t> stream_;
  curandGenerator_t curand_;
  cublasHandle_t cublas_;
  cusparseHandle_t cusparse_;
  cusparseMatDescr_t cusparse_desc_;
  cusolverDnHandle_t cusolver_dn_;
  std::mt19937 rng_;
  float *device_s_one_;
  float *device_s_zero_;

  static Engine *instance_;
};

// Some convenience functions for GPU kernels.

// Returns number of blocks and number of threads given n things to work on.
void BlocksThreads(int max_blocks, int max_threads, int n, int *num_blocks,
                   int *num_threads);
}