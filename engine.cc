#include "engine.h"

#include <omp.h>

#include "engine.h"

namespace gi {

Engine *Engine::instance_ = nullptr;

Engine::Engine(const EngineOptions &opt) {
  CUDA_CALL(cudaSetDevice(opt.device));
  stream_.resize(opt.num_streams);
  for (int i = 0; i < opt.num_streams; ++i) {
    CUDA_CALL(cudaStreamCreate(&stream_[i]));
  }

  CURAND_CALL(curandCreateGenerator(&curand_, CURAND_RNG_PSEUDO_DEFAULT));
  LOG(INFO) << "Rng seed: " << opt.rng_seed;
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(curand_, opt.rng_seed));

  CUBLAS_CALL(cublasCreate(&cublas_));

  CUSOLVER_CALL(cusolverDnCreate(&cusolver_dn_));

  CUSPARSE_CALL(cusparseCreate(&cusparse_));
  CUSPARSE_CALL(cusparseCreateMatDescr(&cusparse_desc_));
  cusparseSetMatType(cusparse_desc_, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(cusparse_desc_, CUSPARSE_INDEX_BASE_ZERO);

  magma_init();

  // Keep "one" in device memory.
  CUDA_CALL(cudaMalloc(&device_s_one_, sizeof(float)));
  float one = 1.0;
  CUDA_CALL(
      cudaMemcpy(device_s_one_, &one, sizeof(float), cudaMemcpyHostToDevice));

  // Keep "zero" in device memory.
  CUDA_CALL(cudaMalloc(&device_s_zero_, sizeof(float)));
  float zero = 0;
  CUDA_CALL(
      cudaMemcpy(device_s_zero_, &zero, sizeof(float), cudaMemcpyHostToDevice));

  // Not needed now. But in the future, if we use OpenMP, this will be useful.
  omp_set_num_threads(opt.omp_num_threads);
}

Engine::~Engine() {
  CUDA_CALL(cudaFree(device_s_one_));
  CUDA_CALL(cudaFree(device_s_zero_));

  magma_finalize();

  for (cudaStream_t s : stream_) {
    CUDA_CALL(cudaStreamDestroy(s));
  }
  CURAND_CALL(curandDestroyGenerator(curand_));
  CUBLAS_CALL(cublasDestroy(cublas_));
  CUSOLVER_CALL(cusolverDnDestroy(cusolver_dn_));
  CUSPARSE_CALL(cusparseDestroy(cusparse_));
}

void BlocksThreads(int max_blocks, int max_threads, int n, int *num_blocks,
                   int *num_threads) {
  if (n < max_threads) {
    *num_blocks = 1;
    *num_threads = n;
  }
  *num_blocks = min(max_blocks, (n + max_threads - 1) / max_threads);
  *num_threads = max_threads;
}

} // namespace gi