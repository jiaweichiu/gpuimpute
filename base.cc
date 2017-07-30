#include "base.h"

namespace gi {

void MainInit(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
}

Engine *Engine::instance_ = nullptr;

Engine::Engine(const EngineOptions &opt) {
  CHECK(!instance_) << "Duplicate instance";
  instance_ = this;

  CUDA_CALL(cudaSetDevice(opt.device));
  LOG(INFO) << "CUDA device " << opt.device << " ready";
  stream_.resize(opt.num_streams);
  for (int i = 0; i < opt.num_streams; ++i) {
    CUDA_CALL(cudaStreamCreate(&stream_[i]));
  }
  LOG(INFO) << "CUDA streams " << opt.num_streams << " ready";

  CURAND_CALL(curandCreateGenerator(&curand_, CURAND_RNG_PSEUDO_DEFAULT));
  LOG(INFO) << "CURAND rng seed: " << opt.rng_seed;
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(curand_, opt.rng_seed));

  CUBLAS_CALL(cublasCreate(&cublas_));

  CUSOLVER_CALL(cusolverDnCreate(&cusolver_dn_));

  CUSPARSE_CALL(cusparseCreate(&cusparse_));
  CUSPARSE_CALL(cusparseCreateMatDescr(&cusparse_desc_));
  cusparseSetMatType(cusparse_desc_, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(cusparse_desc_, CUSPARSE_INDEX_BASE_ZERO);
  LOG(INFO) << "CUDA ready";

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
  LOG(INFO) << "OMP num_threads: " << opt.omp_num_threads;

  CHECK_EQ(0, magma_init());
  LOG(INFO) << "Magma ready";
}

Engine::~Engine() {
  magma_finalize();
  LOG(INFO) << "Magma done";

  CUDA_CALL(cudaFree(device_s_one_));
  CUDA_CALL(cudaFree(device_s_zero_));

  for (cudaStream_t s : stream_) {
    CUDA_CALL(cudaStreamDestroy(s));
  }
  LOG(INFO) << "CUDA streams done";
  CURAND_CALL(curandDestroyGenerator(curand_));
  CUBLAS_CALL(cublasDestroy(cublas_));
  CUSOLVER_CALL(cusolverDnDestroy(cusolver_dn_));
  CUSPARSE_CALL(cusparseDestroy(cusparse_));
  LOG(INFO) << "CUDA done";
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