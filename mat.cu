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
#include "mat.h"

namespace gi {

// Max number of blocks for SampleAndUpdateKernel.
constexpr int kSampleAndUpdateNumBlocks = (1 << 12);

template<int block_size>
__device__ void WarpReduce(volatile float* buf, int tid) {
  if (block_size >= 64) {
    buf[tid] += buf[tid + 32];
  }
  if (block_size >= 32) {
    buf[tid] += buf[tid + 16];
  }
  if (block_size >= 16) {
    buf[tid] += buf[tid + 8];
  }
  if (block_size >= 8) {
    buf[tid] += buf[tid + 4];
  }
  if (block_size >= 4) {
    buf[tid] += buf[tid + 2];
  }
  if (block_size >= 2) {
    buf[tid] += buf[tid + 1];
  }
}

template<int block_size>
__global__
void SampleAndUpdateKernel(const int nnz,
                           const int* __restrict__ d_row,
                           const int* __restrict__ d_col,
                           float* d_value,
                           const float* __restrict__ d_ut,
                           const int lda_ut,
                           const float* __restrict__ d_vt,
                           const int lda_vt,
                           const float* __restrict__ d_s,
                           float alpha, float beta) {
  extern __shared__ float buf[];
  const int tid = threadIdx.x;

  for (int i = blockIdx.x; i < nnz; i += gridDim.x) {
    // i is the inner index and 0 <= i < nnz.
    const int row = d_row[i];
    const int col = d_col[i];
    buf[tid] = d_ut[tid + row * lda_ut] * d_vt[tid + col * lda_vt] * d_s[tid];
    __syncthreads();

    // Assume block size never exceeds 1024.
    if (block_size >= 512) {
      if (tid < 256) {
        buf[tid] += buf[tid + 256];
        __syncthreads();
      }
    }
    if (block_size >= 256) {
      if (tid < 128) {
        buf[tid] += buf[tid + 128];
        __syncthreads();
      }
    }
    if (block_size >= 128) {
      if (tid < 64) {
        buf[tid] += buf[tid + 64];
        __syncthreads();
      }
    }
    if (tid < 32) {
      WarpReduce<block_size>(buf, tid);
    }
    if (tid == 0) {
      d_value[i] = beta * d_value[i] + alpha * buf[0];
    }
  }
}

// Need to perform nnz inner products. Each inner product includes k
// multiplications. Each block has k threads. Assume k is a small power of 2.
void DeviceSampleAndUpdateHelper(
    const IVec& row, const IVec& col, const SVec& value,
    float alpha, const SMat& ut, const SMat& vt, const SVec& s, float beta) {
  const int nnz = value.size();
  const int k = s.size();
  const int num_blocks = std::min(kSampleAndUpdateNumBlocks, nnz);
  switch (k) {
  case 512:
    SampleAndUpdateKernel<512><<<num_blocks, k, k>>>(
        nnz, row.data(), col.data(), value.data(),
        ut.data(), ut.lda(), vt.data(), vt.lda(), s.data(), alpha, beta);
    break;
  case 256:
    SampleAndUpdateKernel<256><<<num_blocks, k, k>>>(
        nnz, row.data(), col.data(), value.data(),
        ut.data(), ut.lda(), vt.data(), vt.lda(), s.data(), alpha, beta);
    break;
  case 128:
    SampleAndUpdateKernel<128><<<num_blocks, k, k>>>(
        nnz, row.data(), col.data(), value.data(),
        ut.data(), ut.lda(), vt.data(), vt.lda(), s.data(), alpha, beta);
    break;
  case 64:
    SampleAndUpdateKernel<64><<<num_blocks, k, k>>>(
        nnz, row.data(), col.data(), value.data(),
        ut.data(), ut.lda(), vt.data(), vt.lda(), s.data(), alpha, beta);
    break;
  case 32:
    SampleAndUpdateKernel<32><<<num_blocks, k, k>>>(
        nnz, row.data(), col.data(), value.data(),
        ut.data(), ut.lda(), vt.data(), vt.lda(), s.data(), alpha, beta);
    break;
  case 16:
    SampleAndUpdateKernel<16><<<num_blocks, k, k>>>(
        nnz, row.data(), col.data(), value.data(),
        ut.data(), ut.lda(), vt.data(), vt.lda(), s.data(), alpha, beta);
    break;
  case 8:
    SampleAndUpdateKernel<8><<<num_blocks, k, k>>>(
        nnz, row.data(), col.data(), value.data(),
        ut.data(), ut.lda(), vt.data(), vt.lda(), s.data(), alpha, beta);
    break;
  case 4:
    SampleAndUpdateKernel<4><<<num_blocks, k, k>>>(
        nnz, row.data(), col.data(), value.data(),
        ut.data(), ut.lda(), vt.data(), vt.lda(), s.data(), alpha, beta);
    break;
  case 2:
    SampleAndUpdateKernel<2><<<num_blocks, k, k>>>(
        nnz, row.data(), col.data(), value.data(),
        ut.data(), ut.lda(), vt.data(), vt.lda(), s.data(), alpha, beta);
    break;
  case 1:
    SampleAndUpdateKernel<1><<<num_blocks, k, k>>>(
        nnz, row.data(), col.data(), value.data(),
        ut.data(), ut.lda(), vt.data(), vt.lda(), s.data(), alpha, beta);
    break;
  default:
    LOG(FATAL) <<
        "Device SampleAndUpdate expects k to be a lower power of 2.";
  }
}

}  // namespace gi