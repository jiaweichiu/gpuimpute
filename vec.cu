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
#include "vec.h"

namespace gi {

__global__
void SVecFillKernel(float* x, const int n, const float value) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    x[i] = value;
  }
}

void SVec::DeviceFill(float value) {
  int num_blocks;
  int num_threads;
  BlocksThreads(256, 256, size(), &num_blocks, &num_threads);
  SVecFillKernel<<<num_blocks, num_threads>>>(data(), size(), value);
}

__global__
void SVecSetToSum3Kernel(float* x, const int n,
                         const float alpha, const float* __restrict__ a,
                         const float beta, const float* __restrict__ b,
                         const float gamma, const float* __restrict__ c) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    x[i] = alpha * a[i] + beta * b[i] + gamma * c[i];
  }
}

void SVec::DeviceSetToSum3(float alpha, const SVec& a, float beta,
                           const SVec& b, float gamma, const SVec& c) {
  int num_blocks;
  int num_threads;
  BlocksThreads(256, 256, size(), &num_blocks, &num_threads);
  SVecSetToSum3Kernel<<<num_blocks, num_threads>>>(
      data(), size(), alpha, a.data(), beta, b.data(), gamma, c.data());
}

__global__
void SetToPermuteKernel(const float* __restrict__ input,
                        const int* __restrict__ perm,
                        float* output, const int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    output[i] = input[perm[i]];
  }
}

void SVec::DeviceSetToPermute(const IVec& perm, const SVec& src) {
  int num_blocks;
  int num_threads;
  BlocksThreads(256, 256, size(), &num_blocks, &num_threads);
  SetToPermuteKernel<<<num_blocks, num_threads>>>(
      src.data(), perm.data(), data(), size());
}

__global__
void SoftThresholdKernel(float* output, const int n, const float threshold) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    float v = output[i];
    if (v > threshold) {
      v -= threshold;
    } else if (v < -threshold) {
      v += threshold;
    } else {
      v = 0;
    }
    output[i] = v;
  }
}

void SVec::DeviceSoftThreshold(float threshold) {
  int num_blocks;
  int num_threads;
  BlocksThreads(256, 256, size(), &num_blocks, &num_threads);
  SoftThresholdKernel<<<num_blocks, num_threads>>>(data(), size(), threshold);
}

__global__
void HardThresholdKernel(float* output, const int n, const float threshold) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    float v = output[i];
    // Assume threshold is positive.
    if (v >= -threshold && v <= threshold) {
      output[i] = 0;
    }
  }
}

void SVec::DeviceHardThreshold(float threshold) {
  int num_blocks;
  int num_threads;
  BlocksThreads(256, 256, size(), &num_blocks, &num_threads);
  HardThresholdKernel<<<num_blocks, num_threads>>>(data(), size(), threshold);
}

__global__
void VecMultiplyKernel(float* output, const int n, const float alpha) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    output[i] *= alpha;
  }
}

void SVec::DeviceMultiply(float alpha) {
  int num_blocks;
  int num_threads;
  BlocksThreads(256, 256, size(), &num_blocks, &num_threads);
  VecMultiplyKernel<<<num_blocks, num_threads>>>(data(), size(), alpha);
}

__global__
void VecInvertKernel(float* output, const int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    output[i] = 1.0 / output[i];
  }
}

void SVec::DeviceInvert() {
  int num_blocks;
  int num_threads;
  BlocksThreads(256, 256, size(), &num_blocks, &num_threads);
  VecInvertKernel<<<num_blocks, num_threads>>>(data(), size());
}

}  // namespace gi