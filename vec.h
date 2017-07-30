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

#include "base.h"

namespace gi {

enum MemType {
  MEM_DEVICE,
  MEM_HOST,
};

template <class T> class Vec {
public:
  // owned=true.
  Vec(int size, MemType mem_type)
      : size_(size), mem_type_(mem_type), owned_(true) {
    if (mem_type_ == MEM_DEVICE) {
      CUDA_CALL(cudaMalloc(&data_, sizeof(T) * size_));
    } else {
      CUDA_CALL(cudaMallocHost(&data_, sizeof(T) * size_));
    }
  }

  // owned=true.
  Vec(const Vec<T> &src, MemType mem_type) : Vec<T>(src.size(), mem_type) {
    CopyFrom(src);
  }

  // owned=false. Just wrap around the data.
  Vec(int size, T *data, MemType mem_type)
      : size_(size), mem_type_(mem_type), data_(data), owned_(false) {}

  virtual ~Vec() {
    if (!owned_) {
      return;
    }
    if (mem_type_ == MEM_DEVICE) {
      CUDA_CALL(cudaFree(data_));
    } else {
      CUDA_CALL(cudaFreeHost(data_));
    }
  }

  // Getters, setters.
  void set(int i, T x) { data_[i] = x; }
  T get(int i) const { return data_[i]; }
  int size() const { return size_; }
  MemType mem_type() const { return mem_type_; }
  T *data() const { return data_; }

  // Some common operations.
  virtual void Clear() {
    if (mem_type() == MEM_DEVICE) {
      CUDA_CALL(cudaMemset(data(), 0, size() * sizeof(float)));
    } else {
      CHECK(memset(data(), 0, size() * sizeof(T)));
    }
  }

  void CopyFrom(const Vec<T> &src) {
    if (mem_type_ == MEM_DEVICE) {
      if (src.mem_type() == MEM_DEVICE) {
        CUDA_CALL(cudaMemcpy(data_, src.data(), size_ * sizeof(T),
                             cudaMemcpyDeviceToDevice));
      } else {
        CUDA_CALL(cudaMemcpy(data_, src.data(), size_ * sizeof(T),
                             cudaMemcpyHostToDevice));
      }
    } else {
      if (src.mem_type() == MEM_DEVICE) {
        CUDA_CALL(cudaMemcpy(data_, src.data(), size_ * sizeof(T),
                             cudaMemcpyDeviceToHost));
      } else {
        CHECK(memcpy(data_, src.data(), size_ * sizeof(T)));
      }
    }
  }

  virtual void Read(istream &is) {
    for (int i = 0; i < size_; ++i) {
      CHECK(is >> data_[i]);
    }
  }

  virtual void Write(ostream &os) const {
    CHECK_EQ(mem_type_, MEM_HOST);
    for (int i = 0; i < size_; ++i) {
      os << data_[i] << " ";
    }
  }

  string DebugString() const {
    ostringstream os;
    Write(os);
    return os.str();
  }

private:
  int size_;
  MemType mem_type_;
  T *data_;
  bool owned_; // Do we own data_? By default, true.
};

template <class T> istream &operator>>(istream &is, Vec<T> &x) {
  x.Read(is);
  return is;
}

template <class T> ostream &operator<<(ostream &os, const Vec<T> &x) {
  x.Write(os);
  return os;
}

typedef Vec<int> IVec;

typedef Eigen::Map<Eigen::VectorXf> SEigenVecMap;

// Vec<float> has some extra functionality.
class SVec : public Vec<float> {
public:
  SVec(int size, MemType mem_type);
  SVec(const SVec &src, MemType mem_type);
  SVec(int size, float *data, MemType mem_type);
  ~SVec() override = default;

  SEigenVecMap *vec_map() const { return vec_map_.get(); }

  // Operations.
  void RandNormal() { RandNormal(0, 1); }
  void RandNormal(float mean, float stddev);
  void RandUniform(); // Uniform between 0 and 1.
  // x[i] = value.
  void Fill(float value);
  // x[i] = alpha * a[i] + beta * b[i] + gamma * c[i];
  void SetToSum3(float alpha, const SVec &a, float beta, const SVec &b,
                 float gamma, const SVec &c);
  // x[i] *= alpha.
  void Multiply(float alpha);
  // x[i] = 1.0 / x[i].
  void Invert();
  float AbsSum() const;
  float Sum() const;
  float Dot(const SVec &v) const;
  float Mean() const;
  float MeanSquare() const;
  float AbsMean() const;
  float RootMeanSquare() const;

  // this[x] = src[perm[x]].
  void SetToPermute(const IVec &perm, const SVec &src);

  // Do thresholding on singular values.
  void SoftThreshold(float threshold);
  void HardThreshold(float threshold);

private:
  void DeviceFill(float value);
  void DeviceSetToSum3(float alpha, const SVec &a, float beta, const SVec &b,
                       float gamma, const SVec &c);
  void DeviceSetToPermute(const IVec &perm, const SVec &src);
  void DeviceSoftThreshold(float threshold);
  void DeviceHardThreshold(float threshold);
  void DeviceMultiply(float alpha);
  void DeviceInvert();
  void InitSEigenVecMap();

  unique_ptr<SEigenVecMap> vec_map_; // Eigen. Only for host, not device.
};

} // namespace gi