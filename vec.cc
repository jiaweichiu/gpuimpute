#include "base.h"
#include "vec.h"

namespace gi {

SVec::SVec(int size, MemType mem_type) : Vec<float>(size, mem_type) {
  InitSEigenVecMap();
}

SVec::SVec(const SVec &src, MemType mem_type) : Vec<float>(src, mem_type) {
  InitSEigenVecMap();
}

SVec::SVec(int size, float *data, MemType mem_type)
    : Vec<float>(size, data, mem_type) {
  InitSEigenVecMap();
}

void SVec::InitSEigenVecMap() {
  if (mem_type() == MEM_HOST) {
    vec_map_.reset(new SEigenVecMap(data(), size()));
  }
}

void SVec::RandNormal(float mean, float stddev) {
  if (mem_type() == MEM_DEVICE) {
    CURAND_CALL(
        curandGenerateNormal(Engine::curand(), data(), size(), mean, stddev));
  } else {
    std::normal_distribution<float> dist(mean, stddev);
    auto &rng = Engine::rng();
    for (int i = 0; i < size(); ++i) {
      data()[i] = dist(rng);
    }
  }
}

void SVec::RandUniform() {
  if (mem_type() == MEM_DEVICE) {
    CURAND_CALL(curandGenerateUniform(Engine::curand(), data(), size()));
  } else {
    std::uniform_real_distribution<float> dist;
    auto &rng = Engine::rng();
    for (int i = 0; i < size(); ++i) {
      data()[i] = dist(rng);
    }
  }
}

float SVec::AbsSum() const {
  float result;
  if (mem_type() == MEM_DEVICE) {
    CUBLAS_CALL(cublasSasum(Engine::cublas(), size(), data(), 1, &result));
  } else {
    result = vec_map_->cwiseAbs().sum();
  }
  return result;
}

float SVec::Sum() const {
  float result;
  if (mem_type() == MEM_DEVICE) {
    CUBLAS_CALL(cublasSdot(Engine::cublas(), size(), data(), 1,
                           Engine::device_s_one(), 0, &result));
  } else {
    result = vec_map_->sum();
  }
  return result;
}

float SVec::Dot(const SVec &v) const {
  float result;
  if (mem_type() == MEM_DEVICE) {
    CUBLAS_CALL(
        cublasSdot(Engine::cublas(), size(), data(), 1, v.data(), 1, &result));
  } else {
    // result = cblas_sdot(size(), data(), 1, v.data(), 1);
    result = vec_map_->dot(*(v.vec_map_));
  }
  return result;
}

float SVec::Mean() const { return Sum() / static_cast<float>(size()); }
float SVec::MeanSquare() const {
  return Dot(*this) / static_cast<float>(size());
}
float SVec::AbsMean() const { return AbsSum() / static_cast<float>(size()); }
float SVec::RootMeanSquare() const { return sqrt(MeanSquare()); }

void SVec::Fill(float value) {
  if (mem_type() == MEM_DEVICE) {
    DeviceFill(value);
  } else {
    vec_map_->setConstant(value);
  }
}

void SVec::SetToSum3(float alpha, const SVec &a, float beta, const SVec &b,
                     float gamma, const SVec &c) {
  if (mem_type() == MEM_DEVICE) {
    DeviceSetToSum3(alpha, a, beta, b, gamma, c);
  } else {
    *vec_map_ =
        alpha * *a.vec_map() + beta * *b.vec_map() + gamma * *c.vec_map();
  }
}

void SVec::Multiply(float alpha) {
  if (mem_type() == MEM_DEVICE) {
    DeviceMultiply(alpha);
  } else {
    *vec_map_ *= alpha;
  }
}

void SVec::Invert() {
  if (mem_type() == MEM_DEVICE) {
    DeviceInvert();
  } else {
    *vec_map_ = vec_map_->array().inverse();
  }
}

void SVec::SetToPermute(const IVec &perm, const SVec &src) {
  CHECK_EQ(perm.mem_type(), mem_type());
  CHECK_EQ(src.mem_type(), mem_type());

  const int n = size();
  CHECK_EQ(n, perm.size());
  CHECK_EQ(n, src.size());

  if (mem_type() == MEM_DEVICE) {
    DeviceSetToPermute(perm, src);
  } else {
    for (int i = 0; i < n; ++i) {
      data()[i] = src.data()[perm.get(i)];
    }
  }
}

void SVec::SoftThreshold(float threshold) {
  if (mem_type() == MEM_DEVICE) {
    DeviceSoftThreshold(threshold);
  } else {
    for (int i = 0; i < size(); ++i) {
      float v = data()[i];
      if (v > threshold) {
        v -= threshold;
      } else if (v < -threshold) {
        v += threshold;
      } else {
        v = 0;
      }
      data()[i] = v;
    }
  }
}

void SVec::HardThreshold(float threshold) {
  if (mem_type() == MEM_DEVICE) {
    DeviceHardThreshold(threshold);
  } else {
    for (int i = 0; i < size(); ++i) {
      float v = std::abs(data()[i]);
      if (v <= threshold) {
        data()[i] = 0;
      }
    }
  }
}

} // namespace gi