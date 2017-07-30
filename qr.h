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
#include "mat.h"
#include "vec.h"

namespace gi {

constexpr int kDefaultMagmaQR = 1;

class QRHelperImpl {
public:
  QRHelperImpl(SMat *a);
  virtual ~QRHelperImpl() = default;
  virtual void Compute() = 0;

protected:
  SMat *a_;
};

class QRHelper {
public:
  // magma_qr is ignored if a is on host memory.
  QRHelper(SMat *a, int magma_qr = kDefaultMagmaQR);
  void Compute();

private:
  unique_ptr<QRHelperImpl> impl_;
};

// Creates QRHelper and calls Compute. Convenient.
void QRCompute(SMat *a);

} // namespace gi