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

// Dense SVD.
class SVDHelperImpl {
public:
  SVDHelperImpl(const SMat &a, SMat *u, SMat *vt, SVec *s);
  virtual ~SVDHelperImpl() = default;
  virtual void Compute() = 0;

protected:
  const SMat *a_;
  SMat *u_;
  SMat *vt_;
  SVec *s_;
};

// CAUTION: We only support full SVD, not economic SVD! If you want to do
// economic SVD, please do QR first and then run SVD on the square R matrix.
// NOTE: For max speed, you should create this and reuse in each iteration.
class SVDHelper {
public:
  SVDHelper(const SMat &a, SMat *u, SMat *vt, SVec *s);
  void Compute();

private:
  unique_ptr<SVDHelperImpl> impl_;
};

// If memory is limited, you might want to create a new SVDHelper each time you
// need to perform SVD, so that workspace memory is returned.
void SVDCompute(const SMat &a, SMat *u, SMat *vt, SVec *s);

} // namespace gi