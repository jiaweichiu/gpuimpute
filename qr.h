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