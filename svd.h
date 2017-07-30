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