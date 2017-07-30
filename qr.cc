#include "qr.h"

namespace gi {

namespace {

class QRHelperMagmaImpl : public QRHelperImpl {
public:
  QRHelperMagmaImpl(SMat *a, int magma_qr = kDefaultMagmaQR)
      : QRHelperImpl(a), magma_qr_(magma_qr) {
    const int n = a->n();
    int dwork_size;
    switch (magma_qr_) {
    case 1:
      dwork_size = n * n;
      // dwork_size = 3 * n * n + n + 2;
      break;
    case 2:
      dwork_size = 3 * n * n + n + 2;
      break;
    case 3:
      dwork_size = 1; // Not used.
      break;
    case 4:
      dwork_size = n * n;
      break;
    default:
      LOG(FATAL) << "Unknown qr ikind";
    }
    dwork_.reset(new SVec(dwork_size, MEM_DEVICE));
    work_.reset(new SVec(3 * n * n, MEM_HOST));
  }

  ~QRHelperMagmaImpl() override = default;

  void Compute() {
    magma_int_t info;
    magma_sgegqr_gpu(magma_qr_, a_->m(), a_->n(), a_->data(), a_->lda(),
                     dwork_->data(), work_->data(), &info);
    CHECK_EQ(0, info) << info;
  }

private:
  unique_ptr<SVec> dwork_;
  unique_ptr<SVec> work_;
  int magma_qr_;
};

typedef Eigen::HouseholderQR<Eigen::MatrixXf> EigenQR;

class QRHelperEigenImpl : public QRHelperImpl {
public:
  QRHelperEigenImpl(SMat *a) : QRHelperImpl(a) {
    eigen_qr_.reset(new EigenQR(a->m(), a->n()));
  }

  ~QRHelperEigenImpl() override = default;

  void Compute() {
    eigen_qr_->compute(*(a_->mat_map()));
    *(a_->mat_map()) =
        eigen_qr_->householderQ() * Eigen::MatrixXf::Identity(a_->m(), a_->n());
  }

private:
  unique_ptr<EigenQR> eigen_qr_;
};

} // namespace

QRHelperImpl::QRHelperImpl(SMat *a) : a_(a) {
  CHECK_GE(a->m(), a->n()) << "We do not expect fat matrices: " << a->m() << " "
                           << a->n();
}

QRHelper::QRHelper(SMat *a, int magma_qr) {
  if (a->mem_type() == MEM_DEVICE) {
    CHECK_GE(magma_qr, 1);
    CHECK_LE(magma_qr, 4);
    impl_.reset(new QRHelperMagmaImpl(a, magma_qr));
  } else {
    impl_.reset(new QRHelperEigenImpl(a));
  }
}

void QRHelper::Compute() { impl_->Compute(); }

void QRCompute(SMat *a) {
  QRHelper qr_helper(a);
  qr_helper.Compute();
}

} // namespace gi