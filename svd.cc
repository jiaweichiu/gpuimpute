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
#include "svd.h"

namespace gi {

namespace {

class SVDHelperCudaImpl : public SVDHelperImpl {
public:
  SVDHelperCudaImpl(const SMat &a, SMat *u, SMat *vt, SVec *s)
      : SVDHelperImpl(a, u, vt, s) {
    CHECK_EQ(a.mem_type(), MEM_DEVICE);
    CUSOLVER_CALL(cusolverDnSgesvd_bufferSize(Engine::cusolver_dn(), a_->m(),
                                              a_->n(), &lwork_));
    work_.reset(new SVec(lwork_, MEM_DEVICE));
    dev_info_.reset(new IVec(1, MEM_DEVICE));
    host_info_.reset(new IVec(1, MEM_HOST));
  }

  ~SVDHelperCudaImpl() override = default;

  void Compute() override {
    // As of 2016Q1, cusolver does not support economic SVD yet.
    float rwork; // Needed only for complex types.
    CUSOLVER_CALL(cusolverDnSgesvd(
        Engine::cusolver_dn(), 'A', 'A', a_->m(), a_->n(), a_->data(),
        a_->lda(), s_->data(), u_->data(), u_->lda(), vt_->data(), vt_->lda(),
        work_->data(), lwork_, &rwork, dev_info_->data()));
    host_info_->CopyFrom(*dev_info_);
    CHECK_EQ(host_info_->get(0), 0);
  }

protected:
  int lwork_;
  unique_ptr<SVec> work_;
  unique_ptr<IVec> dev_info_;
  unique_ptr<IVec> host_info_;
};

class SVDHelperLapackImpl : public SVDHelperImpl {
public:
  SVDHelperLapackImpl(const SMat &a, SMat *u, SMat *vt, SVec *s)
      : SVDHelperImpl(a, u, vt, s) {
    CHECK_EQ(a.mem_type(), MEM_HOST);
    float work_query;
    lwork_ = -1;
    CHECK_EQ(0, LAPACKE_sgesvd_work(LAPACK_COL_MAJOR, 'A', 'A', a_->m(),
                                    a_->n(), a_->data(), a_->lda(), s_->data(),
                                    u_->data(), u_->lda(), vt_->data(),
                                    vt_->lda(), &work_query, lwork_));
    lwork_ = static_cast<int>(work_query);
    work_.reset(new SVec(lwork_, MEM_HOST));
  }

  ~SVDHelperLapackImpl() override = default;

  void Compute() override {
    CHECK_EQ(0, LAPACKE_sgesvd_work(LAPACK_COL_MAJOR, 'A', 'A', a_->m(),
                                    a_->n(), a_->data(), a_->lda(), s_->data(),
                                    u_->data(), u_->lda(), vt_->data(),
                                    vt_->lda(), work_->data(), lwork_));
  }

private:
  int lwork_;
  unique_ptr<SVec> work_;
};

// Uses SVDHelperLapackerImpl to do it in host and copy to device.
class SVDHelperDummyDeviceImpl : public SVDHelperImpl {
public:
  SVDHelperDummyDeviceImpl(const SMat &a, SMat *u, SMat *vt, SVec *s)
      : SVDHelperImpl(a, u, vt, s), h_a_(a.m(), a.n(), MEM_HOST),
        h_u_(u->m(), u->n(), MEM_HOST), h_vt_(vt->m(), vt->n(), MEM_HOST),
        h_s_(s->size(), MEM_HOST) {
    helper_.reset(new SVDHelperLapackImpl(h_a_, &h_u_, &h_vt_, &h_s_));
  }

  ~SVDHelperDummyDeviceImpl() override = default;

  void Compute() override {
    h_a_.CopyFrom(*a_);
    helper_->Compute();
    u_->CopyFrom(h_u_);
    vt_->CopyFrom(h_vt_);
    s_->CopyFrom(h_s_);
  }

private:
  unique_ptr<SVDHelperImpl> helper_;
  SMat h_a_;
  SMat h_u_;
  SMat h_vt_;
  SVec h_s_;
};

} // namespace

SVDHelperImpl::SVDHelperImpl(const SMat &a, SMat *u, SMat *vt, SVec *s)
    : a_(&a), u_(u), vt_(vt), s_(s) {
  CHECK_EQ(a.mem_type(), u->mem_type());
  CHECK_EQ(a.mem_type(), vt->mem_type());
  CHECK_EQ(a.mem_type(), s->mem_type());

  // Dimensions check. We support only full SVD, not economic SVD.
  CHECK_EQ(std::min(a.m(), a.n()), s->size());
  CHECK_EQ(u->m(), u->n());   // Square u.
  CHECK_EQ(vt->m(), vt->n()); // Square vt.
  CHECK_EQ(u->m(), a.m());
  CHECK_EQ(vt->m(), a.n());
}

SVDHelper::SVDHelper(const SMat &a, SMat *u, SMat *vt, SVec *s) {
  if (a.mem_type() == MEM_DEVICE) {
    impl_.reset(new SVDHelperDummyDeviceImpl(a, u, vt, s));
    // impl_.reset(new SVDHelperCudaImpl(a, u, vt, s));
  } else {
    impl_.reset(new SVDHelperLapackImpl(a, u, vt, s));
  }
}

void SVDHelper::Compute() { impl_->Compute(); }

void SVDCompute(const SMat &a, SMat *u, SMat *vt, SVec *s) {
  SVDHelper svd_helper(a, u, vt, s);
  svd_helper.Compute();
}

} // namespace gi