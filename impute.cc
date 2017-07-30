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
#include "impute.h"
#include "base.h"
#include "mat.h"
#include "qr.h"
#include "svd.h"
#include "vec.h"

namespace gi {

Impute::Impute(const ImputeOptions &opt) : opt_(opt) {
  CHECK_GT(opt.sv_threshold, 0);
  CHECK(!opt.output_filename.empty());
  CHECK(!opt.train_filename.empty());
  CHECK(!opt.train_t_filename.empty());
  CHECK(!opt.test_filename.empty());
  CHECK(!opt.train_perm_filename.empty());

  mem_type_ = opt.use_gpu ? MEM_DEVICE : MEM_HOST;
  a_train_.reset(SSpMat::ReadInput(opt.train_filename, mem_type_));
  a_train_t_.reset(SSpMat::ReadInput(opt.train_t_filename, mem_type_));
  a_test_.reset(SSpMat::ReadInput(opt.test_filename, mem_type_));
  {
    ifstream fin(opt.train_perm_filename);
    IVec v(a_train_->nnz(), MEM_HOST);
    fin >> v;
    train_perm_.reset(new IVec(v, mem_type_));
  }

  const int m = a_train_->m();
  const int n = a_train_->n();
  CHECK_EQ(m, a_test_->m());
  CHECK_EQ(n, a_test_->n());

  LOG(INFO) << "Read: " << opt.train_filename;
  LOG(INFO) << "Number of nonzero elements: " << a_train_->nnz();
  LOG(INFO) << "use_gpu=" << opt.use_gpu;
  LOG(INFO) << "randomize_init=" << opt_.randomize_init;
}

void Impute::Run() {
  if (opt_.accelerated) {
    RunAccelerated();
  }
  RunNormal();
}

void Impute::ResetTestMatrix(int iter, const SMat &vt, SMat *out) {
  if ((iter % opt_.randn_iters) == 0) {
    out->RandNormal(0, 1);
  } else {
    out->CopyFrom(vt);
  }
}

void Impute::RunNormal() {
  const int m = a_train_->m();
  const int n = a_train_->n();
  CHECK_EQ(m, a_test_->m());
  CHECK_EQ(n, a_test_->n());
  const int k = opt_.k;

  // Extra variables.
  SMat ut(k, m, mem_type_); // ut.
  SMat vt(k, n, mem_type_); // vt.
  SVec s(k, mem_type_);     // Singular values.
  SMat mk(m, k, mem_type_);
  SMat q(m, k, mem_type_);
  SMat qt(k, m, mem_type_);
  SMat nk(n, k, mem_type_);
  SMat nk2(n, k, mem_type_);
  SMat kn(k, n, mem_type_);
  SMat kk(k, k, mem_type_);
  SMat kk2(k, k, mem_type_);
  SMat r(k, k, mem_type_);
  SVec h_s(k, MEM_HOST);

  QRHelper mk_qr(&mk);

  // Currently, cusolver doesn't support economic SVD, so we considered using
  // CULA's svd. However, we suspect this procedure is using CPU a lot and doing
  // a lot of HtoD and DtoH transfers. To mitigate this, we do a QR first.
  // nk is essentially A.T * Q.
  // Say nk = Q' R' where Q' is n by k and R' is k by k.
  // Run SVD on R' to get R' = U' S' V'.T

  // ========== Method 1 ==========
  // SVDHelper svd_helper(nk, &nk2, &kk, &s);
  // ========== Method 2 ==========
  SVDHelper svd_helper(r, &kk2, &kk, &s);
  QRHelper nk2_qr(&nk2);
  QRHelper nk_qr(&nk);

  // ========== End ==========
  if (!opt_.randomize_init) {
    // Zero initialization.
    ut.Clear();
    vt.Clear();
    s.Clear();
  } else {
    // RandUniform initialization.
    ut.RandUniform();
    vt.RandUniform();
    ut.SetToProduct(1.0 / static_cast<float>(k), ut);
    vt.SetToProduct(1.0 / static_cast<float>(k), vt);
    s.Fill(1.0);
  }

  // Backup sparse matrix values.
  SVec a_train_value(*a_train_->value(), mem_type_);
  SVec a_test_value(*a_test_->value(), mem_type_);

  // Collect rmse and timing info over iterations.
  Timer timer;
  vector<int> l_iter;
  vector<float> rmse;
  vector<float> timing;
  vector<float> s_min; // Smallest singular value. Want this to be zero.
  float time_elapsed = 0;

  for (int iter = 0;; ++iter) {
    if ((iter % opt_.log_every_n) == 0) {
      cudaDeviceSynchronize();

      // Get time elapsed.
      time_elapsed += timer.elapsed();
      timing.push_back(time_elapsed);
      l_iter.push_back(iter);

      // Compute error.
      a_test_->CopyValuesFrom(a_test_value);
      a_test_->SampleAndUpdate(-1.0, ut, vt, s, 1.0);
      rmse.push_back(a_test_->value()->RootMeanSquare());

      h_s.CopyFrom(s);
      s_min.push_back(h_s.get(k - 1));

      // Output.
      LOG(INFO) << "Iter: " << l_iter.back();
      LOG(INFO) << " SMin: " << s_min.back();
      LOG(INFO) << " Time elapsed: " << timing.back() << " sec";
      LOG(INFO) << " RMSE: " << rmse.back() << "\n";

      // Reset timer.
      timer.reset();

      // TODO: We could add a postprocessing step here.
      if (timing.back() > opt_.max_time) {
        break;
      }
    }

    a_train_->CopyValuesFrom(a_train_value);

    // Decrement sparse matrix by sampling u*diag(s)*v.T on its support.
    a_train_->SampleAndUpdate(-1.0, ut, vt, s, 1.0);

    // Update transpose of train matrix by permuting.
    a_train_t_->value()->SetToPermute(*train_perm_, *(a_train_->value()));

    // Create random matrix.
    kn.RandNormal(0, 1);

    // Apply U.diag(S).VT to random matrix.
    // Apply sparse matrix. We apply to random matrix transposed because that is
    // about three times faster than the random matrix not-transposed!
    kk.SetToProduct(1.0, NO_TRANS, vt, TRANS, kn, 0);
    kk.SetToProduct(s, kk);
    mk.SetToProduct(1.0, TRANS, ut, NO_TRANS, kk, 0);
    mk.SetToProduct(1.0, NO_TRANS, *a_train_, TRANS, kn, 1.0);

    for (int i = 0; i < opt_.num_gram; ++i) {
      // QRCompute(&mk);
      mk_qr.Compute();

      kk.SetToProduct(1.0, NO_TRANS, ut, NO_TRANS, mk, 0);
      kk.SetToProduct(s, kk);
      nk.SetToProduct(1.0, TRANS, vt, NO_TRANS, kk, 0);
      nk.SetToProduct(1.0, NO_TRANS, *a_train_t_, NO_TRANS, mk, 1.0);

      // Re-orthonormalize.
      nk_qr.Compute();

      kk.SetToProduct(1.0, NO_TRANS, vt, NO_TRANS, nk, 0);
      kk.SetToProduct(s, kk);
      mk.SetToProduct(1.0, TRANS, ut, NO_TRANS, kk, 0);
      mk.SetToProduct(1.0, NO_TRANS, *a_train_, NO_TRANS, nk, 1.0);
    }

    // Do QR. Transpose the result to get k by m matrix.
    // QRCompute(&mk);
    mk_qr.Compute();
    // Sparse*transpose of matrix is much faster than sparse*matrix.
    // So we prefer working with q transpose instead of q.
    qt.SetToSum(1.0, TRANS, mk, 0, NO_TRANS, qt);

    // Compute A.transpose to Q where A = sparse + sampled svd matrix.
    kk.SetToProduct(1.0, NO_TRANS, ut, TRANS, qt, 0);
    kk.SetToProduct(s, kk);
    nk.SetToProduct(1.0, TRANS, vt, NO_TRANS, kk, 0);
    nk.SetToProduct(1.0, NO_TRANS, *a_train_t_, TRANS, qt, 1.0);

    // ========== Method 1 ==========
    // A.T Q = U1 S1 V1.T
    // A ~ Q Q.T A = (Q V1) S1 U1.T.
    // If A ~ U S V.T, then:
    //   V.T = U1.T.
    //   U.T = V1.T Q.T.
    /*svd_helper.Compute();
    vt.SetToSum(1.0, TRANS, nk2, 0, NO_TRANS, vt);
    ut.SetToProduct(1.0, NO_TRANS, kk, NO_TRANS, qt, 0);*/
    // ========== Method 2 ==========
    // Run SVD.
    // (You might want to disable this to track the gaps in GPU usage.)

    // SVD(r) = kk2 * diag(s) ** kk.
    // SVD(nk) = U1 S1 V1.T = nk2 * kk2 * diag(s) ** kk
    // So U1 = nk2 * kk2 and V1.T = kk.
    // V.T = U1.T = kk2.T * nk2.T.
    // U.T = V1.T Q.T = kk * qt.
    nk2.CopyFrom(nk);
    nk2_qr.Compute(); // nk2 = QR(nk)
    r.SetToProduct(1.0, TRANS, nk2, NO_TRANS, nk, 0);
    svd_helper.Compute();
    vt.SetToProduct(1.0, TRANS, kk2, TRANS, nk2, 0);
    ut.SetToProduct(1.0, NO_TRANS, kk, NO_TRANS, qt, 0);
    // ========== End ==========

    if (opt_.soft_threshold) {
      s.SoftThreshold(opt_.sv_threshold);
    } else {
      s.HardThreshold(opt_.sv_threshold);
    }
  }
  LOG(INFO) << "Total time taken: " << time_elapsed << "\n";

  CHECK_EQ(timing.size(), rmse.size());
  CHECK_EQ(timing.size(), l_iter.size());
  ofstream os(opt_.output_filename);
  os << "iter\tsmin\ttime\trmse\n";
  for (size_t i = 0; i < timing.size(); ++i) {
    os << l_iter[i] << "\t" << s_min[i] << "\t" << timing[i] << "\t" << rmse[i]
       << "\n";
  }
}

void Impute::RunAccelerated() {
  const int m = a_train_->m();
  const int n = a_train_->n();
  CHECK_EQ(m, a_test_->m());
  CHECK_EQ(n, a_test_->n());
  const int k = opt_.k;

  // Solution variables.
  SMat ut1(k, m, mem_type_); // ut.
  SMat vt1(k, n, mem_type_); // vt.
  SVec s1(k, mem_type_);     // Singular values.
  SMat ut2(k, m, mem_type_); // ut2.
  SMat vt2(k, n, mem_type_); // vt2.
  SVec s2(k, mem_type_);     // Singular values.
  // Work with ut_a, ..., ut_b, ..., instead of ut1, ..., ut2, ...
  SMat *ut_a = &ut1; // Last solution.
  SMat *vt_a = &vt1;
  SVec *s_a = &s1;
  SMat *ut_b = &ut2; // Last last solution.
  SMat *vt_b = &vt2;
  SVec *s_b = &s2;
  SVec d_s(k, mem_type_);
  SVec h_s(k, MEM_HOST);

  // Training errors.
  float last_err = a_train_->value()->RootMeanSquare();

  // Intermediate data. TODO(jiawei): Use a few temp buffers to store all these.
  SMat mk(m, k, mem_type_);
  SMat q(m, k, mem_type_);
  SMat qt(k, m, mem_type_);
  SMat nk(n, k, mem_type_);
  SMat nk2(n, k, mem_type_);
  SMat kn(k, n, mem_type_);
  SMat kk(k, k, mem_type_);
  SMat kk2(k, k, mem_type_);
  SMat r(k, k, mem_type_);

  // QRs, SVD.
  QRHelper mk_qr(&mk);
  QRHelper nk_qr(&nk);

  // Currently, cusolver doesn't support economic SVD, so we considered using
  // CULA's svd. However, we suspect this procedure is using CPU a lot and doing
  // a lot of HtoD and DtoH transfers. To mitigate this, we do a QR first.
  // nk is essentially A.T * Q.
  // Say nk = Q' R' where Q' is n by k and R' is k by k.
  // Run SVD on R' to get R' = U' S' V'.T
  SVDHelper svd_helper(r, &kk2, &kk, &d_s);
  ut_b->Clear();
  vt_b->Clear();
  s_b->Clear();
  if (!opt_.randomize_init) {
    // Zero initialization.
    ut_a->Clear();
    vt_a->Clear();
    s_a->Clear();
  } else {
    // RandUniform initialization.
    ut_a->RandUniform();
    vt_a->RandUniform();
    ut_a->SetToProduct(1.0 / static_cast<float>(k), *ut_a);
    vt_a->SetToProduct(1.0 / static_cast<float>(k), *vt_a);
    s_a->Fill(1.0);
  }

  // Backup sparse matrix values.
  SVec a_train_value(*a_train_->value(), mem_type_);
  SVec a_test_value(*a_test_->value(), mem_type_);

  // Store SampleAndUpdate results for each U, S, V.T.
  SVec usv_sampled1(a_train_->nnz(), mem_type_);
  SVec usv_sampled2(a_train_->nnz(), mem_type_);
  SVec *usv_sampled_a = &usv_sampled1;
  SVec *usv_sampled_b = &usv_sampled2;
  usv_sampled1.Clear();
  usv_sampled2.Clear();

  // Collect rmse and timing info over iterations.
  Timer timer;
  vector<int> l_iter;
  vector<float> rmse;
  vector<float> timing;
  vector<float> s_min; // Smallest singular value. Want this to be zero.
  float time_elapsed = 0;
  int theta_c = 1;

  for (int iter = 0;; ++iter) {
    if ((iter % opt_.log_every_n) == 0) {
      cudaDeviceSynchronize();

      // Get time elapsed.
      time_elapsed += timer.elapsed();
      timing.push_back(time_elapsed);
      l_iter.push_back(iter);

      // Compute error.
      a_test_->CopyValuesFrom(a_test_value);
      a_test_->SampleAndUpdate(-1.0, *ut_a, *vt_a, *s_a, 1.0);
      rmse.push_back(a_test_->value()->RootMeanSquare());

      h_s.CopyFrom(*s_a);
      s_min.push_back(h_s.get(k - 1));

      // Output.
      LOG(INFO) << "Iter: " << l_iter.back();
      LOG(INFO) << " SMin: " << s_min.back();
      LOG(INFO) << " Time elapsed: " << timing.back() << " sec";
      LOG(INFO) << " RMSE: " << rmse.back();

      // Reset timer.
      timer.reset();

      if (timing.back() > opt_.max_time) {
        break;
      }
    }

    // Set a_train to be observed matrix - (1+theta)X_t + theta X_{t-1}.
    SampleAndUpdateHelper(*a_train_->row_ind(), *a_train_->row(),
                          *a_train_->col(), *usv_sampled_a, 1.0, *ut_a, *vt_a,
                          *s_a, 0);
    // TODO(jiawei): Add SetToSum2.
    a_train_->value()->SetToSum3(1.0, a_train_value, -1.0, *usv_sampled_a, 0,
                                 *usv_sampled_a);
    float this_err = a_train_->value()->RootMeanSquare();
    if (this_err > last_err) {
      theta_c = 1;
    } else {
      ++theta_c;
    }
    std::swap(this_err, last_err);
    const float theta = static_cast<float>(theta_c - 1) / (theta_c + 2);

    a_train_->value()->SetToSum3(1.0, a_train_value, -(1.0 + theta),
                                 *usv_sampled_a, theta, *usv_sampled_b);

    // Update transpose of train matrix by permuting.
    a_train_t_->value()->SetToPermute(*train_perm_, *(a_train_->value()));

    // Create random matrix if necessary, as test vectors for rnadomized SVD.
    ResetTestMatrix(iter, *vt_a, &kn);

    // Apply U.diag(S).VT to random matrix.
    kk.SetToProduct(1.0, NO_TRANS, *vt_a, TRANS, kn, 0);
    kk.SetToProduct(*s_a, kk);
    mk.SetToProduct(1.0, TRANS, *ut_a, NO_TRANS, kk, 0);
    kk.SetToProduct(1.0, NO_TRANS, *vt_b, TRANS, kn, 0);
    kk.SetToProduct(*s_b, kk);
    mk.SetToProduct(-theta, TRANS, *ut_b, NO_TRANS, kk, 1.0 + theta);

    // Apply sparse matrix. We apply to random matrix transposed because that is
    // about three times faster than the random matrix not-transposed!
    mk.SetToProduct(1.0, NO_TRANS, *a_train_, TRANS, kn, 1.0);

    for (int i = 0; i < opt_.num_gram; ++i) {
      mk_qr.Compute();

      // Apply transpose.
      kk.SetToProduct(1.0, NO_TRANS, *ut_a, NO_TRANS, mk, 0);
      kk.SetToProduct(*s_a, kk);
      nk.SetToProduct(1.0, TRANS, *vt_a, NO_TRANS, kk, 0);
      kk.SetToProduct(1.0, NO_TRANS, *ut_b, NO_TRANS, mk, 0);
      kk.SetToProduct(*s_b, kk);
      nk.SetToProduct(-theta, TRANS, *vt_b, NO_TRANS, kk, 1.0 + theta);
      nk.SetToProduct(1.0, NO_TRANS, *a_train_t_, NO_TRANS, mk, 1.0);

      // Re-orthonormalize.
      nk_qr.Compute();

      // Apply non-transpose.
      kk.SetToProduct(1.0, NO_TRANS, *vt_a, NO_TRANS, nk, 0);
      kk.SetToProduct(*s_a, kk);
      mk.SetToProduct(1.0, TRANS, *ut_a, NO_TRANS, kk, 0);
      kk.SetToProduct(1.0, NO_TRANS, *vt_b, NO_TRANS, nk, 0);
      kk.SetToProduct(*s_b, kk);
      mk.SetToProduct(-theta, TRANS, *ut_b, NO_TRANS, kk, 1.0 + theta);
      mk.SetToProduct(1.0, NO_TRANS, *a_train_, NO_TRANS, nk, 1.0);
    }

    // Do QR. Transpose the result to get k by m matrix.
    mk_qr.Compute();
    // Sparse*transpose of matrix is much faster than sparse*matrix.
    // So we prefer working with q transpose instead of q.
    qt.SetToSum(1.0, TRANS, mk, 0, NO_TRANS, qt);

    // Compute A.transpose to Q where A = sparse + sampled svd matrix.
    kk.SetToProduct(1.0, NO_TRANS, *ut_a, TRANS, qt, 0);
    kk.SetToProduct(*s_a, kk);
    nk.SetToProduct(1.0, TRANS, *vt_a, NO_TRANS, kk, 0);
    kk.SetToProduct(1.0, NO_TRANS, *ut_b, TRANS, qt, 0);
    kk.SetToProduct(*s_b, kk);
    nk.SetToProduct(-theta, TRANS, *vt_b, NO_TRANS, kk, 1.0 + theta);
    nk.SetToProduct(1.0, NO_TRANS, *a_train_t_, TRANS, qt, 1.0);

    // Run SVD.
    // (You might want to disable this to track the gaps in GPU usage.)

    // SVD(r) = kk2 * diag(s) ** kk.
    // SVD(nk) = U1 S1 V1.T = nk2 * kk2 * diag(s) ** kk
    // So U1 = nk2 * kk2 and V1.T = kk.
    // V.T = U1.T = kk2.T * nk2.T.
    // U.T = V1.T Q.T = kk * qt.

    nk2.CopyFrom(nk);
    nk_qr.Compute();
    r.SetToProduct(1.0, TRANS, nk, NO_TRANS, nk2, 0);
    svd_helper.Compute();

    // Push new solution as "last last solution".
    s_b->CopyFrom(d_s);
    if (opt_.soft_threshold) {
      s_b->SoftThreshold(opt_.sv_threshold);
    } else {
      s_b->HardThreshold(opt_.sv_threshold);
    }
    vt_b->SetToProduct(1.0, TRANS, kk2, TRANS, nk, 0);
    ut_b->SetToProduct(1.0, NO_TRANS, kk, NO_TRANS, qt, 0);

    // Swap last with (last last solution / new solution).
    std::swap(ut_a, ut_b);
    std::swap(vt_a, vt_b);
    std::swap(s_a, s_b);
    std::swap(usv_sampled_a, usv_sampled_b);
  }
  LOG(INFO) << "Total time taken: " << time_elapsed << "\n";

  CHECK_EQ(timing.size(), rmse.size());
  CHECK_EQ(timing.size(), l_iter.size());
  CHECK_EQ(timing.size(), s_min.size());
  ofstream os(opt_.output_filename);
  os << "iter\tsmin\ttime\trmse\n";
  for (size_t i = 0; i < timing.size(); ++i) {
    os << l_iter[i] << "\t" << s_min[i] << "\t" << timing[i] << "\t" << rmse[i]
       << "\n";
  }
}

} // namespace gi