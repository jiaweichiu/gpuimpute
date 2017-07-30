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
#include "base.h"
#include "vec.h"
#include "mat.h"
#include "qr.h"
#include "svd.h"

namespace gi {

struct ImputeOptions {
  float sv_threshold = 5.0;
  int k = 32;
  int num_gram = 1;
  bool use_gpu = false;
  string init = "zero";
  string output_filename;
  string train_filename;
  string train_t_filename;
  string test_filename;
  string train_perm_filename;

  // Evaluate error, state etc every this many iterations.
  int log_every_n = 100;

  // Max running time in seconds.
  double max_time = 300;
};

void Impute(const ImputeOptions& opt) {
  CHECK_GT(opt.sv_threshold, 0);

  CHECK(!opt.output_filename.empty());
  CHECK(!opt.train_filename.empty());
  CHECK(!opt.train_t_filename.empty());
  CHECK(!opt.test_filename.empty());
  CHECK(!opt.train_perm_filename.empty());

  const MemType mem_type = opt.use_gpu ? MEM_DEVICE : MEM_HOST;
  unique_ptr<SSpMat> a_train(SSpMat::ReadInput(opt.train_filename, mem_type));
  unique_ptr<SSpMat> a_train_t(SSpMat::ReadInput(opt.train_t_filename,
                                               mem_type));
  unique_ptr<SSpMat> a_test(SSpMat::ReadInput(opt.test_filename, mem_type));
  unique_ptr<IVec> train_perm;
  {
    ifstream fin(opt.train_perm_filename);
    IVec v(a_train->nnz(), MEM_HOST);
    fin >> v;
    train_perm.reset(new IVec(v, mem_type));
  }
  LOG(INFO) << "Read: " << opt.train_filename;
  LOG(INFO) << "Number of nonzero elements: " << a_train->nnz();

  const int m = a_train->m();
  const int n = a_train->n();
  CHECK_EQ(m, a_test->m());
  CHECK_EQ(n, a_test->n());
  const int k = opt.k;

  // Extra variables.
  SMat ut(k, m, mem_type); // ut.
  SMat vt(k, n, mem_type); // vt.
  SVec s(k, mem_type);     // Singular values.
  SMat mk(m, k, mem_type);
  SMat q(m, k, mem_type);
  SMat qt(k, m, mem_type);
  SMat nk(n, k, mem_type);
  SMat nk2(n, k, mem_type);
  SMat kn(k, n, mem_type);
  SMat kk(k, k, mem_type);
  SMat kk2(k, k, mem_type);
  SMat r(k, k, mem_type);
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
  LOG(INFO) << "SoftImpute\n";
  LOG(INFO) << "use_gpu=" << opt.use_gpu;
  cout << " init=" << opt.init << "\n";
  if (opt.init == "zero") {
    // Zero initialization.
    ut.Clear();
    vt.Clear();
    s.Clear();
  } else if (opt.init == "rand") {
    // RandUniform initialization.
    ut.RandUniform();
    vt.RandUniform();
    ut.SetToProduct(1.0 / static_cast<float>(k), ut);
    vt.SetToProduct(1.0 / static_cast<float>(k), vt);
    s.Fill(1.0);
  } else {
    LOG(FATAL) << "Unknown --init: " << opt.init;
  }

  // Backup sparse matrix values.
  SVec a_train_value(*a_train->value(), mem_type);
  SVec a_test_value(*a_test->value(), mem_type);

  // Collect rmse and timing info over iterations.
  Timer timer;
  vector<int> l_iter;
  vector<float> rmse;
  vector<float> timing;
  vector<float> s_min; // Smallest singular value. Want this to be zero.
  float time_elapsed = 0;

  for (int iter = 0;; ++iter) {
    if ((iter % opt.log_every_n) == 0) {
      cudaDeviceSynchronize();

      // Get time elapsed.
      time_elapsed += timer.elapsed();
      timing.push_back(time_elapsed);
      l_iter.push_back(iter);

      // Compute error.
      a_test->CopyValuesFrom(a_test_value);
      a_test->SampleAndUpdate(-1.0, ut, vt, s, 1.0);
      rmse.push_back(a_test->value()->RootMeanSquare());

      h_s.CopyFrom(s);
      s_min.push_back(h_s.get(k - 1));

      // Output.
      cout << "Iter: " << l_iter.back();
      cout << " SMin: " << s_min.back();
      cout << " Time elapsed: " << timing.back() << " sec";
      cout << " RMSE: " << rmse.back() << "\n";

      // Reset timer.
      timer.reset();

      // TODO: We could add a postprocessing step here.
      if (timing.back() > opt.max_time) {
        break;
      }
    }

    a_train->CopyValuesFrom(a_train_value);

    // Decrement sparse matrix by sampling u*diag(s)*v.T on its support.
    a_train->SampleAndUpdate(-1.0, ut, vt, s, 1.0);

    // Update transpose of train matrix by permuting.
    a_train_t->value()->SetToPermute(*train_perm, *(a_train->value()));

    // Create random matrix.
    kn.RandNormal(0, 1);

    // Apply U.diag(S).VT to random matrix.
    // Apply sparse matrix. We apply to random matrix transposed because that is
    // about three times faster than the random matrix not-transposed!
    kk.SetToProduct(1.0, NO_TRANS, vt, TRANS, kn, 0);
    kk.SetToProduct(s, kk);
    mk.SetToProduct(1.0, TRANS, ut, NO_TRANS, kk, 0);
    mk.SetToProduct(1.0, NO_TRANS, *a_train, TRANS, kn, 1.0);

    for (int i = 0; i < opt.num_gram; ++i) {
      // QRCompute(&mk);
      mk_qr.Compute();

      kk.SetToProduct(1.0, NO_TRANS, ut, NO_TRANS, mk, 0);
      kk.SetToProduct(s, kk);
      nk.SetToProduct(1.0, TRANS, vt, NO_TRANS, kk, 0);
      nk.SetToProduct(1.0, NO_TRANS, *a_train_t, NO_TRANS, mk, 1.0);

      // Re-orthonormalize.
      nk_qr.Compute();

      kk.SetToProduct(1.0, NO_TRANS, vt, NO_TRANS, nk, 0);
      kk.SetToProduct(s, kk);
      mk.SetToProduct(1.0, TRANS, ut, NO_TRANS, kk, 0);
      mk.SetToProduct(1.0, NO_TRANS, *a_train, NO_TRANS, nk, 1.0);
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
    nk.SetToProduct(1.0, NO_TRANS, *a_train_t, TRANS, qt, 1.0);

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

    // Soft threshold.
    s.SoftThreshold(opt.sv_threshold);
  }
  cout << "Total time taken: " << time_elapsed << "\n";

  CHECK_EQ(timing.size(), rmse.size());
  CHECK_EQ(timing.size(), l_iter.size());
  ofstream os(opt.output_filename);
  os << "iter\tsmin\ttime\trmse\n";
  for (size_t i = 0; i < timing.size(); ++i) {
    os << l_iter[i] << "\t" << s_min[i] << "\t" << timing[i] << "\t" << rmse[i]
       << "\n";
  }
}

} // namespace gi