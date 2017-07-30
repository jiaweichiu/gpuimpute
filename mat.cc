#include "mat.h"
#include "base.h"
#include "vec.h"

namespace gi {

namespace {

cublasOperation_t CublasTrans(Orientation o) {
  return (o == NO_TRANS) ? CUBLAS_OP_N : CUBLAS_OP_T;
}

cusparseOperation_t CusparseTrans(Orientation o) {
  return (o == NO_TRANS) ? CUSPARSE_OPERATION_NON_TRANSPOSE
                         : CUSPARSE_OPERATION_TRANSPOSE;
}

constexpr int kSampleAndUpdateMaxPower = 6;

} // namespace

SMat::SMat(int m, int n, MemType mem_type)
    : SVec(RoundToAlign(m) * n, mem_type), lda_(RoundToAlign(m)), m_(m), n_(n) {
  InitSEigenMap();
}

SMat::SMat(const SMat &src, MemType mem_type)
    : SMat(src.m(), src.n(), mem_type) {
  CopyFrom(src);
}

SMat::SMat(int m, int n, float *data, int lda, MemType mem_type)
    : SVec(lda * n, data, mem_type), lda_(lda), m_(m), n_(n) {
  InitSEigenMap();
}

void SMat::InitSEigenMap() {
  if (mem_type() == MEM_HOST) {
    mat_map_.reset(new SEigenMap(data(), m_, n_, Eigen::OuterStride<>(lda_)));
  }
}

void SMat::CopyFrom(const SMat &src) {
  int m = min(m_, src.m());
  int n = min(n_, src.n());
  if (mem_type() == MEM_DEVICE) {
    if (src.mem_type() == MEM_DEVICE) {
      CUDA_CALL(cudaMemcpy2D(data(), lda_ * sizeof(float), src.data(),
                             src.lda() * sizeof(float), m * sizeof(float), n,
                             cudaMemcpyDeviceToDevice));
    } else {
      CUDA_CALL(cudaMemcpy2D(data(), lda_ * sizeof(float), src.data(),
                             src.lda() * sizeof(float), m * sizeof(float), n,
                             cudaMemcpyHostToDevice));
    }
  } else {
    if (src.mem_type() == MEM_DEVICE) {
      CUDA_CALL(cudaMemcpy2D(data(), lda_ * sizeof(float), src.data(),
                             src.lda() * sizeof(float), m * sizeof(float), n,
                             cudaMemcpyDeviceToHost));
    } else {
      for (int i = 0; i < n; ++i) {
        CHECK(memcpy(data() + i * lda_, src.data() + i * src.lda(),
                     m * sizeof(float)));
      }
    }
  }
}

void SMat::Read(istream &is) {
  for (int i = 0; i < m_; ++i) {   // For each row.
    for (int j = 0; j < n_; ++j) { // For each column.
      float x;
      CHECK(is >> x);
      set(i, j, x);
    }
  }
}

void SMat::Write(ostream &os) const {
  for (int i = 0; i < m_; ++i) {   // For each column.
    for (int j = 0; j < n_; ++j) { // For each row.
      CHECK(os << get(i, j) << " ");
    }
    os << "\n";
  }
}

void SMat::SetToSum(float alpha, Orientation oa, const SMat &a, float beta,
                    Orientation ob, const SMat &b) {
  // Check that device types are consistent.
  CHECK_EQ(mem_type(), a.mem_type());
  CHECK_EQ(mem_type(), b.mem_type());

  // Dimensions check.
  int am = oa == NO_TRANS ? a.m() : a.n();
  int an = oa == NO_TRANS ? a.n() : a.m();
  int bm = ob == NO_TRANS ? b.m() : b.n();
  int bn = ob == NO_TRANS ? b.n() : b.m();
  CHECK_EQ(am, m_);
  CHECK_EQ(an, n_);
  CHECK_EQ(bm, m_);
  CHECK_EQ(bn, n_);

  if (mem_type() == MEM_DEVICE) {
    CUBLAS_CALL(cublasSgeam(Engine::cublas(), CublasTrans(oa), CublasTrans(ob),
                            m_, n_, &alpha, a.data(), a.lda(), &beta, b.data(),
                            b.lda(), data(), lda_));
  } else {
    if (alpha == 0 && beta == 0) {
      mat_map_->setZero();
    } else if (alpha == 0) {
      // oa == NO_TRANS ? *(a.mat_map()) : a.mat_map()->transpose() seems to
      // not work. I am not sure why! Hence, we need lots of cases here.
      if (ob == NO_TRANS) {
        mat_map_->noalias() = beta * *b.mat_map();
      } else {
        mat_map_->noalias() = beta * b.mat_map()->transpose();
      }
    } else if (beta == 0) {
      if (oa == NO_TRANS) {
        mat_map_->noalias() = alpha * *a.mat_map();
      } else {
        mat_map_->noalias() = alpha * a.mat_map()->transpose();
      }
    } else {
      if (oa == NO_TRANS) {
        if (ob == NO_TRANS) {
          mat_map_->noalias() = alpha * *a.mat_map() + beta * *b.mat_map();
        } else {
          mat_map_->noalias() =
              alpha * *a.mat_map() + beta * b.mat_map()->transpose();
        }
      } else {
        if (ob == NO_TRANS) {
          mat_map_->noalias() =
              alpha * a.mat_map()->transpose() + beta * *b.mat_map();
        } else {
          mat_map_->noalias() = alpha * a.mat_map()->transpose() +
                                beta * b.mat_map()->transpose();
        }
      }
    }
  }
}

void SMat::SetToProduct(float alpha, Orientation oa, const SMat &a,
                        Orientation ob, const SMat &b, float beta) {
  // Check that mem_types are consistent.
  CHECK_EQ(mem_type(), a.mem_type());
  CHECK_EQ(mem_type(), b.mem_type());

  // Dimensions check.
  int am = oa == NO_TRANS ? a.m() : a.n();
  int an = oa == NO_TRANS ? a.n() : a.m();
  int bm = ob == NO_TRANS ? b.m() : b.n();
  int bn = ob == NO_TRANS ? b.n() : b.m();
  CHECK_EQ(am, m_);
  CHECK_EQ(bn, n_);
  CHECK_EQ(an, bm);

  if (mem_type() == MEM_DEVICE) {
    CUBLAS_CALL(cublasSgemm(Engine::cublas(), CublasTrans(oa), CublasTrans(ob),
                            m_, n_, an, &alpha, a.data(), a.lda(), b.data(),
                            b.lda(), &beta, data(), lda_));
  } else {
    // Assume no aliasing to force lazy evaluation.
    // We use a lot of cases because I am not sure why something like
    // "NO_TRANS ? *a.mat_map() : a.mat_map()->transpose()" is not working.
    if (alpha == 0 && beta == 0) {
      mat_map_->setZero();
    } else if (beta == 0) {
      if (oa == NO_TRANS) {
        if (ob == NO_TRANS) {
          mat_map_->noalias() = alpha * *a.mat_map() * *b.mat_map();
        } else {
          mat_map_->noalias() = alpha * *a.mat_map() * b.mat_map()->transpose();
        }
      } else {
        if (ob == NO_TRANS) {
          mat_map_->noalias() = alpha * a.mat_map()->transpose() * *b.mat_map();
        } else {
          mat_map_->noalias() =
              alpha * a.mat_map()->transpose() * b.mat_map()->transpose();
        }
      }
    } else {
      if (oa == NO_TRANS) {
        if (ob == NO_TRANS) {
          mat_map_->noalias() =
              beta * *mat_map_ + alpha * *a.mat_map() * *b.mat_map();
        } else {
          mat_map_->noalias() = beta * *mat_map_ +
                                alpha * *a.mat_map() * b.mat_map()->transpose();
        }
      } else {
        if (ob == NO_TRANS) {
          mat_map_->noalias() = beta * *mat_map_ +
                                alpha * a.mat_map()->transpose() * *b.mat_map();
        } else {
          mat_map_->noalias() =
              beta * *mat_map_ +
              alpha * a.mat_map()->transpose() * b.mat_map()->transpose();
        }
      }
    }
  }
}

void SMat::SetToProduct(float alpha, Orientation oa, const SSpMat &a,
                        Orientation ob, const SMat &b, float beta) {
  // Check that mem_types are consistent.
  CHECK_EQ(mem_type(), a.mem_type());
  CHECK_EQ(mem_type(), b.mem_type());

  // Dimensions check.
  int am = oa == NO_TRANS ? a.m() : a.n();
  int an = oa == NO_TRANS ? a.n() : a.m();
  int bm = ob == NO_TRANS ? b.m() : b.n();
  int bn = ob == NO_TRANS ? b.n() : b.m();
  CHECK_EQ(am, m_);
  CHECK_EQ(bn, n_);
  CHECK_EQ(an, bm);

  if (mem_type() == MEM_DEVICE) {
    CUSPARSE_CALL(cusparseScsrmm2(
        Engine::cusparse(), CusparseTrans(oa), CusparseTrans(ob), a.m(), n_,
        a.n(), a.nnz(), &alpha, Engine::cusparse_desc(), a.value()->data(),
        a.row_ind()->data(), a.col()->data(), b.data(), b.lda(), &beta, data(),
        lda()));
  } else {
    if (beta == 0) {
      if (oa == NO_TRANS) {
        if (ob == NO_TRANS) {
          mat_map_->noalias() = *(a.eigen_sp_mat()) * *(b.mat_map());
        } else {
          mat_map_->noalias() = *(a.eigen_sp_mat()) * b.mat_map()->transpose();
        }
      } else {
        if (ob == NO_TRANS) {
          mat_map_->noalias() = a.eigen_sp_mat()->transpose() * *(b.mat_map());
        } else {
          mat_map_->noalias() =
              (a.eigen_sp_mat()->transpose() * b.mat_map()->transpose());
        }
      }
    } else {
      if (oa == NO_TRANS) {
        if (ob == NO_TRANS) {
          mat_map_->noalias() =
              (beta * *mat_map_ + *(a.eigen_sp_mat()) * *(b.mat_map()));
        } else {
          mat_map_->noalias() =
              (beta * *mat_map_ +
               *(a.eigen_sp_mat()) * b.mat_map()->transpose());
        }
      } else {
        if (ob == NO_TRANS) {
          mat_map_->noalias() =
              (beta * *mat_map_ +
               a.eigen_sp_mat()->transpose() * *(b.mat_map()));
        } else {
          mat_map_->noalias() =
              (beta * *mat_map_ +
               (a.eigen_sp_mat()->transpose() * b.mat_map()->transpose()));
        }
      }
    }
  }
}

void SMat::SetToProduct(const SVec &d, const SMat &a) {
  CHECK_EQ(d.mem_type(), mem_type());
  CHECK_EQ(a.mem_type(), mem_type());

  // Dimensions check.
  CHECK_EQ(d.size(), a.m());
  CHECK_EQ(m_, a.m());
  CHECK_EQ(n_, a.n());

  if (mem_type() == MEM_DEVICE) {
    CUBLAS_CALL(cublasSdgmm(Engine::cublas(), CUBLAS_SIDE_LEFT, a.m(), a.n(),
                            a.data(), a.lda(), d.data(), 1, data(), lda()));
  } else {
    mat_map_->noalias() = d.vec_map()->asDiagonal() * *(a.mat_map());
  }
}

void SMat::SetToProduct(float alpha, const SMat &a) {
  CHECK_EQ(a.mem_type(), mem_type());

  // Dimensions check.
  CHECK_EQ(m_, a.m());
  CHECK_EQ(n_, a.n());

  float beta = 0;

  if (mem_type() == MEM_DEVICE) {
    CUBLAS_CALL(cublasSgeam(Engine::cublas(), CublasTrans(NO_TRANS),
                            CublasTrans(NO_TRANS), m_, n_, &alpha, a.data(),
                            a.lda(), &beta, data(), lda_, data(), lda_));
  } else {
    mat_map_->noalias() = alpha * *(a.mat_map());
  }
}

SSpMat::SSpMat(istream &is) {
  int m;
  int n;
  int nnz;
  CHECK(is >> m >> n >> nnz);
  vector<float> value(nnz);
  vector<int> row_ind(m + 1);
  vector<int> col(nnz);
  for (int i = 0; i < nnz; ++i) {
    CHECK(is >> value[i]);
  }
  for (int i = 0; i < nnz; ++i) {
    CHECK(is >> col[i]);
  }
  for (int i = 0; i <= m; ++i) {
    CHECK(is >> row_ind[i]);
  }
  InitOnHost(m, n, nnz, row_ind, col, value);
}

SSpMat::SSpMat(int m, int n, int nnz, const vector<int> &row_ind,
               const vector<int> &col, const vector<float> &value) {
  InitOnHost(m, n, nnz, row_ind, col, value);
}

void SSpMat::InitOnHost(int m, int n, int nnz, const vector<int> &row_ind,
                        const vector<int> &col, const vector<float> &value) {
  mem_type_ = MEM_HOST; // For now, only support host, not device.
  m_ = m;
  n_ = n;
  nnz_ = nnz;
  CHECK_EQ(m + 1, row_ind.size());
  CHECK_EQ(nnz, col.size());
  CHECK_EQ(nnz, value.size());

  // Prepare triplets and also update row_.
  row_.reset(new IVec(nnz_, mem_type_));
  vector<Eigen::Triplet<float>> triplets;
  for (int i = 0; i < m_; ++i) {
    for (int j = row_ind[i]; j < row_ind[i + 1]; ++j) {
      triplets.emplace_back(i, col[j], value[j]);
      row_->set(j, i);
    }
  }

  // Prepare a_, our Eigen sparse matrix.
  a_.reset(new EigenSpMat(m_, n_));
  a_->reserve(nnz_);
  a_->setFromTriplets(triplets.begin(), triplets.end());
  a_->makeCompressed();

  // Verify that a_'s data matches our original input in CSR format.
  CHECK_EQ(a_->nonZeros(), nnz_);
  CHECK_EQ(a_->outerSize(), m_);
  CHECK_EQ(a_->innerSize(), n_);
  for (int i = 0; i <= m_; ++i) {
    CHECK_EQ(a_->outerIndexPtr()[i], row_ind[i]);
  }
  for (int i = 0; i < nnz_; ++i) {
    CHECK_EQ(a_->innerIndexPtr()[i], col[i]);
    CHECK_EQ(a_->valuePtr()[i], value[i]);
  }

  // Do not own. They belong to a_.
  row_ind_.reset(new IVec(m_ + 1, a_->outerIndexPtr(), MEM_HOST));
  col_.reset(new IVec(nnz_, a_->innerIndexPtr(), MEM_HOST));
  value_.reset(new SVec(nnz_, a_->valuePtr(), MEM_HOST));
}

SSpMat::SSpMat(const SSpMat &src, MemType mem_type)
    : mem_type_(mem_type), nnz_(src.nnz()), m_(src.m()), n_(src.n()) {
  if (mem_type == MEM_DEVICE) {
    // Host/device copying to device.
    row_ind_.reset(new IVec(*src.row_ind(), MEM_DEVICE));
    col_.reset(new IVec(*src.col(), MEM_DEVICE));
    row_.reset(new IVec(*src.row(), MEM_DEVICE));
    value_.reset(new SVec(*src.value(), MEM_DEVICE));
  } else {
    CHECK_EQ(src.mem_type(), MEM_HOST);

    // Host copying to host only.
    a_.reset(new EigenSpMat(*(src.a_)));
    row_.reset(new IVec(*(src.row_), MEM_HOST));

    // Do not own. They belong to a_.
    row_ind_.reset(new IVec(m_ + 1, a_->outerIndexPtr(), MEM_HOST));
    col_.reset(new IVec(nnz_, a_->innerIndexPtr(), MEM_HOST));
    value_.reset(new SVec(nnz_, a_->valuePtr(), MEM_HOST));
  }
}

SSpMat *SSpMat::ReadInput(const string &filename, MemType mem_type) {
  ifstream fin(filename);
  unique_ptr<SSpMat> mat(new SSpMat(fin));
  if (mem_type == mat->mem_type()) {
    return mat.release();
  }
  return new SSpMat(*mat, mem_type); // Make a copy.
}

void SSpMat::Write(ostream &os) const {
  CHECK_EQ(mem_type_, MEM_HOST);
  os << "m=" << m_ << " n=" << n_ << " nnz=" << nnz_ << "\n";
  os << *row_ind_ << "\n";
  os << *row_ << "\n";
  os << *col_ << "\n";
  os << *value_ << "\n";
}

ostream &operator<<(ostream &os, const SSpMat &x) {
  x.Write(os);
  return os;
}

string SSpMat::DebugString() const {
  ostringstream os;
  Write(os);
  return os.str();
}

void SSpMat::CopyValuesFrom(const SSpMat &src) {
  value_->CopyFrom(*(src.value_));
}

void SSpMat::CopyValuesFrom(const SVec &src) { value_->CopyFrom(src); }

void SSpMat::ClearValues() { value_->Clear(); }

void SSpMat::SampleAndUpdate(float alpha, const SMat &ut, const SMat &vt,
                             const SVec &s, float beta) {
  SampleAndUpdateHelper(*row_ind_, *row_, *col_, *value_, alpha, ut, vt, s,
                        beta);
}

void SampleAndUpdateHelper(const IVec &row_ind, const IVec &row,
                           const IVec &col, const SVec &value, float alpha,
                           const SMat &ut, const SMat &vt, const SVec &s,
                           float beta) {
  // Check that mem_types are consistent.
  CHECK_EQ(row_ind.mem_type(), ut.mem_type());
  CHECK_EQ(row_ind.mem_type(), vt.mem_type());
  CHECK_EQ(row_ind.mem_type(), s.mem_type());

  CHECK_EQ(row_ind.mem_type(), row.mem_type());
  CHECK_EQ(row_ind.mem_type(), col.mem_type());
  CHECK_EQ(row_ind.mem_type(), value.mem_type());

  // Dimensions check.
  const int m = ut.n();
  CHECK_EQ(row_ind.size(), m + 1);
  CHECK_EQ(row.size(), col.size());
  CHECK_EQ(row.size(), value.size());
  const int k = s.size();
  CHECK_EQ(ut.m(), k);
  CHECK_EQ(vt.m(), k);

  if (row_ind.mem_type() == MEM_DEVICE) {
    DeviceSampleAndUpdateHelper(row, col, value, alpha, ut, vt, s, beta);
  } else {
    CHECK_LE(k, 1 << kSampleAndUpdateMaxPower);
    float buf[1 << kSampleAndUpdateMaxPower];
    for (int i = 0; i < m; ++i) { // Row of output is "i".
      for (int l = 0; l < k; ++l) {
        buf[l] = alpha * ut.get(l, i) * s.get(l);
      }

      const int j0 = row_ind.get(i);
      const int j1 = row_ind.get(i + 1);
      for (int j = j0; j < j1; ++j) {
        // Column of output is "col".
        const int column = col.get(j);
        // Compute inner product over k elements.
        float increment = 0;
        for (int l = 0; l < k; ++l) {
          increment += (buf[l] * vt.get(l, column));
        }
        // Update the sparse matrix at row i, column col, inner index j.
        float *x = value.data() + j;
        *x = *x * beta + increment;
      }
    }
  }
}

} // namespace riftbolt