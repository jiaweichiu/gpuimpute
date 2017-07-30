#pragma once

#include "base.h"
#include "vec.h"

namespace gi {

enum Orientation {
  NO_TRANS,
  TRANS,
};

constexpr int kAlignment = 32;
inline int RoundToAlign(int x) {
  return (((x + kAlignment - 1) / kAlignment)) * kAlignment;
}

typedef Eigen::Map<Eigen::MatrixXf, 0, Eigen::OuterStride<>> SEigenMap;

class SSpMat;

class SMat : public SVec {
public:
  // owned=true.
  SMat(int m, int n, MemType mem_type);
  SMat(const SMat &src, MemType mem_type);

  // owned=false.
  SMat(int m, int n, float *data, int lda, MemType mem_type);

  ~SMat() override = default;

  // Getters, setters.
  void set(int i, int j, float x) { SVec::set(i + j * lda_, x); }
  // Get row i, col j.
  float get(int i, int j) const { return SVec::get(i + j * lda_); }
  int lda() const { return lda_; }
  int m() const { return m_; }
  int n() const { return n_; }
  SEigenMap *mat_map() const { return mat_map_.get(); }

  // Operations.
  // Be careful with aliasing! There is no aliasing here.
  // this <- alpha * op(a) + beta * op(b).
  void SetToSum(float alpha, Orientation oa, const SMat &a, float beta,
                Orientation ob, const SMat &b);

  // this <- beta * this + alpha * op(a) * op(b).
  void SetToProduct(float alpha, Orientation oa, const SMat &a, Orientation ob,
                    const SMat &b, float beta);

  // this <- beta * this + alpha * op(a) * op(b).
  void SetToProduct(float alpha, Orientation oa, const SSpMat &a,
                    Orientation ob, const SMat &b, float beta);

  // this <- diag(d) * a.
  void SetToProduct(const SVec &d, const SMat &a);

  // this <- alpha * a.
  void SetToProduct(float alpha, const SMat &a);

  void CopyFrom(const SMat &src);
  void Read(istream &is) override;
  void Write(ostream &os) const override;

private:
  void InitSEigenMap();

  int lda_;
  int m_;
  int n_;
  unique_ptr<SEigenMap> mat_map_; // Eigen. Only for host, not device.
};

typedef Eigen::SparseMatrix<float, Eigen::RowMajor> EigenSpMat;

// Single-precision sparse matrix.
class SSpMat {
public:
  // For now, only support mem_type=MEM_HOST for the following constructors.
  SSpMat(istream &is);
  SSpMat(int m, int n, int nnz, const vector<int> &row_ind,
         const vector<int> &col, const vector<float> &value);

  // For now, only support src.mem_type=MEM_HOST.
  SSpMat(const SSpMat &src, MemType mem_type);

  MemType mem_type() const { return mem_type_; }
  int nnz() const { return nnz_; }
  int m() const { return m_; }
  int n() const { return n_; }

  IVec *row_ind() const { return row_ind_.get(); }
  IVec *col() const { return col_.get(); }
  IVec *row() const { return row_.get(); }
  SVec *value() const { return value_.get(); }
  EigenSpMat *eigen_sp_mat() const { return a_.get(); }

  // this <- beta * this + alpha * USV.T sampled on support of this.
  void SampleAndUpdate(float alpha, const SMat &ut, const SMat &vt,
                       const SVec &s, float beta);

  void Write(ostream &os) const;

  string DebugString() const;

  void CopyValuesFrom(const SSpMat &src);
  void CopyValuesFrom(const SVec &src);

  void ClearValues();

  static SSpMat *ReadInput(const string &filename, MemType mem_type);

private:
  void InitOnHost(int m, int n, int nnz, const vector<int> &row_ind,
                  const vector<int> &col, const vector<float> &value);

  MemType mem_type_;
  int nnz_;
  int m_;
  int n_;
  unique_ptr<IVec> row_ind_;
  unique_ptr<IVec> col_;
  unique_ptr<IVec> row_;
  unique_ptr<SVec> value_;

  // Host.
  unique_ptr<EigenSpMat> a_;
};

ostream &operator<<(ostream &os, const SSpMat &x);

// Sparse matrix defined by (row, col, value).
// Sample U*S*V.T on support of sparse matrix. Call this z.
// Set value <- beta * value + alpha * z.
void SampleAndUpdateHelper(const IVec &row_ind, const IVec &row,
                           const IVec &col, const SVec &value, float alpha,
                           const SMat &ut, const SMat &vt, const SVec &s,
                           float beta);

void DeviceSampleAndUpdateHelper(const IVec &row, const IVec &col,
                                 const SVec &value, float alpha, const SMat &ut,
                                 const SMat &vt, const SVec &s, float beta);

} // namespace gi