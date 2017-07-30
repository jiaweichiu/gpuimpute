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
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "base.h"

DEFINE_string(train_filename,
              "/usr/local/google/home/jiawei/cfgpu/riftbolt/ml-1m/train.txt",
              "Filename for training dataset.");
DEFINE_string(test_filename,
              "/usr/local/google/home/jiawei/cfgpu/riftbolt/ml-1m/test.txt",
              "Filename for training dataset.");

DEFINE_int32(log_every_n, 2, "Report test error every this many iterations.");

DEFINE_string(output_filename, "", "Output file for stats.");

DEFINE_double(learning_rate, 0.005, "Learning rate for SGD.");

DEFINE_double(lambda, 0.1, "Lambda for SGD, used for regularization.");

DEFINE_int32(k, 32, "Number of columns for U and V that we want to recover.");

DEFINE_int32(rng_seed, 3255245, "Seed for rng.");

DEFINE_double(max_time, 100000, "Maximum time in seconds.");

namespace gi {

typedef float FloatType;

using Eigen::MatrixBase;

typedef Eigen::MatrixXd Mat;
typedef Eigen::Map<Mat> MatMap;
typedef Eigen::VectorXd Vec;
typedef Eigen::Map<Vec> VecMap;

typedef Eigen::SparseMatrix<FloatType, Eigen::RowMajor> SpMat;
typedef Eigen::Triplet<FloatType> Triplet;

class RngHelper {
public:
  RngHelper();
  RngHelper(uint_fast32_t seed);
  FloatType Normal();
  void Normal(Mat &m);
  FloatType Uniform();
  void Uniform(Mat &m);

  static RngHelper &Get();

private:
  std::mt19937 rng_;
  std::normal_distribution<FloatType> normal_dist_;
  std::uniform_real_distribution<FloatType> uniform_dist_;
};

RngHelper::RngHelper() : RngHelper(FLAGS_rng_seed) {}

RngHelper::RngHelper(uint_fast32_t seed) : rng_(seed) {}

FloatType RngHelper::Normal() { return normal_dist_(rng_); }

void RngHelper::Normal(Mat &m) {
  int size = m.rows() * m.cols();
  for (int i = 0; i < size; ++i) {
    m.data()[i] = Normal();
  }
}

FloatType RngHelper::Uniform() { return uniform_dist_(rng_); }

void RngHelper::Uniform(Mat &m) {
  int size = m.rows() * m.cols();
  for (int i = 0; i < size; ++i) {
    m.data()[i] = Uniform();
  }
}

RngHelper &RngHelper::Get() {
  static RngHelper rng_helper;
  return rng_helper;
}

SpMat NewSpMat(const string &filename) {
  ifstream is(filename);

  // Read in matrix in CSR format.
  int m;
  int n;
  int nnz;
  CHECK(is >> m >> n >> nnz);
  vector<FloatType> value(nnz);
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

  // Convert to triplets.
  vector<Triplet> triplets;
  for (int i = 0; i < m; ++i) {
    for (int j = row_ind[i]; j < row_ind[i + 1]; ++j) {
      triplets.emplace_back(i, col[j], value[j]);
    }
  }

  // Create our sparse matrix.
  SpMat out(m, n);
  out.reserve(nnz);
  out.setFromTriplets(triplets.begin(), triplets.end());
  out.makeCompressed();

  // Verify that the matrix built matches the input we get.
  CHECK_EQ(out.nonZeros(), nnz);
  CHECK_EQ(out.outerSize(), m);
  CHECK_EQ(out.innerSize(), n);
  for (int i = 0; i <= m; ++i) {
    CHECK_EQ(out.outerIndexPtr()[i], row_ind[i]);
  }
  for (int i = 0; i < nnz; ++i) {
    CHECK_EQ(out.innerIndexPtr()[i], col[i]);
    CHECK_EQ(out.valuePtr()[i], value[i]);
  }

  LOG(INFO) << "Read " << nnz << " entries from " << filename << "\n";
  return out;
}

FloatType Square(FloatType x) { return x * x; }

FloatType RmsError(const SpMat &a, Mat &ut, Mat &vt) {
  FloatType sum = 0;
  for (int q = 0; q < a.outerSize(); ++q) {
    for (SpMat::InnerIterator it(a, q); it; ++it) {
      const int i = it.row();
      const int j = it.col();
      sum += Square(it.value() - ut.col(i).dot(vt.col(j)));
    }
  }
  return sqrt(sum / a.nonZeros());
}

void Main() {
  CHECK(!FLAGS_output_filename.empty());
  CHECK(!FLAGS_train_filename.empty());
  CHECK(!FLAGS_test_filename.empty());

  SpMat a_train = NewSpMat(FLAGS_train_filename);
  SpMat a_test = NewSpMat(FLAGS_test_filename);

  const int k = FLAGS_k;

  // Dimensions check.
  const int m = a_train.rows();
  const int n = a_train.cols();
  CHECK_EQ(m, a_test.rows());
  CHECK_EQ(n, a_test.cols());

  RngHelper &rng_helper = RngHelper::Get();
  Mat ut(k, m);
  Mat vt(k, n);

  // Extra check for dimensions, to be sure.
  CHECK_EQ(a_train.rows(), a_train.outerSize());
  CHECK_EQ(a_train.rows(), ut.cols());
  CHECK_EQ(a_train.cols(), vt.cols());
  CHECK_EQ(ut.rows(), vt.rows());

  // Initialize ut, vt with random numbers.
  rng_helper.Uniform(ut);
  rng_helper.Uniform(vt);
  ut /= static_cast<FloatType>(k);
  vt /= static_cast<FloatType>(k);

  // Collect rmse and timing info over iterations.
  Timer timer;
  vector<int> l_iter;
  vector<FloatType> rmse;
  vector<float> timing;
  float time_elapsed = 0;

  for (int iter = 0;; ++iter) {
    if ((iter % FLAGS_log_every_n) == 0) {
      // Get time elapsed.
      time_elapsed += timer.elapsed();
      timing.push_back(time_elapsed);
      l_iter.push_back(iter);

      // Compute error.
      rmse.push_back(RmsError(a_test, ut, vt));

      // Output.
      LOG(INFO) << "Iter: " << l_iter.back()
                << " Time elapsed: " << timing.back()
                << " sec RMSE: " << rmse.back() << "\n";

      // Reset timer.
      timer.reset();

      if (timing.back() > FLAGS_max_time) {
        break;
      }
    }

    for (int q = 0; q < a_train.outerSize(); ++q) {
      for (SpMat::InnerIterator it(a_train, q); it; ++it) {
        const int i = it.row();
        const int j = it.col();
        const FloatType err = it.value() - ut.col(i).dot(vt.col(j));
        ut.col(i) +=
            FLAGS_learning_rate * (err * vt.col(j) - FLAGS_lambda * ut.col(i));
        vt.col(j) +=
            FLAGS_learning_rate * (err * ut.col(i) - FLAGS_lambda * vt.col(j));
      }
    }
  }

  LOG(INFO) << "Total time taken: " << time_elapsed << "\n";

  CHECK_EQ(timing.size(), rmse.size());
  CHECK_EQ(timing.size(), l_iter.size());
  ofstream os(FLAGS_output_filename);
  os << "iter\ttime\trmse\n";
  for (size_t i = 0; i < timing.size(); ++i) {
    os << l_iter[i] << "\t" << timing[i] << "\t" << rmse[i] << "\n";
  }
}

} // namespace gi

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  gi::Main();
}