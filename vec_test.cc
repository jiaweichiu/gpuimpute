#include <gtest/gtest.h>

#include "base.h"
#include "vec.h"

namespace gi {

TEST(IVecTest, Basic) {
  IVec h_x(10, MEM_HOST);
  h_x.Clear();
  EXPECT_EQ(h_x.DebugString(), "0 0 0 0 0 0 0 0 0 0 ");

  h_x.set(3, 77);
  EXPECT_FLOAT_EQ(h_x.get(3), 77);
  EXPECT_EQ(h_x.DebugString(), "0 0 0 77 0 0 0 0 0 0 ");

  IVec d_y(10, MEM_DEVICE);
  d_y.CopyFrom(h_x);

  IVec h_z(10, MEM_HOST);
  h_z.Clear();
  EXPECT_EQ(h_z.DebugString(), "0 0 0 0 0 0 0 0 0 0 ");
  h_z.CopyFrom(d_y);
  EXPECT_EQ(h_z.DebugString(), "0 0 0 77 0 0 0 0 0 0 ");
}

TEST(IVecTest, NoOwn) {
  vector<int> h_v(3, 5);
  {
    IVec h_v2(3, h_v.data(), MEM_HOST); // We do not own data.
    EXPECT_EQ(h_v2.DebugString(), "5 5 5 ");
    IVec d_v2(h_v2, MEM_DEVICE);
    d_v2.Clear();
    h_v2.CopyFrom(d_v2);
    // Destroy h_v2.
  }
  for (float x : h_v) {
    EXPECT_FLOAT_EQ(x, 0);
  }
}

TEST(IVecTest, ReadWrite) {
  istringstream is("9 8 7 6 5 0 1 2 3 4");
  IVec h_x(10, MEM_HOST);
  EXPECT_TRUE(is >> h_x);
  EXPECT_EQ(h_x.DebugString(), "9 8 7 6 5 0 1 2 3 4 ");
}

TEST(SVecTest, Basic) {
  SVec h_x(10, MEM_HOST);

  h_x.Fill(3);
  EXPECT_EQ(h_x.DebugString(), "3 3 3 3 3 3 3 3 3 3 ");
  h_x.Clear();
  EXPECT_EQ(h_x.DebugString(), "0 0 0 0 0 0 0 0 0 0 ");

  SVec d_x(10, MEM_DEVICE);
  d_x.Fill(6);
  h_x.CopyFrom(d_x);
  EXPECT_EQ(h_x.DebugString(), "6 6 6 6 6 6 6 6 6 6 ");

  h_x.Clear();
  EXPECT_EQ(h_x.DebugString(), "0 0 0 0 0 0 0 0 0 0 ");
  h_x.set(3, 1.0);
  EXPECT_FLOAT_EQ(h_x.get(3), 1.0);
  EXPECT_EQ(h_x.DebugString(), "0 0 0 1 0 0 0 0 0 0 ");

  d_x.CopyFrom(h_x);
  SVec h_z(10, MEM_HOST);
  h_z.Clear();
  EXPECT_EQ(h_z.DebugString(), "0 0 0 0 0 0 0 0 0 0 ");
  h_z.CopyFrom(d_x);
  EXPECT_EQ(h_z.DebugString(), "0 0 0 1 0 0 0 0 0 0 ");
}

TEST(SVecTest, MassiveFill) {
  constexpr int n = 5000000;
  SVec h_x(n, MEM_HOST);
  SVec d_x(n, MEM_DEVICE);
  h_x.Fill(3);
  d_x.Fill(10);
  h_x.CopyFrom(d_x);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(h_x.get(i), 10);
  }
}

void CheckMeanAndRMS(const SVec &x, const SVec &y, float expected_mean,
                     float expected_rms) {
  // Check that mean is close to 0.
  const float mu = y.Mean();
  EXPECT_NEAR(mu, expected_mean, 1e-2);

  // Check that std dev is close to 1.
  const float rms = y.RootMeanSquare();
  EXPECT_NEAR(rms, expected_rms, 1e-2);

  // Check that RootMeanSquare and Mean and AbsMean are the same on host and
  // device.
  EXPECT_NEAR(mu, x.Mean(), 1e-5);
  EXPECT_NEAR(rms, x.RootMeanSquare(), 1e-5);
  EXPECT_NEAR(y.AbsMean(), x.AbsMean(), 1e-5);
}

TEST(SVecTest, RandNormal) {
  constexpr int n = 50000;
  SVec h_x(n, MEM_HOST);
  SVec d_x(n, MEM_DEVICE);

  h_x.RandNormal();
  d_x.CopyFrom(h_x);
  CheckMeanAndRMS(h_x, d_x, 0, 1);

  d_x.RandNormal();
  h_x.CopyFrom(d_x);
  CheckMeanAndRMS(h_x, d_x, 0, 1);
}

TEST(SVecTest, RandUniform) {
  constexpr int n = 50000;
  SVec h_x(n, MEM_HOST);
  SVec d_x(n, MEM_DEVICE);

  // Variance is 1/12.
  const float expected_rms = sqrt(0.5 * 0.5 + 1.0 / 12.0);
  h_x.RandUniform();
  d_x.CopyFrom(h_x);
  CheckMeanAndRMS(h_x, d_x, 0.5, expected_rms);

  d_x.RandUniform();
  h_x.CopyFrom(d_x);
  CheckMeanAndRMS(h_x, d_x, 0.5, expected_rms);
}

TEST(SVecTest, DotSums) {
  SVec h_x(3, MEM_HOST);
  h_x.set(0, 1);
  h_x.set(1, 5);
  h_x.set(2, -3);
  SVec h_y(3, MEM_HOST);
  h_y.set(0, -2);
  h_y.set(1, 4);
  h_y.set(2, -6);
  SVec d_x(h_x, MEM_DEVICE);
  SVec d_y(h_y, MEM_DEVICE);

  EXPECT_NEAR(h_x.Dot(h_y), 36, 1e-5);
  EXPECT_NEAR(d_x.Dot(d_y), 36, 1e-5);

  EXPECT_NEAR(h_x.Sum(), 3, 1e-5);
  EXPECT_NEAR(d_x.Sum(), 3, 1e-5);

  EXPECT_NEAR(h_y.Sum(), -4, 1e-5);
  EXPECT_NEAR(d_y.Sum(), -4, 1e-5);

  EXPECT_NEAR(h_x.AbsSum(), 9, 1e-5);
  EXPECT_NEAR(d_x.AbsSum(), 9, 1e-5);

  EXPECT_NEAR(h_x.MeanSquare(), 11.6666666667, 1e-5);
  EXPECT_NEAR(d_x.MeanSquare(), 11.6666666667, 1e-5);
}

TEST(SVecTest, SetToPermute) {
  SVec h_v(5, MEM_HOST);
  SVec h_w(5, MEM_HOST);
  h_v.set(0, 1);
  h_v.set(1, 3);
  h_v.set(2, 5);
  h_v.set(3, 7);
  h_v.set(4, 9);
  IVec h_perm(5, MEM_HOST);
  h_perm.set(0, 3);
  h_perm.set(1, 1);
  h_perm.set(2, 2);
  h_perm.set(3, 0);
  h_perm.set(4, 4);

  // Test on host.
  h_w.SetToPermute(h_perm, h_v);
  EXPECT_EQ(h_w.DebugString(), "7 3 5 1 9 ");

  // Test on device.
  SVec d_v(h_v, MEM_DEVICE);
  SVec d_w(5, MEM_DEVICE);
  IVec d_perm(h_perm, MEM_DEVICE);
  d_w.SetToPermute(d_perm, d_v);
  h_w.Clear();
  EXPECT_EQ(h_w.DebugString(), "0 0 0 0 0 ");
  h_w.CopyFrom(d_w);
  EXPECT_EQ(h_w.DebugString(), "7 3 5 1 9 ");
}

TEST(SVecTest, Multiply) {
  constexpr int n = 10;
  SVec h_x(n, MEM_HOST);
  SVec d_x(n, MEM_DEVICE);

  istringstream is("9 8 7 6 5 0 1 2 3 4");
  EXPECT_TRUE(is >> h_x);
  h_x.Multiply(2);
  EXPECT_EQ(h_x.DebugString(), "18 16 14 12 10 0 2 4 6 8 ");

  d_x.CopyFrom(h_x);
  d_x.Multiply(3);
  h_x.CopyFrom(d_x);
  EXPECT_EQ(h_x.DebugString(), "54 48 42 36 30 0 6 12 18 24 ");
}

TEST(SVecTest, Invert) {
  constexpr int n = 4;
  SVec h_x(n, MEM_HOST);
  SVec d_x(n, MEM_DEVICE);

  istringstream is("1 0.5 0.25 0.125");
  EXPECT_TRUE(is >> h_x);
  d_x.CopyFrom(h_x);
  h_x.Invert();
  EXPECT_EQ(h_x.DebugString(), "1 2 4 8 ");

  d_x.Invert();
  h_x.Clear();
  h_x.CopyFrom(d_x);
  EXPECT_EQ(h_x.DebugString(), "1 2 4 8 ");
}

TEST(SVecTest, SetToSum3) {
  constexpr int n = 10;
  SVec h_x(n, MEM_HOST);
  SVec d_x(n, MEM_DEVICE);
  SVec h_y(n, MEM_HOST);
  SVec d_y(n, MEM_DEVICE);
  SVec h_z(n, MEM_HOST);
  SVec d_z(n, MEM_DEVICE);

  {
    istringstream is("9 8 7 6 5 0 1 2 3 4");
    EXPECT_TRUE(is >> h_x);
  }
  {
    istringstream is("5 4 7 1 5 8 2 9 3 5");
    EXPECT_TRUE(is >> h_y);
  }
  {
    istringstream is("1 1 2 2 3 3 8 8 0 2");
    EXPECT_TRUE(is >> h_z);
  }
  h_x.SetToSum3(2, h_x, -1, h_y, 3, h_z);
  EXPECT_EQ(h_x.DebugString(), "16 15 13 17 14 1 24 19 3 9 ");

  {
    istringstream is("9 8 7 6 5 0 1 2 3 4");
    EXPECT_TRUE(is >> h_x);
  }
  d_x.CopyFrom(h_x);
  d_y.CopyFrom(h_y);
  d_z.CopyFrom(h_z);
  d_x.SetToSum3(2, d_x, -1, d_y, 3, d_z);
  h_x.Clear();
  h_x.CopyFrom(d_x);
  EXPECT_EQ(h_x.DebugString(), "16 15 13 17 14 1 24 19 3 9 ");
}

TEST(SVecTest, SoftThreshold) {
  constexpr int n = 11;
  SVec h_x(n, MEM_HOST);
  {
    istringstream is("0 1 -1 2 -2 3 -3 4 -4 5 -5");
    EXPECT_TRUE(is >> h_x);
  }
  SVec d_x(h_x, MEM_DEVICE); // Make a copy to device first.

  // Threshold on host copy.
  h_x.SoftThreshold(2.5);
  EXPECT_EQ(h_x.DebugString(), "0 0 0 0 0 0.5 -0.5 1.5 -1.5 2.5 -2.5 ");

  // Threshold on device copy.
  d_x.SoftThreshold(2.5);
  h_x.Clear();
  EXPECT_EQ(h_x.DebugString(), "0 0 0 0 0 0 0 0 0 0 0 ");
  h_x.CopyFrom(d_x);
  EXPECT_EQ(h_x.DebugString(), "0 0 0 0 0 0.5 -0.5 1.5 -1.5 2.5 -2.5 ");
}

TEST(SVecTest, HardThreshold) {
  constexpr int n = 11;
  SVec h_x(n, MEM_HOST);
  {
    istringstream is("0 1 -1 2 -2 3 -3 4 -4 5 -5");
    EXPECT_TRUE(is >> h_x);
  }
  SVec d_x(h_x, MEM_DEVICE); // Make a copy to device first.

  // Threshold on host copy.
  h_x.HardThreshold(2.5);
  EXPECT_EQ(h_x.DebugString(), "0 0 0 0 0 3 -3 4 -4 5 -5 ");

  // Threshold on device copy.
  d_x.HardThreshold(2.5);
  h_x.Clear();
  EXPECT_EQ(h_x.DebugString(), "0 0 0 0 0 0 0 0 0 0 0 ");
  h_x.CopyFrom(d_x);
  EXPECT_EQ(h_x.DebugString(), "0 0 0 0 0 3 -3 4 -4 5 -5 ");
}

} // namespace gi

int main(int argc, char **argv) {
  gi::MainInit(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  gi::EngineOptions opt;
  gi::Engine engine(opt);
  return RUN_ALL_TESTS();
}
