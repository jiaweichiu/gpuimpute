#include <gtest/gtest.h>

#include "base.h"
#include "mat.h"

namespace gi {

namespace {

SMat* NewSMat(int m, int n, float seed) {
  SMat* a = new SMat(m, n, MEM_HOST);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      a->set(i, j, seed++);
    }
  }
  return a;
}

SMat* NewSMatAlt(int m, int n, MemType mem_type) {
  SMat a(m, n, MEM_HOST);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      a.set(i, j, static_cast<float>(i + j * m));
    }
  }
  return new SMat(a, mem_type);
}

}  // namespace

TEST(SMatTest, Copy) {
  unique_ptr<SMat> h_x(NewSMat(3, 5, 1.1));
  EXPECT_EQ(h_x->DebugString(),
            "1.1 2.1 3.1 4.1 5.1 \n"
            "6.1 7.1 8.1 9.1 10.1 \n"
            "11.1 12.1 13.1 14.1 15.1 \n");

  // Host to host copy. Copy only a subset.
  SMat h_y(2, 4, MEM_HOST);
  h_y.CopyFrom(*h_x);
  EXPECT_EQ(h_y.DebugString(),
            "1.1 2.1 3.1 4.1 \n"
            "6.1 7.1 8.1 9.1 \n");

  // Copy from host to device.
  SMat d_y(2, 4, MEM_DEVICE);
  d_y.CopyFrom(*h_x);
  SMat h_z(3, 5, MEM_HOST);
  h_z.Clear();
  h_z.CopyFrom(d_y);
  EXPECT_EQ(h_z.DebugString(),
            "1.1 2.1 3.1 4.1 0 \n"
            "6.1 7.1 8.1 9.1 0 \n"
            "0 0 0 0 0 \n");
}

TEST(SMatTest, Fill) {
  SMat h_x(3, 2, MEM_HOST);
  h_x.Fill(2.0);
  EXPECT_EQ(h_x.DebugString(),
            "2 2 \n"
            "2 2 \n"
            "2 2 \n");

  SMat d_x(h_x, MEM_DEVICE);
  d_x.Fill(3.0);
  h_x.CopyFrom(d_x);
  EXPECT_EQ(h_x.DebugString(),
            "3 3 \n"
            "3 3 \n"
            "3 3 \n");
}

TEST(SMatTest, Transpose) {
  unique_ptr<SMat> h_x(NewSMat(3, 5, 1.1));
  SMat h_y(5, 3, MEM_HOST);
  h_y.SetToSum(2.0, TRANS, *h_x, 0, TRANS, *h_x);
  EXPECT_EQ(h_y.DebugString(),
            "2.2 12.2 22.2 \n"
            "4.2 14.2 24.2 \n"
            "6.2 16.2 26.2 \n"
            "8.2 18.2 28.2 \n"
            "10.2 20.2 30.2 \n");

  // Do the same test on device.
  SMat d_x(*h_x, MEM_DEVICE);
  SMat d_y(5, 3, MEM_DEVICE);
  d_y.SetToSum(2.0, TRANS, d_x, 0, TRANS, d_x);
  h_y.Clear();
  h_y.CopyFrom(d_y);
  EXPECT_EQ(h_y.DebugString(),
            "2.2 12.2 22.2 \n"
            "4.2 14.2 24.2 \n"
            "6.2 16.2 26.2 \n"
            "8.2 18.2 28.2 \n"
            "10.2 20.2 30.2 \n");
}

TEST(SMatTest, TransposeAlt) {
  unique_ptr<SMat> h_x(NewSMat(3, 5, 1.1));
  SMat h_y(5, 3, MEM_HOST);
  h_y.SetToSum(0, TRANS, *h_x, 2.0, TRANS, *h_x);
  EXPECT_EQ(h_y.DebugString(),
            "2.2 12.2 22.2 \n"
            "4.2 14.2 24.2 \n"
            "6.2 16.2 26.2 \n"
            "8.2 18.2 28.2 \n"
            "10.2 20.2 30.2 \n");

  // Do the same test on device.
  SMat d_x(*h_x, MEM_DEVICE);
  SMat d_y(5, 3, MEM_DEVICE);
  d_y.SetToSum(0, TRANS, d_x, 2.0, TRANS, d_x);
  h_y.Clear();
  h_y.CopyFrom(d_y);
  EXPECT_EQ(h_y.DebugString(),
            "2.2 12.2 22.2 \n"
            "4.2 14.2 24.2 \n"
            "6.2 16.2 26.2 \n"
            "8.2 18.2 28.2 \n"
            "10.2 20.2 30.2 \n");
}

TEST(SMatTest, SumNoTransNoTrans) {
  unique_ptr<SMat> h_x(NewSMat(3, 5, 1.1));
  SMat h_y(3, 5, MEM_HOST);
  h_y.Fill(3.0);
  h_y.SetToSum(5.0, NO_TRANS, h_y, 2.0, NO_TRANS, *h_x);
  EXPECT_EQ(h_y.DebugString(),
            "17.2 19.2 21.2 23.2 25.2 \n"
            "27.2 29.2 31.2 33.2 35.2 \n"
            "37.2 39.2 41.2 43.2 45.2 \n");

  // Do the same test on device.
  SMat d_x(*h_x, MEM_DEVICE);
  SMat d_y(3, 5, MEM_DEVICE);
  d_y.Fill(3.0);
  d_y.SetToSum(5.0, NO_TRANS, d_y, 2.0, NO_TRANS, d_x);
  h_y.Clear();
  h_y.CopyFrom(d_y);
  EXPECT_EQ(h_y.DebugString(),
            "17.2 19.2 21.2 23.2 25.2 \n"
            "27.2 29.2 31.2 33.2 35.2 \n"
            "37.2 39.2 41.2 43.2 45.2 \n");
}

TEST(SMatTest, SumNoTransTrans) {
  unique_ptr<SMat> h_x(NewSMat(3, 5, 1.1));
  SMat h_y(5, 3, MEM_HOST);
  h_y.Fill(3.0);
  h_y.SetToSum(5.0, NO_TRANS, h_y, 2.0, TRANS, *h_x);
  EXPECT_EQ(h_y.DebugString(),
            "17.2 27.2 37.2 \n"
            "19.2 29.2 39.2 \n"
            "21.2 31.2 41.2 \n"
            "23.2 33.2 43.2 \n"
            "25.2 35.2 45.2 \n");

  // Do the same test on device.
  SMat d_x(*h_x, MEM_DEVICE);
  SMat d_y(5, 3, MEM_DEVICE);
  d_y.Fill(3.0);
  d_y.SetToSum(5.0, NO_TRANS, d_y, 2.0, TRANS, d_x);
  h_y.Clear();
  h_y.CopyFrom(d_y);
  EXPECT_EQ(h_y.DebugString(),
            "17.2 27.2 37.2 \n"
            "19.2 29.2 39.2 \n"
            "21.2 31.2 41.2 \n"
            "23.2 33.2 43.2 \n"
            "25.2 35.2 45.2 \n");
}

TEST(SMatTest, SumTransNoTrans) {
  unique_ptr<SMat> h_x(NewSMat(3, 5, 1.1));
  SMat h_y(5, 3, MEM_HOST);
  h_y.Fill(3.0);
  h_y.SetToSum(2.0, TRANS, *h_x, 5.0, NO_TRANS, h_y);
  EXPECT_EQ(h_y.DebugString(),
            "17.2 27.2 37.2 \n"
            "19.2 29.2 39.2 \n"
            "21.2 31.2 41.2 \n"
            "23.2 33.2 43.2 \n"
            "25.2 35.2 45.2 \n");

  // Do the same test on device.
  SMat d_x(*h_x, MEM_DEVICE);
  SMat d_y(5, 3, MEM_DEVICE);
  d_y.Fill(3.0);
  d_y.SetToSum(2.0, TRANS, d_x, 5.0, NO_TRANS, d_y);
  h_y.Clear();
  h_y.CopyFrom(d_y);
  EXPECT_EQ(h_y.DebugString(),
            "17.2 27.2 37.2 \n"
            "19.2 29.2 39.2 \n"
            "21.2 31.2 41.2 \n"
            "23.2 33.2 43.2 \n"
            "25.2 35.2 45.2 \n");
}

TEST(SMatTest, SumTransTrans) {
  unique_ptr<SMat> h_x(NewSMat(3, 5, 1.1));
  SMat h_y(5, 3, MEM_HOST);
  SMat h_z(3, 5, MEM_HOST);
  h_z.Fill(3.0);
  h_y.SetToSum(2.0, TRANS, *h_x, 5.0, TRANS, h_z);
  EXPECT_EQ(h_y.DebugString(),
            "17.2 27.2 37.2 \n"
            "19.2 29.2 39.2 \n"
            "21.2 31.2 41.2 \n"
            "23.2 33.2 43.2 \n"
            "25.2 35.2 45.2 \n");

  // Do the same test on device.
  SMat d_x(*h_x, MEM_DEVICE);
  SMat d_y(5, 3, MEM_DEVICE);
  SMat d_z(h_z, MEM_DEVICE);
  d_z.Fill(3.0);
  d_y.SetToSum(2.0, TRANS, d_x, 5.0, TRANS, d_z);
  h_y.Clear();
  h_y.CopyFrom(d_y);
  EXPECT_EQ(h_y.DebugString(),
            "17.2 27.2 37.2 \n"
            "19.2 29.2 39.2 \n"
            "21.2 31.2 41.2 \n"
            "23.2 33.2 43.2 \n"
            "25.2 35.2 45.2 \n");
}

TEST(SMatTest, MultiplyByDiag) {
  unique_ptr<SMat> h_x(NewSMat(3, 5, 1.1));
  SVec h_d(3, MEM_HOST);  // Diagonal.
  h_d.set(0, 2.0);
  h_d.set(1, -1.0);
  h_d.set(2, 3.0);

  h_x->SetToProduct(h_d, *h_x);  // Multiply diag on left.
  EXPECT_EQ(h_x->DebugString(),
            "2.2 4.2 6.2 8.2 10.2 \n"
            "-6.1 -7.1 -8.1 -9.1 -10.1 \n"
            "33.3 36.3 39.3 42.3 45.3 \n");

  // Do the same test on device.
  h_x.reset(NewSMat(3, 5, 1.1));
  SMat d_x(*h_x, MEM_DEVICE);
  SVec d_d(h_d, MEM_DEVICE);
  d_x.SetToProduct(d_d, d_x);
  h_x->Clear();
  h_x->CopyFrom(d_x);
  EXPECT_EQ(h_x->DebugString(),
            "2.2 4.2 6.2 8.2 10.2 \n"
            "-6.1 -7.1 -8.1 -9.1 -10.1 \n"
            "33.3 36.3 39.3 42.3 45.3 \n");
}

TEST(SMatTest, MultiplyByScalar) {
  unique_ptr<SMat> h_x(NewSMat(3, 5, 1.1));

  h_x->SetToProduct(2.0, *h_x);
  EXPECT_EQ(h_x->DebugString(),
            "2.2 4.2 6.2 8.2 10.2 \n"
            "12.2 14.2 16.2 18.2 20.2 \n"
            "22.2 24.2 26.2 28.2 30.2 \n");

  // Do the same test on device.
  h_x.reset(NewSMat(3, 5, 1.1));
  SMat d_x(*h_x, MEM_DEVICE);
  d_x.SetToProduct(2.0, d_x);
  h_x->Clear();
  h_x->CopyFrom(d_x);
  EXPECT_EQ(h_x->DebugString(),
            "2.2 4.2 6.2 8.2 10.2 \n"
            "12.2 14.2 16.2 18.2 20.2 \n"
            "22.2 24.2 26.2 28.2 30.2 \n");
}

TEST(SMatTest, DenseMultiply) {
  unique_ptr<SMat> h_x(NewSMat(3, 5, 1.1));
  unique_ptr<SMat> h_y(NewSMat(4, 5, -2.0));

  SMat h_z(3, 4, MEM_HOST);
  h_z.SetToProduct(2.0, NO_TRANS, *h_x, TRANS, *h_y, 0);
  EXPECT_EQ(h_z.DebugString(),
            "20 175 330 485 \n"
            "20 425 830 1235 \n"
            "20 675 1330 1985 \n");

  // Do the same test on device.
  SMat d_x(*h_x, MEM_DEVICE);
  SMat d_y(*h_y, MEM_DEVICE);
  SMat d_z(3, 4, MEM_DEVICE);
  d_z.SetToProduct(2.0, NO_TRANS, d_x, TRANS, d_y, 0);
  h_z.Clear();
  h_z.CopyFrom(d_z);
  EXPECT_EQ(h_z.DebugString(),
            "20 175 330 485 \n"
            "20 425 830 1235 \n"
            "20 675 1330 1985 \n");
}

TEST(SSpMatTest, Basic) {
  istringstream is("6 10 5\n1 3 5 -2 -4\n1 3 5 9 6\n0 2 2 4 4 4 5");
  SSpMat h_a(is);  // On host.
  SSpMat d_a(h_a, MEM_DEVICE);
  SSpMat h_b(h_a, MEM_HOST);
  EXPECT_EQ(h_b.DebugString(),
            "m=6 n=10 nnz=5\n"
            "0 2 2 4 4 4 5 \n"
            "0 0 2 2 5 \n"
            "1 3 5 9 6 \n"
            "1 3 5 -2 -4 \n");
}

TEST(SSpMatTest, MultiplyNoTransNoTrans) {
  istringstream is("6 10 5\n1 3 5 -2 -4\n1 3 5 9 6\n0 2 2 4 4 4 5");
  SSpMat h_a(is);
  SSpMat d_a(h_a, MEM_DEVICE);

  // Test on host.
  unique_ptr<SMat> h_b(NewSMatAlt(10, 2, MEM_HOST));
  SMat h_c(6, 2, MEM_HOST);
  h_c.SetToProduct(1.0, NO_TRANS, h_a, NO_TRANS, *h_b, 0);
  EXPECT_EQ(h_c.DebugString(),
            "10 50 \n"
            "0 0 \n"
            "7 37 \n"
            "0 0 \n"
            "0 0 \n"
            "-24 -64 \n");

  // Test on device.
  unique_ptr<SMat> d_b(NewSMatAlt(10, 2, MEM_DEVICE));
  SMat d_c(6, 2, MEM_DEVICE);
  d_c.SetToProduct(1.0, NO_TRANS, d_a, NO_TRANS, *d_b, 0);
  h_c.Clear();
  h_c.CopyFrom(d_c);
  EXPECT_EQ(h_c.DebugString(),
            "10 50 \n"
            "0 0 \n"
            "7 37 \n"
            "0 0 \n"
            "0 0 \n"
            "-24 -64 \n");
}

TEST(SSpMatTest, MultiplyTransNoTrans) {
  istringstream is("6 10 5\n1 3 5 -2 -4\n1 3 5 9 6\n0 2 2 4 4 4 5");
  SSpMat h_a(is);
  SSpMat d_a(h_a, MEM_DEVICE);

  // Test on host.
  unique_ptr<SMat> h_b(NewSMatAlt(6, 2, MEM_HOST));
  SMat h_c(10, 2, MEM_HOST);
  h_c.SetToProduct(1.0, TRANS, h_a, NO_TRANS, *h_b, 0);
  EXPECT_EQ(h_c.DebugString(),
            "0 0 \n"
            "0 6 \n"
            "0 0 \n"
            "0 18 \n"
            "0 0 \n"
            "10 40 \n"
            "-20 -44 \n"
            "0 0 \n"
            "0 0 \n"
            "-4 -16 \n");

  // Test on device.
  unique_ptr<SMat> d_b(NewSMatAlt(6, 2, MEM_DEVICE));
  SMat d_c(10, 2, MEM_DEVICE);
  d_c.SetToProduct(1.0, TRANS, d_a, NO_TRANS, *d_b, 0);
  h_c.Clear();
  h_c.CopyFrom(d_c);
  EXPECT_EQ(h_c.DebugString(),
            "0 0 \n"
            "0 6 \n"
            "0 0 \n"
            "0 18 \n"
            "0 0 \n"
            "10 40 \n"
            "-20 -44 \n"
            "0 0 \n"
            "0 0 \n"
            "-4 -16 \n");
}

TEST(SSpMatTest, MultiplyNoTransTrans) {
  istringstream is("6 10 5\n1 3 5 -2 -4\n1 3 5 9 6\n0 2 2 4 4 4 5");
  SSpMat h_a(is);
  SSpMat d_a(h_a, MEM_DEVICE);

  // Test on host.
  unique_ptr<SMat> h_b(NewSMatAlt(2, 10, MEM_HOST));
  SMat h_c(6, 2, MEM_HOST);
  h_c.SetToProduct(1.0, NO_TRANS, h_a, TRANS, *h_b, 0);
  EXPECT_EQ(h_c.DebugString(),
            "20 24 \n"
            "0 0 \n"
            "14 17 \n"
            "0 0 \n"
            "0 0 \n"
            "-48 -52 \n");

  // Test on device.
  unique_ptr<SMat> d_b(NewSMatAlt(2, 10, MEM_DEVICE));
  SMat d_c(6, 2, MEM_DEVICE);
  d_c.SetToProduct(1.0, NO_TRANS, d_a, TRANS, *d_b, 0);
  h_c.Clear();
  h_c.CopyFrom(d_c);
  EXPECT_EQ(h_c.DebugString(),
            "20 24 \n"
            "0 0 \n"
            "14 17 \n"
            "0 0 \n"
            "0 0 \n"
            "-48 -52 \n");
}

TEST(SSpMatTest, SampleAndUpdate) {
  SMat h_ut(4, 4, MEM_HOST);
  h_ut.set(0, 0, 1);
  h_ut.set(1, 0, 2);
  h_ut.set(2, 0, -1);
  h_ut.set(3, 0, -6);
  h_ut.set(0, 1, -2);
  h_ut.set(1, 1, 3);
  h_ut.set(2, 1, 6);
  h_ut.set(3, 1, 8);
  h_ut.set(0, 2, 9);
  h_ut.set(1, 2, 5);
  h_ut.set(2, 2, -3);
  h_ut.set(3, 2, -1);
  h_ut.set(0, 3, 2);
  h_ut.set(1, 3, -3);
  h_ut.set(2, 3, -6);
  h_ut.set(3, 3, 5);

  SMat h_vt(4, 3, MEM_HOST);
  h_vt.set(0, 0, -2);
  h_vt.set(1, 0, 7);
  h_vt.set(2, 0, 1);
  h_vt.set(3, 0, 2);
  h_vt.set(0, 1, 3);
  h_vt.set(1, 1, -8);
  h_vt.set(2, 1, 8);
  h_vt.set(3, 1, 7);
  h_vt.set(0, 2, 7);
  h_vt.set(1, 2, -3);
  h_vt.set(2, 2, -4);
  h_vt.set(3, 2, -4);

  SVec h_s(4, MEM_HOST);
  h_s.set(0, 5);
  h_s.set(1, -9);
  h_s.set(2, 1);
  h_s.set(3, 3);

  istringstream is(
      "4 3 5\n"
      "100.0 200.0 300.0 400.0 500.0\n"
      "0 2 1 1 2\n"
      "0 0 2 3 5\n");
  SSpMat h_a(is);
  SSpMat d_a(h_a, MEM_DEVICE);
  h_a.SampleAndUpdate(2.0, h_ut, h_vt, h_s, 1.0);
  EXPECT_EQ(h_a.DebugString(),
            "m=4 n=3 nnz=5\n"
            "0 0 2 3 5 \n"
            "1 1 2 3 3 \n"
            "0 2 1 1 2 \n"
            "-130 -18 1200 142 406 \n");

  // Run the same test on device.
  SMat d_ut(h_ut, MEM_DEVICE);
  SMat d_vt(h_vt, MEM_DEVICE);
  SVec d_s(h_s, MEM_DEVICE);
  d_a.SampleAndUpdate(2.0, d_ut, d_vt, d_s, 1.0);
  h_a.ClearValues();
  EXPECT_EQ(h_a.DebugString(),
            "m=4 n=3 nnz=5\n"
            "0 0 2 3 5 \n"
            "1 1 2 3 3 \n"
            "0 2 1 1 2 \n"
            "0 0 0 0 0 \n");

  h_a.CopyValuesFrom(d_a);
  EXPECT_EQ(h_a.DebugString(),
            "m=4 n=3 nnz=5\n"
            "0 0 2 3 5 \n"
            "1 1 2 3 3 \n"
            "0 2 1 1 2 \n"
            "-130 -18 1200 142 406 \n");
}

}  // namespace gi

int main(int argc, char **argv) {
  gi::MainInit(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  gi::EngineOptions opt;
  gi::Engine engine(opt);
  return RUN_ALL_TESTS();
}
