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
#include <gtest/gtest.h>

#include "qr.h"

namespace gi {

namespace {

void Compare(const SMat &a, const vector<vector<float>> &b) {
  EXPECT_EQ(a.m(), b.size());
  for (int i = 0; i < a.m(); ++i) {
    EXPECT_EQ(a.n(), b[i].size());
    for (int j = 0; j < a.n(); ++j) {
      EXPECT_NEAR(a.get(i, j), b[i][j], 1e-3);
    }
  }
}

} // namespace

class QRTest : public ::testing::Test {
protected:
  void SetUp() {
    // Prepare our inputs and outputs.
    h_a_.reset(new SMat(4, 2, MEM_HOST));
    h_a_->set(0, 0, 1.4);
    h_a_->set(0, 1, 2.3);
    h_a_->set(1, 0, -2.1);
    h_a_->set(1, 1, 3.7);
    h_a_->set(2, 0, 9.0);
    h_a_->set(2, 1, 5.6);
    h_a_->set(3, 0, 2.8);
    h_a_->set(3, 1, -3.5);

    d_a_.reset(new SMat(*h_a_, MEM_DEVICE));
  }

  void CheckAnswers() {
    // Normalize each column.
    for (int i = 0; i < 2; ++i) { // Column i.
      if (h_a_->get(0, i) < 0) {
        for (int j = 0; j < 4; ++j) { // Row j.
          h_a_->set(j, i, -h_a_->get(j, i));
        }
      }
    }
    Compare(*h_a_, {{0.143478, 0.252999},
                    {-0.215218, 0.642555},
                    {0.922362, 0.313373},
                    {0.286957, -0.651853}});
  }

  unique_ptr<SMat> h_a_;
  unique_ptr<SMat> d_a_;
};

TEST_F(QRTest, Host) {
  QRHelper qr_helper(h_a_.get());
  qr_helper.Compute();
  CheckAnswers();
}

TEST_F(QRTest, Device1) {
  QRHelper qr_helper(d_a_.get(), 1);
  qr_helper.Compute();
  h_a_->CopyFrom(*d_a_);
  CheckAnswers();
}

TEST_F(QRTest, Device2) {
  QRHelper qr_helper(d_a_.get(), 2);
  qr_helper.Compute();
  h_a_->CopyFrom(*d_a_);
  CheckAnswers();
}

TEST_F(QRTest, Device3) {
  QRHelper qr_helper(d_a_.get(), 3);
  qr_helper.Compute();
  h_a_->CopyFrom(*d_a_);
  CheckAnswers();
}

TEST_F(QRTest, Device4) {
  QRHelper qr_helper(d_a_.get(), 4);
  qr_helper.Compute();
  h_a_->CopyFrom(*d_a_);
  CheckAnswers();
}

} // namespace gi

int main(int argc, char **argv) {
  gi::MainInit(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  gi::EngineOptions opt;
  gi::Engine engine(opt);
  return RUN_ALL_TESTS();
}
