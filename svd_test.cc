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

#include "svd.h"

namespace gi {

void Compare(const SMat &a, const vector<vector<float>> &b) {
  EXPECT_EQ(a.m(), b.size());
  for (int i = 0; i < a.m(); ++i) {
    EXPECT_EQ(a.n(), b[i].size());
    for (int j = 0; j < a.n(); ++j) {
      EXPECT_NEAR(a.get(i, j), b[i][j], 1e-3);
    }
  }
}

void Compare(const SVec &a, const vector<float> &b) {
  EXPECT_EQ(a.size(), b.size());
  for (int i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(a.get(i), b[i], 1e-3);
  }
}

class SVDTest : public ::testing::Test {
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

    h_u_.reset(new SMat(4, 4, MEM_HOST));
    h_vt_.reset(new SMat(2, 2, MEM_HOST));
    h_s_.reset(new SVec(2, MEM_HOST));

    d_a_.reset(new SMat(*h_a_, MEM_DEVICE));
    d_u_.reset(new SMat(4, 4, MEM_DEVICE));
    d_vt_.reset(new SMat(2, 2, MEM_DEVICE));
    d_s_.reset(new SVec(2, MEM_DEVICE));
  }

  void CheckAnswers() {
    Compare(*h_u_, {{-0.222642, -0.187149, -0.952639, -0.0887906},
                    {-0.0221634, -0.677277, 0.0700023, 0.732055},
                    {-0.97375, 0.0276561, 0.224507, -0.0253624},
                    {-0.0418402, 0.710989, -0.192807, 0.674958}});
    Compare(*h_vt_, {{-0.840298, -0.542124}, {0.542124, -0.840298}});
    Compare(*h_s_, {10.8843, 6.27153});
  }

  unique_ptr<SMat> h_a_;
  unique_ptr<SMat> h_u_;
  unique_ptr<SMat> h_vt_;
  unique_ptr<SVec> h_s_;

  unique_ptr<SMat> d_a_;
  unique_ptr<SMat> d_u_;
  unique_ptr<SMat> d_vt_;
  unique_ptr<SVec> d_s_;
};

TEST_F(SVDTest, Device) {
  SVDHelper svd_helper(*d_a_, d_u_.get(), d_vt_.get(), d_s_.get());
  svd_helper.Compute();
  h_u_->CopyFrom(*d_u_);
  h_vt_->CopyFrom(*d_vt_);
  h_s_->CopyFrom(*d_s_);
  CheckAnswers();
}

TEST_F(SVDTest, Host) {
  SVDHelper svd_helper(*h_a_, h_u_.get(), h_vt_.get(), h_s_.get());
  svd_helper.Compute();
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
