// Dummy program to see if we can start and stop CUDA, Magma properly.
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "base.h"

TEST(BasicTest, Basic) { EXPECT_TRUE(1 == 1); }

int main(int argc, char **argv) {
  gi::MainInit(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  gi::EngineOptions opt;
  gi::Engine engine(opt);
  return RUN_ALL_TESTS();
}
