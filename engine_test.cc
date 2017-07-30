#include <gtest/gtest.h>

#include "common.h"
#include "engine.h"

namespace gi {

TEST(BasicTest, Basic) {
  EXPECT_TRUE(1 == 1);
}

}  // namespace gi

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  ::testing::InitGoogleTest(&argc, argv);
  
  gi::EngineOptions opt;
  gi::Engine engine(opt);
  return RUN_ALL_TESTS();
}
