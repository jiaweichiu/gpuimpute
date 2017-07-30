#include <glog/logging.h>

#include <magma_v2.h>
#include <magma_lapack.h>

#include "engine.h"

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  gi::EngineOptions opt;
  gi::Engine engine(opt);
}
