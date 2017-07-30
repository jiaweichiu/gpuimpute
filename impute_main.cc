#include "impute.h"
#include "base.h"

DEFINE_string(output_filename, "", "Output file for stats.");

DEFINE_string(train_filename, "", "Filename for training dataset.");
DEFINE_string(train_t_filename, "",
              "Filename for training dataset, transposed.");
DEFINE_string(test_filename, "", "Filename for training dataset.");
DEFINE_string(train_perm_filename, "",
              "Filename for training data permutation.");

DEFINE_bool(use_gpu, true,
            "Use GPU for our computations. If false, we use CPU only.");

DEFINE_int32(
    k, 32,
    "Number of singular values computed per iteration. Should be powers of 2.");

DEFINE_double(sv_threshold, 10.0, "Singular value threshold.");

DEFINE_int32(num_gram, 1,
             "Apply Gram matrix this many times. Recommend either 0 or 1.");

DEFINE_int32(log_every_n, 200, "Report test error every this many iterations.");

DEFINE_string(init, "zero",
              "How to initialize U and V? Possibilities: zero, rand");

DEFINE_double(max_time, 100000, "Maximum time in seconds.");

DEFINE_int32(
    randn, 1000000,
    "SVD is computed by applying A to a random matrix. We often reuse this "
    "matrix from the previous iteration. Every this many iterations, we will "
    "re-initialize this random matrix.");

DEFINE_bool(soft_threshold, true, "Soft threshold? Otherwise, hard threshold.");

DEFINE_bool(randomize_init, false, "Randomize the initial solution?");

DEFINE_bool(accelerated, true, "Use accelerated version?");

namespace gi {

void Main() {
  EngineOptions e_opt;
  Engine engine(e_opt);
  ImputeOptions opt;
  opt.output_filename = FLAGS_output_filename;
  opt.train_filename = FLAGS_train_filename;
  opt.train_t_filename = FLAGS_train_t_filename;
  opt.test_filename = FLAGS_test_filename;
  opt.train_perm_filename = FLAGS_train_perm_filename;

  opt.sv_threshold = FLAGS_sv_threshold;
  opt.k = FLAGS_k;
  opt.num_gram = FLAGS_num_gram;
  opt.use_gpu = FLAGS_use_gpu;
  opt.randomize_init = FLAGS_randomize_init;
  opt.log_every_n = FLAGS_log_every_n;
  opt.max_time = FLAGS_max_time;
  opt.soft_threshold = FLAGS_soft_threshold;
  opt.accelerated = FLAGS_accelerated;

  Impute impute(opt);
  impute.Run();
}

} // namespace gi

int main(int argc, char **argv) {
  gi::MainInit(argc, argv);
  gi::Main();
}
