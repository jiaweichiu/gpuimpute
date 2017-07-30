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
#include "base.h"

namespace gi {

struct ImputeOptions {
  string output_filename;
  string train_filename;
  string train_t_filename;
  string test_filename;
  string train_perm_filename;

  float sv_threshold = 5.0;
  int k = 32;
  int num_gram = 1;
  bool use_gpu = false;

  // If true, populate solution with random numbers initially.
  // Otherwise, just zero out.
  bool randomize_init = false;

  // Evaluate error, state etc every this many iterations.
  int log_every_n = 100;

  // Max running time in seconds.
  double max_time = 300;

  // If true, do soft threshold. Otherwise, do hard threshold.
  bool soft_threshold = true;
};

void Impute(const ImputeOptions &opt);

} // namespace gi