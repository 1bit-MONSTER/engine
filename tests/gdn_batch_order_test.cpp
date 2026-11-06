// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// gdn_batch_order_test.cpp -- the batched GDN recurrence is head-major (one parallel region per layer) instead of
// row-major (one region per layer AND row), and that reordering must not move a single element of CORE.
//
// The engine asserts it itself under NPU_GDN_CHECK=1: both nestings run through the same per-(head,row) body and
// the results are compared bit for bit; a mismatch fails the call with an err_.  So this test only has to enable
// the assertion and make the engine run the batched path on a container with GDN layers.
//
// Host-only (no device, no XRT) and skipped unless ONEBIT_TEST_Q4NX names a hybrid container:
//
//     ONEBIT_TEST_Q4NX=~/models/npu/d3-lite-conv ./gdn_batch_order_test
//
// Why this and not the row-major path only: an M-row batch is 192 parallel regions per verify pass and 6144 per
// M=128 prefill chunk as written before, and region count is what makes a loaded box pay -- measured 16.8 vs 4.1
// ms/row at equal contention, bit-identical (`~/briefs/mtploop-gdn-row-bench.cpp`).  This test is the reason the
// change can be claimed exact rather than argued exact.
#include "q4nx_forward.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    const char* dir = std::getenv("ONEBIT_TEST_Q4NX");
    if (!dir || !*dir) {
        std::printf("SKIP: set ONEBIT_TEST_Q4NX to a Q4NX model directory with linear-attention layers\n");
        return 77;
    }
    // Ask the engine to compare its two nestings on every GDN layer it touches.
    setenv("NPU_GDN_CHECK", "1", 1);
    const std::string model = std::string(dir) + "/model.q4nx";
    Q4nxNpuForward fwd;
    if (!fwd.init_host(model.c_str(), std::string(dir) + "/kernels")) {
        std::printf("SKIP: init_host failed for %s (%s)\n", model.c_str(), fwd.last_error());
        return 77;
    }
    // A short batch at position 0: enough rows for row-major to differ from head-major if it were going to, and
    // cheap enough that a 27B container is not needed for CI.
    const std::vector<int> batch = {1, 2, 3, 4};
    std::vector<float> all;
    if (!fwd.step_batch_all(batch, 0, all)) {
        std::printf("FAIL  step_batch_all failed: %s\n", fwd.last_error());
        return 1;
    }
    if (all.empty()) {
        std::printf("FAIL  step_batch_all returned no logits\n");
        return 1;
    }
    std::printf("PASS  %zu tokens x %zu logits: every GDN layer's head-major nesting matched row-major bit for "
                "bit\n", batch.size(), all.size() / batch.size());
    return 0;
}
