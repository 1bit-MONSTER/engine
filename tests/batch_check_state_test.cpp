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

// batch_check_state_test.cpp -- the NPU_BATCH_CHECK oracle must leave the forward exactly as it found it (engine#1905).
//
// The oracle replays a batch with sequential step() calls before the batched pass runs, then has to put the state
// back. It used reset(), which is the right state only at start_pos 0: for a later chunk it emptied the prefix, so the
// replay compared "position p from scratch" with "position p continuing the prefix" (a divergence on every chunk but
// the first) and the batched pass itself then ran with no prefix -- the debug flag changed the answer it was checking.
//
// Host-only: the forward runs in host-GEMM mode (no device, no XRT).  Needs a Q4NX container, so it is skipped (77)
// unless ONEBIT_TEST_Q4NX points at one:
//
//     ONEBIT_TEST_Q4NX=~/models/npu/Qwen3-0.6B ./batch_check_state_test
//
// NPU_BATCH_CHECK is read once per process (a static in step_batch), so it is set before the first call and the
// reference is plain step(), which the oracle does not touch.  Checked here, for a prompt split the way the serve path
// splits it (a chunk, then a later chunk, then the one-token tail):
//   [1] after every chunk the forward holds exactly the positions run so far -- a reset in the oracle leaves it
//       holding only the last chunk's;
//   [2] the last chunk's logits equal the sequential reference's at that position, bit for bit: in host-GEMM mode
//       both sides are the same arithmetic in the same order, so any tolerance would only hide a one-row prefix leak.
//
// Where it runs: the row is registered under if(ONEBIT_NPU) (it links onebit_npu_forward) and ci.yml configures
// without it, so it runs on a dev box with ONEBIT_NPU=ON and a container, not in CI.
#include "q4nx_forward.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

double rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 1e9;
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); i++) {
        num += std::fabs((double)a[i] - (double)b[i]);
        den += std::fabs((double)a[i]);
    }
    return num / (den + 1e-30);
}

int argmax(const std::vector<float>& v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); i++)
        if (v[i] > v[best]) best = (int)i;
    return best;
}

int fails = 0;
void check(bool cond, const std::string& what) {
    std::printf("%s  %s\n", cond ? "PASS" : "FAIL", what.c_str());
    if (!cond) fails++;
}

// The forward holds exactly `n` positions: a snapshot of n succeeds and one of n + 1 does not.
bool holds_exactly(const Q4nxNpuForward& f, int n) {
    Q4nxNpuForward::StateSnapshot s;
    return f.snapshot_state(n, s) && !f.snapshot_state(n + 1, s);
}

}  // namespace

int main() {
    const char* dir = std::getenv("ONEBIT_TEST_Q4NX");
    if (!dir || !*dir) {
        std::printf("SKIP: set ONEBIT_TEST_Q4NX to a Q4NX model directory to run this\n");
        return 77;
    }
    setenv("NPU_BATCH_CHECK", "1", 1);   // before the first step_batch: the flag is read once per process
    const std::string model = std::string(dir) + "/model.q4nx";
    Q4nxNpuForward fwd;
    if (!fwd.init_host(model.c_str(), std::string(dir) + "/kernels")) {
        std::printf("SKIP: init_host failed for %s (%s)\n", model.c_str(), fwd.last_error());
        return 77;
    }

    // Values do not matter; the split does. Three chunks: [0,3), [3,5), [5,6) -- the last is the serve path's
    // split-off final token, which is a step_batch at start_pos > 0 on every request.
    const std::vector<int> ids = {1, 2, 3, 4, 5, 6};
    const std::vector<std::pair<int, int>> chunks = {{0, 3}, {3, 5}, {5, 6}};

    fwd.reset();
    std::vector<float> ref;
    for (size_t m = 0; m < ids.size(); m++) {
        if (!fwd.step(ids[m], (int)m, ref)) {
            std::printf("FAIL  sequential step %zu failed: %s\n", m, fwd.last_error());
            return 1;
        }
    }

    fwd.reset();
    std::vector<float> logits;
    for (const auto& c : chunks) {
        const std::vector<int> chunk(ids.begin() + c.first, ids.begin() + c.second);
        if (!fwd.step_batch(chunk, c.first, logits)) {
            std::printf("FAIL  step_batch(start=%d, M=%zu) failed: %s\n", c.first, chunk.size(), fwd.last_error());
            return 1;
        }
        check(holds_exactly(fwd, c.second),
              "after the chunk at start " + std::to_string(c.first) + " the forward holds exactly " +
                  std::to_string(c.second) + " positions");
    }
    const double rel = rel_diff(logits, ref);
    char msg[200];
    std::snprintf(msg, sizeof msg, "last chunk's logits vs the sequential reference: rel-L1 %.2e, argmax %d vs %d", rel,
                  argmax(logits), argmax(ref));
    check(argmax(logits) == argmax(ref) && rel == 0.0, msg);

    std::printf("%s (%d failed)\n", fails == 0 ? "ALL PASS" : "FAILED", fails);
    return fails == 0 ? 0 : 1;
}
