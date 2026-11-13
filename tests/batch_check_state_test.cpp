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
// Where it runs: in CI (engine#1946), built against the public runtime stub and run by tests/batch_check_state_ci.py
// on a two-layer container that script fabricates; and on a dev box with ONEBIT_NPU=ON, on the container named by
// ONEBIT_TEST_Q4NX.
#include "file_create.h"
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
    // A caller that has its own reference forward of the container (tests/batch_check_state_ci.py lfm3) asks for the
    // sequential logits: batched == sequential cannot see a rule both paths get wrong, a reference can.
    if (const char* out = std::getenv("ONEBIT_TEST_LOGITS_OUT")) {
        FILE* f = onebit_fopen_private_wb(out);   // 0600, never through a symlink (file_create.h)
        if (!f || std::fwrite(ref.data(), sizeof(float), ref.size(), f) != ref.size()) {
            std::printf("FAIL  cannot write the sequential logits to %s\n", out);
            if (f) std::fclose(f);
            return 1;
        }
        std::fclose(f);
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

    // engine#1780: the verify a speculative decoder runs. A prefix, then k drafted rows in one step_batch_all(): every
    // row's logits must be the sequential step's at that position, bit for bit; then a partial accept rolls the cache
    // back to the accepted rows, and the next step must continue exactly as a forward that only ever saw those rows.
    // The rollback is where a model with two attention geometries (Gemma4) lost half its full layers' positions.
    {
        const std::vector<int> prefix = {1, 2, 3}, drafts = {4, 5, 6, 7};
        const int next_tok = 8, accepted = 2;
        const int P = (int)prefix.size(), K = (int)drafts.size();
        fwd.reset();
        std::vector<float> seq, rows_ref;
        for (int i = 0; i < P; i++)
            if (!fwd.step(prefix[(size_t)i], i, seq)) { std::printf("FAIL  prefix step: %s\n", fwd.last_error()); return 1; }
        for (int i = 0; i < K; i++) {
            if (!fwd.step(drafts[(size_t)i], P + i, seq)) { std::printf("FAIL  draft step: %s\n", fwd.last_error()); return 1; }
            rows_ref.insert(rows_ref.end(), seq.begin(), seq.end());
        }
        // The forward that only ever saw the accepted rows, then the next token.
        fwd.reset();
        std::vector<float> next_ref;
        for (int i = 0; i < P; i++) fwd.step(prefix[(size_t)i], i, seq);
        for (int i = 0; i < accepted; i++) fwd.step(drafts[(size_t)i], P + i, seq);
        if (!fwd.step(next_tok, P + accepted, next_ref)) { std::printf("FAIL  reference step: %s\n", fwd.last_error()); return 1; }

        fwd.reset();
        for (int i = 0; i < P; i++) fwd.step(prefix[(size_t)i], i, seq);
        std::vector<float> rows;
        fwd.begin_verify(K, P);
        if (!fwd.step_batch_all(drafts, P, rows)) { std::printf("FAIL  step_batch_all: %s\n", fwd.last_error()); return 1; }
        check(rows.size() == rows_ref.size() && rows == rows_ref,
              "verify: every drafted row's logits are the sequential step's, bit for bit (" + std::to_string(K) + " rows)");
        if (!fwd.commit_verify_rows(accepted)) { std::printf("FAIL  commit_verify_rows: %s\n", fwd.last_error()); return 1; }
        check(holds_exactly(fwd, P + accepted),
              "verify: a partial accept leaves exactly prefix + accepted positions (" + std::to_string(P + accepted) + ")");
        std::vector<float> next;
        if (!fwd.step(next_tok, P + accepted, next)) { std::printf("FAIL  step after commit: %s\n", fwd.last_error()); return 1; }
        check(next == next_ref, "verify: the step after a partial accept equals the forward that saw only those rows");
    }

    std::printf("%s (%d failed)\n", fails == 0 ? "ALL PASS" : "FAILED", fails);
    return fails == 0 ? 0 : 1;
}
