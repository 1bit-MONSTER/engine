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

// mtp_batch_all_test.cpp -- step_batch_all() must return, for every row, the logits a sequential decode
// would have produced at that position.  That is the whole contract speculative verification rests on: a
// draft token is accepted only when the model's OWN choice at that position matches it, so a per-row
// logit that is merely "close" is not enough -- the argmax has to agree.
//
// Host-only: the forward runs in host-GEMM mode (no device, no XRT).  Needs a Q4NX container, so it is
// skipped (return 77) unless ONEBIT_TEST_Q4NX points at one:
//
//     ONEBIT_TEST_Q4NX=~/models/npu/Qwen3-0.6B ./mtp_batch_all_test
//
// Checked here:
//   [1] row m of the batch equals the logits of a sequential step at position start+m (rel diff reported,
//       argmax must match exactly);
//   [2] the last row equals what plain step_batch() returns -- the batched prefill path is untouched;
//   [3] the argmax of every row is stable across a repeat run (nothing depends on call order).
#include "q4nx_forward.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Q4nxNpuForward is at global scope in this header (npu/forward/include/q4nx_forward.h).

namespace {

double rel_diff(const std::vector<float>& a, const std::vector<float>& b, double* maxabs = nullptr) {
    if (a.size() != b.size() || a.empty()) return 1e9;
    double num = 0, den = 0, ma = 0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = std::fabs((double)a[i] - (double)b[i]);
        num += d;
        den += std::fabs((double)a[i]);
        if (a[i] > ma) ma = a[i];
        if (maxabs && d > *maxabs) *maxabs = d;
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

}  // namespace

int main() {
    const char* dir = std::getenv("ONEBIT_TEST_Q4NX");
    if (!dir || !*dir) {
        std::printf("SKIP: set ONEBIT_TEST_Q4NX to a Q4NX model directory to run this\n");
        return 77;
    }
    const std::string model = std::string(dir) + "/model.q4nx";
    Q4nxNpuForward fwd;
    if (!fwd.init_host(model.c_str(), std::string(dir) + "/kernels")) {
        std::printf("SKIP: init_host failed for %s (%s)\n", model.c_str(), fwd.last_error());
        return 77;   // a container this build cannot read is not a failure of step_batch_all
    }
    std::fprintf(stderr, "[test] phase 0: one token for V\n");
    const int V = [&] {
        std::vector<float> lg;
        fwd.reset();
        if (!fwd.step(1, 0, lg)) {
            std::printf("SKIP: step() failed (%s)\n", fwd.last_error());
            std::exit(77);
        }
        return (int)lg.size();
    }();
    std::printf("container %s, vocab %d\n", model.c_str(), V);

    // A short batch of plausible tokens.  Values do not matter: the check is batched-vs-sequential.
    // Two rows is enough to check the contract and keeps a host-GEMM run of a 27B container bounded.
    const std::vector<int> batch = {1, 2 % V};

    // Sequential reference: reset, then feed the same tokens one at a time.
    std::fprintf(stderr, "[test] phase 1: sequential steps\n");
    fwd.reset();
    std::vector<std::vector<float>> seq;
    for (size_t m = 0; m < batch.size(); m++) {
        std::vector<float> lg;
        if (!fwd.step(batch[m], (int)m, lg)) {
            std::printf("FAIL  sequential step %zu failed: %s\n", m, fwd.last_error());
            return 1;
        }
        seq.push_back(lg);
    }

    // [1] the batch, all rows
    std::fprintf(stderr, "[test] phase 2: step_batch_all\n");
    fwd.reset();
    std::vector<float> all;
    if (!fwd.step_batch_all(batch, 0, all)) {
        std::printf("FAIL  step_batch_all failed: %s\n", fwd.last_error());
        return 1;
    }
    if ((int)all.size() != (int)batch.size() * V) {
        std::printf("FAIL  step_batch_all returned %zu floats, want %zu\n", all.size(), batch.size() * (size_t)V);
        return 1;
    }
    for (size_t m = 0; m < batch.size(); m++) {
        std::vector<float> row(all.begin() + m * (size_t)V, all.begin() + (m + 1) * (size_t)V);
        double maxabs = 0;
        const double rel = rel_diff(row, seq[m], &maxabs);
        char msg[200];
        std::snprintf(msg, sizeof msg,
                      "row %zu: batched vs sequential rel-L1 %.2e (max|d| %.2e), argmax %d vs %d", m, rel,
                      maxabs, argmax(row), argmax(seq[m]));
        // the argmax is the contract; the values must be the same computation, so the tolerance is tight
        check(argmax(row) == argmax(seq[m]) && rel < 5e-3, msg);
    }

    // [2] the last row is exactly what plain step_batch() returns
    std::fprintf(stderr, "[test] phase 3: step_batch\n");
    fwd.reset();
    std::vector<float> tail;
    if (!fwd.step_batch(batch, 0, tail)) {
        std::printf("FAIL  step_batch failed: %s\n", fwd.last_error());
        return 1;
    }
    {
        std::vector<float> lastrow(all.begin() + (batch.size() - 1) * (size_t)V, all.end());
        double maxabs = 0;
        const double rel = rel_diff(lastrow, tail, &maxabs);
        char msg[200];
        std::snprintf(msg, sizeof msg, "last row == step_batch(): rel-L1 %.2e, argmax %d vs %d", rel,
                      argmax(lastrow), argmax(tail));
        check(argmax(lastrow) == argmax(tail) && rel < 1e-6, msg);
    }

    // [3] repeatability
    std::fprintf(stderr, "[test] phase 4: repeat\n");
    fwd.reset();
    std::vector<float> again;
    if (fwd.step_batch_all(batch, 0, again)) {
        double maxabs = 0;
        const double rel = rel_diff(all, again, &maxabs);
        char msg[200];
        std::snprintf(msg, sizeof msg, "repeat run identical: rel-L1 %.2e", rel);
        check(rel == 0.0, msg);
    } else {
        check(false, "repeat step_batch_all failed");
    }

    std::printf(fails ? "\n%d CHECK(S) FAILED\n" : "\nall checks passed\n", fails);
    return fails ? 1 : 0;
}
