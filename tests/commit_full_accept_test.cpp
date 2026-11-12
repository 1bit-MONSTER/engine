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

// commit_full_accept_test.cpp -- a fully accepted verify batch keeps the state its own recurrence left
// (Q4nxNpuForward::commit_verify_rows, engine#1781), and that state is the one the snapshot restore + replay builds.
//
// Host-GEMM, no device; skips (77) unless ONEBIT_TEST_Q4NX names a causal model with linear-attention (GDN) layers.
// The two commit paths are chosen by NPU_COMMIT_FULL_REPLAY, which the forward reads once per process, so the test
// runs itself twice, once on each path, and compares what the two runs wrote: the whole state after the commit
// (attention KV, conv window and recurrent state of every layer) and the next step's logits, bit for bit.
//
//   ONEBIT_TEST_Q4NX=~/models/npu/MiniCPM-V-4.7-1B-NPU2 ./commit_full_accept_test

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "q4nx_forward.h"

namespace {

// One run on whichever path the environment selects: prefill, one fully accepted verify batch, one step after it.
int run_once(const char* dir, const char* out) {
    setenv("NPU_INFER_HOST_GEMM", "1", 1);
    Q4nxNpuForward fwd;
    const std::string model = std::string(dir) + "/model.q4nx";
    if (!fwd.init_host(model.c_str(), std::string(dir) + "/kernels")) {
        std::printf("SKIP: init_host failed for %s (%s)\n", model.c_str(), fwd.last_error());
        return 77;
    }
    if (fwd.noncausal()) { std::printf("SKIP: a noncausal pack has no verify pass\n"); return 77; }
    const std::vector<int> prompt = {1, 2, 3, 4, 5};
    const std::vector<int> block = {6, 7, 8, 9};
    std::vector<float> logits, all;
    if (!fwd.step_batch(prompt, 0, logits)) { std::printf("FAIL prefill: %s\n", fwd.last_error()); return 1; }
    fwd.begin_verify(int(block.size()), int(prompt.size()));
    if (!fwd.step_batch_all(block, int(prompt.size()), all)) { std::printf("FAIL verify: %s\n", fwd.last_error()); return 1; }
    if (!fwd.commit_verify_rows(int(block.size()))) { std::printf("FAIL commit: %s\n", fwd.last_error()); return 1; }
    const int pos = int(prompt.size() + block.size());
    Q4nxNpuForward::StateSnapshot snap;
    if (!fwd.snapshot_state(pos, snap)) { std::printf("FAIL snapshot\n"); return 1; }
    if (snap.rec.empty() || snap.rec[0].empty()) {
        bool any = false;
        for (const auto& r : snap.rec) any = any || !r.empty();
        if (!any) { std::printf("SKIP: the model has no recurrent (GDN) state\n"); return 77; }
    }
    if (!fwd.step(10, pos, logits)) { std::printf("FAIL step after the commit: %s\n", fwd.last_error()); return 1; }
    // The negative control: each run must have taken the path it was started for, or the comparison below would
    // compare a path with itself and pass while testing nothing.
    const bool replay = std::getenv("NPU_COMMIT_FULL_REPLAY") != nullptr;
    if (fwd.full_accepts_kept() != (replay ? 0 : 1)) {
        std::printf("FAIL the %s run took the other commit path (%ld full accept(s) kept)\n", replay ? "replay" : "kept",
                    fwd.full_accepts_kept());
        return 1;
    }
    FILE* f = std::fopen(out, "wb");   // file-create-ok: the test's own scratch file beside its build, removed below
    if (!f) return 1;
    auto put = [&](const std::vector<std::vector<float>>& vv) {
        for (const auto& v : vv) {
            const uint64_t n = v.size();
            std::fwrite(&n, sizeof n, 1, f);
            std::fwrite(v.data(), sizeof(float), v.size(), f);
        }
    };
    put(snap.k); put(snap.v); put(snap.conv); put(snap.rec);
    std::fwrite(logits.data(), sizeof(float), logits.size(), f);
    std::fclose(f);
    return 0;
}

std::vector<char> slurp(const std::string& p) {
    std::vector<char> b;
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return b;
    char buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + n);
    std::fclose(f);
    return b;
}

int child(const char* self, const char* out, bool replay) {
    const pid_t pid = fork();
    if (pid == 0) {
        if (replay) setenv("NPU_COMMIT_FULL_REPLAY", "1", 1); else unsetenv("NPU_COMMIT_FULL_REPLAY");
        unsetenv("NPU_REPLAY_CHECK");
        execl(self, self, "--one", out, (char*)nullptr);
        _exit(127);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
}

}  // namespace

int main(int argc, char** argv) {
    const char* dir = std::getenv("ONEBIT_TEST_Q4NX");
    if (!dir || !*dir) { std::printf("SKIP: set ONEBIT_TEST_Q4NX to a causal model with GDN layers\n"); return 77; }
    if (argc == 3 && std::string(argv[1]) == "--one") return run_once(dir, argv[2]);
    const std::string base = std::string("commit_full_accept.") + std::to_string(getpid());
    const std::string a = base + ".kept", b = base + ".replayed";
    const int ra = child(argv[0], a.c_str(), false);
    const int rb = ra == 0 ? child(argv[0], b.c_str(), true) : ra;
    int rc = 0;
    if (ra == 77 || rb == 77) rc = 77;
    else if (ra != 0 || rb != 0) { std::printf("FAIL a run failed (rc %d, %d)\n", ra, rb); rc = 1; }
    else {
        const std::vector<char> x = slurp(a), y = slurp(b);
        if (x.empty() || x != y) {
            std::printf("FAIL the kept state and the replayed state differ (%zu and %zu bytes)\n", x.size(), y.size());
            rc = 1;
        } else {
            std::printf("PASS  commit_full_accept: after a fully accepted 4-row verify batch, the kept state and the "
                        "restored + replayed state are identical (%zu bytes: KV, conv, recurrent, next logits)\n", x.size());
        }
    }
    std::remove(a.c_str());
    std::remove(b.c_str());
    return rc;
}
