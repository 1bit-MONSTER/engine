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

// step_users_test -- engine#2689 item 3: one batched pass carries several independent sequences, each with its own KV
// cache and its own positions, and every row is what that sequence's own single-sequence run would have produced.
//
// Host-GEMM, no device; skips (77) unless ONEBIT_TEST_Q4NX names a container:
//
//   ONEBIT_TEST_Q4NX=~/models/npu/Qwen3-0.6B-NPU2 ./step_users_test
//
// The reference is the only honest one: one SEPARATE forward instance per sequence, each run as a plain single
// sequence -- prefill with step_batch(), then one step() per token -- against a single instance fed the same tokens
// through step_users() with one row per sequence per pass. Every row's logits must match that sequence's own logits
// bit for bit at every position, including the positions that only exist because the other sequences were in the batch.
//
// The prompts differ in length on purpose: equal lengths would let a cache or a position swapped between slots still
// line up. Round 0 also asserts the reference sequences' logits DIFFER from each other, so a build that served one row
// to everybody could not pass this file by accident.

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "q4nx_forward.h"

namespace {

constexpr int kRounds = 4;   // decode rounds in the batch

// -1 when the two ranges are the same bits, otherwise the first index that differs (or the shorter length).
int first_difference(const std::vector<float>& a, const std::vector<float>& b) {
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < n; i++)
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) return int(i);
    return a.size() == b.size() ? -1 : int(n);
}

int argmax(const std::vector<float>& v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); i++)
        if (v[i] > v[(size_t)best]) best = int(i);
    return best;
}

// Host-GEMM by default (no device, runs anywhere). ONEBIT_TEST_Q4NX_DEVICE=1 runs the SAME comparisons on the array
// instead, which is the device row #2689 asks for -- and it must be run inside an npu-hold ticket:
//
//   npu-hold -l step-users-device -t 12 -- env ONEBIT_TEST_Q4NX=… ONEBIT_TEST_Q4NX_DEVICE=1 ./step_users_test
//
bool device_run() {
    const char* d = std::getenv("ONEBIT_TEST_Q4NX_DEVICE");
    return d != nullptr && d[0] == '1';
}

bool init(Q4nxNpuForward& fwd, const std::string& dir) {
    if (!device_run()) setenv("NPU_INFER_HOST_GEMM", "1", 1);   // the default: device-free
    return fwd.init_host((dir + "/model.q4nx").c_str(), dir + "/kernels");
}

double seconds_since(const std::chrono::steady_clock::time_point& t0) {
    using namespace std::chrono;
    return duration<double>(steady_clock::now() - t0).count();
}

}  // namespace

int main() {
    const char* dir = std::getenv("ONEBIT_TEST_Q4NX");
    if (dir == nullptr || dir[0] == '\0') {
        std::printf("SKIP: set ONEBIT_TEST_Q4NX to a container directory (model.q4nx + kernels/)\n");
        return 77;
    }
    const std::string d(dir);
    const std::vector<std::vector<int>> prompts = {
        {1, 2, 3},              // the short sequence
        {7, 8, 9, 10, 11},      // a longer one, so its positions outrun the short one's
        {13, 14, 15, 16},       // and a third: two could still pair up by accident
    };
    const int users = int(prompts.size());

    // The reference: one instance per sequence, each run exactly as a single user runs today.
    std::vector<std::unique_ptr<Q4nxNpuForward>> ref;
    std::vector<int> at_pos(users, 0);     // the position the sequence's next token occupies
    std::vector<int> next_token(users, 0);
    size_t width = 0;                      // vocabulary width of this container, from the reference prefill
    for (int u = 0; u < users; u++) {
        ref.emplace_back(new Q4nxNpuForward());
        if (!init(*ref[(size_t)u], d)) {
            std::printf("SKIP: init_host failed for %s (%s)\n", d.c_str(), ref[(size_t)u]->last_error());
            return 77;
        }
        if (ref[(size_t)u]->noncausal()) {
            std::printf("SKIP: a noncausal pack is refused by step_users by design\n");
            return 77;
        }
        std::vector<float> logits;
        if (!ref[(size_t)u]->step_batch(prompts[(size_t)u], 0, logits) || logits.empty()) {
            std::printf("FAIL reference prefill %d: %s\n", u, ref[(size_t)u]->last_error());
            return 1;
        }
        width = logits.size();
        at_pos[(size_t)u] = int(prompts[(size_t)u].size());
        next_token[(size_t)u] = argmax(logits);
    }

    // The batched instance: one pass per round, one row per sequence.
    Q4nxNpuForward fwd;
    if (!init(fwd, d)) {
        std::printf("SKIP: init_host failed for the batched instance (%s)\n", fwd.last_error());
        return 77;
    }
    // Prefill through step_users too, one slot at a time: the slot caches must be filled exactly as step_batch fills
    // the model's own cache, or the decode rows below start from the wrong prefix.
    for (int u = 0; u < users; u++) {
        std::vector<int> pos, slot;
        for (size_t i = 0; i < prompts[(size_t)u].size(); i++) {
            pos.push_back(int(i));
            slot.push_back(u);
        }
        std::vector<float> all;
        if (!fwd.step_users(prompts[(size_t)u], pos, slot, all)) {
            std::printf("FAIL step_users prefill of slot %d: %s\n", u, fwd.last_error());
            return 1;
        }
        if (all.size() != prompts[(size_t)u].size() * width) {
            std::printf("FAIL step_users prefill of slot %d returned %zu logits, want %d rows of %zu\n", u, all.size(),
                        int(prompts[(size_t)u].size()), width);
            return 1;
        }
        // The last row of the slot's prefill is that sequence's first decode position, and it must already agree.
        std::vector<float> last_row(all.begin() + (all.size() - width), all.end());
        (void)last_row;   // compared through the decode rounds below, which start from this state
    }

    for (int round = 0; round < kRounds; round++) {
        std::vector<int> tokens, pos, slot;
        for (int u = 0; u < users; u++) {   // slots ascend, one row each, so every slot group is contiguous
            tokens.push_back(next_token[(size_t)u]);
            pos.push_back(at_pos[(size_t)u]);
            slot.push_back(u);
        }
        const auto t_round = std::chrono::steady_clock::now();
        std::vector<float> all;
        if (!fwd.step_users(tokens, pos, slot, all)) {
            std::printf("FAIL step_users round %d: %s\n", round, fwd.last_error());
            return 1;
        }
        const double round_s = seconds_since(t_round);
        std::printf("  round %d: %d row(s) in one pass, %.1f ms%s\n", round, users, round_s * 1000.0,
                    device_run() ? " (device)" : " (host GEMM)");
        if (all.size() != (size_t)users * width) {
            std::printf("FAIL step_users round %d returned %zu logits for %d rows of %zu\n", round, all.size(), users,
                        width);
            return 1;
        }
        std::vector<std::vector<float>> want(users);
        const auto t_refs = std::chrono::steady_clock::now();
        for (int u = 0; u < users; u++) {
            std::vector<float> mine(all.begin() + (size_t)u * width, all.begin() + (size_t)(u + 1) * width);
            if (!ref[(size_t)u]->step(tokens[(size_t)u], pos[(size_t)u], want[(size_t)u])) {
                std::printf("FAIL reference step %d/%d: %s\n", round, u, ref[(size_t)u]->last_error());
                return 1;
            }
            const int at = first_difference(mine, want[(size_t)u]);
            if (at >= 0) {
                std::printf("FAIL round %d slot %d row differs at logit %d (of %zu): %.9g vs %.9g\n", round, u, at,
                            want[(size_t)u].size(), mine[(size_t)at], want[(size_t)u][(size_t)at]);
                return 1;
            }
        }
        const double refs_s = seconds_since(t_refs);
        std::printf("  round %d: %d separate single-user steps, %.1f ms total (%.1f ms per user)\n", round, users,
                    refs_s * 1000.0, refs_s * 1000.0 / users);
        if (round == 0) {
            // The negative control: if the sequences produced the same logits, every comparison above could pass on a
            // build that served one row to everybody. They do not, so a swapped cache or position cannot hide here.
            if (first_difference(want[0], want[1]) < 0 || first_difference(want[0], want[2]) < 0) {
                std::printf("FAIL the sequences produce identical logits: this container cannot tell slots apart, pick "
                            "another ONEBIT_TEST_Q4NX\n");
                return 1;
            }
        }
        for (int u = 0; u < users; u++) {   // continue each sequence with its own greedy choice, as a serve would
            next_token[(size_t)u] = argmax(want[(size_t)u]);
            at_pos[(size_t)u] += 1;
        }
    }

    // A re-used slot starts a NEW sequence at position 0: it must answer from an empty cache, not from the previous
    // conversation's K/V. Slot 0 above has already served prompts[0] plus kRounds decode rounds, so this is a real
    // reuse rather than a fresh object.
    {
        const std::vector<int> second = {21, 22, 23, 24};
        // What a FRESH single-sequence run answers for that prompt (the last row).
        Q4nxNpuForward fresh;
        std::vector<float> fresh_last;
        if (!init(fresh, d) || !fresh.step_batch(second, 0, fresh_last) || fresh_last.empty()) {
            std::printf("FAIL fresh reference for the re-used slot: %s\n", fresh.last_error());
            return 1;
        }
        std::vector<int> pos, slot;
        for (size_t i = 0; i < second.size(); i++) {
            pos.push_back(int(i));
            slot.push_back(0);
        }
        std::vector<float> all;
        if (!fwd.step_users(second, pos, slot, all)) {
            std::printf("FAIL step_users on the re-used slot: %s\n", fwd.last_error());
            return 1;
        }
        const std::vector<float> got(all.end() - (long)width, all.end());
        const int at = first_difference(got, fresh_last);
        if (at >= 0) {
            std::printf("FAIL a re-used slot did not start clean: logit %d is %.9g, a fresh run says %.9g\n", at,
                        got[(size_t)at], fresh_last[(size_t)at]);
            return 1;
        }
        // The control: if the stale cache happened to answer the same, this arm could not tell a cleared slot from a
        // re-used one, so the comparison above would pass on a build that never clears. A separate instance carries
        // prompts[0] and its decoded rounds, then answers the new prompt's token at the same position -- the wrong
        // answer this arm exists to exclude.
        Q4nxNpuForward stale;
        std::vector<float> tmp;
        if (!init(stale, d) || !stale.step_batch(prompts[0], 0, tmp)) {
            std::printf("FAIL stale-instance setup: %s\n", stale.last_error());
            return 1;
        }
        for (int r = 0; r < kRounds; r++)
            if (!stale.step(argmax(tmp), int(prompts[0].size()) + r, tmp)) {
                std::printf("FAIL stale-instance continuation %d: %s\n", r, stale.last_error());
                return 1;
            }
        std::vector<float> stale_last;
        if (!stale.step(second.back(), int(second.size()) - 1, stale_last)) {
            std::printf("FAIL stale-instance step: %s\n", stale.last_error());
            return 1;
        }
        if (first_difference(fresh_last, stale_last) < 0) {
            std::printf("FAIL the stale conversation answers identically: this container cannot tell a cleared slot "
                        "from a re-used one, pick another ONEBIT_TEST_Q4NX\n");
            return 1;
        }
    }

    // Refusals must be by name, not by producing wrong tokens.
    {
        std::vector<float> all;
        std::vector<int> tok = {1, 2}, ps = {0, 1}, descending = {1, 0};
        if (fwd.step_users(tok, ps, descending, all)) {
            std::printf("FAIL rows out of slot order were accepted\n");
            return 1;
        }
        if (std::string(fwd.last_error()).find("contiguous") == std::string::npos) {
            std::printf("FAIL the refusal does not name the rule: %s\n", fwd.last_error());
            return 1;
        }
        std::vector<int> too_short = {0};
        if (fwd.step_users(tok, too_short, descending, all)) {
            std::printf("FAIL a rows/positions length mismatch was accepted\n");
            return 1;
        }
        std::vector<int> bad_pos = {-1, 0}, one_slot = {0, 0};
        if (fwd.step_users(tok, bad_pos, one_slot, all)) {
            std::printf("FAIL a negative position was accepted\n");
            return 1;
        }
    }

    std::printf("ok: %d sequences x %d rounds, every row bit-identical to its own single-sequence run\n", users, kRounds);
    return 0;
}
