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

// npu_batch_decode_test -- engine#2689 item 3b: several in-flight requests share one decode pass, and each still gets
// exactly the logits its own single-sequence run produces.
//
// Host-GEMM, no device; skips (77) unless ONEBIT_TEST_Q4NX names a container:
//
//   ONEBIT_TEST_Q4NX=~/models/npu/Qwen3-0.6B ./npu_batch_decode_test
//
// Three requests run as three THREADS against one NpuDecodeBatch (that is what a serve does: one worker per request),
// each with its own slot, and their greedy token sequences and every row's logits are compared bit for bit against three
// separate single-sequence forwards. The rounds counter has to come out well below the number of decode steps, or the
// requests ran one after another and the file would pass while testing nothing.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <chrono>
#include <thread>
#include <vector>

#include "npu_batch_decode.h"
#include "q4nx_forward.h"

namespace {

constexpr int kRounds = 4;   // decode steps per request

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

bool init(Q4nxNpuForward& fwd, const std::string& dir) {
    setenv("NPU_INFER_HOST_GEMM", "1", 1);
    return fwd.init_host((dir + "/model.q4nx").c_str(), dir + "/kernels");
}

}  // namespace

int main() {
    const char* dir = std::getenv("ONEBIT_TEST_Q4NX");
    if (dir == nullptr || dir[0] == '\0') {
        std::printf("SKIP: set ONEBIT_TEST_Q4NX to a container directory (model.q4nx + kernels/)\n");
        return 77;
    }
    const std::string d(dir);
    const std::vector<std::vector<int>> prompts = {{1, 2, 3}, {7, 8, 9, 10}, {13, 14, 15}};
    const int users = int(prompts.size());

    // The reference: one forward per request, run as a plain single sequence.
    std::vector<std::unique_ptr<Q4nxNpuForward>> ref;
    std::vector<int> ref_next(users, 0);
    size_t width = 0;
    for (int u = 0; u < users; u++) {
        ref.emplace_back(new Q4nxNpuForward());
        if (!init(*ref[(size_t)u], d)) {
            std::printf("SKIP: init_host failed for %s (%s)\n", d.c_str(), ref[(size_t)u]->last_error());
            return 77;
        }
        std::vector<float> logits;
        if (!ref[(size_t)u]->step_batch(prompts[(size_t)u], 0, logits) || logits.empty()) {
            std::printf("FAIL reference prefill %d: %s\n", u, ref[(size_t)u]->last_error());
            return 1;
        }
        width = logits.size();
        ref_next[(size_t)u] = argmax(logits);
    }

    Q4nxNpuForward fwd;
    if (!init(fwd, d)) {
        std::printf("SKIP: init_host failed for the batched instance (%s)\n", fwd.last_error());
        return 77;
    }
    {
        onebit::NpuDecodeBatch batch(fwd, 4);
        // The refusals first: they must be by name, and they must not need a device or a round to observe.
        {
            std::string why;
            std::vector<float> lg;
            if (batch.prefill(9, {1, 2}, &why)) {
                std::printf("FAIL a prefill for an unheld slot was accepted\n");
                return 1;
            }
            if (why.find("not held") == std::string::npos) {
                std::printf("FAIL the unheld-slot refusal does not say so: %s\n", why.c_str());
                return 1;
            }
            if (batch.step(-1, 1, 0, &lg, &why)) {
                std::printf("FAIL a step without a slot was accepted\n");
                return 1;
            }
        }
        std::vector<int> slots(users, -1);
        for (int u = 0; u < users; u++) {
            slots[(size_t)u] = batch.acquire();
            if (slots[(size_t)u] < 0) {
                std::printf("FAIL acquire %d returned no slot (max %d)\n", u, batch.max_users());
                return 1;
            }
        }
        if (batch.acquire() >= 0) {   // 4th slot exists; take it so the next call is the refusal
            if (batch.acquire() >= 0) {
                std::printf("FAIL acquire past max_users was accepted\n");
                return 1;
            }
        }

        std::vector<std::vector<std::vector<float>>> got(users);   // [user][round] logits
        std::vector<std::string> errors(users);
        std::vector<std::thread> threads;
        for (int u = 0; u < users; u++) {
            threads.emplace_back([&, u] {
                std::string why;
                if (!batch.prefill(slots[(size_t)u], prompts[(size_t)u], &why)) {
                    errors[(size_t)u] = "prefill: " + why;
                    return;
                }
                int token = ref_next[(size_t)u];
                for (int r = 0; r < kRounds; r++) {
                    std::vector<float> row;
                    const int pos = int(prompts[(size_t)u].size()) + r;
                    if (!batch.step(slots[(size_t)u], token, pos, &row, &why)) {
                        errors[(size_t)u] = "round " + std::to_string(r) + ": " + why;
                        return;
                    }
                    got[(size_t)u].push_back(row);
                    token = argmax(row);
                }
            });
        }
        for (auto& t : threads) t.join();
        for (int u = 0; u < users; u++) {
            if (!errors[(size_t)u].empty()) {
                std::printf("FAIL request %d: %s\n", u, errors[(size_t)u].c_str());
                return 1;
            }
        }
        // Every row against that request's own single-sequence step.
        for (int u = 0; u < users; u++) {
            int token = ref_next[(size_t)u];
            for (int r = 0; r < kRounds; r++) {
                std::vector<float> want;
                if (!ref[(size_t)u]->step(token, int(prompts[(size_t)u].size()) + r, want)) {
                    std::printf("FAIL reference step %d/%d: %s\n", r, u, ref[(size_t)u]->last_error());
                    return 1;
                }
                const auto& mine = got[(size_t)u][(size_t)r];
                const int at = first_difference(mine, want);
                if (at >= 0) {
                    std::printf("FAIL request %d round %d differs at logit %d (of %zu): %.9g vs %.9g\n", u, r, at,
                                want.size(), mine[(size_t)at], want[(size_t)at]);
                    return 1;
                }
                token = argmax(want);
            }
        }
        // The batching itself: 12 blocking steps across 3 threads must not have cost 12 passes.
        const long steps_total = long(users) * kRounds;
        std::printf("ok: %d requests x %d steps = %ld steps in %ld pass(es)\n", users, kRounds, steps_total,
                    batch.rounds());
        if (batch.rounds() >= steps_total) {
            std::printf("FAIL every step got its own pass: the requests were serialised, so this file tested nothing\n");
            return 1;
        }
        for (int u = 0; u < users; u++) batch.release(slots[(size_t)u]);
    }

    // Steady state of a real serve: a NEW request's prompt arrives while other slots are mid-decode, so its prefill runs
    // between their rounds. window_us = 0 makes that interleaving frequent on purpose, and every row is still compared
    // against that request's own single-sequence run -- a prefill sharing the pass with live decode rows is where a
    // slot mix-up would show as garbage rather than as a subtly wrong number.
    {
        Q4nxNpuForward srv;
        if (!init(srv, d)) return 77;
        onebit::NpuDecodeBatch batch(srv, 4, 0);
        std::vector<std::vector<std::vector<float>>> got2(users);
        std::vector<std::string> err2(users);
        std::vector<std::thread> th;
        for (int u = 0; u < users; u++) {
            th.emplace_back([&, u] {
                std::this_thread::sleep_for(std::chrono::milliseconds(120 * u));   // slot u joins mid-stream
                std::string why;
                const int slot = batch.acquire();
                if (slot < 0) { err2[(size_t)u] = "no slot"; return; }
                if (!batch.prefill(slot, prompts[(size_t)u], &why)) { err2[(size_t)u] = "prefill: " + why; return; }
                int token = ref_next[(size_t)u];
                for (int r = 0; r < kRounds; r++) {
                    std::vector<float> row;
                    const int pos = int(prompts[(size_t)u].size()) + r;
                    if (!batch.step(slot, token, pos, &row, &why)) { err2[(size_t)u] = "step: " + why; return; }
                    got2[(size_t)u].push_back(row);
                    token = argmax(row);
                }
                batch.release(slot);
            });
        }
        for (auto& t : th) t.join();
        for (int u = 0; u < users; u++) {
            if (!err2[(size_t)u].empty()) {
                std::printf("FAIL staggered request %d: %s\n", u, err2[(size_t)u].c_str());
                return 1;
            }
        }
        for (int u = 0; u < users; u++) {
            int token = ref_next[(size_t)u];
            for (int r = 0; r < kRounds; r++) {
                std::vector<float> want;
                if (!ref[(size_t)u]->step(token, int(prompts[(size_t)u].size()) + r, want)) {
                    std::printf("FAIL staggered reference %d/%d\n", r, u);
                    return 1;
                }
                const int at = first_difference(got2[(size_t)u][(size_t)r], want);
                if (at >= 0) {
                    std::printf("FAIL staggered request %d round %d differs at logit %d: %.9g vs %.9g\n", u, r, at,
                                got2[(size_t)u][(size_t)r][(size_t)at], want[(size_t)at]);
                    return 1;
                }
                token = argmax(want);
            }
        }
        std::printf("ok: %d staggered requests (prompt arriving mid-decode) stay bit-identical in %ld pass(es)\n", users,
                    batch.rounds());
    }

    // A full batch QUEUES a new request rather than failing it: two slots, two holders, a third caller that must wait
    // until one releases. This is the serve case -- a busy batch is not a bad request.
    {
        Q4nxNpuForward srv;
        if (!init(srv, d)) return 77;
        onebit::NpuDecodeBatch batch(srv, 2);
        const int a = batch.acquire();
        const int b2 = batch.acquire();
        if (a < 0 || b2 < 0) {
            std::printf("FAIL two slots on a fresh batch\n");
            return 1;
        }
        if (batch.acquire(0) >= 0) {   // wait_ms = 0: still nothing free right now
            std::printf("FAIL acquire(0) on a full batch returned a slot\n");
            return 1;
        }
        std::chrono::steady_clock::time_point waited_until;
        bool got = false;
        std::thread waiter([&] {
            const int s2 = batch.acquire(10000);   // queues until a release
            waited_until = std::chrono::steady_clock::now();
            got = s2 >= 0;
            if (got) batch.release(s2);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (got) {
            std::printf("FAIL the queued acquire returned while the batch was still full\n");
            return 1;
        }
        const auto t_release = std::chrono::steady_clock::now();
        batch.release(a);
        waiter.join();
        if (!got) {
            std::printf("FAIL the queued acquire never got the freed slot\n");
            return 1;
        }
        const double after_ms =
            1000.0 * std::chrono::duration<double>(waited_until - t_release).count();
        if (after_ms > 500) {
            std::printf("FAIL the queued acquire took %.0f ms to notice a release\n", after_ms);
            return 1;
        }
        batch.release(b2);
    }

    // A second row for a slot whose first is still in flight is refused (it would interleave one sequence's positions).
    {
        Q4nxNpuForward solo;
        if (!init(solo, d)) return 77;
        onebit::NpuDecodeBatch batch(solo, 2);
        const int s = batch.acquire();
        if (s < 0) {
            std::printf("FAIL acquire on a fresh batch\n");
            return 1;
        }
        std::string why;
        std::vector<float> row;
        // Two threads race the same slot: exactly one gets through, the other is refused by name.
        int accepted = 0, refused = 0;
        std::string refused_why;
        std::vector<std::thread> t;
        for (int k = 0; k < 2; k++)
            t.emplace_back([&] {
                std::string w;
                std::vector<float> lg;
                if (batch.step(s, 3, 0, &lg, &w)) {
                    __atomic_fetch_add(&accepted, 1, __ATOMIC_SEQ_CST);
                } else {
                    __atomic_fetch_add(&refused, 1, __ATOMIC_SEQ_CST);
                    refused_why = w;
                }
            });
        for (auto& x : t) x.join();
        if (accepted != 1 || refused != 1) {
            std::printf("FAIL two in-flight rows for one slot: %d accepted, %d refused\n", accepted, refused);
            return 1;
        }
        if (refused_why.find("already has a row waiting") == std::string::npos) {
            std::printf("FAIL the duplicate-slot refusal does not say why: %s\n", refused_why.c_str());
            return 1;
        }
        batch.release(s);
    }
    return 0;
}
