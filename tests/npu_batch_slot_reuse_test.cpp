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

// npu_batch_slot_reuse_test -- engine#2978: two consecutive requests on ONE slot of the decode batcher.
//
// This is what a serve with --npu-batch-users does all day: a request takes the lowest free slot, runs its prompt and
// its decode steps, releases the slot, and the next request gets the same slot and starts at position 0. Between
// engine#2734 and the fix for engine#2978 the forward refused that second prompt ("slot 0 holds N position(s), but this
// pass starts it at 0"), so every request after the first on a slot failed, and no test drove the batcher that way:
// npu_batch_decode_test acquires fresh slots for every prompt it runs.
//
// The calls are made one after another from one thread, so nothing here depends on how long a row takes: it runs the
// same on a real container and on the two-layer one CI fabricates (tests/batch_check_state_ci.py).
//
//   ONEBIT_TEST_Q4NX=<container directory> ./npu_batch_slot_reuse_test       (host GEMM, no device; 77 without it)
//
// What is compared, every time against a SEPARATE forward run as a plain single sequence:
//   - the first request on the slot (so the reference itself is known to agree before the slot is re-used);
//   - the second request's prompt row and its first decode row: bit-identical to a fresh run of that prompt;
//   - a neighbour that holds the other slot across the hand-over: its next row is bit-identical to its own run, so the
//     restart cleared one slot and not the bank;
//   - the control: the first conversation, asked the second prompt's last token at the same position, answers
//     differently, so a slot that was never cleared could not have passed the comparison above.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "npu_batch_decode.h"
#include "q4nx_forward.h"

namespace {

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

bool init(Q4nxNpuForward& fwd, const std::string& dir) {
    setenv("NPU_INFER_HOST_GEMM", "1", 1);   // device-free
    return fwd.init_host((dir + "/model.q4nx").c_str(), dir + "/kernels");
}

// One row against its reference; prints the FAIL line itself.
bool same(const char* what, const std::vector<float>& got, const std::vector<float>& want) {
    if (want.empty()) {
        std::printf("FAIL %s: the reference returned no logits\n", what);
        return false;
    }
    const int at = first_difference(got, want);
    if (at < 0) return true;
    if ((size_t)at < got.size() && (size_t)at < want.size())
        std::printf("FAIL %s: logit %d is %.9g, its own single-sequence run says %.9g\n", what, at, got[(size_t)at],
                    want[(size_t)at]);
    else
        std::printf("FAIL %s: %zu logits, its own single-sequence run has %zu\n", what, got.size(), want.size());
    return false;
}

}  // namespace

int main() {
    const char* dir = std::getenv("ONEBIT_TEST_Q4NX");
    if (dir == nullptr || dir[0] == '\0') {
        std::printf("SKIP: set ONEBIT_TEST_Q4NX to a container directory (model.q4nx + kernels/)\n");
        return 77;
    }
    const std::string d(dir);
    const std::vector<int> first = {1, 2, 3};             // the request that uses the slot first
    const std::vector<int> second = {21, 22, 23, 24};     // the request that gets the same slot afterwards
    const std::vector<int> other = {7, 8, 9, 10, 11};     // the neighbour, in flight across the hand-over

    Q4nxNpuForward fwd;
    if (!init(fwd, d)) {
        std::printf("SKIP: init_host failed for %s (%s)\n", d.c_str(), fwd.last_error());
        return 77;
    }
    if (!fwd.step_users_supported()) {
        std::printf("SKIP: this container cannot share a decode pass, so the batcher is refused for it by design\n");
        return 77;
    }
    // The references: one forward per sequence, each run exactly as a single user runs.
    Q4nxNpuForward ref_first, ref_second, ref_other;
    std::vector<float> want_first, want_second, want_other;
    if (!init(ref_first, d) || !ref_first.step_batch(first, 0, want_first) || !init(ref_second, d) ||
        !ref_second.step_batch(second, 0, want_second) || !init(ref_other, d) ||
        !ref_other.step_batch(other, 0, want_other)) {
        std::printf("FAIL reference prefill: %s %s %s\n", ref_first.last_error(), ref_second.last_error(),
                    ref_other.last_error());
        return 1;
    }

    onebit::NpuDecodeBatch batch(fwd, 2, 0);
    std::string why;
    std::vector<float> got;

    // Request 1 takes a slot; the neighbour takes the other one and keeps it to the end.
    const int slot_a = batch.acquire();
    const int slot_n = batch.acquire();
    if (slot_a < 0 || slot_n < 0 || slot_a == slot_n) {
        std::printf("FAIL two slots on a fresh batch (%d, %d)\n", slot_a, slot_n);
        return 1;
    }
    if (!batch.prefill(slot_a, first, &got, &why)) {
        std::printf("FAIL the first request's prompt: %s\n", why.c_str());
        return 1;
    }
    if (!same("the first request's prompt row", got, want_first)) return 1;
    if (!batch.prefill(slot_n, other, &got, &why)) {
        std::printf("FAIL the neighbour's prompt: %s\n", why.c_str());
        return 1;
    }
    if (!same("the neighbour's prompt row", got, want_other)) return 1;
    int tok_first = argmax(want_first);
    for (int r = 0; r < 2; r++) {   // request 1 decodes two tokens, as a short answer would
        const int at = int(first.size()) + r;
        if (!batch.step(slot_a, tok_first, at, &got, &why)) {
            std::printf("FAIL the first request's decode step %d: %s\n", r, why.c_str());
            return 1;
        }
        if (!ref_first.step(tok_first, at, want_first)) {
            std::printf("FAIL reference step %d of the first request: %s\n", r, ref_first.last_error());
            return 1;
        }
        if (!same("a decode row of the first request", got, want_first)) return 1;
        tok_first = argmax(want_first);
    }
    int tok_other = argmax(want_other);
    int at_other = int(other.size());
    if (!batch.step(slot_n, tok_other, at_other, &got, &why) || !ref_other.step(tok_other, at_other, want_other)) {
        std::printf("FAIL the neighbour's first decode step: %s %s\n", why.c_str(), ref_other.last_error());
        return 1;
    }
    if (!same("the neighbour's first decode row", got, want_other)) return 1;
    tok_other = argmax(want_other);
    at_other++;

    // Request 1 is done. The next request must get the SAME slot, or this run re-used nothing.
    if (!batch.release(slot_a)) {
        std::printf("FAIL the first request's slot was not released\n");
        return 1;
    }
    const int slot_b = batch.acquire();
    if (slot_b != slot_a) {
        std::printf("FAIL the freed slot %d was not handed out again (got %d), so this run did not re-use a slot\n",
                    slot_a, slot_b);
        return 1;
    }
    // The line engine#2978 is about: a prompt at position 0 on a slot that still holds the last request's positions.
    if (!batch.prefill(slot_b, second, &got, &why)) {
        std::printf("FAIL the second request on slot %d was refused: %s\n", slot_b, why.c_str());
        return 1;
    }
    if (!same("the second request's prompt row on the re-used slot", got, want_second)) return 1;
    // The control. ref_first still holds the first conversation; asked the new prompt's last token at the same
    // position it gives the answer a slot that was NOT cleared would give. If that equals the fresh answer, the
    // comparison above proves nothing on this container.
    std::vector<float> stale;
    if (!ref_first.step(second.back(), int(second.size()) - 1, stale)) {
        std::printf("FAIL the stale-conversation control: %s\n", ref_first.last_error());
        return 1;
    }
    if (first_difference(stale, want_second) < 0) {
        std::printf("FAIL the previous conversation answers identically: this container cannot tell a cleared slot "
                    "from a re-used one, pick another ONEBIT_TEST_Q4NX\n");
        return 1;
    }
    const int tok_second = argmax(want_second);
    if (!batch.step(slot_b, tok_second, int(second.size()), &got, &why) ||
        !ref_second.step(tok_second, int(second.size()), want_second)) {
        std::printf("FAIL the second request's decode step: %s %s\n", why.c_str(), ref_second.last_error());
        return 1;
    }
    if (!same("the second request's decode row on the re-used slot", got, want_second)) return 1;
    // The neighbour held its slot through all of that and continues its own sequence.
    if (!batch.step(slot_n, tok_other, at_other, &got, &why) || !ref_other.step(tok_other, at_other, want_other)) {
        std::printf("FAIL the neighbour's decode step after the other slot restarted: %s %s\n", why.c_str(),
                    ref_other.last_error());
        return 1;
    }
    if (!same("the neighbour's decode row after the other slot restarted", got, want_other)) return 1;
    batch.release(slot_b);
    batch.release(slot_n);
    std::printf("ok: two consecutive requests on slot %d, the second bit-identical to a fresh run, the neighbour on "
                "slot %d undisturbed (%ld passes)\n", slot_a, slot_n, batch.rounds());
    return 0;
}
