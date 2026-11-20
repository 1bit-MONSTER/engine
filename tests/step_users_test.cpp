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
    // A container whose per-sequence state is not kept per slot cannot share a pass, and then the one thing to check
    // is that step_users() says so by name instead of answering. A model let through by mistake does not crash: it
    // returns rows that are not that sequence's own (short-convolution layers did, before each slot had its own
    // windows), which is what the rest of this file fails on when the predicate and the pass disagree.
    if (!fwd.step_users_supported()) {
        std::vector<float> all;
        if (fwd.step_users({1, 2}, {0, 1}, {0, 0}, all) || !all.empty()) {
            std::printf("FAIL step_users ran a pass on a container step_users_supported() rules out\n");
            return 1;
        }
        if (std::string(fwd.last_error()).find("step_users: ") != 0) {
            std::printf("FAIL the refusal does not name step_users: %s\n", fwd.last_error());
            return 1;
        }
        std::printf("ok: this container cannot share a pass and step_users refuses it by name: %s\n", fwd.last_error());
        return 0;
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


    // A pass that touches only slot 0 must not disturb the other slots' caches (engine#2689 item 3b found this: the
    // bank used to be resized to the highest slot of the CURRENT call, so a single-user call between multi-user rounds
    // dropped slots 1..3 and their next row attended over an empty cache at a position above 0 -- NaN in the serve
    // batcher). Slots 0..2 all hold conversations at this point; one call for slot 0 alone, then each of the other two
    // must still continue its own sequence bit for bit.
    {
        std::vector<int> last(users, 0);
        for (int u = 0; u < users; u++) last[(size_t)u] = next_token[(size_t)u];
        // A one-slot call, on the LOWEST slot, in the middle of everything.
        std::vector<float> throwaway;
        if (!fwd.step_users({last[0]}, {at_pos[0]}, {0}, throwaway)) {
            std::printf("FAIL single-slot pass on slot 0: %s\n", fwd.last_error());
            return 1;
        }
        for (int u = 1; u < users; u++) {   // slot 0 was the one the single-slot call advanced; 1..n-1 must be untouched
            std::vector<float> want;
            if (!ref[(size_t)u]->step(last[(size_t)u], at_pos[(size_t)u], want)) {
                std::printf("FAIL reference after the single-slot pass %d: %s\n", u, ref[(size_t)u]->last_error());
                return 1;
            }
            std::vector<float> row;
            if (!fwd.step_users({last[(size_t)u]}, {at_pos[(size_t)u]}, {u}, row)) {
                std::printf("FAIL slot %d after the single-slot pass: %s\n", u, fwd.last_error());
                return 1;
            }
            const int at = first_difference(row, want);
            if (at >= 0) {
                std::printf("FAIL slot %d lost its cache when slot 0 ran alone: logit %d is %.9g, its own run says %.9g\n",
                            u, at, row[(size_t)at], want[(size_t)at]);
                return 1;
            }
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

    // This arm runs LAST on purpose: it starts a new sequence at position 0 on a slot that already holds one, so
    // everything after it would be reading a cache the earlier rounds did not build (found by running it before the
    // single-slot arm: that arm then asked for slot 0's old length and was correctly refused).
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

    // The arm above runs slot 0 ALONE, so on its own it cannot tell "slot 0 was cleared" from "every slot was cleared",
    // and it never puts a restart in the same pass as a sequence that continues (review of engine#2978's fix). One
    // pass, three slots, three different things:
    //   slot 0 continues the sequence it restarted above (position 4 of `second`),
    //   slot 1 restarts at position 0 while it still holds its first conversation,
    //   slot 2 continues the conversation it has held since the top of this file.
    // Every row has its own single-sequence reference, so a clear that reaches a neighbour, or a neighbour's cache
    // read by the restarted slot, shows as a differing row.
    {
        const std::vector<int> second = {21, 22, 23, 24};   // what slot 0 holds now
        const std::vector<int> third = {31, 32, 33};        // slot 1's new prompt
        const int tok0 = 25, tok2 = 17;                     // any ids: each reference is fed the same one
        const int pos0 = int(second.size());
        const int pos2 = at_pos[2] + 1;   // the single-slot arm advanced slots 1 and 2 (and their references) by one row
        Q4nxNpuForward cont0, fresh1;
        std::vector<float> tmp, want0, want1, want2;
        if (!init(cont0, d) || !cont0.step_batch(second, 0, tmp) || !cont0.step(tok0, pos0, want0)) {
            std::printf("FAIL reference for the restarted sequence's next row: %s\n", cont0.last_error());
            return 1;
        }
        if (!init(fresh1, d) || !fresh1.step_batch(third, 0, want1) || want1.empty()) {
            std::printf("FAIL fresh reference for the slot that restarts in a shared pass: %s\n", fresh1.last_error());
            return 1;
        }
        if (!ref[2]->step(tok2, pos2, want2)) {
            std::printf("FAIL reference for the neighbour that continues: %s\n", ref[2]->last_error());
            return 1;
        }
        const std::vector<int> tokens = {tok0, third[0], third[1], third[2], tok2};
        const std::vector<int> pos = {pos0, 0, 1, 2, pos2};
        const std::vector<int> slot = {0, 1, 1, 1, 2};
        std::vector<float> all;
        if (!fwd.step_users(tokens, pos, slot, all)) {
            std::printf("FAIL a pass where one slot restarts and its neighbours continue: %s\n", fwd.last_error());
            return 1;
        }
        if (all.size() != tokens.size() * width) {
            std::printf("FAIL the mixed pass returned %zu logits, want %zu rows of %zu\n", all.size(), tokens.size(),
                        width);
            return 1;
        }
        const struct {
            size_t row;
            const std::vector<float>* want;
            const char* what;
        } rows[] = {
            {0, &want0, "slot 0 (continuing below a slot that restarts)"},
            {3, &want1, "slot 1 (restarting between two slots that continue)"},
            {4, &want2, "slot 2 (continuing above a slot that restarts)"},
        };
        for (const auto& r : rows) {
            const std::vector<float> got(all.begin() + long(r.row * width), all.begin() + long((r.row + 1) * width));
            const int at = first_difference(got, *r.want);
            if (at >= 0) {
                std::printf("FAIL %s differs at logit %d: %.9g, its own single-sequence run says %.9g\n", r.what, at,
                            got[(size_t)at], (*r.want)[(size_t)at]);
                return 1;
            }
        }
        // The exemption is for position 0 only. A start INSIDE the sequence a slot holds is still refused (slot 0 holds
        // 5 positions now), and the refusal leaves the slot as it was: its next row is still its own.
        std::vector<float> row;
        if (fwd.step_users({26}, {2}, {0}, row)) {
            std::printf("FAIL a start inside the slot's sequence (position 2 of 5) was accepted\n");
            return 1;
        }
        if (std::string(fwd.last_error()).find("holds 5 position(s)") == std::string::npos) {
            std::printf("FAIL the refusal does not say what the slot holds: %s\n", fwd.last_error());
            return 1;
        }
        if (!cont0.step(26, pos0 + 1, want0) || !fwd.step_users({26}, {pos0 + 1}, {0}, row)) {
            std::printf("FAIL slot 0 after a refused pass: %s / %s\n", cont0.last_error(), fwd.last_error());
            return 1;
        }
        const int at = first_difference(row, want0);
        if (at >= 0) {
            std::printf("FAIL a refused pass changed the slot it refused: logit %d is %.9g, its own run says %.9g\n", at,
                        row[(size_t)at], want0[(size_t)at]);
            return 1;
        }

        // A PASS THAT FAILS AFTER IT STARTED leaves the slots it carried empty, and says so. The layers it reached hold
        // this pass's rows and the others do not, so continuing would read rows of a pass that never completed; the
        // slot's old length must not survive it. NPU_NO_BATCH makes the batched body refuse a step_users pass after the
        // entry checks, which is the cheapest failure there is that happens inside the pass. Slot 2 is in the pass;
        // slot 1 is not and must keep everything.
        {
            const int next2 = pos2 + 1;   // where slot 2's sequence ends now
            const int next1 = int(third.size());
            setenv("NPU_NO_BATCH", "1", 1);
            const bool ran = fwd.step_users({27}, {next2}, {2}, row);
            const std::string said = fwd.last_error();
            unsetenv("NPU_NO_BATCH");
            if (ran) {
                std::printf("FAIL a pass ran with NPU_NO_BATCH set, so this arm has no failing pass to look at\n");
                return 1;
            }
            if (said.find("emptied") == std::string::npos) {
                std::printf("FAIL a failed pass does not say what it did to its slots: %s\n", said.c_str());
                return 1;
            }
            if (fwd.step_users({27}, {next2}, {2}, row)) {
                std::printf("FAIL a slot continued from where it was before a pass that failed\n");
                return 1;
            }
            if (std::string(fwd.last_error()).find("holds 0 position(s)") == std::string::npos) {
                std::printf("FAIL the slot of a failed pass is not empty: %s\n", fwd.last_error());
                return 1;
            }
            // The slot starts again at 0 and answers as a fresh run does ...
            std::vector<float> all2;
            if (!fwd.step_users(third, {0, 1, 2}, {2, 2, 2}, all2) || all2.size() != third.size() * width) {
                std::printf("FAIL the slot of a failed pass did not start again at position 0: %s\n", fwd.last_error());
                return 1;
            }
            const std::vector<float> got2(all2.end() - (long)width, all2.end());
            int at2 = first_difference(got2, want1);   // want1 is a fresh run of `third`, from above
            if (at2 >= 0) {
                std::printf("FAIL the slot of a failed pass did not start clean: logit %d is %.9g, a fresh run says %.9g\n",
                            at2, got2[(size_t)at2], want1[(size_t)at2]);
                return 1;
            }
            // ... and the slot that was not in the failed pass continues its own sequence.
            std::vector<float> want_n;
            if (!fresh1.step(34, next1, want_n) || !fwd.step_users({34}, {next1}, {1}, row)) {
                std::printf("FAIL a slot outside the failed pass: %s / %s\n", fresh1.last_error(), fwd.last_error());
                return 1;
            }
            at2 = first_difference(row, want_n);
            if (at2 >= 0) {
                std::printf("FAIL a failed pass changed a slot it did not carry: logit %d is %.9g, its own run says %.9g\n",
                            at2, row[(size_t)at2], want_n[(size_t)at2]);
                return 1;
            }
        }
    }

    // The single-sequence path, for the same question: an instance that has served one conversation and is given a new
    // prompt at position 0 answers as a fresh instance does -- in one batch, and in two chunks, which is how a serve
    // sends a prompt longer than one pass. This is what every request of a serve without --npu-batch-users does, and on
    // a short-convolution model it is the convolution windows that have to be cleared with the K/V.
    {
        const std::vector<int> second = {21, 22, 23, 24};
        Q4nxNpuForward fresh;
        std::vector<float> want, got;
        if (!init(fresh, d) || !fresh.step_batch(second, 0, want) || want.empty()) {
            std::printf("FAIL fresh reference for the single-sequence restart: %s\n", fresh.last_error());
            return 1;
        }
        Q4nxNpuForward& used = *ref[0];   // has served prompts[0] and its decode rounds
        if (!used.step_batch(second, 0, got)) {
            std::printf("FAIL a second request on a used instance: %s\n", used.last_error());
            return 1;
        }
        int at = first_difference(got, want);
        if (at >= 0) {
            std::printf("FAIL a second request on a used single-sequence instance is not a fresh run: logit %d is %.9g, "
                        "fresh says %.9g\n", at, got[(size_t)at], want[(size_t)at]);
            return 1;
        }
        const std::vector<int> head(second.begin(), second.begin() + 2), tail(second.begin() + 2, second.end());
        if (!used.step_batch(head, 0, got) || !used.step_batch(tail, int(head.size()), got)) {
            std::printf("FAIL a second request sent in two chunks: %s\n", used.last_error());
            return 1;
        }
        at = first_difference(got, want);
        if (at >= 0) {
            std::printf("FAIL a second request sent in two chunks is not a fresh run: logit %d is %.9g, fresh says %.9g\n",
                        at, got[(size_t)at], want[(size_t)at]);
            return 1;
        }
    }
    std::printf("ok: %d sequences x %d rounds, every row bit-identical to its own single-sequence run\n", users, kRounds);
    return 0;
}
