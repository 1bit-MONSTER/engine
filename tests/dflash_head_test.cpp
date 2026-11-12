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

// dflash_head_test.cpp -- the DFlash draft as mtp::run()'s drafter (npu/dflash_head.h, engine#1781). Host only.
//
// The base is a toy that writes one tap row per position it runs -- the rejected rows of a verify batch too, as the
// forward's tap sink does -- and whose greedy choice is a fixed function of the token. The draft is a small random
// DraftWeights; its pick rule is the base's own rule (every draft right), that rule with every third draft wrong, or
// always wrong. Checked, for each, at k = 2 and 7 with a block of 8:
//   [1] the text equals plain decode's;
//   [2] the sink ends holding exactly the committed stream's rows (prompt plus every emitted token), bit for bit: no
//       rejected row survives a commit;
//   [3] every draft equals draft_forward() over the committed stream's rows alone, computed from scratch, bit for bit;
//   [4] the right-everywhere rule has every draft accepted, the always-wrong one none;
// and that a draft at a position the sink does not end at is refused, with a reason.

#include "dflash_head.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using onebit::npu::dflash::DFlashHead;
using onebit::npu::dflash::DraftLayerWeights;
using onebit::npu::dflash::DraftWeights;
using onebit::npu::dflash::HeadHooks;
using onebit::npu::mtp::BaseModel;
using onebit::npu::mtp::DraftHead;
using onebit::npu::mtp::Options;
using onebit::npu::mtp::Result;

namespace {

int fails = 0;
void check(bool cond, const std::string& what) {
    std::printf("%s  %s\n", cond ? "PASS" : "FAIL", what.c_str());
    if (!cond) fails++;
}

constexpr int V = 16, H = 8, T = 2, BLOCK = 8;

int next_of(int token) { return (token * 5 + 3) % V; }

float hashf(uint64_t a, uint64_t b) {
    uint64_t u = a * 0x9E3779B97F4A7C15ULL ^ (b + 0x632BE59BD9B4E019ULL);
    u ^= u >> 29; u *= 0xBF58476D1CE4E5B9ULL; u ^= u >> 32;
    return float(double(u >> 11) / double(1ULL << 53)) * 2.0f - 1.0f;
}
// The tap row the base writes for `token` run at `pos`.
void tap_row(int token, int pos, float* out) {
    for (int i = 0; i < T * H; i++) out[i] = hashf(uint64_t(token) * 1000003ULL + uint64_t(pos), uint64_t(i));
}

class TapBase : public BaseModel {
public:
    std::vector<float> taps;   // the sink
    int vocab() const override { return V; }
    int hidden_size() const override { return 1; }
    int max_batch_rows() const override { return 16; }
    bool step(int token, int pos, std::vector<float>& logits) override {
        if (pos != consumed_) return false;
        put(token, pos);
        consumed_++;
        logits.assign(V, 0.0f);
        logits[size_t(next_of(token))] = 1.0f;
        hidden_.assign(1, float(consumed_));
        return true;
    }
    bool step_batch_all(const std::vector<int>& tokens, int start, std::vector<float>& all) override {
        if (start != consumed_) return false;
        all.assign(tokens.size() * size_t(V), 0.0f);
        rows_.clear();
        for (size_t m = 0; m < tokens.size(); m++) {
            put(tokens[m], start + int(m));
            all[m * size_t(V) + size_t(next_of(tokens[m]))] = 1.0f;
            rows_.push_back(float(start + int(m) + 1));
        }
        pending_ = int(tokens.size());
        consumed_ += pending_;
        return true;
    }
    bool commit_verify(int a) override {   // positions only; the sink is the drafter's to trim, as on the forward
        if (a < 0 || a > pending_) return false;
        consumed_ -= pending_ - a;
        rows_.resize(size_t(a));
        pending_ = 0;
        return true;
    }
    const std::vector<float>& last_hidden() const override { return hidden_; }
    const std::vector<float>& last_hidden_rows() const override { return rows_; }

private:
    void put(int token, int pos) {
        const size_t o = taps.size();
        taps.resize(o + size_t(T * H));
        tap_row(token, pos, taps.data() + o);
    }
    int consumed_ = 0, pending_ = 0;
    std::vector<float> hidden_, rows_;
};

DraftWeights random_model() {
    DraftWeights w;
    w.hidden = H; w.intermediate = 12; w.heads = 4; w.kv_heads = 2; w.head_dim = 4; w.taps = T;
    w.rms_eps = 1e-6f; w.rope_theta = 10000.0f;
    uint64_t c = 0;
    auto vec = [&](size_t n, float s) {
        std::vector<float> v(n);
        for (float& f : v) f = s * hashf(7, c++);
        return v;
    };
    auto gain = [&](size_t n) {
        std::vector<float> g = vec(n, 0.3f);
        for (float& f : g) f += 1.0f;
        return g;
    };
    w.fc = vec(size_t(H) * H * T, 0.5f);
    w.hidden_norm = gain(H);
    w.norm = gain(H);
    for (int l = 0; l < 2; l++) {
        DraftLayerWeights L;
        L.q = vec(16 * H, 0.6f); L.k = vec(8 * H, 0.6f); L.v = vec(8 * H, 0.6f); L.o = vec(H * 16, 0.4f);
        L.gate = vec(12 * H, 0.5f); L.up = vec(12 * H, 0.5f); L.down = vec(H * 12, 0.4f);
        L.input_norm = gain(H); L.post_norm = gain(H); L.q_norm = gain(4); L.k_norm = gain(4);
        w.layers.push_back(L);
    }
    return w;
}

void embed(int token, std::vector<float>& row) {
    row.resize(H);
    for (int i = 0; i < H; i++) row[size_t(i)] = hashf(uint64_t(token) + 99, uint64_t(i));
}

// Records each draft's rows and the context it drafted over, so [3] can recompute them afterwards.
struct Logged : DraftHead {
    DFlashHead& h;
    std::vector<int> ctx, last;
    std::vector<std::vector<float>> out;
    explicit Logged(DFlashHead& x) : h(x) {}
    bool needs_last_hidden() const override { return h.needs_last_hidden(); }
    int draft(const std::vector<float>& hid, int lt, int pos, int k, std::vector<int>& o,
              std::vector<float>* conf) override {
        const int n = h.draft(hid, lt, pos, k, o, conf);
        // The context a draft at `pos` must see is the pos - 1 committed rows before it, whatever the head reports.
        if (n > 0) { ctx.push_back(pos - 1); last.push_back(lt); out.push_back(h.last_draft_rows()); }
        return n;
    }
    void commit_rows(const std::vector<int>& t, const std::vector<float>& r, int M) override { h.commit_rows(t, r, M); }
};

}  // namespace

int main() {
    const DraftWeights w = random_model();
    const std::vector<int> prompt = {1, 6, 11, 2};
    const int WANT = 30;
    Options o;
    o.max_tokens = WANT;
    o.tau = 0.0f;
    TapBase pb;
    const Result plain = onebit::npu::mtp::run_plain(pb, prompt, o);
    check(plain.ok && int(plain.tokens.size()) == WANT, "plain decode on the tap base");
    const char* names[] = {"right", "every third wrong", "wrong"};
    for (int mode = 0; mode < 3; mode++) {
        for (int k : {2, 7}) {
            TapBase b;
            int calls = 0;
            HeadHooks hooks;
            hooks.embed = [](int t, std::vector<float>& r) { embed(t, r); return t >= 0 && t < V; };
            hooks.pick = [&](const float*, int prev) {
                const int t = next_of(prev);
                const bool wrong = mode == 2 || (mode == 1 && ++calls % 3 == 0);
                return wrong ? (t + 1) % V : t;
            };
            DFlashHead head(w, b.taps, hooks, /*mask*/ 15, BLOCK);
            Logged lg(head);
            o.k = k;
            const Result r = onebit::npu::mtp::run(b, &lg, prompt, o);
            const std::string tag = std::string("k=") + std::to_string(k) + " pick " + names[mode] + ": ";
            check(r.ok && r.tokens == plain.tokens && head.error().empty(), tag + "text equals plain decode" +
                                                                                (head.error().empty() ? "" : " (" + head.error() + ")"));
            // [2] the committed stream, rebuilt from scratch
            std::vector<int> stream = prompt;
            stream.insert(stream.end(), r.tokens.begin(), r.tokens.end());
            std::vector<float> want(stream.size() * size_t(T * H));
            for (size_t p = 0; p < stream.size(); p++) tap_row(stream[p], int(p), want.data() + p * size_t(T * H));
            check(b.taps == want, tag + "the sink holds exactly the " + std::to_string(stream.size()) +
                                      " committed rows (has " + std::to_string(b.taps.size() / size_t(T * H)) + ")");
            // [3] every draft against draft_forward over the committed rows it should have seen
            int same = 0;
            for (size_t i = 0; i < lg.out.size(); i++) {
                std::vector<float> noise(size_t(BLOCK) * H), row, mask;
                embed(lg.last[i], row);
                embed(15, mask);
                std::copy(row.begin(), row.end(), noise.begin());
                for (int j = 1; j < BLOCK; j++) std::copy(mask.begin(), mask.end(), noise.begin() + j * H);
                const std::vector<float> ref =
                    onebit::npu::dflash::draft_forward(w, want.data(), lg.ctx[i], noise.data(), BLOCK, 0);
                same += ref == lg.out[i] ? 1 : 0;
            }
            check(!lg.out.empty() && same == int(lg.out.size()),
                  tag + std::to_string(same) + " of " + std::to_string(lg.out.size()) +
                      " drafts equal draft_forward over the committed rows, bit for bit");
            // [4]
            if (mode == 0) check(r.stats.drafted > 0 && r.stats.accepted == r.stats.drafted, tag + "every draft accepted");
            if (mode == 1) check(r.stats.accepted > 0 && r.stats.accepted < r.stats.drafted, tag + "a prefix accepted");
            if (mode == 2) check(r.stats.drafted > 0 && r.stats.accepted == 0, tag + "no draft accepted");
            std::printf("INFO %saccepted %d of %d drafts in %d rounds\n", tag.c_str(), r.stats.accepted,
                        r.stats.drafted, r.stats.iterations);
        }
    }
    {
        std::vector<float> taps(size_t(3 * T * H), 0.25f);
        HeadHooks hooks;
        hooks.embed = [](int t, std::vector<float>& r) { embed(t, r); return true; };
        hooks.pick = [](const float*, int prev) { return next_of(prev); };
        DFlashHead head(w, taps, hooks, 15, BLOCK);
        std::vector<int> out;
        const int n = head.draft({}, 1, 6, 4, out);
        check(n == 0 && out.empty() && head.error().find("position 6 over 3") != std::string::npos,
              "a draft at a position the sink does not end at is refused: " + head.error());
        check(head.draft({}, 1, 4, 4, out) == 4 && head.error().empty() && head.last_context_rows() == 3,
              "the same head at the position after its 3 rows drafts 4 tokens over 3 context rows");
    }
    std::printf(fails ? "\n%d CHECK(S) FAILED\n" : "\nall checks passed\n", fails);
    return fails ? 1 : 0;
}
