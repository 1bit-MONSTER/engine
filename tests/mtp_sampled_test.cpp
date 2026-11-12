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


// mtp_sampled_test -- mtp::run_from() under a sampler (npu/mtp/draft.h RowSampler, engine#1781 gate 2): the tokens a
// speculative run emits follow the base's sampling distribution, as plain sampling's do, for both round shapes.
//
// The base is a Markov chain over V tokens (a row's logits depend only on the token just consumed), so the exact
// probability of every 3-token reply is a product of three softmax entries. Each arm runs N seeded requests and
// compares the histogram of replies with that product (total variation distance), next to plain sampling through the
// same loop as the control. The drafter proposes the chain's SECOND most likely token, so drafts are accepted some of
// the time and refused some of the time, and both branches of the rule are exercised in every arm.
//   [1] plain sampling (k = 0) matches the exact distribution (the control: the histogram and the bound are sound);
//   [2] the one-pass round (needs_last_hidden() == false) matches it, with accepted drafts and refusals both seen;
//   [3] the two-pass round matches it, likewise;
//   [4] every run leaves the base holding exactly the emitted tokens (prompt + reply, nothing rejected kept);
//   [5] the same seed gives the same reply (the loop draws only through the caller's sampler);
//   [6] a sampler that keeps every draft is caught by the same bound (the test can fail);
//   [7] over 24-token replies (many rounds each, k = 1 and 4), every transition prev -> next follows the chain's row for prev,
//       which a round that carries a stale refusal into a later round breaks;
//   [8] forward-serve's own sampler (app/npu_spec_sampler.h, sample_token() and sample_probs() underneath) with
//       temperature, top_k and top_p: both round shapes against the exact product of sample_probs() rows.
#include "mtp/draft.h"
#include "npu_spec_sampler.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

using onebit::npu::mtp::SpecStep;
using onebit::npu::mtp::BaseModel;
using onebit::npu::mtp::DraftHead;
using onebit::npu::mtp::Options;
using onebit::npu::mtp::Result;
using onebit::npu::mtp::RowSampler;

namespace {

constexpr int V = 5;
constexpr int kReply = 3;
constexpr int kRuns = 200000;
constexpr double kBound = 0.02;   // TV; the three arms land at 0.008-0.009 at this N

// Logits of the row after consuming token t: distinct per row so every transition matters.
float logit(int t, int v) {
    static const float L[V][V] = {{1.2f, 0.9f, 0.1f, -0.4f, 0.3f},
                                  {0.2f, 1.0f, 0.8f, -0.2f, -1.0f},
                                  {-0.5f, 0.4f, 1.1f, 0.9f, 0.0f},
                                  {0.7f, -0.3f, 0.2f, 1.3f, 1.0f},
                                  {1.0f, 0.1f, -0.6f, 0.5f, 0.8f}};
    return L[t][v];
}
std::vector<float> softmax_row(int t) {
    std::vector<float> p(V);
    double m = -1e30, s = 0;
    for (int v = 0; v < V; v++) m = std::max(m, (double)logit(t, v));
    for (int v = 0; v < V; v++) s += std::exp(logit(t, v) - m);
    for (int v = 0; v < V; v++) p[(size_t)v] = float(std::exp(logit(t, v) - m) / s);
    return p;
}
int second_best(int t) {
    std::vector<float> p = softmax_row(t);
    int b = 0;
    for (int v = 1; v < V; v++) if (p[(size_t)v] > p[(size_t)b]) b = v;
    int s = b == 0 ? 1 : 0;
    for (int v = 0; v < V; v++) if (v != b && p[(size_t)v] > p[(size_t)s]) s = v;
    return s;
}

class ChainBase : public BaseModel {
public:
    std::vector<int> held;   // tokens consumed, by position
    int vocab() const override { return V; }
    int hidden_size() const override { return 1; }
    int max_batch_rows() const override { return 8; }
    bool step(int token, int pos, std::vector<float>& logits) override {
        if (pos != (int)held.size()) return false;
        held.push_back(token);
        logits.resize(V);
        for (int v = 0; v < V; v++) logits[(size_t)v] = logit(token, v);
        return true;
    }
    bool step_batch_all(const std::vector<int>& tokens, int start, std::vector<float>& all) override {
        if (start != (int)held.size()) return false;
        start_ = start;
        all.clear();
        for (int t : tokens) {
            held.push_back(t);
            for (int v = 0; v < V; v++) all.push_back(logit(t, v));
        }
        return true;
    }
    bool commit_verify(int accepted) override {
        held.resize((size_t)(start_ + accepted));
        return true;
    }
    const std::vector<float>& last_hidden() const override { return h_; }

private:
    int start_ = 0;
    std::vector<float> h_{0.0f};
};

// Proposes the chain's second most likely successor, k times over its own proposals.
class SecondHead : public DraftHead {
public:
    explicit SecondHead(bool one_pass) : one_pass_(one_pass) {}
    bool needs_last_hidden() const override { return !one_pass_; }
    int draft(const std::vector<float>&, int last, int, int k, std::vector<int>& out, std::vector<float>* conf) override {
        out.clear();
        if (conf) conf->clear();
        for (int i = 0; i < k; i++) out.push_back(last = second_best(last));
        return k;
    }

private:
    bool one_pass_;
};

// Softmax at temperature 1 drawn with the request's generator; check() is spec_accept.h's rule with a one-hot q.
class SoftmaxSampler : public RowSampler {
public:
    explicit SoftmaxSampler(uint64_t seed, bool keep_all = false) : rng_(seed), keep_all_(keep_all) {}
    int pick(const float* logits, int n) override {
        std::vector<float> p = probs(logits, n);
        return onebit::npu::mtp::spec_residual_sample(p.data(), nullptr, n, u());
    }
    SpecStep check(const float* logits, int n, int x) override {
        std::vector<float> p = probs(logits, n), q((size_t)n, 0.0f);
        q[(size_t)x] = 1.0f;
        const double ua = u(), ur = u();
        if (keep_all_) return {true, x};
        ++checks;
        SpecStep s = onebit::npu::mtp::spec_accept_step(p.data(), q.data(), n, x, ua, ur);
        if (s.accepted) ++kept;
        return s;
    }
    int checks = 0, kept = 0;

private:
    static std::vector<float> probs(const float* l, int n) {
        double m = -1e30, s = 0;
        for (int v = 0; v < n; v++) m = std::max(m, (double)l[v]);
        std::vector<float> p((size_t)n);
        for (int v = 0; v < n; v++) s += std::exp(l[v] - m);
        for (int v = 0; v < n; v++) p[(size_t)v] = float(std::exp(l[v] - m) / s);
        return p;
    }
    double u() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng_); }
    std::mt19937_64 rng_;
    bool keep_all_;
};

int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) g_fail++;
}

constexpr int kPrompt = 0;

double exact(int a, int b, int c) {
    return (double)softmax_row(kPrompt)[(size_t)a] * softmax_row(a)[(size_t)b] * softmax_row(b)[(size_t)c];
}

struct Arm {
    double tv = 1.0;
    int checks = 0, kept = 0, bad_state = 0, errors = 0;
};

// mode: 0 plain, 1 one-pass, 2 two-pass
Arm run_arm(int mode, int k, bool keep_all = false) {
    Arm arm;
    std::map<int, int> hist;
    for (int s = 0; s < kRuns; s++) {
        ChainBase base;
        SoftmaxSampler smp(0x9e3779b97f4a7c15ull * (uint64_t)(s + 1) + (uint64_t)mode, keep_all);
        SecondHead head(mode == 1);
        Options o;
        o.k = mode == 0 ? 0 : k;
        o.max_tokens = kReply;
        o.tau = 0.0f;
        o.sampler = &smp;
        std::vector<float> logits;
        base.step(kPrompt, 0, logits);
        Result r = onebit::npu::mtp::run_from(base, mode == 0 ? nullptr : &head, 1, logits, o);
        if (!r.ok || (int)r.tokens.size() != kReply) { arm.errors++; continue; }
        std::vector<int> want{kPrompt};
        want.insert(want.end(), r.tokens.begin(), r.tokens.end());
        if (base.held != want) arm.bad_state++;
        hist[(r.tokens[0] * V + r.tokens[1]) * V + r.tokens[2]]++;
        arm.checks += smp.checks;
        arm.kept += smp.kept;
    }
    double tv = 0;
    for (int a = 0; a < V; a++)
        for (int b = 0; b < V; b++)
            for (int c = 0; c < V; c++) {
                auto it = hist.find((a * V + b) * V + c);
                const double f = it == hist.end() ? 0.0 : double(it->second) / kRuns;
                tv += std::fabs(f - exact(a, b, c));
            }
    arm.tv = tv / 2;
    return arm;
}

// [8]: the serve's sampler. mode 0 plain, 1 one-pass, 2 two-pass; TV of the 3-token replies against the product of
// sample_probs() rows under `sp`.
double serve_sampler_tv(int mode, const onebit::npu::SampleParams& sp) {
    auto row = [&](int t) {
        std::vector<float> l(V);
        for (int v = 0; v < V; v++) l[(size_t)v] = logit(t, v);
        return onebit::npu::sample_probs(l.data(), V, sp, V);
    };
    std::map<int, int> hist;
    for (int s = 0; s < kRuns; s++) {
        ChainBase base;
        std::mt19937_64 rng(0x2545f4914f6cdd1dull * (uint64_t)(s + 1) + (uint64_t)mode);
        onebit::npu::SpecSampler smp(sp, rng);
        SecondHead head(mode == 1);
        Options o;
        o.k = mode == 0 ? 0 : 3;
        o.max_tokens = kReply;
        o.tau = 0.0f;
        o.sampler = &smp;
        std::vector<float> logits;
        base.step(kPrompt, 0, logits);
        Result r = onebit::npu::mtp::run_from(base, mode == 0 ? nullptr : &head, 1, logits, o);
        if (!r.ok || (int)r.tokens.size() != kReply) return 1.0;
        hist[(r.tokens[0] * V + r.tokens[1]) * V + r.tokens[2]]++;
    }
    double tv = 0;
    const std::vector<float> p0 = row(kPrompt);
    for (int a = 0; a < V; a++) {
        const std::vector<float> pa = row(a);
        for (int b = 0; b < V; b++) {
            const std::vector<float> pb = row(b);
            for (int c = 0; c < V; c++) {
                auto it = hist.find((a * V + b) * V + c);
                const double f = it == hist.end() ? 0.0 : double(it->second) / kRuns;
                tv += std::fabs(f - (double)p0[(size_t)a] * pa[(size_t)b] * pb[(size_t)c]);
            }
        }
    }
    return tv / 2;
}

// [7]: worst row's TV between the observed next-token frequencies after `prev` and softmax_row(prev), over every
// consecutive pair (the prompt included) of `runs` replies of `len` tokens.
double transition_tv(int mode, int k, int runs, int len) {
    std::vector<std::vector<double>> n(V, std::vector<double>(V, 0.0));
    for (int s = 0; s < runs; s++) {
        ChainBase base;
        SoftmaxSampler smp(0x51ed270b27f4a7c1ull * (uint64_t)(s + 1) + (uint64_t)mode);
        SecondHead head(mode == 1);
        Options o;
        o.k = k;
        o.max_tokens = len;
        o.tau = 0.0f;
        o.sampler = &smp;
        std::vector<float> logits;
        base.step(kPrompt, 0, logits);
        Result r = onebit::npu::mtp::run_from(base, &head, 1, logits, o);
        if (!r.ok) return 1.0;
        int prev = kPrompt;
        for (int t : r.tokens) { n[(size_t)prev][(size_t)t] += 1; prev = t; }
    }
    double worst = 0;
    for (int a = 0; a < V; a++) {
        double tot = 0, tv = 0;
        for (int b = 0; b < V; b++) tot += n[(size_t)a][(size_t)b];
        const std::vector<float> p = softmax_row(a);
        for (int b = 0; b < V; b++) tv += std::fabs(n[(size_t)a][(size_t)b] / tot - p[(size_t)b]);
        worst = std::max(worst, tv / 2);
    }
    return worst;
}

}  // namespace

int main() {
    const char* names[] = {"plain", "one-pass", "two-pass"};
    for (int mode = 0; mode < 3; mode++) {
        const Arm a = run_arm(mode, 3);
        std::printf("  %-8s tv %.4f  checks %d kept %d  errors %d bad_state %d\n", names[mode], a.tv, a.checks, a.kept,
                    a.errors, a.bad_state);
        char what[160];
        std::snprintf(what, sizeof what, "[%d] %s: reply distribution within TV %.2f of the exact product", mode + 1,
                      names[mode], kBound);
        check(a.errors == 0 && a.tv < kBound, what);
        if (mode > 0) {
            std::snprintf(what, sizeof what, "[%d] %s: drafts both kept and refused", mode + 1, names[mode]);
            check(a.kept > 0 && a.kept < a.checks, what);
        }
        std::snprintf(what, sizeof what, "[4] %s: the base holds exactly prompt + reply", names[mode]);
        check(a.bad_state == 0, what);
    }

    {   // [5] reproducible from the seed
        auto once = [](uint64_t seed) {
            ChainBase base;
            SoftmaxSampler smp(seed);
            SecondHead head(true);
            Options o;
            o.k = 4;
            o.max_tokens = 24;
            o.tau = 0.0f;
            o.sampler = &smp;
            std::vector<float> logits;
            base.step(kPrompt, 0, logits);
            return onebit::npu::mtp::run_from(base, &head, 1, logits, o).tokens;
        };
        const auto a = once(7), b = once(7), c = once(8);
        check(a == b && a.size() == 24, "[5] the same seed gives the same 24-token reply");
        check(a != c, "[5] a different seed gives a different reply (the sampler is in the loop)");
    }

    {   // [6] keeping every draft is a different distribution, and the bound sees it
        const Arm a = run_arm(1, 3, true);
        std::printf("  keep-all tv %.4f\n", a.tv);
        check(a.tv > 3 * kBound, "[6] a sampler that keeps every draft fails the bound");
    }

    // [7] at k = 1 (a round keeps its only draft about a quarter of the time, so a round after a fully kept one is
    // common) and k = 4 (long chains, refusals deep in them).
    for (int mode = 1; mode < 3; mode++)
        for (int k : {1, 4}) {
            const double tv = transition_tv(mode, k, 20000, 24);
            std::printf("  %-8s k=%d transitions, worst row tv %.4f\n", names[mode], k, tv);
            char what[160];
            std::snprintf(what, sizeof what,
                          "[7] %s k=%d: every transition of 24-token replies within TV %.2f of its row", names[mode],
                          k, kBound);
            check(tv < kBound, what);
        }

    {   // [8]
        onebit::npu::SampleParams sp;
        sp.temperature = 0.8f;
        sp.top_k = 4;
        sp.top_p = 0.9f;
        for (int mode = 0; mode < 3; mode++) {
            const double tv = serve_sampler_tv(mode, sp);
            std::printf("  %-8s serve sampler (t 0.8, top_k 4, top_p 0.9) tv %.4f\n", names[mode], tv);
            char what[160];
            std::snprintf(what, sizeof what, "[8] %s: forward-serve's sampler within TV %.2f of the exact product",
                          names[mode], kBound);
            check(tv < kBound, what);
        }
    }

    std::printf(g_fail ? "%d check(s) FAILED\n" : "all checks passed\n", g_fail);
    return g_fail ? 1 : 0;
}
