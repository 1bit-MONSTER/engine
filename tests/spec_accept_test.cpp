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
//
// spec_accept_test -- npu/mtp/spec_accept.h, the distribution-following acceptance rule (engine#1781, gate 2).
//
// Two properties make the rule what the issue asks for, and each has an arm that a wrong rule fails:
//
//   1. at temperature 0 (one-hot rows) it is exactly "accept iff the draft is the target's argmax, else emit the
//      argmax" -- the contract tests/mtp_draft_test.cpp already pins for the greedy loop;
//   2. with a sampler, the emitted token follows the TARGET's distribution p whatever the drafter's q is -- checked
//      with a drafter that is deliberately wrong, and against the greedy rule, which misses p by 0.55 on the same rows.
//
// No model and no device: a fixed-seed generator, so the counts below are the same on every run and the tolerances
// are bounds on a known sample, not on luck. The sampling error of a frequency over N = 200,000 draws is at most
// sqrt(0.25 / N) = 0.0011 (one sigma), so 0.006 is more than five sigma and 0.3 is nowhere near it.
#include "mtp/spec_accept.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using onebit::npu::mtp::spec_accept_prefix;
using onebit::npu::mtp::spec_accept_step;
using onebit::npu::mtp::spec_residual_sample;
using onebit::npu::mtp::SpecResult;
using onebit::npu::mtp::SpecStep;

static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) g_fail++;
}

struct Rng {   // splitmix64: small, fixed, and good enough to make 200k draws look uniform
    uint64_t s;
    double operator()() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        return double(z >> 11) * (1.0 / 9007199254740992.0);   // [0, 1)
    }
};

static int draw(const std::vector<float>& d, double u) {
    double acc = 0.0;
    for (size_t i = 0; i < d.size(); ++i) { acc += d[i]; if (u < acc) return int(i); }
    return int(d.size()) - 1;
}

// Emitted-token frequencies and the acceptance rate of one position over n draws, with the drafter sampling q.
static double max_dev_from_p(const std::vector<float>& p, const std::vector<float>& q, int n, uint64_t seed,
                             double* accept_rate, bool greedy_rule = false) {
    const int V = int(p.size());
    std::vector<long> cnt(size_t(V), 0);
    long acc = 0;
    Rng r{seed};
    int amax = 0;
    for (int i = 1; i < V; ++i) if (p[size_t(i)] > p[size_t(amax)]) amax = i;
    for (int i = 0; i < n; ++i) {
        const int x = draw(q, r());
        int out;
        if (greedy_rule) {                       // the rule this header replaces: accept iff x is the target's argmax
            const double ua = r(), ur = r(); (void)ua; (void)ur;
            if (x == amax) acc++;
            out = amax;
        } else {
            const SpecStep s = spec_accept_step(p.data(), q.data(), V, x, r(), r());
            if (s.accepted) acc++;
            out = s.token;
        }
        if (out >= 0 && out < V) cnt[size_t(out)]++;
    }
    double worst = 0.0;
    for (int i = 0; i < V; ++i) worst = std::fmax(worst, std::fabs(double(cnt[size_t(i)]) / n - double(p[size_t(i)])));
    if (accept_rate) *accept_rate = double(acc) / n;
    return worst;
}

int main() {
    const int N = 200000;
    char msg[256];

    // ---- 1. temperature 0: one-hot rows, the greedy contract ------------------------------------------------------
    {
        const std::vector<float> pa = {0, 1, 0, 0}, qa = {0, 1, 0, 0}, qb = {0, 0, 1, 0};
        bool all_acc = true, none_acc = true, emit_ok = true;
        Rng r{1};
        for (int i = 0; i < 1000; ++i) {
            const SpecStep s = spec_accept_step(pa.data(), qa.data(), 4, 1, r(), r());
            all_acc &= s.accepted && s.token == 1;
            const SpecStep t = spec_accept_step(pa.data(), qb.data(), 4, 2, r(), r());
            none_acc &= !t.accepted;
            emit_ok &= t.token == 1;
        }
        check(all_acc, "temp 0: a draft equal to the target's argmax is accepted on every draw");
        check(none_acc, "temp 0: a draft that is not the target's argmax is rejected on every draw");
        check(emit_ok, "temp 0: the token emitted at a rejection is the target's argmax");
    }

    // ---- 2. a sampler, with a deliberately wrong drafter: the output still follows p ------------------------------
    {
        const std::vector<float> p = {0.45f, 0.35f, 0.20f}, q = {0.10f, 0.80f, 0.10f};
        double rate = 0.0;
        const double dev = max_dev_from_p(p, q, N, 42, &rate);
        std::snprintf(msg, sizeof msg, "poor drafter: emitted frequencies are within 0.006 of p (max deviation %.4f)", dev);
        check(dev < 0.006, msg);
        std::snprintf(msg, sizeof msg, "poor drafter: acceptance rate is sum(min(p, q)) = 0.55 (measured %.4f)", rate);
        check(std::fabs(rate - 0.55) < 0.006, msg);
        double grate = 0.0;
        const double gdev = max_dev_from_p(p, q, N, 42, &grate, true);
        std::snprintf(msg, sizeof msg, "the GREEDY rule on the same rows misses p by more than 0.3 (max deviation %.4f): "
                                       "this arm tells the two rules apart", gdev);
        check(gdev > 0.3, msg);
    }
    {
        const std::vector<float> p = {0.45f, 0.35f, 0.20f}, q = {0.40f, 0.38f, 0.22f};
        double rate = 0.0;
        const double dev = max_dev_from_p(p, q, N, 7, &rate);
        std::snprintf(msg, sizeof msg, "good drafter: follows p (max deviation %.4f) and accepts 0.95 (measured %.4f)", dev, rate);
        check(dev < 0.006 && std::fabs(rate - 0.95) < 0.006, msg);
    }
    {
        const std::vector<float> p = {0.5f, 0.3f, 0.2f};
        double rate = 0.0;
        const double dev = max_dev_from_p(p, p, N, 9, &rate);
        std::snprintf(msg, sizeof msg, "identical drafter: every draft is accepted (rate %.6f) and the output follows p (%.4f)", rate, dev);
        check(rate == 1.0 && dev < 0.006, msg);
    }
    {
        const std::vector<float> p = {0.6f, 0.4f, 0.0f, 0.0f}, q = {0.0f, 0.0f, 0.5f, 0.5f};
        double rate = 1.0;
        const double dev = max_dev_from_p(p, q, N, 11, &rate);
        std::snprintf(msg, sizeof msg, "disjoint drafter: nothing is accepted (rate %.6f) and the output still follows p (%.4f)", rate, dev);
        check(rate == 0.0 && dev < 0.006, msg);
    }
    {   // a deterministic (argmax) drafter states a one-hot q: accepted with probability p(x), residual is p without x
        const std::vector<float> p = {0.2f, 0.5f, 0.3f}, q = {0.0f, 1.0f, 0.0f};
        double rate = 0.0;
        const double dev = max_dev_from_p(p, q, N, 13, &rate);
        std::snprintf(msg, sizeof msg, "one-hot drafter: accepted with probability p(x) = 0.5 (measured %.4f), output follows p (%.4f)", rate, dev);
        check(std::fabs(rate - 0.5) < 0.006 && dev < 0.006, msg);
    }
    {   // a wide vocabulary with random rows: the property is not an artefact of three tokens
        const int V = 257;
        Rng g{2024};
        std::vector<float> p(V), q(V);
        double sp = 0, sq = 0;
        for (int i = 0; i < V; ++i) { p[size_t(i)] = float(g() * g()); q[size_t(i)] = float(g() * g()); sp += p[size_t(i)]; sq += q[size_t(i)]; }
        double want = 0;
        for (int i = 0; i < V; ++i) { p[size_t(i)] = float(p[size_t(i)] / sp); q[size_t(i)] = float(q[size_t(i)] / sq); }
        for (int i = 0; i < V; ++i) want += std::fmin(p[size_t(i)], q[size_t(i)]);
        double rate = 0.0;
        const double dev = max_dev_from_p(p, q, 2 * N, 5, &rate);
        std::snprintf(msg, sizeof msg, "257 tokens, random rows: output within 0.003 of p (%.4f), acceptance %.4f against sum(min) %.4f", dev, rate, want);
        check(dev < 0.003 && std::fabs(rate - want) < 0.006, msg);
    }

    // ---- 3. the edges a wiring will hit ---------------------------------------------------------------------------
    {
        const std::vector<float> p = {0.5f, 0.5f, 0.0f}, q = {1.0f, 0.0f, 0.0f};
        bool ok = true;
        Rng r{3};
        for (int i = 0; i < 2000; ++i) {
            const SpecStep s = spec_accept_step(p.data(), q.data(), 3, 1, r(), r());   // x has q(x) == 0
            ok &= !s.accepted && (s.token == 0 || s.token == 1);
        }
        check(ok, "a draft the drafter gives no mass to is rejected, and the emitted token stays in p's support");
        const SpecStep bad = spec_accept_step(p.data(), q.data(), 3, 7, 0.0, 0.5);
        check(!bad.accepted && bad.token >= 0 && bad.token < 2, "an out-of-range draft id is rejected and resampled, not read");
    }
    {
        const std::vector<float> p = {0.25f, 0.0f, 0.75f, 0.0f}, q = {0.25f, 0.5f, 0.25f, 0.0f};
        check(spec_residual_sample(p.data(), q.data(), 4, 0.0) == 2 && spec_residual_sample(p.data(), q.data(), 4, 0.999999999) == 2,
              "the residual max(0, p - q) is sampled: only token 2 has residual mass here, at both ends of u");
        const std::vector<float> same = {0.0f, 0.3f, 0.7f, 0.0f};
        check(spec_residual_sample(same.data(), same.data(), 4, 0.0) == 1 && spec_residual_sample(same.data(), same.data(), 4, 0.9999999999) == 2,
              "an empty residual (p == q) falls back to p and never returns a zero-probability token");
        const std::vector<float> zero = {0, 0, 0};
        check(spec_residual_sample(zero.data(), zero.data(), 3, 0.5) == -1 && spec_residual_sample(nullptr, nullptr, 3, 0.5) == -1,
              "a row with no mass, or no row, returns -1 instead of a token");
    }

    // ---- 4. the chain: stop at the first rejection, two draws per position examined ------------------------------
    {
        const int V = 3, k = 3;
        const std::vector<float> P = {0, 1, 0,   0, 0, 1,   1, 0, 0};   // target argmaxes 1, 2, 0
        const std::vector<float> Q = {0, 1, 0,   0, 0, 1,   0, 1, 0};   // drafter agrees twice, then says 1
        const int drafts[3] = {1, 2, 1};
        int draws = 0;
        Rng r{21};
        const SpecResult res = spec_accept_prefix(P.data(), Q.data(), V, drafts, k, [&] { draws++; return r(); });
        check(res.accepted == 2 && res.next_token == 0, "chain: two drafts accepted, the third rejected, and the target's token emitted in its place");
        check(draws == 6, "chain: two uniforms are consumed per position examined");
        const int good[3] = {1, 2, 0};
        const std::vector<float> Q2 = P;
        const SpecResult all = spec_accept_prefix(P.data(), Q2.data(), V, good, k, [&] { return r(); });
        check(all.accepted == 3 && all.next_token == -1, "chain: when all k survive, next_token is -1 (the caller samples the pass's last row)");
        const int first_bad[3] = {0, 2, 0};
        const std::vector<float> Q3 = {1, 0, 0,   0, 0, 1,   1, 0, 0};
        draws = 0;
        const SpecResult none = spec_accept_prefix(P.data(), Q3.data(), V, first_bad, k, [&] { draws++; return r(); });
        check(none.accepted == 0 && none.next_token == 1 && draws == 2, "chain: a rejection at the first position stops there, later rows are not examined");
    }
    {   // the chain preserves the distribution at the position where it stops, too: two sampled positions
        const int V = 3;
        const std::vector<float> P = {0.45f, 0.35f, 0.20f,   0.10f, 0.60f, 0.30f};
        const std::vector<float> Q = {0.10f, 0.80f, 0.10f,   0.30f, 0.30f, 0.40f};
        std::vector<long> c0(3, 0), c1(3, 0);
        long n1 = 0;
        Rng r{77};
        for (int i = 0; i < N; ++i) {
            const int d[2] = {draw({Q[0], Q[1], Q[2]}, r()), draw({Q[3], Q[4], Q[5]}, r())};
            const SpecResult res = spec_accept_prefix(P.data(), Q.data(), V, d, 2, [&] { return r(); });
            const int t0 = res.accepted >= 1 ? d[0] : res.next_token;
            c0[size_t(t0)]++;
            if (res.accepted >= 1) { n1++; c1[size_t(res.accepted == 2 ? d[1] : res.next_token)]++; }
        }
        double dev0 = 0, dev1 = 0;
        for (int i = 0; i < 3; ++i) {
            dev0 = std::fmax(dev0, std::fabs(double(c0[size_t(i)]) / N - double(P[size_t(i)])));
            dev1 = std::fmax(dev1, std::fabs(double(c1[size_t(i)]) / double(n1) - double(P[size_t(3 + i)])));
        }
        std::snprintf(msg, sizeof msg, "chain of two sampled positions: each emitted token follows its own row of p (%.4f, %.4f)", dev0, dev1);
        check(dev0 < 0.006 && dev1 < 0.008, msg);
    }

    std::printf("%s\n", g_fail ? "spec_accept_test: FAILED" : "spec_accept_test: all arms passed");
    return g_fail ? 1 : 0;
}
