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
// spec_accept.h -- the acceptance rule of a speculative decode whose output follows the TARGET's distribution
// (engine#1781, gate 2).
//
// draft.h's loop accepts a drafted token when it equals the base model's argmax. That is exact for greedy decode and
// for nothing else: with a sampler, "accept iff the draft is the target's most likely token, otherwise emit that
// token" always emits the argmax, which is a different model's output (0.55 away from p on p = [.45,.35,.20] with
// q = [.10,.80,.10] -- the arm in tests/spec_accept_test.cpp). The rule that preserves the distribution is the
// standard one, for a drafted token x drawn from the drafter's distribution q at a position where the target's
// distribution is p:
//
//     accept x with probability min(1, p(x) / q(x));
//     on the first rejection, emit a token drawn from normalise(max(0, p - q)) and stop the chain.
//
// The emitted token is then distributed exactly as p, whatever q is; q only decides how often a draft survives
// (the acceptance probability is sum_i min(p_i, q_i)). At temperature 0 both rows are one-hot, the ratio is 0 or 1,
// and the rule IS "accept iff the drafter's token is the target's argmax, else emit the argmax" -- so draft.h's
// greedy contract is this rule's degenerate case and not a second rule.
//
// What the caller owes:
//   * p is the target's FINAL sampling distribution at that position -- after temperature, top-k, top-p and any
//     penalty -- and q is the drafter's, in the same sense. A deterministic drafter (an argmax head) has a one-hot q.
//   * x was drawn from q. If q(x) is 0 the premise is false; the step then rejects and resamples, so the output stays
//     inside p's support, but nothing is promised about a drafter that lies about its own distribution.
//   * the uniforms are independent draws in [0, 1). The functions take them as arguments and own no generator, so a
//     run is reproducible from the caller's seed and a test can drive every branch.
//
// Pure and header-only: no model, no device, no allocation. It decides WHICH rows are kept; begin_verify() /
// commit_verify_rows() decide how the kept rows are committed.
#pragma once

#include <cstddef>

namespace onebit {
namespace npu {
namespace mtp {

struct SpecStep {
    bool accepted;   // the drafted token survives
    int  token;      // the token to emit at this position: the draft when accepted, the residual sample otherwise
};

// A token from normalise(max(0, p - q)), by inverse CDF on u in [0, 1). When the residual has no mass (p == q up to
// rounding, where a rejection has probability 0 and can only come from float error) it samples p itself. Never
// returns an index whose weight is zero, and returns -1 only for V <= 0 or a row with no positive mass at all.
inline int spec_residual_sample(const float* p, const float* q, int V, double u) {
    if (!p || V <= 0) return -1;
    double mass = 0.0;
    if (q)
        for (int i = 0; i < V; ++i) {
            const double r = double(p[i]) - double(q[i]);
            if (r > 0.0) mass += r;
        }
    const bool use_p = !(mass > 0.0);
    if (use_p) {
        mass = 0.0;
        for (int i = 0; i < V; ++i)
            if (p[i] > 0.0f) mass += double(p[i]);
        if (!(mass > 0.0)) return -1;
    }
    if (u < 0.0) u = 0.0;
    const double target = u * mass;
    double acc = 0.0;
    int last = -1;
    for (int i = 0; i < V; ++i) {
        const double w = use_p ? (p[i] > 0.0f ? double(p[i]) : 0.0) : (double(p[i]) - double(q[i]));
        if (!(w > 0.0)) continue;
        acc += w;
        last = i;
        if (target < acc) return i;
    }
    return last;   // u so close to 1 that rounding left `target` at or past the total: the last weighted index
}

// One drafted position. `x` is the token the drafter drew from q.
inline SpecStep spec_accept_step(const float* p, const float* q, int V, int x, double u_accept, double u_resample) {
    if (p && q && x >= 0 && x < V && q[x] > 0.0f) {
        // u < min(1, p/q) written without the division: u * q < p. With p >= q it holds for every u in [0, 1).
        if (u_accept * double(q[x]) < double(p[x])) return {true, x};
    }
    return {false, spec_residual_sample(p, q, V, u_resample)};
}

struct SpecResult {
    int accepted;     // how many leading drafts survive: 0..k
    int next_token;   // the residual sample at the first rejection; -1 when all k survive (the caller then samples
                      // position k from the target's own row there, which is the pass's k+1-th row)
};

// The rule along a chain of k drafted positions. p_rows and q_rows are k rows of V probabilities, row d being the
// distributions at drafted position d; `uniform()` returns the next independent draw in [0, 1). Exactly two draws are
// consumed per position examined, accepted or not, so the number of draws does not leak the outcome into the stream.
template <class Uniform>
inline SpecResult spec_accept_prefix(const float* p_rows, const float* q_rows, int V, const int* drafts, int k,
                                     Uniform&& uniform) {
    for (int d = 0; d < k; ++d) {
        const double ua = uniform();
        const double ur = uniform();
        const SpecStep s = spec_accept_step(p_rows + std::size_t(d) * std::size_t(V),
                                            q_rows + std::size_t(d) * std::size_t(V), V, drafts[d], ua, ur);
        if (!s.accepted) return {d, s.token};
    }
    return {k, -1};
}

}  // namespace mtp
}  // namespace npu
}  // namespace onebit
