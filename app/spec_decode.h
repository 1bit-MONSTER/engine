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
// spec_decode.h -- the speculative round: what a drafter has to hand the verifier, and the decision that turns
// M x NV target logits plus a draft's own distribution into the tokens a round emits (engine#1781, the wiring task).
//
// The pieces this does NOT own, deliberately, because they live elsewhere and are already measured:
//   * the acceptance rule itself -- npu/mtp/spec_accept.h (#1958, merged), which reduces to "accept iff the draft is
//     the target's argmax" at temperature 0;
//   * the verify pass -- app/forward_serve.cpp's begin_verify()/step_batch_all(), which already returns one row of
//     logits per position (`M x NV`), so `p` needs no new plumbing;
//   * the drafter -- #1780's DFlash packages and #1957's gate, which will implement DraftProvider.
//
// What it does own is the bookkeeping that a speedup number rests on: which drafts were accepted, in what order, where
// the chain stopped, and how many were proposed -- because a tok/s figure without the acceptance that produced it is
// not reproducible. The decision function is pure and takes its uniforms from the caller, so the round can be tested
// with no model, no device and no RNG, and so the two draws actually come from the sampler's stream rather than from a
// per-position generator (which would follow `p` in distribution but not in *stream*).
#pragma once

#include <cstddef>
#include <vector>

#include "npu/mtp/spec_accept.h"

namespace onebit::spec {

// One drafter answer: up to k proposed ids, and the drafter's own distribution at each of them, row-major k x V.
struct DraftStep {
    std::vector<int> ids;
    std::vector<float> q;
    int rows() const { return int(ids.size()); }
    bool valid(int V) const { return V > 0 && q.size() == ids.size() * std::size_t(V); }
};

// What a speculative loop needs from a drafter, and nothing else. Returning false means "nothing to say" (no
// drafter, out of range, or a refusal), and the caller then takes exactly the non-speculative path it took before.
class DraftProvider {
  public:
    virtual ~DraftProvider() = default;
    virtual bool propose(const std::vector<int>& ctx, int k, int V, DraftStep& out) = 0;
};

// One round's outcome. `emitted` is the accepted drafts in order, plus the resampled token when the chain stops on a
// rejection -- never more than drafts.size() + 0: the resample *replaces* the rejected draft, so a round of k drafts
// emits between 1 and k tokens.
struct SpecRound {
    std::vector<int> emitted;
    int accepted = 0;
    int proposed = 0;
    bool stopped_on_rejection = false;
};

// Decide one round. `p_rows` and `q_rows` are drafts.size() x V, row-major, both aligned with `drafts`. `ua`/`ur` are
// drafts.size() uniforms in [0,1) for the two draws each decision needs, taken from the caller's stream in order: if
// either is null, or either is outside [0,1), the round refuses (returns proposed == 0) rather than drawing for itself.
// A draft whose q(x) is 0 rejects and resamples (#1958's premise: `x` was not drawn from q), which is the case that
// keeps the division out of the rule.
inline SpecRound decide(const float* p_rows, const float* q_rows, int V, const std::vector<int>& drafts,
                        const double* ua, const double* ur) {
    SpecRound out;
    if (!p_rows || !q_rows || V <= 0 || drafts.empty() || !ua || !ur) return out;
    const std::size_t n = drafts.size();
    for (std::size_t d = 0; d < n; ++d) {
        const double a = ua[d], r = ur[d];
        if (!(a >= 0.0 && a < 1.0) || !(r >= 0.0 && r < 1.0)) return SpecRound{};  // refuse, do not draw here
    }
    out.proposed = int(n);
    const float* p = p_rows;
    const float* q = q_rows;
    for (std::size_t d = 0; d < n; ++d) {
        const npu::mtp::SpecStep s =
            npu::mtp::spec_accept_step(p + d * std::size_t(V), q + d * std::size_t(V), V, drafts[d], ua[d], ur[d]);
        out.emitted.push_back(s.token);
        if (s.accepted) {
            ++out.accepted;
            continue;
        }
        out.stopped_on_rejection = true;
        break;  // the chain stops at the first rejection: every later draft was conditioned on a token that was not chosen
    }
    return out;
}

}  // namespace onebit::spec
