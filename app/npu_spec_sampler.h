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

// npu_spec_sampler.h -- a sampled request's choices inside the speculative loop (engine#1781 gate 2): the
// npu/mtp/draft.h RowSampler that `1bit forward-serve --spec-sample` hands mtp::run_from().
//
// pick() is the plain loop's sample_token() on the row. check() keeps a drafted token x with probability p(x), p being
// what sample_token() draws from under the same parameters (sample_probs()), and otherwise emits a draw from p without
// x: npu/mtp/spec_accept.h's rule for a drafter whose own distribution is one-hot at its proposal, which is what the
// prompt lookup and a DFlash head are. Both uniforms of a check come from the request's generator, in order, so a
// reply is reproducible from its seed. The reply follows the plain loop's distribution, not its random stream.
// tests/mtp_sampled_test.cpp holds the distribution, top_k and top_p included.
//
// Under a constraint (a grammar) every choice is made inside its mask, as the plain loop's sample_token() is: pick() in
// the state the emitted tokens left, and draft i in that state advanced past drafts 0..i-1 (TokenConstraint keeps the
// round's own copy). Greedy parameters are served too: check() then keeps x iff it is the masked argmax, and the
// refusal's token is that argmax, so a constrained greedy reply is the plain loop's token for token.
#pragma once

#include <random>
#include <vector>

#include "../npu/mtp/draft.h"
#include "npu_sampling.h"

namespace onebit::npu {

// Which tokens may come next. The committed state is the one the emitted tokens left (the caller advances it as it
// delivers them); the round's state starts as a copy of it and moves past each kept draft.
class TokenConstraint {
public:
    virtual ~TokenConstraint() = default;
    virtual void mask_committed(std::vector<char>& allowed) = 0;
    virtual void begin_round() = 0;
    virtual void mask_round(std::vector<char>& allowed) = 0;
    virtual void advance_round(int token) = 0;
};

class SpecSampler : public mtp::RowSampler {
public:
    SpecSampler(const SampleParams& p, std::mt19937_64& rng, TokenConstraint* c = nullptr) : sp_(p), rng_(rng), c_(c) {}
    int pick(const float* logits, int n) override {
        if (c_) c_->mask_committed(mask_);
        return sample_token(std::vector<float>(logits, logits + n), sp_, rng_, c_ ? &mask_ : nullptr, n);
    }
    void begin_round() override {
        if (c_) c_->begin_round();
    }
    mtp::SpecStep check(const float* logits, int n, int x) override {
        if (c_) c_->mask_round(mask_);
        const std::vector<float> p = sample_probs(logits, size_t(n), sp_, n, c_ ? &mask_ : nullptr);
        mtp::SpecStep s;
        if (sp_.greedy()) {
            // the masked argmax is the only token with mass; -1 when the mask allows nothing
            int best = -1;
            for (int i = 0; i < n; i++)
                if (p[size_t(i)] > 0.0f) { best = i; break; }
            s = {best >= 0 && best == x, best};
        } else {
            std::vector<float> q(size_t(n), 0.0f);
            if (x >= 0 && x < n) q[size_t(x)] = 1.0f;
            std::uniform_real_distribution<double> u(0.0, 1.0);
            const double ua = u(rng_), ur = u(rng_);
            s = mtp::spec_accept_step(p.data(), q.data(), n, x, ua, ur);
        }
        if (s.accepted && c_) c_->advance_round(x);
        return s;
    }

private:
    const SampleParams& sp_;
    std::mt19937_64& rng_;
    TokenConstraint* c_;
    std::vector<char> mask_;
};

}  // namespace onebit::npu
