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
#pragma once

#include <random>
#include <vector>

#include "../npu/mtp/draft.h"
#include "npu_sampling.h"

namespace onebit::npu {

class SpecSampler : public mtp::RowSampler {
public:
    SpecSampler(const SampleParams& p, std::mt19937_64& rng) : sp_(p), rng_(rng) {}
    int pick(const float* logits, int n) override {
        return sample_token(std::vector<float>(logits, logits + n), sp_, rng_, nullptr, n);
    }
    mtp::SpecStep check(const float* logits, int n, int x) override {
        const std::vector<float> p = sample_probs(logits, size_t(n), sp_, n);
        std::vector<float> q(size_t(n), 0.0f);
        if (x >= 0 && x < n) q[size_t(x)] = 1.0f;
        std::uniform_real_distribution<double> u(0.0, 1.0);
        const double ua = u(rng_), ur = u(rng_);
        return mtp::spec_accept_step(p.data(), q.data(), n, x, ua, ur);
    }

private:
    const SampleParams& sp_;
    std::mt19937_64& rng_;
};

}  // namespace onebit::npu
