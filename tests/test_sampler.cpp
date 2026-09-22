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
#include <map>
#include <vector>

#include "check.h"
#include "sampler.h"

using namespace onebit;

static std::map<int32_t, int> histogram(SamplingParams p, const std::vector<float>& logits, int n) {
    Sampler s(p);
    std::map<int32_t, int> h;
    for (int i = 0; i < n; ++i) ++h[s.sample(logits)];
    return h;
}

int main() {
    const std::vector<float> logits = {1.0f, 3.0f, 2.9f, -5.0f, 0.5f};

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    CHECK(Sampler(greedy).sample(logits) == 1);
    SamplingParams k1;
    k1.top_k = 1;
    CHECK(Sampler(k1).sample(logits) == 1);

    // Same seed, same sequence; different seed, (almost surely) different.
    SamplingParams hot;
    hot.temperature = 1.5f;
    hot.top_k = 0;
    hot.top_p = 1.0f;
    hot.min_p = 0.0f;
    hot.seed = 42;
    Sampler a(hot), b(hot);
    std::vector<int32_t> sa, sb;
    for (int i = 0; i < 64; ++i) {
        sa.push_back(a.sample(logits));
        sb.push_back(b.sample(logits));
    }
    CHECK(sa == sb);

    // top-k 2 only ever yields the two largest logits.
    SamplingParams k2 = hot;
    k2.top_k = 2;
    for (auto [id, n] : histogram(k2, logits, 2000)) CHECK(id == 1 || id == 2);

    // top-p: p(1)~0.49, p(2)~0.44 -> 0.5 needs both; token 0 (~0.05) excluded.
    SamplingParams tp = hot;
    tp.top_p = 0.5f;
    for (auto [id, n] : histogram(tp, logits, 2000)) CHECK(id == 1 || id == 2);

    // min-p 0.5: only candidates with p >= 0.5 * p_max survive (tokens 1 and 2).
    SamplingParams mp = hot;
    mp.min_p = 0.5f;
    for (auto [id, n] : histogram(mp, logits, 2000)) CHECK(id == 1 || id == 2);

    // No filters: every token with non-negligible mass shows up.
    auto all = histogram(hot, logits, 20000);
    CHECK(all.count(0) && all.count(1) && all.count(2) && all.count(4));

    return test_result("test_sampler");
}
