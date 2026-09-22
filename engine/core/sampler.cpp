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
#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace onebit {

int32_t argmax(std::span<const float> logits) {
    return int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

int32_t Sampler::sample(std::span<const float> logits) {
    if (p_.temperature <= 0.0f || p_.top_k == 1) return argmax(logits);

    // Candidates sorted by logit, descending (ties: lower id first).
    std::vector<int32_t> ids(logits.size());
    std::iota(ids.begin(), ids.end(), 0);
    size_t k = ids.size();
    if (p_.top_k > 0 && size_t(p_.top_k) < k) k = size_t(p_.top_k);
    auto by_logit = [&](int32_t a, int32_t b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); };
    std::partial_sort(ids.begin(), ids.begin() + std::ptrdiff_t(k), ids.end(), by_logit);
    ids.resize(k);

    // Untempered probabilities of the candidates.
    const double mx = logits[ids[0]];
    std::vector<double> prob(k);
    double sum = 0.0;
    for (size_t i = 0; i < k; ++i) sum += prob[i] = std::exp(double(logits[ids[i]]) - mx);
    for (double& q : prob) q /= sum;

    // top-p: keep the smallest prefix whose mass reaches top_p.
    if (p_.top_p < 1.0f) {
        double cum = 0.0;
        size_t keep = k;
        for (size_t i = 0; i < k; ++i) {
            cum += prob[i];
            if (cum >= p_.top_p) {
                keep = i + 1;
                break;
            }
        }
        k = keep;
    }
    // min-p: drop candidates below min_p times the top probability.
    if (p_.min_p > 0.0f) {
        const double floor = p_.min_p * prob[0];
        size_t keep = 1;
        while (keep < k && prob[keep] >= floor) ++keep;
        k = keep;
    }

    // Temperature, then draw.
    std::vector<double> w(k);
    for (size_t i = 0; i < k; ++i) w[i] = std::exp((double(logits[ids[i]]) - mx) / p_.temperature);
    std::discrete_distribution<size_t> dist(w.begin(), w.end());
    return ids[dist(rng_)];
}

}  // namespace onebit
