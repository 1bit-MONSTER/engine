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
// sampler.h: next-token selection from logits.
//
// Order follows llama.cpp's default chain: top-k, top-p and min-p filter the
// candidates on the untempered distribution, then temperature is applied and
// one token is drawn. temperature <= 0 selects the argmax (greedy).
#pragma once

#include <cstdint>
#include <random>
#include <span>

namespace onebit {

struct SamplingParams {
    float temperature = 0.8f;
    int32_t top_k = 40;    // <= 0 disables
    float top_p = 0.95f;   // >= 1 disables
    float min_p = 0.05f;   // <= 0 disables
    uint64_t seed = 0;
};

class Sampler {
public:
    explicit Sampler(const SamplingParams& p) : p_(p), rng_(p.seed) {}

    int32_t sample(std::span<const float> logits);

private:
    SamplingParams p_;
    std::mt19937_64 rng_;
};

int32_t argmax(std::span<const float> logits);

}  // namespace onebit
