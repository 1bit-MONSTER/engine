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
// generator.h: runs one generation at a time over a model and tokenizer.
//
// Keeps the previous request's tokens in the KV cache and reuses the longest
// shared prefix (llama-server's cache_n), which is what makes multi-turn chat
// cheap: only the new turn is evaluated.
#pragma once

#include <cstdint>
#include <expected>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "cpu_model.h"
#include "sampler.h"
#include "text_stream.h"
#include "tokenizer.h"

namespace onebit {

struct GenerateRequest {
    std::vector<int32_t> prompt;
    SamplingParams sampling;
    int32_t max_tokens = -1;  // < 0: until end of generation or a full context
    std::vector<std::string> stops;
    bool split_reasoning = false;
    bool reasoning_open = false;
};

struct GenerateResult {
    std::string content, reasoning;
    std::string finish_reason;  // "stop" or "length"
    int32_t prompt_tokens = 0;  // whole prompt, including the reused prefix
    int32_t cache_n = 0;        // prompt tokens reused from the previous request
    int32_t predicted_n = 0;
    double prompt_ms = 0.0, predicted_ms = 0.0;
};

// Called with each delta as it becomes sendable. Return false to cancel
// (client went away); the result then has finish_reason "stop".
using OnDelta = std::function<bool(const TextDelta&)>;

class Generator {
public:
    Generator(CpuModel model, Tokenizer tokenizer);

    const Tokenizer& tokenizer() const { return tok_; }
    uint32_t n_ctx() const { return model_.n_ctx(); }

    std::expected<GenerateResult, std::string> generate(const GenerateRequest& req, const OnDelta& on_delta);

private:
    bool is_end_of_generation(int32_t id) const { return eog_.contains(id); }

    std::mutex mu_;
    CpuModel model_;
    Tokenizer tok_;
    std::set<int32_t> eog_;
    std::vector<int32_t> cached_;  // tokens currently in the KV cache
};

}  // namespace onebit
