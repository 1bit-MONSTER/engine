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

// Token generation on the fast lane: one lane run per prompt token, then decode
// with the next token's runlist prepared while the current one executes.
#pragma once

#include "lane.h"

#include <functional>
#include <vector>

namespace onebit::npu {

struct GenerateOptions {
    int max_tokens = 256;
    // Divides positive (multiplies negative) logits of the distinct ids among the
    // last `penalty_window` of prompt + output. 1 is plain greedy. Greedy decoding
    // of a 0.6B model loops, so the default is on.
    float repetition_penalty = 1.1f;
    int penalty_window = 64;
    bool stop_at_eos = true;
    // Called with the decode step index before each choice (0 = the token the
    // prompt produces), e.g. to record logits.
    std::function<void(int step, Lane&)> on_logits;
    // Called with each generated token; return false to stop.
    std::function<bool(int token)> on_token;
};

struct GenerateResult {
    std::vector<int> tokens;
    double prefill_ms = 0, decode_ms = 0;  // decode_ms covers every generated token
    bool stopped_at_eos = false;
};

// The model's end-of-sequence ids: config.json's, plus <|endoftext|> for Qwen3,
// which ends answers with it although its config lists only <|im_end|>.
std::vector<int> eos_ids(const Model& model);

GenerateResult generate(Lane& lane, const Model& model, const std::vector<int>& prompt, const GenerateOptions& opt);

}  // namespace onebit::npu
