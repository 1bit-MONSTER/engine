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
#include "generator.h"

#include <algorithm>
#include <chrono>
#include <format>

namespace onebit {

namespace {

// Control tokens that end a turn in the chat formats of supported models.
// tokenizer.ggml.eos_token_id is always included as well.
constexpr const char* kEndOfTurn[] = {"<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>",
                                      "<|end|>",    "<end_of_turn>"};

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

Generator::Generator(CpuModel model, Tokenizer tokenizer) : model_(std::move(model)), tok_(std::move(tokenizer)) {
    if (tok_.eos_id() >= 0) eog_.insert(tok_.eos_id());
    for (int32_t id = 0; size_t(id) < tok_.n_vocab(); ++id) {
        if (!tok_.is_control(id)) continue;
        for (const char* t : kEndOfTurn)
            if (tok_.token_text(id) == t) eog_.insert(id);
    }
}

std::expected<GenerateResult, std::string> Generator::generate(const GenerateRequest& req, const OnDelta& on_delta) {
    std::lock_guard lock(mu_);
    if (req.prompt.empty()) return std::unexpected(std::string("prompt is empty"));
    if (req.prompt.size() >= model_.n_ctx())
        return std::unexpected(std::format("prompt is {} tokens; the context holds {}", req.prompt.size(), model_.n_ctx()));

    GenerateResult r;
    r.prompt_tokens = int32_t(req.prompt.size());

    // Reuse the shared prefix; the last prompt token is always re-evaluated
    // because its logits are needed to sample the first new token.
    size_t keep = 0;
    while (keep < cached_.size() && keep < req.prompt.size() && cached_[keep] == req.prompt[keep]) ++keep;
    keep = std::min(keep, req.prompt.size() - 1);
    model_.truncate(uint32_t(keep));
    cached_.resize(keep);
    r.cache_n = int32_t(keep);

    auto t0 = std::chrono::steady_clock::now();
    const std::vector<float>* logits = nullptr;
    for (size_t i = keep; i < req.prompt.size(); ++i) {
        auto out = model_.forward(req.prompt[i]);
        if (!out) {
            cached_.clear();
            model_.reset();
            return std::unexpected(out.error());
        }
        cached_.push_back(req.prompt[i]);
        logits = *out;
    }
    r.prompt_ms = ms_since(t0);

    Sampler sampler(req.sampling);
    TextStream stream(req.stops, req.split_reasoning, req.reasoning_open);
    auto t1 = std::chrono::steady_clock::now();
    bool cancelled = false;
    r.finish_reason = "length";
    for (;;) {
        if (req.max_tokens >= 0 && r.predicted_n >= req.max_tokens) break;
        const int32_t id = sampler.sample(*logits);
        if (is_end_of_generation(id)) {
            r.finish_reason = "stop";
            break;
        }
        ++r.predicted_n;
        const int32_t one[1] = {id};
        auto piece = tok_.decode(one, /*skip_special=*/true);
        if (!piece) return std::unexpected(piece.error());
        TextDelta d = stream.push(*piece);
        if (!d.empty() && on_delta && !on_delta(d)) {
            cancelled = true;
            break;
        }
        if (stream.stopped()) {
            r.finish_reason = "stop";
            break;
        }
        if (model_.n_past() >= model_.n_ctx()) break;  // context full
        auto out = model_.forward(id);
        if (!out) {
            cached_.clear();
            model_.reset();
            return std::unexpected(out.error());
        }
        cached_.push_back(id);
        logits = *out;
    }
    if (!cancelled) {
        TextDelta d = stream.finish();
        if (!d.empty() && on_delta) on_delta(d);
    } else {
        r.finish_reason = "stop";
    }
    r.predicted_ms = ms_since(t1);
    r.content = stream.content();
    r.reasoning = stream.reasoning();
    return r;
}

}  // namespace onebit
