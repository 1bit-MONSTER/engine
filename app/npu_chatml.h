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

// The Qwen3-lane chat prompt of `forward-serve` (issue #832).  It used to live in forward_serve.cpp with
// `(void)thinking;`, on the theory that the model-generic route serves architectures whose tokenizers do
// not know `<think>` -- but that file parses and masks every answer as `AnswerFamily::qwen3`, so the field
// was accepted and dropped.  The visible cost was not the answer text: `content_mask()` (issue #651 item 3)
// gives a reasoning delta no logprob entry, so a Qwen3 reply that opened inside `<think>` reported
// `logprobs.content: []` on every request, indistinguishable from "the model decided to think".
//
// `app/unified.cpp`'s lane has always honoured it, and this emits the same bytes.
#pragma once

#include "tokenizer.h"

#include <nlohmann/json.hpp>

#include <string>

namespace onebit::npu {

// What Qwen3's own chat template emits after the assistant prefix when thinking is off: an empty think
// block, so the model answers directly instead of opening one.  Same literal as app/unified.cpp:112-117.
inline constexpr const char* kEmptyThink = "<think>\n\n</think>\n\n";

// `chat_template_kwargs.enable_thinking`, default true -- a client that never sent the field gets the
// prompt it always got.  A non-boolean throws, and the caller answers 400 rather than guessing.
inline bool thinking_from_body(const nlohmann::json& body) {
    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object())
        return body["chat_template_kwargs"].value("enable_thinking", true);
    return true;
}

inline std::string chatml(const nlohmann::json& messages, bool thinking = true) {
    std::string p;
    for (const auto& m : messages) {
        std::string content;
        const auto& c = m.at("content");
        if (c.is_string()) {
            content = c;
        } else if (c.is_array()) {
            for (const auto& part : c)
                if (part.value("type", "") == "text") content += part.value("text", "");
        }
        p += "<|im_start|>" + m.at("role").get<std::string>() + "\n" + content + "<|im_end|>\n";
    }
    p += "<|im_start|>assistant\n";
    if (!thinking) p += kEmptyThink;
    return p;
}

// Whether the container's tokenizer can represent the prompt above.  `chatml()` is rendered for
// every container (#832), and a tokenizer without the anchors spells `<|im_start|>` as ordinary
// pieces, so the model is prompted with text its own template never produces (#963: on
// Zyphra/ZR1-1.5B, whose specials are DeepSeek's, 33 tokens rendered against 15 under the model's
// template).  The route answers 400 in that case instead of answering plausibly from the wrong
// prompt.
//
// A necessary condition read off the container itself, not a proof that the template *is* ChatML: a
// container may carry both anchors and still use a different template.  Packing the model's own
// `chat_template` is the real fix and a format decision, not this check.
inline bool chatml_supported(const Tokenizer& tok) {
    return tok.token_id("<|im_start|>") >= 0 && tok.token_id("<|im_end|>") >= 0;
}

}  // namespace onebit::npu
