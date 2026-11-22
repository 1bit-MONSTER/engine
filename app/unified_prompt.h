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

// The prompt `1bit unified` builds for a chat request (engine#3193, #963).
//
// Extracted from handle_generate so a device-free test can assert the thing that matters: **this route and the per-op
// forward must send the model the same prompt.** They did not. `unified` hand-rolled ChatML; `forward_serve` renders the
// container's own template, and on the FLM Qwen3 packs the difference is five tokens -- the template opens with an empty
// system turn ("<|im_start|>system\n<|im_end|>\n") and ChatML does not. Qwen3-1.7B then answered "the first five prime
// numbers" with "3, 5, 7, 11, 13" (5/5, greedy, same weights, same tokenizer), while 0.6B and 4B answered the wrong
// prompt "correctly" -- which is why the omission survived until a 1.7B run.
//
// What stays here is only the decision: render the container's template when the build has a renderer for it, else the
// ChatML fallback the pack's own think template asks for.
#pragma once

#include <string>

#include <nlohmann/json.hpp>

#include "npu_template.h"
#include "npu_chatml.h"
#include "think_template.h"

namespace onebit::unified {

// The ChatML fallback, moved here unchanged from unified.cpp so the extraction is behaviour-preserving. It is the
// renderer #963 says must not be used when a container declares a template -- it stays for a container that declares
// none (the FLM packs declare theirs in tokenizer_config.json, which `load_chat_template` reads).
inline std::string chatml_prompt(const nlohmann::ordered_json& messages, ThinkPrompt think) {
    std::string p;
    for (const auto& m : messages) {
        std::string content;
        const auto& c = m.at("content");
        if (c.is_string()) {
            content = c.get<std::string>();
        } else if (c.is_array()) {  // [{type: text, text: ...}, ...]
            for (const auto& part : c)
                if (part.value("type", "") == "text") content += part.value("text", "");
        }
        p += "<|im_start|>" + m.at("role").get<std::string>() + "\n" + content + "<|im_end|>\n";
    }
    p += "<|im_start|>assistant\n";
    if (think == ThinkPrompt::empty_block) p += "<think>\n\n</think>\n\n";
    else if (think == ThinkPrompt::open) p += "<think>\n";
    return p;
}

// The prompt for one chat request.
//   ordered       the request body as parsed, in the caller's key order (a template's `tojson` reads it)
//   tmpl          the container's chat template (npu::load_chat_template)
//   bos_text/eos  the strings the template's {{ bos_token }}/{{ eos_token }} render
//   route_opens_think  whether the route opens a <think> block itself (Engine::open_think)
//   pack_template      what the pack's own template says about enable_thinking (Engine::think_tmpl)
//
// `empty_block` and `open` are the two shapes the ChatML fallback can add; the template path renders its own.
inline std::string build_chat_prompt(const nlohmann::ordered_json& ordered, const npu::ChatTemplate& tmpl,
                                     const std::string& bos_text, const std::string& eos_text,
                                     bool route_opens_think, const ThinkTemplate& pack_template) {
    bool thinking = true;
    if (ordered.contains("chat_template_kwargs") && ordered["chat_template_kwargs"].is_object())
        thinking = ordered["chat_template_kwargs"].value("enable_thinking", true);
    const ThinkPlan plan = think_plan(/*chat=*/true, thinking,
                                      ordered.value("reasoning_format", std::string("auto")) == "none",
                                      route_opens_think, pack_template);
    if (tmpl.present() && npu::renderer_built())
        return npu::render_chat_template(tmpl, npu::messages_for_template(ordered.at("messages")),
                                         /*add_generation_prompt=*/true,
                                         npu::request_template_vars(ordered, thinking), bos_text, eos_text);
    return chatml_prompt(ordered.at("messages"), plan.prompt);
}

}  // namespace onebit::unified
