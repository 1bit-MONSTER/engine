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
//
// think_template.h -- what `chat_template_kwargs.enable_thinking` can mean for a pack, read off its own template
// (engine#2135).
//
// The unified NPU route builds ChatML itself instead of rendering the pack's template, and for a thinking-off request
// it appends an empty block, `<think>\n\n</think>\n\n`, which is what Qwen3's own template emits. That is right only for
// a template that HAS the switch. Two packs on this box do not:
//
//   Qwen3-4B-Thinking-2507  its generation prompt opens `<think>\n` unconditionally and `enable_thinking` does not occur.
//                           Sent the empty block it reasons anyway, in plain text, and closes with `</think>`; with the
//                           splitter off for a thinking-off request, all of it came back as `content`.
//   Qwen3-VL-4B-Instruct    neither `enable_thinking` nor `<think>` occurs. Sent the empty block it answered a turn
//                           that already held a block it has never seen: "I apologize, but I cannot provide that
//                           information ..." to "What is the capital of France?".
//
// So the prompt follows the template: a switch -> unchanged; no switch and the template opens the block -> thinking
// cannot be turned off, the request is served as thinking-on; no switch and no block -> the request adds nothing.
// Header-only and pure, so the rule is host-tested (tests/think_template_test.cpp).
#pragma once

#include <string>

namespace onebit {

// What a chat template says about thinking. `known` is false when no template was read (the rule then changes nothing).
struct ThinkTemplate {
    bool known = false;
    bool has_switch = false;      // the template reads `enable_thinking`
    bool mentions_think = false;  // `<think>` occurs anywhere in it
    bool opens_think = false;     // its generation prompt itself opens `<think>`
};

inline ThinkTemplate think_template_of(const std::string& t) {
    ThinkTemplate tt;
    if (t.empty()) return tt;
    tt.known = true;
    tt.has_switch = t.find("enable_thinking") != std::string::npos;
    tt.mentions_think = t.find("<think>") != std::string::npos;
    // The generation prompt is what the last `add_generation_prompt` block emits, up to its `endif`.
    const size_t g = t.rfind("add_generation_prompt");
    if (g != std::string::npos) {
        const size_t end = t.find("endif", g);
        const std::string block = t.substr(g, end == std::string::npos ? std::string::npos : end - g);
        // Opening means leaving the block open: Qwen3's own generation block also holds `<think>`, but only inside the
        // empty `<think>\n\n</think>` it emits for thinking off, which closes again.
        tt.opens_think = block.find("<think>") != std::string::npos && block.find("</think>") == std::string::npos;
    }
    return tt;
}

// What a chat request's prompt ends with after `<|im_start|>assistant\n`.
enum class ThinkPrompt { none, open, empty_block };

struct ThinkPlan {
    ThinkPrompt prompt = ThinkPrompt::none;
    bool start_inside = false;  // the splitter starts inside the block (the prompt opened it)
    bool split = false;         // reasoning_content is split out of the reply
};

// `want_thinking`: the request's enable_thinking (default true). `route_opens_think`: what the route says its prompt
// does when thinking is on. `reasoning_none`: the request asked for reasoning_format "none".
inline ThinkPlan think_plan(bool chat, bool want_thinking, bool reasoning_none, bool route_opens_think,
                            const ThinkTemplate& tt) {
    ThinkPlan p;
    if (!chat) return p;
    bool thinking = want_thinking;
    bool add_empty_block = !want_thinking;
    if (!want_thinking && tt.known && !tt.has_switch) {
        if (tt.opens_think) {
            thinking = true;             // the template cannot turn thinking off: serve what it would send
            add_empty_block = false;
        } else if (!tt.mentions_think) {
            add_empty_block = false;     // a template with no block at all: thinking off adds nothing
        }
    }
    if (add_empty_block) p.prompt = ThinkPrompt::empty_block;
    else if (thinking && route_opens_think) p.prompt = ThinkPrompt::open;
    p.start_inside = p.prompt == ThinkPrompt::open;
    // Split whenever the caller did not ask for raw text: a model that reasons regardless is still split, and a reply
    // with no block passes through the splitter unchanged (the way llama-server's reasoning_format "auto" works).
    p.split = !reasoning_none;
    return p;
}

}  // namespace onebit
