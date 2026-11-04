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

// npu_chatml_test: the Qwen3-lane chat prompt of `forward-serve` (app/npu_chatml.h, issue #832) -- the
// request's `chat_template_kwargs.enable_thinking` and the bytes it produces -- plus the container check of
// #963: whether the model's tokenizer can represent that prompt at all. No model, no device, no network;
// the two tokenizers are fixtures under tests/data.
//
// What this can and cannot pin: the prompt is the only part of "turn thinking off" the route controls.
// Whether the model then answers outside `<think>` -- and so whether `logprobs.content` comes back
// non-empty -- is a property of the weights and is measured on a serve, not here. So the assertions below
// are the bytes (and that thinking-off is the same suffix the d3 route's template already emits), and the
// PR body carries the served row.
#include "npu_chatml.h"
#include "npu_systemone.h"  // the cross-check: the d3 route's template ends with the same suffix

#include <cstdio>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

using nlohmann::json;
using onebit::npu::chatml;
using onebit::npu::chatml_supported;
using onebit::npu::kEmptyThink;
using onebit::npu::thinking_from_body;

static int failures = 0;

static void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL %s\n", what.c_str());
        ++failures;
    }
}

static bool ends_with(const std::string& s, const std::string& tail) {
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: npu_chatml_test <tokenizer.json with ChatML anchors> <tokenizer.json without>\n");
        return 2;
    }
    const json msgs = json::array({{{"role", "user"}, {"content", "hi"}}});
    const std::string assistant = "<|im_start|>assistant\n";

    // Thinking on is byte-for-byte what the route always sent: no think block of its own, because Qwen3's
    // template leaves it to the model.
    const std::string on = chatml(msgs, true);
    check(ends_with(on, assistant), "thinking on: ends at the bare assistant prefix");
    check(on.find("<think>") == std::string::npos, "thinking on: the template adds no think block");
    check(chatml(msgs) == on, "thinking on: the default is thinking on");

    // Thinking off appends exactly the empty block, and nothing else moves.
    const std::string off = chatml(msgs, false);
    check(ends_with(off, assistant + kEmptyThink), "thinking off: ends with the empty think block");
    check(off == on + kEmptyThink, "thinking off: only the think block is added");
    check(std::string(kEmptyThink) == "<think>\n\n</think>\n\n", "the empty block is Qwen3's own literal");

    // The d3 route's template (app/npu_systemone.h) already emits that same suffix; a change to either must
    // make this row fail rather than drift silently.
    const std::string systemone = onebit::npu::systemone_chat("hi");
    check(ends_with(systemone, "<|im_end|>\n" + assistant + kEmptyThink),
          "the d3 template ends with the same assistant suffix");
    check(ends_with(systemone, off.substr(off.find("<|im_start|>user\n"))),
          "thinking off reproduces the d3 template from its user turn on");

    // The rendering the header took over from forward_serve.cpp: roles, turns and array content.
    const json turns = json::array({{{"role", "system"}, {"content", "be brief"}},
                                    {{"role", "user"},
                                     {"content", json::array({{{"type", "text"}, {"text", "one"}},
                                                              {{"type", "image_url"}, {"image_url", "x"}},
                                                              {{"type", "text"}, {"text", " two"}}})}}});
    check(chatml(turns, false) == "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\none two<|im_end|>\n" +
                                      assistant + kEmptyThink,
          "multi-turn with array content: text parts concatenated, other parts skipped");

    // The request field itself.
    check(thinking_from_body(json::object()) == true, "no chat_template_kwargs: thinking on");
    check(thinking_from_body(json{{"chat_template_kwargs", json::object()}}) == true,
          "empty chat_template_kwargs: thinking on");
    check(thinking_from_body(json{{"chat_template_kwargs", {{"enable_thinking", true}}}}) == true,
          "enable_thinking true");
    check(thinking_from_body(json{{"chat_template_kwargs", {{"enable_thinking", false}}}}) == false,
          "enable_thinking false");
    check(thinking_from_body(json{{"chat_template_kwargs", "off"}}) == true,
          "a non-object chat_template_kwargs leaves thinking on, as the route always did");

    // A non-boolean is refused by nlohmann's own throw, which the route turns into a 400 -- never a guess.
    bool threw = false;
    try {
        (void)thinking_from_body(json{{"chat_template_kwargs", {{"enable_thinking", 1}}}});
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "a non-boolean enable_thinking throws (the caller answers 400)");

    // #963: is this container's tokenizer able to represent the prompt at all?  The anchors have to be
    // vocabulary entries -- an added token stays whole, while text the pre-tokenizer merely splits is what
    // made the render 23 tokens where Qwen3's own tokenizer spends 9 on the same request, and 33 against 15
    // under the model's own template on Zyphra/ZR1-1.5B.
    const onebit::npu::Tokenizer with_anchors(argv[1]);
    const onebit::npu::Tokenizer without(argv[2]);
    check(chatml_supported(with_anchors), "a tokenizer with <|im_start|>/<|im_end|> is served the chat route");
    check(!chatml_supported(without), "a tokenizer without them is refused, not prompted anyway (#963)");
    check(with_anchors.encode("<|im_start|>").size() == 1 && with_anchors.token_id("<|im_start|>") >= 0,
          "with the anchor the marker is one token, which is what ChatML requires");
    check(with_anchors.encode(chatml(msgs, true))[0] == with_anchors.token_id("<|im_start|>"),
          "the rendered prompt opens with that anchor token");
    check(without.token_id("<|im_start|>") < 0 && without.token_id("<|im_end|>") < 0,
          "without it there is no such id: the refusal's condition is the anchor, not the prompt's length");
    check(!without.encode(chatml(msgs, true)).empty(),
          "the refusal is not about whether the prompt tokenizes at all -- it does, into the wrong ids");

    if (failures == 0) std::printf("npu_chatml: ok\n");
    return failures == 0 ? 0 : 1;
}
