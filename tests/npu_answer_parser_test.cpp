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
// Copyright 2026 The 1bit-MONSTER Authors.
// SPDX-License-Identifier: Apache-2.0
//
// npu_answer_parser_test -- engine#1779 fault 4: "Gemma 4 tool-call parsing".
//
// The issue's test: "a template's own tool-call emission parsed into a structured call: name, arguments, and a
// non-tool reply that must not be parsed as one". The parser is header-only (app/npu_answer_parser.h), so this is a
// pure unit test: no server, no model, no device, and it runs on any box.
//
// The arms below are the contract that header states rather than just the happy path:
//   * a `<tool_call>{"name":…,"arguments":{…}}</tool_call>` block becomes a ToolCall with the arguments re-dumped as
//     JSON text (footer note at :49 says `function.arguments` is a string in OpenAI's shape);
//   * a call-only reply sets content_null, an answer with prose does not (:279-281);
//   * WITHOUT tools declared the marker is text and stays in content, marker and all (:193 "Without tools the tag is
//     just text: keep it in content, marker and all") - that is the strongest form of "a non-tool reply must not be
//     parsed as one";
//   * a malformed call body is not dropped: it goes back into content with its markup (:216-221), because what the
//     model wrote must never be lost.

#include <cstdio>
#include <string>

#include "npu_answer_parser.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) failures++;
}

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

}  // namespace

int main() {
    using namespace onebit::npu;

    // 1. A template's own emission, with reasoning and prose around it.
    {
        const auto a = parse_answer(
            "<think>I should check the weather first.</think>Let me look.<tool_call>{\"name\":\"get_weather\","
            "\"arguments\":{\"city\":\"Paris\"}}</tool_call>",
            AnswerFamily::qwen3, true);
        check(a.calls.size() == 1, "a tool_call block becomes exactly one call");
        if (a.calls.size() == 1) {
            check(a.calls[0].name == "get_weather", "the call keeps its name");
            check(contains(a.calls[0].arguments, "Paris"), "the call keeps its arguments as JSON text");
            check(a.calls[0].arguments.size() > 2 && a.calls[0].arguments.front() == '{',
                  "arguments is a JSON object string, not a quoted blob");
            check(!a.calls[0].id.empty(), "the call carries an id");
        }
        check(a.reasoning == "I should check the weather first.", "<think> goes to reasoning");
        check(a.content == "Let me look.", "prose outside the markup goes to content");
        check(!a.content_null, "a reply with prose does not report null content");
    }

    // 2. Call-only: content: null, not "" (the issue's second rule).
    {
        const auto a = parse_answer("<tool_call>{\"name\":\"ping\",\"arguments\":{}}</tool_call>", AnswerFamily::qwen3, true);
        check(a.calls.size() == 1, "a call-only reply still yields the call");
        check(a.content.empty(), "a call-only reply has empty content");
        check(a.content_null, "a call-only reply reports content_null");
    }

    // 3. A non-tool reply, tools declared: prose must not become a call.
    {
        const auto a = parse_answer("The weather in Paris is mild and dry today.", AnswerFamily::qwen3, true);
        check(a.calls.empty(), "a plain answer produces no calls");
        check(a.content == "The weather in Paris is mild and dry today.", "a plain answer is all content");
    }

    // 4. The marker with NO tools declared is text, marker and all - the strongest "not parsed as one" case.
    {
        const std::string text = "Use <tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call> to emit a call.";
        const auto        a    = parse_answer(text, AnswerFamily::qwen3, false);
        check(a.calls.empty(), "without tools declared, tool_call markup is not parsed into a call");
        check(contains(a.content, "<tool_call>"), "without tools, the marker stays visible in content");
        check(contains(a.content, "to emit a call."), "without tools, the rest of the text is kept");
    }

    // 5. Malformed call body: never dropped, goes back into content with its markup.
    {
        const auto a = parse_answer("x<tool_call>not json at all</tool_call>", AnswerFamily::qwen3, true);
        check(a.calls.empty(), "a malformed call body produces no call");
        check(contains(a.content, "<tool_call>") && contains(a.content, "not json at all"),
              "a malformed call body is kept in content, markup and all (nothing is lost)");
    }

    // 6. Two calls in one reply, in order.
    {
        const auto a = parse_answer(
            "<tool_call>{\"name\":\"first\",\"arguments\":{\"n\":1}}</tool_call>"
            "<tool_call>{\"name\":\"second\",\"arguments\":{\"n\":2}}</tool_call>",
            AnswerFamily::qwen3, true);
        check(a.calls.size() == 2, "two blocks become two calls");
        if (a.calls.size() == 2) {
            check(a.calls[0].name == "first" && a.calls[1].name == "second", "calls keep their order");
            check(a.calls[0].id != a.calls[1].id, "each call gets its own id");
        }
        check(a.content_null, "a two-call reply with no prose reports content_null");
    }

    // 7. The family tag set is complete for the one family the header declares, so an unsupported family cannot be
    //    silently parsed as qwen3.
    {
        const auto tags = onebit::npu::answer_detail::tags_of(AnswerFamily::qwen3);
        check(std::string(tags.call_open) == "<tool_call>" && std::string(tags.call_close) == "</tool_call>",
              "the qwen3 family's call tags are the ones this test feeds");
        check(std::string(tags.think_open) == "<think>" && std::string(tags.think_close) == "</think>",
              "the qwen3 family's reasoning tags are the ones this test feeds");
    }

    // A reasoning block the PROMPT opened (engine#1786, FastFlowLM's Qwen3.5-2B-NPU2): the template's generation prompt
    // ends with `<think>`, so the answer starts inside the block and the only tag in it is the closing one.
    {
        const AnswerFamily Q = AnswerFamily::qwen3;
        const std::string on = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n";
        const std::string off = "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        const std::string plain = "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n";
        check(prompt_opens_reasoning(on, Q), "a prompt that ends with <think> leaves the model inside the block");
        check(!prompt_opens_reasoning(off, Q), "one that opens AND closes it (thinking off) does not");
        check(!prompt_opens_reasoning(plain, Q) && !prompt_opens_reasoning("", Q) && !prompt_opens_reasoning("\n \n", Q),
              "nor does one that never mentions it, or an empty one");
        check(!prompt_opens_reasoning("tell me what <think> means<|im_end|>\n<|im_start|>assistant\n", Q),
              "the tag earlier in the prompt is not the prompt's end");
        check(!prompt_opens_reasoning(on, AnswerFamily::gemma4), "the tag is the family's own: <think> opens nothing for gemma4");

        const std::string reply = "Thinking Process:\n1. The capital of France is Paris.\n</think>\n\nParis";
        const auto before = parse_answer(reply, Q, false);   // what the route did: the fault, as an assertion
        check(before.reasoning.empty() && contains(before.content, "Thinking Process") && !contains(before.content, "</think>") &&
                  contains(before.content, "Paris"),
              "without the flag the reasoning is filed under content, with the closing tag dropped, so nothing marks where the answer starts (the measured fault)");
        const auto a = parse_answer(reply, Q, false, /*starts_in_reasoning=*/true);
        check(a.reasoning == "Thinking Process:\n1. The capital of France is Paris.\n", "with it the reasoning is the reasoning");
        check(a.content == "Paris" && !contains(a.content, "think"), "and the content is the answer, with no tag and no layout newlines in it");

        // an answer cut off inside the block: all reasoning, nothing invented as content
        const auto cut = parse_answer("Thinking Process:\n1. Analyze", Q, false, true);
        check(cut.reasoning == "Thinking Process:\n1. Analyze" && cut.content.empty(), "an answer cut inside the block is all reasoning");
        // a call after the prompt-opened block still parses
        const auto call = parse_answer("need the tool</think><tool_call>{\"name\":\"ping\",\"arguments\":{}}</tool_call>", Q, true, true);
        check(call.reasoning == "need the tool" && call.calls.size() == 1 && call.calls[0].name == "ping" && call.content_null,
              "a tool call after the prompt-opened block is still a call");
        // the model re-opening the block itself is harmless
        const auto twice = parse_answer("<think>again</think>ok", Q, false, true);
        check(twice.content == "ok" && contains(twice.reasoning, "again"), "a model that writes the opening tag anyway still ends up split");

        // streamed, split at awkward places, the pieces add up to the same thing
        AnswerStream st(Q, false, true);
        std::vector<std::pair<bool, std::string>> out;
        for (const std::string& piece : {std::string("Thinking Pro"), std::string("cess:\n1. The capital of France is Paris.\n</th"),
                                         std::string("ink>\n\nPar"), std::string("is")})
            st.feed(piece, out);
        st.finish(out);
        check(st.reasoning() == a.reasoning && st.content() == a.content, "the streamed split equals the whole-answer split");
        check(!out.empty() && out.front().first, "and its first piece is reasoning, so a client sees it as such from the first token");

        // per-token mask: reasoning tokens get no logprob entry, content tokens do
        const std::vector<std::string> toks = {"Thinking", " Process", "</think>", "\n\n", "Paris"};
        const std::vector<char> m = content_mask(toks, Q, false, true);
        check(m == std::vector<char>({0, 0, 0, 0, 1}), "the content mask starts at the answer, after the closing tag and its layout");
        check(content_mask(toks, Q, false) == std::vector<char>({1, 1, 0, 1, 1}), "where without the flag the reasoning tokens counted as content");
    }

    // Whitespace around a reasoning block is layout (the rule app/think_split.h applies on the in-process route).
    {
        const AnswerFamily Q = AnswerFamily::qwen3;
        const auto a = parse_answer("<think>\nthe capital is Paris\n</think>\n\nParis", Q, false);
        check(a.reasoning == "the capital is Paris\n" && a.content == "Paris",
              "the newline after <think> and the blank line after </think> are not part of either text");
        const auto none = parse_answer("<think>\n\n</think>\n\nItem 3 is the 3D printer.", Q, false);
        check(none.reasoning.empty() && none.content == "Item 3 is the 3D printer.",
              "a model that thinks nothing sends no reasoning at all, and the answer starts at its first character");
        const auto plain = parse_answer("\n  indented answer", Q, false);
        check(plain.content == "\n  indented answer", "content with no reasoning before it is returned as written");
        const auto mid = parse_answer("First.<think>x</think> Second.", Q, false);
        check(mid.content == "First. Second.", "content that came before a block keeps the space that follows the block");
        const auto inner = parse_answer("<think>a\n\n  b</think>one\n\n two", Q, false);
        check(inner.reasoning == "a\n\n  b" && inner.content == "one\n\n two", "whitespace inside either text is kept");
        const auto ws = parse_answer("<think>x</think>\n\n", Q, false);
        check(ws.content.empty() && ws.reasoning == "x", "an answer that is only layout after the block is empty content");

        // streamed: the layout arrives in its own pieces and none of them is sent
        AnswerStream st(Q, false);
        std::vector<std::pair<bool, std::string>> out;
        for (const std::string& piece : {std::string("<think>"), std::string("\n"), std::string("\n"), std::string("</think>"),
                                         std::string("\n\n"), std::string("Item"), std::string(" 3")})
            st.feed(piece, out);
        st.finish(out);
        check(out.size() == 2 && !out[0].first && out[0].second == "Item" && out[1].second == " 3",
              "streamed, an empty block and its layout produce no chunk; the first chunk is the answer's first word");

        // the per-token mask agrees: layout tokens are not content tokens
        const std::vector<char> m = content_mask({"<think>", "\n", "why", "</think>", "\n\n", "Paris", "\n"}, Q, false);
        check(m == std::vector<char>({0, 0, 0, 0, 0, 1, 1}), "layout tokens after the block get no content entry; the answer's own do");

        // a tool call straight after the block is unaffected
        const auto call = parse_answer("<think>t</think>\n<tool_call>{\"name\":\"ping\",\"arguments\":{}}</tool_call>", Q, true);
        check(call.calls.size() == 1 && call.content_null, "a call after the block and a newline is a call with null content");
        // without tools the marker is text and is not eaten by the trim
        const auto text = parse_answer("<think>t</think>\n<tool_call>x", Q, false);
        check(text.content == "<tool_call>x", "the first text after the block is kept whole, whatever it starts with");
    }

    // engine#2111: usage.completion_tokens_details.reasoning_tokens -- how much of the budget went into reasoning.
    {
        const AnswerFamily Q = AnswerFamily::qwen3;
        // the issue's row: max_tokens 8, the template opened the block, and every token is still inside it
        const std::vector<std::string> eight = {"Okay", ",", " the", " user", " asks", " about", " the", " capital"};
        check(reasoning_token_count(eight, Q, false, true) == 8,
              "an unclosed block that took the whole budget: every token is reasoning (content \"\", 8 of 8)");
        const auto empty = parse_answer("Okay, the user asks about the capital", Q, false, true);
        check(empty.content.empty() && empty.reasoning == "Okay, the user asks about the capital",
              "... and that reply's content is empty while its reasoning carries the text");
        // the model opens the block itself: both tags count with it, the layout and the answer do not
        check(reasoning_token_count({"<think>", "\n", "why", "</think>", "\n\n", "Paris", "\n"}, Q, false) == 4,
              "a closed block: <think>, its text, </think> are reasoning; the layout after it and the answer are not");
        // a tag split across tokens counts with the block on both sides
        check(reasoning_token_count({"<th", "ink>", "plan", "</th", "ink>", "Paris"}, Q, false) == 5,
              "a tag split over two tokens is reasoning on both halves");
        // a token that straddles the boundary reached content, as content_mask says
        check(reasoning_token_count({"<think>", "plan", "</think>Paris"}, Q, false) == 2,
              "a token carrying the closing tag and the first answer text is content");
        // a block this splitter opens after the answer started: its opening tag counts with it
        check(reasoning_token_count({"Hi.", "<think>", "x", "</think>", "Bye"}, Q, false) == 3,
              "a block opened after content: <think>, x, </think> are reasoning; the prose around it is not");
        // no reasoning at all
        check(reasoning_token_count({"Par", "is", "."}, Q, false) == 0, "a reply with no block has 0 reasoning tokens");
        check(reasoning_token_count({}, Q, false, true) == 0, "no tokens, no reasoning tokens");
        // call markup is neither content nor reasoning, and is not counted as reasoning
        check(reasoning_token_count({"<think>", "t", "</think>", "<tool_call>", "{\"name\":\"ping\",\"arguments\":{}}",
                                     "</tool_call>"}, Q, true) == 3,
              "a tool call after the block is not reasoning");
        // the count agrees with the mask: a reasoning token never has a content entry
        const std::vector<std::string> mixed = {"<think>", "a", "b", "</think>", "\n", "X", "Y"};
        const std::vector<char> m = content_mask(mixed, Q, false);
        int content_entries = 0;
        for (char c : m) content_entries += c ? 1 : 0;
        check(reasoning_token_count(mixed, Q, false) + content_entries <= int(mixed.size()),
              "reasoning tokens and content entries never overlap");
    }

    std::printf("%s\n", failures ? "FAILED" : "all cases ok");
    return failures ? 1 : 0;
}
