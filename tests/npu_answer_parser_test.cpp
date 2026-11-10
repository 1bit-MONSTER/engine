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

    std::printf("%s\n", failures ? "FAILED" : "all cases ok");
    return failures ? 1 : 0;
}
