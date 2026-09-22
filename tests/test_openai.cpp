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
#include "check.h"
#include "openai.h"

using namespace onebit;
using json = openai::json;

static void test_parse_chat() {
    auto r = openai::parse_chat(json::parse(R"({
        "model": "x", "stream": true, "stream_options": {"include_usage": true},
        "temperature": 0, "top_k": 5, "max_tokens": 12, "seed": 7, "stop": ["A", "B"],
        "messages": [{"role": "system", "content": "s"},
                     {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}],
        "chat_template_kwargs": {"enable_thinking": false}, "reasoning_format": "none",
        "unknown_field": 1})"));
    REQUIRE(r.has_value());
    CHECK(r->stream && r->include_usage);
    CHECK(r->gen.sampling.temperature == 0.0f);
    CHECK(r->gen.sampling.top_k == 5);
    CHECK(r->gen.sampling.seed == 7);
    CHECK(r->gen.max_tokens == 12);
    CHECK((r->gen.stops == std::vector<std::string>{"A", "B"}));
    CHECK(r->messages[1]["content"] == "a\nb");
    CHECK(r->template_kwargs["enable_thinking"] == false);
    CHECK(!r->split_reasoning);
    CHECK(r->tools.is_null());

    auto bad = [](const char* body) { return !openai::parse_chat(json::parse(body)).has_value(); };
    CHECK(bad(R"({"messages": []})"));
    CHECK(bad(R"({"messages": [{"content": "no role"}]})"));
    CHECK(bad(R"({"messages": [{"role": "user", "content": "x"}], "n": 2})"));
    CHECK(bad(R"({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]})"));
    CHECK(bad(R"({"messages": [{"role": "user", "content": "x"}], "temperature": "hot"})"));
    CHECK(bad(R"({"messages": [{"role": "user", "content": "x"}], "stop": 5})"));
    CHECK(bad(R"([1, 2])"));
    // max_completion_tokens wins over max_tokens.
    auto both = openai::parse_chat(json::parse(
        R"({"messages": [{"role": "user", "content": "x"}], "max_tokens": 5, "max_completion_tokens": 9})"));
    CHECK(both && both->gen.max_tokens == 9);
}

static void test_parse_completion() {
    auto r = openai::parse_completion(json::parse(R"({"prompt": "hi", "stop": "\n"})"));
    REQUIRE(r.has_value());
    CHECK(r->prompt == "hi" && !r->chat && !r->split_reasoning);
    CHECK((r->gen.stops == std::vector<std::string>{"\n"}));
    CHECK(openai::parse_completion(json::parse(R"({"prompt": ["one"]})")).has_value());
    CHECK(!openai::parse_completion(json::parse(R"({"prompt": ["a", "b"]})")).has_value());
    CHECK(!openai::parse_completion(json::parse(R"({"prompt": [1, 2, 3]})")).has_value());
    CHECK(!openai::parse_completion(json::parse(R"({})")).has_value());
}

// Structural checks on every response shape (nlohmann brace literals can turn
// an intended object into an array, so the shapes are checked, not eyeballed).
static void test_shapes() {
    GenerateResult g;
    g.content = "4";
    g.reasoning = "add";
    g.finish_reason = "stop";
    g.prompt_tokens = 10;
    g.cache_n = 3;
    g.predicted_n = 2;
    g.prompt_ms = 70.0;
    g.predicted_ms = 40.0;
    openai::Meta m{"chatcmpl-1", "model", 123};

    json c = openai::chat_response(m, g);
    CHECK(c.is_object() && c["object"] == "chat.completion");
    REQUIRE(c["choices"].is_array() && c["choices"].size() == 1 && c["choices"][0].is_object());
    CHECK(c["choices"][0]["message"].is_object());
    CHECK(c["choices"][0]["message"]["content"] == "4");
    CHECK(c["choices"][0]["message"]["reasoning_content"] == "add");
    CHECK(c["choices"][0]["finish_reason"] == "stop");
    CHECK(c["usage"].is_object() && c["usage"]["prompt_tokens"] == 10 && c["usage"]["completion_tokens"] == 2);
    CHECK(c["usage"]["total_tokens"] == 12);
    CHECK(c["usage"]["prompt_tokens_details"].is_object() && c["usage"]["prompt_tokens_details"]["cached_tokens"] == 3);
    const json& t = c["timings"];
    CHECK(t.is_object() && t["prompt_n"] == 7 && t["cache_n"] == 3 && t["predicted_n"] == 2);
    CHECK(t["prompt_per_second"].get<double>() == 100.0);
    CHECK(t["predicted_per_second"].get<double>() == 50.0);

    json ch = openai::chat_chunk(m, json{{"content", "x"}}, nullptr);
    REQUIRE(ch["choices"].is_array() && ch["choices"][0].is_object());
    CHECK(ch["object"] == "chat.completion.chunk");
    CHECK(ch["choices"][0]["delta"].is_object() && ch["choices"][0]["delta"]["content"] == "x");
    CHECK(ch["choices"][0]["finish_reason"].is_null());
    json role = openai::chat_chunk(m, json{{"role", "assistant"}, {"content", nullptr}}, nullptr);
    CHECK(role["choices"][0]["delta"].is_object() && role["choices"][0]["delta"]["role"] == "assistant");
    json empty = openai::chat_chunk(m, json::object(), "length");
    CHECK(empty["choices"][0]["delta"].is_object() && empty["choices"][0]["delta"].empty());

    json tc = openai::completion_response(m, g);
    REQUIRE(tc["choices"].is_array() && tc["choices"][0].is_object());
    CHECK(tc["object"] == "text_completion" && tc["choices"][0]["text"] == "4");
    json tch = openai::completion_chunk(m, "y", nullptr);
    CHECK(tch["choices"][0].is_object() && tch["choices"][0]["text"] == "y");

    json e = openai::error(400, "bad", "invalid_request_error");
    CHECK(e["error"].is_object() && e["error"]["code"] == 400 && e["error"]["message"] == "bad");
}

int main() {
    test_parse_chat();
    test_parse_completion();
    test_shapes();
    return test_result("test_openai");
}
