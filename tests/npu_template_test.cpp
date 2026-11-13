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

// app/npu_template.h: the container's own chat template, rendered (engine #963 option 2).
//
// The expectations in tests/data/templates/*.json are **not** this renderer's own output: each `expect`
// string was produced by Jinja2 3.1.6 (the implementation Hugging Face's `apply_chat_template` uses)
// rendering the same template with the same variables, so the row that says "we render what the model
// asked for" is a comparison against something that is not us.  Fixtures:
//
//   qwen3_flm.json  FastFlowLM/Qwen3-0.6B-NPU2's template (4449 B, a55ee1b1…'s sibling a136047ba140…),
//                   which is the ChatML family *plus* the `user_system_prompt` branch those packs add
//   zr1.json        Zyphra/ZR1-1.5B's template (2081 B, b6835114…), DeepSeek-style specials
//                   (`<｜User｜>`), i.e. the non-ChatML container #963 measured at 33 tokens against 15
//
// What each row is for: (1) the bytes match a real Jinja2 for the shapes a caller sends, (2) request
// bytes keep their `is_input` mark so the tokenizer will not turn a message into control tokens, (3) a
// template that cannot render is a TemplateError naming its sha256 rather than a fall back to ChatML,
// and (4) a container with no template is refused by this layer rather than rendered with something.
#include "npu_template.h"

#include "../npu/tokenizer.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

onebit::npu::ChatTemplate template_of(const nlohmann::ordered_json& fixture) {
    onebit::npu::ChatTemplate t;
    t.text = fixture.at("template").get<std::string>();
    t.source = "tokenizer_config.json";
    t.sha256 = onebit::sha256_hex(reinterpret_cast<const uint8_t*>(t.text.data()), t.text.size());
    t.wants_user_system_prompt = t.text.find("user_system_prompt") != std::string::npos;
    return t;
}

// The variables a request contributes, minus the ones the renderer owns (messages and the prompt flag).
nlohmann::ordered_json extra_of(const nlohmann::ordered_json& request) {
    nlohmann::ordered_json extra = nlohmann::ordered_json::object();
    for (const auto& [k, v] : request.items()) {
        if (k == "messages" || k == "add_generation_prompt") continue;
        extra[k] = v;
    }
    return extra;
}

void run_fixture(const std::string& path) {
    // ordered: the fixtures carry the request order a caller sends, and `tojson` renders it
    const nlohmann::ordered_json fixture = nlohmann::ordered_json::parse(read_file(path));
    const std::string name = path.substr(path.find_last_of('/') + 1);
    const onebit::npu::ChatTemplate tmpl = template_of(fixture);
    const std::string bos = fixture.at("provenance").value("bos_token", std::string());
    const std::string eos = fixture.at("provenance").value("eos_token", std::string());
    for (const auto& c : fixture.at("cases")) {
        const std::string label = name + " " + c.at("name").get<std::string>();
        const auto& request = c.at("request");
        try {
            const std::string rendered = onebit::npu::render_chat_template(
                tmpl, request.at("messages"), request.value("add_generation_prompt", false), extra_of(request), bos, eos);
            const std::string got = rendered;
            const std::string want = c.at("expect").get<std::string>();
            check(got == want, label + ": the bytes are Jinja2's (" + std::to_string(got.size()) + " B)");
            if (got != want) std::printf("     got:  %s\n     want: %s\n", got.c_str(), want.c_str());
        } catch (const std::exception& e) {
            check(false, label + ": rendered (" + e.what() + ")");
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--no-renderer") {
            // ONEBIT_JINJA=OFF: the claim this test makes is about a renderer that is not in this build.
            std::printf("skip  this build has no chat-template renderer (ONEBIT_JINJA=OFF)\n");
            return 77;
        }
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <tests/data/templates/FIXTURE.json>... [--tokenizer FILE]\n", argv[0]);
        return 2;
    }
    std::string tokenizer_path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--tokenizer" && i + 1 < argc) {
            tokenizer_path = argv[++i];
            continue;
        }
        run_fixture(a);
    }

    // The chat route's own variables (engine#1788): a request's `tools` reach the template, in the
    // caller's key order, and an empty list is "none offered".  The route used to pass only
    // chat_template_kwargs and enable_thinking, so a template's tools block never rendered.
    {
        onebit::npu::ChatTemplate t;
        t.text = "{% if tools %}TOOLS {{ tools | tojson }} {% endif %}{% if enable_thinking %}think {% endif %}"
                 "{{ k }}{{ messages[0].content }}";
        t.source = "test";
        const nlohmann::ordered_json body = nlohmann::ordered_json::parse(
            R"({"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"k":"K "},)"
            R"("tools":[{"type":"function","function":{"name":"get_weather","description":"d","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"string"}}}}}]})");
        try {
            const nlohmann::ordered_json vars = onebit::npu::request_template_vars(body, true);
            const std::string out = onebit::npu::render_chat_template(t, body.at("messages"), true, vars);
            check(out.find("get_weather") != std::string::npos, "request tools: the template sees the request's tools");
            const size_t z = out.find("zeta"), a = out.find("alpha");
            check(z != std::string::npos && a != std::string::npos && z < a, "request tools: in the caller's key order");
            check(out.find("think K hi") != std::string::npos, "request tools: kwargs and enable_thinking still arrive");
            nlohmann::ordered_json none = body;
            none["tools"] = nlohmann::ordered_json::array();
            const std::string plain = onebit::npu::render_chat_template(
                t, none.at("messages"), true, onebit::npu::request_template_vars(none, false));
            check(plain == "K hi", "request tools: an empty list renders no tools block");
        } catch (const std::exception& e) {
            check(false, std::string("request tools: rendered (") + e.what() + ")");
        }
    }

    // An earlier call in the conversation (engine#1788): the API's arguments string reaches the template
    // as the object, so a template that walks it renders; the route used to answer 400 on the second turn.
    {
        onebit::npu::ChatTemplate t;
        t.text = "{% for m in messages %}{% if m.tool_calls %}{% for c in m.tool_calls %}{{ c.function.name }}("
                 "{% for k, v in c.function.arguments|items %}{{ k }}={{ v }};{% endfor %}){% endfor %}{% endif %}{% endfor %}";
        t.source = "test";
        const nlohmann::ordered_json msgs = nlohmann::ordered_json::parse(
            R"([{"role":"user","content":"q"},{"role":"assistant","content":"","tool_calls":[{"id":"call_0","type":"function",)"
            R"("function":{"name":"get_weather","arguments":"{\"zeta\":\"Paris\",\"alpha\":2}"}}]},)"
            R"({"role":"tool","tool_call_id":"call_0","content":"17"}])");
        bool raw_threw = false;
        try {
            (void)onebit::npu::render_chat_template(t, msgs, true);
        } catch (const onebit::npu::TemplateError&) {
            raw_threw = true;
        }
        check(raw_threw, "call arguments: as the API sends them (a string) this template cannot walk them");
        try {
            const std::string out = onebit::npu::render_chat_template(t, onebit::npu::messages_for_template(msgs), true);
            check(out == "get_weather(zeta=Paris;alpha=2;)", "call arguments: the template walks the object, in the string's key order");
        } catch (const std::exception& e) {
            check(false, std::string("call arguments: rendered (") + e.what() + ")");
        }
        nlohmann::ordered_json odd = msgs;
        odd[1]["tool_calls"][0]["function"]["arguments"] = "not json";
        check(onebit::npu::messages_for_template(odd) == odd, "call arguments: a string that is not a JSON object is left as sent");
    }

    // engine#2324: which requests carry a PICTURE. The NPU route parses no pixels, so the route refuses by name instead of
    // rendering the part as text and answering; these arms are the predicate that decides, and both directions matter --
    // over-matching would break the tool-call requests above, under-matching would answer a picture as if it were text.
    {
        using onebit::npu::message_carries_an_image;
        const auto m = [](const char* text) { return nlohmann::ordered_json::parse(text); };
        check(!message_carries_an_image(m(R"([{"role":"user","content":"hello"}])")),
              "an image part: a plain string content is not a picture");
        check(!message_carries_an_image(m(R"([{"role":"user","content":[{"type":"text","text":"hello"}]}])")),
              "an image part: a text-only content ARRAY is not a picture");
        check(message_carries_an_image(
                  m(R"([{"role":"user","content":[{"type":"text","text":"what is this?"},{"type":"image_url","image_url":{"url":"data:image/ppm;base64,AAA"}}]}])")),
              "an image part: the OpenAI image_url part is a picture");
        check(message_carries_an_image(m(R"([{"role":"user","content":[{"type":"image","data":"AAA"}]}])")),
              "an image part: the bare image spelling several local servers accept is a picture too");
        check(message_carries_an_image(
                  m(R"([{"role":"system","content":"x"},{"role":"user","content":[{"type":"image_url","image_url":{"url":"http://h/i.png"}}]}])")),
              "an image part: it is found in a later message, not only the first");
        check(!message_carries_an_image(m(R"({"not":"an array"})")),
              "an image part: a body whose messages are not an array is not a picture");
        check(!message_carries_an_image(m(R"([{"role":"user","content":[{"no_type":1},{"type":2}]}])")),
              "an image part: parts without a string type are not pictures");
        check(!message_carries_an_image(m(R"([{"role":"assistant","tool_calls":[{"function":{"arguments":"{\"type\":\"image_url\"}"}}]}])")),
              "an image part: the string image_url inside tool arguments is not a picture (it is not a content part)");
    }

    // A container that declares no template is this layer's "no", not a ChatML render: the route decides
    // what to do about that (app/forward_serve.cpp: ChatML with anchors, else the #987 400).
    {
        onebit::npu::ChatTemplate empty;
        bool threw = false;
        try {
            nlohmann::ordered_json msgs = nlohmann::ordered_json::array();
            msgs.push_back({{"role", "user"}, {"content", "hi"}});
            (void)onebit::npu::render_chat_template(empty, msgs, true);
        } catch (const onebit::npu::TemplateError& e) {
            threw = std::string(e.what()).find("no chat template") != std::string::npos;
        }
        check(threw, "a container with no template is refused here, not rendered with another one");
    }
    // A template that does not parse is a refusal that names the sha256, never a fall back.
    {
        onebit::npu::ChatTemplate broken;
        broken.text = "{% for m in messages %}{{ m.content }";
        broken.source = "tokenizer_config.json";
        broken.sha256 = onebit::sha256_hex(reinterpret_cast<const uint8_t*>(broken.text.data()), broken.text.size());
        bool named = false;
        try {
            nlohmann::ordered_json msgs = nlohmann::ordered_json::array();
            msgs.push_back({{"role", "user"}, {"content", "hi"}});
            (void)onebit::npu::render_chat_template(broken, msgs, true);
        } catch (const onebit::npu::TemplateError& e) {
            named = std::string(e.what()).find(broken.sha256) != std::string::npos;
        }
        check(named, "a template that does not parse is a TemplateError naming its sha256");
    }

    std::printf("\n%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
