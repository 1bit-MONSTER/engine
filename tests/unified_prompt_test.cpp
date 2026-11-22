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

// engine#3193: the two NPU chat routes must send the model the same prompt, and this is the row that says so.
//
// They did not: `1bit unified` (the private route and the lane) hand-rolled ChatML while the per-op forward rendered the
// container's template. On the FLM Qwen3 packs the template opens with an EMPTY SYSTEM TURN -- five tokens -- and with
// the wrong prompt Qwen3-1.7B answered "the first five prime numbers" with "3, 5, 7, 11, 13" (5/5, greedy). 0.6B and 4B
// answered the wrong prompt "correctly", which is why an answer-level test would have missed it: this one compares BYTES.
//
// No model, no NPU, no HTTP: the fixture is the FLM template the repo already pins, and the renderer is the same one both
// routes use. Without the Jinja renderer (ONEBIT_JINJA=OFF) the claim cannot be checked, so it says 77 rather than skip.

#include "unified_prompt.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

namespace {

using onebit::ThinkPrompt;
using onebit::ThinkTemplate;

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <tests/data/templates/qwen3_flm.json>\n", argv[0]);
        return 2;
    }
    if (!onebit::npu::renderer_built()) {
        std::printf("skip  this build has no chat-template renderer (ONEBIT_JINJA=OFF)\n");
        return 77;
    }
    const nlohmann::ordered_json fixture = nlohmann::ordered_json::parse(read_file(argv[1]));
    onebit::npu::ChatTemplate tmpl;
    tmpl.text = fixture.at("template").get<std::string>();
    tmpl.source = "tokenizer_config.json";
    tmpl.sha256 = onebit::sha256_hex(reinterpret_cast<const uint8_t*>(tmpl.text.data()), tmpl.text.size());

    const std::string bos = fixture.at("provenance").value("bos_token", std::string());
    const std::string eos = fixture.at("provenance").value("eos_token", std::string());
    // The request that failed on 1.7B: the /no_think marker rides in the user text, not in chat_template_kwargs.
    nlohmann::ordered_json request = nlohmann::ordered_json::object();
    nlohmann::ordered_json messages = nlohmann::ordered_json::array();
    messages.push_back({{"role", "user"},
                        {"content", "List the first five prime numbers, comma-separated: /no_think"}});
    request["messages"] = messages;
    request["enable_thinking"] = true;

    const ThinkTemplate no_pack_template{};
    const std::string prompt = onebit::unified::build_chat_prompt(request, tmpl, bos, eos, /*route_opens_think=*/false,
                                                                  no_pack_template);
    // What the per-op forward renders for the same request: the same call the renderer makes, so the row is "the route
    // uses this renderer", not "the renderer equals itself".
    const std::string forward = onebit::npu::render_chat_template(
        tmpl, onebit::npu::messages_for_template(request.at("messages")), /*add_generation_prompt=*/true,
        onebit::npu::request_template_vars(request, true), bos, eos);
    check(prompt == forward, "engine#3193 the unified route's prompt is byte-identical to the per-op forward's");
    check(prompt.find("<|im_start|>system\n<|im_end|>\n") != std::string::npos,
          "engine#3193 …and it carries the template's empty system turn (the five tokens ChatML omitted)");
    // The fallback is still reachable for a container that declares no template, and it must NOT be what a container
    // with one gets: that class is the whole defect.
    const std::string fallback = onebit::unified::chatml_prompt(request.at("messages"), ThinkPrompt::none);
    check(fallback != prompt, "engine#3193 the ChatML fallback is not what a container with a template gets");
    check(fallback.find("<|im_start|>system") == std::string::npos, "engine#3193 …because ChatML emits no system turn");

    std::printf("unified_prompt: %s (%d failure(s))\n", failures ? "FAIL" : "ok", failures);
    return failures ? 1 : 0;
}
