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

// engine#3193: a private NPU route cannot honour `tools`, and used to answer anyway.
//
// `npu/private_route.h`'s `PrivateRequest` carries {max_tokens, repetition_penalty, penalty_window, on_token} and no
// constraint hook, so a route's session generates its own tokens with no grammar. A request that carries `tools`
// therefore reached the session, had its grammar dropped, and came back as prose -- where the per-op forward on the
// same weights returns a proper `tool_calls`. Measured on the dx route: `tool_calls: NONE` even with
// `tool_choice: "required"`; the identical request on the per-op route returns get_weather{"city":"Vienna"}.
//
// The refusal is a pure function of the request body, so this is a decision table with no model, no NPU and no HTTP:
//
//   tools present            refuse, and say the route cannot carry a grammar
//   tool_choice present      refuse too (a choice with no tools is still a constraint that cannot be honoured)
//   tools: null              do NOT refuse: a client that sends null is not asking for a tool call
//   neither                  do NOT refuse: the ordinary chat path
//
// and every refusal must name the way out, because a 400 that does not is just a different dead end.

#include "npu_tool_grammar.h"

#include <cstdio>
#include <string>

using onebit::npu::private_route_tools_refusal;
using json = nlohmann::json;

namespace {

int bad = 0;

void check(bool cond, const char* what) {
    std::printf("%s %s\n", cond ? "ok  " : "FAIL", what);
    if (!cond) ++bad;
}

bool mentions(const std::string& s, const char* needle) { return s.find(needle) != std::string::npos; }

}  // namespace

int main() {
    const std::string route = "Qwen3-0.6B-NPU-1bit";
    const json tools = json::parse(R"([{"type":"function","function":{"name":"get_weather",
        "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}])");

    {
        json body = {{"messages", json::array({{{"role", "user"}, {"content", "weather in Vienna?"}}})},
                     {"tools", tools}};
        const std::string why = private_route_tools_refusal(body, route);
        check(!why.empty(), "a request carrying tools is refused on a private route");
        check(mentions(why, "tools"), "the refusal names the field that caused it");
        check(mentions(why, route.c_str()), "and names the route the model is actually on");
        check(mentions(why, "ONEBIT_NPU_DX_KERNELS"), "and names the way out (serve it through the per-op forward)");
    }
    {
        json body = {{"messages", json::array()}, {"tool_choice", "required"}};
        const std::string why = private_route_tools_refusal(body, route);
        check(!why.empty(), "tool_choice without tools is refused too (still a constraint that cannot be honoured)");
        check(mentions(why, "tool_choice"), "and the message names tool_choice, not tools");
    }
    {
        json body = {{"messages", json::array()}, {"tools", nullptr}};
        check(private_route_tools_refusal(body, route).empty(),
              "tools: null is not a request for a tool call, so it is not refused");
    }
    {
        // a client that always sends the key with an empty list must not lose every chat to the refusal
        json body = {{"messages", json::array()}, {"tools", json::array()}};
        check(private_route_tools_refusal(body, route).empty(),
              "tools: [] carries no constraint, so it is not refused (clients send it on every request)");
    }
    {
        json body = {{"messages", json::array()}, {"tool_choice", "none"}};
        check(private_route_tools_refusal(body, route).empty(), "tool_choice: none asks for no call, so it is not refused");
    }
    {
        json body = {{"messages", json::array()},
                     {"response_format", {{"type", "json_schema"},
                                          {"json_schema", {{"name", "w"}, {"schema", {{"type", "object"}}}}}}}};
        const std::string why = private_route_tools_refusal(body, route);
        check(!why.empty(), "a JSON SCHEMA from the caller is refused too (a private route installs no grammar)");
        check(mentions(why, "json_schema"), "and the message names response_format json_schema");
    }
    {
        json body = {{"messages", json::array()}, {"response_format", {{"type", "json_object"}}}};
        check(private_route_tools_refusal(body, route).empty(),
              "json_object is advisory, not a contract, so it is not refused");
    }
    {
        json body = {{"messages", json::array({{{"role", "user"}, {"content", "hello"}}})}};
        check(private_route_tools_refusal(body, route).empty(), "an ordinary chat request is not refused");
    }

    std::printf("npu_private_route_tools: %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
