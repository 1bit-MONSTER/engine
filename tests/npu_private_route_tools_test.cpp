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

using onebit::npu::npu_constraint_refusal;
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
        const std::string why = npu_constraint_refusal(body, route, true);
        check(!why.empty(), "a request carrying tools is refused on a private route");
        check(mentions(why, "tools"), "the refusal names the field that caused it");
        check(mentions(why, route.c_str()), "and names the route the model is actually on");
        check(mentions(why, "PrivateRequest"), "and cites the struct the private route actually uses");
        check(mentions(why, "forward-serve"), "and names the way out that exists on this server");
    }
    {
        // the same request on the LAYER LANE: `1bit unified` serves lane models through the same handler
        // (app/unified.cpp, Engine::generate dispatches `if (lane)` first), so the sentence must not describe the
        // private route's struct, nor offer an env var `unified.cpp` never reads.
        json body = {{"messages", json::array()}, {"tools", tools}};
        const std::string why = npu_constraint_refusal(body, route, false);
        check(!why.empty(), "a request carrying tools is refused on the layer lane too");
        check(mentions(why, "GenerateOptions"), "and cites the lane's own generator, not the private route's session");
        check(!mentions(why, "PrivateRequest"), "and does not name a struct this request never touched");
        check(!mentions(why, "ONEBIT_NPU_DX_KERNELS"), "and does not offer an env var this server never reads");
        check(mentions(why, "forward-serve"), "and names the per-op forward as the way out");
    }
    {
        json body = {{"messages", json::array()}, {"tool_choice", "required"}};
        const std::string why = npu_constraint_refusal(body, route, true);
        check(!why.empty(), "tool_choice without tools is refused too (still a constraint that cannot be honoured)");
        check(mentions(why, "tool_choice"), "and the message names tool_choice, not tools");
    }
    {
        json body = {{"messages", json::array()}, {"tools", nullptr}};
        check(npu_constraint_refusal(body, route, true).empty(),
              "tools: null is not a request for a tool call, so it is not refused");
    }
    {
        // a client that always sends the key with an empty list must not lose every chat to the refusal
        json body = {{"messages", json::array()}, {"tools", json::array()}};
        check(npu_constraint_refusal(body, route, true).empty(),
              "tools: [] carries no constraint, so it is not refused (clients send it on every request)");
    }
    {
        json body = {{"messages", json::array()}, {"tool_choice", "none"}};
        check(npu_constraint_refusal(body, route, true).empty(), "tool_choice: none asks for no call, so it is not refused");
    }
    {
        json body = {{"messages", json::array()},
                     {"response_format", {{"type", "json_schema"},
                                          {"json_schema", {{"name", "w"}, {"schema", {{"type", "object"}}}}}}}};
        const std::string why = npu_constraint_refusal(body, route, true);
        check(!why.empty(), "a JSON SCHEMA from the caller is refused too (a private route installs no grammar)");
        check(mentions(why, "json_schema"), "and the message names response_format json_schema");
    }
    {
        json body = {{"messages", json::array()}, {"response_format", {{"type", "json_object"}}}};
        check(npu_constraint_refusal(body, route, true).empty(),
              "json_object is advisory, not a contract, so it is not refused");
    }
    {
        // engine#3254 review: `value("type", "")` tolerated a MISSING key only, so a present non-string threw
        // `json.exception.type_error.302` out of the handler (past the try that answers 400) and httplib turned it
        // into a bare 500. An unreadable `type` is not a schema request, and it must not throw.
        json five = {{"messages", json::array()}, {"response_format", {{"type", 5}}}};
        json none = {{"messages", json::array()}, {"response_format", {{"type", nullptr}}}};
        bool threw = false;
        try {
            threw = !npu_constraint_refusal(five, route, true).empty() ||
                    !npu_constraint_refusal(none, route, true).empty();
        } catch (const std::exception&) {
            threw = true;
        }
        check(!threw, "a non-string response_format.type is neither a refusal nor a throw (it used to be a 500)");
    }
    {
        json body = {{"messages", json::array({{{"role", "user"}, {"content", "hello"}}})}};
        check(npu_constraint_refusal(body, route, true).empty(), "an ordinary chat request is not refused");
    }

    std::printf("npu_private_route_tools: %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
