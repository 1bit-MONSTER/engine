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

// app/pm.h -- `1bit serve --pm`: the served model is the Project Manager (docs/pm.md).
//
// The resident model (ZAYA1-8B) keeps every conversation. On POST /v1/chat/completions
// the engine runs a tool loop instead of a plain forward: the PM gets a system prompt
// and two tools, experts() and delegate(expert, task, context); with --pm-spawn-url it also gets
// spawn_worker(worker, box, class, task), which asks a guarded executor to start a long-running
// agent worker on its own local model (docs/pm.md). A delegate call becomes an
// OpenAI chat request to Lemonade for the expert's catalog id (Lemonade loads the expert;
// it allows two LLMs at once, so one expert runs beside the PM); the expert's answer comes
// back as the tool result, and the PM composes the final answer, which is streamed when the
// client asked for a stream. Every other route forwards unchanged. When Lemonade or the
// expert fails, the tool result says so and the PM answers alone, never silently.
#pragma once

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace onebit::pm {

struct Expert {
    std::string key;       // what the PM names in delegate(expert=...)
    std::string id;        // Lemonade catalog id (the `model` of the chat request)
    std::string domain;    // one line, shown to the PM
    std::string use_when;  // when to pick this expert, shown to the PM
    int max_tokens = 0;    // the expert's max_tokens (0: Lemonade's default)
};

// What a --pm-router laya decision says about one request (docs/pm.md, "Routing with Laya").
struct RouteHint {
    std::string label;        // the router's request class (Laya: code, prose, short, long_doc)
    float confidence = 0.0f;  // the router's calibrated confidence in that class
    long ms = 0;              // the router's decision time
};

struct Config {
    std::vector<Expert> experts;
    int max_rounds = 4;                 // PM rounds per request, tool rounds included
    size_t max_result_bytes = 32768;    // an expert's answer is cut here (UTF-8 safe) before the PM sees it
    int delegate_timeout_s = 1800;      // read timeout for one expert answer
    bool thinking = false;              // chat_template_kwargs.enable_thinking for PM rounds (the client's wins)
    std::string system_prompt;          // "" : the built-in prompt (pm.cpp)
    std::string lemonade_url;           // http://host:port of the Lemonade that runs the experts
    std::string spawn_url;              // --pm-spawn-url: the executor that starts agent workers ("" : no spawn_worker tool)
    int spawn_timeout_s = 180;          // read timeout for one spawn_worker request
    std::string chat_path = "/api/v1/chat/completions";
    // --pm-router laya (docs/pm.md): the router's class -> the expert for it, "" leaving the
    // answer to the PM. "model" (the default) is the PM's own choice, as before.
    std::string router = "model";
    std::map<std::string, std::string> routes;
    float routes_min_confidence = 0.16f;  // a class below this leaves the answer to the PM
    // The router, set by serve when --pm-router laya: state -> (label, confidence, ms), false when
    // it could not classify (the PM answers). Empty with the default router.
    std::function<bool(const std::string& state, RouteHint& out)> classify;

    const Expert* find(const std::string& key) const;
};

// config/pm-experts.json as built into this binary.
const char* default_experts_json();

// Parses an experts file (the built-in one when `file` is empty); throws std::runtime_error
// with the reason on a bad file.
Config load_config(const std::string& file, const std::string& lemonade_url);

// The --pm-router values: "model" (the default: the PM decides) and "laya" (the engine asks the
// Laya router first). An empty value is "model"; anything else throws std::runtime_error.
std::string parse_router(const std::string& value);

// The expert a routed class selects at `confidence`, or nullptr when the PM answers itself: no
// route for the class, an empty expert name, an expert the config does not list, or a confidence
// below routes_min_confidence.
const Expert* route_expert(const Config& cfg, const std::string& label, float confidence);

// One tool call the PM made, from llama-server's tool_calls array or from the text of its
// answer (ZAYA's <zyphra_tool_call><function=NAME><parameter=K>V</parameter></function> form,
// Qwen's <tool_call>{"name":..,"arguments":{..}}</tool_call> form, or a bare JSON object).
struct ToolCall {
    std::string id, name;
    nlohmann::json arguments;  // an object; a non-JSON arguments string becomes {"task": text}
};
std::vector<ToolCall> parse_tool_calls(const nlohmann::json& message, const std::vector<std::string>& tool_names);

// The PM handler for POST /v1/chat/completions. `backend_port` is the resident model's
// llama-server; our_id/drop_model/set_model are serve's usual model-name rules.
void handle_chat(const Config& cfg, int backend_port, const std::string& our_id, bool drop_model,
                 const std::string& set_model, const httplib::Request& req, httplib::Response& res);

}  // namespace onebit::pm
