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
// and two tools, experts() and delegate(expert, task, context). A delegate call becomes an
// OpenAI chat request to Lemonade for the expert's catalog id (Lemonade loads the expert;
// it allows two LLMs at once, so one expert runs beside the PM); the expert's answer comes
// back as the tool result, and the PM composes the final answer, which is streamed when the
// client asked for a stream. Every other route forwards unchanged. When Lemonade or the
// expert fails, the tool result says so and the PM answers alone, never silently.
#pragma once

#include <httplib.h>
#include <nlohmann/json.hpp>

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

struct Config {
    std::vector<Expert> experts;
    int max_rounds = 4;                 // PM rounds per request, tool rounds included
    size_t max_result_bytes = 32768;    // an expert's answer is cut here (UTF-8 safe) before the PM sees it
    int delegate_timeout_s = 1800;      // read timeout for one expert answer
    bool thinking = false;              // chat_template_kwargs.enable_thinking for PM rounds (the client's wins)
    std::string system_prompt;          // "" : the built-in prompt (pm.cpp)
    std::string lemonade_url;           // http://host:port of the Lemonade that runs the experts
    std::string chat_path = "/api/v1/chat/completions";

    const Expert* find(const std::string& key) const;
};

// config/pm-experts.json as built into this binary.
const char* default_experts_json();

// Parses an experts file (the built-in one when `file` is empty); throws std::runtime_error
// with the reason on a bad file.
Config load_config(const std::string& file, const std::string& lemonade_url);

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
