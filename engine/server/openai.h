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
// openai.h: OpenAI / llama-server request parsing and response shapes.
//
// Responses carry both OpenAI's `usage` and llama-server's `timings`, which is
// what Lemonade reads for its telemetry.
#pragma once

#include <cstdint>
#include <expected>
#include <string>

#include <nlohmann/json.hpp>

#include "generator.h"

namespace onebit::openai {

using json = nlohmann::ordered_json;

struct Request {
    bool chat = true;
    bool stream = false;
    bool include_usage = false;  // stream_options.include_usage
    json messages;               // chat: text-only messages
    json tools;                  // chat: array or null
    json template_kwargs;        // chat: chat_template_kwargs
    bool split_reasoning = true; // chat: reasoning_format != "none"
    std::string prompt;          // completion
    GenerateRequest gen;         // prompt tokens are filled in by the caller
};

// Parse a /v1/chat/completions or /v1/completions body. Unknown fields are
// ignored; fields whose meaning would be silently dropped (n > 1, non-text
// content parts) are rejected.
std::expected<Request, std::string> parse_chat(const json& body);
std::expected<Request, std::string> parse_completion(const json& body);

struct Meta {
    std::string id;
    std::string model;
    int64_t created = 0;
};

json usage(const GenerateResult& r);
json timings(const GenerateResult& r);

json chat_response(const Meta& m, const GenerateResult& r);
json chat_chunk(const Meta& m, const json& delta, const json& finish_reason);
json completion_response(const Meta& m, const GenerateResult& r);
json completion_chunk(const Meta& m, const std::string& text, const json& finish_reason);

json error(int code, const std::string& message, const std::string& type);

}  // namespace onebit::openai
