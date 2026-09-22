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
#include "openai.h"

#include <format>
#include <random>

namespace onebit::openai {

namespace {

std::expected<void, std::string> parse_common(const json& b, Request& r) {
    if (!b.is_object()) return std::unexpected(std::string("request body must be a JSON object"));
    auto number = [&](const char* key, auto& field) -> std::expected<void, std::string> {
        if (!b.contains(key) || b[key].is_null()) return {};
        if (!b[key].is_number()) return std::unexpected(std::format("'{}' must be a number", key));
        field = b[key].get<std::remove_reference_t<decltype(field)>>();
        return {};
    };
    SamplingParams& s = r.gen.sampling;
    for (auto res : {number("temperature", s.temperature), number("top_p", s.top_p), number("min_p", s.min_p)})
        if (!res) return res;
    if (auto res = number("top_k", s.top_k); !res) return res;
    if (b.contains("seed") && b["seed"].is_number_integer() && b["seed"].get<int64_t>() >= 0) {
        s.seed = b["seed"].get<uint64_t>();
    } else {
        s.seed = std::random_device{}();
    }
    for (const char* key : {"max_completion_tokens", "max_tokens", "n_predict"}) {
        if (b.contains(key) && !b[key].is_null()) {
            if (!b[key].is_number_integer()) return std::unexpected(std::format("'{}' must be an integer", key));
            r.gen.max_tokens = b[key].get<int32_t>();
            break;
        }
    }
    if (b.contains("n") && !b["n"].is_null() && b["n"] != 1)
        return std::unexpected(std::string("only n = 1 is supported"));
    if (b.contains("stop") && !b["stop"].is_null()) {
        const json& st = b["stop"];
        if (st.is_string()) {
            r.gen.stops.push_back(st.get<std::string>());
        } else if (st.is_array()) {
            for (const auto& x : st) {
                if (!x.is_string()) return std::unexpected(std::string("'stop' entries must be strings"));
                r.gen.stops.push_back(x.get<std::string>());
            }
        } else {
            return std::unexpected(std::string("'stop' must be a string or an array of strings"));
        }
    }
    r.stream = b.value("stream", false);
    if (b.contains("stream_options") && b["stream_options"].is_object())
        r.include_usage = b["stream_options"].value("include_usage", false);
    return {};
}

// Content may be a string, null, or an array of parts; only text parts are
// accepted, joined with newlines.
std::expected<json, std::string> text_content(const json& c) {
    if (c.is_string() || c.is_null()) return c;
    if (!c.is_array()) return std::unexpected(std::string("message content must be a string or an array"));
    std::string out;
    for (const auto& part : c) {
        if (!part.is_object() || part.value("type", "") != "text" || !part.contains("text") || !part["text"].is_string())
            return std::unexpected(std::string("only text content parts are supported"));
        if (!out.empty()) out += "\n";
        out += part["text"].get<std::string>();
    }
    return out;
}

}  // namespace

std::expected<Request, std::string> parse_chat(const json& b) {
    Request r;
    r.chat = true;
    if (auto res = parse_common(b, r); !res) return std::unexpected(res.error());
    if (!b.contains("messages") || !b["messages"].is_array() || b["messages"].empty())
        return std::unexpected(std::string("'messages' must be a non-empty array"));
    r.messages = json::array();
    for (const auto& m : b["messages"]) {
        if (!m.is_object() || !m.contains("role") || !m["role"].is_string())
            return std::unexpected(std::string("each message needs a string 'role'"));
        json msg = m;
        if (m.contains("content")) {
            auto c = text_content(m["content"]);
            if (!c) return std::unexpected(c.error());
            msg["content"] = *c;
        }
        r.messages.push_back(std::move(msg));
    }
    r.tools = b.contains("tools") && b["tools"].is_array() && !b["tools"].empty() ? b["tools"] : json(nullptr);
    r.template_kwargs = b.contains("chat_template_kwargs") && b["chat_template_kwargs"].is_object()
                            ? b["chat_template_kwargs"]
                            : json::object();
    r.split_reasoning = b.value("reasoning_format", std::string("auto")) != "none";
    return r;
}

std::expected<Request, std::string> parse_completion(const json& b) {
    Request r;
    r.chat = false;
    r.split_reasoning = false;
    if (auto res = parse_common(b, r); !res) return std::unexpected(res.error());
    if (!b.contains("prompt")) return std::unexpected(std::string("'prompt' is required"));
    const json& p = b["prompt"];
    if (p.is_string()) {
        r.prompt = p.get<std::string>();
    } else if (p.is_array() && p.size() == 1 && p[0].is_string()) {
        r.prompt = p[0].get<std::string>();
    } else {
        return std::unexpected(std::string("'prompt' must be a string (batched prompts are not supported)"));
    }
    return r;
}

json usage(const GenerateResult& r) {
    return {{"prompt_tokens", r.prompt_tokens},
            {"completion_tokens", r.predicted_n},
            {"total_tokens", r.prompt_tokens + r.predicted_n},
            {"prompt_tokens_details", {{"cached_tokens", r.cache_n}}}};
}

json timings(const GenerateResult& r) {
    const int32_t evaluated = r.prompt_tokens - r.cache_n;
    auto per_second = [](int32_t n, double ms) { return ms > 0.0 ? 1000.0 * n / ms : 0.0; };
    return {{"cache_n", r.cache_n},
            {"prompt_n", evaluated},
            {"prompt_ms", r.prompt_ms},
            {"prompt_per_token_ms", evaluated ? r.prompt_ms / evaluated : 0.0},
            {"prompt_per_second", per_second(evaluated, r.prompt_ms)},
            {"predicted_n", r.predicted_n},
            {"predicted_ms", r.predicted_ms},
            {"predicted_per_token_ms", r.predicted_n ? r.predicted_ms / r.predicted_n : 0.0},
            {"predicted_per_second", per_second(r.predicted_n, r.predicted_ms)}};
}

json chat_response(const Meta& m, const GenerateResult& r) {
    json msg = {{"role", "assistant"}, {"content", r.content}};
    if (!r.reasoning.empty()) msg["reasoning_content"] = r.reasoning;
    return {{"id", m.id},
            {"object", "chat.completion"},
            {"created", m.created},
            {"model", m.model},
            {"choices", json::array({{{"index", 0}, {"message", msg}, {"finish_reason", r.finish_reason}}})},
            {"usage", usage(r)},
            {"timings", timings(r)}};
}

json chat_chunk(const Meta& m, const json& delta, const json& finish_reason) {
    return {{"id", m.id},
            {"object", "chat.completion.chunk"},
            {"created", m.created},
            {"model", m.model},
            {"choices", json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", finish_reason}}})}};
}

json completion_response(const Meta& m, const GenerateResult& r) {
    return {{"id", m.id},
            {"object", "text_completion"},
            {"created", m.created},
            {"model", m.model},
            {"choices", json::array({{{"index", 0}, {"text", r.content}, {"finish_reason", r.finish_reason}}})},
            {"usage", usage(r)},
            {"timings", timings(r)}};
}

json completion_chunk(const Meta& m, const std::string& text, const json& finish_reason) {
    return {{"id", m.id},
            {"object", "text_completion"},
            {"created", m.created},
            {"model", m.model},
            {"choices", json::array({{{"index", 0}, {"text", text}, {"finish_reason", finish_reason}}})}};
}

json error(int code, const std::string& message, const std::string& type) {
    return {{"error", {{"code", code}, {"message", message}, {"type", type}}}};
}

}  // namespace onebit::openai
