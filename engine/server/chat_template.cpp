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
#include "chat_template.h"

#include <minja/chat-template.hpp>

namespace onebit {

ChatTemplate::ChatTemplate(ChatTemplate&&) noexcept = default;
ChatTemplate& ChatTemplate::operator=(ChatTemplate&&) noexcept = default;
ChatTemplate::~ChatTemplate() = default;

std::expected<ChatTemplate, std::string> ChatTemplate::from_source(const std::string& source, const std::string& bos,
                                                                   const std::string& eos) {
    ChatTemplate t;
    try {
        t.impl_ = std::make_unique<minja::chat_template>(source, bos, eos);
    } catch (const std::exception& e) {
        return std::unexpected(std::string("chat template does not parse: ") + e.what());
    }
    return t;
}

std::expected<ChatTemplate, std::string> ChatTemplate::from_gguf(const GgufFile& f) {
    auto source = f.get_string("tokenizer.chat_template");
    if (!source) return std::unexpected(std::string("GGUF has no tokenizer.chat_template"));
    auto token_text = [&](const char* key) -> std::string {
        auto id = f.get_uint(key);
        const GgufArray* tokens = f.get_array("tokenizer.ggml.tokens");
        if (!id || !tokens || *id >= tokens->size()) return "";
        const auto* s = std::get_if<std::string>(&(*tokens)[*id].v);
        return s ? *s : "";
    };
    return from_source(*source, token_text("tokenizer.ggml.bos_token_id"), token_text("tokenizer.ggml.eos_token_id"));
}

std::expected<std::string, std::string> ChatTemplate::render(const nlohmann::ordered_json& messages,
                                                             const nlohmann::ordered_json& tools,
                                                             bool add_generation_prompt,
                                                             const nlohmann::ordered_json& kwargs) const {
    minja::chat_template_inputs in;
    in.messages = messages;
    in.tools = tools;
    in.add_generation_prompt = add_generation_prompt;
    in.extra_context = kwargs.is_object() ? kwargs : nlohmann::ordered_json::object();
    try {
        return impl_->apply(in);
    } catch (const std::exception& e) {
        return std::unexpected(std::string("chat template failed: ") + e.what());
    }
}

}  // namespace onebit
