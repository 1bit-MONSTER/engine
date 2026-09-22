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
// chat_template.h: renders OpenAI-style messages with the model's own Jinja
// chat template (GGUF key tokenizer.chat_template), via minja.
#pragma once

#include <expected>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "gguf.h"

namespace minja {
class chat_template;
}

namespace onebit {

class ChatTemplate {
public:
    static std::expected<ChatTemplate, std::string> from_gguf(const GgufFile& f);
    static std::expected<ChatTemplate, std::string> from_source(const std::string& source, const std::string& bos,
                                                                const std::string& eos);

    // messages: array of {role, content, ...}; tools: array or null.
    // kwargs: extra template variables (chat_template_kwargs, e.g. enable_thinking).
    std::expected<std::string, std::string> render(const nlohmann::ordered_json& messages,
                                                   const nlohmann::ordered_json& tools, bool add_generation_prompt,
                                                   const nlohmann::ordered_json& kwargs) const;

    ChatTemplate(ChatTemplate&&) noexcept;
    ChatTemplate& operator=(ChatTemplate&&) noexcept;
    ~ChatTemplate();

private:
    ChatTemplate() = default;
    std::unique_ptr<minja::chat_template> impl_;
};

}  // namespace onebit
