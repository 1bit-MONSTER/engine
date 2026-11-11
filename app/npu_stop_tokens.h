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

// The ids that end a generation: config.json's `eos_token_id`, which HF writes either as one
// integer (Qwen2.5/Qwen3: 151645) or as a list (Llama-3.2: [128001, 128008, 128009]).
//
// `forward-serve` read it with json_int(), which answers its default for anything that is not an
// integer, so on a list-valued container the serve's only stop ids were -1 and whatever
// "<|endoftext|>" resolved to by name.  Measured on the Llama-3.2-1B OGA pack (engine#1044): the
// answer was correct through "The capital of France is Paris." and then the model's own first stop
// token, <|eot_id|> 128009, was *served as text* and the model invented a second turn --
// "The capital of France is Paris.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI'm
// sorry," with finish_reason "length" instead of "stop".  A wrong stop set looks like a model
// defect and is not one: every position before it agreed with the reference.
//
// The whole list is honoured.  A container that names several stop ids means all of them (Llama-3.2
// lists <|endoftext|>, <|eot_id|> and <|start_header_id|>); stopping on the first one only would
// leave the others to be served as text on prompts whose answer ends differently.
//
// Shapes that cannot be read as ids are ignored rather than fatal: this is the container's own
// config.json, written by our importer, and a serve that refuses to load over one malformed entry
// is worse than one that stops on the ids it could read.  A missing key yields an empty list.
#pragma once

#include <nlohmann/json.hpp>

#include <fstream>
#include <string>
#include <vector>

namespace onebit::npu {

/// Every integer in config.json's `eos_token_id`, whether it is one integer or a list of them.
inline std::vector<int> eos_token_ids(const nlohmann::json& cfg) {
    std::vector<int> ids;
    if (!cfg.contains("eos_token_id")) return ids;
    const nlohmann::json& e = cfg["eos_token_id"];
    if (e.is_number_integer()) {
        ids.push_back(e.get<int>());
    } else if (e.is_array()) {
        for (const nlohmann::json& v : e)
            if (v.is_number_integer()) ids.push_back(v.get<int>());
    }
    return ids;
}

/// tokenizer_config.json's `eos_token` as text: a plain string, or the AddedToken object form
/// `{"content": "<|im_end|>", ...}`.  Empty when the key is absent, null or another shape.
inline std::string eos_token_text(const nlohmann::json& tokenizer_config) {
    if (!tokenizer_config.is_object() || !tokenizer_config.contains("eos_token")) return {};
    const nlohmann::json& e = tokenizer_config["eos_token"];
    if (e.is_string()) return e.get<std::string>();
    if (e.is_object() && e.contains("content") && e["content"].is_string()) return e["content"].get<std::string>();
    return {};
}

/// Every id that ends a generation on a container, in the order they were found and without repeats:
///
///   1. config.json's `eos_token_id` (one integer or a list): what eos_token_ids() reads;
///   2. tokenizer_config.json's `eos_token`, looked up in the tokenizer.  The two files are written by
///      different tools and need not agree.  FastFlowLM's Qwen3.5-2B container says `eos_token_id: 248044`
///      (<|endoftext|>) in config.json and `eos_token: "<|im_end|>"` (248046) in tokenizer_config.json, and the
///      model ends a chat turn with the second.  With only the first, the answer came back as
///      "Item 3 is the 3D printer.<|im_end|>\n": the turn end served as text and one more token generated
///      after it (engine#1785's row on that container);
///   3. `chat_turn_end`, when the caller's chat route renders a prompt whose turns end with a marker of its
///      own (ChatML's <|im_end|>): a prompt we close with that marker is answered by a turn that ends with it,
///      whatever the two files say.  Pass an empty string when the route has no such marker.
///
/// `id_of` maps a token's text to its id, or a negative number when the tokenizer has no such token; a name it
/// does not know adds nothing.
template <typename IdOf>
std::vector<int> stop_token_ids(const nlohmann::json& config, const nlohmann::json& tokenizer_config,
                                const std::string& chat_turn_end, IdOf&& id_of) {
    std::vector<int> ids = eos_token_ids(config);
    const auto add = [&](const std::string& text) {
        if (text.empty()) return;
        const int id = id_of(text);
        if (id < 0) return;
        for (int have : ids)
            if (have == id) return;
        ids.push_back(id);
    };
    add(eos_token_text(tokenizer_config));
    add(chat_turn_end);
    return ids;
}

/// A JSON file, or null when it is missing or malformed: both stop-id sources are optional.
inline nlohmann::json read_json_or_null(const std::string& path) {
    std::ifstream f(path);
    if (!f) return nullptr;
    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception&) {
        return nullptr;
    }
    return j;
}

/// eos_token_ids() for a container's config.json.  Unreadable file -> empty.
inline std::vector<int> eos_token_ids_from_config(const std::string& cfg_path) {
    std::ifstream f(cfg_path);
    if (!f) return {};
    nlohmann::json cfg;
    try {
        f >> cfg;
    } catch (const std::exception&) {
        return {};
    }
    return eos_token_ids(cfg);
}

}  // namespace onebit::npu
