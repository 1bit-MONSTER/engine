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
