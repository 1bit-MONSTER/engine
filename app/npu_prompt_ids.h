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

// `/v1/completions`' prompt: text, or the exact token ids the caller already has.
//
// llama.cpp's `/completion` accepts both (`prompt` as a string or as an array of ids), and a
// parity run needs the second form: the same *text* reaches two engines through two tokenizers
// and one prompt-construction rule each, so a text prompt measures the tokenizer and the BOS
// policy as much as it measures the forward.  Measured on Zyphra/ZR1-1.5B (2026-11-03): five raw
// prompts, forward-serve and llama.cpp on the same container weights, and the two agreed on the
// first generated token in 0 of 5 -- the cause was not the arithmetic but a one-token prompt
// difference (`forward-serve` prepends BOS only on the chat route, app/forward_serve.cpp).  With
// the ids supplied by the caller both engines evaluate literally the same positions.
//
// Ids are taken as given: no BOS is inserted and no text is tokenized, because the caller is
// stating the prompt, not asking for it to be derived.  An id outside the vocabulary is a 400 --
// the embedding lookup indexes by id, so a value the vocabulary does not have is a request
// error, never a silently clamped one.  (OpenAI's own `/v1/completions` reads an array as a batch
// of strings; that form is not implemented here, and this rule refuses it by name rather than
// batching it silently.  llama.cpp's `/completion` is what the id shape follows.)
#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace onebit::npu {

// Fills `out` and returns true when `body` carries the prompt as token ids; returns false when
// it is a string (the caller tokenizes it as before).  Throws std::runtime_error naming the
// problem when `prompt` is missing, empty, or neither a string nor an array of ids inside the
// vocabulary.
inline bool prompt_ids_from_body(const nlohmann::json& body, int vocab, std::vector<int>& out) {
    const auto it = body.find("prompt");
    if (it == body.end()) throw std::runtime_error("prompt: missing");
    if (it->is_string()) return false;
    if (!it->is_array()) throw std::runtime_error("prompt must be a string or an array of token ids");
    if (it->empty()) throw std::runtime_error("prompt: an empty token-id array cannot be evaluated");
    out.reserve(it->size());
    for (size_t i = 0; i < it->size(); i++) {
        const nlohmann::json& v = (*it)[i];
        if (!v.is_number_integer())
            throw std::runtime_error("prompt[" + std::to_string(i) + "] is not an integer token id");
        const long long id = v.get<long long>();
        if (id < 0 || (vocab > 0 && id >= vocab))
            throw std::runtime_error("prompt[" + std::to_string(i) + "] = " + std::to_string(id) +
                                     " is outside the vocabulary (0.." + std::to_string(vocab - 1) + ")");
        out.push_back(int(id));
    }
    return true;
}

}  // namespace onebit::npu
