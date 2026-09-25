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

// Byte-level BPE from a Hugging Face tokenizer.json, as Qwen2/Qwen3 use it.
// Ported from 1bit-MONSTER npu-infer/tools/qwen3_tokenizer.cpp.
//
//   added-token pre-split (<|im_start|>, <think> and the like stay whole)
//   -> the pre-tokenizer's Split regex (PCRE2, Unicode classes)
//   -> bytes to the byte-level alphabet -> BPE by merge rank.
// Input is taken as already NFC-normalized.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace onebit::npu {

class Tokenizer {
public:
    explicit Tokenizer(const std::string& tokenizer_json);
    ~Tokenizer();
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    std::vector<int> encode(const std::string& text) const;
    // Ids outside the vocabulary decode to nothing.
    std::string decode(const std::vector<int>& ids) const;
    std::string decode(int id) const { return decode(std::vector<int>{id}); }
    // One past the largest id, added tokens included.
    int size() const { return int(id_to_text_.size()); }
    // The vocabulary id for a token string (added tokens included), or -1 if absent.
    int token_id(const std::string& token) const;

private:
    std::unordered_map<std::string, int> vocab_;             // token text -> id, added tokens included
    std::vector<std::string> id_to_text_;                    // dense, "" for gaps
    std::vector<std::pair<std::string, int>> added_;         // added tokens, longest first
    std::unordered_map<std::string, int> merge_ranks_;       // "a\x1fb" -> rank
    uint32_t byte_to_char_[256];
    std::unordered_map<uint32_t, uint8_t> char_to_byte_;
    struct Regex;
    std::unique_ptr<Regex> split_;

    std::vector<int> bpe(const std::string& chars) const;
};

}  // namespace onebit::npu
