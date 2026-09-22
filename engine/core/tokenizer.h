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
// tokenizer.h: byte-level BPE tokenizer read from GGUF metadata.
//
// Mirrors the HF `tokenizers` pipeline for the pre-tokenizers listed in
// docs/tokenizer.md: added-token split, NFC, regex split, byte-level mapping,
// BPE merges. Other tokenizer.ggml.pre values are refused, not approximated.
#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "gguf.h"

namespace onebit {

class Tokenizer {
public:
    static std::expected<Tokenizer, std::string> from_gguf(const GgufFile& f);

    // parse_special: recognise added tokens such as <|im_start|> in the text
    // (the HF default). When false they are tokenized as plain text.
    std::expected<std::vector<int32_t>, std::string> encode(std::string_view text, bool parse_special = true) const;

    // skip_special drops control tokens (token_type CONTROL) from the output.
    std::expected<std::string, std::string> decode(std::span<const int32_t> ids, bool skip_special = false) const;

    // NFC + pre-token split + byte-level mapping, without added-token
    // handling or BPE. Exposed so the split can be tested on its own.
    std::expected<std::vector<std::string>, std::string> pre_tokenize(std::string_view text) const;

    size_t n_vocab() const { return tokens_.size(); }
    int32_t bos_id() const { return bos_; }
    int32_t eos_id() const { return eos_; }
    const std::string& token_text(int32_t id) const { return tokens_.at(size_t(id)); }
    // CONTROL tokens (<|im_end|>, <|endoftext|>, ...): never rendered as text by default.
    bool is_control(int32_t id) const;

private:
    Tokenizer() = default;

    void split(std::u32string_view text, std::vector<std::string>& pieces) const;
    void encode_plain(std::u32string_view text, std::vector<int32_t>& out) const;
    void bpe(const std::string& piece, std::vector<int32_t>& out) const;

    std::vector<std::string> tokens_;
    std::vector<int32_t> token_type_;
    std::unordered_map<std::string, int32_t> token_id_;
    std::unordered_map<std::string, int32_t> merge_rank_;  // "left right" -> rank
    std::vector<std::pair<std::u32string, int32_t>> added_;  // sorted longest first
    std::string byte_to_unicode_[256];   // byte -> UTF-8 of its mapped code point
    std::unordered_map<char32_t, uint8_t> unicode_to_byte_;
    int32_t bos_ = -1, eos_ = -1;
};

}  // namespace onebit
