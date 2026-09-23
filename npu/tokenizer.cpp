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

#include "tokenizer.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace onebit::npu {

namespace {

// Qwen2/Qwen3's pre-tokenizer pattern, used when tokenizer.json names none.
const char* const kDefaultPattern =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|"
    "\\s+(?!\\S)|\\s+";

void utf8_append(std::string& s, uint32_t cp) {
    if (cp < 0x80) {
        s += char(cp);
    } else if (cp < 0x800) {
        s += char(0xC0 | (cp >> 6));
        s += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += char(0xE0 | (cp >> 12));
        s += char(0x80 | ((cp >> 6) & 0x3F));
        s += char(0x80 | (cp & 0x3F));
    } else {
        s += char(0xF0 | (cp >> 18));
        s += char(0x80 | ((cp >> 12) & 0x3F));
        s += char(0x80 | ((cp >> 6) & 0x3F));
        s += char(0x80 | (cp & 0x3F));
    }
}

// Length of the UTF-8 sequence a lead byte starts; 1 for a stray byte.
size_t utf8_len(unsigned char c) { return (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1; }

// The pattern from tokenizer.json's pre_tokenizer (a Split with a Regex), if any.
std::string split_pattern(const nlohmann::json& pre) {
    if (!pre.is_object()) return "";
    if (pre.value("type", "") == "Split" && pre.contains("pattern") && pre["pattern"].contains("Regex"))
        return pre["pattern"]["Regex"].get<std::string>();
    if (pre.contains("pretokenizers"))
        for (const auto& p : pre["pretokenizers"])
            if (auto s = split_pattern(p); !s.empty()) return s;
    return "";
}

}  // namespace

struct Tokenizer::Regex {
    pcre2_code* re = nullptr;
    explicit Regex(const std::string& pattern) {
        int err = 0;
        PCRE2_SIZE off = 0;
        re = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()), pattern.size(), PCRE2_UTF | PCRE2_UCP, &err,
                           &off, nullptr);
        if (!re) throw std::runtime_error("tokenizer: cannot compile the pre-tokenizer pattern");
    }
    ~Regex() { pcre2_code_free(re); }

    // The matched pieces, plus any text between matches as its own piece, so
    // nothing is dropped.
    std::vector<std::string> split(const std::string& text) const {
        std::vector<std::string> pieces;
        pcre2_match_data* md = pcre2_match_data_create_from_pattern(re, nullptr);
        size_t off = 0, last = 0;
        while (off < text.size()) {
            if (pcre2_match(re, reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(), off, 0, md, nullptr) < 0) break;
            const PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md);
            const size_t start = ov[0], end = ov[1];
            if (end == start) {
                off = start + 1;
                continue;
            }
            if (start > last) pieces.push_back(text.substr(last, start - last));
            pieces.push_back(text.substr(start, end - start));
            last = off = end;
        }
        if (last < text.size()) pieces.push_back(text.substr(last));
        pcre2_match_data_free(md);
        return pieces;
    }
};

Tokenizer::Tokenizer(const std::string& path) {
    // Byte-level alphabet (GPT-2): printable bytes map to themselves, the rest
    // to 256 + n in order.
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        byte_to_char_[b] = printable ? uint32_t(b) : uint32_t(256 + n++);
        char_to_byte_[byte_to_char_[b]] = uint8_t(b);
    }

    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    const auto root = nlohmann::json::parse(f);
    const auto& model = root.at("model");
    int max_id = -1;
    for (const auto& [text, id] : model.at("vocab").items()) {
        vocab_[text] = id.get<int>();
        max_id = std::max(max_id, id.get<int>());
    }
    if (root.contains("added_tokens"))
        for (const auto& t : root["added_tokens"]) {
            const std::string text = t.at("content");
            const int id = t.at("id");
            vocab_[text] = id;
            max_id = std::max(max_id, id);
            // Every added token is matched whole, as Hugging Face tokenizers does,
            // not only the ones flagged special: Qwen3's <think> and </think> are
            // added with special = false. (The ported code split special ones only.)
            added_.emplace_back(text, id);
        }
    std::sort(added_.begin(), added_.end(), [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    // Merges: ["a", "b"] pairs, or "a b" strings in older files. Rank = position.
    const auto& merges = model.at("merges");
    for (size_t i = 0; i < merges.size(); ++i) {
        const auto& m = merges[i];
        std::string a, b;
        if (m.is_array() && m.size() >= 2) {
            a = m[0];
            b = m[1];
        } else if (m.is_string()) {
            const std::string s = m;
            const size_t sp = s.find(' ');
            if (sp == std::string::npos) continue;
            a = s.substr(0, sp);
            b = s.substr(sp + 1);
        } else {
            continue;
        }
        merge_ranks_.emplace(a + "\x1f" + b, int(i));
    }

    id_to_text_.assign(size_t(max_id + 1), "");
    for (const auto& [text, id] : vocab_)
        if (id >= 0) id_to_text_[size_t(id)] = text;

    std::string pattern = root.contains("pre_tokenizer") ? split_pattern(root["pre_tokenizer"]) : "";
    split_ = std::make_unique<Regex>(pattern.empty() ? kDefaultPattern : pattern);
}

Tokenizer::~Tokenizer() = default;

std::vector<int> Tokenizer::bpe(const std::string& chars) const {
    std::vector<std::string> word;
    for (size_t i = 0; i < chars.size();) {
        const size_t len = utf8_len(static_cast<unsigned char>(chars[i]));
        word.push_back(chars.substr(i, len));
        i += len;
    }
    while (word.size() > 1) {  // merge the lowest-ranked adjacent pair until none applies
        int best = -1;
        size_t at = 0;
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            auto it = merge_ranks_.find(word[i] + "\x1f" + word[i + 1]);
            if (it != merge_ranks_.end() && (best < 0 || it->second < best)) {
                best = it->second;
                at = i;
            }
        }
        if (best < 0) break;
        word[at] += word[at + 1];
        word.erase(word.begin() + long(at) + 1);
    }
    std::vector<int> ids;
    for (const auto& w : word)
        if (auto it = vocab_.find(w); it != vocab_.end()) ids.push_back(it->second);
    return ids;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    auto plain = [&](const std::string& run) {
        for (const auto& piece : split_->split(run)) {
            std::string chars;
            for (unsigned char b : piece) utf8_append(chars, byte_to_char_[b]);
            const auto p = bpe(chars);
            ids.insert(ids.end(), p.begin(), p.end());
        }
    };
    size_t pos = 0;
    while (pos < text.size()) {
        // The earliest added token at or after pos; ties go to the longest.
        size_t best = std::string::npos;
        const std::pair<std::string, int>* sp = nullptr;
        for (const auto& s : added_) {
            const size_t at = text.find(s.first, pos);
            if (at != std::string::npos && (best == std::string::npos || at < best)) {
                best = at;
                sp = &s;
            }
        }
        if (!sp) {
            plain(text.substr(pos));
            break;
        }
        if (best > pos) plain(text.substr(pos, best - pos));
        ids.push_back(sp->second);
        pos = best + sp->first.size();
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || size_t(id) >= id_to_text_.size()) continue;
        const std::string& t = id_to_text_[size_t(id)];
        for (size_t i = 0; i < t.size();) {
            const unsigned char c = static_cast<unsigned char>(t[i]);
            const size_t len = utf8_len(c);
            if (i + len > t.size()) {
                out += t[i++];
                continue;
            }
            uint32_t cp = len == 1 ? c : len == 2 ? (c & 0x1Fu) : len == 3 ? (c & 0x0Fu) : (c & 0x07u);
            for (size_t j = 1; j < len; ++j) cp = (cp << 6) | (static_cast<unsigned char>(t[i + j]) & 0x3F);
            // Byte-level characters become their byte; anything else (the text of
            // special and added tokens) passes through as UTF-8.
            if (auto it = char_to_byte_.find(cp); it != char_to_byte_.end()) out += char(it->second);
            else out.append(t, i, len);
            i += len;
        }
    }
    return out;
}

}  // namespace onebit::npu
