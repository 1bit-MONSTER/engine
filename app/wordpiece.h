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

// WordPiece, for the BERT-family embedding encoders (engine#1311).
//
// An embedding model's contract starts at the token ids, so this follows the `BertTokenizer` pipeline the
// reference model ships (`tokenizer.json`: BertNormalizer{clean_text, handle_chinese_chars, strip_accents,
// lowercase} + BertPreTokenizer + WordPiece{unk=[UNK]} + [CLS]/[SEP]) rather than "looks like BERT":
//
//   clean_text        drop NUL/0xFFFD/control, every whitespace becomes one space, trim
//   chinese chars     a CJK character becomes its own token (a space either side)
//   whitespace        split on spaces
//   per word          lowercase, strip accents (NFD minus combining marks), split punctuation out
//   wordpiece         greedy longest-match-first with "##" continuations; a piece that cannot be matched is
//                     [UNK] for the WHOLE word (not per character)
//
// Unicode work is table-driven and deliberately small: ASCII, Latin-1 Supplement and Latin Extended-A for
// case and accents (that is where English text with accents lives), the common punctuation blocks, and the
// CJK ranges the reference splits on. A code point outside the tables passes through unchanged, which is the
// right answer for ASCII and a documented limit elsewhere; `tests/embed_wordpiece_test.cpp` pins the rules and
// the STS-B corpus is the measured proof that they agree with the reference on real text.
#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace onebit::wordpiece {

// ---------------------------------------------------------------- UTF-8

inline void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// Decode one code point; returns the byte length (>=1). Invalid bytes decode as themselves so that no input
// can crash the tokenizer.
inline size_t decode_utf8(std::string_view s, size_t i, uint32_t& cp) {
    const auto c = static_cast<unsigned char>(s[i]);
    size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
    if (i + len > s.size()) return 1;
    cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
    for (size_t k = 1; k < len; ++k) {
        const auto b = static_cast<unsigned char>(s[i + k]);
        if ((b & 0xC0) != 0x80) return 1;
        cp = (cp << 6) | (b & 0x3F);
    }
    return len;
}

// ---------------------------------------------------------------- character classes

// `is_whitespace`: space, tab, newline, carriage return, or the Zs separators we meet in real text.
inline bool is_space(uint32_t cp) {
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0B || cp == 0x0C) return true;
    return cp == 0xA0 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}

// `is_control`: the Cc/Cf range minus the whitespace controls the reference keeps.
inline bool is_control(uint32_t cp) {
    if (cp == 0 || cp == 0xFFFD) return true;
    if (cp == '\t' || cp == '\n' || cp == '\r') return false;
    if (cp < 0x20 || (cp >= 0x7F && cp <= 0x9F)) return true;
    return cp == 0x200B || cp == 0x200E || cp == 0x200F || cp == 0xFEFF || (cp >= 0x202A && cp <= 0x202E) ||
           (cp >= 0x2066 && cp <= 0x2069);
}

// `is_punctuation`: the ASCII ranges the reference uses, plus the Unicode punctuation this tokenizer meets.
inline bool is_punct(uint32_t cp) {
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126))
        return true;
    if (cp >= 0x2000 && cp <= 0x206F) {  // general punctuation: quotes, dashes, ellipsis, bullets, ...
        return cp != 0x200B && cp != 0x200E && cp != 0x200F && !(cp >= 0x2028 && cp <= 0x2029) &&
               cp != 0x202F && cp != 0x205F;
    }
    switch (cp) {
        case 0xA1: case 0xA7: case 0xAB: case 0xB6: case 0xB7: case 0xBB: case 0xBF:
        case 0x2E2E: case 0x3001: case 0x3002: case 0x3003: case 0x3008: case 0x3009: case 0x300A:
        case 0x300B: case 0x300C: case 0x300D: case 0x300E: case 0x300F: case 0x3010: case 0x3011:
        case 0xFF01: case 0xFF02: case 0xFF07: case 0xFF08: case 0xFF09: case 0xFF0C: case 0xFF0E:
        case 0xFF1A: case 0xFF1B: case 0xFF1F: case 0xFF3B: case 0xFF3D: case 0xFF5E:
            return true;
        default:
            return false;
    }
}

// The CJK ranges the reference treats as one-token-per-character.
inline bool is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B920 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

// Lowercase for ASCII, Latin-1 Supplement and Latin Extended-A; anything else passes through.
inline uint32_t lower(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + 32;
    if (cp >= 0xC0 && cp <= 0xDE && cp != 0xD7) return cp + 32;
    if (cp >= 0x100 && cp <= 0x137) return (cp % 2 == 0) ? cp + 1 : cp;          // pairs AaBb...
    if (cp == 0x130) return 'i';                                                 // I with dot above
    if (cp == 0x139 || cp == 0x13B || cp == 0x13D || cp == 0x13F || cp == 0x141) return cp + 1;
    if (cp == 0x143 || cp == 0x145 || cp == 0x147) return cp + 1;
    if (cp >= 0x14A && cp <= 0x177) return (cp % 2 == 0) ? cp + 1 : cp;
    if (cp == 0x178) return 0xFF;
    if (cp == 0x179 || cp == 0x17B || cp == 0x17D) return cp + 1;
    if (cp >= 0x391 && cp <= 0x3A9 && cp != 0x3A2) return cp + 32;               // Greek, for completeness
    if (cp >= 0x410 && cp <= 0x42F) return cp + 32;                              // Cyrillic
    return cp;
}

// The base letter of a precomposed Latin character (NFD minus the combining mark). 0 means "no decomposition
// in the table", and the character is then kept as it is.
inline uint32_t strip_accent(uint32_t cp) {
    switch (cp) {
        case 0xC0: case 0xC1: case 0xC2: case 0xC3: case 0xC4: case 0xC5: return 'a';
        case 0xC7: return 'c';
        case 0xC8: case 0xC9: case 0xCA: case 0xCB: return 'e';
        case 0xCC: case 0xCD: case 0xCE: case 0xCF: return 'i';
        case 0xD1: return 'n';
        case 0xD2: case 0xD3: case 0xD4: case 0xD5: case 0xD6: case 0xD8: return 'o';
        case 0xD9: case 0xDA: case 0xDB: case 0xDC: return 'u';
        case 0xDD: return 'y';
        case 0xE0: case 0xE1: case 0xE2: case 0xE3: case 0xE4: case 0xE5: return 'a';
        case 0xE7: return 'c';
        case 0xE8: case 0xE9: case 0xEA: case 0xEB: return 'e';
        case 0xEC: case 0xED: case 0xEE: case 0xEF: return 'i';
        case 0xF1: return 'n';
        case 0xF2: case 0xF3: case 0xF4: case 0xF5: case 0xF6: case 0xF8: return 'o';
        case 0xF9: case 0xFA: case 0xFB: case 0xFC: return 'u';
        case 0xFD: case 0xFF: return 'y';
        // Latin Extended-A: a-acute .. y-diaeresis, the ones a real corpus carries
        case 0x100: case 0x101: case 0x102: case 0x103: case 0x104: case 0x105: return 'a';
        case 0x106: case 0x107: case 0x10C: case 0x10D: return 'c';
        case 0x10E: case 0x10F: case 0x110: case 0x111: return 'd';
        case 0x112: case 0x113: case 0x114: case 0x115: case 0x116: case 0x117: case 0x118: case 0x119: return 'e';
        case 0x11A: case 0x11B: case 0x11E: case 0x11F: case 0x120: case 0x121: case 0x122: case 0x123: return 'g';
        case 0x124: case 0x125: return 'h';
        case 0x126: case 0x127: return 'i';
        case 0x12A: case 0x12B: case 0x12E: case 0x12F: case 0x130: return 'i';
        case 0x136: case 0x137: case 0x138: return 'k';
        case 0x139: case 0x13A: case 0x13B: case 0x13C: case 0x13D: case 0x13E: case 0x13F: case 0x140:
        case 0x141: case 0x142: return 'l';
        case 0x143: case 0x144: case 0x145: case 0x146: case 0x147: case 0x148: return 'n';
        case 0x14C: case 0x14D: case 0x14E: case 0x14F: case 0x150: case 0x151: return 'o';
        case 0x154: case 0x155: case 0x156: case 0x157: case 0x158: case 0x159: return 'r';
        case 0x15A: case 0x15B: case 0x15C: case 0x15D: case 0x15E: case 0x15F: case 0x160: case 0x161: return 's';
        case 0x162: case 0x163: case 0x164: case 0x165: case 0x166: case 0x167: return 't';
        case 0x168: case 0x169: case 0x16A: case 0x16B: case 0x16C: case 0x16D: case 0x16E: case 0x16F:
        case 0x170: case 0x171: case 0x172: case 0x173: return 'u';
        case 0x174: case 0x175: case 0x176: case 0x177: return 'w';
        case 0x179: case 0x17A: case 0x17B: case 0x17C: case 0x17D: case 0x17E: return 'z';
        default: return 0;
    }
}

// ---------------------------------------------------------------- tokenizer

class BertTokenizer {
public:
    // vocab.txt: one piece per line, id = line number (the BERT convention the reference ships).
    bool load_vocab(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;
        std::string line;
        int id = 0;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            vocab_.emplace(line, id++);
            if (line == "[UNK]") unk_ = id - 1;
            if (line == "[CLS]") cls_ = id - 1;
            if (line == "[SEP]") sep_ = id - 1;
            if (line == "[PAD]") pad_ = id - 1;
        }
        return !vocab_.empty() && unk_ >= 0;
    }

    int vocab_size() const { return static_cast<int>(vocab_.size()); }
    int unk_id() const { return unk_; }
    int cls_id() const { return cls_; }
    int sep_id() const { return sep_; }
    int pad_id() const { return pad_; }

    // The reference's `_clean_text`: NUL/0xFFFD dropped, control dropped, whitespace collapsed to ' '.
    static std::string clean_text(std::string_view in) {
        std::string out;
        for (size_t i = 0; i < in.size();) {
            uint32_t cp = 0;
            const size_t n = decode_utf8(in, i, cp);
            if (cp == 0 || cp == 0xFFFD) {
                // dropped
            } else if (is_space(cp)) {
                out.push_back(' ');
            } else if (!is_control(cp)) {
                out.append(in.substr(i, n));
            }
            i += n;
        }
        const size_t b = out.find_first_not_of(' ');
        if (b == std::string::npos) return {};
        const size_t e = out.find_last_not_of(' ');
        return out.substr(b, e - b + 1);
    }

    // One wordPiece-able piece -> its ids. Exposed for the test.
    std::vector<int> word_ids(const std::vector<uint32_t>& cps) const {
        std::vector<int> ids;
        if (cps.empty()) return ids;
        std::string word;
        for (uint32_t cp : cps) append_utf8(word, cp);
        size_t start = 0;
        bool first = true;
        while (start < word.size()) {
            size_t end = word.size();
            int found = -1;
            size_t found_end = start;
            while (start < end) {
                const std::string piece = (first ? "" : "##") + word.substr(start, end - start);
                const auto it = vocab_.find(piece);
                if (it != vocab_.end()) {
                    found = it->second;
                    found_end = end;
                    break;
                }
                // step back one UTF-8 character
                size_t k = end - 1;
                while (k > start && (static_cast<unsigned char>(word[k]) & 0xC0) == 0x80) --k;
                end = k;
            }
            if (found < 0) return {unk_};  // the whole word is unknown, not the first character
            ids.push_back(found);
            start = found_end;
            first = false;
        }
        return ids;
    }

    // A sentence -> ids with [CLS]/[SEP], truncated to max_tokens including both specials.
    std::vector<int> encode(std::string_view text, int max_tokens, bool add_specials = true) const {
        std::vector<int> ids;
        if (add_specials) ids.push_back(cls_);
        std::string s = clean_text(text);
        // handle_chinese_chars: a CJK character is its own token
        std::string spaced;
        for (size_t i = 0; i < s.size();) {
            uint32_t cp = 0;
            const size_t n = decode_utf8(s, i, cp);
            if (is_cjk(cp)) {
                spaced.push_back(' ');
                spaced.append(s.substr(i, n));
                spaced.push_back(' ');
            } else {
                spaced.append(s.substr(i, n));
            }
            i += n;
        }
        const int room = add_specials ? max_tokens - 2 : max_tokens;
        size_t i = 0;
        while (i < spaced.size() && static_cast<int>(ids.size()) - (add_specials ? 1 : 0) < room) {
            while (i < spaced.size() && spaced[i] == ' ') ++i;
            if (i >= spaced.size()) break;
            size_t j = i;
            while (j < spaced.size() && spaced[j] != ' ') ++j;
            const std::string_view token(spaced.data() + i, j - i);
            i = j;
            for (int id : token_ids(token)) {
                if (static_cast<int>(ids.size()) - (add_specials ? 1 : 0) >= room) break;
                ids.push_back(id);
            }
        }
        if (add_specials) ids.push_back(sep_);
        return ids;
    }

    // lowercase, strip accents, split punctuation out -- the reference's BasicTokenizer per whitespace token.
    static std::vector<uint32_t> basic_token(std::string_view token) {
        std::vector<uint32_t> out;
        for (size_t i = 0; i < token.size();) {
            uint32_t cp = 0;
            const size_t n = decode_utf8(token, i, cp);
            i += n;
            cp = lower(cp);
            const uint32_t base = strip_accent(cp);
            if (base != 0) cp = base;
            if (cp >= 0x300 && cp <= 0x36F) continue;  // a combining mark left over
            if (is_punct(cp)) {
                out.push_back(0xFFFFFFFFu);  // punctuation is its own piece
                out.push_back(cp);
                out.push_back(0xFFFFFFFFu);
                continue;
            }
            out.push_back(cp);
        }
        // the sentinels above are resolved by the caller through split_on_punct()
        return out;
    }

    // The ids of one whitespace token, punctuation split out (the reference's BasicTokenizer + WordPiece).
    std::vector<int> token_ids(std::string_view token) const {
        std::vector<int> ids;
        std::vector<uint32_t> cur;
        const std::vector<uint32_t> cps = basic_token(token);
        for (uint32_t cp : cps) {
            if (cp == 0xFFFFFFFFu) {
                for (int id : word_ids(cur)) ids.push_back(id);
                cur.clear();
                continue;
            }
            cur.push_back(cp);
        }
        for (int id : word_ids(cur)) ids.push_back(id);
        return ids;
    }

private:
    std::unordered_map<std::string, int> vocab_;
    int unk_ = -1, cls_ = -1, sep_ = -1, pad_ = -1;
};

}  // namespace onebit::wordpiece
