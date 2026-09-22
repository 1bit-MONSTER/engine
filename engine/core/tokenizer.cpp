#include "tokenizer.h"

#include <algorithm>
#include <climits>
#include <format>
#include <variant>

#include "unicode.h"

namespace onebit {

namespace {

// llama.cpp token types stored in tokenizer.ggml.token_type.
constexpr int32_t kTypeControl = 3;
constexpr int32_t kTypeUserDefined = 4;

using unicode::is_letter;
using unicode::is_number;
using unicode::is_whitespace;

bool is_newline(char32_t c) { return c == U'\r' || c == U'\n'; }

// Simple case folding for the letters the contraction branch can meet.
// U+017F LATIN SMALL LETTER LONG S folds to 's' under (?i).
char32_t fold(char32_t c) {
    if (c >= U'A' && c <= U'Z') return c + 32;
    if (c == 0x017F) return U's';
    return c;
}

// Length of the pre-token starting at s[i] for the Qwen2 split pattern
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Each branch is tried in order, as the regex engine does, and each returns
// the length its greedy quantifiers settle on after backtracking.
size_t qwen2_match(std::u32string_view s, size_t i) {
    const size_t n = s.size();
    auto at = [&](size_t k) -> char32_t { return k < n ? s[k] : 0; };
    auto is_other = [](char32_t c) { return !is_whitespace(c) && !is_letter(c) && !is_number(c); };

    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (s[i] == U'\'' && i + 1 < n) {
        const char32_t a = fold(at(i + 1)), b = fold(at(i + 2));
        if (a == U's' || a == U't') return 2;
        if ((a == U'r' && b == U'e') || (a == U'v' && b == U'e')) return 3;
        if (a == U'm') return 2;
        if (a == U'l' && b == U'l') return 3;
        if (a == U'd') return 2;
    }
    // [^\r\n\p{L}\p{N}]?\p{L}+
    {
        size_t j = i;
        const char32_t c = s[i];
        if (!is_newline(c) && !is_letter(c) && !is_number(c) && is_letter(at(i + 1))) j = i + 1;
        if (j < n && is_letter(s[j])) {
            while (j < n && is_letter(s[j])) ++j;
            return j - i;
        }
    }
    // \p{N}
    if (is_number(s[i])) return 1;
    // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
    {
        size_t j = i;
        if (s[i] == U' ' && i + 1 < n && is_other(s[i + 1])) j = i + 1;
        if (j < n && is_other(s[j])) {
            while (j < n && is_other(s[j])) ++j;
            while (j < n && is_newline(s[j])) ++j;
            return j - i;
        }
    }
    // The remaining branches all start with whitespace.
    size_t ws_end = i;
    while (ws_end < n && is_whitespace(s[ws_end])) ++ws_end;
    if (ws_end == i) return 1;  // unreachable: every code point matches a branch above
    // \s*[\r\n]+ : ends just after the last newline in the whitespace run.
    for (size_t k = ws_end; k > i; --k)
        if (is_newline(s[k - 1])) return k - i;
    // \s+(?!\S)
    if (ws_end == n) return ws_end - i;
    if (ws_end - i >= 2) return ws_end - i - 1;
    // \s+
    return ws_end - i;
}

// GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to 256+.
std::vector<char32_t> byte_to_codepoint() {
    std::vector<char32_t> map(256, 0);
    std::vector<bool> direct(256, false);
    for (int b = '!'; b <= '~'; ++b) direct[b] = true;
    for (int b = 0xA1; b <= 0xAC; ++b) direct[b] = true;
    for (int b = 0xAE; b <= 0xFF; ++b) direct[b] = true;
    char32_t next = 256;
    for (int b = 0; b < 256; ++b) map[b] = direct[b] ? char32_t(b) : next++;
    return map;
}

}  // namespace

std::expected<Tokenizer, std::string> Tokenizer::from_gguf(const GgufFile& f) {
    auto model = f.get_string("tokenizer.ggml.model");
    if (model != "gpt2") return std::unexpected(std::format("tokenizer model '{}' is not supported", model.value_or("")));
    auto pre = f.get_string("tokenizer.ggml.pre");
    if (pre != "qwen2") return std::unexpected(std::format("pre-tokenizer '{}' is not supported", pre.value_or("")));

    const GgufArray* tokens = f.get_array("tokenizer.ggml.tokens");
    const GgufArray* types = f.get_array("tokenizer.ggml.token_type");
    const GgufArray* merges = f.get_array("tokenizer.ggml.merges");
    if (!tokens || !types || !merges || types->size() != tokens->size())
        return std::unexpected(std::string("GGUF tokenizer arrays missing or inconsistent"));

    Tokenizer t;
    t.tokens_.reserve(tokens->size());
    t.token_type_.reserve(tokens->size());
    for (size_t i = 0; i < tokens->size(); ++i) {
        const auto* s = std::get_if<std::string>(&(*tokens)[i].v);
        const auto* ty = std::get_if<int64_t>(&(*types)[i].v);
        if (!s || !ty) return std::unexpected(std::format("bad tokenizer entry {}", i));
        t.tokens_.push_back(*s);
        t.token_type_.push_back(int32_t(*ty));
        t.token_id_.emplace(*s, int32_t(i));
        if (*ty == kTypeControl || *ty == kTypeUserDefined) {
            auto cps = unicode::decode_utf8(*s);
            if (!cps || cps->empty()) return std::unexpected(std::format("added token {} is not valid UTF-8", i));
            t.added_.emplace_back(std::u32string(cps->begin(), cps->end()), int32_t(i));
        }
    }
    std::stable_sort(t.added_.begin(), t.added_.end(),
                     [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    t.merge_rank_.reserve(merges->size());
    for (size_t i = 0; i < merges->size(); ++i) {
        const auto* m = std::get_if<std::string>(&(*merges)[i].v);
        if (!m || std::count(m->begin(), m->end(), ' ') != 1) return std::unexpected(std::format("bad merge {}", i));
        t.merge_rank_.emplace(*m, int32_t(i));
    }

    const auto map = byte_to_codepoint();
    for (int b = 0; b < 256; ++b) {
        unicode::append_utf8(t.byte_to_unicode_[b], map[b]);
        t.unicode_to_byte_[map[b]] = uint8_t(b);
    }
    for (int b = 0; b < 256; ++b)
        if (!t.token_id_.contains(t.byte_to_unicode_[b]))
            return std::unexpected(std::format("vocab has no token for byte {}", b));

    auto id_or = [&](const char* key) {
        auto v = f.get_uint(key);
        return (v && *v < t.tokens_.size()) ? int32_t(*v) : -1;
    };
    t.bos_ = id_or("tokenizer.ggml.bos_token_id");
    t.eos_ = id_or("tokenizer.ggml.eos_token_id");
    return t;
}

void Tokenizer::bpe(const std::string& piece, std::vector<int32_t>& out) const {
    // Split into the mapped characters (each 1-2 UTF-8 bytes long).
    std::vector<std::string> sym;
    for (size_t i = 0; i < piece.size();) {
        const size_t len = (static_cast<unsigned char>(piece[i]) < 0x80) ? 1 : 2;
        sym.push_back(piece.substr(i, len));
        i += len;
    }
    // Repeatedly merge the adjacent pair with the lowest rank (leftmost on ties).
    std::string key;
    while (sym.size() > 1) {
        int32_t best = INT32_MAX;
        size_t at = 0;
        for (size_t i = 0; i + 1 < sym.size(); ++i) {
            key.assign(sym[i]).append(" ").append(sym[i + 1]);
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best) {
                best = it->second;
                at = i;
            }
        }
        if (best == INT32_MAX) break;
        sym[at] += sym[at + 1];
        sym.erase(sym.begin() + std::ptrdiff_t(at) + 1);
    }
    for (const auto& s : sym) out.push_back(token_id_.at(s));  // single bytes are checked at load
}

void Tokenizer::split(std::u32string_view text, std::vector<std::string>& pieces) const {
    const std::vector<char32_t> norm = unicode::nfc(std::vector<char32_t>(text.begin(), text.end()));
    const std::u32string_view s(norm.data(), norm.size());
    std::string utf8;
    for (size_t i = 0; i < s.size();) {
        const size_t len = qwen2_match(s, i);
        utf8.clear();
        for (size_t k = i; k < i + len; ++k) unicode::append_utf8(utf8, s[k]);
        std::string& piece = pieces.emplace_back();
        for (unsigned char b : utf8) piece += byte_to_unicode_[b];
        i += len;
    }
}

void Tokenizer::encode_plain(std::u32string_view text, std::vector<int32_t>& out) const {
    std::vector<std::string> pieces;
    split(text, pieces);
    for (const auto& piece : pieces) bpe(piece, out);
}

std::expected<std::vector<std::string>, std::string> Tokenizer::pre_tokenize(std::string_view text) const {
    auto cps = unicode::decode_utf8(text);
    if (!cps) return std::unexpected(std::string("input is not valid UTF-8"));
    std::vector<std::string> pieces;
    split(std::u32string_view(cps->data(), cps->size()), pieces);
    return pieces;
}

std::expected<std::vector<int32_t>, std::string> Tokenizer::encode(std::string_view text, bool parse_special) const {
    auto cps = unicode::decode_utf8(text);
    if (!cps) return std::unexpected(std::string("input is not valid UTF-8"));
    const std::u32string_view s(cps->data(), cps->size());

    std::vector<int32_t> out;
    size_t start = 0;
    if (parse_special) {
        // Leftmost-longest match of added tokens, as HF's AddedVocabulary does.
        for (size_t i = 0; i < s.size();) {
            const std::pair<std::u32string, int32_t>* hit = nullptr;
            for (const auto& a : added_)
                if (s.substr(i, a.first.size()) == a.first) {
                    hit = &a;
                    break;
                }
            if (!hit) {
                ++i;
                continue;
            }
            encode_plain(s.substr(start, i - start), out);
            out.push_back(hit->second);
            i += hit->first.size();
            start = i;
        }
    }
    encode_plain(s.substr(start), out);
    return out;
}

std::expected<std::string, std::string> Tokenizer::decode(std::span<const int32_t> ids, bool skip_special) const {
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || size_t(id) >= tokens_.size()) return std::unexpected(std::format("token id {} out of range", id));
        const int32_t type = token_type_[size_t(id)];
        if (type == kTypeControl || type == kTypeUserDefined) {
            if (!(skip_special && type == kTypeControl)) out += tokens_[size_t(id)];
            continue;
        }
        auto cps = unicode::decode_utf8(tokens_[size_t(id)]);
        if (!cps) return std::unexpected(std::format("token {} is not valid UTF-8", id));
        for (char32_t cp : *cps) {
            auto it = unicode_to_byte_.find(cp);
            if (it == unicode_to_byte_.end()) return std::unexpected(std::format("token {} is not byte-level", id));
            out += char(it->second);
        }
    }
    return out;
}

}  // namespace onebit
