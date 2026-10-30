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

// GBNF grammars for `1bit forward-serve`: the format llama.cpp parses in
// third_party/llama.cpp/src/llama-grammar.cpp (`x ::= alternates`, literals, char
// classes, `.`, rule refs, `( )`, and the `* + ? {m,n}` repetition operators),
// compiled to what a token decoder needs -- "does this candidate token's text stay
// inside the grammar?" and "may the answer stop here?".
//
// This mirrors llama.cpp's semantics instead of calling it: llama_sampler_init_grammar
// wants a llama_vocab and llama_sampler_sample wants a llama_context, and the NPU
// route has neither (it holds raw logits and an npu::Tokenizer), while llama.cpp is
// only linked under ONEBIT_HRX.  Matching is byte-wise, so `.` and a char class
// consume one byte where llama.cpp consumes one Unicode code point; an ASCII grammar
// (JSON, tool calls, the usual case) behaves identically, a grammar that spells out
// non-ASCII text can reject a token here that llama.cpp would accept, never the other
// way round.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace onebit::npu {

// One 256-bit byte mask: has(b) says "this byte is matched by this element".
struct ByteMask {
    std::array<uint64_t, 4> w{{0, 0, 0, 0}};

    void set(unsigned b) { w[b >> 6] |= uint64_t(1) << (b & 63); }
    void set_range(unsigned lo, unsigned hi) {
        for (unsigned b = lo; b <= hi; b++) set(b);
    }
    bool has(unsigned b) const { return (w[b >> 6] >> (b & 63)) & 1; }
    void flip_all() { w = {{~w[0], ~w[1], ~w[2], ~w[3]}}; }
};

class Grammar {
public:
    // A parse in progress: the frames still to match, innermost last.  A frame packs
    // the rule id and the element index inside that rule.  Stacks are kept sorted and
    // deduplicated, so two states that hold the same frames compare equal.
    struct State {
        std::vector<std::vector<uint64_t>> stacks;
        bool accepting = false;  // some parse reached the end of the root rule

        bool dead() const { return stacks.empty() && !accepting; }
    };

    // Compile GBNF text.  Throws std::runtime_error("gbnf: <what> at line N"); the
    // message is meant to be handed back to the client as the 400 body.
    explicit Grammar(const std::string& gbnf);

    State start() const;

    // Feed one token's decoded text.  Returns false and leaves `st` untouched when the
    // text leaves the grammar; otherwise `st` holds the resulting parse.
    bool feed(State& st, const std::string& text) const;

    // The answer may stop here only where the grammar can end.
    static bool can_stop(const State& st) { return st.accepting; }

    // The bytes any pending element can match next.  A caller that has to test many
    // candidate strings (one per vocabulary token) uses this first: a candidate whose
    // first byte is not in here cannot match, so it never needs a full feed().
    ByteMask next_bytes(const State& st) const;

private:
    enum ElemType : uint8_t { E_CHAR, E_RULE, E_ALT, E_END };
    struct Elem {
        ElemType type = E_END;
        uint32_t value = 0;  // E_RULE: the referenced rule
        ByteMask mask;       // E_CHAR: the bytes this element matches
    };

    friend class GrammarParser;

    void advance_stack(const std::vector<uint64_t>& stack, std::vector<std::vector<uint64_t>>& out) const;
    bool step(const State& st, unsigned b, State& out) const;
    // Is there an element after `pos` in `rule` that still has to match?  A frame whose
    // alternative ended (pos+1 is an ALT or the rule's END) is not worth keeping: the
    // enclosing frame is already past it.
    bool continuation(uint32_t rule, uint32_t pos) const {
        const uint32_t next = pos + 1;
        return next < rules_[rule].size() && rules_[rule][next].type != E_ALT &&
               rules_[rule][next].type != E_END;
    }

    std::vector<std::vector<Elem>> rules_;
    std::vector<std::vector<uint32_t>> alt_starts_;  // per rule: first element of each alternative
    uint32_t root_ = 0;
};

// Frame = (rule id, element index).
inline uint64_t gbnf_frame(uint32_t rule, uint32_t pos) { return (uint64_t(rule) << 32) | pos; }
inline uint32_t gbnf_rule(uint64_t f) { return uint32_t(f >> 32); }
inline uint32_t gbnf_pos(uint64_t f) { return uint32_t(f & 0xffffffffu); }

// Recursive-descent parser for the GBNF text, emitting into Grammar's rule vectors.
// Repetition uses llama.cpp's rewrite rules (third_party/llama.cpp/src/llama-grammar.cpp,
// handle_repetitions): S{m,n} becomes m copies plus n-m generated tail rules, S* becomes
// a tail rule that recurses or ends.  Every rule stays a flat element list, which is all
// the runtime below understands.
class GrammarParser {
public:
    GrammarParser(Grammar& g, const std::string& src) : g_(g), src_(src) {}

    // Parse the whole grammar; returns the id of the `root` rule.
    uint32_t parse_all();

private:
    using Elem = Grammar::Elem;

    [[noreturn]] void fail(const std::string& what) const;
    char peek() const { return pos_ < src_.size() ? src_[pos_] : '\0'; }
    // Skip whitespace and comments.  A newline ends the rule it is in, so it is only
    // crossed where GBNF allows a rule to wrap: inside ( ), after '|', and right after
    // '::=' (third_party/llama.cpp/src/llama-grammar.cpp, parse_space/newline_ok).
    void space();
    void space_across_lines() {
        const bool saved = newline_ok_;
        newline_ok_ = true;
        space();
        newline_ok_ = saved;
    }
    uint32_t parse_char();
    uint32_t parse_utf8();
    void push_char(uint32_t rule, const ByteMask& m);
    void push_utf8(uint32_t rule, uint32_t cp);
    uint32_t symbol_id(const std::string& name);
    uint32_t generated_id();
    void add_rule(uint32_t id, const std::vector<Elem>& rule);
    void parse_alternates(const std::string& name, uint32_t rule);
    void parse_sequence(const std::string& name, uint32_t rule);
    void repetitions(uint32_t rule, size_t last_sym_start, uint64_t min_times, uint64_t max_times);

    Grammar& g_;
    const std::string& src_;
    size_t pos_ = 0;
    size_t line_ = 1;
    bool newline_ok_ = false;  // may the sequence being parsed run past a newline?
    std::vector<std::pair<std::string, uint32_t>> symbols_;
};

namespace gbnf_detail {
inline bool is_digit_char(char c) { return c >= '0' && c <= '9'; }
inline bool is_word_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '-' || is_digit_char(c);
}
inline bool is_hex_char(char c) {
    return is_digit_char(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}
inline uint32_t hex_val(char c) { return uint32_t(c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10); }
}  // namespace gbnf_detail

inline void GrammarParser::fail(const std::string& what) const {
    throw std::runtime_error("gbnf: " + what + " at line " + std::to_string(line_));
}

inline void GrammarParser::space() {
    for (;;) {
        const char c = peek();
        if (c == '\n') {
            if (!newline_ok_) return;
            line_++;
            pos_++;
        } else if (c == ' ' || c == '\t' || c == '\r') {
            pos_++;
        } else if (c == '#') {
            while (pos_ < src_.size() && src_[pos_] != '\n') pos_++;
        } else {
            return;
        }
    }
}

inline uint32_t GrammarParser::parse_char() {
    using namespace gbnf_detail;
    if (pos_ >= src_.size()) fail("unexpected end of input");
    if (peek() != '\\') return (unsigned char)src_[pos_++];
    pos_++;
    const char c = peek();
    if (pos_ >= src_.size()) fail("unexpected end of input");
    pos_++;
    switch (c) {
        case 'x':
        case 'u': {
            const int n = c == 'x' ? 2 : 4;
            uint32_t v = 0;
            for (int i = 0; i < n; i++) {
                if (!is_hex_char(peek())) fail("expecting hex digits");
                v = v * 16 + hex_val(src_[pos_++]);
            }
            return v;
        }
        case 'n': return '\n';
        case 'r': return '\r';
        case 't': return '\t';
        case '\\': return '\\';
        case '"': return '"';
        case '[': return '[';
        case ']': return ']';
        default: fail(std::string("unknown escape \\") + c);
    }
}

// One UTF-8 encoded code point out of the grammar text (a grammar may hold non-ASCII
// literals); matching stays byte-wise, so push_utf8 turns it back into bytes.
inline uint32_t GrammarParser::parse_utf8() {
    const unsigned char lead = (unsigned char)src_[pos_++];
    int n = 0;
    uint32_t cp = 0;
    if ((lead & 0xE0) == 0xC0) {
        n = 1;
        cp = lead & 0x1F;
    } else if ((lead & 0xF0) == 0xE0) {
        n = 2;
        cp = lead & 0x0F;
    } else if ((lead & 0xF8) == 0xF0) {
        n = 3;
        cp = lead & 0x07;
    } else {
        fail("invalid UTF-8 in a literal");
    }
    for (int i = 0; i < n; i++) {
        if (pos_ >= src_.size()) fail("truncated UTF-8 in a literal");
        const unsigned char c = (unsigned char)src_[pos_];
        if ((c & 0xC0) != 0x80) fail("invalid UTF-8 in a literal");
        cp = (cp << 6) | (c & 0x3F);
        pos_++;
    }
    return cp;
}

inline void GrammarParser::push_char(uint32_t rule, const ByteMask& m) {
    Elem e;
    e.type = Grammar::E_CHAR;
    e.mask = m;
    g_.rules_[rule].push_back(e);
}

// A literal escapes to one code point; matching is byte-wise, so an ASCII code point is
// one element and anything wider becomes its UTF-8 bytes, one element each.
inline void GrammarParser::push_utf8(uint32_t rule, uint32_t cp) {
    if (cp < 0x80) {
        ByteMask m;
        m.set(cp);
        push_char(rule, m);
        return;
    }
    std::string bytes;
    if (cp < 0x800) {
        bytes += char(0xC0 | (cp >> 6));
        bytes += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        bytes += char(0xE0 | (cp >> 12));
        bytes += char(0x80 | ((cp >> 6) & 0x3F));
        bytes += char(0x80 | (cp & 0x3F));
    } else {
        bytes += char(0xF0 | (cp >> 18));
        bytes += char(0x80 | ((cp >> 12) & 0x3F));
        bytes += char(0x80 | ((cp >> 6) & 0x3F));
        bytes += char(0x80 | (cp & 0x3F));
    }
    for (char b : bytes) {
        ByteMask m;
        m.set((unsigned char)b);
        push_char(rule, m);
    }
}

inline uint32_t GrammarParser::symbol_id(const std::string& name) {
    for (const auto& s : symbols_)
        if (s.first == name) return s.second;
    const uint32_t id = uint32_t(symbols_.size());
    symbols_.emplace_back(name, id);
    if (g_.rules_.size() <= id) g_.rules_.resize(id + 1);
    return id;
}

inline uint32_t GrammarParser::generated_id() {
    const std::string name = "rep" + std::to_string(symbols_.size());
    return symbol_id(name);
}

inline void GrammarParser::add_rule(uint32_t id, const std::vector<Elem>& rule) {
    if (g_.rules_.size() <= id) g_.rules_.resize(id + 1);
    g_.rules_[id] = rule;
}

inline void GrammarParser::repetitions(uint32_t rule, size_t last_sym_start, uint64_t min_times, uint64_t max_times) {
    const bool no_max = max_times == std::numeric_limits<uint64_t>::max();
    if (min_times > 2000 || (!no_max && max_times > 2000)) fail("repetition count above 2000");
    if (!no_max && max_times < min_times) fail("repetition maximum below its minimum");
    if (last_sym_start >= g_.rules_[rule].size()) fail("nothing to repeat");
    const std::vector<Elem> prev(g_.rules_[rule].begin() + long(last_sym_start), g_.rules_[rule].end());
    const uint64_t n_opt = no_max ? 1 : max_times - min_times;
    {
        // Scoped: generating the tail rules below resize g_.rules_, which would leave a
        // reference to this rule dangling.
        auto& r = g_.rules_[rule];
        if (min_times == 0) {
            r.resize(last_sym_start);
        } else {
            for (uint64_t i = 1; i < min_times; i++) r.insert(r.end(), prev.begin(), prev.end());
        }
    }
    uint32_t last_rec = 0;
    std::vector<Elem> rec(prev);
    for (uint64_t i = 0; i < n_opt; i++) {
        rec.resize(prev.size());
        const uint32_t id = generated_id();
        if (i > 0 || no_max) {
            Elem ref;
            ref.type = Grammar::E_RULE;
            ref.value = no_max ? id : last_rec;
            rec.push_back(ref);
        }
        Elem alt;
        alt.type = Grammar::E_ALT;
        rec.push_back(alt);
        Elem end;
        end.type = Grammar::E_END;
        rec.push_back(end);
        add_rule(id, rec);
        last_rec = id;
    }
    if (n_opt > 0) {
        Elem ref;
        ref.type = Grammar::E_RULE;
        ref.value = last_rec;
        g_.rules_[rule].push_back(ref);
    }
}

inline void GrammarParser::parse_alternates(const std::string& name, uint32_t rule) {
    parse_sequence(name, rule);
    while (peek() == '|') {
        pos_++;
        Elem alt;
        alt.type = Grammar::E_ALT;
        g_.rules_[rule].push_back(alt);
        space_across_lines();
        parse_sequence(name, rule);
    }
    Elem end;
    end.type = Grammar::E_END;
    g_.rules_[rule].push_back(end);
}

inline void GrammarParser::parse_sequence(const std::string& name, uint32_t rule) {
    using namespace gbnf_detail;
    size_t last_sym_start = g_.rules_[rule].size();
    for (;;) {
        space();
        if (pos_ >= src_.size()) return;
        const char c = peek();
        if (c == '|' || c == ')' || c == '\n') return;  // the rule, alternative or group ends here
        if (c == '"') {
            pos_++;
            last_sym_start = g_.rules_[rule].size();
            while (peek() != '"') {
                if (pos_ >= src_.size()) fail("unterminated literal");
                if ((unsigned char)peek() < 0x80) {
                    push_utf8(rule, parse_char());
                } else {
                    push_utf8(rule, parse_utf8());
                }
            }
            pos_++;  // closing quote
        } else if (c == '[') {
            pos_++;
            bool negate = false;
            if (peek() == '^') {
                negate = true;
                pos_++;
            }
            ByteMask m;
            last_sym_start = g_.rules_[rule].size();
            while (peek() != ']') {
                if (pos_ >= src_.size()) fail("unterminated char class");
                const uint32_t lo = parse_char();
                if (lo > 0xFF) fail("non-ASCII range in a char class");
                if (peek() == '-' && pos_ + 1 < src_.size() && src_[pos_ + 1] != ']') {
                    pos_++;
                    const uint32_t hi = parse_char();
                    if (hi > 0xFF) fail("non-ASCII range in a char class");
                    if (hi < lo) fail("char class range runs backwards");
                    m.set_range(lo, hi);
                } else {
                    m.set(lo);
                }
            }
            pos_++;  // closing bracket
            if (negate) m.flip_all();
            push_char(rule, m);
        } else if (c == '(') {
            pos_++;
            const uint32_t sub = generated_id();
            const bool saved = newline_ok_;
            newline_ok_ = true;  // a group may wrap
            parse_alternates(name, sub);
            newline_ok_ = saved;
            if (peek() != ')') fail("expecting ')'");
            pos_++;
            last_sym_start = g_.rules_[rule].size();
            Elem ref;
            ref.type = Grammar::E_RULE;
            ref.value = sub;
            g_.rules_[rule].push_back(ref);
        } else if (c == '.') {
            pos_++;
            last_sym_start = g_.rules_[rule].size();
            ByteMask m;
            m.flip_all();
            push_char(rule, m);
        } else if (is_word_char(c)) {
            std::string ref;
            while (is_word_char(peek())) ref += src_[pos_++];
            last_sym_start = g_.rules_[rule].size();
            Elem e;
            e.type = Grammar::E_RULE;
            e.value = symbol_id(ref);
            g_.rules_[rule].push_back(e);
        } else if (c == '*' || c == '+' || c == '?' || c == '{') {
            uint64_t min_times = 0;
            uint64_t max_times = std::numeric_limits<uint64_t>::max();
            if (c == '+') min_times = 1;
            if (c == '?') max_times = 1;
            if (c == '{') {
                pos_++;
                space();
                if (!is_digit_char(peek())) fail("expecting a repetition count");
                min_times = 0;
                while (is_digit_char(peek())) min_times = min_times * 10 + uint64_t(src_[pos_++] - '0');
                space();
                if (peek() == '}') {
                    max_times = min_times;
                } else if (peek() == ',') {
                    pos_++;
                    space();
                    if (is_digit_char(peek())) {
                        max_times = 0;
                        while (is_digit_char(peek())) max_times = max_times * 10 + uint64_t(src_[pos_++] - '0');
                        space();
                    }
                    if (peek() != '}') fail("expecting '}'");
                } else {
                    fail("expecting ',' or '}'");
                }
            }
            pos_++;
            repetitions(rule, last_sym_start, min_times, max_times);
            last_sym_start = g_.rules_[rule].size();
        } else {
            fail(std::string("unexpected '") + c + "'");
        }
    }
}

inline uint32_t GrammarParser::parse_all() {
    using namespace gbnf_detail;
    space_across_lines();
    while (pos_ < src_.size()) {
        std::string name;
        while (is_word_char(peek())) name += src_[pos_++];
        if (name.empty()) fail("expecting a rule name");
        space();
        if (src_.compare(pos_, 3, "::=") != 0) fail("expecting '::='");
        pos_ += 3;
        space_across_lines();  // the body may start on the next line
        parse_alternates(name, symbol_id(name));
        if (peek() == '\r') pos_++;
        if (pos_ < src_.size() && peek() != '\n') fail("expecting end of line");
        if (pos_ < src_.size()) {
            line_++;
            pos_++;
        }
        space_across_lines();
    }
    for (const auto& s : symbols_)
        if (g_.rules_.size() > s.second && g_.rules_[s.second].empty())
            fail("rule '" + s.first + "' is referenced but never defined");
    for (const auto& s : symbols_)
        if (s.first == "root") return s.second;
    fail("no root rule");
}

inline Grammar::Grammar(const std::string& gbnf) {
    GrammarParser p(*this, gbnf);
    root_ = p.parse_all();
    if (root_ >= rules_.size()) rules_.resize(root_ + 1);
    alt_starts_.resize(rules_.size());
    for (size_t r = 0; r < rules_.size(); r++) {
        alt_starts_[r].push_back(0);
        for (size_t i = 0; i < rules_[r].size(); i++)
            if (rules_[r][i].type == E_ALT) alt_starts_[r].push_back(uint32_t(i + 1));
    }
}

// Expand a stack until every frame sits on a byte element (or the parse is over).
//  - E_CHAR: keep the frame, it is waiting for a byte;
//  - E_RULE: keep the caller's next element as a frame of its own, then push a frame on
//    each non-empty alternative of the referenced rule (one whose alternative is empty
//    matches without consuming anything, so only the caller's frame stays);
//  - anything else: an alternative that already ended, which the invariant keeps off the
//    stack, so the parse is dropped rather than guessed at.
inline void Grammar::advance_stack(const std::vector<uint64_t>& stack,
                                  std::vector<std::vector<uint64_t>>& out) const {
    std::vector<std::vector<uint64_t>> todo{stack};
    std::set<std::vector<uint64_t>> seen;
    while (!todo.empty()) {
        std::vector<uint64_t> cur = std::move(todo.back());
        todo.pop_back();
        if (!seen.insert(cur).second) continue;
        if (cur.empty()) {
            out.push_back(cur);  // the root rule matched: the answer may stop
            continue;
        }
        const uint32_t r = gbnf_rule(cur.back());
        const uint32_t p = gbnf_pos(cur.back());
        if (r >= rules_.size() || p >= rules_[r].size()) continue;  // nothing to match
        const Elem& e = rules_[r][p];
        if (e.type == E_CHAR) {
            out.push_back(cur);
            continue;
        }
        if (e.type != E_RULE || e.value >= alt_starts_.size()) continue;
        std::vector<uint64_t> base(cur.begin(), cur.end() - 1);
        if (continuation(r, p)) base.push_back(gbnf_frame(r, p + 1));
        for (uint32_t s : alt_starts_[e.value]) {
            if (s < rules_[e.value].size() && rules_[e.value][s].type != E_ALT &&
                rules_[e.value][s].type != E_END) {
                std::vector<uint64_t> next = base;
                next.push_back(gbnf_frame(e.value, s));
                todo.push_back(std::move(next));
            } else {
                todo.push_back(base);  // an empty alternative: the reference is satisfied
            }
        }
    }
}

// The alternatives of a rule are enumerated where the rule is entered: here for the start
// rule, and in advance_stack for every rule reference.
inline Grammar::State Grammar::start() const {
    State st;
    std::vector<std::vector<uint64_t>> out;
    for (uint32_t s : alt_starts_[root_]) {
        if (s < rules_[root_].size() && rules_[root_][s].type != E_ALT && rules_[root_][s].type != E_END)
            advance_stack({gbnf_frame(root_, s)}, out);
        else
            out.push_back({});  // an empty root alternative: the grammar matches at once
    }
    for (auto& s : out) {
        if (s.empty()) st.accepting = true;
        else st.stacks.push_back(std::move(s));
    }
    std::sort(st.stacks.begin(), st.stacks.end());
    st.stacks.erase(std::unique(st.stacks.begin(), st.stacks.end()), st.stacks.end());
    return st;
}

inline ByteMask Grammar::next_bytes(const State& st) const {
    ByteMask m;
    for (const auto& stack : st.stacks) {
        if (stack.empty()) continue;
        const uint32_t r = gbnf_rule(stack.back());
        const uint32_t p = gbnf_pos(stack.back());
        if (r >= rules_.size() || p >= rules_[r].size()) continue;
        const Elem& e = rules_[r][p];
        if (e.type != E_CHAR) continue;
        for (int i = 0; i < 4; i++) m.w[i] |= e.mask.w[i];
    }
    return m;
}

inline bool Grammar::step(const State& st, unsigned b, State& out) const {
    std::vector<std::vector<uint64_t>> expanded;
    for (const auto& stack : st.stacks) {
        if (stack.empty()) continue;
        const uint32_t r = gbnf_rule(stack.back());
        const uint32_t p = gbnf_pos(stack.back());
        if (r >= rules_.size() || p >= rules_[r].size()) continue;
        const Elem& e = rules_[r][p];
        if (e.type != E_CHAR || !e.mask.has(b)) continue;
        std::vector<uint64_t> next(stack.begin(), stack.end() - 1);
        if (continuation(r, p)) next.push_back(gbnf_frame(r, p + 1));
        advance_stack(next, expanded);
    }
    out = State{};
    for (auto& s : expanded) {
        if (s.empty()) out.accepting = true;
        else out.stacks.push_back(std::move(s));
    }
    std::sort(out.stacks.begin(), out.stacks.end());
    out.stacks.erase(std::unique(out.stacks.begin(), out.stacks.end()), out.stacks.end());
    return !out.dead();
}

inline bool Grammar::feed(State& st, const std::string& text) const {
    State cur = st;
    for (unsigned char b : text) {
        State next;
        if (!step(cur, b, next)) return false;
        cur = std::move(next);
    }
    if (cur.dead()) return false;
    st = std::move(cur);
    return true;
}

// The JSON grammar from third_party/llama.cpp/grammars/json.gbnf, for
// response_format {"type": "json_object"}.
inline const char* json_grammar() {
    return R"GB(
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= | " " | "\n" [ \t]{0,20}
)GB";
}

}  // namespace onebit::npu
