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

// The answer parser for `1bit forward-serve` (issue #652 item 3): one place that splits what the
// model wrote into OpenAI's three parts -- `reasoning_content`, `content` and `tool_calls` -- so the
// streamed and the non-streamed reply are the same answer, not two different ones.
//
// One implementation serves both paths: AnswerStream is the incremental splitter, and parse_answer()
// is that splitter fed the whole text at once.  A streamed delta therefore cannot disagree with the
// non-streamed reply, which is exactly the property #652 asks for.
//
// The two rules from the issue, as code rather than convention:
//   * markup counts as a call only when the request carried `tools` (tools_sent); otherwise the
//     model's <tool_call> block is text the caller asked for and stays in `content`;
//   * a reply that is only tool calls has `content: null` (content_null), not "".
//
// Families: Qwen3's markup -- `<think>…</think>` for reasoning and
// `<tool_call>{"name": …, "arguments": {…}}</tool_call>` for calls, which is what this route's ChatML
// models emit -- and Gemma 4's (`<|channel>thought…<channel|>`, `<|tool_call>call:f{…}<tool_call|>`).
// answer_family_of_template() picks one from the container's own chat template.  AnswerFamily exists so
// LFM2 can be added without another signature change.
#pragma once

#include <nlohmann/json.hpp>

#include <cctype>
#include <functional>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace onebit::npu {

enum class AnswerFamily {
    qwen3,
    // Gemma 4 (engine#1779, F10 fault 4): the pack's own tokenizer_config.json states this markup, and it is not
    // qwen3's. Tokens: <|turn>/<turn|>, <|channel>/<channel|>, <|tool_call>/<tool_call|>, escape <|"|>.
    // A call body is `call:NAME{key:value,...}` with string values escaped as <|"|>v<|"|> -- NOT JSON, which is
    // why a JSON-only reader fails on it (the vendor's response_schema ships the same two regexes this follows).
    gemma4,
};

// One tool call, in OpenAI's shape: function.arguments is a JSON *string*, so the parsed object is
// re-dumped rather than kept as a value.
struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments;
};

struct ParsedAnswer {
    std::string reasoning;      // <think>…</think> contents, concatenated
    std::string content;        // everything outside reasoning and outside call markup
    std::vector<ToolCall> calls;
    bool content_null = false;  // the reply was calls only
};

namespace answer_detail {

inline std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\n' || s[b] == '\r' || s[b] == '\t')) ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\n' || s[e - 1] == '\r' || s[e - 1] == '\t')) --e;
    return s.substr(b, e - b);
}

inline const char* family_name(AnswerFamily f) {
    switch (f) {
        case AnswerFamily::qwen3: return "qwen3";
        case AnswerFamily::gemma4: return "gemma4";
    }
    return "?";
}

// The tags of a family, longest first so a prefix test cannot match the shorter one first.
struct FamilyTags {
    const char* think_open;
    const char* think_close;
    const char* call_open;
    const char* call_close;
};

inline FamilyTags tags_of(AnswerFamily f) {
    switch (f) {
        case AnswerFamily::qwen3:
            return FamilyTags{"<think>", "</think>", "<tool_call>", "</tool_call>"};
        case AnswerFamily::gemma4:
            // The reasoning channel opens with <|channel> immediately followed by the literal word `thought`; the
            // template emits exactly `<|channel>thought\n…\n<channel|>`, so the pair is the tag.
            return FamilyTags{"<|channel>thought", "<channel|>", "<|tool_call>", "<tool_call|>"};
    }
    throw std::runtime_error(std::string("answer parser: unsupported family ") + family_name(f));
}

// How many trailing bytes of `buf` could still become a tag: the longest suffix that is a proper
// prefix of one of the family's tags.  Those are held back until the next chunk decides them.
inline size_t holdback(const std::string& buf, const FamilyTags& t) {
    const char* tags[4] = {t.think_open, t.think_close, t.call_open, t.call_close};
    size_t hold = 0;
    for (const char* tag : tags) {
        const size_t n = std::string(tag).size();
        for (size_t len = std::min(n - 1, buf.size()); len > 0; len--)
            if (buf.compare(buf.size() - len, len, tag, len) == 0) {
                hold = std::max(hold, len);
                break;
            }
    }
    return hold;
}

}  // namespace answer_detail

// Which family a container's replies are written in, read from the chat template the container declares: a
// model writes the markup its template taught it.  Gemma 4's template carries `<|tool_call>` and
// `<|channel>thought` as literals (amd/gemma-4-e4b-it's chat_template: one of each); a Qwen3 template carries
// `<tool_call>` and neither of those.  No template, or one with neither marker, is qwen3 -- what every
// container was parsed as before this function existed.
inline AnswerFamily answer_family_of_template(const std::string& template_text) {
    const answer_detail::FamilyTags g = answer_detail::tags_of(AnswerFamily::gemma4);
    if (template_text.find(g.call_open) != std::string::npos ||
        template_text.find(g.think_open) != std::string::npos)
        return AnswerFamily::gemma4;
    return AnswerFamily::qwen3;
}

// The incremental splitter.  feed() appends whatever is now unambiguous to `out` as
// (is_reasoning, text) pieces; finish() flushes what the holdback still held and reports the calls.
class AnswerStream {
public:
    // `starts_in_reasoning`: the PROMPT already opened the reasoning block (prompt_opens_reasoning below), so the
    // answer begins inside it and the first thing to look for is the closing tag.
    AnswerStream(AnswerFamily family, bool tools_sent, bool starts_in_reasoning = false)
        : fam_(family), tags_(answer_detail::tags_of(family)), tools_sent_(tools_sent),
          mode_(starts_in_reasoning ? Mode::reasoning : Mode::content) {}

    void feed(const std::string& chunk, std::vector<std::pair<bool, std::string>>& out) {
        buf_ += chunk;
        const size_t hold = answer_detail::holdback(buf_, tags_);
        const std::string safe = buf_.substr(0, buf_.size() - hold);
        buf_.erase(0, buf_.size() - hold);
        scan(safe, out);
    }

    // End of the answer: flush the rest.  An unterminated <tool_call> is put back into content
    // rather than dropped -- nothing the model wrote may vanish quietly.
    void finish(std::vector<std::pair<bool, std::string>>& out) {
        scan(buf_, out);
        buf_.clear();
        if (mode_ == Mode::call) {
            // Through emit(), so content() and the emitted pieces cannot disagree.
            emit(out, false, std::string(tags_.call_open) + call_text_);
            call_text_.clear();
            mode_ = Mode::content;
        }
    }

    // The calls seen so far, in order.  A call needs tools_sent, per the issue's rule.
    const std::vector<ToolCall>& calls() const { return calls_; }
    bool saw_reasoning() const { return saw_reasoning_; }
    // Inside a reasoning block right now (after the opening tag, before the closing one). reasoning_token_count()
    // reads it before and after each token, so the tags themselves and held-back partial tags count with the block.
    bool in_reasoning() const { return mode_ == Mode::reasoning; }
    bool saw_text() const { return saw_text_; }

    const std::string& reasoning() const { return reasoning_; }
    const std::string& content() const { return content_; }

private:
    enum class Mode { content, reasoning, call };

    // Whitespace around a reasoning block is the model's layout, not part of either text: `<think>\n...\n</think>\n\n`
    // then the answer.  Left in, the answer to "one word" comes back as "\n\nParis", and a model that thinks nothing
    // (`<think>\n\n</think>`) sends a reasoning chunk that is two newlines.  So the whitespace that LEADS the reasoning
    // is dropped, and so is the whitespace that leads the content when that content follows a reasoning block --
    // the rule app/think_split.h has always applied on the in-process route ("whitespace around the block is
    // dropped"), so the two routes return the same strings.  Content with no reasoning before it is not touched.
    void emit(std::vector<std::pair<bool, std::string>>& out, bool is_reasoning, const std::string& raw) {
        if (raw.empty()) return;
        std::string text = raw;
        bool& trim = is_reasoning ? trim_reasoning_ : trim_content_;
        if (trim) {
            const size_t b = text.find_first_not_of(" \t\r\n");
            if (b == std::string::npos) return;   // all whitespace so far: nothing to send yet
            text.erase(0, b);
            trim = false;
        }
        if (is_reasoning) {
            reasoning_ += text;
            saw_reasoning_ = true;
        } else {
            content_ += text;
            saw_text_ = true;
        }
        out.emplace_back(is_reasoning, text);
    }

    void scan(const std::string& safe, std::vector<std::pair<bool, std::string>>& out) {
        size_t i = 0;
        std::string plain;  // consecutive characters of the current mode
        auto flush_plain = [&] {
            if (plain.empty()) return;
            if (mode_ == Mode::call) {
                call_text_ += plain;
            } else {
                emit(out, mode_ == Mode::reasoning, plain);
            }
            plain.clear();
        };
        while (i < safe.size()) {
            if (safe[i] == '<') {
                // Compared at the tag's own length, never through a fixed window: this read `safe.substr(i, 16)`
                // once, and gemma4's thinking tag `<|channel>thought` is 17 bytes, so it could never match -- every
                // tag as long as the window read as text (engine#1779). No width is left here to outgrow.
                auto starts = [&](const char* tag) {
                    const size_t n = std::char_traits<char>::length(tag);
                    return safe.compare(i, n, tag) == 0;
                };
                if (starts(tags_.think_open)) {
                    flush_plain();
                    mode_ = Mode::reasoning;
                    i += std::string(tags_.think_open).size();
                    continue;
                }
                if (starts(tags_.think_close)) {
                    flush_plain();
                    // what follows a block is separated from it by layout whitespace -- unless content came first
                    if (mode_ == Mode::reasoning && !saw_text_) trim_content_ = true;
                    mode_ = Mode::content;
                    i += std::string(tags_.think_close).size();
                    continue;
                }
                if (starts(tags_.call_open)) {
                    flush_plain();
                    if (tools_sent_) {
                        mode_ = Mode::call;
                        call_text_.clear();
                    }
                    // Without tools the tag is just text: keep it in content, marker and all.
                    if (!tools_sent_) plain += tags_.call_open;
                    i += std::string(tags_.call_open).size();
                    continue;
                }
                if (starts(tags_.call_close) && mode_ == Mode::call) {
                    flush_plain();
                    close_call(out);
                    i += std::string(tags_.call_close).size();
                    continue;
                }
            }
            plain += safe[i++];
        }
        flush_plain();
    }

    void close_call(std::vector<std::pair<bool, std::string>>& out) {
        if (fam_ == AnswerFamily::gemma4 && close_gemma_call()) {
            call_text_.clear();
            mode_ = Mode::content;
            return;
        }
        try {
            const nlohmann::json j = nlohmann::json::parse(call_text_);
            ToolCall c;
            c.id = "call_" + std::to_string(calls_.size());
            c.name = j.at("name").get<std::string>();
            c.arguments = j.contains("arguments") ? j.at("arguments").dump() : std::string("{}");
            calls_.push_back(std::move(c));
        } catch (const std::exception&) {
            // Not the object the grammar would have produced: it goes back into content, and out as
            // a delta as well, so what the model wrote is never lost and both paths agree.
            emit(out, false, std::string(tags_.call_open) + call_text_ + tags_.call_close);
        }
        call_text_.clear();
        mode_ = Mode::content;
    }

    // `call:NAME{key:value,...}` -> a ToolCall whose arguments are JSON, so the two routes hand the caller one shape.
    // Throws on a body that is not that shape (the caller puts the markup back into content).
    // The Gemma object syntax is not JSON: keys are bare, string values are wrapped in <|"|>, and nested objects and
    // arrays use the same bare-key form. This rewrites it rather than guessing: <|"|>v<|"|> becomes a JSON string,
    // a bare key gains quotes, and numbers/true/false/null pass through as literals.
    // shortcut: values that are neither quoted, numeric, boolean, null nor a brace/bracket nest are emitted as strings,
    // re-derive from the pack if a writer starts emitting bare tokens that are meant to be something else.
    static std::string gemma_object_to_json(const std::string& body) {
        std::string out;
        size_t i = 0;
        auto space = [&] { while (i < body.size() && (body[i] == ' ' || body[i] == '\n' || body[i] == '\t')) ++i; };
        // one value at `i`
        std::function<void()> value = [&] {
            space();
            if (body.compare(i, 5, "<|\"|>") == 0) {
                i += 5;
                std::string lit;
                while (i < body.size() && body.compare(i, 5, "<|\"|>") != 0) lit += body[i++];
                // A string whose closing escape never arrives is malformed, not a shorter string: refusing it sends
                // the markup back to content (close_call), where accepting it would invent a truncated argument.
                if (i >= body.size()) throw std::runtime_error("gemma4 call: unterminated <|\"|> string");
                i += 5;
                out += nlohmann::json(lit).dump();
                return;
            }
            if (i < body.size() && (body[i] == '{' || body[i] == '[')) {
                const char close_ch = body[i] == '{' ? '}' : ']';
                out += body[i++];
                space();
                bool first = true;
                while (i < body.size() && body[i] != close_ch) {
                    if (!first) out += ',';
                    first = false;
                    space();
                    if (close_ch == '}') {
                        const size_t colon = body.find(':', i);
                        if (colon == std::string::npos) throw std::runtime_error("gemma4 call: a key with no value");
                        std::string key = body.substr(i, colon - i);
                        while (!key.empty() && (key.back() == ' ' || key.back() == '\n')) key.pop_back();
                        if (key.empty()) throw std::runtime_error("gemma4 call: an empty key");
                        out += nlohmann::json(key).dump() + ":";
                        i = colon + 1;
                        value();
                    } else {
                        value();
                    }
                    space();
                    if (i < body.size() && body[i] == ',') ++i;
                }
                if (i >= body.size()) throw std::runtime_error("gemma4 call: an object or array that never closes");
                out += body[i++];
                return;
            }
            std::string raw;
            int depth = 0;
            while (i < body.size()) {
                const char c = body[i];
                if (c == '{' || c == '[') ++depth;
                if (c == '}' || c == ']') { if (depth == 0) break; --depth; }
                if (depth == 0 && (c == ',' || c == '}' || c == ']')) break;
                raw += c; ++i;
            }
            while (!raw.empty() && (raw.back() == ' ' || raw.back() == '\n')) raw.pop_back();
            // An empty value consumed nothing. Refusing it is also what keeps the loops above finite: `[1}` would
            // otherwise ask for a value at the `}` for ever, appending a comma each time.
            if (raw.empty()) throw std::runtime_error("gemma4 call: an empty value");
            if (raw == "true" || raw == "false" || raw == "null" || (!raw.empty() && (std::isdigit((unsigned char)raw[0]) || raw[0] == '-'))) {
                out += raw;
            } else {
                out += nlohmann::json(raw).dump();
            }
        };
        value();
        space();
        if (i != body.size()) throw std::runtime_error("gemma4 call: text after the arguments object");
        return out;
    }

    bool close_gemma_call() {
        const std::string body = answer_detail::trim(call_text_);
        if (body.compare(0, 5, "call:") != 0) return false;
        const size_t brace = body.find('{');
        if (brace == std::string::npos || body.back() != '}') return false;
        ToolCall c;
        c.id = "call_" + std::to_string(calls_.size());
        c.name = answer_detail::trim(body.substr(5, brace - 5));
        if (c.name.empty()) return false;
        try {
            c.arguments = gemma_object_to_json(body.substr(brace, body.size() - brace));
        } catch (const std::exception&) {
            return false;
        }
        calls_.push_back(std::move(c));
        return true;
    }

    AnswerFamily fam_;
    answer_detail::FamilyTags tags_;
    bool tools_sent_ = false;
    Mode mode_ = Mode::content;
    std::string buf_;        // held back: may still become a tag
    std::string call_text_;  // the JSON inside an open <tool_call>
    std::string reasoning_;
    std::string content_;
    std::vector<ToolCall> calls_;
    bool saw_reasoning_ = false;
    bool saw_text_ = false;
    bool trim_reasoning_ = true;   // until the reasoning's first non-whitespace character
    bool trim_content_ = false;    // set when a reasoning block closes before any content; cleared by the first text
};

// Does the rendered prompt leave the model INSIDE a reasoning block?  Some templates end their generation prompt
// with the opening tag itself -- Qwen3.5's ends `<|im_start|>assistant\n<think>\n` when thinking is on -- so the
// model never writes that tag: its answer starts with the reasoning and the first tag in it is the closing one.
// A splitter that waits to see the block opened then files the whole reasoning under `content` and drops the
// closing tag it never saw opened, so nothing in the reply marks where the answer starts (measured on FastFlowLM's
// Qwen3.5-2B-NPU2, engine#1786).  True when the prompt, less trailing whitespace,
// ends with the family's opening tag.  A prompt that opens and closes the block (thinking off:
// `<think>\n\n</think>\n\n`) does not end with it, and neither does one that never mentions it.
inline bool prompt_opens_reasoning(const std::string& prompt, AnswerFamily family) {
    const std::string open = answer_detail::tags_of(family).think_open;
    size_t end = prompt.size();
    while (end > 0 && (prompt[end - 1] == ' ' || prompt[end - 1] == '\n' || prompt[end - 1] == '\r' || prompt[end - 1] == '\t')) --end;
    return !open.empty() && end >= open.size() && prompt.compare(end - open.size(), open.size(), open) == 0;
}

// Which of `pieces` (one generated token's text each, in order) belong to `content` -- not to
// reasoning and not to call markup.  Callers that report a per-token value use this so reasoning and
// call markup get no entry (issue #651 item 3).  A token that straddles a boundary ("</think>The")
// counts as content: its text did reach the caller.
inline std::vector<char> content_mask(const std::vector<std::string>& pieces, AnswerFamily family, bool tools_sent,
                                      bool starts_in_reasoning = false) {
    AnswerStream s(family, tools_sent, starts_in_reasoning);
    std::vector<char> mask;
    mask.reserve(pieces.size());
    std::vector<std::pair<bool, std::string>> out;
    for (const auto& piece : pieces) {
        out.clear();
        s.feed(piece, out);
        bool is_content = false;
        for (const auto& [is_reasoning, text] : out) {
            (void)text;
            is_content = is_content || !is_reasoning;
        }
        mask.push_back(is_content ? 1 : 0);
    }
    // Whatever the holdback still holds at the end belongs to the content being flushed by finish().
    out.clear();
    s.finish(out);
    for (const auto& [is_reasoning, text] : out) {
        (void)text;
        if (!is_reasoning && !mask.empty() && !mask.back()) mask.back() = 1;
    }
    return mask;
}

// How many of `pieces` (one generated token's text each, in order) were reasoning: OpenAI's
// usage.completion_tokens_details.reasoning_tokens (engine#2111). An empty `content` with finish_reason "length" reads
// like "the model said nothing" when the whole budget went into a thinking block, and this is the number that says so.
// A token is reasoning when none of its text reached `content` and the block was open before it, after it, or it put
// text into the block: so the opening and closing tags, a partial tag the holdback keeps, and an unclosed block at the
// end of the budget all count, while call markup and the whitespace after a closed block do not. A token that
// straddles the boundary ("</think>The") is content, as content_mask() says. Same splitter, so the count, the split
// and the logprob entries cannot disagree.
inline int reasoning_token_count(const std::vector<std::string>& pieces, AnswerFamily family, bool tools_sent,
                                 bool starts_in_reasoning = false) {
    AnswerStream s(family, tools_sent, starts_in_reasoning);
    std::vector<std::pair<bool, std::string>> out;
    int n = 0;
    bool last_counted = false;
    // Tokens that produced nothing before anything was produced (leading whitespace, the first half of an opening tag
    // the holdback keeps) belong to whatever comes next: to the block if the next token opens it, to nothing otherwise.
    int pending = 0;
    bool started = false;
    for (const auto& piece : pieces) {
        const bool before = s.in_reasoning();
        out.clear();
        s.feed(piece, out);
        bool content = false, reasoning = false;
        for (const auto& [is_reasoning, text] : out) {
            (void)text;
            (is_reasoning ? reasoning : content) = true;
        }
        last_counted = !content && (reasoning || before || s.in_reasoning());
        if (last_counted) {
            ++n;
            if (!started) n += pending;
            started = true;
        } else if (content) {
            started = true;
        } else if (!started) {
            ++pending;
        }
    }
    // What the holdback still holds is flushed by finish(): if it is content, the last token's text reached `content`.
    out.clear();
    s.finish(out);
    for (const auto& [is_reasoning, text] : out) {
        (void)text;
        if (!is_reasoning && last_counted) {
            --n;
            last_counted = false;
        }
    }
    return n;
}

// The whole answer at once: the same splitter, so the two paths cannot disagree.
inline ParsedAnswer parse_answer(const std::string& text, AnswerFamily family, bool tools_sent,
                                 bool starts_in_reasoning = false) {
    AnswerStream s(family, tools_sent, starts_in_reasoning);
    std::vector<std::pair<bool, std::string>> pieces;
    s.feed(text, pieces);
    s.finish(pieces);
    ParsedAnswer a;
    a.reasoning = s.reasoning();
    a.content = s.content();
    a.calls = s.calls();
    // A call-only reply says content: null rather than "" (the issue's second rule).
    a.content_null = !a.calls.empty() && a.content.empty();
    return a;
}

}  // namespace onebit::npu
