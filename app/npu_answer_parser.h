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
// Families: Qwen3's markup today -- `<think>…</think>` for reasoning and
// `<tool_call>{"name": …, "arguments": {…}}</tool_call>` for calls, which is what this route's ChatML
// models emit.  AnswerFamily exists so Gemma 4 (`call:f{…}`) and LFM2 can be added without another
// signature change; an unsupported family is a 400 the caller sees, never a silent wrong split.
#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace onebit::npu {

enum class AnswerFamily {
    qwen3,
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

inline const char* family_name(AnswerFamily f) {
    switch (f) {
        case AnswerFamily::qwen3: return "qwen3";
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

// The incremental splitter.  feed() appends whatever is now unambiguous to `out` as
// (is_reasoning, text) pieces; finish() flushes what the holdback still held and reports the calls.
class AnswerStream {
public:
    AnswerStream(AnswerFamily family, bool tools_sent)
        : fam_(family), tags_(answer_detail::tags_of(family)), tools_sent_(tools_sent) {}

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
    bool saw_text() const { return saw_text_; }

    const std::string& reasoning() const { return reasoning_; }
    const std::string& content() const { return content_; }

private:
    enum class Mode { content, reasoning, call };

    void emit(std::vector<std::pair<bool, std::string>>& out, bool is_reasoning, const std::string& text) {
        if (text.empty()) return;
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
                const std::string rest = safe.substr(i, 16);
                auto starts = [&](const char* tag) { return rest.rfind(tag, 0) == 0; };
                if (starts(tags_.think_open)) {
                    flush_plain();
                    mode_ = Mode::reasoning;
                    i += std::string(tags_.think_open).size();
                    continue;
                }
                if (starts(tags_.think_close)) {
                    flush_plain();
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
};

// Which of `pieces` (one generated token's text each, in order) belong to `content` -- not to
// reasoning and not to call markup.  Callers that report a per-token value use this so reasoning and
// call markup get no entry (issue #651 item 3).  A token that straddles a boundary ("</think>The")
// counts as content: its text did reach the caller.
inline std::vector<char> content_mask(const std::vector<std::string>& pieces, AnswerFamily family, bool tools_sent) {
    AnswerStream s(family, tools_sent);
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

// The whole answer at once: the same splitter, so the two paths cannot disagree.
inline ParsedAnswer parse_answer(const std::string& text, AnswerFamily family, bool tools_sent) {
    AnswerStream s(family, tools_sent);
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
