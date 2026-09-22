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
#include "text_stream.h"

#include <algorithm>

namespace onebit {

namespace {

constexpr std::string_view kOpen = "<think>";
constexpr std::string_view kClose = "</think>";

bool is_space(char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }

// Length of the longest suffix of `s` that is a proper prefix of `word`.
size_t partial_suffix(std::string_view s, std::string_view word) {
    for (size_t n = std::min(s.size(), word.size() - 1); n > 0; --n)
        if (s.substr(s.size() - n) == word.substr(0, n)) return n;
    return 0;
}

// Bytes at the end of `s` that start an incomplete UTF-8 sequence.
size_t incomplete_utf8_tail(std::string_view s) {
    for (size_t back = 1; back <= std::min<size_t>(3, s.size()); ++back) {
        const auto c = static_cast<unsigned char>(s[s.size() - back]);
        if ((c & 0xC0) == 0x80) continue;  // continuation byte: keep looking back
        size_t need = (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        return need > back ? back : 0;
    }
    return 0;
}

}  // namespace

TextStream::TextStream(std::vector<std::string> stops, bool split_reasoning, bool reasoning_open)
    : stops_(std::move(stops)), split_(split_reasoning) {
    std::erase_if(stops_, [](const std::string& s) { return s.empty(); });
    mode_ = !split_ ? Mode::Content : reasoning_open ? Mode::Reasoning : Mode::Start;
}

void TextStream::emit(TextDelta& out, std::string_view content, std::string_view reasoning) {
    if (!content.empty() && trim_content_start_) {
        size_t skip = 0;
        while (skip < content.size() && is_space(content[skip])) ++skip;
        content.remove_prefix(skip);
        if (!content.empty()) trim_content_start_ = false;
    }
    out.content += content;
    out.reasoning += reasoning;
    all_content_ += content;
    all_reasoning_ += reasoning;
}

void TextStream::route(std::string_view text, bool final, TextDelta& out) {
    route_hold_ += text;
    for (;;) {
        std::string& h = route_hold_;
        if (mode_ == Mode::Content) {
            emit(out, h, {});
            h.clear();
            return;
        }
        if (mode_ == Mode::Start) {
            size_t lead = 0;
            while (lead < h.size() && is_space(h[lead])) ++lead;
            std::string_view rest = std::string_view(h).substr(lead);
            if (rest.starts_with(kOpen)) {
                h.erase(0, lead + kOpen.size());
                mode_ = Mode::Reasoning;
                continue;
            }
            if (!final && kOpen.starts_with(rest)) return;  // may still become <think>
            mode_ = Mode::Content;
            continue;
        }
        // Reasoning: emit up to </think>, trimming the block's outer whitespace.
        if (all_reasoning_.empty()) {
            size_t lead = 0;
            while (lead < h.size() && is_space(h[lead])) ++lead;
            h.erase(0, lead);
        }
        if (size_t close = h.find(kClose); close != std::string::npos) {
            size_t end = close;
            while (end > 0 && is_space(h[end - 1])) --end;
            emit(out, {}, std::string_view(h).substr(0, end));
            h.erase(0, close + kClose.size());
            mode_ = Mode::Content;
            trim_content_start_ = true;
            continue;
        }
        if (final) {  // unterminated block: everything is reasoning
            size_t end = h.size();
            while (end > 0 && is_space(h[end - 1])) --end;
            emit(out, {}, std::string_view(h).substr(0, end));
            h.clear();
            return;
        }
        // Hold a possible partial </think> and trailing whitespace (it may precede the tag).
        size_t keep = partial_suffix(h, kClose);
        size_t end = h.size() - keep;
        while (end > 0 && is_space(h[end - 1])) --end;
        emit(out, {}, std::string_view(h).substr(0, end));
        h.erase(0, end);
        return;
    }
}

TextDelta TextStream::push(std::string_view bytes) {
    TextDelta out;
    if (stopped_) return out;

    std::string text = utf8_tail_ + std::string(bytes);
    const size_t tail = incomplete_utf8_tail(text);
    utf8_tail_ = text.substr(text.size() - tail);
    text.resize(text.size() - tail);

    stop_hold_ += text;
    size_t first = std::string::npos;
    for (const auto& s : stops_) first = std::min(first, stop_hold_.find(s));
    if (first != std::string::npos) {
        stopped_ = true;
        route(std::string_view(stop_hold_).substr(0, first), true, out);
        stop_hold_.clear();
        utf8_tail_.clear();
        return out;
    }
    size_t keep = 0;
    for (const auto& s : stops_) keep = std::max(keep, partial_suffix(stop_hold_, s));
    route(std::string_view(stop_hold_).substr(0, stop_hold_.size() - keep), false, out);
    stop_hold_.erase(0, stop_hold_.size() - keep);
    return out;
}

TextDelta TextStream::finish() {
    TextDelta out;
    if (stopped_) return out;
    route(stop_hold_, true, out);
    stop_hold_.clear();
    utf8_tail_.clear();
    return out;
}

}  // namespace onebit
