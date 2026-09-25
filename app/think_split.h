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
#pragma once

#include <algorithm>
#include <string>

namespace onebit {

// Splits a thinking model's output as llama-server does (reasoning_format "auto"): the text
// inside <think>...</think> is reasoning_content, the rest content. Tags may arrive split across
// tokens, so a possible partial tag is held back; the think block is only entered before any
// content, and whitespace around the block is dropped.
struct ThinkSplit {
    bool in_think;
    bool content_started = false;
    bool reasoning_started = false;
    std::string buf;
    explicit ThinkSplit(bool opened) : in_think(opened) {}

    template <class Emit>  // Emit(bool reasoning, const std::string& text)
    void feed(const std::string& t, Emit&& emit) {
        buf += t;
        for (;;) {
            const std::string tag = in_think ? "</think>" : "<think>";
            const size_t p = buf.find(tag);
            const bool opens = p != std::string::npos && !in_think && !content_started &&
                               buf.find_first_not_of(" \t\r\n") >= p;  // only whitespace before it
            if (p != std::string::npos && (in_think || opens)) {
                out(buf.substr(0, p), emit);
                buf.erase(0, p + tag.size());
                in_think = !in_think;
                continue;
            }
            // hold back the longest suffix that could start the tag
            size_t keep = 0;
            for (size_t k = std::min(tag.size() - 1, buf.size()); k > 0; --k)
                if (buf.compare(buf.size() - k, k, tag, 0, k) == 0) { keep = k; break; }
            out(buf.substr(0, buf.size() - keep), emit);
            buf.erase(0, buf.size() - keep);
            return;
        }
    }
    template <class Emit>
    void flush(Emit&& emit) {
        out(buf, emit);
        buf.clear();
    }

private:
    template <class Emit>
    void out(std::string text, Emit& emit) {
        bool& started = in_think ? reasoning_started : content_started;
        if (!started) {
            const size_t b = text.find_first_not_of(" \t\r\n");
            if (b == std::string::npos) return;
            text.erase(0, b);
            started = true;
        }
        if (!text.empty()) emit(in_think, text);
    }
};

}  // namespace onebit
