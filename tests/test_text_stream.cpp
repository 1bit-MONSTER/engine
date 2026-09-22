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
#include <string>
#include <vector>

#include "check.h"
#include "text_stream.h"

using namespace onebit;

namespace {

struct Run {
    std::string content, reasoning;
    bool stopped = false;
};

// Feeds pieces one at a time, as tokens would arrive, and concatenates deltas.
Run feed(const std::vector<std::string>& pieces, std::vector<std::string> stops = {}, bool split = false,
         bool open = false) {
    TextStream ts(std::move(stops), split, open);
    Run r;
    for (const auto& p : pieces) {
        TextDelta d = ts.push(p);
        r.content += d.content;
        r.reasoning += d.reasoning;
    }
    TextDelta d = ts.finish();
    r.content += d.content;
    r.reasoning += d.reasoning;
    r.stopped = ts.stopped();
    CHECK(r.content == ts.content());
    CHECK(r.reasoning == ts.reasoning());
    return r;
}

// Every way of cutting `s` into single bytes: the strictest streaming case.
std::vector<std::string> bytes(const std::string& s) {
    std::vector<std::string> out;
    for (char c : s) out.emplace_back(1, c);
    return out;
}

}  // namespace

static void test_utf8() {
    // "é" split across two tokens must not be emitted half-way.
    TextStream ts({}, false);
    CHECK(ts.push("caf\xC3").content == "caf");
    CHECK(ts.push("\xA9!").content == "\xC3\xA9!");
    CHECK(feed(bytes("日本語 😀")).content == "日本語 😀");
    // A dangling lead byte at the end is dropped rather than sent broken.
    CHECK(feed({"ok\xE4\xBD"}).content == "ok");
}

static void test_stops() {
    Run r = feed(bytes("Hello world. User: hi"), {"User:"});
    CHECK(r.content == "Hello world. ");
    CHECK(r.stopped);
    r = feed({"ab", "cUs", "er", "x"}, {"User"});  // stop string spans tokens
    CHECK(r.content == "abc");
    CHECK(r.stopped);
    r = feed({"Use", "ful"}, {"User"});  // near miss: held, then released
    CHECK(r.content == "Useful");
    CHECK(!r.stopped);
    r = feed({"Us"}, {"User"});  // ends on a partial stop: released at finish
    CHECK(r.content == "Us");
    r = feed({"a", "STOP", "b"}, {"", "STOP"});  // empty stop strings ignored
    CHECK(r.content == "a");
    // Earliest of several stop strings wins.
    r = feed({"x END y STOP"}, {"STOP", "END"});
    CHECK(r.content == "x ");
}

static void test_reasoning() {
    const std::string text = "<think>\nLet me think.\n</think>\n\nThe answer is 4.";
    for (const auto& pieces : {std::vector<std::string>{text}, bytes(text)}) {
        Run r = feed(pieces, {}, true);
        CHECK(r.reasoning == "Let me think.");
        CHECK(r.content == "The answer is 4.");
    }
    // Split off: the tags stay in content.
    CHECK(feed({text}).content == text);
    // No think block: all content, including text that merely starts like "<t".
    CHECK(feed(bytes("<table>"), {}, true).content == "<table>");
    CHECK(feed({"  Hi"}, {}, true).content == "  Hi");
    // Prompt already opened the block.
    Run r = feed(bytes("step one\n</think>\n\nDone"), {}, true, true);
    CHECK(r.reasoning == "step one");
    CHECK(r.content == "Done");
    // Unterminated block (hit max tokens): all of it is reasoning.
    r = feed(bytes("<think>\nstill going  "), {}, true);
    CHECK(r.reasoning == "still going");
    CHECK(r.content.empty());
    // Interior whitespace survives streaming.
    r = feed(bytes("<think>a\n\nb</think>c"), {}, true);
    CHECK(r.reasoning == "a\n\nb");
    CHECK(r.content == "c");
    // Stop string inside the reasoning ends everything there.
    r = feed(bytes("<think>abc STOP def</think>x"), {"STOP"}, true);
    CHECK(r.reasoning == "abc");
    CHECK(r.content.empty());
}

int main() {
    test_utf8();
    test_stops();
    test_reasoning();
    return test_result("test_text_stream");
}
