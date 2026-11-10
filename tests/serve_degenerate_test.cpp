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
//
// engine#1779 (F10 fault 2): the degenerate-repetition predicate. Header only, no device, no pack -- CI runs it in ms.
//
// The cases are chosen around the two ways this predicate can be wrong: too strict (the reported `//////` slips through --
// which is what a `\1{7,}` rule did in engine#1791, where eight identical characters were required and the symptom is six)
// and too loose (a legitimate short answer, or a long answer that merely *contains* a repeated run, gets flagged).
#include "serve_degenerate.h"

#include <cstdio>
#include <string>

static int failures = 0;

static void expect(bool want, const std::string& core, const char* why) {
    const bool got = onebit::degenerate_reply(core);
    if (got != want) {
        ++failures;
        std::printf("FAIL degenerate_reply(%s) = %s, want %s  (%s)\n", core.c_str(), got ? "true" : "false",
                    want ? "true" : "false", why);
    }
}

int main() {
    // The reported symptom, and the reason the floor is four rather than eight.
    expect(true, "//////", "the reported symptom: six slashes after the context limit");
    expect(true, "////", "four is the floor and it is inclusive");
    expect(false, "///", "three is short enough to be a real answer");
    expect(false, "", "an empty reply is not degeneracy");
    expect(true, "....", "any character, not just slashes");

    // Too-loose guards: a real answer must never be flagged.
    expect(false, "ok", "a short legitimate answer");
    expect(false, "aaaa!", "a repeated run plus punctuation is a sentence, not degeneracy");
    expect(false, "the cat sat", "ordinary text");
    expect(false, "aa", "two characters");

    // Boundary: mixed case and digits are different characters.
    expect(false, "aAaA", "case matters");
    expect(true, "0000", "digits can degenerate too");
    expect(false, "0001", "and one different character is enough to be fine");

    if (failures != 0) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("serve_degenerate: ok\n");
    return 0;
}
