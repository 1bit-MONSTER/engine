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
// A reply that is one repeated character after the context limit (engine#1779, F10 fault 2).
//
// FastFlowLM 1.0.12's notes list "Qwen3.5 answering `//////` after hitting the context limit". The rule here is deliberately
// blunt, because the thing it detects is blunt: a reply whose whole body is one character repeated. Four is the floor --
// `//////` (the reported symptom) is six, and an earlier detector in this tree used `\1{7,}`, i.e. eight, so a server
// answering exactly the reported string passed it (engine#1791, closed; the rule below is pi-hrx-sched's, credited).
//
// It is a predicate, not a policy: callers decide what to do (answer with a refusal, close the stream cleanly), and nothing
// here reads the request, the model or the context. That keeps it testable without a device or a pack -- see
// tests/serve_degenerate_test.cpp.
#ifndef ONEBIT_SERVE_DEGENERATE_H
#define ONEBIT_SERVE_DEGENERATE_H

#include <cstddef>
#include <string>

namespace onebit {

// True when `core` (the reply text with leading/trailing whitespace removed) is at least four characters and every one of
// them is the same character. An empty or short reply is not degenerate -- "ok" is a legitimate answer, and "\n\n\n\n" is
// trimmed to nothing before it gets here.
inline bool degenerate_reply(const std::string& core) {
    if (core.size() < 4) return false;
    for (std::size_t i = 1; i < core.size(); ++i) {
        if (core[i] != core[0]) return false;
    }
    return true;
}

}  // namespace onebit

#endif  // ONEBIT_SERVE_DEGENERATE_H
